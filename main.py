import glob
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

REPO = "m-a-p/YuE2-3B"
MY_BUCKET = "directedbykobyperez/YuE2-mysongs-private"
OUT = "/tmp/yue2_out"
os.makedirs(OUT, exist_ok=True)

pipe = None


class GenerationRequest(BaseModel):
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 0
    cfg_scale: float = 1.0


def send_to_telegram(text):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=3,
        )
    except Exception:
        pass


def is_content_safe(style: str, lyrics: str) -> bool:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    system_prompt = os.getenv("CONTENT_FILTER_PROMPT")
    if not api_key or not system_prompt:
        return True
    try:
        import requests
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Style/Caption: {style}\nLyrics: {lyrics}"},
                ],
                "stream": False,
            },
            timeout=5,
        )
        answer = response.json()["choices"][0]["message"]["content"].strip().upper()
        if "BLOCK" in answer:
            send_to_telegram(f"🚨 Content Blocked!\nStyle: {style}\nLyrics: {lyrics}")
            return False
    except Exception:
        pass
    return True


def _ensure_yue2():
    try:
        import yue2  # noqa: F401
        return
    except ImportError:
        pass
    whl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yue2_infer-0.1.5-py3-none-any.whl")
    print(f"[yue2] installing vendored wheel: {whl}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", whl],
        check=True, capture_output=True, text=True,
    )
    print("[yue2] installed.")


def get_pipeline():
    global pipe
    if pipe is None:
        _ensure_yue2()
        from yue2 import YuE2Pipeline
        print("Loading YuE2-3B pipeline (first run downloads weights)...")
        pipe = YuE2Pipeline.from_pretrained(REPO, device="cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline loaded.")
    return pipe


def flac_to_wav(flac_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-c:a", "pcm_s16le", wav_path],
        capture_output=True, check=True,
    )
    return wav_path


def flac_to_mp3(flac_path, mp3_path, bitrate="320k"):
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-b:a", bitrate, mp3_path],
        capture_output=True, check=True,
    )
    return mp3_path


def export_score(song, song_id):
    import soundfile as _sf  # noqa: F401

    art_dir = os.path.join(OUT, f"art_{song_id}")
    try:
        song.save_artifacts(art_dir)
    except Exception as e:
        print(f"[score] save_artifacts failed: {e}")
        return None, None
    found = sorted(glob.glob(os.path.join(art_dir, "**", "*.abc"), recursive=True))
    if not found:
        print(f"[score] no .abc in {art_dir} (cot=off?)")
        return None, None
    abc_path = os.path.join(OUT, f"{song_id}.abc")
    shutil.copyfile(found[0], abc_path)
    midi_path = os.path.join(OUT, f"{song_id}.mid")
    try:
        subprocess.run(["abc2midi", abc_path, "-o", midi_path], capture_output=True, text=True, check=True)
        if not os.path.exists(midi_path):
            midi_path = None
    except Exception as e:
        print(f"[score] abc2midi failed: {e}")
        midi_path = None
    return abc_path, midi_path


def get_duration_seconds(audio_path):
    try:
        import soundfile as sf
        return float(sf.info(audio_path).duration)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _upload_to_my_bucket(wav_path, meta_dict, song_id, abc_path=None, midi_path=None):
    """Silently upload songs/<id>/<id>.wav (+ .abc/.mid) + meta.json to the private bucket. Never raises."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=token)
        meta_local = os.path.join(OUT, f"{song_id}_meta.json")
        with open(meta_local, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)
        base = f"buckets/{MY_BUCKET}/songs/{song_id}"
        fs.put_file(wav_path, f"{base}/{song_id}.wav")
        if abc_path and os.path.exists(abc_path):
            fs.put_file(abc_path, f"{base}/{song_id}.abc")
        if midi_path and os.path.exists(midi_path):
            fs.put_file(midi_path, f"{base}/{song_id}.mid")
        fs.put_file(meta_local, f"{base}/meta.json")
        print(f"[my-bucket] uploaded {song_id}")
    except Exception as e:
        print(f"[my-bucket] upload failed: {e}")


def bucket_url(song_id, filename):
    return f"https://huggingface.co/buckets/{MY_BUCKET}/resolve/main/songs/{song_id}/{filename}"


@app.post("/")
def generate_song(payload: GenerationRequest):
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    try:
        from torch import cuda
        has_gpu = cuda.is_available()
    except Exception:
        has_gpu = False

    try:
        if not is_content_safe(payload.style, payload.lyrics):
            raise HTTPException(
                status_code=400,
                detail="Error: Nice try! But our firewall is smarter than your prompt. "
                       "Try writing something actual music-related.",
            )

        pipeline = get_pipeline()

        seed = payload.seed if payload.seed > 0 else random.randint(1, 10**6)
        seed = int(seed)

        gen_kwargs = dict(
            style=payload.style,
            lyrics=payload.lyrics,
            cot=payload.cot,
            seed=seed,
        )
        if payload.cfg_scale and float(payload.cfg_scale) != 1.0:
            gen_kwargs["cfg_scale"] = float(payload.cfg_scale)

        song = pipeline(**gen_kwargs)

        song_id = secrets.token_hex(6)
        flac_path = os.path.join(OUT, f"song_{song_id}.flac")
        wav_path = os.path.join(OUT, f"{song_id}.wav")
        song.save(flac_path)
        flac_to_wav(flac_path, wav_path)
        try:
            flac_to_mp3(flac_path, os.path.join(OUT, f"song_{song_id}.mp3"))
        except Exception as e:
            print(f"[mp3] skipped: {e}")

        abc_path, midi_path = export_score(song, song_id)

        words = (payload.style or "").strip().replace("\n", " ").split()[:8]
        title = " ".join(words).title()[:80] if words else f"YuE2 Song {song_id[:6]}"
        actual = get_duration_seconds(wav_path)

        meta = {
            "id": song_id,
            "title": title,
            "description": (payload.style or "").strip(),
            "lyrics": payload.lyrics or "",
            "caption": (payload.style or "").strip(),
            "bpm": 120,
            "language": "",
            "duration": actual,
            "seed": seed,
            "generation_mode": payload.cot or "auto",
            "audio_file": f"{song_id}.wav",
            "abc_file": f"{song_id}.abc" if abc_path else None,
            "midi_file": f"{song_id}.mid" if midi_path else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _upload_to_my_bucket(wav_path, meta, song_id, abc_path=abc_path, midi_path=midi_path)

        links = {"wav": bucket_url(song_id, f"{song_id}.wav")}
        if abc_path and os.path.exists(abc_path):
            links["abc"] = bucket_url(song_id, f"{song_id}.abc")
        if midi_path and os.path.exists(midi_path):
            links["mid"] = bucket_url(song_id, f"{song_id}.mid")
        links["meta"] = bucket_url(song_id, "meta.json")

        return {
            "status": "success",
            "id": song_id,
            "seed": seed,
            "title": title,
            "duration_seconds": round(actual, 1),
            "links": links,
        }

    except HTTPException:
        raise
    except Exception as e:
        send_to_telegram(f"❌ Generation Failed!\nError: {str(e)[:500]}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
