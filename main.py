from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import secrets
import subprocess

app = FastAPI()

pipe = None
OUT = "/tmp/yue2_out"
os.makedirs(OUT, exist_ok=True)

class GenerationRequest(BaseModel):
    style: str
    lyrics: str
    seed: int = 0
    cfg_scale: float = 1.0

def get_pipeline():
    global pipe
    if pipe is None:
        from yue2 import YuE2Pipeline
        print("Loading YuE2-3B pipeline on GCP L4 GPU...")
        pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline loaded successfully.")
    return pipe

@app.post("/")
def generate_song(payload: GenerationRequest):
    try:
        pipeline = get_pipeline()
        seed = payload.seed if payload.seed > 0 else int(torch.randint(0, 10**6, (1,)).item())
        
        song = pipeline(
            style=payload.style,
            lyrics=payload.lyrics,
            seed=seed,
            cfg_scale=payload.cfg_scale
        )

        song_id = secrets.token_hex(6)
        flac_path = os.path.join(OUT, f"song_{song_id}.flac")
        mp3_path = os.path.join(OUT, f"song_{song_id}.mp3")
        
        song.save(flac_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", flac_path, "-b:a", "320k", mp3_path],
            capture_output=True, check=True,
        )

        return {
            "status": "success",
            "seed": seed,
            "mp3_file": mp3_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
