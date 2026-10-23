FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg abcmidi && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY yue2_infer-0.1.5-py3-none-any.whl .
RUN pip install --no-cache-dir ./yue2_infer-0.1.5-py3-none-any.whl

COPY main.py .

ENV HF_HOME=/tmp/hf

EXPOSE 8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]
