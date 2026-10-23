FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# שכפול הריפו והתקנה רשמית של החבילה בתוך פייתון
RUN GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git /tmp/YuE && \
    cd /tmp/YuE && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/YuE

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
