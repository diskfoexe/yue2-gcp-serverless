FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# הורדה ידנית של הריפוזיטורי של YuE והתקנה מקומית שלו
RUN git clone https://github.com/m-a-p/YuE.git /tmp/YuE && \
    cd /tmp/YuE && \
    pip install --no-cache-dir -e .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
