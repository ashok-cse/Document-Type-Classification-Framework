# DTCF inference service — CPU-only image suitable for Easypanel.
# Build:  docker build -t dtcf .
# Run:    docker run -p 8000:8000 -e HF_MODEL_REPO=<user>/<repo> dtcf

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PORT=8000

WORKDIR /app

# System packages required by Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximise Docker layer caching.
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install -r /app/app/requirements.txt

# Copy the application code.
COPY app /app/app

# Persistent model cache lives here — mount as a volume in Easypanel
# so the model isn't re-downloaded on every restart.
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

# Single worker — TF inference is CPU-heavy; scale horizontally instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
