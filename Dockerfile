# DTCF inference service — CPU-only image suitable for Easypanel.
# The trained model is BAKED INTO the image, so the container is self-contained:
# no HuggingFace download and no network needed at runtime.
# Build:  docker build -t dtcf .
# Run:    docker run -p 8000:8000 dtcf
#
# To serve a model from HuggingFace Hub instead, override at runtime:
#   docker run -p 8000:8000 -e LOCAL_MODEL_PATH= -e HF_MODEL_REPO=<user>/<repo> dtcf
# (an empty LOCAL_MODEL_PATH disables the baked-in model and falls through to HF).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    LOCAL_MODEL_PATH=/models/resnet50.keras \
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

# Bake the trained model into the image (resolved via LOCAL_MODEL_PATH above).
# ~100 MB; keeps the container self-contained with no runtime download.
COPY models/resnet50.keras /models/resnet50.keras

# HF cache dir still exists for the optional HuggingFace fallback path — mount
# as a volume in Easypanel if you switch to HF_MODEL_REPO instead of the baked model.
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

# Container-level health check so Easypanel can detect an unhealthy worker.
# Honours $PORT (Render injects it; Easypanel uses the ENV default of 8000).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,sys,json,urllib.request; p=os.environ.get('PORT','8000'); sys.exit(0 if json.load(urllib.request.urlopen(f'http://127.0.0.1:{p}/health'))['model_loaded'] else 1)"

# Single worker — TF inference is CPU-heavy; scale horizontally instead.
# Shell form so $PORT is expanded: Render sets PORT (default 10000); the ENV
# default above keeps Easypanel/local on 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
