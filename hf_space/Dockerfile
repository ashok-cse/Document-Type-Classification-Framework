# Hugging Face Space (Docker SDK) — full-TensorFlow DTCF build with live Grad-CAM.
# This Space repo only needs this Dockerfile + README.md; the app code and the
# trained model are pulled from GitHub at build time, so no large files / git-LFS
# are stored in the Space.
FROM python:3.11-slim

# HF Spaces runs as a non-root user and only /tmp + $HOME are reliably writable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    LOCAL_MODEL_PATH=/app/models/resnet50.keras \
    EAGER_LOAD=1 \
    PORT=7860 \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# git (to clone the source) + Pillow's runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libjpeg62-turbo libpng16-16 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pull the app code and the baked model from the public GitHub repo.
# Bump CACHE_BUST (or use the Space "Factory rebuild") to pick up new commits.
ARG REPO=https://github.com/ashok-cse/Document-Type-Classification-Framework
ARG CACHE_BUST=2026-06-28
RUN git clone --depth 1 "$REPO" /tmp/src \
    && cp -r /tmp/src/app /app/app \
    && mkdir -p /app/models \
    && cp /tmp/src/models/resnet50.keras /app/models/resnet50.keras \
    && rm -rf /tmp/src

RUN pip install -r /app/app/requirements.txt

# World-readable so the non-root Space user can read app + model.
RUN chmod -R a+rX /app && mkdir -p /tmp/hf /tmp/mpl && chmod -R 777 /tmp/hf /tmp/mpl

EXPOSE 7860

# Single worker; HF routes to the port declared as app_port in README.md.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
