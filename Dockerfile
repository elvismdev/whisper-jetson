# syntax=docker/dockerfile:1
# =============================================================
# Whisper ASR Webservice for Jetson AGX Orin
# Base: dustynv/faster-whisper (pre-built CTranslate2 for ARM64)
# Web UI: whisper-asr-webservice (Swagger UI)
# Engine: faster-whisper + CTranslate2 (CUDA)
# =============================================================
ARG BASE_IMAGE=dustynv/faster-whisper:r36.4.0-cu128-24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source=https://github.com/elvismdev/whisper-jetson
LABEL org.opencontainers.image.description="GPU-accelerated Whisper ASR for NVIDIA Jetson"
LABEL org.opencontainers.image.license=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Fix pip to use standard PyPI (base image only has Jetson AI Lab index which may be unreachable)
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu128

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download Swagger UI assets for offline serving
RUN mkdir -p swagger-ui-assets && \
    curl -sL "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.1/swagger-ui.css" \
        -o swagger-ui-assets/swagger-ui.css && \
    curl -sL "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.1/swagger-ui-bundle.js" \
        -o swagger-ui-assets/swagger-ui-bundle.js

# Install Python dependencies the webservice needs
# (faster-whisper and ctranslate2 are already in the base image)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    ffmpeg-python

# ASR configuration
ENV ASR_ENGINE=faster_whisper
ENV ASR_MODEL=large-v3
ENV ASR_QUANTIZATION=float16
ENV ASR_DEVICE=cuda

# Copy vendored app code (patches already applied in source)
COPY app/ /app/app/

# Model cache directory
RUN mkdir -p /root/.cache/huggingface

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:9000/docs > /dev/null || exit 1

CMD ["python3", "-m", "uvicorn", "app.webservice:app", "--host", "0.0.0.0", "--port", "9000"]
