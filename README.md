# Whisper ASR on Jetson

GPU-accelerated speech-to-text service for **NVIDIA Jetson AGX Orin** devices. Provides a web UI (Swagger) for drag-and-drop audio transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2) with CUDA acceleration.

Built on top of:
- [dustynv/faster-whisper](https://github.com/dusty-nv/jetson-containers) — pre-built ARM64 container with CTranslate2 + CUDA for Jetson
- [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) — FastAPI app with Swagger UI for audio transcription

## Features

- **GPU-accelerated** transcription using CTranslate2 (CUDA) — ~4x faster than PyTorch Whisper
- **Web interface** (Swagger UI) for uploading and transcribing audio files
- **large-v3 model** by default for maximum accuracy (especially good for Spanish)
- **float16 quantization** for optimal speed/accuracy balance on Jetson's unified memory
- **Auto-starts on boot** via systemd
- **100+ languages** supported
- **Multiple output formats**: text, JSON, VTT, SRT, TSV

## Requirements

- NVIDIA Jetson AGX Orin (64GB recommended)
- JetPack 6.x (tested on JetPack 6.2.1 / R36.4.7)
- Docker with NVIDIA runtime (`--runtime nvidia`)
- ~20 GB disk space (container image ~17.5 GB + model ~3 GB)

## Quick Start (Pull Pre-built Image)

This is the fastest way to get up and running. The pre-built image is hosted on GitHub Container Registry and includes everything needed.

### 1. Pull the image

```bash
sudo docker pull ghcr.io/elvismdev/whisper-jetson:latest
```

### 2. Pre-download the model (recommended)

This downloads the `large-v3` model (~3 GB) into a persistent Docker volume so the first transcription is instant rather than waiting for the download.

```bash
sudo docker volume create whisper-models

sudo docker run --rm --runtime nvidia \
    -v whisper-models:/root/.cache/huggingface \
    ghcr.io/elvismdev/whisper-jetson:latest \
    python3 -c "
from faster_whisper import WhisperModel
print('Downloading large-v3 model...')
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('Model downloaded and ready.')
"
```

This will take a few minutes depending on your internet speed. The model is cached in the `whisper-models` Docker volume and persists across container restarts and reboots.

### 3. Run the service

```bash
sudo docker run -d --runtime nvidia --name whisper-asr --network host \
    -e ASR_ENGINE=faster_whisper \
    -e ASR_MODEL=large-v3 \
    -e ASR_QUANTIZATION=float16 \
    -e ASR_DEVICE=cuda \
    -v whisper-models:/root/.cache/huggingface \
    --restart unless-stopped \
    ghcr.io/elvismdev/whisper-jetson:latest
```

### 4. Open the web UI

Open your browser and go to:

```
http://<your-jetson-ip>:9000/docs
```

You'll see the Swagger UI where you can:
1. Expand the **POST /asr** endpoint
2. Click **Try it out**
3. Upload an audio file (WAV, MP3, M4A, FLAC, OGG, etc.)
4. Select the language (or leave blank for auto-detect)
5. Click **Execute** to transcribe

## Auto-Start on Boot (systemd)

To have the service start automatically when the Jetson boots, copy the included `whisper-asr.service` file from this repo:

### 1. Copy the service file

```bash
sudo cp whisper-asr.service /etc/systemd/system/
```

### 2. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable whisper-asr.service
sudo systemctl start whisper-asr.service
```

### 3. Check the status

```bash
sudo systemctl status whisper-asr.service
```

### Managing the service

```bash
# View logs
sudo docker logs -f whisper-asr

# Restart the service
sudo systemctl restart whisper-asr.service

# Stop the service
sudo systemctl stop whisper-asr.service

# Disable auto-start on boot
sudo systemctl disable whisper-asr.service
```

## Build from Source

If you prefer to build the Docker image yourself (or want to customize it):

### 1. Clone this repo

```bash
git clone https://github.com/elvismdev/whisper-jetson.git
cd whisper-jetson
```

### 2. Build the image

```bash
sudo docker build -t ghcr.io/elvismdev/whisper-jetson:latest .
```

> **Note:** This must be built on an ARM64 Jetson device (or using ARM64 emulation). The base image `dustynv/faster-whisper:r36.4.0-cu128-24.04` is ARM64/Jetson-specific.

The build takes roughly 5-10 minutes depending on network speed (it downloads the webservice code, Swagger UI assets, and Python dependencies).

### 3. Run

Follow steps 2-4 from the [Quick Start](#quick-start-pull-pre-built-image) section above.

## Configuration

Environment variables you can pass to the container:

| Variable | Default | Description |
|---|---|---|
| `ASR_ENGINE` | `faster_whisper` | ASR engine (`faster_whisper`, `openai_whisper`, `whisperx`) |
| `ASR_MODEL` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) |
| `ASR_QUANTIZATION` | `float16` | Compute type (`float16`, `int8_float16`, `int8`) |
| `ASR_DEVICE` | `cuda` | Device (`cuda` or `cpu`) |

### Model sizes

| Model | Parameters | Required VRAM | Relative Speed |
|---|---|---|---|
| `tiny` | 39M | ~1 GB | ~10x |
| `base` | 74M | ~1 GB | ~7x |
| `small` | 244M | ~2 GB | ~4x |
| `medium` | 769M | ~5 GB | ~2x |
| `large-v3` | 1550M | ~10 GB | 1x |

The Jetson AGX Orin 64GB has more than enough unified memory for `large-v3` with `float16`. For smaller Jetson devices (Orin NX/Nano), consider using `medium` or `small`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/asr` | POST | Transcribe an audio file |
| `/detect-language` | POST | Detect the language of an audio file |
| `/docs` | GET | Swagger UI (web interface) |

### Example: Transcribe via cURL

```bash
curl -X POST "http://<your-jetson-ip>:9000/asr?output=json&language=es" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "audio_file=@recording.wav"
```

### Example: Auto-detect language and transcribe

```bash
curl -X POST "http://<your-jetson-ip>:9000/asr?output=txt" \
    -F "audio_file=@recording.mp3"
```

## CI/CD

This repo uses a **self-hosted GitHub Actions runner** on the Jetson itself. Pushing to `main` (when the `Dockerfile` or workflow changes) automatically:

1. Builds the image on the Jetson
2. Pushes it to GHCR with `:latest` and `:sha-<commit>` tags
3. Restarts the `whisper-asr` systemd service with the new image
4. Cleans up old image tags and build cache

The workflow is defined in `.github/workflows/build-and-push.yml`.

## Updating

If you're running this on a **different Jetson** (not the build runner), pull the latest image and restart:

```bash
sudo docker pull ghcr.io/elvismdev/whisper-jetson:latest
sudo systemctl restart whisper-asr.service
```

If running manually without systemd:

```bash
sudo docker stop whisper-asr && sudo docker rm whisper-asr
# Then run the docker run command from Quick Start step 3
```

## Troubleshooting

### Check if the GPU is being used

```bash
# Watch GPU utilization in real-time
tegrastats

# Look for GR3D_FREQ — should spike to 70-99% during transcription
```

### Check container logs

```bash
sudo docker logs whisper-asr --tail 50
```

### Model not loading / slow first request

If you skipped the model pre-download step, the first transcription request will trigger a download of the model (~3 GB), which can take 1-2 minutes. Run the pre-download command from Quick Start step 2 to avoid this.

### Port 9000 already in use

Change the port by modifying the `docker run` command:

```bash
sudo docker run -d --runtime nvidia --name whisper-asr \
    -p 8080:9000 \
    -e ASR_ENGINE=faster_whisper \
    -e ASR_MODEL=large-v3 \
    -e ASR_QUANTIZATION=float16 \
    -e ASR_DEVICE=cuda \
    -v whisper-models:/root/.cache/huggingface \
    ghcr.io/elvismdev/whisper-jetson:latest
```

Then access the UI at `http://<your-jetson-ip>:8080/docs`.

## License

MIT
