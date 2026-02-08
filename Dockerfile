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
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone whisper-asr-webservice
RUN git clone --depth 1 https://github.com/ahmetoner/whisper-asr-webservice.git /app
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
    click \
    more-itertools \
    ffmpeg-python

# Install openai-whisper without deps (only needed for tokenizer.LANGUAGES)
RUN pip install --no-cache-dir --no-deps openai-whisper 2>/dev/null || true

# Set ASR config early so patches and verifications use the right engine
ENV ASR_ENGINE=faster_whisper
ENV ASR_MODEL=large-v3
ENV ASR_QUANTIZATION=float16
ENV ASR_DEVICE=cuda

# Patch ALL Python files to handle missing torch gracefully
# (base image has CTranslate2/faster-whisper but no PyTorch)
RUN <<'PATCH_SCRIPT'
python3 -c "
import pathlib, glob

# Patch all files that do 'import torch'
for fpath in glob.glob('/app/app/**/*.py', recursive=True):
    p = pathlib.Path(fpath)
    src = p.read_text()
    if 'import torch' in src and 'try:' not in src.split('import torch')[0][-20:]:
        patched = src.replace(
            'import torch',
            'try:\n    import torch\nexcept ImportError:\n    torch = None'
        )
        # Also guard torch.cuda.is_available() calls
        patched = patched.replace(
            'torch.cuda.is_available()',
            '(torch is not None and torch.cuda.is_available())'
        )
        # Guard torch.cuda.empty_cache() calls
        patched = patched.replace(
            'torch.cuda.empty_cache()',
            '(torch.cuda.empty_cache() if torch is not None else None)'
        )
        p.write_text(patched)
        print(f'Patched: {fpath}')
    else:
        print(f'OK (no patch needed): {fpath}')
"
PATCH_SCRIPT

# Patch factory to use lazy imports (avoid loading openai_whisper/whisperx engines we don't use)
RUN <<'FACTORY_PATCH'
python3 -c "
import pathlib
p = pathlib.Path('/app/app/factory/asr_model_factory.py')
p.write_text('''from app.config import CONFIG


class ASRModelFactory:
    @staticmethod
    def create_asr_model():
        if CONFIG.ASR_ENGINE == \"faster_whisper\":
            from app.asr_models.faster_whisper_engine import FasterWhisperASR
            return FasterWhisperASR()
        elif CONFIG.ASR_ENGINE == \"openai_whisper\":
            from app.asr_models.openai_whisper_engine import OpenAIWhisperASR
            return OpenAIWhisperASR()
        elif CONFIG.ASR_ENGINE == \"whisperx\":
            from app.asr_models.mbain_whisperx_engine import WhisperXASR
            return WhisperXASR()
        else:
            raise ValueError(f\"Unsupported ASR engine: {CONFIG.ASR_ENGINE}\")
''')
print('Patched factory for lazy imports')
"
FACTORY_PATCH

# If openai-whisper tokenizer is not available, create a minimal shim
RUN <<'SHIM_SCRIPT'
python3 -c "from whisper import tokenizer; print('whisper tokenizer OK:', len(tokenizer.LANGUAGES), 'languages')" 2>/dev/null && exit 0

python3 -c "
import site, os, pathlib

langs = {
    'en':'english','zh':'chinese','de':'german','es':'spanish','ru':'russian',
    'ko':'korean','fr':'french','ja':'japanese','pt':'portuguese','tr':'turkish',
    'pl':'polish','ca':'catalan','nl':'dutch','ar':'arabic','sv':'swedish',
    'it':'italian','id':'indonesian','hi':'hindi','fi':'finnish','vi':'vietnamese',
    'he':'hebrew','uk':'ukrainian','el':'greek','ms':'malay','cs':'czech',
    'ro':'romanian','da':'danish','hu':'hungarian','ta':'tamil','no':'norwegian',
    'th':'thai','ur':'urdu','hr':'croatian','bg':'bulgarian','lt':'lithuanian',
    'la':'latin','mi':'maori','ml':'malayalam','cy':'welsh','sk':'slovak',
    'te':'telugu','fa':'persian','lv':'latvian','bn':'bengali','sr':'serbian',
    'az':'azerbaijani','sl':'slovenian','kn':'kannada','et':'estonian',
    'mk':'macedonian','br':'breton','eu':'basque','is':'icelandic','hy':'armenian',
    'ne':'nepali','mn':'mongolian','bs':'bosnian','kk':'kazakh','sq':'albanian',
    'sw':'swahili','gl':'galician','mr':'marathi','pa':'panjabi','si':'sinhala',
    'km':'khmer','sn':'shona','yo':'yoruba','so':'somali','af':'afrikaans',
    'oc':'occitan','ka':'georgian','be':'belarusian','tg':'tajik','sd':'sindhi',
    'gu':'gujarati','am':'amharic','yi':'yiddish','lo':'lao','uz':'uzbek',
    'fo':'faroese','ht':'haitian creole','ps':'pashto','tk':'turkmen',
    'nn':'nynorsk','mt':'maltese','sa':'sanskrit','lb':'luxembourgish',
    'my':'myanmar','bo':'tibetan','tl':'tagalog','mg':'malagasy','as':'assamese',
    'tt':'tatar','haw':'hawaiian','ln':'lingala','ha':'hausa','ba':'bashkir',
    'jw':'javanese','su':'sundanese','yue':'cantonese',
}

sp = site.getsitepackages()[0]
whisper_dir = os.path.join(sp, 'whisper')
os.makedirs(whisper_dir, exist_ok=True)
pathlib.Path(os.path.join(whisper_dir, '__init__.py')).write_text(
'''import numpy as np

SAMPLE_RATE = 16000
N_SAMPLES = SAMPLE_RATE * 30  # 30 seconds

def pad_or_trim(array, length=N_SAMPLES, *, axis=-1):
    if array.shape[axis] > length:
        array = array[..., :length]
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)
    return array
''')
pathlib.Path(os.path.join(whisper_dir, 'tokenizer.py')).write_text('LANGUAGES = ' + repr(langs))
print('Created whisper tokenizer shim with', len(langs), 'languages')
"
SHIM_SCRIPT

# Verify critical imports work
RUN python3 -c "from whisper import tokenizer; print('tokenizer OK:', len(tokenizer.LANGUAGES), 'langs')"
RUN python3 -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
RUN python3 -c "from app.config import CONFIG; print('config OK: engine=' + CONFIG.ASR_ENGINE)"

# Install the webservice package (for the CLI entry point)
RUN pip install --no-cache-dir --no-deps -e . 2>/dev/null || \
    (printf '#!/usr/bin/env python3\nfrom app.webservice import start\nstart()\n' > /usr/local/bin/whisper-asr-webservice && \
     chmod +x /usr/local/bin/whisper-asr-webservice)

# Model cache directory
RUN mkdir -p /root/.cache/huggingface

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:9000/docs > /dev/null || exit 1

CMD ["whisper-asr-webservice", "--host", "0.0.0.0", "--port", "9000"]
