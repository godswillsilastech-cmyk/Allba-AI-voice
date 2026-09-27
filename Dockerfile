FROM python:3.10-slim

# git: to clone the yarngpt repo (it's not a pip package)
# wget: to fetch the WavTokenizer config + checkpoint
# ffmpeg: torchaudio's backend for reading/writing wav files
# build-essential: some deps (uroman, inflect) need to compile small extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- YarnGPT is a plain GitHub repo, not a pip package ---
# Cloning it here gives us /app/yarngpt/audiotokenizer.py plus the bundled
# default_speakers_local/*.json speaker prompt files it needs at runtime.
RUN git clone --depth 1 https://github.com/saheedniyi02/yarngpt.git yarngpt

# --- WavTokenizer decoder assets (turns YarnGPT's audio "codes" into a waveform) ---
# Config file: small YAML, straight HF download.
RUN wget -q -O wavtokenizer_config.yaml \
    https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml

# Checkpoint (~1.75GB).
#
# CHANGED: pulled from Hugging Face instead of Google Drive. Google Drive's
# gdown route frequently rate-limits or blocks automated downloads from cloud
# build servers (Render's included), which was the single most likely cause
# of a build failure in the original setup. HF's CDN doesn't have that problem.
#
# NOTE: the upstream repo (novateur/WavTokenizer-large-speech-75token) has
# renamed this file before -- confirm the filename below still matches
# https://huggingface.co/novateur/WavTokenizer-large-speech-75token/tree/main
# before you build, since a 404 here will fail the build.
RUN wget -q -O wavtokenizer_model.ckpt \
    https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_v2.ckpt \
 && test -s wavtokenizer_model.ckpt \
 || (echo "ERROR: WavTokenizer checkpoint download failed or returned an empty file" && exit 1)

# --- Pre-cache the language model + tokenizer weights into the image ---
# This means the running container never has to hit the network for model
# weights: only the (still lazy) step of loading them into RAM happens at
# first request, which is what keeps idle memory low.
ENV HF_HOME=/app/hf_cache
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
MODEL='saheedniyi/YarnGPT2'; \
AutoModelForCausalLM.from_pretrained(MODEL); \
AutoTokenizer.from_pretrained(MODEL)"

COPY app.py .

ENV PORT=8000
EXPOSE 8000

# Render (like Railway) injects $PORT at runtime; sh -c lets us expand it.
#
# CHANGED: added --workers 1 explicitly. The model is a global singleton
# loaded into a single process's RAM on first request. If uvicorn/gunicorn
# ever defaulted to >1 worker, each worker would load its own full copy of
# the model, multiplying memory use and making an OOM crash on small
# instances even more likely.
#
# CHANGED: added --proxy-headers so uvicorn trusts Render's reverse proxy
# for client IP / https scheme (Render terminates TLS in front of your app).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
