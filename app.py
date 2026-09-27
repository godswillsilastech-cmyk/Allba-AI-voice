"""
YarnGPT Yoruba TTS API
----------------------
Wraps the YarnGPT2 model (saheedniyi/YarnGPT2) to turn Yoruba text into
speech. The model is a small (~360M param) language model that predicts
audio "codes", which a separate WavTokenizer decoder turns into an actual
waveform. Both are lazy-loaded on first request to keep idle RAM low.

Real upstream voices for Yoruba are: yoruba_male2, yoruba_female1, yoruba_female2
NOTE: "zainab" is an ENGLISH voice in the real YarnGPT model, not Yoruba.
Using it here would generate English-accented speech, not Yoruba, so it is
intentionally excluded from this Yoruba-only endpoint. Default is
"yoruba_female2" as the closest equivalent.
"""

import os
import uuid
import threading

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from transformers import AutoModelForCausalLM
from yarngpt.audiotokenizer import AudioTokenizerV2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "saheedniyi/YarnGPT2"  # HF hub id, weights are pre-cached in the image at build time

# These paths must match where the Dockerfile downloaded the WavTokenizer assets.
WAV_TOKENIZER_CONFIG_PATH = "/app/wavtokenizer_config.yaml"
WAV_TOKENIZER_MODEL_PATH = "/app/wavtokenizer_model.ckpt"

# Only real Yoruba voices from the upstream model. "zainab" deliberately excluded (see note above).
YORUBA_VOICES = {"yoruba_male2", "yoruba_female1", "yoruba_female2"}
DEFAULT_VOICE = "yoruba_female2"

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "400"))  # keep generations bounded so CPU inference finishes in reasonable time

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="YarnGPT Yoruba TTS")

# Allow the Firebase-hosted frontend (or anything) to call this API.
# allow_credentials must stay False for allow_origins=["*"] to be valid in browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------

_audio_tokenizer = None
_model = None
_load_lock = threading.Lock()          # guards first-time loading
_generate_lock = threading.Lock()      # serializes generation (CPU/RAM constrained trial instance)


def _get_pipeline():
    """Load the audio tokenizer + language model into RAM, once, on first use."""
    global _audio_tokenizer, _model

    if _model is not None:
        return _audio_tokenizer, _model

    with _load_lock:
        # Re-check in case another request loaded it while we waited on the lock.
        if _model is None:
            torch.set_num_threads(os.cpu_count() or 1)

            audio_tokenizer = AudioTokenizerV2(
                MODEL_NAME,
                WAV_TOKENIZER_MODEL_PATH,
                WAV_TOKENIZER_CONFIG_PATH,
            )
            # CHANGED: torch_dtype forced to float32 instead of "auto". "auto"
            # loads whatever dtype the checkpoint was saved in -- if that's
            # float16/bfloat16, several ops are slow or unsupported on CPU.
            # We're CPU-only here (see requirements.txt), so float32 is the
            # safe, predictable choice.
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float32
            ).to(audio_tokenizer.device)
            model.eval()

            _audio_tokenizer, _model = audio_tokenizer, model

    return _audio_tokenizer, _model


def _generate_wav(text: str, voice: str) -> str:
    """Runs the actual (slow, CPU-bound) generation. Returns path to a temp .wav file."""
    audio_tokenizer, model = _get_pipeline()

    with _generate_lock:  # only one generation at a time on this small instance
        prompt = audio_tokenizer.create_prompt(text, lang="yoruba", speaker_name=voice)
        input_ids = audio_tokenizer.tokenize_prompt(prompt)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                temperature=0.1,
                repetition_penalty=1.1,
                max_length=4000,
            )

        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)

    out_path = f"/tmp/{uuid.uuid4().hex}.wav"
    torchaudio.save(out_path, audio, sample_rate=24000)
    return out_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Lightweight health check. Does NOT touch the model, so it stays fast
    even before the model has been loaded, and works as Render's healthcheck.

    NOTE: because this deliberately never touches the model, it will report
    "ok" even on an instance that doesn't have enough RAM to actually run
    inference -- that failure only shows up on the first real /tts call.
    Make sure your Render plan has enough memory headroom for a ~360M param
    model + WavTokenizer decoder + torch itself (see DEPLOYMENT_NOTES)."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/voices")
def list_voices():
    """Lets the frontend populate its voice dropdown without hardcoding it."""
    return {"voices": sorted(YORUBA_VOICES), "default": DEFAULT_VOICE}


@app.post("/tts")
async def tts(
    text: str = Query(..., description="Yoruba text to synthesize"),
    voice: str = Query(DEFAULT_VOICE, description="One of: " + ", ".join(sorted(YORUBA_VOICES))),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"`text` is too long ({len(text)} chars). Limit is {MAX_TEXT_LENGTH} for this test deployment.",
        )
    if voice not in YORUBA_VOICES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown voice '{voice}'. Valid Yoruba voices: {sorted(YORUBA_VOICES)}. "
                "Note: 'zainab' is an English voice in the real model, not Yoruba, so it isn't offered here."
            ),
        )

    try:
        # Run the blocking, CPU-heavy generation in a threadpool so the event
        # loop (and the health check) stays responsive while it runs.
        wav_path = await run_in_threadpool(_generate_wav, text, voice)
    except Exception as exc:  # surface a clean error instead of a raw 500 traceback
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    # Delete the temp file after it's been sent, so /tmp doesn't fill up over the test period.
    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(os.remove, wav_path),
    )
