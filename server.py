"""OpenAI-compatible TTS/STT server with cross-platform backend support.

Supports:
- Kokoro TTS (PyTorch — works on CPU, MPS/Apple Silicon, CUDA)
- faster-whisper STT (CUDA on NVIDIA GPUs)
- mlx-whisper STT (Apple Silicon / MLX)

Auto-detects hardware at startup and picks the best backend.
"""

import io
import json
import logging
import os
import platform
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts-server")

SAMPLE_RATE = 24000

# OpenAI voice name -> Kokoro voice name
VOICE_MAP = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "shimmer": "af_sky",
}

# Language code prefix -> KPipeline lang_code
LANG_PREFIXES = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Portuguese
    "z": "z",  # Mandarin Chinese
}

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# Backend dispatch
pipelines: dict = {}

# STT backends (populated at startup)
# On NVIDIA: whisper_model = faster_whisper.WhisperModel
# On Apple Silicon (MLX): whisper_stt = mlx_whisper module
whisper_backend = "none"  # 'cuda', 'mlx', 'cpu', or 'none'
whisper_model = None  # faster_whisper.WhisperModel (CUDA/CPU)
ml_stt = None  # mlx_stt object (Apple Silicon)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_stt_backend() -> tuple:
    """Detect available STT backend and return name + loaded model/module.

    Priority: CUDA (faster-whisper) > MLX (mlx-whisper) > CPU (faster-whisper).
    Returns (backend_name, model_or_module).
    """
    system = platform.system()
    
    # Check for CUDA first (NVIDIA GPU)
    try:
        import torch
        if torch.cuda.is_available():
            from faster_whisper import WhisperModel
            logger.info("CUDA detected — using faster-whisper (CUDA)")
            return "cuda", WhisperModel(
                "large-v3",
                device="cuda",
                device_index=0,
                compute_type="float16",
            )
    except ImportError:
        pass
    
    # Check for CUDA but not available (NVIDIA driver but no GPU)
    try:
        import torch
        if torch.cuda.is_available():
            from faster_whisper import WhisperModel
            return "cuda", WhisperModel(
                "large-v3",
                device="cuda",
                device_index=0,
                compute_type="float16",
            )
    except ImportError:
        pass
    
    # Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx
        if system == "Darwin" and mx.metal.is_available():
            import mlx_whisper
            logger.info("Apple Silicon (MPS/MLX) detected — using mlx-whisper")
            return "mlx", mlx_whisper
    except ImportError:
        pass
    
    # Check for CUDA availability but CTranslate2 lacks it
    try:
        from faster_whisper import WhisperModel
        import ctranslate2
        # Try CUDA first
        try:
            model = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
            logger.info("faster-whisper (CUDA) available")
            return "cuda", model
        except (ValueError, Exception):
            logger.info("CTranslate2 lacks CUDA — falling back to CPU")
            cpu_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            return "cpu", cpu_model
    except ImportError:
        pass
    
    # Default: CPU fallback
    logger.warning("No GPU/MLX detected — using CPU for STT (slow)")
    try:
        from faster_whisper import WhisperModel
        import ctranslate2
        return "cpu", WhisperModel("large-v3", device="cpu", compute_type="int8")
    except ImportError:
        return "none", None


# ---------------------------------------------------------------------------
# Transcription helper — abstracts away CUDA vs MLX calls
# ---------------------------------------------------------------------------

def transcribe_audio(file_path: str, language: Optional[str] = None) -> tuple:
    """Transcribe an audio file using the detected backend.
    
    Returns (text, info_dict) where info_dict contains:
        language, language_probability, duration
    """
    if whisper_backend == "mlx":
        result = ml_stt.transcribe(
            file_path,
            language=language,
            beam_size=1,
            word_timestamps=False,
        )
        info = {
            "language": result.get("language", "unknown"),
            "language_probability": result.get("language_probs", {}).get("unknown", 0.0),
            "duration": result.get("duration", 0.0),
        }
        return result["text"], info
    
    elif whisper_backend in ("cuda", "cpu"):
        segments, info = whisper_model.transcribe(file_path, language=language)
        text = " ".join(seg.text.strip() for seg in segments)
        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
        return text, info_dict
    
    else:
        raise RuntimeError("No STT backend loaded. Check startup logs.")


# ---------------------------------------------------------------------------
# Kokoro pipeline (always PyTorch — works on CPU, MPS, CUDA)
# ---------------------------------------------------------------------------

def get_pipeline(lang_code: str):
    """Get or create a Kokoro KPipeline for the given language code."""
    from kokoro import KPipeline
    if lang_code not in pipelines:
        logger.info(f"Loading KPipeline lang_code='{lang_code}'...")
        pipelines[lang_code] = KPipeline(lang_code=lang_code)
        logger.info(f"Pipeline '{lang_code}' ready.")
    return pipelines[lang_code]


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all backends on startup."""
    # Load Kokoro (always PyTorch)
    logger.info("Loading Kokoro model (PyTorch)...")
    get_pipeline("a")  # Pre-load American English (most common)
    logger.info("Kokoro TTS ready.")
    
    # Load STT backend (auto-detected)
    global whisper_backend, whisper_model, ml_stt
    whisper_backend, loaded = detect_stt_backend()
    if whisper_backend == "mlx":
        ml_stt = loaded
    else:
        whisper_model = loaded
    logger.info(f"STT backend: {whisper_backend}")
    logger.info("Server ready.")
    
    yield
    
    logger.info("Shutting down.")


app = FastAPI(title="Kokoro TTS + Whisper STT", lifespan=lifespan)


# ---------------------------------------------------------------------------
# TTS endpoint  (always uses PyTorch Kokoro)
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str = Field(..., max_length=4096)
    voice: str = "alloy"
    response_format: Literal["mp3", "wav", "flac", "opus", "pcm", "aac"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)


def encode_audio(samples, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "pcm":
        pcm = (samples * 32767).astype(np.int16)
        buf.write(pcm.tobytes())
    elif fmt == "opus":
        sf.write(buf, samples, SAMPLE_RATE, format="OGG", subtype="VORBIS")
    elif fmt == "mp3":
        sf.write(buf, samples, SAMPLE_RATE, format="MP3")
    elif fmt == "aac":
        sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    else:
        sf.write(buf, samples, SAMPLE_RATE, format=fmt.upper())
    return buf.getvalue()


def resolve_voice(name: str) -> str:
    """Map an OpenAI voice name or pass through a Kokoro voice name."""
    if name in VOICE_MAP:
        return VOICE_MAP[name]
    if len(name) >= 3 and name[0] in LANG_PREFIXES:
        return name
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice '{name}'. OpenAI voices: {list(VOICE_MAP.keys())}. "
               f"Or use a Kokoro voice name directly (e.g. af_heart, am_adam).",
    )


def lang_code_for_voice(voice: str) -> str:
    """Derive the language code from the voice name prefix."""
    prefix = voice[0]
    return LANG_PREFIXES.get(prefix, "a")


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text must not be empty.")
    
    voice = resolve_voice(req.voice)
    lang_code = lang_code_for_voice(voice)
    pipeline = get_pipeline(lang_code)
    
    logger.info(
        f"TTS: voice={voice} lang={lang_code} speed={req.speed} "
        f"fmt={req.response_format} chars={len(req.input)}"
    )
    
    t0 = time.perf_counter()
    chunks = []
    for _gs, _ps, audio in pipeline(req.input, voice=voice, speed=req.speed):
        chunks.append(audio)
    
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated.")
    
    samples = np.concatenate(chunks)
    elapsed = time.perf_counter() - t0
    duration = len(samples) / SAMPLE_RATE
    logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
    
    audio_bytes = encode_audio(samples, req.response_format)
    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[req.response_format],
        headers={"Content-Disposition": f'inline; filename="speech.{req.response_format}"'},
    )


# ---------------------------------------------------------------------------
# STT endpoint  (dispatches to CUDA or MLX backend)
# ---------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    if whisper_backend == "none" or whisper_model is None:
        raise HTTPException(status_code=503, detail=f"STT backend '{whisper_backend}' not loaded.")
    
    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        
        t0 = time.perf_counter()
        text, info = transcribe_audio(tmp.name, language=language)
        elapsed = time.perf_counter() - t0
        
        logger.info(
            f"STT: backend={whisper_backend} lang={info['language']} "
            f"prob={info['language_probability']:.2f} "
            f"duration={info['duration']:.1f}s elapsed={elapsed:.2f}s"
        )
        
        if response_format == "verbose_json":
            return {"text": text, **info}
        return {"text": text}
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/audio/voices")
async def list_voices():
    """List all available voices (not part of OpenAI API, but useful)."""
    return {
        "openai_mapped": VOICE_MAP,
        "kokoro_voices": [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "ef_dora", "em_alex", "em_santa",
            "ff_siwis",
            "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            "if_sara", "im_nicola",
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
            "pf_dora", "pm_alex", "pm_santa",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
        ],
    }


@app.get("/v1/models")
async def list_models():
    """Minimal /v1/models so clients can discover this server."""
    return {
        "object": "list",
        "data": [
            {"id": "kokoro", "object": "model", "created": 0, "owned_by": "local"},
            {"id": "tts-1", "object": "model", "created": 0, "owned_by": "local"},
            {"id": "tts-1-hd", "object": "model", "created": 0, "owned_by": "local"},
            {"id": "whisper-1", "object": "model", "created": 0, "owned_by": "local"},
        ],
    }


@app.get("/health")
async def health():
    """Healthcheck — reports which backends are active."""
    return {
        "status": "ok",
        "tts_backend": "kokoro-pytorch",
        "tts_device": "cuda" if _torch_has_cuda() else "mlx" if _mlx_is_available() else "cpu",
        "stt_backend": whisper_backend,
        "stt_model": "large-v3",
        "stt_device": {"cuda": "nvidia-gpu", "mlx": "apple-silicon", "cpu": "cpu", "none": "none"}.get(
            whisper_backend, "unknown"
        ),
    }


def _torch_has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _mlx_is_available():
    """Check if MLX is available (Apple Silicon)."""
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8880)
