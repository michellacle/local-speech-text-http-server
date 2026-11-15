"""OpenAI-compatible TTS/STT server (GPU-only).

Supports:
- Kokoro TTS (PyTorch — CUDA on NVIDIA, MPS on Apple Silicon)
- faster-whisper STT (CUDA on NVIDIA GPUs)
- mlx-whisper STT (Apple Silicon / Metal)

GPU is mandatory: the server detects CUDA (NVIDIA) or Metal (Apple
Silicon) at startup and refuses to start on CPU. There is no CPU
fallback — a missing/broken GPU is a fatal startup error.

At request time the GPU is re-probed (a GPU can crash or lock up while
the server is running). If the GPU is not usable at that moment, API
requests fail fast with HTTP 500 — they never silently degrade to CPU.
"""

import asyncio
import io
import json
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# In-memory log buffer (last 1000 lines)
LOG_BUFFER: list[str] = []
MAX_LOG_LINES = 1000


class LogBufferHandler(logging.Handler):
    """Custom logging handler that buffers log lines in memory."""

    def emit(self, record):
        try:
            log_line = self.format(record)
            LOG_BUFFER.append(log_line)
            if len(LOG_BUFFER) > MAX_LOG_LINES:
                LOG_BUFFER.pop(0)
        except Exception:
            pass


# Set up logging with both console and buffer handlers
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

buffer_handler = LogBufferHandler()
buffer_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("tts-server")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(buffer_handler)

# Also capture uvicorn access logs
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addHandler(buffer_handler)

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

# GPU device for Kokoro TTS ('cuda' or 'mps'), set at startup.
# This server is GPU-only: startup fails hard if no GPU is found.
TTS_DEVICE: str = ""
GPU_DEVICE_INDEX = int(os.environ.get("GPU_DEVICE_INDEX", "0"))

# Keep interactive transcription responsive. Faster-Whisper defaults to a
# larger beam than the OpenAI Whisper API, which adds latency for voice input.
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", "1"))
STT_VAD_FILTER = os.environ.get("STT_VAD_FILTER", "true").lower() in {
    "1", "true", "yes", "on"
}
STT_CONDITION_ON_PREVIOUS_TEXT = os.environ.get(
    "STT_CONDITION_ON_PREVIOUS_TEXT", "false"
).lower() in {"1", "true", "yes", "on"}

# STT backends (populated at startup)
# On NVIDIA: whisper_model = faster_whisper.WhisperModel (CUDA)
# On Apple Silicon (MLX): ml_stt = mlx_whisper module
whisper_backend = "none"  # 'cuda', 'mlx', or 'none'
whisper_model = None  # faster_whisper.WhisperModel (CUDA)
ml_stt = None  # mlx_whisper module (Apple Silicon)


# ---------------------------------------------------------------------------
# Hardware detection (GPU-only — no CPU fallback)
# ---------------------------------------------------------------------------

def detect_gpu_device() -> str:
    """Detect the GPU device for PyTorch (Kokoro TTS).

    Returns 'cuda' (NVIDIA) or 'mps' (Apple Silicon).
    Raises RuntimeError if no GPU is available — this server
    deliberately refuses to run on CPU.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(f"PyTorch is not installed: {e}") from e
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "No GPU detected (CUDA or Metal). This server is GPU-only and "
        "refuses to run on CPU. Install NVIDIA drivers + a CUDA-enabled "
        "PyTorch build (or run on Apple Silicon) and restart."
    )


def detect_stt_backend() -> tuple:
    """Detect the GPU STT backend and return (name, model_or_module).

    Priority: CUDA (faster-whisper) > MLX (mlx-whisper, Apple Silicon).
    Raises RuntimeError if no GPU backend is available — no CPU fallback.
    Returns (backend_name, model_or_module).
    """
    # NVIDIA GPU → faster-whisper (CTranslate2 CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            from faster_whisper import WhisperModel
            try:
                logger.info("CUDA detected — using faster-whisper (CUDA)")
                return "cuda", WhisperModel(
                    "large-v3",
                    device="cuda",
                    device_index=GPU_DEVICE_INDEX,
                    compute_type="float16",
                )
            except Exception as e:
                err = str(e)
                hint = (
                    "The GPU is out of memory — close other GPU workloads "
                    "and retry."
                    if "out of memory" in err.lower()
                    else "The installed ctranslate2 build likely lacks CUDA "
                    "support (install a CUDA-enabled wheel)."
                )
                raise RuntimeError(
                    f"CUDA is available but faster-whisper failed to load on "
                    f"GPU: {err} {hint} This server is GPU-only — no CPU "
                    f"fallback."
                ) from e
    except ImportError:
        pass

    # Apple Silicon → mlx-whisper (Metal GPU)
    if platform.system() == "Darwin":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                import mlx_whisper
                logger.info("Apple Silicon (Metal/MLX) detected — using mlx-whisper")
                return "mlx", mlx_whisper
        except ImportError:
            pass

    raise RuntimeError(
        "No GPU available for STT (need NVIDIA CUDA or Apple MLX). "
        "This server is GPU-only and refuses to run on CPU."
    )


# ---------------------------------------------------------------------------
# Runtime GPU health (fail fast at request time — no CPU fallback)
# ---------------------------------------------------------------------------

# A GPU can crash or lock up while the server is running (driver errors,
# Xid faults, thermal resets). CUDA calls against a wedged GPU block
# indefinitely, so the probe below runs in a daemon thread with a hard
# timeout: a stuck GPU fails fast (within GPU_PROBE_TIMEOUT seconds)
# instead of hanging the request, and we never fall back to CPU.
GPU_PROBE_TIMEOUT = float(os.environ.get("GPU_PROBE_TIMEOUT", "5"))


def gpu_healthy(timeout: float = GPU_PROBE_TIMEOUT) -> tuple:
    """Check *right now* whether the GPU is usable.

    Returns (ok, reason). The CUDA probe runs a tiny kernel + full
    device synchronize in a daemon thread and joins it for at most
    `timeout` seconds. If the thread is still alive after the timeout,
    the GPU is wedged. (The stuck probe thread is left behind — one per
    failed probe; the service is degraded anyway and needs a restart.)
    """
    try:
        import torch
    except ImportError as e:
        return False, f"PyTorch not installed: {e}"

    if TTS_DEVICE == "cuda":
        if not torch.cuda.is_available():
            return False, (
                "torch.cuda.is_available() is False — GPU unavailable "
                "(driver crash or reset?). Restart the service."
            )
        res: dict = {}

        def probe():
            try:
                dev = f"cuda:{GPU_DEVICE_INDEX}"
                x = torch.zeros(8, device=dev)
                x.sum().item()  # launch a kernel + sync
                torch.cuda.synchronize()
                res["ok"] = True
            except Exception as e:  # surface the actual CUDA error
                res["err"] = str(e)

        t = threading.Thread(target=probe, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return False, (
                f"CUDA probe timed out after {timeout:.0f}s — GPU appears "
                "wedged/locked (Xid error or driver hang). Restart the service."
            )
        if "err" in res:
            return False, f"CUDA probe failed: {res['err']}"
        return True, ""

    if TTS_DEVICE == "mps":
        ok = torch.backends.mps.is_available()
        return (ok, "") if ok else (False, "MPS backend unavailable")

    return False, "No GPU device was detected at startup"


def require_gpu() -> None:
    """Raise HTTP 500 if the GPU is not usable at this moment (fail fast)."""
    ok, reason = gpu_healthy()
    if not ok:
        logger.error(f"GPU unavailable — failing request: {reason}")
        raise HTTPException(status_code=500, detail=f"GPU unavailable: {reason}")


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
    
    elif whisper_backend == "cuda":
        segments, info = whisper_model.transcribe(
            file_path,
            language=language,
            beam_size=STT_BEAM_SIZE,
            vad_filter=STT_VAD_FILTER,
            condition_on_previous_text=STT_CONDITION_ON_PREVIOUS_TEXT,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
        return text, info_dict
    
    else:
        raise RuntimeError("No STT backend loaded. Check startup logs.")


def warm_up_stt_model() -> None:
    """Initialize CUDA kernels before the first user transcription."""
    if whisper_backend != "cuda" or whisper_model is None:
        return

    warmup_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            warmup_path = tmp.name
        # A short tone forces the encoder and decoder to execute without
        # requiring a bundled speech sample or making a network request.
        sample_rate = 16000
        samples = 0.01 * np.sin(
            2 * np.pi * 440 * np.arange(sample_rate, dtype=np.float32) / sample_rate
        )
        sf.write(warmup_path, samples, sample_rate)
        segments, _ = whisper_model.transcribe(
            warmup_path,
            language="en",
            beam_size=STT_BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=STT_CONDITION_ON_PREVIOUS_TEXT,
        )
        for _ in segments:
            pass
        logger.info("STT CUDA warm-up complete.")
    finally:
        if warmup_path:
            try:
                os.unlink(warmup_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Kokoro pipeline (always PyTorch on the GPU — CUDA or MPS)
# ---------------------------------------------------------------------------

def get_pipeline(lang_code: str):
    """Get or create a Kokoro KPipeline for the given language code."""
    from kokoro import KPipeline
    if lang_code not in pipelines:
        logger.info(f"Loading KPipeline lang_code='{lang_code}' device={TTS_DEVICE}...")
        pipelines[lang_code] = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=TTS_DEVICE,  # 'cuda' or 'mps' — never 'cpu'
        )
        logger.info(f"Pipeline '{lang_code}' ready on {TTS_DEVICE}.")
    return pipelines[lang_code]


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all backends on startup.

    GPU-only: fails hard (and systemd restarts the unit) if no GPU is
    available — this server never runs on CPU.
    """
    global TTS_DEVICE, whisper_backend, whisper_model, ml_stt

    # Detect GPU up front — fatal if missing (no CPU fallback)
    TTS_DEVICE = detect_gpu_device()
    logger.info(f"GPU detected — TTS device: {TTS_DEVICE}")

    # Load Kokoro (PyTorch, pinned to the GPU)
    logger.info("Loading Kokoro model (PyTorch, %s)...", TTS_DEVICE)
    get_pipeline("a")  # Pre-load American English (most common)
    logger.info("Kokoro TTS ready on %s.", TTS_DEVICE)

    # Load STT backend (GPU — CUDA or MLX, never CPU)
    whisper_backend, loaded = detect_stt_backend()
    if whisper_backend == "mlx":
        ml_stt = loaded
    else:
        whisper_model = loaded
    logger.info(f"STT backend: {whisper_backend}")
    warm_up_stt_model()
    logger.info("Server ready.")

    yield

    logger.info("Shutting down.")


app = FastAPI(title="Kokoro TTS + Whisper STT", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Home page & observability endpoints
# ---------------------------------------------------------------------------

HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTS/STT Server</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { font-size: 1.8rem; margin-bottom: 8px; color: #58a6ff; }
        h2 { font-size: 1.2rem; margin-bottom: 12px; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
        .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card.full { grid-column: 1 / -1; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-weight: 600; font-size: 0.85rem; }
        td { font-size: 0.9rem; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        .badge-green { background: #123325; color: #3fb950; }
        .badge-yellow { background: #3d2e00; color: #d29922; }
        .badge-red { background: #3d1214; color: #f85149; }
        .badge-blue { background: #0c2d6b; color: #58a6ff; }
        #log-container { max-height: 400px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.6; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
        #log-container div { white-space: pre-wrap; word-break: break-all; }
        #nvidia-smi { max-height: 400px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.4; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; white-space: pre-wrap; word-break: break-all; }
        .refresh-btn { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; margin-bottom: 12px; }
        .refresh-btn:hover { background: #30363d; }
        a { color: #58a6ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        code { background: #1c2128; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }
    </style>
</head>
<body>
    <h1>Kokoro TTS + Whisper STT Server</h1>
    <p class="subtitle">OpenAI-compatible API endpoints</p>

    <div class="grid">
        <div class="card">
            <h2>API Endpoints</h2>
            <table>
                <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
                <tr><td><span class="badge badge-green">POST</span></td><td><code>/v1/audio/speech</code></td><td>Text-to-speech (Kokoro)</td></tr>
                <tr><td><span class="badge badge-green">POST</span></td><td><code>/v1/audio/transcriptions</code></td><td>Speech-to-text (Whisper)</td></tr>
                <tr><td><span class="badge badge-blue">GET</span></td><td><code>/v1/models</code></td><td>List available models</td></tr>
                <tr><td><span class="badge badge-blue">GET</span></td><td><code>/v1/audio/voices</code></td><td>List available voices</td></tr>
                <tr><td><span class="badge badge-blue">GET</span></td><td><code>/health</code></td><td>Health check</td></tr>
                <tr><td><span class="badge badge-blue">GET</span></td><td><code>/api/logs</code></td><td>Last 100 log lines (JSON)</td></tr>
                <tr><td><span class="badge badge-blue">GET</span></td><td><code>/api/nvidia-smi</code></td><td>GPU status (JSON)</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Loaded Models</h2>
            <table id="models-table">
                <tr><th>Model</th><th>Backend</th><th>Device</th><th>Status</th></tr>
            </table>
        </div>

        <div class="card">
            <h2>Available Voices (OpenAI Mapped)</h2>
            <table id="voices-table">
                <tr><th>OpenAI Name</th><th>Kokoro Voice</th></tr>
            </table>
        </div>

        <div class="card full">
            <h2>GPU Status (nvidia-smi)</h2>
            <button class="refresh-btn" onclick="fetchNvidiaSmi()">Refresh</button>
            <div id="nvidia-smi">Loading...</div>
        </div>

        <div class="card full">
            <h2>Service Logs (Last 100 Lines)</h2>
            <button class="refresh-btn" onclick="fetchLogs()">Refresh</button>
            <div id="log-container">Loading...</div>
        </div>
    </div>

    <script>
        async function fetchHealth() {
            const r = await fetch('/health');
            const d = await r.json();
            const modelsTable = document.getElementById('models-table');
            const statusClass = (s) => ['cuda', 'mlx', 'ok'].includes(s) ? 'badge-green' : s === 'none' ? 'badge-red' : 'badge-yellow';
            modelsTable.innerHTML = `
                <tr><th>Model</th><th>Backend</th><th>Device</th><th>Status</th></tr>
                <tr><td>Kokoro TTS</td><td>${d.tts_backend}</td><td>${d.tts_device}</td><td><span class="badge badge-green">active</span></td></tr>
                <tr><td>Whisper STT</td><td>${d.stt_backend}</td><td>${d.stt_device || 'N/A'}</td><td><span class="badge ${statusClass(d.stt_backend)}">${d.stt_backend}</span></td></tr>
            `;
        }

        async function fetchVoices() {
            const r = await fetch('/v1/audio/voices');
            const d = await r.json();
            const table = document.getElementById('voices-table');
            let html = '<tr><th>OpenAI Name</th><th>Kokoro Voice</th></tr>';
            for (const [k, v] of Object.entries(d.openai_mapped)) {
                html += `<tr><td>${k}</td><td>${v}</td></tr>`;
            }
            table.innerHTML = html;
        }

        async function fetchLogs() {
            const r = await fetch('/api/logs');
            const d = await r.json();
            const container = document.getElementById('log-container');
            container.innerHTML = d.logs.map(l => '<div>' + l.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>').join('');
            container.scrollTop = container.scrollHeight;
        }

        async function fetchNvidiaSmi() {
            const r = await fetch('/api/nvidia-smi');
            const d = await r.json();
            const el = document.getElementById('nvidia-smi');
            el.textContent = d.output || d.error || 'nvidia-smi not available';
        }

        // Initial load
        Promise.all([fetchHealth(), fetchVoices(), fetchLogs(), fetchNvidiaSmi()]);

        // Auto-refresh logs and nvidia-smi every 5s
        setInterval(fetchLogs, 5000);
        setInterval(fetchNvidiaSmi, 5000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Home page with endpoints, models, logs, and GPU status."""
    return HOME_PAGE_HTML


@app.get("/api/logs")
async def get_logs():
    """Return the last 100 log lines."""
    return {"logs": LOG_BUFFER[-100:]}


@app.get("/api/nvidia-smi")
async def get_nvidia_smi():
    """Run nvidia-smi and return the output."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(subprocess.run, ["nvidia-smi"], capture_output=True, text=True, timeout=10),
            timeout=12
        )
        return {"output": result.stdout}
    except FileNotFoundError:
        return {"error": "nvidia-smi not found (no NVIDIA driver installed)"}
    except Exception as e:
        return {"error": str(e)}


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

    require_gpu()  # fail fast with 500 if the GPU crashed/locked up since startup

    voice = resolve_voice(req.voice)
    lang_code = lang_code_for_voice(voice)
    pipeline = get_pipeline(lang_code)

    logger.info(
        f"TTS: voice={voice} lang={lang_code} speed={req.speed} "
        f"fmt={req.response_format} chars={len(req.input)}"
    )

    t0 = time.perf_counter()
    chunks = []
    try:
        for _gs, _ps, audio in pipeline(req.input, voice=voice, speed=req.speed):
            chunks.append(audio)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS synthesis failed (GPU error?)")
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed (GPU unavailable or crashed): {e}",
        ) from e
    
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
    if whisper_backend == "none" or (whisper_backend == "cuda" and whisper_model is None):
        raise HTTPException(status_code=503, detail=f"STT backend '{whisper_backend}' not loaded.")

    require_gpu()  # fail fast with 500 if the GPU crashed/locked up since startup

    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()

        t0 = time.perf_counter()
        try:
            text, info = transcribe_audio(tmp.name, language=language)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("STT transcription failed (GPU error?)")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed (GPU unavailable or crashed): {e}",
            ) from e
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
    """Healthcheck — reports which backends are active (real devices).

    Returns 500 when the GPU is not usable at this moment, so monitors
    (systemd, load balancers, cron) can trigger a service restart.
    """
    gpu_ok, gpu_detail = gpu_healthy()
    payload = {
        "status": "ok" if gpu_ok else "gpu-unavailable",
        "tts_backend": "kokoro-pytorch",
        "tts_device": TTS_DEVICE or "uninitialized",
        "stt_backend": whisper_backend,
        "stt_model": "large-v3",
        "stt_device": {"cuda": "nvidia-gpu", "mlx": "apple-silicon"}.get(
            whisper_backend, "unknown"
        ),
        "gpu": {"available": gpu_ok, "detail": gpu_detail or "healthy"},
    }
    if not gpu_ok:
        return JSONResponse(status_code=500, content=payload)
    return payload


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8880)
