"""OpenAI-compatible TTS/STT server with cross-platform backend support.

Supports:
- Kokoro TTS (PyTorch — works on CPU, MPS/Apple Silicon, CUDA)
- faster-whisper STT (CUDA on NVIDIA GPUs)
- mlx-whisper STT (Apple Silicon / MLX)

Auto-detects hardware at startup and picks the best backend.
"""

import asyncio
import io
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
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

        <div class="card">
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
            const statusClass = (s) => s === 'ok' ? 'badge-green' : s === 'none' ? 'badge-red' : 'badge-yellow';
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
