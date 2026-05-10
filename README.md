# Cacique TTS/STT Server

> **Alpha Release** — This project is under active development. APIs and configuration may change between versions.

An OpenAI-compatible Text-to-Speech and Speech-to-Text server for local AI inference. Drop-in replacement for OpenAI's `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints, powered by [Kokoro](https://github.com/hexgrad/kokoro) (TTS) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (STT) running on GPU.

Built for use with [OpenClaw](https://github.com/OpenClaw).

## Features

- **OpenAI API compatible** — works with any client that targets the OpenAI audio endpoints
- **Text-to-Speech** via Kokoro with 50+ voices across 9 languages
- **Speech-to-Text** via faster-whisper (Whisper large-v3) with automatic language detection
- **GPU accelerated** — both TTS and STT run on CUDA
- **Multiple output formats** — MP3, WAV, FLAC, Opus, AAC, PCM

## Supported Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page with status, logs, and GPU info |
| `/v1/audio/speech` | POST | Generate speech from text |
| `/v1/audio/transcriptions` | POST | Transcribe audio to text |
| `/v1/audio/voices` | GET | List available voices |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/api/logs` | GET | Last 100 log lines (JSON) |
| `/api/nvidia-smi` | GET | GPU status (JSON) |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~4 GB VRAM (Kokoro + Whisper large-v3 in float16)

## Installation

```bash
git clone https://github.com/michel-dance/cacique_tts_sst
cd tts-server
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

The server starts on `http://0.0.0.0:8880`.

### TTS Example

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from OpenClaw!", "voice": "alloy"}' \
  -o speech.mp3
```

### STT Example

```bash
curl -X POST http://localhost:8880/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1
```

## Systemd Service (Ubuntu)

Use `install.sh` to install the server as a system daemon. Run from the repository root:

```bash
sudo bash install.sh
```

This installs the server to `/opt/tts_sst`, creates a dedicated service user, sets up a Python virtual environment, and registers a `tts_sst` systemd service.

### Managing the Service

```bash
# Start the service
sudo systemctl start tts_sst

# Stop the service
sudo systemctl stop tts_sst

# Restart the service (after config/code changes)
sudo systemctl restart tts_sst

# Enable auto-start on boot
sudo systemctl enable tts_sst

# Disable auto-start on boot
sudo systemctl disable tts_sst

# Check service status
sudo systemctl status tts_sst

# View live logs
sudo journalctl -u tts_sst -f

# View recent logs
sudo journalctl -u tts_sst -e --no-pager
```

### Reinstalling After Code Changes

If you modify `server.py` or `requirements.txt`, re-run the installer to update the deployed files:

```bash
sudo bash install.sh
```

Then restart the service to pick up the changes:

```bash
sudo systemctl restart tts_sst
```

## Voice Mapping

Standard OpenAI voice names are mapped to Kokoro voices:

| OpenAI Voice | Kokoro Voice |
|---|---|
| alloy | af_alloy |
| echo | am_echo |
| fable | bm_fable |
| nova | af_nova |
| onyx | am_onyx |
| shimmer | af_sky |

You can also use any Kokoro voice name directly (e.g. `af_heart`, `am_adam`). See `/v1/audio/voices` for the full list.

## Supported Languages

American English, British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin Chinese.

## License

Alpha software — released by OpenClaw for local AI inference use.
