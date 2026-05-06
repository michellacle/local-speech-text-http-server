#!/usr/bin/env bash
#
# tts_sst installer — Debian/Ubuntu
# Installs the TTS/STT server as a systemd service under /opt/tts_sst.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVICE_NAME="tts_sst"
INSTALL_DIR="/opt/tts_sst"
VENV_DIR="${INSTALL_DIR}/venv"
SERVER_PORT=8880
HEALTH_URL="http://127.0.0.1:${SERVER_PORT}/health"
SERVICE_USER="tts_sst"
HF_CACHE_DIR="${INSTALL_DIR}/hf_cache"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_HEALTH_RETRIES=200
HEALTH_RETRY_INTERVAL=3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }

require_root() {
    if [[ "${EUID:-}" -ne 0 ]]; then
        error "This script must be run as root (use sudo)."
        exit 1
    fi
}

require_ubuntu() {
    if ! command -v apt-get &>/dev/null; then
        error "Only Ubuntu/Debian systems are currently supported."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
install_system_deps() {
    info "Installing system dependencies…"
    apt-get update -qq
    apt-get install -y -qq \
        python3 python3-venv python3-pip \
        git curl gcc g++ \
        libsndfile1 ffmpeg \
        >/dev/null 2>&1
    ok "System dependencies installed."
}

# ---------------------------------------------------------------------------
# 2. Create service user
# ---------------------------------------------------------------------------
create_service_user() {
    if id "${SERVICE_USER}" &>/dev/null; then
        info "Service user '${SERVICE_USER}' already exists, skipping."
    else
        info "Creating service user '${SERVICE_USER}'…"
        useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
        ok "Service user created."
    fi
}

# ---------------------------------------------------------------------------
# 3. Install application
# ---------------------------------------------------------------------------
install_app() {
    info "Setting up application in ${INSTALL_DIR}…"

    # Create directory
    mkdir -p "${INSTALL_DIR}"

    # Copy source files
    cp "${REPO_DIR}/server.py" "${INSTALL_DIR}/"
    cp "${REPO_DIR}/requirements.txt" "${INSTALL_DIR}/"

    # Create virtual environment
    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi

    # Upgrade pip and install dependencies
    info "Installing Python dependencies (this may take a while)…"
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/pip" install --quiet -r "${INSTALL_DIR}/requirements.txt"

    # Create HF model cache directory
    mkdir -p "${HF_CACHE_DIR}"

    # Set ownership
    chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

    ok "Application installed."
}

# ---------------------------------------------------------------------------
# 4. Create systemd unit
# ---------------------------------------------------------------------------
create_systemd_unit() {
    info "Creating systemd service unit…"

    cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=TTS/STT Server
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/server.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=${HF_CACHE_DIR}
Environment=HUGGINGFACE_HUB_CACHE=${HF_CACHE_DIR}
Environment=TORCH_HOME=${HF_CACHE_DIR}/torch

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    ok "Systemd unit created."
}

# ---------------------------------------------------------------------------
# 5. Start and enable service
# ---------------------------------------------------------------------------
start_service() {
    info "Enabling and starting ${SERVICE_NAME} service…"
    systemctl enable "${SERVICE_NAME}"
    systemctl start "${SERVICE_NAME}"
    ok "Service started."
}

# ---------------------------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------------------------
health_check() {
    info "Waiting for server to become healthy…"
    local attempt=0
    while (( attempt < MAX_HEALTH_RETRIES )); do
        attempt=$((attempt + 1))
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
        if [[ "${http_code}" == "200" ]]; then
            ok "Server is healthy!"
            info "Health endpoint response:"
            curl -s "${HEALTH_URL}" | python3 -m json.tool 2>/dev/null || true
            return 0
        fi
        if (( attempt % 5 == 0 )); then
            info "Still waiting… (attempt ${attempt}/${MAX_HEALTH_RETRIES})"
        fi
        sleep "${HEALTH_RETRY_INTERVAL}"
    done

    error "Server did not become healthy in time."
    info "Check logs with: journalctl -u ${SERVICE_NAME} -e"
    return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "============================================"
    echo "  TTS/STT Server Installer"
    echo "============================================"
    echo ""

    require_root
    require_ubuntu

    install_system_deps
    create_service_user
    install_app
    create_systemd_unit
    start_service
    health_check

    echo ""
    echo "============================================"
    ok "Installation complete!"
    echo "============================================"
    echo ""
    info "Service:       ${SERVICE_NAME}"
    info "Install dir:   ${INSTALL_DIR}"
    info "Server URL:    http://0.0.0.0:${SERVER_PORT}"
    info "Health check:  curl ${HEALTH_URL}"
    info "View logs:     journalctl -u ${SERVICE_NAME} -f"
    info "Restart:       systemctl restart ${SERVICE_NAME}"
    echo ""
}

main "$@"
