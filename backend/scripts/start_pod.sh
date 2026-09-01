#!/usr/bin/env bash
# One-shot, idempotent bootstrap + start script for a RunPod (or similar) GPU
# pod. Safe to re-run any time -- after a fresh clone, or after a pod
# stop/restart wipes everything outside /workspace (Node.js, apt packages,
# any pip install not placed under /workspace).
#
# What it does:
#   - Installs Node.js if missing (fast, ~10s; wiped by pod restarts)
#   - Creates a Python venv at /workspace/venv and installs backend deps into
#     it ONLY if that venv doesn't already exist -- since /workspace persists
#     across pod restarts, this multi-minute step normally runs once, ever.
#   - Starts (or restarts) the backend on :8000 and the frontend on :4173,
#     backgrounded with nohup, logging to /workspace/{backend,frontend}.log.
#
# Usage:
#   1. Create /workspace/.imuse_env once (gitignored, persists under
#      /workspace) with:
#        export IMUSE_API_KEY=<your key>
#        export VITE_API_URL=<your backend's public proxy URL>
#   2. bash backend/scripts/start_pod.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"
FRONTEND_DIR="$REPO_DIR/frontend"
VENV_DIR="/workspace/venv"
LOG_DIR="/workspace"
ENV_FILE="/workspace/.imuse_env"

echo "== imuse-studio pod bootstrap =="

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "No $ENV_FILE found. Create it with:" >&2
  echo "  export IMUSE_API_KEY=<your key>" >&2
  echo "  export VITE_API_URL=<your backend's public proxy URL>" >&2
  exit 1
fi

: "${IMUSE_API_KEY:?IMUSE_API_KEY not set in $ENV_FILE}"
: "${VITE_API_URL:?VITE_API_URL not set in $ENV_FILE}"

if ! command -v node >/dev/null 2>&1; then
  echo "Installing Node.js..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
  apt-get install -y nodejs >/dev/null 2>&1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating venv at $VENV_DIR and installing deps (only happens once)..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip --quiet
  "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt" -r "$BACKEND_DIR/requirements-ml.txt"
else
  echo "Reusing existing venv at $VENV_DIR"
fi

echo "Starting backend..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
cd "$BACKEND_DIR"
IMUSE_API_KEY="$IMUSE_API_KEY" IMUSE_MOCK_ML=0 \
  nohup "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 > "$LOG_DIR/backend.log" 2>&1 &
echo "  backend PID $!, log: $LOG_DIR/backend.log"

echo "Building and starting frontend..."
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  npm install
fi
VITE_API_URL="$VITE_API_URL" VITE_API_KEY="$IMUSE_API_KEY" npm run build

pkill -f "serve -s dist" 2>/dev/null || true
nohup npx --yes serve -s dist -l 4173 > "$LOG_DIR/frontend.log" 2>&1 &
echo "  frontend PID $!, log: $LOG_DIR/frontend.log"

echo "Waiting for services to come up..."
sleep 5

echo "== Health check =="
curl -s http://localhost:8000/api/health || echo "backend not responding yet -- check $LOG_DIR/backend.log"
echo
curl -s -o /dev/null -w "frontend http_code=%{http_code}\n" http://localhost:4173 || true
