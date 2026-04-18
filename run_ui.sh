#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

STREAMLIT_BIN="$ROOT_DIR/.venv/bin/streamlit"

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  echo "Error: repo virtualenv is missing or incomplete. Run: ./scripts/bootstrap_env.sh" >&2
  exit 1
fi

PORT="${1:-8501}"
PID_FILE=".streamlit_ui.pid"
LOG_FILE="streamlit.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Streamlit is already running (pid=$OLD_PID)."
    echo "URL: http://localhost:${PORT}"
    exit 0
  else
    rm -f "$PID_FILE"
  fi
fi

nohup "$STREAMLIT_BIN" run chat_ui.py --server.port "$PORT" --server.headless true > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

sleep 1
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "Started Streamlit (pid=$NEW_PID)"
  echo "URL: http://localhost:${PORT}"
  echo "Log: $ROOT_DIR/$LOG_FILE"
else
  echo "Failed to start Streamlit. Check log: $ROOT_DIR/$LOG_FILE" >&2
  exit 1
fi
