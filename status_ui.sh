#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PID_FILE=".streamlit_ui.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "stopped (no pid file)"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  echo "running pid=$PID"
else
  echo "stopped (stale pid=$PID)"
fi
