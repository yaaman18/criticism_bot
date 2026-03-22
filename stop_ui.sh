#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PID_FILE=".streamlit_ui.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file. Streamlit may already be stopped."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped Streamlit (pid=$PID)"
else
  echo "Process not running (stale pid=$PID)"
fi
rm -f "$PID_FILE"
