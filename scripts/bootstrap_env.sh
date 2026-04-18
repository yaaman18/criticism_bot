#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating repo virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

export PATH="$VENV_DIR/bin:$PATH"

echo "Upgrading pip inside $VENV_DIR"
"$VENV_PY" -m pip install --upgrade pip

echo "Installing repo requirements into $VENV_DIR"
"$VENV_PY" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Running harness doctor with repo Python"
"$VENV_PY" -m trm_pipeline.experiment_harness doctor

echo
echo "Repo runtime is ready."
echo "Run tests with: $VENV_PY -m pytest -q"
echo "Run the app with: $VENV_PY $ROOT_DIR/anthropic_art_critic_chat.py"
