#!/usr/bin/env bash
set -euo pipefail

LENIA_DIR="${LENIA_DIR:-$HOME/Lenia_official/Python}"
LENIA_PY="$LENIA_DIR/LeniaND.py"
LENIA_VENV_PY="$LENIA_DIR/.venv/bin/python"

if [[ ! -f "$LENIA_PY" ]]; then
  echo "LeniaND.py not found: $LENIA_PY" >&2
  exit 1
fi

if [[ ! -x "$LENIA_VENV_PY" ]]; then
  echo "Lenia venv python not found: $LENIA_VENV_PY" >&2
  echo "Create it first: cd \"$LENIA_DIR\" && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

cd "$LENIA_DIR"
exec "$LENIA_VENV_PY" "$LENIA_PY" --erie-env --erie-seed 20260228 "$@"
