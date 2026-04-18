#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/.venv/bin/python" -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --execution-target gpu-handoff \
  --provider vastai \
  "$@"
