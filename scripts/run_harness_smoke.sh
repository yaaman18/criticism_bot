#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="${1:-artifacts/harness_smoke}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: repo virtualenv is missing or incomplete. Run: ./scripts/bootstrap_env.sh" >&2
  exit 1
fi

rm -rf "$OUTPUT_ROOT"

"$VENV_PY" -m trm_pipeline.experiment_harness plan \
  --output-root "$OUTPUT_ROOT" \
  --experiment-name harness_smoke \
  --families toxic_band fragile_boundary \
  --num-seeds 1 \
  --steps 8 \
  --warmup-steps 2

"$VENV_PY" -m trm_pipeline.experiment_harness run \
  --contract "$OUTPUT_ROOT/contract.json"

echo
echo "Smoke harness artifacts:"
echo "  $OUTPUT_ROOT/doctor_report.json"
echo "  $OUTPUT_ROOT/eval_report.json"
echo "  $OUTPUT_ROOT/promotion_decision.json"
echo "  $OUTPUT_ROOT/next_steps.json"
