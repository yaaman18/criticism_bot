#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
PASSIVE_ROOT="${1:-artifacts/dataset_smoke_passive}"
AGENTIC_ROOT="${2:-artifacts/dataset_smoke_agentic}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: repo virtualenv is missing or incomplete. Run: ./scripts/bootstrap_env.sh" >&2
  exit 1
fi

rm -rf "$PASSIVE_ROOT" "$AGENTIC_ROOT"

"$VENV_PY" -m trm_pipeline.dataset_harness plan \
  --output-root "$PASSIVE_ROOT" \
  --dataset-name passive_smoke \
  --dataset-kind passive_lenia_pretrain \
  --num-seeds 4 \
  --image-size 32 \
  --warmup-steps 16 \
  --record-steps 32 \
  --target-radius 8

"$VENV_PY" -m trm_pipeline.dataset_harness run \
  --contract "$PASSIVE_ROOT/contract.json"

"$VENV_PY" -m trm_pipeline.dataset_harness plan \
  --output-root "$AGENTIC_ROOT" \
  --dataset-name agentic_smoke \
  --dataset-kind agentic_bootstrap \
  --episodes 5 \
  --steps 6 \
  --warmup-steps 1 \
  --image-size 32 \
  --target-radius 8 \
  --min-episode-samples 1 \
  --min-distinct-actions 1 \
  --max-dominant-action-fraction 1.0 \
  --min-episode-policy-entropy 0.0

"$VENV_PY" -m trm_pipeline.dataset_harness run \
  --contract "$AGENTIC_ROOT/contract.json"

echo
echo "Dataset smoke artifacts:"
echo "  $PASSIVE_ROOT/collection_decision.json"
echo "  $AGENTIC_ROOT/collection_decision.json"
