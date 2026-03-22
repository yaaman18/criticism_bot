#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SEED_CATALOG="${SEED_CATALOG:-data/lenia_official/animals2d_seeds.json}"
TRM_ROLLOUTS_DIR="${TRM_ROLLOUTS_DIR:-data/trm_rollouts_vast}"
TRM_A_DIR="${TRM_A_DIR:-artifacts/trm_a_vast}"
TRM_B_CACHE_DIR="${TRM_B_CACHE_DIR:-data/trm_b_cache_vast}"
TRM_B_DIR="${TRM_B_DIR:-artifacts/trm_b_vast}"
TRM_VA_CACHE_DIR="${TRM_VA_CACHE_DIR:-data/trm_va_cache_vast}"
TRM_VM_DIR="${TRM_VM_DIR:-artifacts/trm_vm_vast}"
TRM_AS_DIR="${TRM_AS_DIR:-artifacts/trm_as_vast}"
MODULE_MANIFEST="${MODULE_MANIFEST:-artifacts/modules_vm_as_vast.json}"
COMPARE_DIR="${COMPARE_DIR:-artifacts/trm_va_mode_compare_vast}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

STAGE="${1:-all}"

run_prepare_rollouts() {
  python -m trm_pipeline.lenia_data \
    --seed-catalog "$SEED_CATALOG" \
    --output-root "$TRM_ROLLOUTS_DIR"
}

run_train_trm_a() {
  python -m trm_pipeline.train_trm_a \
    --manifest "$TRM_ROLLOUTS_DIR/manifest.jsonl" \
    --output-dir "$TRM_A_DIR" \
    --objective variational \
    --device cuda \
    --grad-clip 1.0 \
    --amp \
    --log-interval "$LOG_INTERVAL"
}

run_prepare_trm_b() {
  python -m trm_pipeline.prepare_trm_b_data \
    --manifest "$TRM_ROLLOUTS_DIR/manifest.jsonl" \
    --checkpoint "$TRM_A_DIR/trm_a.pt" \
    --output-root "$TRM_B_CACHE_DIR"
}

run_train_trm_b() {
  python -m trm_pipeline.train_trm_b \
    --manifest "$TRM_B_CACHE_DIR/manifest.jsonl" \
    --output-dir "$TRM_B_DIR" \
    --device cuda \
    --grad-clip 1.0 \
    --amp \
    --log-interval "$LOG_INTERVAL"
}

run_prepare_trm_va() {
  python -m trm_pipeline.prepare_trm_va_data \
    --seed-catalog "$SEED_CATALOG" \
    --output-root "$TRM_VA_CACHE_DIR" \
    --episodes 64 \
    --steps 32 \
    --warmup-steps 4
}

run_train_trm_vm() {
  python -m trm_pipeline.train_trm_vm \
    --manifest "$TRM_VA_CACHE_DIR/manifest.jsonl" \
    --output-dir "$TRM_VM_DIR" \
    --device cuda \
    --grad-clip 1.0 \
    --amp \
    --log-interval "$LOG_INTERVAL"
}

run_train_trm_as() {
  python -m trm_pipeline.train_trm_as \
    --manifest "$TRM_VA_CACHE_DIR/manifest.jsonl" \
    --output-dir "$TRM_AS_DIR" \
    --device cuda \
    --grad-clip 1.0 \
    --amp \
    --log-interval "$LOG_INTERVAL"
}

run_write_manifest() {
  python - <<'PY'
import json
import os
from pathlib import Path

manifest = [
    {"id": "world_primary", "name": "trm_a", "checkpoint": os.environ["TRM_A_DIR"] + "/trm_a.pt", "primary": True},
    {"id": "boundary_primary", "name": "trm_b", "checkpoint": os.environ["TRM_B_DIR"] + "/trm_b.pt", "primary": True},
    {"id": "vm_primary", "name": "trm_vm", "checkpoint": os.environ["TRM_VM_DIR"] + "/trm_vm.pt", "primary": True},
    {"id": "as_primary", "name": "trm_as", "checkpoint": os.environ["TRM_AS_DIR"] + "/trm_as.pt", "primary": True},
]
path = Path(os.environ["MODULE_MANIFEST"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
PY
}

run_compare_modes() {
  python -m trm_pipeline.compare_trm_va_modes \
    --seed-catalog "$SEED_CATALOG" \
    --module-manifest "$MODULE_MANIFEST" \
    --output-root "$COMPARE_DIR"
}

export TRM_A_DIR TRM_B_DIR TRM_VM_DIR TRM_AS_DIR MODULE_MANIFEST

case "$STAGE" in
  prepare_rollouts) run_prepare_rollouts ;;
  train_trm_a) run_train_trm_a ;;
  prepare_trm_b) run_prepare_trm_b ;;
  train_trm_b) run_train_trm_b ;;
  prepare_trm_va) run_prepare_trm_va ;;
  train_trm_vm) run_train_trm_vm ;;
  train_trm_as) run_train_trm_as ;;
  write_manifest) run_write_manifest ;;
  compare_modes) run_compare_modes ;;
  all)
    run_prepare_rollouts
    run_train_trm_a
    run_prepare_trm_b
    run_train_trm_b
    run_prepare_trm_va
    run_train_trm_vm
    run_train_trm_as
    run_write_manifest
    run_compare_modes
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac
