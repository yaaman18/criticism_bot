#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VIEWER_ROOT="$ROOT_DIR/openframeworks/erie_life_viewer"
OUTPUT_ROOT="$ROOT_DIR/artifacts/of_viewer_live_check"
NPZ_PATH=""
JOBS=4
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_erie_life_viewer.sh --npz <path-to-runtime.npz> [options]

Options:
  --npz <path>             Input ERIE runtime .npz (required)
  --output-root <path>     Export output root (default: artifacts/of_viewer_live_check)
  --viewer-root <path>     openFrameworks viewer root (default: openframeworks/erie_life_viewer)
  --jobs <n>               make parallel jobs (default: 4)
  --skip-build             Skip make step
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npz)
      NPZ_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --viewer-root)
      VIEWER_ROOT="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$NPZ_PATH" ]]; then
  echo "Error: --npz is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$NPZ_PATH" ]]; then
  echo "Error: npz not found: $NPZ_PATH" >&2
  exit 1
fi

if [[ ! -d "$VIEWER_ROOT" ]]; then
  echo "Error: viewer root not found: $VIEWER_ROOT" >&2
  exit 1
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "[1/3] Exporting frames from: $NPZ_PATH"
"$PYTHON_BIN" -m trm_pipeline.export_erie_openframeworks_frames \
  --npz "$NPZ_PATH" \
  --output-root "$OUTPUT_ROOT"

SESSION_DIR="$VIEWER_ROOT/bin/data/session"
echo "[2/3] Syncing session files into: $SESSION_DIR"
mkdir -p "$SESSION_DIR/frames"
cp "$OUTPUT_ROOT/manifest.json" "$SESSION_DIR/manifest.json"
rm -f "$SESSION_DIR/frames/"*.png
cp "$OUTPUT_ROOT/frames/"*.png "$SESSION_DIR/frames/"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[3/3] Building openFrameworks viewer"
  make -j"$JOBS" -C "$VIEWER_ROOT"
else
  echo "[3/3] Build skipped (--skip-build)"
fi

echo "Done."
echo "Run viewer:"
echo "  make -C \"$VIEWER_ROOT\" RunRelease"
