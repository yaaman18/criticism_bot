#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trm_pipeline.common import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit non-zero unless promotion_decision status is promote.")
    parser.add_argument("--decision", required=True, help="Path to promotion_decision.json")
    parser.add_argument("--allow-blocked", action="store_true", help="Treat blocked as success.")
    args = parser.parse_args()

    decision_path = Path(args.decision)
    data = load_json(decision_path)
    status = str(data.get("status", "unknown"))
    ci_summary = str(data.get("ci_summary", ""))
    recommendation = str(data.get("recommendation", ""))

    message = f"promotion_decision status={status}"
    if ci_summary:
        message += f" | {ci_summary}"
    elif recommendation:
        message += f" | {recommendation}"
    print(message)

    if status == "promote":
        return
    if status == "blocked" and args.allow_blocked:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
