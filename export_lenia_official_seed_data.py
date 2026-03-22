#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/Users/yamaguchimitsuyuki/Lenia_official/Python")
OUT_ROOT = Path("/Users/yamaguchimitsuyuki/criticism_bot/data/lenia_official")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def normalize_entry(entry: dict[str, Any], source_file: str, dim: int) -> dict[str, Any] | None:
    if "params" not in entry or "cells" not in entry:
        return None
    params = entry.get("params")
    if not isinstance(params, dict):
        return None
    return {
        "source_file": source_file,
        "dim": dim,
        "code": entry.get("code", ""),
        "name": entry.get("name", ""),
        "cname": entry.get("cname", ""),
        "params": {
            "R": params.get("R"),
            "T": params.get("T"),
            "b": params.get("b"),
            "m": params.get("m"),
            "s": params.get("s"),
            "kn": params.get("kn"),
            "gn": params.get("gn"),
        },
        "cells": entry.get("cells", ""),
    }


def export_source(filename: str, dim: int) -> list[dict[str, Any]]:
    src = ROOT / filename
    rows = load_json(src)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = normalize_entry(row, filename, dim)
        if item is not None:
            out.append(item)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    rows_2d = export_source("animals.json", 2)
    rows_3d = export_source("animals3D.json", 3)
    rows_4d = export_source("animals4D.json", 4)
    all_rows = rows_2d + rows_3d + rows_4d

    write_json(OUT_ROOT / "animals2d_seeds.json", rows_2d)
    write_json(OUT_ROOT / "animals3d_seeds.json", rows_3d)
    write_json(OUT_ROOT / "animals4d_seeds.json", rows_4d)
    write_jsonl(OUT_ROOT / "all_seeds.jsonl", all_rows)

    summary = {
        "source_root": str(ROOT),
        "counts": {
            "animals2d": len(rows_2d),
            "animals3d": len(rows_3d),
            "animals4d": len(rows_4d),
            "all": len(all_rows),
        },
        "output_files": [
            "animals2d_seeds.json",
            "animals3d_seeds.json",
            "animals4d_seeds.json",
            "all_seeds.jsonl",
        ],
    }
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
