"""Run directory and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_DIRS = ("txt", "png", "metrics", "json", "model", "npz", "dat")


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for name in ARTIFACT_DIRS:
        path = run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def save_table(path: Path, array: np.ndarray, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, header=header)
