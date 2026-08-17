"""Reproducibility manifest creation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def hardware_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        details.update(
            {
                "torch": torch.__version__,
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except ImportError:
        details["torch"] = None
    return details


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(result_path: str | Path, fields: dict[str, Any]) -> Path:
    result_path = Path(result_path)
    manifest_path = result_path.with_suffix(result_path.suffix + ".manifest.json")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_file": result_path.name,
        "git_revision": git_revision(),
        "hardware": hardware_details(),
        **fields,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
