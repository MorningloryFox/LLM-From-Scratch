"""Shared helpers for reproducible local Feneco experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def corpus_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        text_files = sorted(item for item in path.rglob("*.txt") if item.is_file())
        for text_file in text_files:
            digest.update(text_file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with text_file.open("rb") as corpus:
                for chunk in iter(lambda: corpus.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    else:
        with path.open("rb") as corpus:
            for chunk in iter(lambda: corpus.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_working_tree_dirty() -> bool | None:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def environment_summary(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["cuda_version"] = torch.version.cuda
    return result


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
