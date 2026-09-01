from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_device(requested: str = "auto") -> str:
    import torch

    allowed = {"auto", "cpu", "mps", "cuda"}
    if requested not in allowed:
        raise ValueError(f"Unknown device {requested!r}; choose one of {sorted(allowed)}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS support")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but PyTorch cannot access Apple's Metal GPU. "
                "Run the command in a normal macOS Terminal and verify with "
                "`python -c \"import torch; print(torch.backends.mps.is_available())\"`."
            )
    return requested


def choose_device() -> str:
    """Select the fastest available device without requiring one explicitly."""
    return resolve_device("auto")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temp_path, output)
    return count


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, output)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
