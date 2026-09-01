#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glossweaver.data.aslg_pc12 import DATASET_ID, PINNED_REVISION, inspect_release
from glossweaver.utils import sha256_file, write_json


def download_aslg_pc12() -> dict[str, object]:
    from huggingface_hub import snapshot_download

    target = PROJECT_ROOT / "datasets/raw/aslg_pc12"
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=PINNED_REVISION,
        local_dir=target,
    )
    report = inspect_release(target)
    parquet = target / "data/train-00000-of-00001.parquet"
    report.update({
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": "CC BY-NC 4.0",
        "parquet_sha256": sha256_file(parquet),
        "local_path": str(target.relative_to(PROJECT_ROOT)),
    })
    write_json(PROJECT_ROOT / "datasets/metadata/aslg_pc12.json", report)
    return report


def status() -> int:
    try:
        report = inspect_release(PROJECT_ROOT / "datasets/raw/aslg_pc12")
        report["available"] = True
        code = 0
    except (FileNotFoundError, ValueError) as exc:
        report = {"dataset_id": DATASET_ID, "available": False, "detail": str(exc)}
        code = 2
    print(json.dumps(report, indent=2))
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the verified active GlossWeaver dataset")
    parser.add_argument("dataset", choices=("status", "aslg-pc12"))
    args = parser.parse_args()
    if args.dataset == "status":
        raise SystemExit(status())
    print(json.dumps(download_aslg_pc12(), indent=2))


if __name__ == "__main__":
    main()
