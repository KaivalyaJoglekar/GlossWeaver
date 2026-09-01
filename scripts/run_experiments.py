#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "config/t5_baseline.yaml",
    "config/t5_augmented.yaml",
    "config/gat5.yaml",
    "config/gat5_augmented.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fair E1-E4 experiment matrix")
    parser.add_argument("--configs", nargs="*", choices=CONFIGS, default=list(CONFIGS))
    args = parser.parse_args()
    for config in args.configs:
        subprocess.run(
            [sys.executable, "-m", "glossweaver.training.train", "--config", config],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()

