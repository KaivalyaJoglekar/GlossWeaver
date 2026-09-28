#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("e1", "e2", "e3", "e4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fair E1-E4 experiment matrix")
    parser.add_argument("--presets", nargs="*", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    for preset in args.presets:
        subprocess.run(
            [sys.executable, "-m", "glossweaver.training.train", "--preset", preset, "--device", args.device],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
