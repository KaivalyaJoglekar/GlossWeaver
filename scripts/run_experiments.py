#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("study-baseline", "study-augmentation", "study-grammar", "study-full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched FLAN-T5 2 × 2 experiment matrix")
    parser.add_argument("--presets", nargs="*", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    for seed in args.seeds:
        for preset in args.presets:
            command = [sys.executable, "-m", "glossweaver.training.train", "--preset", preset,
                       "--device", args.device, "--seed", str(seed), "--output-dir", str(PROJECT_ROOT / "checkpoints" / preset / f"seed-{seed}")]
            if args.max_steps is not None:
                command += ["--max-steps", str(args.max_steps)]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
