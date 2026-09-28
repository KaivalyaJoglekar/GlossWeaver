#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glossweaver.inference import Reconstructor
from glossweaver.model_registry import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct deterministic challenge-set inference")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    challenge = pd.read_csv(ROOT / "data/challenge/glossweaver_challenge.csv")
    reconstructor = Reconstructor(args.model, device=args.device)
    rows = []
    for record in challenge.to_dict(orient="records"):
        prediction, grammar = reconstructor.reconstruct(record["gloss"], args.num_beams)
        row = {**record, "model": args.model, "prediction": prediction}
        row.update({f"grammar_{key.lower()}": value for key, value in grammar.items()})
        rows.append(row)
        print(f"{record['gloss']} -> {prediction}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    print("\nActive checkpoint")
    for key, value in reconstructor.debug_info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
