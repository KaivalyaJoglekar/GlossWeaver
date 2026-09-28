#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.api import InferRequest, infer
from glossweaver.inference import Reconstructor


def main() -> None:
    glosses = [
        "ME GO STORE YESTERDAY",
        "TEACHER EXPLAIN STUDENT MATH",
        "CHILD PLAY OUTSIDE SUNNY",
        "BOY THROW BALL STRONG",
    ]
    direct = Reconstructor("e4_glossweaver", device="auto")
    rows = []
    for gloss in glosses:
        direct_prediction, _ = direct.reconstruct(gloss)
        response = infer(InferRequest(gloss=gloss, checkpoint="e4_glossweaver"))
        rows.append({
            "gloss": gloss,
            "direct_prediction": direct_prediction,
            "api_prediction": response.reconstruction,
            "exact_match": direct_prediction == response.reconstruction,
            "direct_checkpoint": direct.debug_info["checkpoint_path"],
            "api_checkpoint": response.debug["checkpoint_path"],
        })
    output = ROOT / "results/debug/inference_parity.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    if not all(row["exact_match"] for row in rows):
        raise SystemExit("Direct/API inference mismatch")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
