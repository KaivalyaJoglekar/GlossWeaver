#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glossweaver.evaluation.metrics import compute_metrics
from glossweaver.inference import Reconstructor
from glossweaver.settings import get_preset
from glossweaver.training.train import _records, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify the 64-pair memorization gate")
    parser.add_argument("--preset", choices=("overfit-t5", "overfit-flan"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    config = get_preset(args.preset)
    from dataclasses import replace
    config = replace(config, device=args.device)
    if not args.skip_training:
        run_training(config)
    records = _records(config.train_file, max_examples=64, seed=config.seed)
    reconstructor = Reconstructor(str(config.checkpoint_dir), device=args.device)
    rows = []
    for record in records:
        prediction, _ = reconstructor.reconstruct(record["gloss"], num_beams=4)
        rows.append({
            "id": record["id"], "gloss": record["gloss"],
            "reference": record["target"], "prediction": prediction,
        })
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    glosses = [row["gloss"] for row in rows]
    metrics = compute_metrics(predictions, references, include_bertscore=False, glosses=glosses)
    debug_dir = ROOT / "results/debug"
    figure_dir = ROOT / "results/figures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(debug_dir / f"{config.experiment_name}_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "experiment": config.experiment_name}]).to_csv(
        debug_dir / f"{config.experiment_name}_metrics.csv", index=False
    )
    history = pd.read_csv(config.history_csv)
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["global_step"], history["train_generation_loss"], marker="o", label="Train generation loss")
    plt.plot(history["global_step"], history["validation_generation_loss"], marker="o", label="Validation generation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.title(f"64-pair overfit sanity check: {config.experiment_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"{config.experiment_name}_loss.png", dpi=180)
    plt.close()
    print(pd.DataFrame([{**metrics, "experiment": config.experiment_name}]).to_string(index=False))
    print(pd.DataFrame(rows).head(20).to_string(index=False))
    if float(metrics["sacrebleu"]) < 80 or float(metrics["rouge_l"]) < 0.9:
        raise SystemExit("Overfit gate failed: BLEU < 80 or ROUGE-L < 0.90")


if __name__ == "__main__":
    main()
