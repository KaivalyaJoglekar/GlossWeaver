from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from glossweaver.evaluation.grammar_metrics import grammar_diagnostics
from glossweaver.evaluation.metrics import compute_metrics, compute_multi_reference_metrics
from glossweaver.inference import Reconstructor
from glossweaver.model_registry import MODELS
from glossweaver.settings import ROOT
from glossweaver.utils import read_jsonl


def _load(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".csv":
        return pd.read_csv(path).fillna("").to_dict(orient="records")
    return list(read_jsonl(path))


def evaluate_model(
    model_key: str,
    dataset_file: str | Path,
    output_csv: str | Path,
    *,
    dataset_name: str,
    split: str,
    ids_file: str | Path | None = None,
    max_examples: int | None = None,
    num_beams: int = 4,
    batch_size: int = 16,
    include_bertscore: bool = True,
    device: str = "auto",
) -> dict[str, object]:
    records = _load(Path(dataset_file))
    if ids_file:
        ids_path = Path(ids_file)
        if ids_path.suffix == ".csv":
            identifiers = pd.read_csv(ids_path)["id"].astype(str).tolist()
        else:
            with ids_path.open(encoding="utf-8") as handle:
                identifiers = json.load(handle)
        by_id = {str(record["id"]): record for record in records}
        records = [by_id[identifier] for identifier in identifiers]
    if max_examples is not None:
        records = records[:max_examples]
    reconstructor = Reconstructor(model_key, device=device)
    start = perf_counter()
    rows: list[dict[str, object]] = []
    for start_index in range(0, len(records), batch_size):
        batch = records[start_index:start_index + batch_size]
        outputs = reconstructor.reconstruct_batch(
            [str(record["gloss"]) for record in batch], num_beams
        )
        for record, (prediction, grammar) in zip(batch, outputs, strict=True):
            row = {
                "id": record["id"], "gloss": record["gloss"],
                "reference": record.get("target", record.get("reference_1", "")),
                "prediction": prediction, "model": model_key,
            }
            row.update({f"grammar_probability_{key.lower()}": value for key, value in grammar.items()})
            rows.append(row)
    runtime_minutes = (perf_counter() - start) / 60
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    glosses = [str(row["gloss"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    if records and records[0].get("reference_2"):
        reference_sets = [
            [str(record[key]) for key in ("reference_1", "reference_2", "reference_3") if record.get(key)]
            for record in records
        ]
        metrics = compute_multi_reference_metrics(
            predictions, reference_sets, include_bertscore=include_bertscore, glosses=glosses
        )
    else:
        metrics = compute_metrics(
            predictions, references, include_bertscore=include_bertscore, glosses=glosses
        )
    metric_row: dict[str, object] = {
        "model": model_key,
        "dataset": dataset_name,
        "split": split,
        "train_examples": "",
        "seed": 42,
        "bleu": metrics["sacrebleu"],
        "rouge_l": metrics["rouge_l"],
        "meteor": metrics["meteor"],
        "bertscore_f1": metrics.get("bertscore_f1", ""),
        "content_word_recall": metrics["content_word_recall"],
        "copy_ratio": metrics["copy_ratio"],
        "runtime_minutes": runtime_minutes,
        "checkpoint": reconstructor.debug_info["checkpoint_path"],
        "examples": len(rows),
        "num_beams": num_beams,
    }
    metrics_path = ROOT / "results/metrics/model_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        keep = ~(
            (frame["model"] == model_key)
            & (frame["dataset"] == dataset_name)
            & (frame["split"] == split)
            & (frame["num_beams"] == num_beams)
        )
        frame = frame[keep]
        frame = pd.concat([frame, pd.DataFrame([metric_row])], ignore_index=True)
    else:
        frame = pd.DataFrame([metric_row])
    frame.to_csv(metrics_path, index=False)
    diagnostics = grammar_diagnostics(glosses, references, predictions)
    diagnostic_rows = [
        {"model": model_key, "dataset": dataset_name, "split": split, "category": category, **values}
        for category, values in diagnostics.items()
    ]
    diagnostics_path = ROOT / "results/tables/grammar_diagnostics.csv"
    existing = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    if not existing.empty and "model" in existing:
        existing = existing[~((existing["model"] == model_key) & (existing["dataset"] == dataset_name) & (existing["split"] == split))]
    pd.concat([existing, pd.DataFrame(diagnostic_rows)], ignore_index=True).to_csv(diagnostics_path, index=False)
    print(pd.DataFrame([metric_row]).to_string(index=False))
    return metric_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a registered GlossWeaver model")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate_model(
        args.model, args.dataset_file, args.output, dataset_name=args.dataset_name,
        split=args.split, ids_file=args.ids, max_examples=args.max_examples,
        num_beams=args.num_beams, include_bertscore=not args.skip_bertscore,
        device=args.device, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
