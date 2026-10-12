from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from time import perf_counter

import pandas as pd

from glossweaver.evaluation.grammar_metrics import grammar_diagnostics
from glossweaver.evaluation.metrics import compute_metrics, compute_multi_reference_metrics
from glossweaver.inference import Reconstructor
from glossweaver.semantics import validate_meaning
from glossweaver.model_registry import MODELS
from glossweaver.settings import ROOT
from glossweaver.utils import read_jsonl, write_json, sha256_file, code_provenance
from glossweaver.data.split_integrity import stratified_sample


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
    include_bertscore: bool = False,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 1 <= num_beams <= 10:
        raise ValueError("num_beams must be between 1 and 10")
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
        records = stratified_sample(records, max_examples, seed)
    if not records:
        raise ValueError("Evaluation dataset is empty")
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
            diagnostics = validate_meaning(str(record["gloss"]), prediction)
            row["semantic_issues"] = json.dumps(diagnostics["issues"])
            row.update({f"grammar_probability_{key.lower()}": value for key, value in grammar.items()})
            rows.append(row)
    runtime_minutes = (perf_counter() - start) / 60
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    glosses = [str(row["gloss"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    if any(record.get("reference_2") for record in records):
        reference_sets = [
            [str(record[key]) for key in ("reference_1", "reference_2", "reference_3") if record.get(key)] or [str(record.get("target", ""))]
            for record in records
        ]
        metrics = compute_multi_reference_metrics(
            predictions, reference_sets, include_bertscore=include_bertscore, glosses=glosses
        )
    else:
        metrics = compute_metrics(
            predictions, references, include_bertscore=include_bertscore, glosses=glosses
        )
    semantic_rows = [validate_meaning(g, p) for g, p in zip(glosses, predictions, strict=True)]
    semantic_metrics = {}
    for category in ("negation", "numbers", "dates", "temporal", "named_entities"):
        checks = [c for result in semantic_rows for c in result["checks"] if c["category"] == category and c["status"] != "unavailable"]
        semantic_metrics[category + "_no_issue_rate"] = sum(c["status"] == "no_issue_detected" for c in checks) / len(checks) if checks else None
        semantic_metrics[category + "_checked_examples"] = len(checks)
    auxiliary = {"grammar_auxiliary_accuracy": None, "grammar_auxiliary_checked_labels": 0}
    correct = checked = 0
    for record, row in zip(records, rows, strict=True):
        expected = record.get("grammar_labels")
        mask = record.get("grammar_mask")
        if isinstance(expected, str):
            expected = json.loads(expected) if expected else None
        if isinstance(mask, str):
            mask = json.loads(mask) if mask else None
        if expected is None:
            continue
        from glossweaver.data.grammar_labels import LABEL_NAMES
        for index, name in enumerate(LABEL_NAMES):
            value = row.get("grammar_probability_" + name.lower())
            if value is None or index >= len(expected) or expected[index] not in (0, 1) or (mask and not mask[index]):
                continue
            checked += 1
            correct += int((value >= .5) == bool(expected[index]))
    if checked:
        auxiliary = {"grammar_auxiliary_accuracy": correct / checked, "grammar_auxiliary_checked_labels": checked}
    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    checkpoint = reconstructor.info.checkpoint_path
    weight = next((checkpoint / name for name in ("model.safetensors", "pytorch_model.bin") if (checkpoint / name).exists()), None) if checkpoint else None
    checkpoint_sha256 = sha256_file(weight) if weight else "copy-baseline"
    metric_row: dict[str, object] = {
        **semantic_metrics,
        **auxiliary,
        "latency_ms_per_example": runtime_minutes * 60000 / len(rows),
        "input_formatter": reconstructor.info.input_formatter,
        "evaluation_code_revision": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None,
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "evaluation_batch_size": batch_size,
        "evaluation_device": device,
        "evaluation_id": evaluation_id,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": sha256_file(Path(dataset_file)),
        "sample_sha256": hashlib.sha256(json.dumps([r["id"] for r in records]).encode()).hexdigest(),
        "model": model_key,
        "dataset": dataset_name,
        "split": split,
        "train_examples": "",
        "seed": seed,
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
        "chrf": metrics["chrf"], "exact_match": metrics["exact_match"],
        "sacrebleu_signature": metrics["sacrebleu_signature"],
        "bertscore_model": metrics.get("bertscore_model", "skipped"),
        "grammar_label_policy": __import__("os").environ.get("GLOSSWEAVER_GRAMMAR_PARSER", "lexical-v1"),
    }
    if checkpoint and (checkpoint / "training_metadata.csv").exists():
        metadata = pd.read_csv(checkpoint / "training_metadata.csv").iloc[0]
        metric_row["train_examples"] = int(metadata["train_examples"])
    snapshot = ROOT / "results/evaluations" / evaluation_id
    snapshot.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(snapshot / "predictions.csv", index=False)
    write_json(snapshot / "manifest.json", {"metrics": metric_row, "ids": [r["id"] for r in records], "source_file": str(dataset_file), "output": str(output), "code_provenance": code_provenance()})
    metrics_path = ROOT / "results/metrics/model_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        frame = pd.concat([frame, pd.DataFrame([metric_row])], ignore_index=True)
    else:
        frame = pd.DataFrame([metric_row])
    frame.to_csv(metrics_path, index=False)
    diagnostics = grammar_diagnostics(glosses, references, predictions)
    diagnostic_rows = [
        {"evaluation_id": evaluation_id, "model": model_key, "dataset": dataset_name, "split": split, "category": category, **values}
        for category, values in diagnostics.items()
    ]
    diagnostics_path = ROOT / "results/tables/grammar_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    if diagnostic_rows:
        existing = pd.read_csv(diagnostics_path) if diagnostics_path.exists() and diagnostics_path.stat().st_size > 1 else pd.DataFrame()
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
    parser.add_argument("--skip-bertscore", action="store_true", help="Compatibility flag; BERTScore is already disabled by default")
    parser.add_argument("--bertscore", action="store_true", help="Explicitly enable BERTScore; may download its evaluator model")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_model(
        args.model, args.dataset_file, args.output, dataset_name=args.dataset_name,
        split=args.split, ids_file=args.ids, max_examples=args.max_examples,
        num_beams=args.num_beams, include_bertscore=args.bertscore and not args.skip_bertscore,
        device=args.device, batch_size=args.batch_size, seed=args.seed,
    )


if __name__ == "__main__":
    main()
