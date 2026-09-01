from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from glossweaver.evaluation.grammar_metrics import grammar_diagnostics
from glossweaver.evaluation.metrics import compute_metrics
from glossweaver.data.grammar_labels import LABEL_NAMES
from glossweaver.modeling.baseline import CopyHeuristicBaseline
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.utils import read_jsonl, resolve_device, write_json


def evaluate_checkpoint(
    checkpoint: str,
    dataset_file: str,
    output_csv: str,
    output_metrics: str,
    *,
    batch_size: int = 8,
    num_beams: int = 4,
    skip_bertscore: bool = False,
    ids_file: str | None = None,
    max_examples: int | None = None,
    device: str = "auto",
) -> dict[str, object]:
    records = list(read_jsonl(dataset_file))
    if ids_file:
        with Path(ids_file).open(encoding="utf-8") as handle:
            identifiers = json.load(handle)
        by_id = {record["id"]: record for record in records}
        records = [by_id[identifier] for identifier in identifiers]
    if max_examples is not None:
        records = records[:max_examples]
    glosses = [record["gloss"] for record in records]
    references = [record["target"] for record in records]
    grammar_probabilities = None
    if checkpoint == "e0-copy":
        predictions = CopyHeuristicBaseline().predict_batch(glosses)
    else:
        config = AutoConfig.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        grammar_aware = bool(getattr(config, "glossweaver_grammar_aware", False))
        model = GrammarAwareT5.from_pretrained(checkpoint) if grammar_aware else T5ForConditionalGeneration.from_pretrained(checkpoint)
        device = resolve_device(device)
        model.to(device).eval()
        predictions = []
        probability_rows: list[list[float]] = []
        for start in range(0, len(glosses), batch_size):
            batch_glosses = glosses[start:start + batch_size]
            encoded = tokenizer(
                ["reconstruct gloss: " + gloss for gloss in batch_glosses],
                padding=True, truncation=True, max_length=128, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated = model.generate(**encoded, num_beams=num_beams, max_new_tokens=128)
                if grammar_aware:
                    output = model(**encoded)
                    probability_rows.extend(torch.sigmoid(output.grammar_logits).cpu().tolist())
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        grammar_probabilities = probability_rows if grammar_aware else None

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "gloss", "reference", "prediction"]
        if grammar_probabilities is not None:
            fields.extend(f"grammar_probability_{name.lower()}" for name in LABEL_NAMES)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
            row = {
                "id": record["id"], "gloss": record["gloss"],
                "reference": record["target"], "prediction": prediction,
            }
            if grammar_probabilities is not None:
                row.update({
                    f"grammar_probability_{name.lower()}": value
                    for name, value in zip(LABEL_NAMES, grammar_probabilities[index], strict=True)
                })
            writer.writerow(row)
    metrics: dict[str, object] = compute_metrics(
        predictions, references, include_bertscore=not skip_bertscore
    )
    metrics["grammar_diagnostics"] = grammar_diagnostics(glosses, references, predictions)
    metrics["checkpoint"] = checkpoint
    metrics["examples"] = len(records)
    write_json(output_metrics, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E0 or a trained GlossWeaver checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or e0-copy")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--dataset-dir", default="datasets/processed/aslg_pc12")
    parser.add_argument("--ids")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output-name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    name = args.output_name or ("e0_copy" if args.checkpoint == "e0-copy" else Path(args.checkpoint).name)
    metrics = evaluate_checkpoint(
        args.checkpoint,
        str(Path(args.dataset_dir) / f"{args.split}.jsonl"),
        f"results/predictions/{name}.csv",
        f"results/metrics/{name}.json",
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        skip_bertscore=args.skip_bertscore,
        ids_file=args.ids,
        max_examples=args.max_examples,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
