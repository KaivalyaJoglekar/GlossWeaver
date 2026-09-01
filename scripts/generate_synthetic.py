#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glossweaver.data.corruption import corrupt_sentence
from glossweaver.data.grammar_labels import extract_grammar_labels
from glossweaver.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training-only telegraphic augmentation")
    parser.add_argument("--input", default="datasets/processed/aslg_pc12/train.jsonl")
    parser.add_argument("--ids", default="datasets/processed/aslg_pc12/research_train_ids.json")
    parser.add_argument("--output", default="datasets/synthetic/aslg_pc12/research_train_25.jsonl")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--probability", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")
    records = list(read_jsonl(args.input))
    if any(record.get("split") != "train" for record in records):
        raise ValueError("Synthetic data may only be generated from training records")
    if args.ids:
        with Path(args.ids).open(encoding="utf-8") as handle:
            selected_ids = json.load(handle)
        by_id = {record["id"]: record for record in records}
        missing = [identifier for identifier in selected_ids if identifier not in by_id]
        if missing:
            raise ValueError(f"ID manifest contains {len(missing)} records absent from training data")
        records = [by_id[identifier] for identifier in selected_ids]
    rng = random.Random(args.seed)
    sample_size = round(len(records) * args.ratio)
    selected = rng.sample(records, sample_size)

    def generated():
        for index, record in enumerate(selected):
            result = corrupt_sentence(
                record["target"], seed=args.seed + index, probability=args.probability
            )
            yield {
                "id": f"synthetic:{record['id']}",
                "dataset": "synthetic_telegraphic",
                "gloss": result.fragment,
                "target": result.target,
                "split": "train",
                "source_partition": "training-target-corruption",
                "parent_id": record["id"],
                "operations": list(result.operations),
                "grammar_labels": extract_grammar_labels(result.fragment, result.target),
                "synthetic": True,
            }

    count = write_jsonl(args.output, generated())
    print(f"Wrote {count:,} training-only synthetic records to {args.output}")


if __name__ == "__main__":
    main()
