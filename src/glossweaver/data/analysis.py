from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glossweaver.utils import read_jsonl, write_json

from .grammar_labels import LABEL_NAMES, tokenize


def analyze(dataset_dir: str | Path, figure_dir: str | Path) -> dict[str, object]:
    source = Path(dataset_dir)
    figures = Path(figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    records = [record for split in ("train", "validation", "test") for record in read_jsonl(source / f"{split}.jsonl")]
    if not records:
        raise ValueError("Cannot run EDA on an empty dataset")
    gloss_lengths = [len(tokenize(record["gloss"])) for record in records]
    target_lengths = [len(tokenize(record["target"])) for record in records]
    gloss_tokens = Counter(token for record in records for token in tokenize(record["gloss"]))
    target_tokens = Counter(token for record in records for token in tokenize(record["target"]))
    labels = Counter()
    for record in records:
        for name, value in zip(LABEL_NAMES, record.get("grammar_labels", []), strict=True):
            labels[name] += value

    for values, title, filename in (
        (gloss_lengths, "Gloss length", "gloss_length.png"),
        (target_lengths, "Target length", "target_length.png"),
    ):
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50)
        plt.xlabel("Tokens")
        plt.ylabel("Examples")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=160)
        plt.close()
    plt.figure(figsize=(7, 6))
    plt.scatter(gloss_lengths, target_lengths, alpha=0.15, s=8)
    plt.xlabel("Gloss tokens")
    plt.ylabel("Target tokens")
    plt.title("Gloss vs target length")
    plt.tight_layout()
    plt.savefig(figures / "length_scatter.png", dpi=160)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.bar(labels.keys(), labels.values())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Positive weak labels")
    plt.title("Grammar recovery label distribution")
    plt.tight_layout()
    plt.savefig(figures / "grammar_labels.png", dpi=160)
    plt.close()

    stats = {
        "examples": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "missing_gloss": sum(not record["gloss"].strip() for record in records),
        "missing_target": sum(not record["target"].strip() for record in records),
        "gloss_vocabulary": len(gloss_tokens),
        "target_vocabulary": len(target_tokens),
        "top_gloss_tokens": gloss_tokens.most_common(50),
        "top_target_tokens": target_tokens.most_common(50),
        "grammar_labels": dict(labels),
        "gloss_length": {"min": min(gloss_lengths), "max": max(gloss_lengths), "mean": sum(gloss_lengths) / len(gloss_lengths)},
        "target_length": {"min": min(target_lengths), "max": max(target_lengths), "mean": sum(target_lengths) / len(target_lengths)},
    }
    write_json(figures / "eda_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-fabricated EDA on processed splits")
    parser.add_argument("--dataset-dir", default="datasets/processed/aslg_pc12")
    parser.add_argument("--figure-dir", default="results/figures")
    args = parser.parse_args()
    stats = analyze(args.dataset_dir, args.figure_dir)
    print(f"Analyzed {stats['examples']:,} examples")


if __name__ == "__main__":
    main()
