from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from glossweaver.utils import write_json, write_jsonl

from .base import CanonicalRecord
from .grammar_labels import LABEL_NAMES, extract_grammar_labels
from .aslg_pc12 import PINNED_REVISION, inspect_release, iter_records

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", normalized).strip()


def normalized_duplicate_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", " ", normalized).strip()


def normalized_pair_hash(gloss: str, target: str) -> str:
    payload = normalized_duplicate_key(gloss) + "\0" + normalized_duplicate_key(target)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_records(records: Iterable[CanonicalRecord]) -> tuple[list[CanonicalRecord], dict[str, int]]:
    stats = Counter({
        "raw_rows": 0,
        "removed_empty": 0,
        "removed_exact_duplicate": 0,
        "removed_normalized_duplicate": 0,
    })
    cleaned: list[CanonicalRecord] = []
    exact_seen: set[tuple[str, str]] = set()
    normalized_seen: set[tuple[str, str]] = set()

    for record in records:
        stats["raw_rows"] += 1
        gloss = normalize_text(record.gloss)
        target = normalize_text(record.target)
        if not gloss or not target:
            stats["removed_empty"] += 1
            continue
        exact = (gloss, target)
        normalized = (normalized_duplicate_key(gloss), normalized_duplicate_key(target))
        if exact in exact_seen:
            stats["removed_exact_duplicate"] += 1
            continue
        if normalized in normalized_seen:
            stats["removed_normalized_duplicate"] += 1
            continue
        exact_seen.add(exact)
        normalized_seen.add(normalized)
        stable_id = f"{record.dataset}:{normalized_pair_hash(gloss, target)[:20]}"
        labels = extract_grammar_labels(gloss, target)
        cleaned.append(replace(
            record,
            id=stable_id,
            gloss=gloss,
            target=target,
            grammar_labels=labels,
        ))
    stats["final_unique_pairs"] = len(cleaned)
    return cleaned, dict(stats)


def remove_cross_split_leakage(
    records: Iterable[CanonicalRecord],
) -> tuple[list[CanonicalRecord], dict[str, int]]:
    """Keep the highest-priority split occurrence: test, validation, then train."""
    grouped: dict[tuple[str, str], list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        grouped[(
            normalized_duplicate_key(record.gloss),
            normalized_duplicate_key(record.target),
        )].append(record)
    priority = {"test": 0, "validation": 1, "train": 2}
    kept: list[CanonicalRecord] = []
    removed = Counter()
    for group in grouped.values():
        splits = {item.split for item in group}
        if len(splits) == 1:
            kept.extend(group)
            continue
        winner = min(splits, key=priority.__getitem__)
        for item in group:
            if item.split == winner:
                kept.append(item)
            else:
                removed[f"removed_from_{item.split}"] += 1
    return kept, dict(removed)


def assert_no_cross_split_leakage(records: Iterable[CanonicalRecord]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for record in records:
        key = (
            normalized_duplicate_key(record.gloss),
            normalized_duplicate_key(record.target),
        )
        prior = owners.setdefault(key, record.split)
        if prior != record.split:
            raise AssertionError(f"Normalized pair occurs in both {prior} and {record.split}")


def deterministic_split(
    records: list[CanonicalRecord],
    *,
    seed: int = 42,
) -> list[CanonicalRecord]:
    ordered = sorted(records, key=lambda record: record.id)
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * 0.8)
    validation_end = train_end + int(len(ordered) * 0.1)
    result = []
    for index, record in enumerate(ordered):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        result.append(replace(record, split=split))
    return result


def ambiguous_gloss_stats(records: Iterable[CanonicalRecord]) -> dict[str, int]:
    targets_by_gloss: dict[str, set[str]] = defaultdict(set)
    for record in records:
        targets_by_gloss[normalized_duplicate_key(record.gloss)].add(
            normalized_duplicate_key(record.target)
        )
    ambiguous = {gloss: targets for gloss, targets in targets_by_gloss.items() if len(targets) > 1}
    return {
        "unique_glosses": len(targets_by_gloss),
        "glosses_with_multiple_targets": len(ambiguous),
        "pairs_under_ambiguous_glosses": sum(len(targets) for targets in ambiguous.values()),
    }


def _save_research_ids(
    output: Path,
    records: list[CanonicalRecord],
    *,
    seed: int,
    train_size: int = 20_000,
    validation_size: int = 2_500,
    test_size: int = 2_500,
) -> dict[str, int]:
    requested = {"train": train_size, "validation": validation_size, "test": test_size}
    counts = {}
    for offset, split in enumerate(("train", "validation", "test")):
        candidates = sorted(record.id for record in records if record.split == split)
        random.Random(seed + offset).shuffle(candidates)
        selected = candidates[: min(requested[split], len(candidates))]
        write_json(output / f"research_{split}_ids.json", selected)
        counts[split] = len(selected)
    return counts


def preprocess_aslg_pc12(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    sample_count: int = 10,
) -> dict[str, object]:
    release = inspect_release(raw_dir)
    raw_records = list(iter_records(raw_dir))
    cleaned, cleaning_stats = clean_records(raw_records)
    split_records = deterministic_split(cleaned, seed=seed)
    assert_no_cross_split_leakage(split_records)

    output = Path(output_dir)
    split_counts = Counter(record.split for record in split_records)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", (
            record.as_dict() for record in split_records if record.split == split
        ))

    label_counts = Counter()
    for record in split_records:
        for label, value in zip(LABEL_NAMES, record.grammar_labels or [], strict=True):
            label_counts[label] += value
    stats: dict[str, object] = {
        "dataset": "aslg_pc12",
        "source": release,
        "raw_rows": len(raw_records),
        "split_counts": dict(split_counts),
        "cleaning": cleaning_stats,
        "ambiguous_glosses": ambiguous_gloss_stats(split_records),
        "leakage_check": {
            "normalized_pair_overlap": 0,
            "passed": True,
        },
        "grammar_label_positive_counts": dict(label_counts),
        "split_method": {"official_splits_available": False, "seed": seed, "ratios": [0.8, 0.1, 0.1]},
    }
    manifest = {
        "dataset": "aslg_pc12",
        "source_revision": PINNED_REVISION,
        "seed": seed,
        "pair_hash": "sha256(normalized_gloss + NUL + normalized_target)",
        "splits": {
            split: [record.id for record in split_records if record.split == split]
            for split in ("train", "validation", "test")
        },
    }
    write_json(output / "split_manifest.json", manifest)
    stats["research_subset_counts"] = _save_research_ids(output, split_records, seed=seed)
    write_json(output / "stats.json", stats)

    sampled = random.Random(seed).sample(split_records, min(sample_count, len(split_records)))
    for record in sampled:
        print("-" * 39)
        print(f"Gloss:\n{record.gloss}\n\nEnglish:\n{record.target}")
    print("-" * 39)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare verified ASLG-PC12 gloss-to-English pairs")
    parser.add_argument("--raw-dir", default="datasets/raw/aslg_pc12")
    parser.add_argument("--output-dir", default="datasets/processed/aslg_pc12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()
    stats = preprocess_aslg_pc12(
        args.raw_dir, args.output_dir, seed=args.seed, sample_count=args.sample_count
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
