"""Source-group isolation and reproducible, origin-stratified evaluation slices."""
from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from typing import Any


def source_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def lineage_keys(record: dict) -> set[str]:
    """Link explicit lineage, normalized targets, and identical fragments."""
    keys = {"gloss:" + source_key(str(record["gloss"]))}
    for field in ("source_sentence", "target"):
        if record.get(field):
            keys.add("sentence:" + source_key(str(record[field])))
    for field in ("source_id", "source_sentence_id", "lineage_id"):
        if record.get(field):
            keys.add("lineage:" + str(record[field]))
    return keys


def group_split(records: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    # Connected components keep variants AND multiple references in one split.
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, record in enumerate(records):
        for key in lineage_keys(record):
            if key in owners:
                a, b = root(i), root(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    ordered = sorted(groups.values(), key=lambda group: sorted(k for r in group for k in lineage_keys(r)))
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * .8)
    val_end = train_end + int(len(ordered) * .1)
    return [dict(r, split="train" if i < train_end else "validation" if i < val_end else "test") for i, group in enumerate(ordered) for r in group]


def assert_source_isolation(train: list[dict], validation: list[dict], test: list[dict] | None = None) -> None:
    owners = {}
    for split, records in (("train", train), ("validation", validation), ("test", test or [])):
        for record in records:
            for key in lineage_keys(record):
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"Source overlap between {previous} and {split}; rebuild grouped splits before training")


def stratified_sample(records: list[dict], maximum: int | None, seed: int = 42) -> list[dict]:
    if maximum is None or maximum >= len(records):
        return list(records)
    if maximum < 1:
        raise ValueError("Sample size must be positive")
    groups = defaultdict(list)
    for record in sorted(records, key=lambda item: str(item["id"])):
        groups[str(record.get("corpus_origin") or record.get("dataset") or "unknown")].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < maximum:
        for key in sorted(groups):
            if groups[key] and len(result) < maximum:
                result.append(groups[key].pop())
    return result
