"""Strict human gloss → English adapter preserving sign annotation notation."""
from __future__ import annotations
from pathlib import Path
from glossweaver.utils import read_jsonl, write_jsonl, write_json
from .preprocess import normalize_text
from .split_integrity import source_key


def prepare_flores(raw_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    owners, counts, review = {}, {}, []
    for split in ("devtest", "dev"):
        records, seen = [], set()
        for index, row in enumerate(read_jsonl(raw_dir / f"{split}.jsonl")):
            if not isinstance(row.get("gloss"), str) or not isinstance(row.get("sentence"), str):
                raise ValueError(f"Invalid gloss/sentence schema in {split} row {index}")
            gloss, target = normalize_text(row["gloss"]), normalize_text(row["sentence"])
            if not gloss or not target:
                review.append(dict(row, split=split, reason="empty_pair"))
                continue
            identity = (gloss, target, row.get("signer"))
            if identity in seen:
                continue
            seen.add(identity)
            key = source_key(gloss)
            if key in owners and owners[key] != split:
                review.append(dict(row, split=split, reason="cross_split_source_overlap_keep_devtest"))
                continue
            owners[key] = split
            records.append({
                "id": f"2m_flores_asl:{split}:{row['id']}:{index}",
                "dataset": "2m_flores_asl", "gloss": gloss, "target": target,
                "split": split, "source_partition": split, "annotation_type": "human",
                "document_url": row.get("URL"), "topic": row.get("topic"),
                "signer_local_id": row.get("signer"),
            })
        write_jsonl(output / f"{split}.jsonl", records)
        counts[split] = len(records)
    write_jsonl(output / "quarantine.jsonl", review)
    write_json(output / "manifest.json", {
        "dataset": "2m_flores_asl", "normalization": "NFC/whitespace only; sign symbols preserved",
        "counts": counts, "quarantined": len(review), "normalized_source_overlap": 0,
        "usage": "dev for external development; devtest frozen for final evaluation; no train split",
        "signer_warning": "signer is not a global signer identifier",
    })
    return counts
