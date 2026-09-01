from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .base import CanonicalRecord


def iter_aligned_text_pairs(path: str | Path, split: str) -> Iterator[CanonicalRecord]:
    """Load an explicitly aligned text/gloss export; videos are never required."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        gloss_key = next((key for key in ("gloss", "asl_gloss", "SENTENCE") if key in fields), None)
        text_key = next((key for key in ("text", "translation", "english") if key in fields), None)
        if not gloss_key or not text_key:
            raise ValueError(
                "How2Sign text integration requires a verified aligned export with "
                "a gloss column and an English translation column."
            )
        for index, row in enumerate(reader):
            yield CanonicalRecord(
                id=f"how2sign:{split}:{index:07d}",
                dataset="how2sign",
                gloss=row[gloss_key],
                target=row[text_key],
                split=split,
                source_partition=split,
            )

