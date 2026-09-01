from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .base import CanonicalRecord

DATASET_ID = "achrafothman/aslg_pc12"
PINNED_REVISION = "cb7cd272db8fcd4004ee04ddf50e194c15ea24d6"
EXPECTED_ROWS = 87_710
EXPECTED_COLUMNS = {"gloss", "text"}


class ASLGPC12SchemaError(ValueError):
    pass


def resolve_columns(column_names: list[str]) -> tuple[str, str]:
    """Resolve orientation only when the live schema is unambiguous."""
    available = set(column_names)
    if EXPECTED_COLUMNS.issubset(available):
        return "gloss", "text"
    gloss_candidates = available & {"gloss", "asl", "asl_gloss", "source"}
    text_candidates = available & {"text", "english", "translation", "target"}
    if len(gloss_candidates) == 1 and len(text_candidates) == 1:
        return next(iter(gloss_candidates)), next(iter(text_candidates))
    raise ASLGPC12SchemaError(
        "Cannot safely identify ASLG-PC12 orientation. Expected verified columns "
        f"'gloss' and 'text'; found {sorted(available)}."
    )


def load_raw_dataset(
    raw_dir: str | Path = "datasets/raw/aslg_pc12",
    *,
    allow_remote: bool = False,
) -> Any:
    from datasets import load_dataset

    root = Path(raw_dir)
    cache_dir = Path(__file__).resolve().parents[3] / ".hf-cache/datasets"
    parquet_files = sorted((root / "data").glob("*.parquet"))
    if parquet_files:
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            cache_dir=str(cache_dir),
        )
    elif allow_remote:
        dataset = load_dataset(
            DATASET_ID, revision=PINNED_REVISION, cache_dir=str(cache_dir)
        )
    else:
        raise FileNotFoundError(
            f"ASLG-PC12 snapshot not found under {root}. Run "
            "`python scripts/download_data.py aslg-pc12`."
        )
    if set(dataset.keys()) != {"train"}:
        raise ASLGPC12SchemaError(
            f"Verified revision has one train split; found {sorted(dataset.keys())}. "
            "Inspect a new revision before deciding whether its splits are official."
        )
    rows = dataset["train"]
    resolve_columns(rows.column_names)
    if len(rows) != EXPECTED_ROWS:
        raise ASLGPC12SchemaError(
            f"Pinned revision should contain {EXPECTED_ROWS:,} rows; found {len(rows):,}."
        )
    return dataset


def inspect_release(raw_dir: str | Path = "datasets/raw/aslg_pc12") -> dict[str, object]:
    dataset = load_raw_dataset(raw_dir)
    rows = dataset["train"]
    gloss_column, target_column = resolve_columns(rows.column_names)
    return {
        "dataset_id": DATASET_ID,
        "revision": PINNED_REVISION,
        "splits": {"train": len(rows)},
        "columns": list(rows.column_names),
        "gloss_column": gloss_column,
        "target_column": target_column,
        "orientation": "gloss -> English text",
    }


def iter_records(raw_dir: str | Path = "datasets/raw/aslg_pc12") -> Iterator[CanonicalRecord]:
    dataset = load_raw_dataset(raw_dir)
    rows = dataset["train"]
    gloss_column, target_column = resolve_columns(rows.column_names)
    for index, row in enumerate(rows):
        gloss = row.get(gloss_column)
        target = row.get(target_column)
        if gloss is not None and not isinstance(gloss, str):
            raise ASLGPC12SchemaError(f"Row {index} gloss is not a string")
        if target is not None and not isinstance(target, str):
            raise ASLGPC12SchemaError(f"Row {index} target is not a string")
        yield CanonicalRecord(
            id=f"aslg_pc12:raw:{index:06d}",
            dataset="aslg_pc12",
            gloss=gloss or "",
            target=target or "",
            split="unsplit",
            source_partition="huggingface-train",
            metadata={"source_row": index},
        )
