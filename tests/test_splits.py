from glossweaver.data.base import CanonicalRecord
from glossweaver.data.preprocess import (
    ambiguous_gloss_stats,
    assert_no_cross_split_leakage,
    clean_records,
    deterministic_split,
)


def raw(index: int, gloss: str | None = None, target: str | None = None) -> CanonicalRecord:
    return CanonicalRecord(
        id=f"raw:{index}",
        dataset="aslg_pc12",
        gloss=gloss or f"GLOSS {index}",
        target=target or f"English sentence {index}.",
        split="unsplit",
        source_partition="huggingface-train",
    )


def test_deterministic_80_10_10_split_has_zero_pair_overlap():
    cleaned, _ = clean_records(raw(index) for index in range(100))
    first = deterministic_split(cleaned, seed=42)
    second = deterministic_split(cleaned, seed=42)
    assert [(item.id, item.split) for item in first] == [(item.id, item.split) for item in second]
    assert sum(item.split == "train" for item in first) == 80
    assert sum(item.split == "validation" for item in first) == 10
    assert sum(item.split == "test" for item in first) == 10
    assert_no_cross_split_leakage(first)


def test_same_gloss_different_targets_is_retained_and_reported():
    cleaned, _ = clean_records([
        raw(1, "BOOK READ", "I read a book."),
        raw(2, "BOOK READ", "She reads the book."),
    ])
    assert len(cleaned) == 2
    report = ambiguous_gloss_stats(cleaned)
    assert report["glosses_with_multiple_targets"] == 1
    assert report["pairs_under_ambiguous_glosses"] == 2
