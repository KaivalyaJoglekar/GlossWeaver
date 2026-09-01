from glossweaver.data.base import CanonicalRecord
from glossweaver.data.preprocess import (
    assert_no_cross_split_leakage,
    clean_records,
    normalize_text,
    remove_cross_split_leakage,
)


def record(identifier: str, split: str, gloss: str, target: str) -> CanonicalRecord:
    return CanonicalRecord(identifier, "test", gloss, target, split, split)


def test_normalization_preserves_gloss_case_and_normalizes_space():
    assert normalize_text("  ME\tGO  STORE  ") == "ME GO STORE"


def test_empty_and_same_split_duplicates_are_removed():
    records, stats = clean_records([
        record("1", "train", "ME GO", "I go."),
        record("2", "train", "me-go", "I GO"),
        record("3", "train", "", "Empty source"),
    ])
    assert len(records) == 1
    assert stats["removed_normalized_duplicate"] == 1
    assert stats["removed_empty"] == 1


def test_cross_split_pair_is_kept_in_test_not_train():
    records = [
        record("train", "train", "BOY GO", "The boy went."),
        record("test", "test", "BOY GO", "The boy went."),
    ]
    deduplicated, stats = remove_cross_split_leakage(records)
    assert [item.id for item in deduplicated] == ["test"]
    assert stats["removed_from_train"] == 1
    assert_no_cross_split_leakage(deduplicated)
