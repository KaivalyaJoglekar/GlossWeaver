from datasets import Dataset

from glossweaver.data.aslg_pc12 import (
    ASLGPC12SchemaError,
    iter_records,
    resolve_columns,
)


def test_verified_live_column_orientation():
    assert resolve_columns(["gloss", "text"]) == ("gloss", "text")


def test_local_parquet_maps_gloss_to_english(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Dataset.from_dict({
        "gloss": ["ME GO STORE"],
        "text": ["I went to the store."],
    }).to_parquet(data_dir / "train-00000-of-00001.parquet")
    monkeypatch.setattr("glossweaver.data.aslg_pc12.EXPECTED_ROWS", 1)
    first = next(iter_records(tmp_path))
    assert first.dataset == "aslg_pc12"
    assert first.gloss == "ME GO STORE"
    assert first.target == "I went to the store."


def test_unknown_schema_is_rejected():
    try:
        resolve_columns(["sentence_a", "sentence_b"])
    except ASLGPC12SchemaError:
        pass
    else:
        raise AssertionError("Ambiguous orientation must not be guessed")

