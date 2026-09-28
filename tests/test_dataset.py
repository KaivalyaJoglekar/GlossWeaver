from datasets import Dataset

from glossweaver.data.aslg_pc12 import (
    ASLGPC12SchemaError,
    iter_records,
    resolve_columns,
)
from glossweaver.data.dataset import GlossTextDataset, Seq2SeqCollator


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text=None, text_target=None, **kwargs):
        value = text if text is not None else text_target
        return {"input_ids": [len(str(value)), 1], "attention_mask": [1, 1]}


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


def test_grammar_enabled_dataset_labels_synthetic_records():
    records = [{
        "gloss": "GIRL GO SCHOOL MORNING",
        "target": "The girl goes to school in the morning.",
    }]
    dataset = GlossTextDataset(records, FakeTokenizer(), include_grammar_labels=True)
    batch = Seq2SeqCollator(FakeTokenizer(), include_grammar=True)([dataset[0]])
    assert tuple(batch["grammar_labels"].shape) == (1, 7)
