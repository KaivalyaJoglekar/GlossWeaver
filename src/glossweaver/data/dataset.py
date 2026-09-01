from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from glossweaver.utils import read_jsonl


class GlossTextDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        prefix: str = "reconstruct gloss: ",
        max_source_length: int = 128,
        max_target_length: int = 128,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Any, **kwargs: Any):
        return cls(list(read_jsonl(path)), tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source = self.tokenizer(
            self.prefix + record["gloss"],
            max_length=self.max_source_length,
            truncation=True,
        )
        target = self.tokenizer(
            text_target=record["target"],
            max_length=self.max_target_length,
            truncation=True,
        )
        item = {
            "input_ids": torch.tensor(source["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target["input_ids"], dtype=torch.long),
        }
        labels = record.get("grammar_labels")
        if labels is not None:
            item["grammar_labels"] = torch.tensor(labels, dtype=torch.float)
        return item


class Seq2SeqCollator:
    def __init__(self, tokenizer: Any, include_grammar: bool = True) -> None:
        self.tokenizer = tokenizer
        self.include_grammar = include_grammar

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        grammar = [feature.get("grammar_labels") for feature in features]
        batch = {
            "input_ids": pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [feature["attention_mask"] for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=-100,
            ),
        }
        if self.include_grammar and all(item is not None for item in grammar):
            batch["grammar_labels"] = torch.stack(grammar)  # type: ignore[arg-type]
        return batch
