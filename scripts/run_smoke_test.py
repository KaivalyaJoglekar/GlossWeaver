#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from glossweaver.data.dataset import GlossTextDataset, Seq2SeqCollator
from glossweaver.evaluation.metrics import exact_match
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.training.trainer import move_batch


RECORDS = [
    {"id": "smoke-1", "gloss": "ME GO STORE YESTERDAY", "target": "I went to the store yesterday.", "grammar_labels": [1, 1, 0, 1, 1, 0]},
    {"id": "smoke-2", "gloss": "CHILD PLAY GARDEN", "target": "The child was playing in the garden.", "grammar_labels": [1, 1, 1, 0, 1, 0]},
]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    corpus = ["reconstruct gloss: " + row["gloss"] for row in RECORDS] + [row["target"] for row in RECORDS]
    for text in corpus:
        for token in text.replace(":", " :").replace(".", " .").split():
            words.setdefault(token, len(words))
    tokenizer = Tokenizer(WordLevel(words, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
    )


def main() -> None:
    tokenizer = tiny_tokenizer()
    config = T5Config(
        vocab_size=len(tokenizer), d_model=32, d_ff=64, num_layers=1,
        num_decoder_layers=1, num_heads=2, decoder_start_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    for model_class in (T5ForConditionalGeneration, GrammarAwareT5):
        model = model_class(config)
        dataset = GlossTextDataset(RECORDS, tokenizer, max_source_length=24, max_target_length=24)
        loader = DataLoader(dataset, batch_size=2, collate_fn=Seq2SeqCollator(tokenizer))
        batch = next(iter(loader))
        if model_class is T5ForConditionalGeneration:
            batch.pop("grammar_labels")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        output = model(**batch)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=4
            )
        assert generated.shape[0] == 2
        with tempfile.TemporaryDirectory(prefix="glossweaver-smoke-") as temp_dir:
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            reloaded = model_class.from_pretrained(temp_dir)
            reloaded.eval()
            with torch.no_grad():
                second = reloaded.generate(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=2
                )
            assert second.shape[0] == 2
    assert exact_match(["ok"], ["ok"]) == 1.0
    print("Smoke test passed: tokenization, baseline and grammar-aware forward/backward, optimizer, save/reload, generation, evaluation, and inference path.")


if __name__ == "__main__":
    main()

