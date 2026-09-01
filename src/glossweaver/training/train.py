from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from glossweaver.config import load_config
from glossweaver.data.dataset import GlossTextDataset, Seq2SeqCollator
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.training.trainer import train_epochs
from glossweaver.utils import read_jsonl, resolve_device, set_seed, write_json


def _records(
    path: str,
    synthetic_path: str | None = None,
    *,
    ids_path: str | None = None,
    max_examples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    records = list(read_jsonl(path))
    if ids_path:
        with Path(ids_path).open(encoding="utf-8") as handle:
            identifiers = json.load(handle)
        by_id = {record["id"]: record for record in records}
        missing = [identifier for identifier in identifiers if identifier not in by_id]
        if missing:
            raise ValueError(f"ID manifest references {len(missing)} absent records")
        records = [by_id[identifier] for identifier in identifiers]
    if max_examples is not None and len(records) > max_examples:
        records = sorted(records, key=lambda record: record["id"])
        random.Random(seed).shuffle(records)
        records = records[:max_examples]
    if synthetic_path:
        synthetic = list(read_jsonl(synthetic_path))
        if any(item.get("split") != "train" for item in synthetic):
            raise ValueError("Synthetic augmentation contains non-training records")
        records.extend(synthetic)
    return records


def run_training(config: dict) -> dict[str, object]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    model_name = config["model"]
    grammar = config.get("grammar", {})
    grammar_enabled = bool(grammar.get("enabled", False))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if grammar_enabled:
        model = GrammarAwareT5.from_pretrained(
            model_name,
            num_grammar_labels=int(grammar.get("num_labels", 6)),
            lambda_grammar=float(grammar.get("lambda", 0.25)),
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = resolve_device(config.get("device", "auto"))
    model.to(device)

    data = config["data"]
    synthetic_path = data.get("synthetic_train") if data.get("use_synthetic") else None
    train_records = _records(
        data["train"], synthetic_path,
        ids_path=data.get("train_ids"),
        max_examples=data.get("max_train_examples"),
        seed=seed,
    )
    validation_records = _records(
        data["validation"],
        ids_path=data.get("validation_ids"),
        max_examples=data.get("max_validation_examples"),
        seed=seed + 1,
    )
    dataset_kwargs = {
        "prefix": config.get("prefix", "reconstruct gloss: "),
        "max_source_length": int(config.get("max_source_length", 128)),
        "max_target_length": int(config.get("max_target_length", 128)),
    }
    train_data = GlossTextDataset(train_records, tokenizer, **dataset_kwargs)
    validation_data = GlossTextDataset(validation_records, tokenizer, **dataset_kwargs)
    collator = Seq2SeqCollator(tokenizer, include_grammar=grammar_enabled)
    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(train_loader) * int(config.get("epochs", 3)))
    if config.get("max_steps") is not None:
        total_steps = min(total_steps, int(config["max_steps"]))
    warmup_steps = round(total_steps * float(config.get("warmup_ratio", 0.05)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    output = Path(config.get("output_dir", f"checkpoints/{config['experiment_name']}"))
    history = train_epochs(
        model, train_loader, validation_loader, optimizer, scheduler,
        device=device,
        epochs=int(config.get("epochs", 3)),
        output_dir=output,
        max_steps=config.get("max_steps"),
    )
    tokenizer.save_pretrained(output)
    metadata: dict[str, object] = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": device,
        "model": model_name,
        "grammar_enabled": grammar_enabled,
        "synthetic_enabled": bool(synthetic_path),
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "history": history,
    }
    write_json(output / "glossweaver_training.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GlossWeaver experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    print(json.dumps(run_training(config), indent=2))


if __name__ == "__main__":
    main()
