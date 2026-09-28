from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from glossweaver.data.dataset import GlossTextDataset, Seq2SeqCollator
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.settings import PRESETS, ROOT, TrainConfig, get_preset
from glossweaver.training.trainer import train_epochs
from glossweaver.utils import read_jsonl, resolve_device, set_seed


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        return pd.read_csv(path).fillna("").to_dict(orient="records")
    return list(read_jsonl(path))


def _records(
    path: Path, *, ids_path: Path | None = None,
    max_examples: int | None = None, seed: int = 42,
) -> list[dict[str, Any]]:
    records = _load_records(path)
    if ids_path:
        if ids_path.suffix == ".csv":
            identifiers = pd.read_csv(ids_path)["id"].astype(str).tolist()
        else:
            with ids_path.open(encoding="utf-8") as handle:
                identifiers = json.load(handle)
        by_id = {record["id"]: record for record in records}
        missing = [identifier for identifier in identifiers if identifier not in by_id]
        if missing:
            raise ValueError(f"ID manifest references {len(missing)} absent records")
        records = [by_id[identifier] for identifier in identifiers]
    if max_examples is not None and len(records) > max_examples:
        # Preserve coverage from every documented synthetic origin instead of
        # allowing a large public subcorpus to drown out controlled diagnostics.
        origins = sorted({str(record.get("corpus_origin", "")) for record in records if record.get("corpus_origin")})
        if len(origins) > 1:
            rng = random.Random(seed)
            groups = {origin: [] for origin in origins}
            for record in sorted(records, key=lambda item: str(item["id"])):
                groups[str(record["corpus_origin"])].append(record)
            for group in groups.values():
                rng.shuffle(group)
            selected: list[dict[str, Any]] = []
            while len(selected) < max_examples and any(groups.values()):
                for origin in origins:
                    if groups[origin] and len(selected) < max_examples:
                        selected.append(groups[origin].pop())
            records = selected
        else:
            records = sorted(records, key=lambda record: str(record["id"]))
            random.Random(seed).shuffle(records)
            records = records[:max_examples]
    return records


def _mixed_records(
    base_path: Path, synthetic_path: Path | None, *, ids_path: Path | None,
    per_source_limit: int | None, seed: int,
) -> list[dict[str, Any]]:
    base = _records(base_path, ids_path=ids_path, max_examples=per_source_limit, seed=seed)
    for record in base:
        record.setdefault("source_type", "aslg_pc12")
    if synthetic_path is None:
        return base
    return base + _records(synthetic_path, max_examples=per_source_limit, seed=seed + 17)


def run_training(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    cache_dir = ROOT / ".hf-cache/hub"
    initialization_source = config.initial_checkpoint or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(initialization_source, cache_dir=cache_dir)
    if config.grammar_enabled:
        model = GrammarAwareT5.from_pretrained(
            initialization_source, cache_dir=cache_dir,
            num_grammar_labels=config.num_grammar_labels,
            lambda_grammar=config.grammar_lambda,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(initialization_source, cache_dir=cache_dir)
    device = resolve_device(config.device)
    model.to(device)
    train_records = _mixed_records(
        config.train_file, config.synthetic_train_file, ids_path=config.train_ids_file,
        per_source_limit=config.train_size, seed=config.seed,
    )
    validation_records = _mixed_records(
        config.validation_file, config.synthetic_validation_file,
        ids_path=config.validation_ids_file, per_source_limit=config.validation_size,
        seed=config.seed + 1,
    )
    if config.experiment_name.startswith("overfit_"):
        validation_records = list(train_records)
    kwargs = {
        "max_source_length": config.max_source_length,
        "max_target_length": config.max_target_length,
        "include_grammar_labels": config.grammar_enabled,
    }
    train_data = GlossTextDataset(train_records, tokenizer, **kwargs)
    validation_data = GlossTextDataset(validation_records, tokenizer, **kwargs)
    collator = Seq2SeqCollator(tokenizer, include_grammar=config.grammar_enabled)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = max(1, (len(train_loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(total_steps * config.warmup_ratio),
        num_training_steps=max(1, total_steps),
    )
    history = train_epochs(
        model, train_loader, validation_loader, optimizer, tokenizer, scheduler,
        device=device, epochs=config.epochs, output_dir=config.checkpoint_dir,
        history_csv=config.history_csv,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        early_stopping_patience=config.early_stopping_patience, max_steps=config.max_steps,
    )
    metadata = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "initial_checkpoint": str(config.initial_checkpoint or "pretrained base model"),
        "seed": config.seed,
        "device": device,
        "grammar_enabled": config.grammar_enabled,
        "grammar_lambda": config.grammar_lambda,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "training_data_description": "ASLG-PC12" + (" + synthetic telegraphic English" if config.synthetic_train_file else ""),
        "best_validation_epoch": min(history, key=lambda row: float(row["validation_generation_loss"]))["epoch"],
    }
    pd.DataFrame([metadata]).to_csv(config.checkpoint_dir / "training_metadata.csv", index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Python-configured GlossWeaver experiment")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = get_preset(args.preset)
    overrides: dict[str, object] = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.output_dir:
        overrides["output_dir"] = Path(args.output_dir)
    metadata = run_training(replace(config, **overrides))
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
