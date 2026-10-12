from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from glossweaver.data.dataset import GlossTextDataset, Seq2SeqCollator
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.settings import PRESETS, ROOT, TrainConfig, get_preset
from glossweaver.training.trainer import train_epochs
from glossweaver.utils import read_jsonl, resolve_device, set_seed, write_json, sha256_file, code_provenance
from glossweaver.data.split_integrity import assert_source_isolation
from glossweaver.model_registry import resolve_checkpoint


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        records = pd.read_csv(path).fillna("").to_dict(orient="records")
        for row in records:
            if isinstance(row.get("grammar_labels"), str):
                row["grammar_labels"] = json.loads(row["grammar_labels"]) if row["grammar_labels"] else None
        return records
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


def run_training(config: TrainConfig, resume: Path | None = None) -> dict[str, object]:
    required = [config.train_file, config.validation_file, config.synthetic_train_file, config.synthetic_validation_file]
    absent = [str(path) for path in required if path is not None and not path.is_file()]
    if absent:
        raise FileNotFoundError("Training data missing: " + ", ".join(absent) + ". Run scripts/prepare_research_data.py for study-v2 corpora.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    # Each continuation gets a new artifact directory, never overwriting its parent.
    config = replace(config, output_dir=config.checkpoint_dir / run_id)
    if resume:
        manifest = json.loads((resume / "run_manifest.json").read_text())
        previous = manifest["config"]
        for field in ("model_name", "grammar_enabled", "grammar_lambda", "max_source_length", "max_target_length", "seed", "dataset_seed", "batch_size", "gradient_accumulation_steps"):
            if previous.get(field) != getattr(config, field):
                raise ValueError(f"Resume configuration differs in {field}; use an explicit new experiment instead")
        for field in ("train_file", "validation_file", "synthetic_train_file", "synthetic_validation_file", "train_ids_file", "validation_ids_file"):
            current = getattr(config, field)
            if previous.get(field) != (str(current) if current is not None else None):
                raise ValueError(f"Resume dataset path differs in {field}; use a new experiment")
        for path, digest in manifest["data_sha256"].items():
            if not Path(path).is_file() or sha256_file(path) != digest:
                raise ValueError("Resume dataset content changed; use a new experiment")
    set_seed(config.seed)
    cache_dir = ROOT / ".hf-cache/hub"
    initialization_source = resume or (resolve_checkpoint(config.initial_checkpoint) if config.initial_checkpoint else config.model_name)
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
        per_source_limit=config.train_size, seed=config.dataset_seed,
    )
    validation_records = _mixed_records(
        config.validation_file, config.synthetic_validation_file,
        ids_path=config.validation_ids_file, per_source_limit=config.validation_size,
        seed=config.dataset_seed + 1,
    )
    if config.experiment_name.startswith("overfit_"):
        validation_records = list(train_records)
    else:
        assert_source_isolation(train_records, validation_records)
    if not train_records or not validation_records:
        raise ValueError("Training and validation must both contain records")
    model.config.glossweaver_grammar_aware = config.grammar_enabled
    model.config.glossweaver_max_source_length = config.max_source_length
    model.config.glossweaver_max_target_length = config.max_target_length
    inputs = [config.train_file, config.validation_file, config.synthetic_train_file,
              config.synthetic_validation_file, config.train_ids_file, config.validation_ids_file]
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.checkpoint_dir / "run_manifest.json", {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_revision": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None,
        "code_provenance": code_provenance(),
        "input_formatter": "shared_v1",
        "selection_criterion": "minimum validation generation loss",
        "validation_decoding": {"num_beams": 1, "max_new_tokens": 64, "do_sample": False},
        "resumed_from": str(resume) if resume else None,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "data_sha256": {str(path): sha256_file(path) for path in inputs if path},
        "train_ids": [r["id"] for r in train_records],
        "validation_ids": [r["id"] for r in validation_records],
        "initialization": str(initialization_source),
        "initialization_sha256": {
            p.name: sha256_file(p) for p in Path(initialization_source).glob("*")
            if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"}
        } if Path(initialization_source).is_dir() else {},
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "grammar_label_policy": __import__("os").environ.get("GLOSSWEAVER_GRAMMAR_PARSER", "lexical-v1"),
    })
    kwargs = {
        "max_source_length": config.max_source_length,
        "max_target_length": config.max_target_length,
        "include_grammar_labels": config.grammar_enabled,
    }
    train_data = GlossTextDataset(train_records, tokenizer, **kwargs)
    validation_data = GlossTextDataset(validation_records, tokenizer, **kwargs)
    collator = Seq2SeqCollator(tokenizer, include_grammar=config.grammar_enabled)
    # Matched studies validate at identical optimizer steps despite corpus-size differences.
    sampler = RandomSampler(train_data, replacement=True, num_samples=(
        config.optimizer_steps_per_epoch * config.gradient_accumulation_steps * config.batch_size
    )) if config.optimizer_steps_per_epoch else None
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collator)
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
    resume_state = torch.load(resume / "optimizer_state.pt", map_location="cpu", weights_only=False) if resume else None
    history = train_epochs(
        model, train_loader, validation_loader, optimizer, tokenizer, scheduler,
        device=device, epochs=config.epochs, output_dir=config.checkpoint_dir,
        history_csv=config.history_csv,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        early_stopping_patience=config.early_stopping_patience, max_steps=config.max_steps, resume_state=resume_state,
    )
    metadata = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "checkpoint_path": str(config.checkpoint_dir),
        "model_name": config.model_name,
        "initial_checkpoint": str(config.initial_checkpoint or "pretrained base model"),
        "seed": config.seed,
        "device": device,
        "grammar_enabled": config.grammar_enabled,
        "grammar_lambda": config.grammar_lambda,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "training_data_description": config.train_file.parent.name + (f" + {config.synthetic_train_file.parent.name}" if config.synthetic_train_file else ""),
        "best_validation_epoch": min(history, key=lambda row: float(row["validation_generation_loss"]))["epoch"],
    }
    pd.DataFrame([metadata]).to_csv(config.checkpoint_dir / "training_metadata.csv", index=False)
    write_json(config.checkpoint_dir.parent / "latest.json", {"run_directory": config.checkpoint_dir.name, "run_id": run_id})
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Python-configured GlossWeaver experiment")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="Trusted local best checkpoint with optimizer_state.pt")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = get_preset(args.preset)
    overrides: dict[str, object] = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.output_dir:
        overrides["output_dir"] = Path(args.output_dir)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    metadata = run_training(replace(config, **overrides), resume=args.resume)
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
