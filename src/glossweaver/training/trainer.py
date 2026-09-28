from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glossweaver.evaluation.metrics import compute_metrics


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _losses(result: Any) -> tuple[float, float, float]:
    generation = getattr(result, "generation_loss", None)
    grammar = getattr(result, "grammar_loss", None)
    return (
        float((generation if generation is not None else result.loss).detach().float()),
        float(grammar.detach().float()) if grammar is not None else 0.0,
        float(result.loss.detach().float()),
    )


def _validation_predictions(
    model: Any, loader: DataLoader, tokenizer: Any, device: str
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            generated = model.generate(
                input_ids=moved["input_ids"], attention_mask=moved["attention_mask"],
                num_beams=1, do_sample=False, max_new_tokens=64,
            )
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            predictions.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))
            references.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    return predictions, references


def train_epochs(
    model: Any,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    scheduler: Any | None = None,
    *,
    device: str,
    epochs: int,
    output_dir: str | Path,
    history_csv: str | Path,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 2,
    max_steps: int | None = None,
) -> list[dict[str, float | int]]:
    output = Path(output_dir)
    history_path = Path(history_csv)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    stale_epochs = global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_gen = train_grammar = train_total = 0.0
        epoch_batches = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"train {epoch}/{epochs}"), 1):
            result = model(**move_batch(batch, device))
            (result.loss / gradient_accumulation_steps).backward()
            generation_loss, grammar_loss, total_loss = _losses(result)
            train_gen += generation_loss
            train_grammar += grammar_loss
            train_total += total_loss
            epoch_batches += 1
            if batch_index % gradient_accumulation_steps == 0 or batch_index == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        model.eval()
        val_gen = val_grammar = val_total = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                result = model(**move_batch(batch, device))
                generation_loss, grammar_loss, total_loss = _losses(result)
                val_gen += generation_loss
                val_grammar += grammar_loss
                val_total += total_loss
        val_batches = max(1, len(validation_loader))
        predictions, references = _validation_predictions(model, validation_loader, tokenizer, device)
        metrics = compute_metrics(predictions, references, include_bertscore=False)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_generation_loss": train_gen / max(1, epoch_batches),
            "train_grammar_loss": train_grammar / max(1, epoch_batches),
            "train_total_loss": train_total / max(1, epoch_batches),
            "validation_generation_loss": val_gen / val_batches,
            "validation_grammar_loss": val_grammar / val_batches,
            "validation_total_loss": val_total / val_batches,
            "validation_bleu": float(metrics["sacrebleu"]),
            "validation_rouge_l": float(metrics["rouge_l"]),
            "validation_meteor": float(metrics["meteor"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        current = float(row["validation_generation_loss"])
        if current < best_loss:
            best_loss = current
            stale_epochs = 0
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
        else:
            stale_epochs += 1
        if stale_epochs >= early_stopping_patience:
            break
        if max_steps is not None and global_step >= max_steps:
            break
    return history
