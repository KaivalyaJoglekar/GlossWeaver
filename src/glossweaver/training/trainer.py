from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epochs(
    model: Any,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    *,
    device: str,
    epochs: int,
    output_dir: str | Path,
    max_steps: int | None = None,
) -> dict[str, list[float]]:
    output = Path(output_dir)
    history = {"train_loss": [], "validation_loss": []}
    best_loss = float("inf")
    global_step = 0
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        epoch_steps = 0
        for batch in tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            result = model(**move_batch(batch, device))
            result.loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_total += result.loss.detach().float().item()
            global_step += 1
            epoch_steps += 1
            if max_steps is not None and global_step >= max_steps:
                break
        history["train_loss"].append(train_total / max(1, epoch_steps))

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                result = model(**move_batch(batch, device))
                validation_total += result.loss.detach().float().item()
        validation_loss = validation_total / max(1, len(validation_loader))
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output)
        if max_steps is not None and global_step >= max_steps:
            break
    return history
