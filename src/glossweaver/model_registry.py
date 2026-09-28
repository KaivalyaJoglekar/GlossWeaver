from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from glossweaver.settings import ROOT


@dataclass(frozen=True)
class ModelInfo:
    key: str
    display_name: str
    checkpoint_path: Path | None
    model_type: str
    input_formatter: str
    grammar_enabled: bool
    experiment_name: str
    training_data_description: str
    default: bool = False

    @property
    def available(self) -> bool:
        if self.model_type == "copy":
            return True
        return bool(self.checkpoint_path and checkpoint_has_weights(self.checkpoint_path))

    def debug_dict(self) -> dict[str, object]:
        modified = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            modified = datetime.fromtimestamp(
                self.checkpoint_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        return {
            "model_name": self.display_name,
            "experiment_name": self.experiment_name,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "e0-copy",
            "checkpoint_modified_time": modified,
            "training_data_description": self.training_data_description,
            "model_type": self.model_type,
            "grammar_enabled": self.grammar_enabled,
        }


def checkpoint_has_weights(path: Path) -> bool:
    return path.is_dir() and any(
        (path / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


MODELS: dict[str, ModelInfo] = {
    "e0_copy": ModelInfo(
        "e0_copy", "E0 – Copy Baseline", None, "copy", "none", False,
        "e0_copy", "No training; lexical copy heuristic.",
    ),
    "e1_t5": ModelInfo(
        "e1_t5", "E1 – T5-small", ROOT / "checkpoints/e1_t5", "t5",
        "shared_v1", False, "e1_t5", "Fixed ASLG-PC12 base split.",
    ),
    "e2_flan": ModelInfo(
        "e2_flan", "E2 – FLAN-T5 hybrid", ROOT / "checkpoints/e2_flan", "flan-t5",
        "shared_v1", False, "e2_flan",
        "Fixed ASLG-PC12 split plus clean synthetic telegraphic English.",
    ),
    "e3_grammar_flan": ModelInfo(
        "e3_grammar_flan", "E3 – Grammar-aware FLAN-T5", ROOT / "checkpoints/e3_grammar_flan",
        "grammar-flan-t5", "shared_v1", True, "e3_grammar_flan",
        "E2 hybrid data with auxiliary grammar supervision.",
    ),
    "e4_glossweaver": ModelInfo(
        "e4_glossweaver", "E4 – GlossWeaver", ROOT / "checkpoints/e4_glossweaver",
        "grammar-flan-t5", "shared_v1", True, "e4_glossweaver",
        "Best validated grammar-aware model and augmentation mix.", True,
    ),
    # Archived models remain directly inspectable, but are never silent fallbacks.
    "legacy_e1": ModelInfo(
        "legacy_e1", "Legacy E1 – broken-domain T5", ROOT / "checkpoints/t5_baseline", "t5",
        "legacy_reconstruct_gloss", False, "t5_baseline", "ASLG-PC12 only; legacy prompt.",
    ),
    "legacy_e4": ModelInfo(
        "legacy_e4", "Legacy E4 – broken-domain GA-T5", ROOT / "checkpoints/gat5_augmented",
        "grammar-t5", "legacy_reconstruct_gloss", True, "gat5_augmented",
        "ASLG-PC12 plus malformed in-domain corruption; legacy prompt.",
    ),
}


def get_model(key: str) -> ModelInfo:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model {key!r}; choose from {sorted(MODELS)}") from exc


def default_available_model() -> ModelInfo:
    candidates = [model for model in MODELS.values() if model.default and model.available]
    if candidates:
        return candidates[0]
    for key in ("e4_glossweaver", "e3_grammar_flan", "e2_flan", "e1_t5", "e0_copy"):
        if MODELS[key].available:
            return MODELS[key]
    raise RuntimeError("No model is available")
