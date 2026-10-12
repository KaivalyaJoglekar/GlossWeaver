from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

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
        return bool(self.checkpoint_path and checkpoint_has_weights(resolve_checkpoint(self.checkpoint_path)))

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


def resolve_checkpoint(path: Path) -> Path:
    """A completed run pointer selects a versioned checkpoint; no cross-model fallback."""
    marker = path / "latest.json"
    if not marker.exists() and not checkpoint_has_weights(path):
        candidates = sorted(path.glob("seed-*/latest.json"))
        if candidates:
            return resolve_checkpoint(candidates[0].parent)
    if marker.exists():
        try:
            candidate = (path / json.loads(marker.read_text())["run_directory"]).resolve()
            if candidate.is_relative_to(path.resolve()) and checkpoint_has_weights(candidate):
                return candidate
        except (ValueError, KeyError, OSError):
            pass
    return path


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
        "Historical staged grammar-aware model and augmentation mix.", True,
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

for key, grammar in (("study-baseline", False), ("study-augmentation", False), ("study-grammar", True), ("study-full", True)):
    MODELS[key] = ModelInfo(key, key.replace("study-", "Study · ").title() + " · fine-tuned", ROOT / "checkpoints" / key,
                          "grammar-flan-t5" if grammar else "flan-t5", "shared_v1", grammar, key,
                          "Matched FLAN study configuration. Existing installed weights may predate study-v2; inspect saved run provenance.")



def cached_pretrained_checkpoint(repository: str) -> Path:
    """Expose local Hugging Face base weights without downloading or relabeling runs."""
    cache = ROOT / ".hf-cache/hub" / ("models--" + repository.replace("/", "--"))
    reference = cache / "refs/main"
    if reference.is_file():
        revision = reference.read_text().strip()
        if revision and all(char in "0123456789abcdef" for char in revision):
            return cache / "snapshots" / revision
    return cache / "snapshots/unavailable"


for key, repository, name, formatter in (
    ("pretrained_flan_t5", "google/flan-t5-small", "FLAN-T5-small · pretrained", "flan_grammar_v1"),
    ("pretrained_t5", "google-t5/t5-small", "T5-small · pretrained", "shared_v1"),
):
    MODELS[key] = ModelInfo(
        key, name, cached_pretrained_checkpoint(repository), "pretrained", formatter, False, key,
        "Original pretrained weights; not fine-tuned for GlossWeaver. Exploratory results only.",
    )


MODELS['reconstruction-flan'] = ModelInfo(
    'reconstruction-flan', 'FLAN-T5-small · reconstruction', ROOT / 'checkpoints/reconstruction-flan',
    'flan-t5', 'shared_v1', False, 'reconstruction-flan',
    'Fine-tuned reconstruction model with separate synthetic time/quantity adaptation; not a matched ablation.',
)


MODELS['balanced-flan'] = ModelInfo(
    'balanced-flan', 'FLAN-T5-small · balanced', ROOT / 'checkpoints/balanced-flan',
    'flan-t5', 'shared_v1', False, 'balanced-flan',
    'Deployment continuation on broader reconstruction data and untimed present-tense examples; not a matched ablation.',
)


def get_model(key: str) -> ModelInfo:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model {key!r}; choose from {sorted(MODELS)}") from exc


def default_available_model() -> ModelInfo:
    for key in ("balanced-flan", "reconstruction-flan", "study-full", "study-augmentation", "study-grammar", "study-baseline"):
        if MODELS[key].available:
            return MODELS[key]
    candidates = [model for model in MODELS.values() if model.default and model.available]
    if candidates:
        return candidates[0]
    for key in ("e4_glossweaver", "e3_grammar_flan", "e2_flan", "e1_t5", "pretrained_flan_t5", "pretrained_t5", "e0_copy"):
        if MODELS[key].available:
            return MODELS[key]
    raise RuntimeError("No model is available")
