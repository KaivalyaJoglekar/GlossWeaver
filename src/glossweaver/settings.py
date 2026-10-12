from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os


ROOT = Path(os.environ.get("GLOSSWEAVER_ROOT", Path(__file__).resolve().parents[2])).resolve()


@dataclass(frozen=True)
class GenerationConfig:
    num_beams: int = 4
    do_sample: bool = False
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if not 1 <= self.num_beams <= 10 or self.max_new_tokens < 1:
            raise ValueError("Beam width must be 1–10 and output length positive")


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    model_name: str
    initial_checkpoint: Path | None = None
    train_file: Path = ROOT / "datasets/processed/aslg_pc12/train.jsonl"
    validation_file: Path = ROOT / "datasets/processed/aslg_pc12/validation.jsonl"
    train_ids_file: Path | None = None
    validation_ids_file: Path | None = None
    synthetic_train_file: Path | None = None
    synthetic_validation_file: Path | None = None
    train_size: int | None = None
    validation_size: int | None = None
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_source_length: int = 96
    max_target_length: int = 96
    seed: int = 42
    dataset_seed: int = 42
    grammar_enabled: bool = False
    grammar_lambda: float = 0.1
    num_grammar_labels: int = 7
    output_dir: Path | None = None
    max_steps: int | None = None
    optimizer_steps_per_epoch: int | None = None
    early_stopping_patience: int = 2
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_source_length", "max_target_length", "early_stopping_patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.grammar_lambda < 0:
            raise ValueError("Learning rate must be positive and grammar lambda non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        for name in ("train_size", "validation_size", "max_steps", "optimizer_steps_per_epoch"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir or ROOT / "checkpoints" / self.experiment_name

    @property
    def history_csv(self) -> Path:
        return self.checkpoint_dir / "history.csv"


RESEARCH_TRAIN_IDS = ROOT / "datasets/processed/aslg_pc12/research_train_ids.json"
RESEARCH_VALIDATION_IDS = ROOT / "datasets/processed/aslg_pc12/research_validation_ids.json"
SYNTHETIC_TRAIN = ROOT / "datasets/processed/reconstruction/train.csv"
SYNTHETIC_VALIDATION = ROOT / "datasets/processed/reconstruction/validation.csv"

SMOKE_CONFIG = TrainConfig(
    experiment_name="smoke",
    model_name="google-t5/t5-small",
    train_size=64,
    validation_size=32,
    epochs=1,
    batch_size=4,
    gradient_accumulation_steps=1,
    max_steps=8,
)

OVERFIT_T5_CONFIG = replace(
    SMOKE_CONFIG,
    experiment_name="overfit_t5_64",
    validation_file=ROOT / "datasets/processed/aslg_pc12/train.jsonl",
    validation_size=64,
    epochs=30,
    batch_size=8,
    learning_rate=1e-3,
    max_steps=300,
    early_stopping_patience=30,
)

OVERFIT_FLAN_CONFIG = replace(
    OVERFIT_T5_CONFIG,
    experiment_name="overfit_flan_64",
    model_name="google/flan-t5-small",
    epochs=60,
    max_steps=480,
)

BASELINE_CONFIG = TrainConfig(
    experiment_name="e1_t5",
    model_name="google-t5/t5-small",
    train_ids_file=RESEARCH_TRAIN_IDS,
    validation_ids_file=RESEARCH_VALIDATION_IDS,
    train_size=5_000,
    validation_size=500,
    epochs=3,
    batch_size=16,
    gradient_accumulation_steps=1,
)

AUGMENTED_CONFIG = replace(
    BASELINE_CONFIG,
    experiment_name="e2_flan",
    model_name="google/flan-t5-small",
    synthetic_train_file=SYNTHETIC_TRAIN,
    synthetic_validation_file=SYNTHETIC_VALIDATION,
)

POLISH_CONFIG = replace(
    AUGMENTED_CONFIG,
    epochs=2,
    learning_rate=1e-4,
    initial_checkpoint=ROOT / "checkpoints/e2_flan",
)

FINAL_POLISH_CONFIG = replace(POLISH_CONFIG, epochs=1, learning_rate=8e-5)

GRAMMAR_CONFIG = replace(
    AUGMENTED_CONFIG,
    experiment_name="e3_grammar_flan",
    initial_checkpoint=ROOT / "checkpoints/e2_flan",
    grammar_enabled=True,
    grammar_lambda=0.1,
    epochs=2,
    learning_rate=1e-4,
)

FULL_CONFIG = replace(
    GRAMMAR_CONFIG,
    experiment_name="e4_glossweaver",
    initial_checkpoint=ROOT / "checkpoints/e3_grammar_flan",
    train_size=20_000,
    validation_size=2_000,
)

PRESETS = {
    "smoke": SMOKE_CONFIG,
    "overfit-t5": OVERFIT_T5_CONFIG,
    "overfit-flan": OVERFIT_FLAN_CONFIG,
    "e1": BASELINE_CONFIG,
    "e2": AUGMENTED_CONFIG,
    "e2-polish": POLISH_CONFIG,
    "e2-final-polish": FINAL_POLISH_CONFIG,
    "e3": GRAMMAR_CONFIG,
    "e4": FULL_CONFIG,
}

# New studies use repaired group-disjoint datasets and a shared validation mix.
# Historical E1–E4 presets remain unchanged for provenance.
MATCHED_BASE = replace(
    BASELINE_CONFIG, model_name="google/flan-t5-small",
    train_file=ROOT / "datasets/processed/study_v2/aslg_grouped/train.jsonl",
    validation_file=ROOT / "datasets/processed/study_v2/aslg_grouped/validation.jsonl",
    train_ids_file=None, validation_ids_file=None,
    synthetic_validation_file=ROOT / "datasets/processed/study_v2/reconstruction_grouped/validation.csv",
    epochs=4, max_steps=1000, optimizer_steps_per_epoch=250, early_stopping_patience=4,
)
for name, augmented, grammar in (
    ("study-baseline", False, False), ("study-augmentation", True, False),
    ("study-grammar", False, True), ("study-full", True, True),
):
    PRESETS[name] = replace(
        MATCHED_BASE, experiment_name=name,
        synthetic_train_file=ROOT / "datasets/processed/study_v2/reconstruction_grouped/train.csv" if augmented else None,
        grammar_enabled=grammar,
    )


PRESETS['demo-adaptation'] = replace(
    MATCHED_BASE, experiment_name='reconstruction-flan',
    initial_checkpoint=ROOT / 'checkpoints/study-augmentation',
    train_file=ROOT / 'datasets/processed/demo_adaptation/train.csv',
    validation_file=ROOT / 'datasets/processed/demo_adaptation/validation.csv',
    synthetic_train_file=None, synthetic_validation_file=None,
    train_size=None, validation_size=400, learning_rate=1e-4,
    epochs=3, max_steps=300, optimizer_steps_per_epoch=100, early_stopping_patience=3,
)


PRESETS['balanced-adaptation'] = replace(
    PRESETS['demo-adaptation'], experiment_name='balanced-flan',
    initial_checkpoint=ROOT / 'checkpoints/reconstruction-flan',
    train_file=ROOT / 'datasets/processed/balanced_adaptation/train.csv',
    validation_file=ROOT / 'datasets/processed/balanced_adaptation/validation.csv',
    validation_size=None, learning_rate=8e-5,
    epochs=4, max_steps=600, optimizer_steps_per_epoch=150, early_stopping_patience=4,
)


def get_preset(name: str) -> TrainConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}") from exc
