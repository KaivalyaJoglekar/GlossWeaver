# GlossWeaver

GlossWeaver reconstructs fluent English from textual ASL-gloss-style input.

```text
ME GO STORE YESTERDAY -> I went to the store yesterday.
```

It is text-only: video recognition, pose estimation, hand tracking, sign
detection, and speech synthesis are out of scope.

## Research design

The research question is whether grammar-aware auxiliary supervision and
controlled telegraphic augmentation improve Gloss-to-Text reconstruction over
a matched `google-t5/t5-small` baseline.

| ID | System | Grammar loss | Augmentation |
|---|---|---:|---:|
| E0 | Copy/normalization | No | No |
| E1 | T5-small | No | No |
| E2 | T5-small | No | Yes |
| E3 | Grammar-Aware T5 | Yes | No |
| E4 | GlossWeaver | Yes | Yes |

The grammar-aware model shares a T5 encoder between the decoder and a six-label
head: ARTICLE, PREPOSITION, AUXILIARY, PRONOUN, TENSE_ASPECT, and
AGREEMENT_INFLECTION. Target-derived weak labels supervise training only;
inference requires the gloss alone.

## Primary dataset

The active dataset is ASLG-PC12 from the pinned public Hugging Face revision
`cb7cd272db8fcd4004ee04ddf50e194c15ea24d6`. It contains 87,710 raw rows in one
split with verified `gloss` and English `text` columns.

ASLG-PC12 is useful for controlled large-scale experimentation, but its glosses
are rule-generated. It is not a fully natural human-produced ASL corpus or a
real-world ASL benchmark. LibriSpeech-Gloss is related work only because its
official gloss files were not publicly accessible during implementation.
See [DATASETS.md](DATASETS.md) for exact provenance, checksum, license, cleaning
counts, and splits.

## Setup

The existing environment is ready:

```bash
cd /Users/kaivalyajoglekar/Desktop/Projects/GlossWeaver
source .venv/bin/activate
```

To recreate it:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m spacy download en_core_web_sm
```

The supplied training configs require Apple MPS so a research run cannot
silently fall back to CPU. Verify Metal access from a normal macOS Terminal:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

The result must be `True`. Codex's sandbox may report `False` even when the
same environment can use MPS in Terminal. Every training, evaluation, and
inference command also accepts `--device mps` (or `auto`, `cpu`, `cuda`).
No CUDA-only mixed-precision setting is assumed on MPS.

## Acquire and inspect ASLG-PC12

```bash
python scripts/download_data.py aslg-pc12
python scripts/download_data.py status
```

The pinned snapshot is stored in `datasets/raw/aslg_pc12/` rather than being
left only in a global cache.

## Preprocess

```bash
python scripts/prepare_data.py
```

This cleans all pairs before splitting, assigns stable IDs, creates a seeded
80/10/10 split, verifies zero normalized-pair overlap, reports repeated glosses
with different targets, prints ten orientation samples, and saves fixed
20k/2.5k/2.5k research ID manifests.

Actual retained counts are 64,813 train, 8,101 validation, and 8,103 test from
81,017 unique pairs.

## Verify the implementation

```bash
pytest -q
python scripts/run_smoke_test.py
```

The first command runs schema, orientation, duplicate, leakage, augmentation,
model, and metric tests. The second is an offline tiny-model architecture test.

## Real-data E1 smoke test

`config/smoke.yaml` selects 512 training and 64 validation examples and stops
after eight optimizer steps:

```bash
python -m glossweaver.training.train --config config/smoke.yaml --device mps

python -m glossweaver.evaluation.evaluate \
  --checkpoint checkpoints/smoke \
  --split test \
  --max-examples 64 \
  --output-name e1_smoke \
  --device mps

python -m glossweaver.inference \
  --checkpoint checkpoints/smoke \
  --text 'X-I WANT BOOK' \
  --device mps
```

## E0

Evaluate E0 on the fixed research test subset:

```bash
python -m glossweaver.evaluation.evaluate \
  --checkpoint e0-copy \
  --split test \
  --ids datasets/processed/aslg_pc12/research_test_ids.json \
  --output-name e0
```

## Development experiment

After smoke validation, run the 5k/500 E1 development experiment:

```bash
python -m glossweaver.training.train --config config/t5_development.yaml --device mps
```

This is a pipeline check, not a final reported result.

## Fixed research experiments

Generate conservative 25% training-only augmentation:

```bash
python scripts/generate_synthetic.py --ratio 0.25 --seed 42
```

Then run the matched 20k/2.5k base subsets:

```bash
python -m glossweaver.training.train --config config/t5_baseline.yaml --device mps
python -m glossweaver.training.train --config config/t5_augmented.yaml --device mps
python -m glossweaver.training.train --config config/gat5.yaml --device mps
python -m glossweaver.training.train --config config/gat5_augmented.yaml --device mps
```

Do not jump directly to E4. Evaluate and inspect E1 before proceeding through
E2, E3, and E4.

## Evaluation and inference

```bash
python -m glossweaver.evaluation.evaluate \
  --checkpoint checkpoints/gat5_augmented \
  --split test \
  --ids datasets/processed/aslg_pc12/research_test_ids.json \
  --device mps

python -m glossweaver.inference \
  --checkpoint checkpoints/gat5_augmented \
  --text 'X-I WANT BOOK' \
  --device mps
```

Evaluation includes SacreBLEU, chrF, ROUGE-L, METEOR, BERTScore F1, exact
match, and heuristic grammar diagnostics. Predictions and metrics are saved
under `results/`.

## Notebooks

The notebooks are optional research-analysis interfaces; they are not used to
train models.

- `notebooks/01_data_analysis.ipynb` reruns EDA over the processed ASLG-PC12
  data and writes dataset plots/statistics to `results/figures/`.
- `notebooks/02_error_analysis.ipynb` opens the manual annotation template for
  categorizing errors after E1 and E4 prediction CSVs exist.

The Python scripts and YAML configs remain the authoritative reproducible
pipeline. Start Jupyter with `jupyter lab` only when you want interactive data
or error analysis.

## Optional external validation

ASLLRP/NCSLGR may later provide a human-annotated external generalization test
if legally accessible manual glosses can be cleanly aligned with English text.
It must never be mixed into ASLG-PC12 training.

## Status and limitations

- ASLG-PC12 acquisition, cleaning, splitting, manifests, and leakage checks are
  implemented and reproducible.
- E0 is complete on the fixed 2,500-example research test manifest. The E1
  eight-step smoke and 5K/500 development runs also complete end to end; the
  latter is a development check, not a final research result.
- Model results must not be reported until their prediction and metric files
  exist.
- Rule-generated glosses, weak grammar labels, automatic diagnostics, domain
  limitations, and hallucination risk must be stated in the paper.
- A frontend is intentionally deferred until E0–E4 and evaluation are complete.

Final fixed-subset E1–E4 results remain pending experiment execution. See
`results/REPORT.md` for completed, traceable non-final runs.
