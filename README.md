# GlossWeaver

GlossWeaver reconstructs fluent English from textual ASL-gloss-style input.

```text
ME GO STORE YESTERDAY -> I went to the store yesterday.
```

The project is text-only. Video recognition, pose estimation, hand tracking,
sign detection, and speech synthesis are outside its scope.

## Current result

The recovered E4 model passes the isolated nine-example challenge set and is
the default served checkpoint. Its three-reference challenge metrics are:

- SacreBLEU: 90.92
- ROUGE-L: 0.956
- METEOR: 0.953
- BERTScore F1: 0.990
- content-word recall: 1.000

See [`results/REPORT.md`](results/REPORT.md) for the root-cause analysis,
dataset audit, E0–E4 comparison, limitations, and exact evaluation scope.
For a file-by-file explanation of the complete system in plain and technical
language, see [`PROJECT_WALKTHROUGH.md`](PROJECT_WALKTHROUGH.md).

## Reproducible Python configuration

The active pipeline does not use YAML. Experiment presets are typed Python
dataclasses in `src/glossweaver/settings.py`:

| Preset | System | Initialization | Data | Grammar loss |
|---|---|---|---|---:|
| `e1` | T5-small baseline | `google-t5/t5-small` | ASLG-PC12 | No |
| `e2` | FLAN hybrid | `google/flan-t5-small` | ASLG + clean reconstruction | No |
| `e3` | grammar-aware FLAN | passing E2 | same hybrid mix | Yes |
| `e4` | GlossWeaver | E3 | scaled hybrid mix | Yes |

All training, evaluation, API, and UI inference paths share the formatter in
`src/glossweaver/text_format.py` and registry in
`src/glossweaver/model_registry.py`. A requested missing checkpoint raises an
explicit error; it never silently falls back to another model.

## Setup and verification

```bash
cd /Users/kaivalyajoglekar/Desktop/Projects/GlossWeaver
source .venv/bin/activate
pytest -q
```

Apple MPS is supported:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Data audit and construction

```bash
python -m glossweaver.data.audit_dataset
python -m glossweaver.data.build_reconstruction_data \
  --maximum 12000 --controlled-count 8000 \
  --conditional-count 1000 --schedule-count 800
python -m glossweaver.data.audit_grammar_labels
```

ASLG-PC12 orientation is verified as `gloss -> English`. The deterministic
seed-42 split contains 64,813 train, 8,101 validation, and 8,103 test pairs.
The reconstruction builder asserts that no exact challenge input or reference
enters training data.

## Pipeline gates

Run the two mandatory 64-pair overfit gates before larger training:

```bash
python scripts/run_overfit_check.py --preset overfit-t5 --device mps
python scripts/run_overfit_check.py --preset overfit-flan --device mps
```

Then run staged experiments in order:

```bash
python -m glossweaver.training.train --preset e1 --device mps
python -m glossweaver.training.train --preset e2 --device mps
python -m glossweaver.training.train --preset e3 --device mps
python -m glossweaver.training.train --preset e4 --device mps
```

Training saves checkpoints and CSV histories. The final research artifacts are
rebuilt with:

```bash
python scripts/build_research_artifacts.py
```

## Evaluation and inference

```bash
python -m glossweaver.evaluation.evaluate \
  --model e4_glossweaver \
  --dataset-file data/challenge/glossweaver_challenge.csv \
  --dataset-name challenge --split test \
  --device mps --output results/predictions/e4_challenge.csv

python -m glossweaver.inference \
  --model e4_glossweaver \
  --text 'TEACHER EXPLAIN STUDENT MATH' \
  --device mps
```

Evaluation writes CSV predictions, consolidated CSV metrics, diagnostic CSVs,
and PNG figures. Historical YAML configs and JSON result logs are retained
under `archive/` only and are not active pipeline inputs.

## Frontend and API

```bash
cd frontend && npm run build
cd .. && uvicorn app.api:app --reload
```

The UI calls the same API/registry as direct inference and displays the active
checkpoint metadata. Use `python scripts/verify_inference_parity.py` to verify
that direct and API outputs match exactly.

## Limitations

ASLG-PC12 is rule-generated parliamentary text rather than a natural ASL
benchmark. The reconstruction supplement is synthetic, grammar targets are
weak labels, and the nine-example challenge is deliberately small. Automatic
metrics and this controlled challenge do not replace evaluation by fluent ASL
users or professional interpreters.
