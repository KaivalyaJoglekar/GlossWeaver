# GlossWeaver — Entire Project Summary

**Snapshot date:** 12 October 2026  
**Package:** `glossweaver` 0.1.0  
**Scope:** Current local source code, dataset manifests, checkpoint availability, and saved research artifacts.

## 1. Project overview

GlossWeaver is a text-only English reconstruction system. It takes incomplete, telegraphic, or gloss-like text and attempts to produce a fluent English sentence while preserving meaning.

```text
ME GO STORE YESTERDAY → I went to the store yesterday.
BOY NOT WANT FOOD    → The boy does not want food.
```

These examples appear in saved development probes; they are not guarantees for arbitrary inputs.

The project combines a Python machine-learning package, reproducible data preparation and evaluation tools, a FastAPI backend, and a React workbench. Sign-language gloss reconstruction is a research application. ASL has its own complete grammar; the task is not to “correct” ASL. Video recognition, pose estimation, sign detection, and speech synthesis are outside the implemented scope.

The current default is **FLAN-T5-small · balanced** (`balanced-flan`). Its weights are present locally. Historical E1–E4 results remain in the repository, but their original checkpoints are unavailable.

## 2. Problem and research motivation

Telegraphic text often omits articles, prepositions, auxiliaries, pronouns, and inflections. English reconstruction may also require changes to word order and tense. Several completions can be valid when context is missing.

An earlier pipeline learned primarily from long parliamentary ASLG-PC12 examples, while the product accepted short everyday fragments. Historical audits found poor coverage of household and school vocabulary and malformed synthetic corruption. Strong scores on parliamentary data therefore did not reliably predict useful workbench output.

The current implementation addresses this through broader reconstruction data, source-grouped partitions, controlled synthetic examples, explicit model identity, shared input formatting, reproducible evaluation, and separate deployment adaptations.

The research question is whether synthetic training examples and an auxiliary grammar objective improve reconstruction over a common FLAN-T5 baseline. The new matched study is configured, but only its augmentation run currently has completed weights. The full factorial comparison remains unfinished.

## 3. Technology stack

| Layer | Technologies and role |
|---|---|
| Language/runtime | Python 3.10+; JavaScript with JSX |
| Neural models | PyTorch, Hugging Face Transformers, T5-small, FLAN-T5-small |
| Training | AdamW, linear warmup/decay, seeded sampling, gradient accumulation |
| Data | Hugging Face Datasets, pandas, NumPy, CSV and JSONL artifacts |
| Language processing | Lexical grammar heuristics; optional spaCy parser |
| Evaluation | SacreBLEU, chrF, ROUGE-L, METEOR, optional BERTScore |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | React 19, Vite 8, CSS, oxlint |
| Verification | pytest, local tiny-model smoke tests, GitHub Actions |
| Devices | CPU, Apple MPS, NVIDIA CUDA; automatic device selection supported |

`pyproject.toml` declares Python dependencies and the `api` and `dev` extras. `frontend/package.json` and its lockfile define frontend dependencies. The package exposes `glossweaver-train`, `glossweaver-evaluate`, and `glossweaver-infer` command-line entry points.

## 4. Architecture and execution flow

### Runtime reconstruction

1. The user enters text and selects an available model in the React workbench.
2. The frontend submits the text, checkpoint key, beam width, and grammar threshold to `POST /api/infer`.
3. FastAPI validates the request and checks the model registry.
4. `Reconstructor` resolves the completed checkpoint and formats the input.
5. The tokenizer checks the model's input limit; oversized inputs are rejected rather than silently truncated.
6. T5 generates an English reconstruction. Grammar-aware checkpoints additionally predict grammar features from encoder states.
7. The API returns text, model identity, elapsed time, notices, and any grammar predictions.
8. The frontend displays the result and records it in the current page session.

### Offline research pipeline

```text
Acquire source data
  → normalize and audit pairs
  → group sources into isolated splits
  → generate synthetic fragments and weak grammar labels
  → train a versioned checkpoint
  → select by validation generation loss
  → evaluate a fixed sample
  → save predictions, metrics, hashes, and manifests
  → expose available models and saved results in the workbench
```

Training, evaluation, CLI inference, and API inference share the new fine-tuned input formatter:

```text
Convert this gloss-like text into a fluent English sentence: <input>
```

Pretrained FLAN uses a separate exploratory prompt and lowercases all-capital input. Legacy checkpoints retain their historical formatter. Prompt differences matter when interpreting comparisons.

## 5. Repository structure

| Path | Responsibility |
|---|---|
| `src/glossweaver/settings.py` | Generation/training configuration, presets, artifact root |
| `src/glossweaver/model_registry.py` | Model identity, availability, checkpoint resolution, default selection |
| `src/glossweaver/inference.py` | Shared single/batch reconstruction and CLI |
| `src/glossweaver/text_format.py` | Whitespace normalization and fine-tuned input prompt |
| `src/glossweaver/data/` | Corpus adapters, preprocessing, corruption, grammar labels, audits, split integrity |
| `src/glossweaver/modeling/` | Copy baseline and grammar-aware T5 implementation |
| `src/glossweaver/training/` | Data loading, sampling, optimization, checkpointing, training manifests |
| `src/glossweaver/evaluation/` | Evaluation, reference metrics, grammar diagnostics, error analysis |
| `app/api.py` | HTTP API and production frontend serving |
| `frontend/src/` | Workbench, benchmarks, datasets, methodology, shared styles |
| `scripts/` | Acquisition, preparation, training gates, deployment comparisons, research reporting |
| `datasets/` | Raw downloads, processed splits, synthetic data, provenance |
| `data/challenge/` | Nine-example development challenge CSV |
| `checkpoints/` | Local versioned weights, tokenizer files, manifests, training state |
| `results/` | Metrics, evaluation snapshots, comparisons, reports, logs, UI previews |
| `tests/` | Python unit and integration tests |
| `notebooks/` | Exploratory data and error-analysis notebooks |
| `archive/` | Legacy configuration/results and invalid or superseded recovery artifacts |
| `.github/workflows/checks.yml` | Automated Python and frontend checks |

Large downloads, caches, checkpoint weights, and many generated datasets are Git-ignored. A cloned repository is not equivalent to this populated local workspace; artifact acquisition or training may be necessary.

## 6. Frontend and user experience

The app uses a minimal dark research-workbench design with charcoal surfaces, indigo actions, restrained borders, and responsive input/output panes.

Its four tabs are:

- **Workbench:** Custom text, example inputs, runnable model selection, reconstruction output, elapsed time, copy action, model refresh, and collapsible session history with JSON export.
- **Benchmarks:** Saved metrics organized by dataset and split, sample counts, checkpoint availability, and grammar diagnostics. Results are read from files, not recomputed by the page.
- **Datasets:** Local processed folders, manifest counts, data type, purpose, availability, and license information.
- **Methodology:** Explanation of models, experimental scope, and limitations.

Cmd/Ctrl + Enter submits reconstruction. Session history remains in memory across tab switches and clears on reload. Accessibility features include labeled native controls, visible focus, a skip link, tab semantics, live status, reduced-motion support, and mobile stacking. Wide dataset and benchmark tables scroll within their containers.

## 7. Backend API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Serve the production frontend when built; otherwise API status |
| GET | `/health` | Health, neural readiness, and default model |
| GET | `/api/models` | Registered models, availability, default, and descriptions |
| GET | `/api/datasets` | Dataset catalogue populated from local manifests |
| POST | `/api/infer` | Reconstruct text using the requested model |
| GET | `/api/metrics` | Saved benchmark rows; optional dataset/split filters |
| GET | `/api/metrics/{experiment}` | Saved results for one model key |
| GET | `/docs` | FastAPI interactive API documentation |

Inference accepts `gloss`, `checkpoint`, `num_beams` (default 4), and `threshold` (default 0.5). Input must be nonblank and at most 2,000 characters. Beam width is restricted to 1–10; the grammar threshold must be finite and between 0 and 1.

Neural input limits default to 96 tokens including the instruction, or the checkpoint's recorded limit. Generation defaults to deterministic four-beam decoding, a three-token repetition constraint, length penalty 1.0, and up to 64 new tokens, bounded by the checkpoint's target limit.

The backend retains one resident model and serializes loading/inference. A competing inference request receives HTTP 503 with retry guidance. Unknown models return 400, unavailable checkpoints return 404, and invalid input returns 422. Internal failures are logged and return a generic 500 response. Explicitly requested unavailable models do not silently switch to another model, and requests do not download pretrained weights.

This is a local research service. Authentication and a production multi-user deployment layer are not implemented.

## 8. Model families and local availability

Availability below was checked against the current registry and local weight files for this summary. It does not constitute a new generation-quality test.

| Model key | Current status | Role |
|---|---|---|
| `balanced-flan` | Available; default | Broader deployment continuation with untimed present-tense coverage |
| `reconstruction-flan` | Available | Time, negation, and quantity adaptation |
| `study-augmentation` | Available | Completed synthetic-augmentation arm of the new matched study |
| `pretrained_flan_t5` | Available in local cache | Original FLAN-T5-small comparison baseline |
| `pretrained_t5` | Available in local cache | Original T5-small comparison baseline |
| `e0_copy` | Always available | Text-normalization heuristic; no learned reconstruction |
| `study-baseline`, `study-grammar`, `study-full` | No completed weights | Remaining matched-study arms |
| `e1_t5`, `e2_flan`, `e3_grammar_flan`, `e4_glossweaver` | Original weights missing | Historical experiments with preserved results |
| `legacy_e1`, `legacy_e4` | Weights unavailable | Older domain/prompt configurations |

Current training lineage:

```text
google/flan-t5-small
  → study-augmentation: 1,000 optimizer updates
  → reconstruction-flan: 300 additional updates
  → balanced-flan: 600 additional updates
```

The completed run directories are:

- `checkpoints/study-augmentation/20261010T182714Z-0c55b514/`
- `checkpoints/reconstruction-flan/20261010T184103Z-d5f247db/`
- `checkpoints/balanced-flan/20261010T185808Z-8b1dacec/`

Each model family uses `latest.json` to select its completed versioned run. The three available fine-tuned models above do **not** have an auxiliary grammar classifier.

## 9. Grammar-aware model

`GrammarAwareT5` extends `T5ForConditionalGeneration` with masked mean pooling over encoder states and a linear multi-label classifier. It combines normal sequence-generation loss with binary cross-entropy grammar loss:

```text
total loss = generation loss + grammar_lambda × grammar loss
```

Active research presets use seven labels and `grammar_lambda = 0.1`:

| Label | Intended reconstruction feature |
|---|---|
| `ARTICLE` | Insert articles such as a, an, or the |
| `PREPOSITION` | Restore prepositional relationships |
| `AUXILIARY` | Add auxiliary/modal verbs |
| `PRONOUN` | Restore or realize pronouns |
| `TENSE_ASPECT` | Realize tense/aspect distinctions |
| `AGREEMENT_INFLECTION` | Restore agreement and inflected forms |
| `WORD_ORDER` | Reorder shared content words |

Labels are deterministic weak supervision derived from source/target comparisons, not human gold annotations. The default policy is `lexical-v1`; spaCy parsing requires explicit configuration and an installed parser. At inference, sigmoid outputs estimate grammatical features. They are not calibrated confidence in meaning or translation quality.

The historical 200-pair audit reported stronger agreement for articles, prepositions, auxiliaries, pronouns, and word order than for tense/aspect and agreement/inflection. Weak-label reliability remains a research limitation.

## 10. Datasets and split policy

Counts below come from current processed manifests.

| Corpus | Local processed folder | Counts | Intended use |
|---|---|---|---|
| ASLG-PC12 grouped | `datasets/processed/aslg_grouped/` | 64,789 train; 8,104 validation; 8,100 test | Rule-generated gloss/English research data |
| 2M-Flores-ASL cleaned | `datasets/processed/2m_flores_asl/` | 994 dev; 1,016 devtest | Human-annotated external development and reserved final evaluation |
| Reconstruction grouped | `datasets/processed/reconstruction_grouped/` | 17,421 train; 2,168 validation; 2,196 test | Synthetic English fragments from public text and controlled constructions |
| Time/quantity adaptation | `datasets/processed/demo_adaptation/` | 7,368 train; 1,430 validation; 462 test | Separate synthetic deployment recipe; training samples 400 validation pairs |
| Balanced adaptation | `datasets/processed/balanced_adaptation/` | 15,968 train; 709 validation; 63 untimed test | Broader mixture with repeated untimed training examples |

**ASLG-PC12:** The acquired raw corpus has 87,710 rows and 81,017 unique cleaned pairs in the historical audit. It is rule-generated parliamentary text, not a direct measure of natural ASL quality. The repaired split groups normalized gloss sources so variants remain in one partition; 24 nonlexical sources were quarantined. Recorded license: CC BY-NC 4.0.

**2M-Flores-ASL:** The text-only export contains 999 raw dev and 1,016 raw devtest rows. Five dev rows overlapping devtest were quarantined. Normalization preserves gloss markers, fingerspelling, numbers, punctuation, and classifiers. No training split is invented and videos are excluded. Devtest is reserved for final evaluation. Recorded license: CC BY-SA 4.0.

**Synthetic reconstruction:** Includes public LibriSpeech English text and controlled compositional, condition, and schedule examples. These are telegraphic reconstruction examples, not authentic ASL. Negation and quantities are protected by corruption checks.

**Adaptation corpora:** Kept separate from the matched study. Existing validation/test sources and literal development probes are excluded from training. Present tense is an explicit default for untimed examples, not evidence of the intended tense in real discourse.

**Other sources:** LibriSpeech-Gloss official gloss files were not acquired for experimental use. Public LibriSpeech-PC English manifests must not be labeled as that gloss dataset. ASLLRP/NCSLGR remains an optional, unacquired external evaluation source; How2Sign adapter code exists but is not part of the current completed training lineage.

Split manifests record IDs, checksums, seeds, counts, and quarantine decisions. Source-isolation assertions prevent normalized gloss overlap between ordinary training and validation sets. Intentional overfit checks are the explicit exception. Repaired splits do not retroactively validate historical results.

## 11. Training and reproducibility

The four new matched presets share FLAN-T5-small initialization source, grouped base data, fixed data-selection seed 42, a common validation mixture, independent initialization, and 1,000 optimizer updates. Seeded uniform sampling with replacement provides 250 updates per validation interval.

| Preset | Base training sample | Added synthetic sample | Grammar objective |
|---|---:|---:|---|
| `study-baseline` | 5,000 | 0 | No |
| `study-augmentation` | 5,000 | 5,000 | No |
| `study-grammar` | 5,000 | 0 | Yes |
| `study-full` | 5,000 | 5,000 | Yes |

All use 500 base plus 500 synthetic validation examples. Synthetic examples are added rather than replacing the base sample. Only `study-augmentation` currently has completed local weights.

The training loop supports gradient accumulation, correct partial accumulation groups, gradient clipping, nonfinite-loss failure, warmup/decay, and selection by validation generation loss. Runs save weights/tokenizer files, history, training metadata, full configuration, selected IDs, dataset hashes, initialization lineage, library versions, and optimizer/scheduler state.

Exact automatic resume is not implemented. Continued training creates a new run and resets the optimizer. Historical E1–E4 presets remain identifiable, but new research claims should use the repaired data and matched study.

The completed balanced continuation used MPS, batch size 16, accumulation 1, learning rate `8e-5`, and 600 updates. Its fourth validation interval had the lowest generation loss.

## 12. Evaluation and measured results

Evaluation supports BLEU with scorer signature, chrF, ROUGE-L, METEOR, exact match, optional BERTScore, heuristic content-word recall, source-copy ratio, multi-reference challenge scoring, and grammar diagnostics.

Samples are deterministic and cover corpus origins rather than simply taking file prefixes. Each evaluation saves predictions and a manifest under `results/evaluations/<evaluation-id>/`, including checkpoint, dataset, and sample hashes. These snapshots are authoritative; `results/metrics/model_metrics.csv` is a convenience index. Concurrent evaluation writers should be avoided.

### Current deployment comparisons

Saved comparisons use identical sample/data hashes and beam settings within each model pair.

| Development diagnostic | Examples | Reconstruction BLEU | Balanced BLEU | Reconstruction exact match | Balanced exact match |
|---|---:|---:|---:|---:|---:|
| Balanced validation sample | 120 | 64.47 | 68.24 | 34.2% | 42.5% |
| Untimed present templates | 63 | 63.87 | 99.46 | 19.0% | 98.4% |
| Time/quantity regression | 80 | 100.00 | 100.00 | 100.0% | 100.0% |

Balanced validation also shows heuristic content recall improving from approximately 0.874 to 0.907. These are development diagnostics. The broad validation set guided checkpoint selection, and the other slices use narrow generators. They do not establish authentic-ASL or unrestricted reconstruction accuracy.

### Earlier protected reconstruction test sample

The saved 100-example comparison uses a common source-isolated sample, seed 42, four beams, and batch size 16:

| Model | BLEU | ROUGE-L | Exact match | Heuristic content recall |
|---|---:|---:|---:|---:|
| Pretrained FLAN-T5-small | 9.25 | 0.652 | 0.00 | 0.990 |
| Study augmentation | 56.56 | 0.856 | 0.32 | 0.877 |
| Reconstruction adaptation | 56.35 | 0.845 | 0.31 | 0.865 |

Pretrained and fine-tuned prompts and devices differed, so this is a workbench comparison rather than a controlled paper ablation. Better fluency did not guarantee better lexical preservation; genuine hallucinations remain on long or rare fragments. Balanced-model numbers for this exact protected sample are not reported in this table.

### Historical E0–E4 results

`results/REPORT.md` preserves the earlier recovery study:

| System | ASLG BLEU | Synthetic BLEU | Nine-example challenge BLEU |
|---|---:|---:|---:|
| E0 copy | 20.24 | 21.48 | 11.93 |
| E1 T5-small | 54.48 | 7.13 | 2.61 |
| E2 FLAN hybrid | 65.45 | 23.91 | 84.40 |
| E3 grammar continuation | 67.02 | 25.11 | 84.40 |
| E4 scaled recipe | 72.93 | 29.55 | 90.92 |

ASLG and synthetic results used 500-example slices; the challenge had nine inputs and three references each. The stages had different training histories and compute budgets, so they are not a matched ablation. The original neural weights are missing. The historical report's statement that E4 was the serving default describes that earlier state and does not describe this checkout today.

## 13. Setup and common commands

Run commands from the repository root unless a `cd` is shown.

### Start the integrated app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[api,dev]'
cd frontend
npm ci
npm run build
cd ..
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. This uses existing local artifacts; installing dependencies does not restore missing checkpoints.

For frontend development, run `npm run dev` inside `frontend` while the backend runs on port 8000. Vite's development server proxies API requests. `npm run preview` alone does not provide that API proxy.

### Acquire and prepare data

```bash
python scripts/download_data.py aslg-pc12
python scripts/prepare_data.py
python scripts/download_external_data.py
python scripts/prepare_research_data.py
python scripts/prepare_demo_adaptation.py
python scripts/prepare_balanced_adaptation.py
```

External preprocessing can reuse a downloaded text export with `python scripts/download_external_data.py --offline`. Preparation steps have input dependencies; they are not interchangeable. Consult the dataset guides before rebuilding historical folders.

### Train and reconstruct

```bash
python -m glossweaver.training.train --preset study-augmentation --device mps --seed 42
python -m glossweaver.training.train --preset demo-adaptation --device mps
python -m glossweaver.training.train --preset balanced-adaptation --device mps

python -m glossweaver.inference \
  --model balanced-flan \
  --text "ME GO STORE YESTERDAY" \
  --device auto
```

Use `cuda` for an NVIDIA accelerator or `cpu` without an accelerator. The two adaptation presets depend on their preceding completed checkpoints.

### Evaluate external development data

```bash
python -m glossweaver.evaluation.evaluate \
  --model study-augmentation \
  --dataset-file datasets/processed/2m_flores_asl/dev.jsonl \
  --dataset-name 2m_flores_asl \
  --split dev \
  --device mps \
  --output results/predictions/study_augmentation_flores_dev.csv
```

This is a command example, not a claim that this evaluation has been completed. Reserve devtest for the final frozen model. BERTScore may require a separate model download; `--skip-bertscore` explicitly omits it.

`GLOSSWEAVER_ROOT` selects the package's artifact root. `GLOSSWEAVER_CORS_ORIGINS` configures permitted frontend origins. `GLOSSWEAVER_GRAMMAR_PARSER` controls the optional grammar parser policy.

## 14. Testing and implementation evidence

The repository has tests for preprocessing, datasets and collation, corruption, labels, split isolation, model behavior, metrics, inference limits, API contracts, training edge cases, and evaluation artifacts. The smoke script exercises a tiny local T5 and grammar-aware model through forward/backward, optimization, save/reload, and generation without downloading pretrained weights.

```bash
python -m pytest
python scripts/run_smoke_test.py
cd frontend
npm run lint
npm run build
```

`results/VERIFICATION.md` records 39 passing Python tests during the latest implementation, subsequent targeted API/evaluation checks, successful frontend lint/build, and real browser/API checks at desktop and 390px mobile widths. Saved UI images are under `results/ui/`.

GitHub Actions is configured for Python 3.11 and Node 24, running Python tests/smoke and frontend lint/build. The verification report does not establish that remote CI has run.

For this documentation task, registry availability and current dataset manifests were inspected directly. Training, neural inference, and the full test suite were not rerun; the verification outcomes above are recorded prior results.

## 15. Current status and remaining work

**Implemented and locally available:** The integrated workbench, API, CLI inference, explicit model availability, three completed fine-tuned checkpoints, source-grouped datasets, human-annotated external text data, reproducible training/evaluation artifacts, deployment diagnostics, and automated check configuration.

**Remaining research work:** Complete the other matched-study arms, repeat across training seeds, perform independent human review, evaluate authentic ASL with appropriate methodology, freeze the final model before devtest use, and write claims that match the resulting evidence. Restoring historical E1–E4 reproduction requires their original weights.

**Remaining engineering limits:** Exact optimizer resume, a production multi-user service, and durable server-side session history are not implemented. Serving keeps one resident model and handles one reconstruction at a time.

**Model limitations:** Ambiguous tense, unfamiliar vocabulary, names, long inputs, quantities, semantic roles, and negation still require inspection. Present tense without a time cue is a deployment convention. High automatic scores on synthetic templates do not prove generalization, and fluent output can change meaning.

## 16. Documentation and artifact guide

| File or folder | What to consult it for |
|---|---|
| `README.md` | Current setup, app operation, study commands, limitations |
| `datasets/README.md` | Latest dataset counts, preprocessing and split policy |
| `DATASETS.md` | Source provenance and historical data context |
| `PRODUCT.md`, `DESIGN.md` | Workbench purpose and interface direction |
| `plan.md` | Implementation checklist and follow-up progress |
| `PROJECT_AUDIT.md` | Original review and research/engineering concerns |
| `PROJECT_WALKTHROUGH.md` | Earlier extensive walkthrough; verify time-sensitive claims against current code |
| `results/RECONSTRUCTION_TRAINING.md` | Completed training, deployment comparisons, measured scope |
| `results/VERIFICATION.md` | Recorded code, browser, and data verification |
| `results/REPORT.md` | Historical recovery study; not current checkpoint availability |
| `results/deployment_comparison/` | Paired development metrics and predictions |
| `results/evaluations/` | Immutable evaluation snapshots with provenance |
| `checkpoints/*/<run-id>/run_manifest.json` | Run configuration, selected IDs, hashes, lineage |
| `archive/` | Preserved legacy and superseded experiments |

Older documents contain statements that were true before new training completed. Current registry availability, versioned run manifests, processed dataset manifests, and immutable evaluation snapshots provide the most concrete evidence for the present project state.
