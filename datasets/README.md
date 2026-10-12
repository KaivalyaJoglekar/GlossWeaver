# Local datasets and preprocessing

## Non-synthetic external data: 2M-Flores-ASL

Official source: https://huggingface.co/datasets/facebook/2M-Flores-ASL
License: CC BY-SA 4.0. Cite the 2M-BELEBELE paper referenced by the official card.
Human translators/native signers create glosses, followed by expert harmonization.
This is distinct from rule-generated ASLG-PC12 and local synthetic reconstruction.

Raw text-only acquisition: `raw/2m_flores_asl/dev.jsonl` (999 rows),
`devtest.jsonl` (1,016 rows). `provenance.json` records source revision, acquisition
method, row counts and SHA-256. Videos are not downloaded.

Processed: `processed/2m_flores_asl/dev.jsonl` (994 rows), `devtest.jsonl` (1,016 rows).
Preprocessing normalizes Unicode NFC, removes BOMs and collapses whitespace.
Gloss markers, fingerspelling, numbers, punctuation and classifiers are preserved.
Exact pair/signer duplicates are removed. Five dev rows have normalized sources
also present in devtest; they are quarantined to protect devtest. Alternative
translations within a split remain. No new train split is invented. `signer` is
not a global signer ID. Dev is for external development; devtest is frozen final
evaluation. The processed corpus intentionally differs from the raw official
corpus by the documented quarantine and must be identified as a cleaned subset.

Reproduce: `python scripts/download_external_data.py`; reuse downloaded text:
`python scripts/download_external_data.py --offline`.

## Existing ASLG-PC12 (rule-generated)

Raw: `raw/aslg_pc12/`, pinned revision recorded in `metadata/aslg_pc12.json`.
Historical processed splits remain in `processed/aslg_pc12/`.
New grouped research splits live in `processed/aslg_grouped/`; duplicate-source
variants belong to one partition. 24 sources without lexical content are
quarantined; meaningful numbers in lexical examples remain. Current counts:
64,789 train / 8,104 validation / 8,100 test.

Grouping does not certify language/alignment quality. Manual review or a validated
language-identification pass remains necessary for multilingual anomalies.

## Synthetic reconstruction (separate)

Historical: `processed/reconstruction/`. Repaired: `processed/reconstruction_grouped/`
with 17,421 train / 2,168 validation / 2,196 test. Source grouping prevents identical
normalized glosses in different partitions. Both grouped corpora regenerate and
store seven deterministic lexical weak labels. These are not human grammar labels.

Run `python scripts/prepare_research_data.py` to rebuild the two grouped corpora
from local historical inputs. Manifests record source checksums, selected IDs,
seed, counts, quarantine, label policy and zero source overlap. Original data and
saved evaluation results are not rewritten by this script.

Raw and processed data are local artifacts. Acquisition/preprocessing code and
README provenance enable reproduction; large generated files are ignored by Git.

## Separate deployment adaptation

`processed/demo_adaptation/` combines selected existing controlled training pairs
with generated travel/action pairs covering past/future markers, negation and
quantities. This is explicitly synthetic and kept separate from external data.
Development probes (including ME GO STORE YESTERDAY) are excluded as literal
sources. Original validation/test partitions are protected; new validation/test
sources also exclude all initial-corpus training sources. It is a deployment
recipe rather than a matched research ablation. Reproduce with
`python scripts/prepare_demo_adaptation.py`, then train `--preset demo-adaptation`
after a completed study-augmentation checkpoint exists.

## Balanced deployment continuation

`processed/balanced_adaptation/` contains 15,968 training records, 709 validation records and 63 held-out untimed diagnostic records. Run `.venv/bin/python scripts/prepare_balanced_adaptation.py` to rebuild it deterministically. The manifest records source hashes and seed 57. It mixes length-filtered ASLG and public English reconstruction examples with controlled examples and time/quantity adaptation. Untimed examples are repeated within training for coverage; present tense is an explicit default when no time is supplied, not a claim about the speaker's intended tense. Existing validation/test sources and development probes are excluded from training. Flores remains unchanged.

Train with `.venv/bin/python -m glossweaver.training.train --preset balanced-adaptation --device mps`. This is a deployment continuation, separate from the matched research ablations. `scripts/compare_deployment_models.py` saves reproducible development comparisons and predictions; these are not final paper results.

The app's **Datasets** tab reads these local manifests and shows the folders and counts. Raw downloads remain under `datasets/raw/` and cleaned files under `datasets/processed/`.
