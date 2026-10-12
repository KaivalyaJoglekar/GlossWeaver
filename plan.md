# GlossWeaver improvement plan

The project reconstructs English from incomplete/telegraphic text. Authentic sign-language gloss reconstruction is an application requiring independent validation. This plan follows PROJECT_AUDIT.md and replaces the older implementation checklist.

## Implementation order

1. **Safe, usable custom-input workbench**
   - Validate text, beam width and thresholds; reject token overflow instead of truncating.
   - Expose model readiness, honest copy-baseline behavior, and actionable missing-checkpoint status.
   - Bound and synchronize the model cache; return safe errors.
   - Build a dark, responsive input/output workbench with sample categories, keyboard submit, elapsed time, copy, session history and JSON export.
2. **Repair benchmark display**
   - Use registry model keys and explicit dataset/split selection.
   - Display actual example counts, saved-result provenance and grammar diagnostics.
   - Show unavailable results and API errors without hiding them.
3. **Data correctness**
   - Group new splits by normalized source; audit train/validation/test input overlap.
   - Keep alternative translations together; record versioned split manifests.
   - Consolidate safe corruption, preserve negation/numbers, and reject ambiguous How2Sign schemas.
   - Add a strict text-only 2M-Flores adapter with explicit acquisition/export instructions.
   - Preserve existing datasets/results; repaired data is generated separately before retraining.
4. **Research-ready experiments and reporting**
   - Add matched FLAN baseline/augmentation/grammar/full presets with identical initialization and optimizer-step budget.
   - Validate training configs and prevent source overlap except intentional overfit gates.
   - Preserve run configs, selected IDs, hashes, initialization lineage and checkpoint sequence limits.
   - Correct partial gradient accumulation, fail on nonfinite loss, and clip gradients.
   - Use deterministic stratified evaluation sampling; preserve scoring configuration, chrF/exact match and per-run manifests.
   - Remove unconditional qualitative verdicts from report generation.
5. **Verification and handoff**
   - Add meaningful tests for HTTP contracts, limits, source isolation, corruption, evaluation sampling and training edge cases.
   - Run Python tests, tiny-model smoke, frontend lint/build, and browser checks at desktop/mobile sizes.
   - Document setup, custom testing, artifact limitations, data rebuilding and new experiments.

## Design direction

A dark research workbench retaining the indigo identity. The main screen prioritizes writing input and reading output. Model status and limitations stay visible; grammar predictions are separate from translation confidence. Benchmarks and methodology live on separate tabs. Accessible native controls, visible focus, reduced motion and mobile stacking are required.

## Acceptance criteria

- A custom input can be submitted through the frontend to the real API; returned text identifies the exact model used.
- Missing trained weights are explicitly visible; copy output is never presented as neural reconstruction.
- Invalid parameters and overlong tokenized input receive actionable validation responses.
- Benchmark rows match saved CSVs and show the real dataset and sample size.
- Newly built datasets have no normalized-source overlap; existing artifacts are not silently rewritten.
- New experiment presets support a reproducible matched comparison; historical E1–E4 remain identifiable.
- Verification passes; any remaining artifact/data/training work is explicitly documented.

## Requires subsequent experiments

Restoring trained weights needs a real artifact source; none is present in this checkout. New model-quality claims require rebuilding clean splits and retraining. Human annotation, blinded ratings, authentic-ASL validation and final paper writing remain research work, not claims that code changes can establish.

## Progress

- [x] Plan written before implementation.
- [x] API and inference safety.
- [x] Frontend workbench and benchmarks.
- [x] Data pipeline and external adapter.
- [x] Research configuration and reporting.
- [x] Tests, browser verification and documentation.

## Dataset acquisition and preparation completed

- Text-only 2M-Flores-ASL downloaded locally under `datasets/raw/2m_flores_asl/`: 999 dev and 1,016 devtest rows, with revision/license/checksums.
- Cleaned external corpus under `datasets/processed/2m_flores_asl/`: 994 dev and 1,016 devtest; five overlapping dev rows quarantined, original gloss notation preserved.
- ASLG source-grouped splits: 64,789 / 8,104 / 8,100; 24 nonlexical inputs quarantined.
- Synthetic corpus kept separate, source-grouped splits: 17,421 / 2,168 / 2,196.
- New study presets use 1,000 optimizer updates with validation every 250 steps, identical initialization source and fixed data selection seed.

Training the new presets and independent human evaluation remain subsequent experiments; no neural model was trained as part of this implementation.

## Reconstruction quality follow-up

- Completed study-augmentation training: 1,000 M4/MPS optimizer updates, with best checkpoint selected by validation generation loss.
- Completed a separate 300-update deployment adaptation for time markers, negation and quantities. The original matched-study checkpoint is preserved.
- Default serving model is `reconstruction-flan`; cached pretrained weights remain comparison baselines.
- Source-isolated test scores and development probes are documented in `results/RECONSTRUCTION_TRAINING.md`. These checks do not constitute authentic-ASL or paper-quality human validation.

## Minimal interface and balanced model follow-up

- [x] Show local dataset locations, counts and purpose in a Datasets tab.
- [x] Use Emil Design Engineering and Impeccable to simplify the dark workbench; compact editor/status, runnable-only model selector and collapsed session history.
- [x] Prepare a reproducible balanced deployment mixture with untimed present defaults and preserved source partitions.
- [x] Train a separate 600-step MPS continuation and preserve the earlier checkpoint.
- [x] Compare identical development samples; promote balanced-flan after improved untimed and broad diagnostics with preserved time/quantity regression results.
- [x] Explicitly distinguish development/template diagnostics from final authentic-ASL results.
