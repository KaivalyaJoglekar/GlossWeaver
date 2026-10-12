# GlossWeaver codebase and dataset audit

Reviewed 10 October 2026. This is an analysis report; application and training code were not changed.

## Assessment

GlossWeaver is a useful, compact research prototype for reconstructing English from gloss-like text. Its module boundaries, shared input formatter, pinned ASLG revision, explicit model registry, and separate generation/grammar loss logging are good foundations. Its strongest current evidence concerns synthetic English reconstruction. The existing results do not establish robust natural-ASL translation or isolate the benefit of grammar supervision.

The next investment should be evaluation integrity and human gloss data, before larger models or longer training.

## Scope and checks

Reviewed the active Python package (data, modeling, training, evaluation, inference, configuration), API, React components and styles, scripts, tests, project documentation, dataset metadata, local processed splits, checkpoint metadata, and saved result tables. Historical archives were treated as provenance, not as active code. No full retraining, exhaustive historical notebook execution, or browser visual/accessibility audit was performed.

Executed:

- Python suite: **23 tests passed**.
- Tiny-model smoke script: passed forward/backward, optimizer, generation, save/reload.
- Frontend production build: passed.
- Frontend lint command: exited successfully.
- FastAPI TestClient checks: models and health return 200; `/api/metrics/e4` returns 404; `/api/metrics/e4_glossweaver` returns 200; E4 inference returns 404 because weights are absent.
- Normalized source-overlap audit of all local training/validation/test data.

All neural checkpoints in the registry are unavailable in this checkout: tokenizer/configuration/metadata files exist, but no recognized weight files exist. `/api/models` currently marks `e0_copy` as default. Consequently, saved neural scores were inspected, not reproduced.

## Priority findings

### 1. High: the metrics UI does not match the backend

Locations: `frontend/src/components/MetricsPanel.jsx:5`, `:23`, `:40`; `app/api.py:123`.

There are multiple independent contract failures:

- UI requests `e0`–`e4`; CSV model names are `e0_copy`, `e1_t5`, `e2_flan`, `e3_grammar_flan`, `e4_glossweaver`. All current requests return 404.
- API returns `{rows: [...]}`; UI reads metric fields directly from the response.
- UI expects `sacrebleu`; API supplies `bleu`. UI also expects chrF and exact match, which the consolidated metrics writer does not persist.
- UI expects nested grammar diagnostics; this endpoint never supplies them.
- Per-request errors are swallowed and filtered away, leaving an empty table instead of a useful error state.
- UI states 2,500 ASLG test examples; saved active evaluations contain 500 ASLG examples, 500 synthetic examples, or nine challenge examples.

Improvement: define one typed response schema, use registry keys, require dataset/split selection, return diagnostics explicitly, display actual sample sizes, and show failures. Add an API-to-component contract test.

### 2. High: synthetic split assignment permits identical sources across splits

Location: `src/glossweaver/data/build_reconstruction_data.py:360`–`:371`.

Assignment hashes the English target, while tense neutralization and function-word removal map different targets to identical glosses. The full local corpus has:

| Splits | Shared normalized glosses | Rows in second split with shared gloss |
|---|---:|---:|
| Train / validation | 596 | 651 / 2,195 |
| Train / test | 610 | 666 / 2,202 |
| Validation / test | 100 | 105 / 2,202 |

Example: `AUNT BAKE BREAD THIS EVENING` appears with “The aunt will bake bread this evening.” in train and “The aunt bakes bread this evening.” in test.

This is source overlap, not exact source-target pair duplication. It weakens claims of generalization to unseen gloss inputs and creates contradictory single-target supervision for ambiguous inputs.

Important qualification: the first 500 synthetic test rows contain **zero** training-gloss overlap. They are all `public_librispeech_clean`. The full-corpus overlap therefore does not by itself show that the published 500-row synthetic BLEU is inflated. It does affect the broader split design and synthetic validation.

ASLG also has 16 train/validation and 12 train/test shared normalized glosses. Its existing leakage assertion checks pairs, not sources.

Improvement: split by normalized gloss group; keep all acceptable translations together; additionally reserve templates/compositions and source documents for harder generalization tests. Store source/document IDs when constructing public-text examples. Separate valid ambiguity from duplicate contamination.

### 3. High: the nine-example challenge is insufficient as the main quality claim

Locations: `results/REPORT.md`; `src/glossweaver/data/build_reconstruction_data.py:197`, `:335`; `scripts/build_research_artifacts.py`.

The recovery recipe explicitly covers the challenge's vocabulary and constructions, including mother/cooking, teacher/math, tail adjectives, sunny conditions, and school schedules. Exact challenge rows are excluded, which is useful, but exact exclusion does not make a development-informed challenge a blind external test. The report itself describes iterative repair of uncovered constructions.

This does not establish intentional leakage or contradict the narrower assertion that challenge scores were not used to choose the best validation epoch. It means the challenge should be described as a development/regression set, with a fresh hidden test used for generalization claims.

The saved synthetic test slice excludes every controlled origin, despite these origins being central to the recovery recipe. `evaluate.py:49` simply selects a prefix; builder output order places public examples first.

Improvement: freeze stratified manifests, evaluate the entire clean test set, report each origin separately, add unseen vocabulary/compositions, and use an independently annotated ASL test. Report confidence intervals and several training seeds.

### 4. High: experiments do not isolate grammar supervision

Locations: `src/glossweaver/settings.py`; `src/glossweaver/training/train.py`; checkpoint training metadata.

E1 uses T5 and 5,000 ASLG examples. E2 changes initialization to FLAN, adds synthetic data, doubles the initial dataset size, and undergoes additional staged training. E3 continues E2; E4 continues E3 with more data. E2 versus E3 therefore also changes training duration, and E3 versus E4 changes scale and cumulative training.

The report acknowledges unequal compute, which is good. However, the measured E3 gain cannot be attributed solely to the grammar objective without a same-duration continuation of E2 with grammar loss disabled.

The mixer uses a per-source cap: E2 is 5,000 base + 5,000 synthetic, not 25% augmentation. E4 is 20,000 base + 17,388 synthetic, about 46.5% synthetic rows. Dataset documentation still describes 25% defaults.

Improvement: compare a single backbone across data/no-data and grammar/no-grammar conditions with matched initialization, optimizer steps, examples, validation manifests, and decoding. Add a continued-E2 control and lambda sweep including zero; then test scaling separately.

### 5. High: released artifacts and setup do not support the documented result

Locations: `README.md`; `.gitignore`; `pyproject.toml`; `src/glossweaver/model_registry.py:93`.

The README states E4 is served by default, but this checkout defaults to E0 because no trained weights exist. Explicit E4 requests correctly fail instead of silently substituting another model.

Python dependencies omit FastAPI and Uvicorn, although the documented product needs both. API tests additionally need an HTTP test client. The README does not give a complete installation/download sequence. Active reconstruction building depends on a LibriSpeech-PC file for which the current downloader has no acquisition command. There is no Python lockfile or CI workflow.

Improvement: provide a versioned checkpoint release with checksum and acquisition command; add API extras; document installation and corpus acquisition; distinguish liveness from model readiness. Preserve inference configuration and tokenizer hashes with model artifacts. Resolve data directories through configuration rather than installed source-file ancestry, which does not reliably identify a project directory in a wheel installation.

### 6. Medium: an exposed corruption path still corrupts non-verbs

Locations: `src/glossweaver/data/corruption.py:62`; `scripts/generate_synthetic.py`.

Verified reproduction:

```python
corrupt_sentence("This analysis is important tonight.", seed=42, probability=1).fragment
# THI ANALYSI IMPORTANT TONIGHT
```

The active reconstruction builder has safer verb-aware rules, but the separate augmentation script still calls `simple_lemma` on every token. A user can regenerate malformed data with a maintained script. `simple_lemma` also affects weak labels and content metrics.

Improvement: consolidate corruption into one implementation with POS-aware morphology and explicit transformations; add regressions for nouns, pronouns, irregular verbs, negation, numbers, and punctuation.

### 7. Medium: cleaned data is not consistently meaningful English reconstruction data

Location: `src/glossweaver/data/preprocess.py:35`.

Cleaning checks empty strings and duplicates, but does not detect language mismatch or sources without lexical content. Across local ASLG splits, **24 glosses contain no ASCII alphabetic characters**; this count includes number-only examples and should not be called purely punctuation-only. One training row maps `.` to a Greek sentence. Another maps `! ?` to Spanish text.

Improvement: quarantine language/alignment anomalies and uninformative sources with reviewable reasons. Preserve meaningful numbers and multilingual named entities. Measure tokenized truncation, domain/vocabulary coverage, and source-target alignment quality before training.

### 8. Medium: grammar predictions and diagnostics overstate what is measured

Locations: `src/glossweaver/data/grammar_labels.py`; `src/glossweaver/evaluation/grammar_metrics.py`; `frontend/src/components/GrammarBars.jsx`.

Grammar targets are heuristics and change depending on whether `en_core_web_sm` happens to be installed. The parser mode is not frozen with processed data/checkpoints. The known-operation audit is a consistency check against synthetic operations, not independent human annotation.

`grammar_diagnostics` compares heuristic features of generated text with heuristic features of one reference; it does **not** evaluate the model's classification logits. On the multi-reference challenge it still uses only the primary reference. Grammar bars describe recovery actions, although a classifier probability does not establish that the generated output correctly performed an action.

Improvement: version the parser and labeling rules, cache labels, add a human-labeled validation subset, distinguish output diagnostics from auxiliary-head classification, and calibrate category thresholds. Treat bars as predicted grammatical features rather than verified explanations or confidence in the translation. About/Metrics still describe six categories, while active data has seven.

### 9. Medium: API inputs and model resource usage are unbounded

Location: `app/api.py:55`–`:59`, `:32`–`:52`, `:110`.

`InferRequest` accepts arbitrarily long glosses, negative or extremely large beam widths, and thresholds outside [0,1]. Invalid decoding values become 500 errors; excessive beam search can consume large amounts of memory. The global cache retains every selected model and has no initialization lock or eviction policy. Concurrent cold requests can load duplicate models. Detailed exceptions and absolute filesystem paths are returned to clients.

Improvement: validate gloss size, beam range, and finite bounded thresholds; return validation errors; limit concurrent inference and resident checkpoints; synchronize loading; return opaque model versions and log full errors server-side. Restrict CORS to intended origins for deployment. These are deployment concerns; no deployed outage or exploit was demonstrated.

### 10. Medium: inference silently truncates input

Location: `src/glossweaver/inference.py:72`, `:102`.

Both inference paths hardcode 96 source tokens regardless of checkpoint training configuration; excess input is silently dropped. Generation is limited to 64 new tokens while training permits 96 target tokens. Names, numbers, negation, or later clauses can disappear.

Improvement: persist sequence lengths with checkpoints, measure truncation rates, reject or explicitly surface unsupported lengths, and avoid sentence chunking that discards context. Reuse encoder outputs for auxiliary grammar predictions instead of a second encoder pass after generation.

### 11. Medium: run history and results are not immutable/reproducible

Locations: `src/glossweaver/training/train.py:132`; `trainer.py:115`; `evaluation/evaluate.py:88`; `scripts/build_research_artifacts.py`.

Continuation overwrites history and metadata in the same experiment directory. Final E2 metadata points to its own mutable checkpoint as initialization; it does not reconstruct the staged lineage. Full config, dataset hashes, selected IDs, library/model revisions, optimizer/scheduler/RNG state, and cumulative steps are not stored together. Existing presets do not encode the complete historical recipe described in the report.

Evaluation hardcodes seed 42, replaces results by model/dataset/split/beams without a checkpoint hash, omits returned chrF/exact match and scoring-model metadata, and labels `str(bleu)` as a SacreBLEU signature. That string is a score summary, not a proper scorer configuration signature. Concurrent CSV updates can overwrite one another. On a fresh output tree, diagnostic output assumes `results/tables` already exists.

The research-artifact builder hardcodes training counts and classifies every new challenge prediction as `VALID_PARAPHRASE`, irrespective of its content. It would repeat this claim after a model regression. This is a concrete reporting defect.

Improvement: immutable run IDs; full manifest and checkpoint hashes; append-only lineage; resumable optimizer state; actual scorer signatures; immutable evaluation files; independently reviewed qualitative labels. Generate prose only from validated records, with no unconditional quality verdicts.

### 12. Lower priority: training and frontend edge cases

- `trainer.py:77`: every microbatch is divided by the full accumulation factor, including a smaller final group. This underweights the last optimizer update when the loader length is not divisible by that factor. Current E1–E4 use accumulation 1, so this does not explain their reported gains.
- Validation losses average batch means equally rather than accounting for different batch/token counts. Empty validation data, nonfinite loss, invalid configuration, and missing grammar labels need explicit failures.
- No gradient clipping or numerical failure checks; consider these before mixed precision/longer training. Precompute tokenization and synthetic grammar labels to reduce repeated work.
- Validation uses greedy generation while serving uses four beams and repetition controls. Record and align decoding when interpreting validation metrics.
- Inference UI accepts edits/model changes during an outstanding request and can display an old result under a new selection. Snapshot request state or cancel/ignore stale responses.
- `parseFloat(value) || 0.5` prevents choosing threshold zero. Clipboard failures are reported as “Copied”.
- Navigation has tab roles but lacks associated tab panels and keyboard tab behavior. Announce loading/errors/results, respect reduced motion, and add keyboard and narrow-screen checks.

## Dataset recommendations

Sources were checked online on 10 October 2026. Recommendations distinguish actual gloss/English pairs from video datasets and synthetic corpora. Do not infer that an advertised video/text corpus provides a usable gloss column.

| Priority | Dataset | Fit and intended use | Acquisition/limitations |
|---|---|---|---|
| 1 | **2M-Flores-ASL** | Strongest immediately verified external gloss-to-English candidate. Dataset card exposes `gloss`, `sentence`, source URL/topic, and dev/devtest splits; glosses are created by ASL translators/native signers and checked by expert annotators. | 2,015 rows shown in the current card, not two million examples. CC BY-SA 4.0. Preserve official splits and reserve devtest for final testing. Access text columns without decoding/downloading the full 326 GB video payload. [Official dataset](https://huggingface.co/datasets/facebook/2M-Flores-ASL). |
| 2 | **ASLLRP / NCSLGR continuous signing** | Authentic linguistic annotations; strong option for natural-ASL constructions, classifiers and non-manual markers. NCSLGR documents 1,002 narrative + 885 elicited utterances. | XML/SignStream adapter required. Verify translation tiers and usable alignment for each release before counting parallel pairs; preserve story/signer grouping and applicable project terms. Existing project policy reserves it for external evaluation. [Official download documentation](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html), [current resource overview](https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html). |
| 3 | **A new human-annotated everyday ASL corpus** | Best match to the current product examples. Collect school, household, appointments, shopping, questions, negation, roles, time, quantities and ambiguous cases. | Proposed project work, not an existing public download. Start with 500–1,000 independently reviewed test items and a separate adaptation set. Involve fluent Deaf ASL users; collect context and multiple acceptable English renderings. |
| 4 | **LibriSpeech-Gloss** | Larger synthetic reconstruction/pretraining candidate: paper reports 266,655 utterances across seven partitions. | Automatically generated, not natural-ASL gold data. The publication landing page did not establish a verified downloadable artifact or dataset license. Acquire and inspect author-provided release first; the local English LibriSpeech-PC files are not this dataset. [Primary publication](https://infocomp.dcc.ufla.br/index.php/infocomp/article/view/5453). |
| Conditional | **How2Sign** | Instructional domain; official project describes over 80 hours of ASL and aligned English text. | CC BY-NC 4.0. Public download page does not establish an immediately usable full gloss/English export. Confirm annotation access before integration; otherwise useful for text augmentation or a separate video project. Existing `how2sign.py` misleadingly accepts `SENTENCE` as a gloss alias; require explicit gloss columns. [Official project](https://how2sign.github.io/). |
| Supporting | **ASLLVD / ASL Sign Bank** | Gloss conventions, lexical variants and vocabulary normalization. | Lexical/citation-form signs do not provide sentence translation supervision. Keep this as a normalization resource rather than adding isolated signs as full-sentence training pairs. [Official ASLLVD documentation](https://www.bu.edu/asllrp/av/dai-asllvd.html). |
| Separate experiment | **RWTH-PHOENIX-2014T** | Established human gloss/translation architecture benchmark. | German Sign Language → German, in a weather domain; not interchangeable with ASL → English. Use a separate model/configuration if testing cross-language generality. [Official corpus](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). |
| Future video scope | **OpenASL** | Broad ASL video/English resource. | Official repository distributes clip metadata/text; it does not establish a ready gold gloss/English corpus for this text-only task. Requires annotation or a separate video translation effort. [Official repository](https://github.com/chevalierNoir/OpenASL). |

Existing ASLG-PC12 should remain a labeled synthetic/rule-generated baseline, with its current provenance and license record preserved. More generated English pairs may expand vocabulary but cannot substitute for independent authentic-ASL testing.

For every new corpus, record source revision/checksum, license, language pair, real versus generated gloss, annotation convention, alignment method, document/signer identity where available, and an immutable split manifest. Preserve fingerspelling, indexing, negation, classifiers, numbers, and non-manual markers; generic punctuation stripping would destroy useful distinctions. The 2M-Flores `signer` field is explicitly not a global signer ID, so it cannot support a naive signer-holdout claim.

## Recommended implementation order

1. Restore/version trained artifacts and make setup reproducible; fix the metrics contract and sample-count labels.
2. Freeze group-disjoint and composition-held-out splits; quarantine malformed examples; replace prefix-only synthetic testing with stratified/full evaluation.
3. Add a schema-verified 2M-Flores text adapter and frozen external test; develop an ASLLRP XML adapter under the existing evaluation-only policy.
4. Consolidate corruption and freeze grammar-label generation. Add focused API, data-isolation, continuation, and reporting tests; run them in CI.
5. Run matched FLAN experiments with/without augmentation and grammar loss, using multiple seeds and immutable lineage.
6. Evaluate semantic fidelity independently: negation, who did what to whom, names, numbers, time, ambiguity, and unsupported input. Use blinded human ratings alongside BLEU/chrF/semantic metrics.
7. Only then compare a larger backbone, parameter-efficient fine-tuning, decoding alternatives, and domain-balanced sampling. Treat grammar probabilities separately from reconstruction quality.

Success should mean measurable gains on new natural-ASL glosses and preserved meaning, with reproducible artifacts and a truthful working interface—not merely higher scores on the existing nine examples.
