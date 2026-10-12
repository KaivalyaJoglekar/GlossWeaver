# Reconstruction training and serving follow-up

Completed locally on M4/MPS. The original E1–E4 weight files remain unavailable; these are new runs.

- Study augmentation: 10,000 training pairs, common 1,000-pair validation mix, 1,000 optimizer updates; best validation epoch 4.
- Deployment adaptation: 7,368 synthetic training pairs, 400 sampled validation pairs, 300 additional updates; best validation epoch 3.
- Adaptation covers explicit time markers, negation and quantities. It is a separate deployment recipe and must not be presented as the matched study ablation.
- Default UI model: **FLAN-T5-small · reconstruction** (`reconstruction-flan`). All checkpoints, optimizer states, data IDs and initialization hashes are stored locally.

## Protected original test sample

Same 100 source-isolated examples and sample checksum; seed 42, beams 4, batch size 16. Pretrained prompt differs from the fine-tuned training prompt. Pretrained scoring ran on CPU; trained scoring ran on MPS. This is a workbench comparison, not a controlled paper ablation.

| Model | BLEU | ROUGE-L | Exact match | Heuristic content recall |
|---|---:|---:|---:|---:|
| pretrained_flan_t5 | 9.25 | 0.652 | 0.00 | 0.990 |
| study-augmentation | 56.56 | 0.856 | 0.32 | 0.877 |
| reconstruction-flan | 56.35 | 0.845 | 0.31 | 0.865 |

Lexical recall decreased despite improved fluency. Some changes are inflection mismatches in the heuristic, but genuine hallucinations remain on long/rare literary fragments. Automatic scores do not establish semantic fidelity.

## Targeted synthetic holdout

100 examples from the separate adaptation test split: BLEU 100, exact match 100%. This is a narrow, fixed-template synthetic test of time/quantity constructions; it is not evidence of general ASL or unrestricted English translation accuracy. Sources were checked against both the initial training selection and adaptation training.

## Development probes

These are debugging examples, not an independent final benchmark. The five reported probes are excluded literally from generated adaptation training.

| Input | Final CPU output |
|---|---|
| ME GO STORE YESTERDAY | I went to the store yesterday. |
| BOY NOT WANT FOOD | The boy will not want food. |
| i going school tomorrow | I will go to the school tomorrow. |
| WOMAN BUY CAR RED | The woman bought a red car. |
| I NOT BUY 3 BOOK YESTERDAY | I did not buy 3 books yesterday. |
| SHE NOT GO HOSPITAL TOMORROW | She will not go to the hospital tomorrow. |
| WE READ 5 BOOK LAST NIGHT | We read 5 books last night. |
| ME WALK LIBRARY NEXT WEEK | I will walk to the library next week. |

The final real browser/API check produced **I went to the store yesterday.** on MPS. Tense remains ambiguous when no time cue is present; for example the negation probe received a future-tense completion.

Code checks: 39 Python tests passed. Both the new default selection and HTTP contract remain covered. Frontend lint/build passed in the preceding implementation. The adaptation does not add a grammar classifier.

Reproduce preparation: `python scripts/prepare_demo_adaptation.py`. Reproduce adaptation after study training: `python -m glossweaver.training.train --preset demo-adaptation --device mps`.

## Balanced continuation (2026-10-11 local time)

The minimal workbench defaults to `balanced-flan`. Its preserved checkpoint is `checkpoints/balanced-flan/20261010T185808Z-8b1dacec`, initialized from the previous reconstruction checkpoint, trained on MPS for 600 optimizer steps, batch size 16, accumulation 1, learning rate 8e-5. The recipe mixes 15,968 records, including repeated untimed examples, and validates on 709 records; epoch 4 had the lowest validation generation loss. The previous model remains selectable.

`results/deployment_comparison/summary.json` stores six evaluations with identical samples/settings per model, plus debugging probes. Immutable evaluation snapshots and hashes are in `results/evaluations/`; the app's Benchmarks tab shows the saved rows.

| Development diagnostic | Previous BLEU | Balanced BLEU | Previous exact match | Balanced exact match |
|---|---:|---:|---:|---:|
| Balanced validation sample (120) | 64.47 | 68.24 | 34.2% | 42.5% |
| Held-out untimed template sample (63) | 63.87 | 99.46 | 19.0% | 98.4% |
| Time/quantity regression sample (80) | 100.00 | 100.00 | 100.0% | 100.0% |

The balanced validation set guided early stopping; it is not an independent final test. Untimed and time/quantity samples are narrow generator diagnostics, not authentic ASL. No Flores devtest examples were used for training or selection. No claim of real-ASL quality follows from these results. Present tense is a documented default when time is missing; context can support different reconstructions. Novel vocabulary, long sentences and names still require inspection.

Example improvement: `BOY NOT WANT FOOD` changes from “The boy will not want food.” to “The boy does not want food.” `ME GO STORE YESTERDAY` stays “I went to the store yesterday.” These debugging probes are separate from reported metric samples.
