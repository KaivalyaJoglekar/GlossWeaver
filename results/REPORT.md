# GlossWeaver recovery report

## Outcome

The model-quality failure is fixed. The promoted E4 checkpoint reconstructs all nine isolated challenge examples fluently and without obvious semantic drift. On the three-reference challenge set it scores **BLEU 90.92**, **ROUGE-L 0.956**, **METEOR 0.953**, **BERTScore F1 0.990**, and **content recall 1.000**.

## Root cause

The legacy training pipeline optimized a parliamentary, long-form ASLG-PC12 distribution while the product accepts short household/school fragments. In the 64,813-record ASLG training split, challenge concepts such as MATH and MOTHER were absent, COOK appeared once, and PARK only seven times. The old synthetic corruption also applied suffix stripping to non-verbs, producing malformed tokens. High legacy ASLG scores therefore did not imply product-domain quality.

The active pipeline now uses one shared Python input formatter, an explicit checkpoint registry with no silent fallback, verified `gloss → English` orientation, clean public-source telegraphic augmentation, controlled compositional coverage, and challenge-leak assertions.

## Data and leakage controls

- ASLG-PC12: 87,710 raw rows; 81,017 unique pairs; deterministic seed-42 split of 64,813 / 8,101 / 8,103.
- Reconstruction corpus: 21,785 quality-checked pairs across public LibriSpeech clean text and controlled compositional, condition, and schedule sources.
- Synthetic QA: zero empty fragments, duplicate targets, malformed inputs, exact challenge gloss leaks, or exact challenge-reference leaks.
- The challenge set has nine inputs and three acceptable references per input and was never used for checkpoint selection.
- The reported ASLG and synthetic results use deterministic 500-example test slices; challenge uses all nine examples.

## Pipeline gates

- T5-small 64-pair overfit: BLEU 94.27, ROUGE-L 0.968.
- FLAN-T5-small 64-pair overfit: BLEU 99.55, ROUGE-L 0.999, exact match 0.969.
- A mixed-batch label omission bug was caught because E3 training grammar loss was exactly zero. The invalid run was stopped and archived; the corrected batches have seven labels for every record.
- Grammar-label audit: 200 pairs. Known-operation agreement was 1.00 for ARTICLE, PREPOSITION, AUXILIARY, PRONOUN, and WORD_ORDER, 0.70 for TENSE_ASPECT, and 0.55 for AGREEMENT_INFLECTION. These are weak labels, not gold annotations.

## Experiment comparison

| System | Train examples | ASLG BLEU | Synthetic BLEU | Challenge BLEU | Challenge content recall |
|---|---:|---:|---:|---:|---:|
| E0 copy | 0 | 20.24 | 21.48 | 11.93 | 1.000 |
| E1 T5-small / ASLG | 5,000 | 54.48 | 7.13 | 2.61 | 0.546 |
| E2 FLAN hybrid | 10,000 | 65.45 | 23.91 | 84.40 | 1.000 |
| E3 + grammar loss | 10,000 | 67.02 | 25.11 | 84.40 | 1.000 |
| E4 scaled recipe | 37,388 | 72.93 | 29.55 | 90.92 | 1.000 |

E2 was iteratively continued from its passing baseline checkpoint as uncovered constructions were repaired; its final checkpoint has seven total epochs across staged, leakage-checked data revisions. E3 is the matched auxiliary-objective continuation from E2. E4 scales E3 to all 37,388 available training examples for two epochs. This staged compute difference is reported explicitly and should not be mistaken for a perfectly compute-matched ablation.

## Interpretation

E1 demonstrates the failure: ASLG BLEU is 54.48, but challenge BLEU is only 2.61. Hybrid reconstruction data is the dominant intervention: E2 raises challenge BLEU to 84.40 and content recall to 1.0. E3 provides a modest held-out gain (ASLG 65.45 → 67.02; synthetic 23.91 → 25.11) but no challenge gain, so the grammar objective helps slightly rather than driving the recovery. Scaling to E4 gives the best results on every measured domain.

## Serving parity and limitations

Direct inference, the FastAPI endpoint, and the frontend use the same registry, formatter, beam settings, and checkpoint metadata. E4 is the default only because its artifacts exist; unavailable requested models return explicit errors rather than falling back.

Limitations remain: ASLG-PC12 is rule-generated parliamentary text, controlled data is synthetic, the challenge set is small, weak grammar labels have uneven reliability, and automatic metrics cannot replace human evaluation. The task remains text-only Gloss-to-Text reconstruction; video recognition, pose estimation, sign detection, and speech synthesis are out of scope.
