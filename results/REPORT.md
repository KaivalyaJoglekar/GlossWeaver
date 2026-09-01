# GlossWeaver execution report

## Dataset preparation

The pinned `achrafothman/aslg_pc12` revision contains 87,710 raw rows with
verified `gloss` and `text` columns. Cleaning retained 81,017 unique pairs and
removed 6,521 exact plus 172 normalized duplicate pairs. The deterministic
seed-42 split contains 64,813 train, 8,101 validation, and 8,103 test records.
Normalized pair overlap is zero. See
`datasets/processed/aslg_pc12/stats.json` and `split_manifest.json`.

## Completed evaluations

E0 was evaluated on the fixed 2,500-record research test manifest. Its saved
metrics are `results/metrics/e0.json` and predictions are
`results/predictions/e0.csv`.

The E1 real-data smoke run completed eight optimizer steps on 512/64 examples,
then evaluated 64 test records. This confirms pipeline functionality, not model
quality; its output still echoes or repeats prompt text.

The E1 development run trained for one epoch on 5,000 records and validated on
500. Training loss was 1.5685054255485535 and validation loss was
0.8605855447905404. On 500 held-out examples it achieved SacreBLEU
48.63559524727392, ROUGE-L 0.7550269945708843, METEOR 0.5770247608405786,
BERTScore F1 0.9228526949882507, and exact match 0.002. These are development
pipeline results, not final research results. Manual inspection confirms
English-oriented outputs but also lexical substitutions and hallucinations.

## Pending research experiments

The fixed 20k/2.5k/2.5k E1–E4 experiments have not yet been executed. Their
metrics and ablations must not be inferred from the development run.
