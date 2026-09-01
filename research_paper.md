# GlossWeaver: paper blueprint

## Abstract

To be written after experiments. It must describe ASLG-PC12 as a rule-generated
English–ASL-gloss parallel corpus and include only results traceable to saved
metric and prediction files.

## Research question

Does grammar-aware auxiliary supervision, combined with controlled
telegraphic-text augmentation, improve reconstruction of fluent English from
gloss-style input compared with a standard T5 baseline?

## Method

Compare E0–E4 under matched T5-small configurations. Derive six weak recovery
labels from training targets, supervise an encoder-side multi-label head, and
never provide gold labels to the generator at evaluation or inference. Generate
synthetic fragments from training targets only and record every transformation.

## Dataset

Although ASLG-PC12 is an earlier corpus, it remains useful for controlled
large-scale Gloss-to-Text experimentation because it provides aligned English
and ASL-gloss sequences. Document its CC BY-NC 4.0 license, pinned Hugging Face
revision, rule-generated nature, actual cleaning statistics, deterministic
splits, and zero-overlap leakage audit. LibriSpeech-Gloss is recent related work
but is not experimentally used because its gloss files were inaccessible.

## Evaluation

Report SacreBLEU, ROUGE-L, METEOR, BERTScore F1, exact match, six heuristic
grammar diagnostics, qualitative errors, and possible hallucinations. Use
validation data for hyperparameters and untouched test data exactly once per
final configuration. Multiple seeds are preferred when compute permits.

## Results

Results pending experiment execution. Do not use phrases such as “state of the
art,” “first,” “significantly better,” or “solves sign-language translation”
without independent evidence.

## Limitations and responsible framing

Discuss synthetic/rule-derived glosses, domain mismatch, lack of native-signer
validation, omitted non-manual markers, weak-label noise, metric limitations,
hallucination risk, and the absence of user studies.
