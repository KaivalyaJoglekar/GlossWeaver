# ReconstructG2T — Historical Project Technical Specification

> Superseded for implementation by the English-only GlossWeaver 2026 brief,
> `README.md`, `DATASETS.md`, and `plan.md`. References below to ASLG-PC12,
> PHOENIX, multilingual work, or the older repository layout are retained only
> as planning history and are not active implementation requirements.

## 1. Project Name

**ReconstructG2T: Grammar-Aware Sentence Reconstruction from Sign Language Glosses**

---

## 2. Problem Statement

Sign-language translation systems often represent signed content using an intermediate symbolic form called a **gloss**. Gloss sequences can differ substantially from fluent spoken-language sentences in word order, morphology, articles, auxiliaries, prepositions, tense realization, and other grammatical information.

For example:

```text
Gloss:
ME GO STORE YESTERDAY

Fluent English:
I went to the store yesterday.
```

The objective of this project is to build a text-to-text NLP system that maps a sign-language gloss or telegraphic fragment to a fluent sentence while preserving the intended semantic content.

Formally:

```text
Given:
G = (g1, g2, ..., gm)

Generate:
Y = (y1, y2, ..., yn)
```

where:

- `G` is a gloss or telegraphic token sequence
- `Y` is the corresponding fluent spoken-language sentence

The implementation will focus on the **Gloss-to-Text** stage only.

---

## 3. Scope

### In scope

- gloss-to-text translation
- fragmented sentence reconstruction
- pretrained encoder-decoder transformers
- grammar-aware auxiliary training
- synthetic telegraphic augmentation
- automatic evaluation
- error analysis
- optional previous-sentence context
- reproducible experiment pipeline

### Out of scope

- sign-language video recognition
- pose estimation
- hand tracking
- sign detection
- speech synthesis
- real-time video translation
- claims about replacing professional human interpreters

---

## 4. Baseline Literature

The implementation is inspired by existing Gloss-to-Text research but must not simply reproduce it.

### Gloss2Text — EMNLP Findings 2024

Fayyazsanavi et al. focus on the Gloss-to-Text stage and combine:
- pretrained language models
- paraphrasing
- back-translation/pseudo-gloss augmentation
- Semantically Aware Label Smoothing (SALS)

Official paper:
https://aclanthology.org/2024.findings-emnlp.947/

Official implementation:
https://github.com/pooyafayyaz/Gloss2Text

The repository uses PHOENIX-2014T and includes scripts for paraphrasing, text-to-gloss back-translation, NLLB-based training, LoRA, evaluation, and SALS.

### Contextual SLT

Recent work has also investigated contextual cues such as previous sentence translations in sign-language translation:

**Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues**, CVPR 2025.

https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html

Context is relevant to this project but will be implemented only after the core reconstruction experiments.

---

## 5. Proposed Contribution

### Main contribution

A **Grammar-Aware Multi-Task Sequence-to-Sequence model** for Gloss-to-Text reconstruction.

The model performs two tasks:

1. reconstruct the fluent sentence
2. predict which linguistic recovery categories are required

The auxiliary task encourages the encoder to learn structures associated with grammatical reconstruction.

### Auxiliary labels

Suggested label vector:

```text
[
  ARTICLE,
  PREPOSITION,
  AUXILIARY,
  PRONOUN,
  TENSE_ASPECT,
  AGREEMENT_INFLECTION
]
```

For an example such as:

```text
Gloss:
BOY GO SCHOOL YESTERDAY

Target:
The boy went to school yesterday.
```

a weak target vector could be:

```text
ARTICLE = 1
PREPOSITION = 1
AUXILIARY = 0
PRONOUN = 0
TENSE_ASPECT = 1
AGREEMENT_INFLECTION = 0
```

These labels are used as **training supervision**, not as gold test-time input.

---

## 6. Proposed Architecture

### 6.1 Base encoder-decoder

Use Hugging Face T5.

Recommended starting model:

```text
t5-small
```

Potential secondary model:

```text
google/flan-t5-small
```

or, if GPU memory allows:

```text
google/flan-t5-base
```

### 6.2 Multi-task extension

Architecture:

```text
Input gloss
   |
   v
+----------------+
| T5 Tokenizer   |
+----------------+
   |
   v
+----------------+
| T5 Encoder     |
+----------------+
   |          |
   |          +----------------------+
   |                                 |
   v                                 v
T5 Decoder                    Pooling Layer
   |                                 |
   v                                 v
Fluent sentence               Linear classifier
                                     |
                                     v
                              Grammar labels
```

### 6.3 Loss

Generation loss:

```text
L_gen = token-level cross entropy
```

Grammar loss:

```text
L_grammar = binary cross entropy with logits
```

Combined loss:

```text
L_total = L_gen + lambda * L_grammar
```

Default:

```text
lambda = 0.25
```

Tune on validation data.

---

## 7. Input Formats

### Baseline

```text
reconstruct gloss: ME GO STORE YESTERDAY
```

### Optional task prefix

```text
gloss to text: ME GO STORE YESTERDAY
```

Use one format consistently for a given experiment.

### Optional context model

```text
previous: John needed vegetables for dinner.
gloss: HE GO MARKET
reconstruct:
```

Do not train a context model unless context is genuinely available or the paper clearly labels the setup as synthetic.

---

## 8. Dataset Interface

Internal example schema:

```python
{
    "id": "sample_000001",
    "gloss": "ME GO STORE YESTERDAY",
    "target": "I went to the store yesterday.",
    "source": "aslg_pc12",
    "grammar_labels": {
        "article": 1,
        "preposition": 1,
        "auxiliary": 0,
        "pronoun": 1,
        "tense_aspect": 1,
        "agreement_inflection": 0
    }
}
```

Serialized CSV version:

```text
id,gloss,target,source,article,preposition,auxiliary,pronoun,tense_aspect,agreement
```

JSONL is preferable when metadata becomes more complex.

---

## 9. Repository Structure

```text
ReconstructG2T/
│
├── README.md
├── plan.md
├── project.md
├── research_paper.md
├── requirements.txt
│
├── config/
│   ├── t5_baseline.yaml
│   ├── t5_augmented.yaml
│   ├── gat5.yaml
│   └── gat5_augmented.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── synthetic/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── grammar_labels.py
│   ├── corruption.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_grammar_labels.py
│   └── test_corruption.py
│
├── checkpoints/
│
└── results/
    ├── metrics/
    ├── predictions/
    ├── figures/
    └── tables/
```

---

## 10. Environment

Suggested dependencies:

```text
python>=3.10
torch
transformers
datasets
accelerate
sentencepiece
evaluate
sacrebleu
rouge-score
nltk
bert-score
spacy
pandas
numpy
scikit-learn
matplotlib
tqdm
pyyaml
```

Download a suitable spaCy English pipeline if using spaCy-based label extraction.

GPU is strongly recommended for fine-tuning.

---

## 11. `src/preprocess.py`

Responsibilities:

```text
load raw data
validate required columns
normalize text
remove empty pairs
detect duplicates
produce deterministic splits
save processed dataset
print dataset statistics
```

Pseudo-interface:

```bash
python -m src.preprocess \
  --input data/raw/aslg.csv \
  --output_dir data/processed \
  --seed 42
```

Validation rules:

- gloss cannot be empty
- target cannot be empty
- minimum token length configurable
- maximum sequence length configurable
- duplicates logged
- no test examples used in training augmentation

---

## 12. `src/grammar_labels.py`

Responsibilities:

1. tokenize target
2. POS/dependency tag target
3. normalize gloss tokens
4. compare lexical/morphological information
5. output weak multi-label supervision

Suggested functions:

```python
extract_grammar_labels(gloss: str, target: str) -> dict
normalize_gloss(gloss: str) -> list[str]
target_features(target: str) -> dict
batch_label_dataset(df) -> DataFrame
```

### Weak-label strategy

Use transparent deterministic rules first.

Example target:

```text
The girl was sitting in the room.
```

Gloss:

```text
GIRL SIT ROOM
```

Likely weak labels:

```text
ARTICLE
AUXILIARY
TENSE_ASPECT
PREPOSITION
```

Do not present the heuristic labels as gold linguistic annotations.

---

## 13. `src/corruption.py`

Purpose:

Generate telegraphic training examples from fluent text.

Function:

```python
corrupt_sentence(
    sentence,
    drop_articles=True,
    drop_prepositions=True,
    drop_auxiliaries=True,
    simplify_verbs=True,
    remove_punctuation=True,
    corruption_probability=...
)
```

Return:

```python
{
    "fragment": ...,
    "target": ...,
    "operations": [...]
}
```

### Quality controls

- never output an empty fragment
- retain key content words
- cap corruption severity
- log transformation distribution
- manually inspect 100 random generated examples

Avoid claiming the synthetic fragments are authentic ASL gloss.

Correct terminology:

> gloss-like / telegraphic synthetic fragments

---

## 14. `src/dataset.py`

Implement Hugging Face-compatible dataset preparation.

Input:

```text
gloss
target
grammar label vector
```

Output tensors:

```text
input_ids
attention_mask
labels
grammar_labels
```

Tokenization:

```python
tokenizer(
    input_text,
    max_length=max_source_length,
    truncation=True
)
```

Target tokenization:

```python
tokenizer(
    text_target=target,
    max_length=max_target_length,
    truncation=True
)
```

---

## 15. `src/model.py`

Implement:

```python
class GrammarAwareT5(...)
```

Suggested behavior:

1. call T5 encoder
2. obtain hidden states
3. mask-aware mean pool encoder states
4. feed pooled representation into a multi-label classifier
5. run normal T5 decoding
6. calculate generation loss
7. calculate grammar BCE loss
8. return combined loss and component losses

Output should include:

```text
loss
generation_loss
grammar_loss
logits
grammar_logits
```

The model should still support `.generate()`.

---

## 16. `src/train.py`

Use either:

- Hugging Face `Seq2SeqTrainer`, extended for auxiliary loss
- custom PyTorch training loop

Recommended: start with `Seq2SeqTrainer` for baseline, then subclass/customize once the baseline is stable.

Command:

```bash
python -m src.train --config config/t5_baseline.yaml
```

Config fields:

```yaml
experiment_name: t5_baseline
model_name: t5-small
seed: 42

data:
  train: data/processed/train.jsonl
  val: data/processed/val.jsonl

training:
  epochs: 5
  learning_rate: 0.0003
  batch_size: 8
  gradient_accumulation_steps: 2
  weight_decay: 0.01
  warmup_ratio: 0.05
  fp16: true

generation:
  num_beams: 4
  max_new_tokens: 128
```

For grammar-aware model:

```yaml
grammar:
  enabled: true
  num_labels: 6
  lambda: 0.25
```

---

## 17. `src/metrics.py`

Implement a single evaluation interface.

Return:

```python
{
    "bleu": ...,
    "rouge_l": ...,
    "meteor": ...,
    "bertscore_f1": ...,
    "exact_match": ...
}
```

Recommended:
- SacreBLEU instead of custom BLEU
- ROUGE-L
- NLTK/evaluate METEOR
- BERTScore

Store metric configuration/version in result metadata.

---

## 18. Grammar Diagnostic Metrics

Separate from generation metrics.

Potential automatic diagnostics:

```text
article_recovery_precision
article_recovery_recall
article_recovery_f1

preposition_recovery_f1
auxiliary_recovery_f1
pronoun_recovery_f1
tense_recovery_score
```

Implementation strategy:

1. POS/dependency parse reference and prediction.
2. extract grammatical feature sets.
3. compare category-specific feature presence.

Because exact tense evaluation can be difficult, describe these as **diagnostic heuristics**.

---

## 19. `src/evaluate.py`

Command:

```bash
python -m src.evaluate \
  --checkpoint checkpoints/t5_baseline \
  --test_file data/processed/test.jsonl \
  --output results/predictions/t5_baseline.csv
```

Prediction CSV:

```text
id
gloss
reference
prediction
source
```

Add grammar predictions for GA-T5:

```text
pred_article
pred_preposition
pred_auxiliary
pred_pronoun
pred_tense
pred_agreement
```

---

## 20. `src/inference.py`

CLI:

```bash
python -m src.inference \
  --checkpoint checkpoints/gat5_full \
  --text "ME GO STORE YESTERDAY"
```

Output:

```text
Input:
ME GO STORE YESTERDAY

Reconstructed:
I went to the store yesterday.

Predicted recovery categories:
TENSE_ASPECT, ARTICLE, PREPOSITION
```

Optional interactive mode:

```bash
python -m src.inference --interactive
```

---

## 21. Experiment Matrix

### E0 — Copy baseline

No training.

### E1 — T5 baseline

```text
real gloss pairs only
grammar objective: no
synthetic augmentation: no
```

### E2 — T5 + augmentation

```text
real + synthetic
grammar objective: no
```

### E3 — GA-T5

```text
real pairs
grammar objective: yes
```

### E4 — Full proposed model

```text
real + synthetic
grammar objective: yes
```

### E5 — Optional context

```text
full proposed model + previous text
```

Main comparison:

```text
E1 vs E2 -> augmentation effect
E1 vs E3 -> grammar supervision effect
E1 vs E4 -> complete method effect
E4 vs E5 -> context effect
```

---

## 22. Multi-Seed Evaluation

If resources permit, run important models with:

```text
seed = 13
seed = 42
seed = 87
```

Report:

```text
mean ± standard deviation
```

At minimum, repeat:
- T5 baseline
- full proposed model

This provides stronger evidence than a single lucky training run.

---

## 23. Statistical Testing

Optional but useful.

Use paired bootstrap resampling or another appropriate paired test on test-set predictions.

Compare:

```text
baseline vs proposed
```

Do not claim an improvement is statistically significant unless it was actually tested.

---

## 24. Synthetic Augmentation Ratios

Try a limited study:

```text
0%
25%
50%
100%
```

where percentage means number of synthetic pairs relative to real training examples.

Pick the best setting using validation performance.

Do not use test data to choose the ratio.

---

## 25. Model Selection

Primary selection metric:

- BERTScore or BLEU depending on instructor expectation

Recommended practical rule:

```text
Select checkpoint using validation loss or validation BLEU,
then report all metrics on the untouched test set.
```

If grammar correctness is a primary contribution, also inspect grammar diagnostic scores during development.

---

## 26. Qualitative Evaluation

Create:

```text
results/predictions/error_analysis.csv
```

Columns:

```text
id
gloss
reference
baseline_prediction
proposed_prediction
error_category
notes
```

Sample categories:

```text
ARTICLE
PREPOSITION
TENSE
PRONOUN
WORD_ORDER
LEXICAL
HALLUCINATION
INCOMPLETE
PARAPHRASE
OTHER
```

Use the same annotation criteria consistently.

---

## 27. Demo Options

### Simple CLI
Recommended first.

### Streamlit
Optional UI:

Input box:
```text
ME GO STORE YESTERDAY
```

Output card:
```text
I went to the store yesterday.
```

Additional section:
```text
Predicted grammatical recovery:
- tense/aspect
- article
- preposition
```

Do not spend substantial time on UI before experiments are complete.

---

## 28. Expected Challenges

### Dataset authenticity
Some large text-only gloss datasets can contain synthetic/rule-generated glosses.

Mitigation:
- state this clearly
- optionally validate on PHOENIX
- avoid generalizing to all real-world sign-language translation

### Weak grammar labels
Automatic labels will contain noise.

Mitigation:
- document rules
- manually inspect a sample
- treat labels as weak supervision
- run ablations

### Hallucination
A language model can produce fluent but unsupported content.

Mitigation:
- compare semantic fidelity
- manually annotate hallucination
- use conservative generation settings

### Evaluation ambiguity
There may be multiple valid translations.

Mitigation:
- use semantic metrics
- qualitative examples
- human/manual evaluation

### Compute constraints
Large LLMs may be unnecessary.

Mitigation:
- T5-small first
- FLAN-T5-small/base only if needed
- use gradient accumulation and mixed precision

---

## 29. Research Paper Claims That Are Safe

After successful experiments:

> We evaluate whether auxiliary grammar-recovery supervision improves Gloss-to-Text generation.

> We introduce a controlled telegraphic corruption pipeline for data augmentation.

> Our ablation study measures the individual contribution of grammar supervision and synthetic augmentation.

Do not write without evidence:

> Our model understands sign language grammar.

> Our model solves sign-language translation.

> Our approach is the first grammar-aware Gloss-to-Text method.

> The system is ready for real-world Deaf communication.

---

## 30. Final Deliverables

Code:
- preprocessing
- augmentation
- grammar labeling
- baseline training
- proposed model
- evaluation
- inference

Data artifacts:
- clean splits
- synthetic training set
- grammar labels

Results:
- metric tables
- ablation table
- prediction files
- error analysis
- figures

Documentation:
- README
- plan
- project specification
- research paper draft

Demo:
- CLI or Streamlit reconstruction interface

---

## 31. Reference Starting Points

### Gloss2Text
Paper:
https://aclanthology.org/2024.findings-emnlp.947/

Code:
https://github.com/pooyafayyaz/Gloss2Text

### RWTH-PHOENIX-Weather 2014T
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

### Contextual cues in SLT
https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html

### ASLG-PC12 background
Search for the original ASLG-PC12 corpus publication and verify the exact redistribution/license terms before committing dataset files to GitHub.
