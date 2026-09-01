# ReconstructG2T — Historical Implementation Plan

> Superseded by `plan.md` and the English-only GlossWeaver 2026 specification.
> The ASLG-PC12, PHOENIX, and multilingual directions below are inactive.

## 1. Project Goal

Build an NLP system that reconstructs fluent, grammatically complete spoken-language sentences from sign-language glosses or other telegraphic/fragmented text.

Example:

```text
Input gloss:
ME GO STORE YESTERDAY

Target:
I went to the store yesterday.
```

The project will focus on the **Gloss-to-Text (G2T)** stage of sign-language translation. Video recognition is intentionally out of scope.

The research version of the project will not simply fine-tune an existing seq2seq model. It will investigate whether **grammar-aware supervision** and **telegraphic-text augmentation** improve reconstruction quality.

Working title:

> **ReconstructG2T: Grammar-Aware Sentence Reconstruction from Sign Language Glosses**

---

## 2. Core Research Idea

### Baseline
Fine-tune a pretrained text-to-text model such as T5 on:

```text
gloss -> fluent sentence
```

### Proposed contribution
Add two mechanisms:

1. **Grammar-aware auxiliary supervision**
   - Automatically identify which grammatical functions must be recovered between a gloss and its fluent target.
   - Train the model to predict these linguistic recovery categories in addition to generating the final sentence.
   - Example categories:
     - ARTICLE
     - PREPOSITION
     - AUXILIARY
     - PRONOUN
     - TENSE_ASPECT
     - AGREEMENT_INFLECTION

2. **Synthetic telegraphic-text augmentation**
   - Start with grammatical English sentences.
   - Remove or simplify selected grammatical information.
   - Train the model to reconstruct the original sentence.
   - Example:

```text
Original:
The students were studying in the library yesterday.

Synthetic fragment:
STUDENTS STUDY LIBRARY YESTERDAY

Recovery labels:
ARTICLE, AUXILIARY, TENSE_ASPECT, PREPOSITION
```

### Optional extension
Context-aware reconstruction:

```text
Previous sentence:
John needed vegetables for dinner.

Current gloss:
HE GO MARKET

Output:
He went to the market.
```

This should be treated as an extension, not the first milestone, because many gloss datasets do not provide reliable document-level context.

---

## 3. Important Research Design Rule — Avoid Target Leakage

Do **not** derive grammar cues from the gold target and then give those gold cues directly to the generator during test-time inference.

That would leak information unavailable in a real system.

Instead:

### Recommended design

During training:

```text
                    -> sentence generation loss
Gloss -> T5 encoder
                    -> grammar-cue prediction loss
```

The grammar labels may be automatically derived from the training target, but they are used only as **supervision** for the auxiliary prediction task.

At inference:

```text
Gloss -> trained model -> fluent sentence
```

No gold target or gold grammar labels are required.

A simpler two-stage alternative is:

```text
Gloss -> cue classifier -> predicted cues
Gloss + predicted cues -> T5 -> sentence
```

The multi-task version is preferred because it is cleaner and avoids error propagation between two separately trained models.

---

## 4. Research Questions

### RQ1
How well does a pretrained seq2seq model reconstruct fluent text from sign-language glosses compared with a simpler baseline?

### RQ2
Does grammar-aware auxiliary supervision improve Gloss-to-Text translation?

### RQ3
Does synthetic telegraphic-text augmentation improve robustness and generalization?

### RQ4
Which grammatical phenomena remain difficult after fine-tuning?

Suggested categories:

- articles
- prepositions
- auxiliaries
- pronouns
- tense/aspect
- agreement/inflection

### Optional RQ5
Does preceding textual context improve reconstruction of ambiguous glosses?

---

## 5. Hypotheses

- **H1:** T5 fine-tuning will substantially outperform a randomly initialized seq2seq/Transformer baseline.
- **H2:** Grammar-aware multi-task training will improve grammatical correctness, especially on function-word recovery.
- **H3:** Synthetic telegraphic augmentation will improve robustness on fragmented inputs that differ from the training gloss distribution.
- **H4:** The full model will improve semantic metrics without sacrificing grammaticality.
- **H5, optional:** Context will primarily help pronouns, lexical ambiguity, tense interpretation, and discourse continuity.

---

## 6. Dataset Plan

### Primary dataset — ASLG-PC12

Use ASLG-PC12 for the main English Gloss-to-Text experiment.

Why:
- ASL gloss to English pairing.
- Much larger than PHOENIX-2014T.
- Suitable for text-only sequence-to-sequence experimentation.

Important caveat:
- ASLG-PC12 contains automatically/rule-derived material and should not be described as equivalent to manually produced natural signing data.
- This limitation must be stated in the paper.

### Secondary benchmark — RWTH-PHOENIX-Weather 2014T

Optional but valuable.

Why:
- Standard sign-language translation benchmark.
- Contains gloss and spoken-language text.
- Used by Gloss2Text.
- Allows comparison with a well-known benchmark.

Caveat:
- German, not English.
- Much smaller.
- Weather-domain language is narrow.

### Synthetic dataset

Generate additional pairs from fluent English sentences.

Possible sources:
- target sentences from ASLG-PC12
- WikiText or another permitted general English corpus
- selected public text corpus with a clear license

For the first version, synthetic corruption of the **existing training targets** is enough.

Never generate augmented examples from test targets and include them in training.

---

## 7. Data Splitting

Use fixed, reproducible splits.

Preferred order:

1. Use official train/dev/test splits if supplied.
2. If unavailable:
   - train: 80%
   - validation: 10%
   - test: 10%
3. Use a fixed random seed, e.g. `42`.
4. Check for duplicate glosses and duplicate targets across splits.
5. Remove exact cross-split duplicates where necessary.

Save:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Suggested columns:

```text
id
gloss
target
source
article_label
preposition_label
auxiliary_label
pronoun_label
tense_aspect_label
agreement_label
```

---

## 8. Data Analysis Before Training

Create `notebooks/01_data_analysis.ipynb`.

Analyse:

- number of examples
- missing values
- duplicates
- gloss length distribution
- target length distribution
- gloss vocabulary size
- target vocabulary size
- ratio of gloss length to target length
- top gloss tokens
- top target tokens
- frequency of grammar-recovery labels
- unusual symbols / annotations
- sentence-length outliers

Produce at least these plots:

1. Gloss length histogram
2. Target length histogram
3. Grammar-label frequency chart
4. Gloss vs target length scatter plot

Save plot images into:

```text
results/figures/
```

---

## 9. Preprocessing

Create `src/preprocess.py`.

Tasks:

- Unicode normalization
- whitespace normalization
- preserve meaningful gloss notation
- strip empty samples
- optional punctuation normalization on target side
- duplicate detection
- dataset splitting
- save processed CSV/JSONL

Avoid aggressive lowercasing until dataset inspection is complete because gloss capitalization may carry formatting conventions.

Do not remove annotation symbols without understanding what they mean.

---

## 10. Grammar Label Extraction

Create:

```text
src/grammar_labels.py
```

Goal:

Automatically create weak supervision labels indicating which grammatical functions are present in the fluent target but absent or underspecified in the gloss.

Recommended tools:

- spaCy for POS/dependency information
- simple deterministic rules
- optional lemmatization

### Suggested labels

#### ARTICLE
Target includes determiners such as `a`, `an`, `the` not represented in gloss.

#### PREPOSITION
Target contains an adposition such as `to`, `in`, `on`, `at`, `from`, `with` that is missing or not lexicalized in the gloss.

#### AUXILIARY
Target requires words such as:
- is
- are
- was
- were
- have
- has
- do
- did
- will

#### PRONOUN
Pronoun normalization or insertion is needed.

#### TENSE_ASPECT
Target verb morphology carries tense/aspect not explicit in the gloss.

#### AGREEMENT_INFLECTION
Plural, third-person agreement, possessive morphology, etc.

### Important methodological note

These are **engineering recovery labels**, not claims that sign languages simply "omit grammar."

Sign languages have their own grammar, and gloss notation is an imperfect textual representation of it.

Use wording such as:

> linguistic information that must be recovered when mapping the available gloss representation to fluent spoken-language text

instead of:

> grammar missing from sign language

---

## 11. Synthetic Telegraphic Corruption

Create:

```text
src/corruption.py
```

Implement controllable transformations.

### C1 — Article deletion

```text
The boy reads a book.
-> BOY READ BOOK
```

### C2 — Preposition deletion

```text
She went to the office.
-> SHE GO OFFICE
```

### C3 — Auxiliary deletion

```text
They were playing outside.
-> THEY PLAY OUTSIDE
```

### C4 — Verb simplification

```text
He walked home.
-> HE WALK HOME
```

### C5 — Pronoun manipulation

Use carefully. Do not always delete pronouns.

### C6 — Punctuation removal

```text
Where are you going?
-> WHERE YOU GO
```

### C7 — Controlled word deletion

Delete only selected low-information/function tokens, not arbitrary content words by default.

### C8 — Optional mild reordering

Use sparingly because artificial reordering can create unrealistic "sign language" data.

Store the operations used:

```json
{
  "fragment": "STUDENT STUDY LIBRARY YESTERDAY",
  "target": "The student was studying in the library yesterday.",
  "operations": [
    "ARTICLE",
    "AUXILIARY",
    "TENSE_ASPECT",
    "PREPOSITION"
  ]
}
```

---

## 12. Baselines

### Baseline 0 — Copy / heuristic baseline

Return the gloss with simple casing and punctuation.

Purpose:
- establishes a trivial lower bound

### Baseline 1 — Vanilla encoder-decoder Transformer

Optional if time allows.

Purpose:
- show the benefit of pretraining

### Baseline 2 — T5-small

Primary baseline.

Input format:

```text
reconstruct gloss: ME GO STORE YESTERDAY
```

Target:

```text
I went to the store yesterday.
```

### Baseline 3 — FLAN-T5-small or FLAN-T5-base

If compute permits.

Prompt:

```text
Convert the following sign-language gloss into a fluent sentence:
ME GO STORE YESTERDAY
```

Do not compare an excessive number of models. A clean experimental comparison is more valuable than ten partially tuned models.

---

## 13. Proposed Model — Grammar-Aware Multi-Task T5

Suggested name:

> **GA-T5** — Grammar-Aware T5

### Architecture

```text
                     +--------------------+
Gloss -------------->| T5 Encoder         |
                     +--------------------+
                        |              |
                        |              +--> Grammar prediction head
                        |                    -> multi-label probabilities
                        |
                        +--> T5 Decoder
                             -> fluent sentence
```

### Training objective

```text
L_total = L_generation + lambda * L_grammar
```

Where:

- `L_generation` = standard seq2seq cross-entropy loss
- `L_grammar` = binary cross-entropy over grammar-recovery labels
- `lambda` = tunable weight

Try:

```text
lambda ∈ {0.1, 0.25, 0.5, 1.0}
```

Tune only using the validation set.

### Why this is useful

The auxiliary objective encourages the encoder to represent the linguistic transformations required for reconstruction without exposing gold labels at inference time.

---

## 14. Implementation Modules

Recommended repository:

```text
ReconstructG2T/
│
├── README.md
├── plan.md
├── project.md
├── research_paper.md
├── requirements.txt
├── config/
│   ├── baseline.yaml
│   ├── grammar_t5.yaml
│   └── augmentation.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_error_analysis.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── grammar_labels.py
│   ├── corruption.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── inference.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_corruption.py
│   └── test_metrics.py
│
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── tables/
│
└── paper/
    ├── figures/
    ├── tables/
    └── notes/
```

---

## 15. Training Pipeline

### Stage A — Data

```text
raw pairs
-> clean
-> split
-> label grammar recovery
-> synthetic augmentation
-> final train/val/test files
```

### Stage B — Baseline

Train T5-small on original pairs only.

Save:
- validation loss
- best checkpoint
- test predictions
- metrics

### Stage C — Synthetic augmentation

Train same T5 configuration with:

```text
real training pairs + synthetic fragments
```

Keep validation/test real and untouched.

### Stage D — Grammar-aware model

Train GA-T5 with the multi-task loss.

### Stage E — Full model

```text
GA-T5 + synthetic augmentation
```

### Stage F — Context extension

Only after the main experiments are stable.

---

## 16. Recommended Hyperparameter Starting Point

For T5-small:

```text
model: t5-small
max_source_length: 128
max_target_length: 128
batch_size: 8 or 16
gradient_accumulation_steps: as required
learning_rate: 3e-4 to start
epochs: 3-8
weight_decay: 0.01
warmup_ratio: 0.05
beam_size: 4
early_stopping: validation metric/loss
seed: 42
```

Do not tune on the test set.

Use mixed precision if GPU supports it.

Keep all experimental configurations in YAML or JSON.

---

## 17. Evaluation

### Translation/reconstruction metrics

Use at minimum:

- SacreBLEU
- ROUGE-L
- METEOR
- BERTScore

Optional:
- chrF
- COMET if practical and linguistically appropriate

### Grammar-focused metrics

Implement category-specific analysis:

- article recovery F1
- preposition recovery F1
- auxiliary recovery F1
- tense/aspect recovery score
- agreement recovery score

These may require rules rather than perfect automatic evaluation.

Report them as diagnostic metrics, not absolute linguistic truth.

### Human evaluation

Strongly recommended for the paper.

Sample 50-100 predictions and ask evaluators to score:

1. grammaticality
2. fluency
3. meaning preservation
4. hallucination / unsupported information

Suggested scale:

```text
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent
```

If human evaluation is not possible, conduct careful manual error analysis.

---

## 18. Main Experimental Table

Prepare a table like:

| Model | BLEU | ROUGE-L | METEOR | BERTScore | Grammar Score |
|---|---:|---:|---:|---:|---:|
| Copy baseline |  |  |  |  |  |
| T5-small |  |  |  |  |  |
| T5 + Synthetic |  |  |  |  |  |
| GA-T5 |  |  |  |  |  |
| **GA-T5 + Synthetic** |  |  |  |  |  |

Do not fill values until experiments are actually run.

---

## 19. Ablation Study

Required if you want the project to read like research.

| Experiment | Synthetic Data | Grammar Loss | Context |
|---|---|---|---|
| T5 | No | No | No |
| T5 + Aug | Yes | No | No |
| GA-T5 | No | Yes | No |
| Full | Yes | Yes | No |
| Full + Context | Yes | Yes | Yes |

Additional ablation:

```text
lambda = 0.1
lambda = 0.25
lambda = 0.5
lambda = 1.0
```

Do not run every possible combination if compute is limited.

---

## 20. Error Analysis

Create error categories:

- missing function word
- wrong tense
- wrong pronoun
- lexical substitution
- word order
- repetition
- hallucinated content
- incomplete output
- named entity corruption
- semantically valid paraphrase penalized by reference metric

For each model, manually inspect at least 50-100 examples.

Example table:

| Gloss | Reference | Baseline | Proposed | Error Analysis |
|---|---|---|---|---|
| ME GO STORE YESTERDAY | I went to the store yesterday. | I go store yesterday. | I went to the store yesterday. | tense/article/preposition fixed |

Include both success cases and failure cases.

---

## 21. Reproducibility Requirements

Every experiment must record:

```text
model checkpoint
dataset version
train/val/test split
seed
learning rate
batch size
epochs
max lengths
beam size
augmentation ratio
grammar-loss lambda
library versions
hardware
runtime
```

Save predictions for every reported model.

Never report a metric if the matching prediction file/checkpoint cannot be traced.

---

## 22. Research Integrity Checklist

Before calling a method "novel":

- search ACL Anthology
- search arXiv
- search Google Scholar
- compare against grammar-guided generation work
- compare against denoising/text-reconstruction literature
- compare against current G2T literature

Safer language before the literature review is complete:

> "We propose and evaluate..."

rather than:

> "We are the first..."

Do not claim direct accessibility improvements without user studies involving Deaf/sign-language communities.

---

## 23. Milestones

### Milestone 1 — Repository + dataset
Deliverables:
- repository structure
- environment
- raw dataset
- preprocessing script
- EDA notebook

### Milestone 2 — Baseline
Deliverables:
- T5 training
- test predictions
- BLEU/ROUGE/METEOR/BERTScore
- baseline table

### Milestone 3 — Synthetic corruption
Deliverables:
- corruption module
- unit tests
- augmented dataset
- augmentation experiment

### Milestone 4 — Grammar supervision
Deliverables:
- grammar label extraction
- label statistics
- GA-T5 model
- grammar-aware experiment

### Milestone 5 — Ablation + error analysis
Deliverables:
- final comparison
- ablation table
- qualitative examples
- failure taxonomy

### Milestone 6 — Research paper
Deliverables:
- abstract
- introduction
- related work
- methods
- experimental setup
- results
- limitations
- conclusion

### Milestone 7 — Final demo
Demo should accept:

```text
ME GO STORE YESTERDAY
```

and display:

```text
Reconstructed:
I went to the store yesterday.
```

Optionally also show:
- beam alternatives
- predicted grammatical recovery categories
- confidence values

---

## 24. Minimal Viable Project vs Full Research Project

### MVP

Must have:
- ASLG-PC12 or another valid gloss/text dataset
- T5-small baseline
- train/validation/test evaluation
- inference script
- BLEU + ROUGE + METEOR
- qualitative examples

### Research-ready version

Add:
- grammar recovery labels
- multi-task GA-T5
- synthetic telegraphic augmentation
- BERTScore
- ablation study
- error analysis
- statistical or multi-seed validation if compute permits

### Stretch version

Add:
- real previous-sentence context
- second dataset
- human evaluation
- multilingual experiment

---

## 25. Final Definition of Done

The project is complete only when all of the following exist:

- reproducible dataset preparation
- working baseline
- proposed model
- saved predictions
- quantitative comparison
- ablation study
- grammar-specific analysis
- failure examples
- inference/demo script
- complete README
- research paper draft
- references and limitations
