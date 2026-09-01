# Dataset provenance and experimental use

All downloaded, processed, and generated dataset artifacts are stored under
`datasets/`. Active training is English-only.

## ASLG-PC12 — primary experimental dataset

ASLG-PC12 is a large English–ASL-gloss parallel corpus used here for
controlled Gloss-to-Text experimentation. Its glosses were created using a
rule-based transformation process; they are not fully natural, human-produced
ASL and do not establish real-world ASL translation quality.

### Verified source

- Official project page: <https://achrafothman.net/site/english-asl-gloss-parallel-corpus-2012-aslg-pc12/>
- Publication: Achraf Othman and Mohamed Jemni, *English-ASL Gloss Parallel
  Corpus 2012: ASLG-PC12*, LREC sign-language workshop, 2012.
- Accessible dataset used: <https://huggingface.co/datasets/achrafothman/aslg_pc12>
- Pinned Hugging Face revision:
  `cb7cd272db8fcd4004ee04ddf50e194c15ea24d6`
- Revision last modified: 9 January 2024, according to Hugging Face metadata
  queried on 1 September 2026.
- License: CC BY-NC 4.0, declared by both the official project page and the
  Hugging Face dataset metadata.
- Raw splits: one split named `train`.
- Raw rows: 87,710.
- Verified columns: `gloss` and `text`, both strings.
- Active orientation: `gloss` → fluent English `text`.
- Parquet SHA-256:
  `a4661fe4a0ad021884f68be05a496e58fdcf9aeba0c627a2c5f807c4c2accfcc`.
- Local snapshot: `datasets/raw/aslg_pc12/`.

The adapter validates the live schema instead of silently assuming column
names. An ambiguous or changed schema causes preprocessing to stop.

### Cleaning and deterministic splits

The single raw split is cleaned before splitting. Unicode and whitespace are
normalized, BOM characters are removed, empty pairs are removed, and exact and
normalized duplicate **pairs** are removed. Repeated glosses with different
English targets are retained and reported separately.

Actual preprocessing audit, seed 42:

| Item | Count |
|---|---:|
| Raw rows | 87,710 |
| Empty pairs removed | 0 |
| Exact duplicate pairs removed | 6,521 |
| Additional normalized duplicate pairs removed | 172 |
| Final unique pairs | 81,017 |
| Unique gloss strings | 80,941 |
| Glosses with multiple targets | 74 |
| Pairs under ambiguous glosses | 150 |

Since the accessible copy contains no official development/test split, the
81,017 unique pairs are deterministically divided 80/10/10:

| Split | Count |
|---|---:|
| Train | 64,813 |
| Validation | 8,101 |
| Test | 8,103 |

Normalized source-target pair overlap between every split is zero. Stable IDs
are SHA-256-derived before experimentation. `split_manifest.json` records every
ID assignment. Fixed research manifests contain 20,000 train, 2,500 validation,
and 2,500 test IDs so E1–E4 use the same base examples.

Processed files are in `datasets/processed/aslg_pc12/`.

## ASLLRP / NCSLGR — optional external evaluation

ASLLRP/NCSLGR is reserved for optional external evaluation using legally
accessible, human-annotated manual glosses aligned to English translations.
It must not be mixed into ASLG-PC12 training. No text-only aligned artifact has
been acquired yet, and this does not block E0–E4.

## LibriSpeech-Gloss (2026) — related work only

LibriSpeech-Gloss is a recent automatically generated English–gloss dataset
(DOI `10.18760/v25.5453`). It is not experimentally used because the official
gloss files were not publicly accessible during implementation. The public
LibriSpeech-PC English manifests are not treated as gloss data.

## Synthetic telegraphic reconstruction examples

Augmentation is generated only from selected ASLG-PC12 training targets and is
stored separately under `datasets/synthetic/aslg_pc12/`. Starting ratios are
0%, 25%, and 50%; E2/E4 default to 25%. These fragments are controlled
telegraphic reconstruction examples, not authentic ASL.
