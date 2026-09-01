# Local dataset layout

```text
datasets/
├── metadata/aslg_pc12.json
├── raw/aslg_pc12/                   pinned Hugging Face snapshot
├── processed/aslg_pc12/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── stats.json
│   ├── split_manifest.json
│   └── research_*_ids.json
└── synthetic/aslg_pc12/             training-target corruptions only
```

ASLG-PC12 is the only active development dataset. It is a rule-generated
English–ASL-gloss corpus used for controlled text reconstruction experiments.
The task direction is always `gloss -> English`.

Run `python scripts/download_data.py status` to validate the snapshot and
`python scripts/prepare_data.py` to reproduce the cleaned splits.

Older LibriSpeech-PC files may remain locally as provenance from the initial
dataset investigation, but no active loader, config, or experiment uses them.
