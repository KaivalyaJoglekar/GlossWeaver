# Implementation verification — 10 October 2026

- Python suite: 36 passed. One dependency deprecation warning from Starlette's HTTPX test-client integration.
- Tiny local T5 and grammar-aware smoke: forward/backward, optimizer update, save/reload, generation passed; no pretrained download required.
- Frontend: oxlint and production Vite build passed.
- Browser: real custom text submitted to the local FastAPI endpoint; exact copy-baseline identity and limitations visible. Copy action and session JSON download verified. Session survives tab changes.
- Benchmarks: actual saved CSV model names, dataset/split and 500-example counts displayed; missing chrF/exact values shown as absent rather than fabricated.
- Responsive checks: desktop and 390 × 844 viewport; benchmark table scrolls inside its container, no horizontal page overflow.
- Data: external raw hashes/counts match acquisition provenance. Processed external dev/devtest are source-isolated. ASLG and synthetic grouped manifests record zero source overlap; all four study train/validation selections independently checked.
- Study budget: seeded replacement sampling gives each recipe 250 updates per validation interval and 1,000 total updates.
- Added CI workflow for Python tests/smoke and frontend lint/build. The remote CI workflow has not run in this session.

Raw external data: 999 dev / 1,016 devtest. Cleaned external data: 994 dev / 1,016 devtest; five overlapping development rows quarantined.
ASLG grouped: 64,789 train / 8,104 validation / 8,100 test; 24 nonlexical sources quarantined.
Synthetic grouped, kept separate: 17,421 / 2,168 / 2,196.

No neural reconstruction quality claim follows from these checks. The checkout contains no trained neural weights. A restored checkpoint or new study training run is needed for fluent reconstruction. Independent human review, authentic-ASL evaluation and multiple-seed experiments remain required for the research paper.

Desktop preview: `results/ui/workbench.jpg`.

## Pretrained testing follow-up

Cached original T5-small and FLAN-T5-small weights were found and exposed separately in the selector. Both were exercised through the real browser/API. FLAN is the default when no project checkpoint exists. These are not the fine-tuned historical E1–E4 models, and neither establishes useful reconstruction quality. Test suite after this change: 38 passed; frontend lint/build passed.

## Completed reconstruction training

Study augmentation and a separate deployment adaptation have now completed. The default `reconstruction-flan` model is available and fixes the reported yesterday example through the real browser/API. There are now trained weights in this checkout. The latest Python suite has 39 passing tests. See `RECONSTRUCTION_TRAINING.md` for actual scores and remaining semantic errors.

## Minimal workbench and balanced continuation

The balanced MPS training run completed 600 optimizer steps. Six comparisons confirm identical sample/data hashes and beam settings between previous and balanced models across three diagnostic slices. The balanced model improves broader validation BLEU 64.47→68.24 and untimed template exact match 19.0%→98.4%, preserving 100% exact match on the 80-example time/quantity regression slice. These are development results, not final authentic-ASL claims.

The default model now selects balanced-flan; the previous model is preserved. Browser/API verification confirms `BOY NOT WANT FOOD` → `The boy does not want food.` with the balanced model selected. Dataset navigation shows real manifest counts. Desktop and 390px mobile layouts were reviewed; page width remained within the viewport, including the horizontally scrollable dataset table. The 39-test suite passed during implementation; all 9 targeted API/evaluation tests passed again after promotion. Frontend lint/build and Python compilation passed. Saved preview: `results/ui/minimal-balanced-workbench.jpg`.
