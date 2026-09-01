# GlossWeaver implementation plan

This plan supersedes the older `plan (1).md`, whose ASLG-PC12 and multilingual
assumptions are retained only as historical context.

1. Pin and document the accessible ASLG-PC12 source and license.
2. Clean ASLG-PC12 before deterministic splitting and audit leakage.
3. Freeze one 20k/2.5k/2.5k research subset for every E1–E4 comparison.
4. Implement E0 through E4 with configuration parity.
5. Run a local, tiny-model smoke test before any full experiment.
6. Produce metrics, predictions, diagnostics, EDA, and error-analysis artifacts
   only from completed executions.

LibriSpeech-Gloss is retained as 2026 related work only because its official
gloss files were inaccessible. It is not an active training dependency.
