# GlossWeaver 2.0 implementation audit

The initial repository had extensive uncommitted changes; those were preserved. The starting suite had 39 passing tests. Existing functionality included local checkpoint discovery, shared formatting, grouped data preparation, matched training presets, optimizer scheduling, best generation-loss selection, versioned evaluation snapshots, dataset catalogue, and a basic React workbench.

Implementation phases: (1) tighten lineage isolation and missing-label handling; (2) add advisory semantic checks and unique beam candidates; (3) extend the API without breaking legacy request fields; (4) introduce the application shell and six product areas plus datasets; (5) verify local model inference, integration, responsive behavior, and offline training paths; (6) document deployments and scientific limits.

## Research integrity

Historical E1–E4 are staged experiments with unavailable original weights. New matched presets share FLAN-T5-small initialization and optimizer-step budgets. Deployment adaptations are outside the matched study. No full training was executed for this change; tiny local baseline and grammar-aware training/resume runs were tested. Stronger isolation now detects repeated targets and explicit sentence lineage as well as repeated fragments. Earlier datasets and manifests remain intact. A separate jointly grouped study_v2 dataset was prepared and audited during this implementation; presets now point to it. Existing trained study weights remain attributable to the earlier corpus.

Semantic checks are lexical rules with a small documented synonym set. Name checks use conservative capitalization cues and are unavailable for all-caps inputs. Date-format variants and valid paraphrases may produce notices. These checks cannot establish entailment, participant-role preservation, grammatical validity, or calibrated confidence. Candidate ranking prioritizes fewer high-severity rule notices, then original beam order; it is not a calibrated grammar score. Surface alignment identifies exact article/auxiliary insertions; other edits remain uncategorized.

Grammar heads retain the existing seven weak reconstruction-feature labels. Explicit null, negative, or nonfinite labels and supplied per-label masks now suppress uncertain supervision. This is not a completed tense/polarity classifier ablation. Existing labels must be audited before any claim about linguistic learning.

Training resume starts from a trusted local best checkpoint at the recorded epoch boundary, restoring optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and selection state into a new run directory. MPS RNG state is not restored. CUDA BF16 is used only on compatible hardware; CPU/MPS and other CUDA devices use FP32. Mid-epoch resume is not supported. Resuming a run stopped mid-epoch starts the next epoch; it does not replay the unfinished sampling sequence.

## Deployment limits

Docker targets CPU, runs one Uvicorn worker, serves the built React app, and mounts checkpoints/data/results read-only. Local model caching serializes inference and retains one resident checkpoint. Busy requests return 503 with Retry-After. Browser cancellation stops waiting; it cannot interrupt an already executing PyTorch generation. Reverse-proxy timeouts and request limits should be configured for the deployment. No external deployment or full training is implied.

## Remaining research work

Run all four matched configurations across several seeds on rebuilt, jointly audited corpora; evaluate an untouched test set with matched sample hashes and decoding; measure auxiliary classification quality; gather blinded human fluency/adequacy/hallucination ratings; evaluate entity-aware and learned semantic validators. No research superiority claim follows from existing development diagnostics.
