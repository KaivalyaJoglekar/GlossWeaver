"""Local research API; neural availability and saved results are explicit."""
from __future__ import annotations

import logging
import json
import os
import sys
from pathlib import Path
from threading import Lock
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from glossweaver.inference import Reconstructor
from glossweaver.model_registry import MODELS, default_available_model, get_model, resolve_checkpoint
from glossweaver.settings import ROOT
from glossweaver.semantics import validate_meaning, align_changes

logger = logging.getLogger(__name__)
app = FastAPI(title="GlossWeaver API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("GLOSSWEAVER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)
class RequestSizeLimit:
    """Bound JSON bodies even when clients omit Content-Length."""
    def __init__(self, app, maximum: int = 32768):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.maximum:
                return await JSONResponse({"detail": "Request body exceeds 32 KiB"}, status_code=413)(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, replay, send)


app.add_middleware(RequestSizeLimit)
FRONTEND_DIST = PROJECT_ROOT / "frontend/dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# One resident model; serialize loading/inference to avoid concurrent GPU spikes.
_reconstructors: dict[str, tuple[tuple, Reconstructor]] = {}
_model_lock = Lock()


def _checkpoint_stamp(model_key: str) -> tuple:
    path = get_model(model_key).checkpoint_path
    if path is None:
        return ()
    path = resolve_checkpoint(path)
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(path.glob("*")) if p.is_file())


def _get_reconstructor(model_key: str) -> Reconstructor:
    stamp = _checkpoint_stamp(model_key)
    cached = _reconstructors.get(model_key)
    if cached is None or cached[0] != stamp:
        _reconstructors.clear()
        cached = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _reconstructors[model_key] = (stamp, Reconstructor(model_key, device="auto"))
    return _reconstructors[model_key][1]


def public_model(info) -> dict:
    return {
        "model_name": info.display_name, "experiment_name": info.experiment_name,
        "model_type": info.model_type, "grammar_enabled": info.grammar_enabled,
        "training_data_description": info.training_data_description,
    }


class InferRequest(BaseModel):
    gloss: str = Field(min_length=1, max_length=2000, validation_alias=AliasChoices("gloss", "text"))
    checkpoint: str = Field(min_length=1, max_length=100, validation_alias=AliasChoices("checkpoint", "model_key"))
    num_beams: int = Field(default=4, ge=1, le=10)
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    alternatives: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def valid_candidates(self):
        if self.alternatives > self.num_beams:
            raise ValueError("Alternative count cannot exceed beam width")
        return self

    @field_validator("gloss")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter some text to reconstruct")
        return value.strip()


class InferResponse(BaseModel):
    gloss: str
    reconstruction: str
    grammar: dict[str, float]
    grammar_active: list[str]
    checkpoint: str
    grammar_aware: bool
    debug: dict[str, object]
    elapsed_ms: float
    notices: list[str]
    semantic: dict[str, object]
    alternatives: list[dict[str, object]] = Field(default_factory=list)
    changes: list[dict[str, str]] = Field(default_factory=list)
    model_identifier: str


@app.get("/")
def index():
    if (FRONTEND_DIST / "index.html").exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return {"name": "GlossWeaver API", "status": "running", "docs": "/docs"}


@app.get("/api/models")
def list_models():
    default_key = default_available_model().key
    ordered = sorted(MODELS.values(), key=lambda model: (model.key != default_key, model.key))
    return {
        "neural_ready": any(m.available and m.model_type != "copy" for m in MODELS.values()),
        "models": [{
            "name": m.display_name, "checkpoint": m.key, "available": m.available,
            "default": m.key == default_key, "debug": public_model(m),
            "key": m.key, "architecture": "copy" if m.model_type == "copy" else "T5-small" if m.key in {"e1_t5", "legacy_e1", "pretrained_t5"} else "FLAN-T5-small",
            "grammar_capability": m.grammar_enabled, "description": m.training_data_description,
            "input_formatter": m.input_formatter, "version": "local-checkpoint",
            "checkpoint_identity": str(resolve_checkpoint(m.checkpoint_path).relative_to(ROOT)) if m.checkpoint_path and resolve_checkpoint(m.checkpoint_path).is_relative_to(ROOT) else m.key,
            "unavailability_reason": None if m.available else "Checkpoint weights are not installed.",
            "historical": m.key.startswith(("e1", "e2", "e3", "e4", "legacy")),
        } for m in ordered],
    }


@app.get("/api/datasets")
def datasets():
    catalogue = [
        ("study_v2/aslg_grouped", "Study v2 · ASLG", "Rule-generated gloss / English", "Joint lineage-isolated matched study", "CC-BY-NC-4.0"),
        ("study_v2/reconstruction_grouped", "Study v2 · Reconstruction", "Synthetic English fragments", "Joint lineage-isolated matched study", "Inherits source licenses"),
        ("aslg_grouped", "ASLG-PC12", "Rule-generated gloss / English", "Research training and validation", "CC-BY-NC-4.0"),
        ("2m_flores_asl", "2M-Flores ASL", "Human-annotated ASL gloss / English", "External development; devtest reserved for final evaluation", "CC-BY-SA-4.0"),
        ("reconstruction_grouped", "Reconstruction corpus", "Synthetic fragments from English and controlled examples", "Broad reconstruction training", "See datasets/README.md for source licenses"),
        ("demo_adaptation", "Time and quantity", "Synthetic controlled examples", "Deployment adaptation", "Generated locally"),
        ("balanced_adaptation", "Balanced adaptation", "Mixture plus untimed examples", "Deployment continuation; present tense is the untimed default", "Inherits source licenses"),
    ]
    rows = []
    for folder, name, kind, usage, license_name in catalogue:
        manifest = ROOT / "datasets/processed" / folder / "manifest.json"
        available = manifest.is_file()
        data = json.loads(manifest.read_text()) if available else {}
        rows.append({"name": name, "kind": kind, "usage": usage, "license": license_name,
                     "folder": f"datasets/processed/{folder}/", "available": available,
                     "counts": data.get("counts", {})})
    return {"root": "datasets/", "guide": "datasets/README.md", "datasets": rows}


@app.post("/api/infer", response_model=InferResponse)
def infer(request: InferRequest):
    if request.checkpoint not in MODELS:
        raise HTTPException(400, "Unknown model")
    info = get_model(request.checkpoint)
    if not info.available:
        raise HTTPException(404, f"{info.display_name} has no trained weights. Restore or train its checkpoint, then refresh model status.")
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(503, "Another reconstruction is running. Try again shortly.", headers={"Retry-After": "2"})
    start = perf_counter()
    try:
        reconstructor = _get_reconstructor(request.checkpoint)
        candidates = []
        if request.alternatives > 1:
            candidates, grammar = reconstructor.reconstruct_candidates(request.gloss, request.num_beams, request.alternatives)
            prediction = candidates[0]["text"]
        else:
            prediction, grammar = reconstructor.reconstruct(request.gloss, num_beams=request.num_beams)
        grammar_aware = reconstructor.grammar_aware
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for %s", request.checkpoint)
        raise HTTPException(500, "The model could not reconstruct this input. Check the server log.") from exc
    finally:
        _model_lock.release()
    notices = ["Missing context can allow multiple valid reconstructions. Check meaning, negation, names and numbers."]
    if info.model_type == "pretrained":
        notices.insert(0, "Pretrained model: these weights have not been fine-tuned on GlossWeaver data. Output may copy the input or change its meaning.")
    if info.model_type == "copy":
        notices.insert(0, "Copy baseline: this output normalizes text; it does not restore missing grammar.")
    return InferResponse(
        gloss=request.gloss, reconstruction=prediction, grammar=grammar,
        grammar_active=[key for key, value in grammar.items() if value >= request.threshold],
        checkpoint=request.checkpoint, grammar_aware=grammar_aware, debug=public_model(info),
        elapsed_ms=round((perf_counter() - start) * 1000, 1), notices=notices,
        semantic=validate_meaning(request.gloss, prediction), alternatives=candidates[1:],
        changes=align_changes(request.gloss, prediction),
        model_identifier=str(resolve_checkpoint(info.checkpoint_path).relative_to(ROOT)) if info.checkpoint_path and resolve_checkpoint(info.checkpoint_path).is_relative_to(ROOT) else info.key,
    )


def _metric_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


@app.get("/api/metrics")
def all_metrics(dataset: str | None = None, split: str | None = None):
    path = ROOT / "results/metrics/model_metrics.csv"
    if not path.exists():
        raise HTTPException(404, "No saved evaluation results are available")
    frame = pd.read_csv(path)
    options = _metric_records(frame[["dataset", "split"]].drop_duplicates())
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if split is not None:
        frame = frame[frame["split"] == split]
    diagnostics_path = ROOT / "results/tables/grammar_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    rows = _metric_records(frame.drop(columns=["checkpoint"], errors="ignore"))
    for row in rows:
        if not diagnostics.empty:
            subset = diagnostics[(diagnostics.model == row["model"]) & (diagnostics.dataset == row["dataset"]) & (diagnostics.split == row["split"])]
            if "evaluation_id" in diagnostics:
                subset = subset[subset.evaluation_id.fillna("") == (row.get("evaluation_id") or "")]
            row["grammar_diagnostics"] = {item.pop("category"): item for item in _metric_records(subset)}
        row["display_name"] = MODELS[row["model"]].display_name if row["model"] in MODELS else row["model"]
        row["checkpoint_available"] = row["model"] in MODELS and MODELS[row["model"]].available
    return {"rows": rows, "evaluations": options, "provenance": "Saved evaluations; not recomputed by this page."}


@app.get("/api/metrics/{experiment}")
def get_metrics(experiment: str):
    result = all_metrics()
    result["rows"] = [row for row in result["rows"] if row["model"] == experiment]
    if not result["rows"]:
        raise HTTPException(404, f"Metrics not found for {experiment}")
    return result


@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "ok", "neural_ready": any(m.available and m.model_type != "copy" for m in MODELS.values()), "default_model": default_available_model().key}


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(FRONTEND_DIST / "favicon.svg")


class ValidateRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2000)
    generated: str = Field(min_length=1, max_length=2000)

    @field_validator("source", "generated")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank")
        return value.strip()


class SemanticResponse(BaseModel):
    status: str
    issues: list[dict[str, object]]
    checks: list[dict[str, str]]
    advisory: str
    validator_version: str


@app.post("/api/validate", response_model=SemanticResponse)
def validate(request: ValidateRequest):
    return validate_meaning(request.source, request.generated)


class CompareRequest(BaseModel):
    gloss: str = Field(min_length=1, max_length=2000, validation_alias=AliasChoices("gloss", "text"))
    models: list[str] = Field(min_length=1, max_length=4)

    @field_validator("gloss")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank")
        return value.strip()
    num_beams: int = Field(default=4, ge=1, le=10)
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class CompareResponse(BaseModel):
    gloss: str
    num_beams: int
    results: list[dict[str, object]]


@app.post("/api/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    results = []
    for key in dict.fromkeys(request.models):
        try:
            item = infer(InferRequest(gloss=request.gloss, checkpoint=key, num_beams=request.num_beams, threshold=request.threshold))
            results.append({"model": key, "status": "ok", "result": item.model_dump()})
        except HTTPException as exc:
            results.append({"model": key, "status": "error", "status_code": exc.status_code, "error": exc.detail})
    return {"gloss": request.gloss, "num_beams": request.num_beams, "results": results}


@app.get("/api/evaluations")
def evaluations():
    from glossweaver.settings import PRESETS
    try:
        result = all_metrics()
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        result = {"rows": [], "evaluations": [], "provenance": "No saved evaluations yet."}
    result["experiments"] = [{"dataset_version": "study-v2", "data_prepared": config.train_file.is_file() and config.validation_file.is_file(), "checkpoint_matches_current_data": None, "key": key, "architecture": config.model_name,
        "augmentation": config.synthetic_train_file is not None, "grammar": config.grammar_enabled,
        "seed": config.seed, "max_steps": config.max_steps,
        "available": MODELS[key].available, "evaluation_count": sum(r["model"] == key for r in result["rows"])}
        for key, config in PRESETS.items() if key.startswith("study-")]
    for row in result["rows"]:
        row["historical"] = not bool(row.get("evaluation_id")) or row["model"].startswith(("e1", "e2", "e3", "e4", "legacy"))
    return result


@app.get("/api/evaluations/{evaluation_id}/examples")
def evaluation_examples(evaluation_id: str, category: str | None = None, limit: int = 100):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", evaluation_id) or not 1 <= limit <= 500:
        raise HTTPException(422, "Invalid evaluation identifier or limit")
    path = ROOT / "results/evaluations" / evaluation_id / "predictions.csv"
    if not path.is_file():
        raise HTTPException(404, "No saved predictions for this evaluation")
    rows = []
    for row in _metric_records(pd.read_csv(path).head(10000)):
        row["semantic"] = validate_meaning(str(row.get("gloss") or ""), str(row.get("prediction") or ""))
        if category and not any(i["category"] == category for i in row["semantic"]["issues"]):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return {"evaluation_id": evaluation_id, "rows": rows, "diagnostics": "Computed advisory lexical checks; saved predictions are unchanged."}
