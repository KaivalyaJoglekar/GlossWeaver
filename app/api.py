"""FastAPI backend using the same Python model registry as direct inference."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from glossweaver.inference import Reconstructor
from glossweaver.model_registry import MODELS, default_available_model, get_model


app = FastAPI(title="GlossWeaver API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
FRONTEND_DIST = ROOT / "frontend/dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

_reconstructors: dict[str, tuple[int, Reconstructor]] = {}


def _checkpoint_stamp(model_key: str) -> int:
    info = get_model(model_key)
    if info.checkpoint_path is None:
        return 0
    weight = next(
        (info.checkpoint_path / name for name in ("model.safetensors", "pytorch_model.bin")
         if (info.checkpoint_path / name).exists()),
        None,
    )
    return weight.stat().st_mtime_ns if weight else -1


def _get_reconstructor(model_key: str) -> Reconstructor:
    stamp = _checkpoint_stamp(model_key)
    cached = _reconstructors.get(model_key)
    if cached is None or cached[0] != stamp:
        _reconstructors[model_key] = (stamp, Reconstructor(model_key, device="auto"))
    return _reconstructors[model_key][1]


class InferRequest(BaseModel):
    gloss: str
    checkpoint: str
    num_beams: int = 4
    threshold: float = 0.5


class InferResponse(BaseModel):
    gloss: str
    reconstruction: str
    grammar: dict[str, float]
    grammar_active: list[str]
    checkpoint: str
    grammar_aware: bool
    debug: dict[str, object]


@app.get("/")
def index():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"name": "GlossWeaver API", "status": "running", "docs": "/docs"}


@app.get("/api/models")
def list_models():
    default_key = default_available_model().key
    ordered = sorted(MODELS.values(), key=lambda model: (model.key != default_key, model.key))
    return {"models": [{
        "name": model.display_name,
        "checkpoint": model.key,
        "available": model.available,
        "default": model.key == default_key,
        "debug": model.debug_dict(),
    } for model in ordered]}


@app.post("/api/infer", response_model=InferResponse)
def infer(request: InferRequest):
    if request.checkpoint not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.checkpoint}")
    info = get_model(request.checkpoint)
    if not info.available:
        raise HTTPException(
            status_code=404,
            detail=f"Requested checkpoint is unavailable; no fallback was loaded: {info.checkpoint_path}",
        )
    if not request.gloss.strip():
        raise HTTPException(status_code=400, detail="Gloss input is empty")
    try:
        reconstructor = _get_reconstructor(request.checkpoint)
        prediction, grammar = reconstructor.reconstruct(
            request.gloss.strip(), num_beams=request.num_beams
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return InferResponse(
        gloss=request.gloss.strip(), reconstruction=prediction, grammar=grammar,
        grammar_active=[key for key, value in grammar.items() if value >= request.threshold],
        checkpoint=request.checkpoint, grammar_aware=reconstructor.grammar_aware,
        debug=reconstructor.debug_info,
    )


@app.get("/api/metrics/{experiment}")
def get_metrics(experiment: str):
    path = ROOT / "results/metrics/model_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Consolidated metrics are not available yet")
    rows = pd.read_csv(path)
    selected = rows[rows["model"] == experiment]
    if selected.empty:
        raise HTTPException(status_code=404, detail=f"Metrics not found for {experiment}")
    return {"rows": selected.fillna("").to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok"}
