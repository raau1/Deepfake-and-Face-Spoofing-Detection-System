"""
FastAPI application entrypoint.

Run with:
    python scripts/run_api.py
or directly:
    uvicorn src.api.main:app --reload

This module wires the InferenceService to three endpoints and serves a
minimal HTML upload page. It deliberately keeps all heavy work (model
loading, preprocessing, inference) inside InferenceService so that the
routes here stay thin and readable.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.inference import InferenceService
from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictionResponse,
)


# ---------- config loading ----------

# Resolve config.yaml relative to the project root (two parents up from this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

API_CFG = CONFIG.get("api", {})
UPLOAD_DIR = Path(API_CFG.get("upload_dir", PROJECT_ROOT / "outputs" / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(API_CFG.get("max_file_size_mb", 200)) * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ext.lower() for ext in API_CFG.get("allowed_extensions", [".mp4"])
}


# ---------- app lifespan: load models once at startup ----------

# Module-level holder so routes can access the service. Populated by lifespan.
_service: Optional[InferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every configured checkpoint once when the server boots."""
    global _service
    print("[api] Initialising InferenceService , loading checkpoints...")
    _service = InferenceService(CONFIG)
    print(
        f"[api] Ready. Loaded models: {_service.loaded_model_names()}, "
        f"default: {_service.default_model}"
    )
    yield
    # Shutdown: nothing to clean up, torch frees GPU memory on process exit.
    print("[api] Shutting down.")


app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "Video deepfake detection powered by XceptionNet, an ensemble with "
        "EfficientNet-B4, and a temporal LSTM. Upload a video, choose a model, "
        "receive a verdict."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable the default /docs so we can serve a dark-themed override below.
    docs_url=None,
    redoc_url=None,
)


# Templates and static files live inside src/api/ so the package is self-contained.
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
# Shared image assets live at src/images/ (one level up from the api package)
# so that models, scripts, and notebooks can reference them without touching
# the api bundle.
IMAGES_DIR = PROJECT_ROOT / "src" / "images"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


def _get_service() -> InferenceService:
    """Guard against routes being called before lifespan completes."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _service


# ---------- routes ----------

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Dark-themed Swagger UI matching the main app palette."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Docs",
        swagger_css_url="/static/swagger-dark.css",
        swagger_ui_parameters={"syntaxHighlight.theme": "monokai"},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the upload form. The template picks up the current model list."""
    svc = _get_service()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "loaded_models": svc.loaded_model_names(),
            "default_model": svc.default_model,
            "max_mb": API_CFG.get("max_file_size_mb", 200),
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        },
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Liveness and model-loading status for monitoring/smoke tests."""
    svc = _get_service()
    return HealthResponse(
        status="ok",
        device=str(svc.device),
        cuda_available=torch.cuda.is_available(),
        loaded_models=svc.loaded_model_names(),
        default_model=svc.default_model,
    )


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """Describe the available model variants with their headline metrics."""
    svc = _get_service()
    # Headline metrics from Section 6 of the dissertation. Hard-coded here
    # rather than re-read from disk because these are reporting numbers,
    # not runtime-mutable values.
    metadata = {
        "mixed": {
            "description": "XceptionNet trained on mixed FF++ All Types + Celeb-DF-v2",
            "parameters": 20_811_050,
            "ff_auc": 0.9933,
            "celebdf_auc": 0.9990,
        },
        "ensemble": {
            "description": "XceptionNet + EfficientNet-B4 ensemble with mean fusion",
            "parameters": 38_363_252,
            "ff_auc": 0.9948,
            "celebdf_auc": 0.9994,
        },
        "temporal": {
            "description": "LSTM over frozen mixed-trained XceptionNet backbone",
            "parameters": 28_160_300,
            "ff_auc": 0.9987,
            "celebdf_auc": 0.9999,
        },
        "robust": {
            "description": "XceptionNet fine-tuned from mixed with compression-aware augmentation (JPEG, downscale, noise) to close the in-the-wild / YouTube generalisation gap",
            "parameters": 20_811_050,
            "ff_auc": 0.9963,
            "celebdf_auc": 0.9992,
        },
    }
    models = [
        ModelInfo(
            name=name,
            description=meta["description"],
            parameters=meta["parameters"],
            ff_auc=meta["ff_auc"],
            celebdf_auc=meta["celebdf_auc"],
            is_loaded=svc.is_model_loaded(name),
            is_default=(name == svc.default_model),
        )
        for name, meta in metadata.items()
    ]
    return ModelsResponse(models=models)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    video: UploadFile = File(..., description="Video file to analyse"),
    model: Optional[str] = Form(None, description="mixed | ensemble | temporal | robust"),
    include_gradcam: bool = Form(
        False,
        description=(
            "If true, produce Grad-CAM heatmap overlays for a subset of frames "
            "and include them base64-encoded in the response. Adds 1–3 seconds "
            "to latency on a consumer GPU."
        ),
    ),
    gradcam_max_frames: int = Form(
        8,
        description="Maximum number of overlays to return (uniformly sampled across detected frames).",
        ge=1,
        le=32,
    ),
    use_tta: bool = Form(
        False,
        description=(
            "If true, apply horizontal-flip test-time augmentation: softmax "
            "outputs are averaged over the original sequence and its mirror. "
            "Roughly doubles inference cost and tends to smooth out borderline "
            "verdicts by cancelling MTCNN crop-asymmetry artefacts."
        ),
    ),
):
    """Accept a video upload, return a JSON verdict."""
    svc = _get_service()

    # Validate extension early - cheaper than saving a bad file first.
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if model and model.lower() not in InferenceService.SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Allowed: {list(InferenceService.SUPPORTED_MODELS)}",
        )
    if model and not svc.is_model_loaded(model.lower()):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' was not loaded at startup (checkpoint missing).",
        )

    # Save upload to disk because OpenCV (used by VideoProcessor) needs a file path,
    # not a Python file-like object.
    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = UPLOAD_DIR / tmp_name

    total_bytes = 0
    try:
        with tmp_path.open("wb") as out:
            # Stream in chunks; enforce the size cap without reading the full
            # upload into memory.
            while True:
                chunk = await video.read(1024 * 1024)  # 1 MiB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
                    )
                out.write(chunk)

        try:
            result = svc.predict(
                str(tmp_path),
                model_name=model,
                include_gradcam=include_gradcam,
                gradcam_max_frames=gradcam_max_frames,
                use_tta=use_tta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return PredictionResponse(**result)

    finally:
        # Always clean up the uploaded file - keeping them would fill the disk
        # and the results we return are already independent of the source video.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
