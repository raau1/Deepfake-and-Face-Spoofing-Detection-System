"""
Pydantic request and response schemas for the FastAPI deployment.
FastAPI uses these both for request validation (422 on bad input)
and for response serialisation (automatic JSON conversion).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FramePrediction(BaseModel):
    """Per-frame fake probability, returned as part of a PredictionResponse."""
    frame_index: int = Field(..., description="Zero-based index within the sampled sequence")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Model probability that this frame is fake")


class GradCamFrame(BaseModel):
    """Per-frame Grad-CAM overlay, returned when include_gradcam is requested."""
    frame_index: int = Field(..., description="Zero-based index within the sampled sequence")
    fake_probability: float = Field(..., description="Fake probability for this frame (matches FramePrediction entry)")
    target_class: int = Field(..., description="Class whose score was explained: 0=real, 1=fake (the model's own argmax)")
    overlay_png_base64: str = Field(..., description="Base64-encoded PNG of the heatmap overlay (no data URI prefix)")


class PredictionResponse(BaseModel):
    """Response body for POST /api/predict."""
    verdict: str = Field(..., description='"REAL", "FAKE", or "INCONCLUSIVE" when probability falls inside the inconclusive band')
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Aggregated video-level fake probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Distance from decision boundary (0.5), scaled to [0, 1]")
    model: str = Field(..., description="Model used for this prediction: mixed, ensemble, temporal, or robust")
    frames_analysed: int = Field(..., description="Number of frames successfully extracted and classified")
    faces_detected: int = Field(..., description="Number of frames where a face was detected")
    processing_time_seconds: float = Field(..., description="End-to-end time from upload to verdict")
    per_frame: List[FramePrediction] = Field(default_factory=list, description="Per-frame fake probabilities")
    gradcam_frames: List[GradCamFrame] = Field(default_factory=list, description="Per-frame Grad-CAM overlay PNGs (empty unless include_gradcam=true)")
    threshold: float = Field(..., description="Decision threshold used to produce verdict")
    inconclusive_band: List[float] = Field(
        default_factory=lambda: [0.4, 0.6],
        description="Lower and upper bounds of the inconclusive probability band; verdict=INCONCLUSIVE when fake_probability falls inside",
    )
    tta_applied: bool = Field(False, description="True when horizontal-flip test-time augmentation was applied (softmax averaging over original + mirror)")
    fake_probability_no_tta: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Video-level fake probability from the original (un-flipped) input only. Populated only when tta_applied is true, so the UI can render a before/after diagnostic; None otherwise",
    )


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""
    status: str
    device: str
    cuda_available: bool
    loaded_models: List[str]
    default_model: str


class ModelInfo(BaseModel):
    """Metadata for a single model, returned by GET /api/models."""
    name: str
    description: str
    parameters: int
    ff_auc: Optional[float] = None
    celebdf_auc: Optional[float] = None
    is_loaded: bool
    is_default: bool


class ModelsResponse(BaseModel):
    """Response body for GET /api/models."""
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    """Generic error response used for 4xx replies."""
    detail: str
