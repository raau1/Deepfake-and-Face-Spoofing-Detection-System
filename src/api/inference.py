"""
InferenceService , runs a video through preprocessing and a chosen model,
returning a single verdict. Used by the FastAPI endpoint in src/api/main.py.

This module encapsulates everything that must happen between "user uploads
a video" and "JSON response goes back":

    1. Preprocess the video (MTCNN face detection, MediaPipe alignment,
       32 uniformly-sampled aligned faces)
    2. Normalise and batch the face tensors
    3. Run the chosen model (mixed XceptionNet / ensemble / temporal LSTM)
    4. Aggregate per-frame probabilities into a single video verdict

All three model variants are loaded once at application startup so that
per-request latency is dominated by preprocessing (MTCNN is the slow step).
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.xception import create_xception
from src.models.ensemble import create_ensemble
from src.models.temporal import create_temporal
from src.preprocessing.pipeline import PreprocessingPipeline


class InferenceService:
    """
    Loads one or more trained models and serves per-video predictions.

    A single instance is constructed at FastAPI startup (see src/api/main.py
    lifespan handler) and shared across requests. Each model is loaded lazily:
    if a checkpoint is missing at startup the service logs a warning and
    continues without that model, rather than refusing to start.
    """

    SUPPORTED_MODELS = ("mixed", "ensemble", "temporal", "robust")

    def __init__(
        self,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        api_cfg = config.get("api", {})
        preproc_cfg = config.get("preprocessing", {})
        image_cfg = preproc_cfg.get("image", {})
        model_cfg = config.get("model", {})

        inference_cfg = config.get("inference", {})
        self.threshold: float = inference_cfg.get("default_threshold", 0.5)
        # Inconclusive band: probabilities inside [lower, upper] map to
        # verdict == "INCONCLUSIVE" rather than forced REAL/FAKE. Keeps
        # verdicts honest on borderline clips (mixed in-the-wild content,
        # OOD generators, heavy compression) where the model's own calibration
        # says "not sure" - the v3 line's 57.7%–99.8% confidence spread
        # reported in §6.10.2 surfaces exactly this middle band.
        self.inconclusive_lower: float = float(
            inference_cfg.get("inconclusive_lower", 0.4)
        )
        self.inconclusive_upper: float = float(
            inference_cfg.get("inconclusive_upper", 0.6)
        )
        self.default_model: str = api_cfg.get("default_model", "temporal")
        self.frames_per_video: int = preproc_cfg.get("frames_per_video", 32)
        # Temporal model was trained with this fixed sequence length.
        # At inference we pad/truncate to match (cycle-repeat for short videos).
        self.temporal_sequence_length: int = model_cfg.get("temporal", {}).get(
            "sequence_length", 32
        )

        self.checkpoints: Dict[str, str] = api_cfg.get("checkpoints", {})

        # Models are loaded into this dict by _load_models().
        # Missing checkpoints produce a warning but do not abort startup.
        self.models: Dict[str, nn.Module] = {}

        # Preprocessing pipeline , same MTCNN + MediaPipe flow used for training,
        # just configured for inference (single video at a time, no progress bar).
        self.pipeline = PreprocessingPipeline(
            output_size=image_cfg.get("size", 299),
            frames_per_video=self.frames_per_video,
            sampling_strategy=preproc_cfg.get("frame_sampling", "uniform"),
            margin=preproc_cfg.get("alignment", {}).get("margin", 0.3),
            use_alignment=preproc_cfg.get("alignment", {}).get("enabled", True),
            device=str(self.device),
            min_face_size=preproc_cfg.get("min_face_size", 60),
        )

        # ImageNet-style normalisation , must match the values used during
        # training (config.preprocessing.image.normalize_mean/std).
        mean = image_cfg.get("normalize_mean", [0.5, 0.5, 0.5])
        std = image_cfg.get("normalize_std", [0.5, 0.5, 0.5])
        self.normalise = transforms.Normalize(mean=mean, std=std)

        self._load_models()

    # ---------- model loading ----------

    def _load_models(self) -> None:
        """Load every checkpoint declared in config.api.checkpoints."""
        for name, ckpt_path in self.checkpoints.items():
            if name not in self.SUPPORTED_MODELS:
                print(f"[InferenceService] Skipping unknown model '{name}'")
                continue
            if not Path(ckpt_path).exists():
                print(f"[InferenceService] Checkpoint missing for '{name}': {ckpt_path}")
                continue
            try:
                model = self._build_and_load(name, ckpt_path)
                model.eval()
                self.models[name] = model
                print(f"[InferenceService] Loaded '{name}' from {ckpt_path}")
            except Exception as exc:
                print(f"[InferenceService] Failed to load '{name}': {exc}")

        if not self.models:
            raise RuntimeError(
                "No models could be loaded. Check config.api.checkpoints paths."
            )
        if self.default_model not in self.models:
            # Fall back to whichever model did load, so the API still works.
            self.default_model = next(iter(self.models))
            print(
                f"[InferenceService] Default model unavailable, "
                f"falling back to '{self.default_model}'"
            )

    def _build_and_load(self, name: str, ckpt_path: str) -> nn.Module:
        """Instantiate the correct architecture and load its weights."""
        model_cfg = self.config.get("model", {})

        if name in ("mixed", "robust"):
            # The robust checkpoint is architecturally identical to mixed
            # (XceptionNet); the difference is the compression-aware fine-tuning.
            model = create_xception(
                num_classes=model_cfg.get("xception", {}).get("num_classes", 2),
                dropout=model_cfg.get("xception", {}).get("dropout", 0.5),
                pretrained=False,  # weights come from the checkpoint
            )
        elif name == "ensemble":
            ens_cfg = model_cfg.get("ensemble", {})
            model = create_ensemble(
                num_classes=2,
                dropout=model_cfg.get("xception", {}).get("dropout", 0.5),
                pretrained=False,
                fusion=ens_cfg.get("fusion", "mean"),
                xception_weight=ens_cfg.get("xception_weight", 0.5),
            )
        elif name == "temporal":
            temp_cfg = model_cfg.get("temporal", {})
            model = create_temporal(
                num_classes=2,
                backbone_dropout=model_cfg.get("xception", {}).get("dropout", 0.5),
                pretrained_backbone=False,
                lstm_hidden=temp_cfg.get("lstm_hidden", 512),
                lstm_layers=temp_cfg.get("lstm_layers", 2),
                lstm_dropout=temp_cfg.get("lstm_dropout", 0.3),
                classifier_dropout=temp_cfg.get("classifier_dropout", 0.5),
                bidirectional=temp_cfg.get("bidirectional", False),
                freeze_backbone=True,
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Checkpoints saved before the temporal refactor have no prefix mismatch,
        # so strict loading is fine here.
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        return model

    # ---------- public helpers ----------

    def loaded_model_names(self) -> List[str]:
        return list(self.models.keys())

    def is_model_loaded(self, name: str) -> bool:
        return name in self.models

    # ---------- main inference entrypoint ----------

    def predict(
        self,
        video_path: str,
        model_name: Optional[str] = None,
        include_gradcam: bool = False,
        gradcam_max_frames: int = 8,
        use_tta: bool = False,
    ) -> Dict:
        """
        Run full pipeline: preprocess → infer → aggregate.

        Args:
            video_path: Absolute path to the uploaded video
            model_name: "mixed", "ensemble", "temporal", or "robust"
                (defaults to self.default_model)
            include_gradcam: If True, produce Grad-CAM overlay PNGs for a
                subset of frames and return them base64-encoded alongside
                the per-frame probabilities.
            gradcam_max_frames: Upper bound on number of frames for which
                overlays are computed (uniformly sampled across detected
                frames). Keeps response size and wall-time predictable.
            use_tta: If True, apply horizontal-flip test-time augmentation:
                run the verdict pipeline on the original face sequence and
                its left-right mirror, then average the softmax outputs.
                Roughly doubles inference cost but stabilises the verdict
                on borderline clips. Does not affect Grad-CAM - overlays
                are always computed against the original (un-flipped) input
                so their spatial coordinates match the thumbnail axes the
                UI uses.

        Returns:
            Dict matching schemas.PredictionResponse
        """
        start = time.perf_counter()

        name = (model_name or self.default_model).lower()
        if name not in self.models:
            raise ValueError(
                f"Model '{name}' is not loaded. Available: {self.loaded_model_names()}"
            )
        model = self.models[name]

        faces = self._extract_faces(video_path)
        if not faces:
            raise RuntimeError(
                "No faces could be extracted from the video. "
                "Check that it contains a clear, frontal face."
            )

        tensor = self._faces_to_tensor(faces)  # [T, 3, H, W]

        if name == "temporal":
            fake_prob, per_frame, fake_prob_no_tta = self._infer_temporal(
                model, tensor, use_tta=use_tta
            )
        else:
            fake_prob, per_frame, fake_prob_no_tta = self._infer_framewise(
                model, tensor, use_tta=use_tta
            )

        gradcam_frames: List[Dict] = []
        if include_gradcam:
            try:
                gradcam_frames = self._compute_gradcams(
                    model_name=name,
                    face_tensor=tensor,
                    face_rgbs=faces,
                    per_frame_probs=per_frame,
                    max_frames=max(1, int(gradcam_max_frames)),
                )
            except Exception as exc:
                # Grad-CAM is a non-essential explanation layer - if it
                # fails we still return the verdict, just without overlays.
                print(f"[InferenceService] Grad-CAM generation failed: {exc}")
                gradcam_frames = []

        verdict = self._verdict_from_prob(fake_prob)
        # Confidence = how far the probability is from the 0.5 boundary,
        # rescaled to [0, 1]. A probability of 0.5 (maximally uncertain)
        # gives confidence 0.0; 0.0 or 1.0 give confidence 1.0.
        confidence = float(abs(fake_prob - 0.5) * 2.0)

        elapsed = time.perf_counter() - start

        return {
            "verdict": verdict,
            "fake_probability": float(fake_prob),
            "confidence": confidence,
            "model": name,
            "frames_analysed": len(faces),
            "faces_detected": len(faces),
            "processing_time_seconds": float(elapsed),
            "per_frame": [
                {"frame_index": i, "fake_probability": float(p)}
                for i, p in enumerate(per_frame)
            ],
            "gradcam_frames": gradcam_frames,
            "threshold": self.threshold,
            "inconclusive_band": [self.inconclusive_lower, self.inconclusive_upper],
            "tta_applied": bool(use_tta),
            "fake_probability_no_tta": fake_prob_no_tta,
        }

    def _verdict_from_prob(self, fake_prob: float) -> str:
        """Map a probability to a three-way verdict, using the configured band.

        If the band is degenerate (lower == upper) we fall back to the
        binary threshold. Otherwise a probability inside [lower, upper]
        maps to INCONCLUSIVE, below lower maps to REAL, and above upper
        maps to FAKE.
        """
        if self.inconclusive_lower >= self.inconclusive_upper:
            return "FAKE" if fake_prob >= self.threshold else "REAL"
        if fake_prob < self.inconclusive_lower:
            return "REAL"
        if fake_prob > self.inconclusive_upper:
            return "FAKE"
        return "INCONCLUSIVE"

    # ---------- preprocessing + inference internals ----------

    def _extract_faces(self, video_path: str) -> List[np.ndarray]:
        """Run the preprocessing pipeline on a single video."""
        # process_video returns (faces, metadata) where faces is a list of
        # HWC uint8 RGB face arrays (one per frame where a face was detected).
        faces, metadata = self.pipeline.process_video(video_path)
        if "error" in metadata:
            raise RuntimeError(f"Video preprocessing failed: {metadata['error']}")
        return [f for f in faces if f is not None]

    def _faces_to_tensor(self, faces: List[np.ndarray]) -> torch.Tensor:
        """Convert a list of RGB face arrays to a normalised [T, 3, H, W] tensor."""
        tensors = []
        for face in faces:
            # faces come out of the pipeline as HWC uint8 RGB arrays.
            t = torch.from_numpy(face).float().permute(2, 0, 1) / 255.0
            t = self.normalise(t)
            tensors.append(t)
        return torch.stack(tensors, dim=0).to(self.device)  # [T, 3, H, W]

    @torch.no_grad()
    def _infer_framewise(
        self, model: nn.Module, frames: torch.Tensor, use_tta: bool = False
    ) -> Tuple[float, List[float], Optional[float]]:
        """Per-frame inference used by the mixed and ensemble models.

        When ``use_tta`` is set we average softmax outputs over the original
        and its horizontal mirror. Flipping the width axis (dim=3) is safe
        for face imagery - left/right cheek swap is semantically equivalent
        for the real-vs-fake decision - and tends to smooth out single-frame
        jitter from MTCNN crop asymmetries.

        The third return value is the un-averaged video-level probability
        (original input only) when ``use_tta`` is True, or ``None`` when TTA
        is disabled. The UI uses it to render a before/after diagnostic so
        the user can see how much the averaging actually moved the verdict.
        """
        probs_orig_t = torch.softmax(model(frames), dim=1)[:, 1]
        probs_orig = probs_orig_t.cpu().numpy()
        if use_tta:
            probs_flip = torch.softmax(model(torch.flip(frames, dims=[3])), dim=1)[:, 1]
            probs = ((probs_orig_t + probs_flip) / 2.0).cpu().numpy()
            fake_prob_no_tta: Optional[float] = float(np.mean(probs_orig))
        else:
            probs = probs_orig
            fake_prob_no_tta = None
        per_frame = probs.tolist()
        # Video-level aggregation: mean probability (matches the evaluation default).
        fake_prob = float(np.mean(probs))
        return fake_prob, per_frame, fake_prob_no_tta

    @torch.no_grad()
    def _infer_temporal(
        self, model: nn.Module, frames: torch.Tensor, use_tta: bool = False
    ) -> Tuple[float, List[float], Optional[float]]:
        """
        Temporal-model inference: one forward pass over the whole sequence.

        The LSTM was trained on fixed-length sequences (self.temporal_sequence_length).
        If fewer faces were detected, cycle-repeat to reach that length (matches
        SequenceDataset behaviour). If more were detected (unusual), truncate.
        The video verdict comes from the LSTM's final hidden state; per-frame
        probabilities for the UI timeline are taken from the frozen backbone's
        classifier head applied to the (un-padded) detected frames.

        With ``use_tta`` we run the sequence-level forward twice (original
        and horizontally-flipped sequence) and average the softmaxes, and do
        the same for the per-frame backbone timeline. The third return is
        the un-averaged LSTM probability so the UI can show a before/after
        diagnostic.
        """
        target_T = self.temporal_sequence_length
        seq_frames = self._pad_or_truncate(frames, target_T)

        seq = seq_frames.unsqueeze(0)  # [1, T, 3, 299, 299]
        video_prob_orig = torch.softmax(model(seq), dim=1)[0, 1]
        if use_tta:
            seq_flip = torch.flip(seq, dims=[4])  # width axis in [B, T, 3, H, W]
            video_prob_flip = torch.softmax(model(seq_flip), dim=1)[0, 1]
            video_prob = float(((video_prob_orig + video_prob_flip) / 2.0).item())
            fake_prob_no_tta: Optional[float] = float(video_prob_orig.item())
        else:
            video_prob = float(video_prob_orig.item())
            fake_prob_no_tta = None

        # Per-frame timeline: only report probabilities for real detected frames
        # (not the padded repeats), so the UI does not show duplicate bars.
        probs_orig = torch.softmax(model.backbone(frames), dim=1)[:, 1]
        if use_tta:
            probs_flip = torch.softmax(
                model.backbone(torch.flip(frames, dims=[3])), dim=1
            )[:, 1]
            per_frame = ((probs_orig + probs_flip) / 2.0).cpu().numpy().tolist()
        else:
            per_frame = probs_orig.cpu().numpy().tolist()

        return video_prob, per_frame, fake_prob_no_tta

    @staticmethod
    def _pad_or_truncate(frames: torch.Tensor, target_T: int) -> torch.Tensor:
        """Cycle-repeat (pad) or slice (truncate) to exactly target_T frames."""
        T = frames.size(0)
        if T == target_T:
            return frames
        if T > target_T:
            # Uniform sub-sample to target_T.
            indices = torch.linspace(0, T - 1, target_T).long()
            return frames[indices]
        # T < target_T: cycle-repeat (e.g. [a, b, c] -> [a, b, c, a, b, c, a]).
        repeats = (target_T + T - 1) // T
        return frames.repeat(repeats, 1, 1, 1)[:target_T]

    # ---------- Grad-CAM explainability ----------

    def _gradcam_target(self, model_name: str, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the (explanation_model, target_layer) pair for Grad-CAM.

        For the temporal model the LSTM head is downstream of a frozen
        per-frame backbone, so the spatial explanation runs against the
        backbone itself rather than the full temporal model. For ensemble
        we explain the XceptionNet branch (the EfficientNet branch would
        give a second, differently-routed heatmap; we pick one for the UI
        to keep the response compact).
        """
        if model_name in ("mixed", "robust"):
            return model, model.backbone.act4
        if model_name == "ensemble":
            return model.xception, model.xception.backbone.act4
        if model_name == "temporal":
            return model.backbone, model.backbone.backbone.act4
        raise ValueError(f"Grad-CAM not supported for model '{model_name}'")

    def _compute_gradcams(
        self,
        model_name: str,
        face_tensor: torch.Tensor,
        face_rgbs: List[np.ndarray],
        per_frame_probs: List[float],
        max_frames: int,
    ) -> List[Dict]:
        """Build per-frame Grad-CAM overlays for a uniformly-sampled subset.

        Args:
            model_name: The chosen model. Drives target-layer resolution.
            face_tensor: [T, 3, H, W] normalised tensor from _faces_to_tensor.
            face_rgbs: List of HWC uint8 RGB face crops (pre-normalisation).
            per_frame_probs: Fake probabilities returned alongside the verdict.
                Length is either T (framewise models) or a shorter prefix for
                the temporal model's detected-only timeline.
            max_frames: Upper bound on the number of overlays to produce.

        Returns:
            List of {frame_index, fake_probability, overlay_png_base64} dicts,
            ordered by frame index.
        """
        model = self.models[model_name]
        explanation_model, target_layer = self._gradcam_target(model_name, model)

        T = face_tensor.shape[0]
        # Uniformly pick at most max_frames frames across the detected range.
        if T <= max_frames:
            indices = list(range(T))
        else:
            indices = [int(round(i)) for i in np.linspace(0, T - 1, max_frames)]

        # Prepare the preprocessed face images (un-normalised, [0, 1] float)
        # for overlay compositing. face_rgbs are uint8 RGB at the model input
        # size (output of the preprocessing pipeline), so no resize is needed.
        overlays: List[Dict] = []

        # GradCAM handles putting the model in eval mode and enabling grads
        # on the input tensor. We use a context manager so the CAM object's
        # forward/backward hooks are detached at the end.
        with GradCAM(model=explanation_model, target_layers=[target_layer]) as cam:
            for idx in indices:
                # Single-frame forward through the explanation model to pick
                # the predicted class as the Grad-CAM target. Using the
                # model's own argmax (rather than always explaining class 1)
                # produces heatmaps that read as "why the model said X" for
                # both REAL and FAKE verdicts.
                input_tensor = face_tensor[idx : idx + 1].clone().requires_grad_(True)

                with torch.enable_grad():
                    logits = explanation_model(input_tensor)
                    target_class = int(torch.softmax(logits, dim=1).argmax(dim=1).item())

                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(target_class)],
                    )[0]  # [H, W] float32 in [0, 1]

                # Composite the heatmap over the original (un-normalised) face.
                face_float = face_rgbs[idx].astype(np.float32) / 255.0
                overlay = show_cam_on_image(face_float, grayscale_cam, use_rgb=True)
                png_b64 = self._rgb_to_png_base64(overlay)

                # Defensive bounds-check on per_frame_probs - for the temporal
                # model this list only covers detected frames, so idx is valid
                # by construction (face_tensor has the same length).
                frame_prob = (
                    float(per_frame_probs[idx])
                    if idx < len(per_frame_probs)
                    else float("nan")
                )

                overlays.append(
                    {
                        "frame_index": int(idx),
                        "fake_probability": frame_prob,
                        "target_class": target_class,  # 0=real, 1=fake
                        "overlay_png_base64": png_b64,
                    }
                )

        return overlays

    @staticmethod
    def _rgb_to_png_base64(rgb: np.ndarray) -> str:
        """Encode an HWC uint8 RGB array as a base64 PNG string (no data URI prefix)."""
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
