# Deepfake Detection System

**CO3201 Computer Science Final Year Project, University of Leicester.**

Raul Blanco Vazquez

A four-checkpoint deepfake detection system covering frame-level
classification (XceptionNet baseline, ensemble with EfficientNet-B4),
sequence-level classification (LSTM over a frozen XceptionNet backbone),
and a compression-aware fine-tune for in-the-wild robustness. The system is
served end-to-end through a CUDA-enabled FastAPI container with per-frame
Grad-CAM explainability, a calibrated inconclusive band, and opt-in
test-time augmentation.


## Headline results

| Dataset | Best model | AUC | EER | Source |
|---|---|---|---|---|
| FaceForensics++ (in-domain) | Temporal | 99.87% | 1.20% | section 6.7 |
| Celeb-DF v2 (cross-manipulation) | Temporal | 99.99% | 0.23% | section 6.7 |
| DeepFakeDetection | Temporal | 99.86% | 1.97% | section 6.10.3 |
| WildDeepfake (in-the-wild) | Temporal | 91.54% | 13.86% | section 6.10.3 |
| DFDC (out-of-family) | Mixed v2 | 71.61% | 35.94% | section 6.10.6 |

The mixed-dataset training strategy contributes a **27.36 pp cross-dataset
AUC lift** on Celeb-DF v2 against a single-dataset baseline; the temporal
model adds another ~3 pp on the toughest in-the-wild split. The DFDC drop
to ~70% AUC reflects that DFDC's manipulation methods (DFAE, NTH, FSGAN,
StyleGAN swaps) never appear in training, and is consistent with the
Dolhansky et al. published XceptionNet baseline band.

## Repository layout

The full repository is ~2.3 GB on disk.


```
FINAL YEAR PROJECT/
  config/             centralised YAML config + Docker override
  src/                preprocessing, models, training, FastAPI service
  scripts/            entry points: train, evaluate, profile, visualise
  data/               see data/README.md (datasets are NOT redistributed)
    demo_samples/     short clips for live API verification
  outputs/
    results/          per-dataset evaluation JSONs + plots 
    figures/          rendered dissertation figures
    gradcam/          source PNGs for figures 6.8 to 6.12 
    profiling/        figure 5.5 raw timings
    models/           trained checkpoints
  Dockerfile          API image build recipe
  docker-compose.yml  GPU-enabled service definition
  requirements.txt    pinned Python dependencies
  setup.py            installable Python package
```

## Quick start from a fresh clone

Two independent paths. **Path A** runs the live FastAPI demo via Docker. **Path B** sets up a Python venv for training, evaluation, figure regeneration, and any other script. Pick whichever you need; you can do both.

### Path A - FastAPI live demo (Docker only)

1. **Install Docker Desktop** (Windows: WSL2 backend; Linux: Docker Engine + `nvidia-container-toolkit`).
2. **Install the NVIDIA GPU driver** for the host. The container will fail to start without an NVIDIA GPU and a recent driver. Download from https://www.nvidia.com/Download/index.aspx
3. **Verify GPU passthrough**: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` should print the GPU table. Fix this before continuing if it fails.
4. **Unzip the project** (avoid spaces in the path on Windows where possible).
5. **Open a terminal in the project root** (where `docker-compose.yml` sits).
6. **Build the image**: `docker compose build` (first run downloads the ~5 GB PyTorch CUDA base image, ~6 to 10 minutes).
7. **Start the container**: `docker compose up -d`.
8. **Open `http://localhost:8000/`** in any browser. Upload any clip from `demo_samples/` to verify.

To stop: `docker compose down`. To see live logs: `docker compose logs -f api`. The four trained checkpoints ship in `outputs/models/`; no separate download is required.

### Path B - Training, evaluation, figure regeneration (venv only)

Docker does not cover these workflows. You need a Python virtual environment on the host.

1. **Install Python 3.10 or 3.11**. The pinned dependencies (especially `torch==2.3.1+cu121` and `mediapipe>=0.10.0,<0.10.15`) ship wheels for both; Python 3.12+ does not have matching wheels and will not work. The reference development environment uses Python 3.11.9.
   - Windows: download the 3.10.x or 3.11.x installer from https://www.python.org/downloads/windows/, run it, **tick "Add Python to PATH"** on the first installer screen.
   - Linux/macOS: `pyenv install 3.11` is the cleanest route, or use the system package (`sudo apt install python3.11 python3.11-venv` on Debian/Ubuntu).
2. **Verify the install**: open a fresh terminal and run `python --version` (or `python3 --version`). It must print `Python 3.10.x` or `3.11.x`. Then `python -m pip --version` to confirm pip is available; pip ships with the official Python installers.
3. **Install the NVIDIA GPU driver** (same as Path A step 2). The CPU-only fallback works for some scripts but training is impractical without a GPU.
4. **Open a terminal in the project root**.
5. **Create the venv**: `python -m venv venv`. This creates a `venv/` directory in the project root.
6. **Activate the venv**:
   - Windows (CMD or PowerShell): `venv\Scripts\activate`
   - Windows (Git Bash): `source venv/Scripts/activate`
   - Linux/macOS: `source venv/bin/activate`

   The terminal prompt should now show `(venv)` at the start.
7. **Upgrade pip** (optional but recommended): `python -m pip install --upgrade pip`.
8. **Install pinned dependencies**: `pip install -r requirements.txt`. Takes 5 to 15 minutes; downloads ~3 GB of CUDA-enabled PyTorch wheels from PyTorch's CDN. The `--extra-index-url` line at the top of `requirements.txt` routes torch to the right wheel server automatically.
9. **Install the project as an editable package**: `pip install -e .`. This makes `from src.X import Y` work from any script.
10. **Update host paths in `config/config.yaml` for venv workflows.** The committed config records the original Windows development paths under `data:`, `training.checkpoint_dir`, `training_robust`, `api.upload_dir`, `api.checkpoints`, and `output:`. If the project is not located at the same absolute path on your machine, replace those values with your own paths before running preprocessing, training, evaluation, or the API from the venv. Docker users do not need this step because `config/config.docker.yaml` is mounted inside the container.
11. **Verify GPU torch**:
    ```
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    ```
    Expected output: `True NVIDIA GeForce RTX 4070` (or whichever GPU the host has). If it prints `False`, the driver and torch CUDA build do not agree; a driver upgrade usually fixes it.

Every subsequent terminal session must **re-activate the venv** (step 6 only) before running any script.

#### Common venv tasks after setup

- **Re-evaluate a checkpoint**: `python scripts/evaluate_model.py --checkpoint outputs/models/mixed/best_model_mixed.pth --dataset celebdf --output-dir outputs/results/mixed_eval_celebdf`
- **Re-render a figure**: `python scripts/visualise_roc_grid.py --results-dir outputs/results --prefix ensemble_v1 --title "Ensemble v1 ROC curves across four evaluation datasets" --output outputs/figures/fig_6_4_ensemble_roc.png`
- **Profile inference throughput**: `python scripts/profile_and_plot.py --json-out outputs/profiling/profile_all.json --fig-out outputs/figures/fig_5_5_stages.png --from-json`
- **Retrain a model**: `python scripts/train_xception_mixed.py` (needs the full datasets in `data/` per `data/README.md`)



## Model checkpoints

All trained checkpoints are committed to this repository under
`outputs/models/`. After cloning, the FastAPI container picks them up
automatically through the bind mount declared in `docker-compose.yml`; no
separate download step is required.

The four FastAPI-served checkpoints (the production line):

```
outputs/models/
  mixed/best_model_mixed.pth                   # XceptionNet, FF++ + Celeb-DF, headline
  ensemble_v3/best_model_ensemble.pth          # XceptionNet + EfficientNet-B4
  temporal_v3/best_model_temporal.pth          # frozen backbone + LSTM
  robust_v3/best_model_robust.pth              # compression-aware fine-tune
```

Additional checkpoints used to back specific dissertation figures (kept for
reproducibility, not loaded by the API):

```
outputs/models/
  best_model.pth                               # baseline (figures 6.1, 6.2)
  ensemble/best_model_ensemble.pth             # original ensemble (figure 6.4)
  temporal/best_model_temporal.pth             # original temporal (figure 6.5)
  robust_v2/best_model_robust.pth              # robust v2 (figure 6.7)
  robust_v3_cbam/best_model_robust.pth         # CBAM null-result (figure 6.12)
```

The container will refuse to start if any of the four production
checkpoints is missing; the additional checkpoints are only consulted when
re-running the visualisation scripts under `scripts/`.

## Demo samples

[demo_samples/](demo_samples/) holds short verification clips for each
dataset, picked so the marker can exercise every model family without
having to download any full dataset. See
[demo_samples/README.md](demo_samples/README.md) for the per-clip expected
verdict tables.

The smoke-test reference clip is `demo_samples/dfdc_fake/aassnaulhq.mp4`,
the same file referenced in section 5.7 of the dissertation. Uploading it
through the FastAPI UI reproduces the documented multi-model verdict
disagreement (mixed FAKE, ensemble REAL, robust REAL, temporal REAL),
which is the design feature being demonstrated.

## Reproducibility

### Re-evaluating a checkpoint

```
venv/Scripts/python.exe scripts/evaluate_model.py \
  --checkpoint outputs/models/mixed/best_model_mixed.pth \
  --dataset celebdf \
  --output-dir outputs/results/mixed_eval_celebdf
```

Evaluates a single checkpoint on one dataset, writes the JSON metrics +
ROC plot to `outputs/results/<output-dir>/`. See `--help` for all options.

### Re-rendering a figure

Most figures in the dissertation are generated by scripts under
`scripts/visualise_*.py` and read from JSONs in `outputs/results/` or
`outputs/profiling/`. For example, to re-render figure 6.4:

```
venv/Scripts/python.exe scripts/visualise_roc_grid.py \
  --results-dir outputs/results \
  --prefix ensemble_v1 \
  --title "Ensemble (mixed-trained) - ROC curves across four evaluation datasets" \
  --output outputs/figures/fig_6_4_ensemble_roc.png
```

### Re-training a model

Training requires the full datasets in `data/`. Begin with the mixed
checkpoint (cheapest; baseline + cross-dataset jump):

```
venv/Scripts/python.exe scripts/train_xception_mixed.py
```

Then derived checkpoints:

```
venv/Scripts/python.exe scripts/train_xception_robust.py     # robust fine-tune
venv/Scripts/python.exe scripts/train_ensemble.py            # XceptionNet + EfficientNet-B4
venv/Scripts/python.exe scripts/train_temporal.py            # LSTM over frozen backbone
```

## Datasets

Five datasets are used for training and evaluation: FaceForensics++,
Celeb-DF v2, the standalone DeepFakeDetection release, DFDC, and a
WildDeepfake test subset. Sources, licensing, and the expected directory
layout are documented in [data/README.md](data/README.md). None of the raw
clips are committed to this repository or mirrored on OneDrive; each must
be obtained directly from its original source.

## Citation

If you reference this project, please cite the dissertation:

> Blanco Vazquez, Raul. "Deepfake and Spoofing Detection System." CO3201 Final Year
> Project, University of Leicester, 2026.

## Licence and academic-evaluation notice

Source code in this repository is provided under the MIT licence for
academic use. The trained checkpoints, demo samples, and processed face
crops are derivative works of academic deepfake-detection datasets each
released under terms that prohibit further redistribution; they are
provided to the dissertation marker exclusively for evaluation of this
submission and must not be redistributed or reposted in any form.
