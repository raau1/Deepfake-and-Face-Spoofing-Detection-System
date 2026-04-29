# Outputs and Model Checkpoints

This folder stores generated project artifacts such as evaluation results,
figures, Grad-CAM examples, profiling data, and trained model checkpoints.

The trained PyTorch checkpoint files are large, so they are hosted on Kaggle
instead of being stored directly in the GitHub repository.

## Download the Models

Download the model files from Kaggle:

https://www.kaggle.com/models/raulblancovazquez/deepfake-detection-system-models/

You may need to sign in to Kaggle before downloading.

## Where to Put the Files

After downloading the Kaggle model archive, create a `models` folder inside
`outputs` if it does not already exist:

```text
outputs/models/
```

Then extract or move the downloaded checkpoint folders into that folder.

```text
outputs/models/
```

The expected structure is:

```text
outputs/models/
  mixed/
    best_model_mixed.pth
  ensemble_v3/
    best_model_ensemble.pth
  temporal_v3/
    best_model_temporal.pth
  robust_v3/
    best_model_robust.pth
```

These four checkpoints are the production models used by the FastAPI inference
service.

Additional dissertation/reproducibility checkpoints may also be included:

```text
outputs/models/
  best_model.pth
  ensemble/
    best_model_ensemble.pth
  temporal/
    best_model_temporal.pth
  robust_v2/
    best_model_robust.pth
  robust_v3_cbam/
    best_model_robust.pth
```

These extra checkpoints are used for reproducing figures and ablation results.
They are not required for the default API demo.

## Verify the Download

From the project root, check that the production checkpoints exist:

```powershell
Get-ChildItem outputs\models -Recurse -Filter *.pth
```

On Linux/macOS:

```bash
find outputs/models -name "*.pth"
```

If the files are in the correct location, the Docker/FastAPI demo can load them
using the paths configured in `config/config.docker.yaml`.

## Using the Models

The checkpoints are PyTorch `.pth` files. The repository code expects them to be
loaded through the model classes in `src/models/` and the inference service in
`src/api/inference.py`.

Recommended usage:

1. Clone the GitHub repository.
2. Download the model checkpoints from Kaggle.
3. Extract them into `outputs/models/`.
4. Build and run the Docker service, or run the Python evaluation scripts from a
   local virtual environment.

See the main project `README.md` for full setup, Docker, evaluation, and
retraining instructions.

## Kaggle Citation

If you use these checkpoints, cite the Kaggle model and dissertation release:

- Kaggle model: https://www.kaggle.com/models/raulblancovazquez/deepfake-detection-system-models/
- Dissertation release: https://github.com/raau1/Deepfake-and-Face-Spoofing-Detection-System/releases/tag/dissertation
