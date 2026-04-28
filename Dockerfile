# Deepfake Detection - FastAPI inference service
#
# Base: official PyTorch image with CUDA 12.1 runtime + cuDNN 8. We still let
# pip reinstall torch==2.3.1+cu121 from the PyTorch wheel index because the
# conda-installed torch in the base image lacks the +cu121 local-version tag,
# and stripping the pin lets transitive deps (facenet-pytorch, triton) drag
# torch back down to a CPU-only wheel.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Non-interactive apt and a saner Python runtime.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries required by OpenCV (libGL, libglib) and MediaPipe (libsm,
# libxext, libxrender). ffmpeg covers H.264/H.265 decode for the profiler clip
# and the upload path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependency manifests first so the heavy pip install layer is
# cached across source-only edits.
COPY requirements.txt setup.py ./

# Install Python deps as-is from requirements.txt. The +cu121 pins and the
# --extra-index-url inside that file route torch / torchvision to the PyTorch
# CDN, which is what we want. grad-cam is now in requirements.txt directly so
# venv users get the explainability path without a separate install step.
RUN pip install --no-cache-dir -r requirements.txt

# Copy source. config/, src/, and scripts/ are the only directories the API
# touches at runtime; outputs/ and data/ are mounted as volumes (see
# docker-compose.yml).
COPY src/    ./src/
COPY scripts ./scripts/
COPY config  ./config/

# Install the project package so `from src.X import Y` works inside the container.
RUN pip install --no-cache-dir -e .

# Default config rewrites the Windows-host paths in config.yaml to /app/... ones.
# docker-compose mounts ./config/config.docker.yaml over this path at runtime.
EXPOSE 8000

# Bind to 0.0.0.0 so the published container port is reachable from the host.
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0"]
