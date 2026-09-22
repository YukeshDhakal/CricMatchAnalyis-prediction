# Video pipeline API only -- not the Streamlit console (see README's "Video pipeline API"
# section for why these are separate deployables: this image needs to stay up and
# reachable from the public internet; the Streamlit console is a local dev tool).
#
# Heavy on purpose: torch/torchvision/ultralytics/opencv are real ML dependencies, not
# trimmable. Expect a multi-GB image and a slow first build even with the CPU-only torch
# pin below (see that RUN step for why it's there). Pick a host plan with at least 2GB
# RAM -- Keypoint R-CNN + YOLO loaded together will not fit in less.
FROM python:3.12-slim

# ffmpeg: opencv's video decoding backend needs it for many real-world codecs (see
# MISTAKES.md's ffmpeg-on-PATH entries from earlier in this project's history -- this
# is the same dependency, just installed system-wide instead of via winget).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# pyproject.toml pins torch/torchvision generically (>=2.6,<3) so local dev on a
# CUDA-capable machine can still get GPU wheels. PyPI's default linux wheel for recent
# torch versions is the CUDA build, which drags in ~2GB of nvidia-*/cudnn/cuda_toolkit
# packages as real pip dependencies even though this container has no GPU to use them
# on -- confirmed the hard way: a build without this line pulled a 554MB GPU torch wheel
# plus a 553MB nvidia_cudnn wheel before being caught and killed. Installing the CPU-only
# wheels first, from PyTorch's own CPU index, satisfies the pyproject.toml version
# constraint (pip treats "2.6.0+cpu" as satisfying ">=2.6,<3") and the plain install
# below then finds torch/torchvision already present and skips reinstalling them.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# `.[api]` pulls in fastapi/uvicorn/python-multipart alongside the video-engine stack
# already declared in pyproject.toml's base dependencies -- one install, one source of
# truth for versions, no separate requirements-api.txt to drift out of sync.
RUN pip install --no-cache-dir -e ".[api]"

# The trained ball+stumps checkpoint is committed as a deliberate exception to the
# repo's normal *.pt gitignore rule (see .gitignore's comment) so a git-based build host
# like Railway actually has it -- without this, YoloDetector falls back to player-only
# detection automatically; the API still works, it just won't classify shots.
COPY weights/ball_stumps_n.pt ./weights/ball_stumps_n.pt

EXPOSE 8000

# THIRD_UMPIRE_API_KEY, THIRD_UMPIRE_ALLOWED_ORIGINS, THIRD_UMPIRE_MAX_UPLOAD_MB are
# read from the environment at runtime -- set them on whatever host runs this, never
# bake a real API key into the image.
CMD ["uvicorn", "api.server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
