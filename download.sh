#!/usr/bin/env bash
# Download the checkpoint at the pinned revision.
#   ./download.sh       (MODELS_DIR=./models by default; ~170 GB)
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-W4A16-AutoRound}
mkdir -p "$MODEL_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$MODEL_DIR:/model" --entrypoint python3 "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
print('model ready')"
