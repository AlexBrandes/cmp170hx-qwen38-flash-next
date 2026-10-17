#!/usr/bin/env bash
# Download the checkpoint at the pinned revision and build the MTP draft head.
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
run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
          -e HF_TOKEN="${HF_TOKEN:-}" -v "$MODELS_DIR:/models" \
          -v "$MODEL_DIR:/model" -v "$PWD/tools:/tools:ro" --entrypoint python3 "$IMAGE" "$@"; }
run -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
print('model ready')"
out=$MODELS_DIR/draft-head-w4a16.safetensors
if [ ! -f "$out" ] || [ "$(sha256sum "$out" | cut -d' ' -f1)" != "$DRAFT_HEAD_SHA256" ]; then
  run /tools/quantize_lm_head.py --model-dir /model \
      --out /models/draft-head-w4a16.safetensors --mse-scale-search
fi
[ "$(sha256sum "$out" | cut -d' ' -f1)" = "$DRAFT_HEAD_SHA256" ] \
  || { echo "draft head hash mismatch" >&2; exit 1; }
echo "draft head ready: $out"
