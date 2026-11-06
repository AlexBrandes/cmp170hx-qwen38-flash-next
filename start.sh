#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next (W4A16) pipeline-parallel on two CMP 170HX cards.
#   ./start.sh          tune the host (sudo), start the container, wait for /health
# Settings: .env (see .env.example) or environment variables.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env

IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-W4A16-AutoRound}
GPUS=${GPUS:-0,1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NAME=${NAME:-cmp170hx-qwen38-flash-next}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.8-flash-next-w4a16}
HOST_TUNE=${HOST_TUNE:-1}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

die() { echo "start.sh: $*" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || die "no model at $MODEL_DIR (run ./download.sh)"
[ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" = 2 ] || die "GPUS must name two cards, e.g. 0,1"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

if [ "$HOST_TUNE" = 1 ]; then
  sudo ./host-tune.sh "$GPUS" "$MODEL_DIR"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --shm-size 1g \
  -p "$HOST:$PORT:8000" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_PP_LAYER_PARTITION=26,22 \
  -e VLLM_MIXED_PREFILL_CHUNK=560 \
  -v "$MODEL_DIR:/models/Qwen3.8-Flash-Next-W4A16-AutoRound:ro" \
  "$IMAGE" \
  vllm serve /models/Qwen3.8-Flash-Next-W4A16-AutoRound \
    --pipeline-parallel-size 2 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len 262144 \
    --block-size 1680 \
    --mamba-cache-mode align \
    --mamba-ssm-cache-dtype float32 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.95 \
    --max-parallel-loading-workers 1 \
    --moe-backend auto \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --no-enable-flashinfer-autotune \
    "-cc=$(tr -d ' \n' < config/compilation.json)" \
    '--engram-config={"disk_offload":true}' \
    '--speculative-config={"method":"mtp","num_speculative_tokens":5,"draft_lm_head_quantization":"nvfp4"}' \
  >/dev/null

echo "started $NAME ($IMAGE) on $HOST:$PORT; loading (several minutes)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fs "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps -q -f "name=^$NAME$" | grep -q .; then
    docker logs --tail 50 "$NAME" >&2 || true
    die "container exited during startup"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || die "not ready after ${READY_TIMEOUT}s (docker logs $NAME)"
  sleep 5
done
echo "ready: http://$HOST:$PORT/v1 (model $SERVED_MODEL_NAME)"
