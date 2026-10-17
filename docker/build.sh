#!/usr/bin/env bash
# Build the engine image from the fork at a pinned commit.
#   docker/build.sh [--fork-dir DIR] [--tag TAG]
# Clones https://github.com/AlexBrandes/vllm at FORK_COMMIT unless --fork-dir
# points at an existing checkout, copies the files the fork changes relative
# to v0.30.0 into the build context, and builds on the pinned official image.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../versions.env"
fork_dir="" tag="$ENGINE_IMAGE_LOCAL"
while [ $# -gt 0 ]; do
  case "$1" in
    --fork-dir) fork_dir=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
if [ -z "$fork_dir" ]; then
  fork_dir="$ctx/vllm"
  git clone --filter=blob:none --no-checkout "$FORK_REPO" "$fork_dir"
fi
git -C "$fork_dir" fetch -q origin "$FORK_COMMIT" 2>/dev/null || true
git -C "$fork_dir" -c advice.detachedHead=false checkout -q "$FORK_COMMIT"
files=$(git -C "$fork_dir" diff --name-only --diff-filter=AM "$VLLM_BASE_TAG" "$FORK_COMMIT" \
        -- vllm vllm_ple_mmap_qwen4.py)
mkdir -p "$ctx/overlay"
for f in $files; do
  mkdir -p "$ctx/overlay/$(dirname "$f")"
  cp "$fork_dir/$f" "$ctx/overlay/$f"
done
printf '%s\n' $files > "$ctx/overlay.files"
cp "$here/Dockerfile" "$ctx/Dockerfile"
docker build --build-arg BASE="$VLLM_BASE_IMAGE" --build-arg FORK_COMMIT="$FORK_COMMIT" \
  -t "$tag" "$ctx"
echo "built $tag from $FORK_REPO@$FORK_COMMIT ($(echo $files | wc -w) files)"
