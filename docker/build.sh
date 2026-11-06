#!/usr/bin/env bash
# Build the engine image from the pinned sources.
#   docker/build.sh [--tag TAG]
# The Dockerfile fetches every source by the commit and sha256 in versions.env:
# upstream vLLM's wheel, the fork's vLLM branch and the vllm-cmp170hx plugin.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../versions.env"
tag="$ENGINE_IMAGE_LOCAL"
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) tag=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
cp "$here/Dockerfile" "$ctx/Dockerfile"
args=()
for v in UPSTREAM_COMMIT VLLM_WHEEL VLLM_WHEEL_SHA256 VLLM_BRANCH_COMMIT VLLM_BRANCH_SHA256 \
         PLUGIN_COMMIT PLUGIN_SHA256; do
  args+=(--build-arg "$v=${!v}")
done
docker build "${args[@]}" -t "$tag" "$ctx"
echo "built $tag: upstream $UPSTREAM_COMMIT, vLLM branch $VLLM_BRANCH_COMMIT, plugin $PLUGIN_COMMIT"
