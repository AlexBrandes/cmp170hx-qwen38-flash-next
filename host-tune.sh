#!/usr/bin/env bash
# Host settings the published numbers were measured with (run as root;
# start.sh calls it through sudo unless HOST_TUNE=0).
#   host-tune.sh <gpu list, e.g. 0,1> <model dir>
# 1. 170 W power limit on both cards.
# 2. 16 KB readahead on the drive holding the model: the PLE table is read
#    in small random rows, and a large readahead turns each into a big read.
# 3. Drop the page cache and compact memory, so the engine's pinned host
#    buffers get contiguous pages. Without it decode often starts in a slower
#    mode (~28 vs ~25 ms per step on our host). This empties the whole page
#    cache, so the weights load from disk again (~100 s).
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "host-tune.sh: run as root" >&2; exit 1; }
gpus=${1:?gpu list} model_dir=${2:?model dir}
for g in $(echo "$gpus" | tr ',' ' '); do
  nvidia-smi -i "$g" -pl 170 >/dev/null
done
src=$(df --output=source "$model_dir" | tail -1)
disk=$(lsblk -no pkname "$src" 2>/dev/null | head -1)
dev=/dev/${disk:-$(basename "$src")}
[ -b "$dev" ] && blockdev --setra 32 "$dev"
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
echo "host tuned: 170 W on GPUs $gpus, readahead 16 KB on $dev, page cache dropped"
