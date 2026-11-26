#!/usr/bin/env bash
# Host settings the published numbers were measured with (run as root;
# start.sh calls it through sudo unless HOST_TUNE=0).
#   host-tune.sh <gpu list, e.g. 0,1> <model dir> [PLE path]
# 1. 170 W power limit on both cards.
# 2. 16 KB readahead on the drive holding the PLE table (default: model drive).
#    PLE reads small random rows, so large readahead turns each into a big read.
# 3. Drop the page cache and compact memory, so the engine's pinned host
#    buffers get contiguous pages. Without it decode often starts in a slower
#    mode (~28 vs ~25 ms per step on our host). This empties the whole page
#    cache, so the weights load from disk again (~100 s).
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "host-tune.sh: run as root" >&2; exit 1; }
gpus=${1:?gpu list} model_dir=${2:?model dir}
ple_path=$(realpath -e -- "${3:-$model_dir}")
src=$(findmnt --first-only --noheadings --output SOURCE --nofsroot --target "$ple_path")
for g in $(echo "$gpus" | tr ',' ' '); do
  nvidia-smi -i "$g" -pl 170 >/dev/null
done
if [ -b "$src" ]; then
  disk=$(lsblk -dnro PKNAME "$src" | sed -n '1p')
  dev=$src
  if [ -n "$disk" ]; then
    dev=/dev/$disk
  fi
  blockdev --setra 32 "$dev"
  echo "host tuned: readahead 16 KB on $dev (PLE path: $ple_path)"
else
  echo "host-tune.sh: skipping readahead for $ple_path (mount source $src is not a block device)" >&2
fi
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
echo "host tuned: 170 W on GPUs $gpus, page cache dropped"
