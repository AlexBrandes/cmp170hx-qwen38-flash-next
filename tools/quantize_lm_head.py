#!/usr/bin/env python3
"""Quantize the Qwen3.8-Flash-Next-W4A16-AutoRound `lm_head` to a packed
W4A16 (int4, group-128, symmetric) GPTQ/AutoGPTQ checkpoint for the MTP
draft head.

Reads ONLY the `lm_head.weight` tensor (bf16, [vocab=248320, hidden=2560],
~1.27 GiB) from the model's safetensors shard; everything else in the
checkpoint is never touched. Output tensors use exactly the GPTQ layout that
vLLM's `AutoGPTQLinearMethod` / Marlin path consumes (same as the model's
own W4A16 linears):

    lm_head.qweight   int32        [K // 8,    N]      8 nibbles / word, LSB first
    lm_head.scales    float16      [K // 128,  N]      one scale per 128-input group
    lm_head.qzeros    int32        [K // 128, N // 8]  constant zero-point 8
                                                    (nibble 8 in all positions;
                                                    the symmetric Marlin kernel
                                                    ignores it)

with K = hidden_size, N = vocab_size. Quantization is RTN by default;
`--mse-scale-search` adds a cheap GPTQ-free improvement: per group, the
scale factor f in a small grid (f * amax/7) is chosen by minimizing the
reconstruction MSE instead of fixed round-to-nearest scaling.

Validation here is deliberately light (the file is re-read and must match
the in-memory tensors exactly; shapes/dtypes are asserted).

CPU-only, peak RAM kept under ~2 GB. Run inside the engine image with the
model dir mounted read-only (see ../download.sh).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import safetensors.torch as st
from safetensors import safe_open

BITS = 4
GROUP = 128
CLIP_LO, CLIP_HI = -8.0, 7.0
# Per-group scale candidates: f * (amax / 7), f = 1.0 is plain RTN.
DEFAULT_SCALE_GRID = [1.0, 0.975, 0.95, 0.925, 0.9, 0.85, 0.8, 0.75, 0.7]
NIBBLE_SHIFTS = [0, 4, 8, 12, 16, 20, 24, 28]


def read_source_weight(model_dir: Path) -> torch.Tensor:
    """Load just lm_head.weight (bf16 [N, K]) from the shard named by the
    safetensors index."""
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"]["lm_head.weight"]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")
    assert w.ndim == 2, w.shape
    return w.contiguous()


def quantize(
    w: torch.Tensor,
    mse_scale_search: bool,
    grid: list[float],
    search_chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 with per-group scales.

    w: [N, K] fp32. Returns (q_u8 [N, K] values 0..15, scales [N, K/G] fp32).
    """
    N, K = w.shape
    assert K % GROUP == 0
    G = K // GROUP
    groups = w.view(N, G, GROUP)

    q = torch.empty(N, K, dtype=torch.uint8)
    scales = torch.empty(N, G, dtype=torch.float32)

    amax = groups.abs().amax(dim=2)  # [N, G]
    amax = amax.clamp_min(torch.finfo(torch.float32).tiny)

    f_grid = torch.tensor(grid, dtype=torch.float32)
    base = amax / 7.0  # [N, G]

    for r0 in range(0, N, search_chunk_rows):
        r1 = min(r0 + search_chunk_rows, N)
        w_c = groups[r0:r1]  # [c, G, 128]
        b_c = base[r0:r1]  # [c, G]
        if not mse_scale_search:
            s_keep = b_c.unsqueeze(2)  # [c, G, 1]
        else:
            # Per-candidate reconstruction error; track the best per group
            # sequentially over candidates so memory stays one [c,G,128] fp32
            # tensor at a time.
            err = torch.full((r1 - r0, G), float("inf"), dtype=torch.float32)
            best_f = torch.zeros(r1 - r0, G, dtype=torch.int64)
            for fi, f in enumerate(f_grid):
                s = b_c * f  # [c, G]
                qv = torch.clamp(
                    torch.round(w_c / s.unsqueeze(2)), CLIP_LO, CLIP_HI
                )
                e = (w_c - qv * s.unsqueeze(2)).pow(2).sum(dim=2)  # [c, G]
                m = e < err
                err[m] = e[m]
                best_f[m] = fi
            s_keep = (b_c * f_grid[best_f]).unsqueeze(2)  # [c, G, 1]

        q_c = torch.clamp(torch.round(w_c / s_keep), CLIP_LO, CLIP_HI)
        q[r0:r1] = (q_c + 8.0).to(torch.uint8).view(r1 - r0, K)
        scales[r0:r1] = s_keep.squeeze(2)
    return q, scales


def pack(q_u8: torch.Tensor, scales: torch.Tensor, block: int = 32768) -> dict:
    """GPTQ/AutoGPTQ packed layout, all tensors in the dtypes vLLM's
    AutoGPTQ path creates: qweight int32 [K/8, N] (LSB-first nibble words,
    two's-complement wrap for high bit patterns), scales fp16 [K/G, N],
    qzeros int32 [K/G, N/8] constant 0x88888888 (zero-point 8)."""
    N, K = q_u8.shape
    G = K // GROUP
    qweight = torch.empty(N, K // 8, dtype=torch.int32)
    for r0 in range(0, N, block):
        r1 = min(r0 + block, N)
        blk = q_u8[r0:r1].view(r1 - r0, K // 8, 8).to(torch.int64)
        acc = torch.zeros(r1 - r0, K // 8, dtype=torch.int64)
        for i in range(8):
            acc |= blk[:, :, i] << (4 * i)
        qweight[r0:r1] = acc.to(torch.int32)
    qzeros = torch.full((G, N // 8), -2001583616, dtype=torch.int32)
    return {
        "lm_head.qweight": qweight.t().contiguous(),
        "lm_head.scales": scales.t().contiguous().to(torch.float16),
        "lm_head.qzeros": qzeros,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--group-size", type=int, default=GROUP)
    ap.add_argument("--mse-scale-search", action="store_true",
                    help="GPTQ-free per-group scale search (MSE-optimal f grid)")
    ap.add_argument("--scale-grid", type=float, nargs="+", default=DEFAULT_SCALE_GRID)
    ap.add_argument("--search-chunk-rows", type=int, default=256)
    args = ap.parse_args()

    assert args.group_size == GROUP
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))

    t0 = time.time()
    w_bf16 = read_source_weight(args.model_dir)  # [N, K] bf16
    N, K = w_bf16.shape
    w = w_bf16.to(torch.float32)
    del w_bf16
    print(
        f"source lm_head.weight: shape=({N}, {K}) file={args.model_dir.name}",
        flush=True,
    )

    q_u8, scales = quantize(
        w,
        mse_scale_search=args.mse_scale_search,
        grid=args.scale_grid,
        search_chunk_rows=args.search_chunk_rows,
    )
    del w
    packd = pack(q_u8, scales)
    del q_u8, scales

    # ---- write + cheap exact round-trip ---------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    st.save_file(packd, str(args.out))

    with safe_open(str(args.out), framework="pt", device="cpu") as f:
        rt = {k: f.get_tensor(k) for k in packd}
    assert set(rt) == set(packd)
    for k in packd:
        assert rt[k].dtype == packd[k].dtype, (k, rt[k].dtype, packd[k].dtype)
        assert tuple(rt[k].shape) == tuple(packd[k].shape), (
            k,
            rt[k].shape,
            packd[k].shape,
        )
        assert torch.equal(rt[k], packd[k]), f"round-trip mismatch in {k}"
    del rt

    sz = args.out.stat().st_size
    h = hashlib.sha256()
    with open(args.out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    report = {
        "file": str(args.out),
        "sha256": h.hexdigest(),
        "bytes": sz,
        "method": "MSE scale search (GPTQ-free)" if args.mse_scale_search else "RTN",
        "scale_grid": args.scale_grid if args.mse_scale_search else None,
        "bits": BITS,
        "group_size": args.group_size,
        "symmetric": True,
        "source_index": str(args.model_dir / "model.safetensors.index.json"),
        "shape_NK": [N, K],
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in packd.items()
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    report_path = args.out.with_name(args.out.name + ".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
