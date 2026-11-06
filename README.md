# Qwen3.8-Flash-Next on 2× CMP 170HX

**Qwen3.8-Flash-Next (W4A16) on two NVIDIA CMP 170HX cards, pipeline-parallel
with vLLM: 234 tok/s for one user (code), 827 tok/s across eight, 6,200–7,400
tok/s prefill from 4k to 64k prompts, the full 262,144-token window, image and
video input, and HumanEval 157/164, GSM8K 1,272/1,319.** Measured on PCIe Gen2 x4
links.

Serves [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
with upstream vLLM and the [vllm-cmp170hx](https://github.com/AlexBrandes/vllm)
plugin in a container: an OpenAI-compatible API with text, image and video
input, native MTP speculative decoding, reasoning and tool calls. The ~102 GB
N-gram (PLE) table stays on the NVMe drive and is read through the page cache;
everything else is on the two GPUs.

```bash
git clone https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next
cd cmp170hx-qwen38-flash-next
./download.sh     # checkpoint at the pinned revision (~170 GB)
./start.sh        # tunes the host (sudo), starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `qwen3.8-flash-next-w4a16`.
`./stop.sh` stops it. Settings go in `.env` (see `.env.example`).

## Requirements

- 2× NVIDIA CMP 170HX, each exposing 64 GiB. No P2P or link modification needed:
  pipeline parallelism sends one activation per step between the cards.
- Linux, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~180 GB free on an **NVMe SSD** for the checkpoint. The PLE table is read from
  it during every forward pass, so keep other heavy writers off that drive.
- Enough RAM for a useful page cache over the PLE table (measured with 93 GB).
- `sudo` for `host-tune.sh` (or `HOST_TUNE=0`; see below).

## Benchmarks

Measured 2026-11-05 with the files in this repository (image
`ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:a87f558b57e60b4f3c796d7be11d12199ce77f393df6ed1c40181ce932717757`, started by
`start.sh`), using the [sparkDash](https://github.com/MiaAI-Lab/sparkDash)
benchmark protocol at commit `3e6d4ef`: greedy, thinking off, output forced to 400
tokens, sent to the engine endpoint. Each cell is the **median of three runs, with
the range in brackets**; sparkDash itself makes one pass per level.

**Decode, code**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 234.4 (232.4–241.3) | 234.4 (232.4–241.3) | 79 ms |
| 2 | 377.7 (377.5–380.5) | 193.8 (191.5–194.9) | 105 ms |
| 4 | 581.9 (562.0–588.7) | 158.1 (152.3–160.0) | 171 ms |
| 8 | 827.3 (809.8–830.2) | 115.6 (111.1–115.9) | 243 ms |

**Decode, json**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 208.1 (207.9–213.1) | 208.1 (207.9–213.1) | 74 ms |
| 2 | 335.3 (330.7–345.1) | 172.3 (169.7–176.2) | 106 ms |
| 4 | 543.4 (542.8–555.3) | 144.0 (143.4–145.2) | 169 ms |
| 8 | 877.1 (868.5–889.1) | 117.1 (116.0–118.6) | 227 ms |

**Decode, prose**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 162.5 (161.5–163.7) | 162.5 (161.5–163.7) | 65 ms |
| 2 | 225.2 (224.7–231.6) | 118.7 (117.0–120.9) | 94 ms |
| 4 | 372.2 (362.2–379.9) | 99.6 (96.3–101.3) | 143 ms |
| 8 | 564.5 (541.7–577.4) | 75.9 (75.1–78.7) | 205 ms |

**Decode, structured**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 264.1 (262.6–264.4) | 264.1 (262.6–264.4) | 67 ms |
| 2 | 434.9 (386.5–436.1) | 221.7 (208.8–222.3) | 99 ms |
| 4 | 697.6 (695.7–698.2) | 180.4 (180.4–180.5) | 135 ms |
| 8 | 1,087.0 (1,079.2–1,087.4) | 144.6 (142.3–145.7) | 197 ms |

**Prefill** (one cold request)

| Prompt | Tokens | Prefill tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,134 | 6,219 (5,884–6,426) | 0.74 s |
| 8k | 8,227 | 6,689 (6,611–7,124) | 1.31 s |
| 16k | 16,423 | 7,286 (7,202–7,352) | 2.33 s |
| 32k | 32,806 | 7,426 (7,379–7,428) | 4.50 s |
| 64k | 65,575 | 6,437 (6,322–6,491) | 10.27 s |
| 128k | 131,112 | 5,381 (5,281–5,482) | 24.45 s |

**Decode at depth** (our addition, not part of sparkDash: one request, the prose
prompt after filler, 400 tokens)

| Context | Tokens | Decode tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,125 | 162.9 (157.6–166.5) | 0.77 s |
| 8k | 8,224 | 161.4 (161.0–170.5) | 1.25 s |
| 16k | 16,415 | 159.7 (144.4–161.9) | 2.36 s |
| 32k | 32,797 | 156.2 (150.9–162.0) | 4.54 s |
| 64k | 65,566 | 153.0 (147.5–153.6) | 9.61 s |
| 128k | 131,104 | 126.3 (121.8–146.8) | 22.16 s |

### Quality

Same container and session, greedy, thinking off.

| Check | Result |
|---|---|
| HumanEval, pass@1 (164 problems) | **157/164** (95.7%) |
| GSM8K (1,319 test questions) | **1,272/1,319** (96.4%) |
| Image and video input | 7/7: a text control, three single images (4000×3000 to 64×64), two images in one message, two short videos |
| Needle retrieval at 4k, 16k, 32k and 64k | 16/16 |
| 262,144-token window | 262,143 prompt tokens + 1 output token accepted; 262,144 refused |
| Long prompt (173,273 tokens) | answered; 26.0 s to first token |

- **GSM8K** copies vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in chat mode: 5-shot,
  1,024 tokens, last integer in the reply.
- **HumanEval** uses EvalPlus's instruct prompt and OpenAI's `check_correctness`
  program (3 s limit).
- **Chat mode only.** This is a chat-only model: on a raw `/v1/completions`
  prompt it ends its turn at once (`<|im_end|>`, p ≈ 0.99), so completion-mode
  scores (GSM8K 49, HumanEval 129) measure the prompt format, not the model.
- **For scale:** Intel's model card reports GSM8K 0.9682 for this checkpoint
  (0.9673 for BF16), measured with its own harness.

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's vLLM branch at [`ea2ef399a6`](https://github.com/AlexBrandes/vllm/commit/ea2ef399a614ea72b65b15804000342f629a850f), Python files over the wheel; and the vllm-cmp170hx plugin at [`bbba5c27cf`](https://github.com/AlexBrandes/vllm/commit/bbba5c27cf9e367e3652b876d53ea7e1c80f16e5) |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:a87f558b57e60b4f3c796d7be11d12199ce77f393df6ed1c40181ce932717757` (tag `engine-ea2ef399a6-bbba5c27cf`): CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it |
| Checkpoint | [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897) at revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, used unmodified |
| MTP draft head | the target's `lm_head`, quantized to NVFP4 for the drafter at load (`draft_lm_head_quantization`) |
| PLE table | vLLM's `disk_offload` (`--engram-config {"disk_offload":true}`): rows stay on the NVMe; the plugin gathers each chunk's rows from the page cache on 32 threads (`VLLM_PLE_DISK_READ_THREADS`) |
| Parallelism | pipeline parallel 2, layer split 26/22 (`VLLM_PP_LAYER_PARTITION`) |
| Scheduling | a new prompt prefills at most 560 tokens per step beside running decodes (`VLLM_MIXED_PREFILL_CHUNK`) |
| Cache | BF16 KV, prefix caching, block size 1,680, FP32 Mamba state, up to 8 sequences |
| Graphs | `FULL_AND_PIECEWISE` CUDA graphs with breakable PLE segments |

The vLLM branch is upstream plus the fork's commits; the
[plugin](https://github.com/AlexBrandes/vllm/tree/plugin) registers the CMP 170HX
(sm_80) platform, the scheduler hooks and the PLE reader. Among the fork's changes
this model uses:

1. **Bounded prefill beside running decodes.** A new long prompt keeps advancing
   while other requests decode.
2. **Threaded page-cache gather for PLE rows.** Replaces `disk_offload`'s serial
   per-row reads; `VLLM_PLE_DISK_READ_THREADS=0` restores them.
3. **The MTP drafter shares the target's embedding under pipeline parallelism**,
   where its own is a placeholder.
4. **Equal sub-block prefill chunks** when a Mamba block is wider than one chunk.

### Host settings (`host-tune.sh`)

`start.sh` runs it through `sudo`; `HOST_TUNE=0` skips it.

- **170 W power limit** on both cards.
- **16 KB readahead** on the model's drive. PLE rows are small random reads.
- **Drop the page cache and compact memory before start.** The engine's pinned
  host buffers then land on contiguous pages; without it decode often starts in
  a mode about 10% slower. It empties the whole page cache, so other workloads
  on the host lose theirs, and the weights load from disk again.

### Reasoning

The checkpoint's chat template is used unmodified. Requests that enable thinking
get the template's default effort, `xhigh`; send `reasoning_effort` (`low`,
`medium`, `xhigh`) to change it. All numbers above have thinking off.

## Caveats

- One host: 2× CMP 170HX at PCIe Gen2 x4, Intel Core Ultra 5 225, 93 GB RAM, a
  consumer QLC NVMe (Kingston NV3 2 TB), Ubuntu 24.04, driver 610.43.02. Other
  hosts have not been measured.
- The PLE table is read from the NVMe while serving. Another heavy writer on the
  same drive stalls it: on our host a KV-cache disk tier on that drive cut
  long-prompt prefill by two thirds.
- Greedy decoding at 8 requests in flight with MTP is not bitwise repeatable;
  HumanEval's failing set can vary by a task between sessions.
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)). The engine is Apache-2.0, the model
has its own licence; see [NOTICE](NOTICE) for every component and credit.
