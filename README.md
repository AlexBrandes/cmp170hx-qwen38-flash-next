# Qwen3.8-Flash-Next on 2× CMP 170HX

**Qwen3.8-Flash-Next (W4A16) on two NVIDIA CMP 170HX cards, pipeline-parallel
with vLLM: 226 tok/s for one user (code), 840 tok/s across eight, 6,300–7,100
tok/s prefill from 4k to 64k prompts, the full 262,144-token window, and
HumanEval 157/164, GSM8K 1,271/1,319.** Measured on PCIe Gen2 x4 links.

Serves [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
with a [vLLM v0.30.0 fork](https://github.com/AlexBrandes/vllm/tree/cmp170hx-qwen38-flash-next)
in a container: an OpenAI-compatible API with native MTP speculative decoding,
reasoning and tool calls. The ~102 GB N-gram (PLE) table stays on the NVMe
drive and is read through an mmap; everything else is on the two GPUs.

```bash
git clone https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next
cd cmp170hx-qwen38-flash-next
./download.sh     # checkpoint at the pinned revision (~170 GB) + the MTP draft head
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

Measured 2026-10-17 with the files in this repository (image
`ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957`, started by
`start.sh`), using the [sparkDash](https://github.com/MiaAI-Lab/sparkDash)
benchmark protocol at commit `3e6d4ef`: greedy, thinking off, output forced to 400
tokens, sent to the engine endpoint. Each cell is the **median of three runs, with
the range in brackets**; sparkDash itself makes one pass per level.

**Decode, code**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 226.2 (223.3–228.0) | 226.2 (223.3–228.0) | 80 ms |
| 2 | 362.9 (359.7–369.4) | 185.0 (183.7–188.1) | 109 ms |
| 4 | 560.2 (547.8–590.1) | 150.5 (149.3–160.4) | 171 ms |
| 8 | 840.4 (760.3–883.0) | 116.0 (106.5–119.1) | 231 ms |

**Decode, json**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 198.0 (197.2–198.9) | 198.0 (197.2–198.9) | 70 ms |
| 2 | 320.3 (310.7–325.3) | 164.8 (161.1–166.6) | 110 ms |
| 4 | 536.7 (487.6–554.9) | 144.9 (138.8–145.7) | 152 ms |
| 8 | 873.2 (837.5–891.9) | 118.0 (115.9–120.5) | 220 ms |

**Decode, prose**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 157.4 (154.4–157.9) | 157.4 (154.4–157.9) | 67 ms |
| 2 | 212.2 (208.3–229.3) | 113.2 (109.8–117.6) | 97 ms |
| 4 | 384.8 (320.2–390.0) | 101.3 (91.3–104.1) | 132 ms |
| 8 | 566.8 (541.4–574.7) | 76.1 (72.9–78.6) | 198 ms |

**Decode, structured**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 250.8 (232.7–251.8) | 250.8 (232.7–251.8) | 69 ms |
| 2 | 406.1 (376.4–413.4) | 207.3 (201.4–210.6) | 95 ms |
| 4 | 697.4 (672.4–707.7) | 177.7 (174.9–180.1) | 134 ms |
| 8 | 999.5 (984.6–1,001.4) | 139.1 (134.6–147.5) | 429 ms |

**Prefill** (one cold request)

| Prompt | Tokens | Prefill tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,133 | 6,295 (6,281–6,499) | 0.74 s |
| 8k | 8,229 | 6,893 (6,874–6,977) | 1.28 s |
| 16k | 16,422 | 7,096 (6,982–7,200) | 2.40 s |
| 32k | 32,808 | 7,104 (6,933–7,222) | 4.72 s |
| 64k | 65,573 | 6,901 (6,328–6,981) | 9.59 s |
| 128k | 131,108 | 5,885 (5,448–6,405) | 22.38 s |

**Decode at depth** (our addition, not part of sparkDash: one request, the prose
prompt after filler, 400 tokens)

| Context | Tokens | Decode tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,125 | 157.9 (145.4–160.7) | 0.74 s |
| 8k | 8,221 | 150.1 (149.7–151.4) | 1.27 s |
| 16k | 16,414 | 152.5 (151.2–156.3) | 2.39 s |
| 32k | 32,797 | 148.4 (146.2–157.6) | 4.63 s |
| 64k | 65,564 | 147.1 (142.2–150.9) | 9.39 s |
| 128k | 131,102 | 141.8 (140.7–142.3) | 21.43 s |

### Quality

Same container and session, greedy, thinking off.

| Check | Result |
|---|---|
| HumanEval, pass@1 (164 problems) | **157/164** (95.7%) |
| GSM8K (1,319 test questions) | **1,271/1,319** (96.4%) |
| Needle retrieval at 8k, 32k and 128k | 12/12 |
| 262,144-token window | 262,143 prompt tokens + 1 output token accepted; 262,144 refused |
| Long prompt (173,273 tokens) | answered; 26.3 s to first token |

- **GSM8K** copies vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in chat mode: 5-shot,
  1,024 tokens, last integer in the reply.
- **HumanEval** uses EvalPlus's instruct prompt and OpenAI's `check_correctness`
  program (3 s limit).
- **Chat mode only.** This is a chat-only model: on a raw `/v1/completions`
  prompt it ends its turn at once (`<|im_end|>`, p ≈ 0.99), so completion-mode
  scores (GSM8K 22, HumanEval 134) measure the prompt format, not the model.
- **For scale:** Intel's model card reports GSM8K 0.9682 for this checkpoint
  (0.9673 for BF16), measured with its own harness.

## What runs

| | |
|---|---|
| Engine | [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm/tree/cmp170hx-qwen38-flash-next) branch `cmp170hx-qwen38-flash-next` at [`02acf67ca10bb0c6efddd5454facb3bfe5dba7d9`](https://github.com/AlexBrandes/vllm/commit/02acf67ca10bb0c6efddd5454facb3bfe5dba7d9): vLLM v0.30.0 + 6 commits (below) |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957` (tag `engine-02acf67ca1`): the official `vllm/vllm-openai:v0.30.0` image, pinned by digest, plus the fork's 18 changed Python files. `docker/build.sh` rebuilds it |
| Checkpoint | [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897) at revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, used unmodified |
| MTP draft head | `lm_head` quantized to int4 (group 128, symmetric) by `tools/quantize_lm_head.py` from the checkpoint, ~6 s on CPU; the file's sha256 is pinned in `versions.env` |
| Parallelism | pipeline parallel 2, layer split 26/22 (`VLLM_PP_LAYER_PARTITION`) |
| Cache | BF16 KV, prefix caching, block size 1,680, FP32 Mamba state, up to 8 sequences |
| Graphs | `FULL_AND_PIECEWISE` CUDA graphs with breakable PLE segments |

The fork's commits, oldest first:

1. **Pipeline parallelism and an NVMe-mmapped PLE table.** Builds on the community
   CMP 170HX work credited in [NOTICE](NOTICE).
2. **Mamba state under pipeline parallelism.** Deferred recurrent-state copies use
   request-slot block tables; batch rows had changed owners by then.
3. **W4A16 draft-only `lm_head` for MTP.**
4. **CUDA-graph-safe PLE lookup and host-side N-gram ids.** Includes the
   stale-tail fix that made graph-mode outputs depend on request history.
5. **Bounded prefill beside running decodes.** A new long prompt keeps advancing
   while other requests decode.
6. **Fused QSA prepare.** A backport of vllm-project/vllm#57097.

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
  HumanEval scored 157 in three sessions with the failing set varying by a task.
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)). The engine is Apache-2.0, the model
has its own licence; see [NOTICE](NOTICE) for every component and credit.
