# Flags used and what they do

Flags fall into two groups:

- **"Correctness"/Determinism.** These are what you need for batch invariant, reproducible results: `VLLM_BATCH_INVARIANT`, `--no-async-scheduling`, `--no-enable-prefix-caching`, and *not* passing `--enforce-eager`. Touch any of these and your hashes change.
- **Speed and memory only.** These make inference faster without affecting determinism: `--moe-backend`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--no-scheduler-reserve-full-isl`, the n-gram `--speculative-config`, the KV cache dtype, and the Marlin `block_size_m`. Tune them freely, results stay consistent and valid.

## Contents

**"Correctness"/Determinism**

- [`VLLM_BATCH_INVARIANT=1`](#vllm_batch_invariant1)
- [`--no-async-scheduling`](#--no-async-scheduling)
- [`--no-enable-prefix-caching`](#--no-enable-prefix-caching)
- [No `--enforce-eager`: leave CUDA graphs ON](#no---enforce-eager-leave-cuda-graphs-on)

**Speed and memory only**

- [`--moe-backend=marlin`](#--moe-backendmarlin)
- [`--gpu-memory-utilization=0.98`](#--gpu-memory-utilization098)
- [`--max-model-len=65536`](#--max-model-len65536)
- [`--max-num-seqs=32`](#--max-num-seqs32)
- [`--max-num-batched-tokens=8192`](#--max-num-batched-tokens8192)
- [`--no-scheduler-reserve-full-isl`](#--no-scheduler-reserve-full-isl)
- [`--speculative-config`: n-gram, and why not EAGLE3](#--speculative-config-n-gram-and-why-not-eagle3)
- [No `--kv-cache-dtype`: leave bf16](#no---kv-cache-dtype-leave-bf16)

[Additional Details](#additional-details)

---

# "Correctness"/Determinism

## `VLLM_BATCH_INVARIANT=1`

Swaps in batch-size-independent matmul, softmax, RMSNorm and attention kernels (the mechanism from Thinking Machines' *Defeating Nondeterminism in LLM Inference*, merged
into vLLM behind this flag). `gpt-oss-120b` is on vLLM's explicitly validated list, and sm80 meets the >= 8.0 requirement. Set it on the **server**, in the serving process's environment.

**It is necessary but NOT sufficient!** It pins the *kernels*, not the *scheduler*, and the docs never mention scheduler configuration at all... (at the time of writing: 07/08/2026; see [latest](https://docs.vllm.ai/en/latest/features/batch_invariance/)/[stable](https://docs.vllm.ai/en/stable/features/batch_invariance/))

> *Note* - On its own:
>
> - Short prompts were perfectly reproducible, since a ~20-token prompt only decodes a handful of tokens and each token is one argmax. With so few decisions, the odds that any of them sits on a near-tie, where a sub-ULP difference flips which token wins, are tiny. The request also lives for so few scheduler steps that the batch around it barely changes underneath it.
> - At real prompt sizes (~12,000 tokens) it does **not** give reproducible output, because the flag pins the kernels but not the scheduler. A long prompt is split across chunked-prefill steps and then decodes for hundreds or thousands of tokens, so the request spans many steps whose batch membership depends on wall-clock timing (see [`--no-async-scheduling`](#--no-async-scheduling) below). Every extra token is another chance to land on a near-tie, and one flipped token cascades into a different answer from there on.

## `--no-async-scheduling`

Turns async scheduling off, so the scheduler only composes a step's batch once the previous step has retired. Batch membership then depends on which requests exist instead of when the CPU happened to get there.

vLLM enables async scheduling by default (`SchedulerConfig.async_scheduling = None` means `on`). It overlaps CPU scheduling with GPU execution: the next step's batch is composed before the current step has retired, so **which batch a given request lands in depends on wall-clock timing**.

This is undocumented, unguarded, and upstream's own determinism tests do not catch it. Fix sent upstream: [vllm-project/vllm#51287](https://github.com/vllm-project/vllm/pull/51287).

## `--no-enable-prefix-caching`

Turns off KV prefix reuse, so every request computes its own prompt KV instead of picking up blocks an earlier request already wrote.

Every request in this workload shares a long identical system prefix, and we did not want the amount of work a request does to depend on who arrived first: with caching on, a request that hits prefills only the uncached suffix and attends to someone else's blocks as context, while the same request arriving first prefills the whole thing.

> **Note:** We never isolated this one, it was off from the start and never tested on its own. Under complete batch invariance those two paths should be bitwise identical, so this may well be unnecessary. Upstream disables it in all three of their determinism tests (`enable_prefix_caching=False`), and vLLM does not auto-disable it under `VLLM_BATCH_INVARIANT`, but nothing documents it either way.

## No `--enforce-eager`: leave CUDA graphs ON

`--enforce-eager` disables CUDA graph capture and runs every forward through the normal eager path. It is often reached for as the "maximum determinism" lever. Here it seems to be the opposite: **CUDA graph capture pads the decode batch to fixed capture sizes**, which removes the batch-composition churn that "incomplete" batch invariance was exposing. The padded shape is the same regardless of how many sequences are really resident, so graphs do for free what upstream batch invariance does not yet do. **We do not pass it**.

---

# Speed and memory only

## `--moe-backend=marlin`

Forces the MXFP4 matmul directly against the packed 4-bit weights, unpacking them inside the GEMM, instead of materializing a bf16 copy of every expert's weights on each forward. Big speed up and it also drops the ~3.96 GiB per-forward transient that forces `--enforce-eager` on the emulation path.

The emulation backend (weights stay MXFP4-packed, dequantized to bf16 per forward) also stays "correct"/deterministic, just slow:

|                             | emulation  | marlin                  |
| --------------------------- | ---------- | ----------------------- |
| 10 documents, 10 concurrent | 594 s      | 115 s (**5.2×**) |
| single-stream decode        | 3.68 tok/s | 21 tok/s (5.7×)        |

> Needs the PR #46639 build ([image/](../image/README.md)). Without it, the fused-MoE gate (`modular_kernel.py::_supports_batch_invariance()`) prunes every MXFP4 MoE kernel that has not opted in, which on sm80 leaves only the Triton emulation one, so `--moe-backend auto` raises *"No MXFP4 MoE backend supports the deployment configuration"*. This **is not a hardware limit**, Marlin is gpt-oss's normal A100 path.

## `--gpu-memory-utilization=0.98`

Sets the fraction of **total** GPU memory vLLM may use for everything: weights, activations and the KV cache. KV is the residual, whatever is left after weights and the activation reserve, so raising this grows the cache.

0.98 is a safe ceiling on an 80GB A100 for this model, tested in both directions:

| util           | outcome                                                                     |
| -------------- | --------------------------------------------------------------------------- |
| 0.97           | safe, doubles the OOM margin, ~9% slower when the queue is deep             |
| **0.98** | **adopted**                                                           |
| 0.985          | starts clean, then OOMs**mid-inference** on a long prompt             |
| 0.9922         | OOM during CUDA graph capture (the value vLLM's startup message recommends) |

## `--max-model-len=65536`

Caps the longest sequence the server will accept, prompt plus output. Anything longer gets rejected.

Sized to the workload: longest document ~44k tokens + system prompt + 9k output ≈ 54.4k worst case. The full context window is 131072, but in this case leaving it there inflates startup profiling and the graph capture range. Does not itself consume KV.

## `--max-num-seqs=32`

Caps how many sequences the scheduler runs at once.

An oversized value inflates startup profiling activation and the CUDA graph capture range, consuming KV: 256 gives 232,576 KV tokens, **32 gives 256,466 (+10.3%)**. The GPU saturates around this many concurrent sequences, and with prompts this long KV only holds ~14 to 23 resident at a time.

Do not read 32 as a concurrency limit for clients. vLLM queues beyond it, though beware that queued requests can sit long enough to hit a client timeout. Set the client's request timeout above ~600 s and they will wait rather than die.

## `--max-num-batched-tokens=8192`

Caps how many tokens go into a single forward pass, prefill chunks and decode tokens together.

Empiric rule is to take the biggest value that does not break for a batch job. Smaller splits the same prompt across more steps, which is usually slower for a batch workload, and the memory it frees buys a few percent of KV at best. Bigger is where it breaks: 16384 consumes ~a quarter of the KV cache, starves decode to 5-6 tok/s against 220-250, and **profiles fine and starts clean, then OOMs on the first real batch and kills the EngineCore** (`EngineDeadError`). 8192 is the largest found that holds.

## `--no-scheduler-reserve-full-isl`

Stops the scheduler reserving KV for each request's full declared input length (ISL) up front. With `--max-model-len=65536` and typical prompts being a fraction of that, the reservation is enormous dead space and admission stalls needlessly.

## `--speculative-config`: n-gram, and why not EAGLE3

Drafts candidate tokens by looking for repeats already in the prompt, then verifies them in a single pass, so a step can emit several tokens instead of one.

`n-gram` yields a free **1.4×**, and unlike a draft model it costs essentially no KV (-0.6%). It suits this workload well, since long JSON outputs with repeating keys that quote the source text verbatim are exactly what prompt-lookup drafts well. **It is output-neutral at temperature 0**, verified bitwise identical.

EAGLE3 is deterministic too, just not worth it here: the draft model costs ~75k KV tokens, cutting residency from ~23 to ~14, which cancels its 1.81 acceptance length. Draft models tend to work well when concurrency-limited; this workload is KV/bandwidth-limited.

## No `--kv-cache-dtype`: leave bf16

Would quantize the KV cache to a smaller dtype, trading precision for more cache. Both were tested but did not work well.

**FP8 is not available on sm80.** `fp8` / `fp8_e4m3` fails to JIT in the Triton KV-write kernel (`type fp8e4nv not supported in this architecture`). `fp8_e5m2` is rejected too (by `attention.py:467`).

**`int8_per_token_head` works and is determinism-clean** (+88% KV, residency 23 → 32), but it runs **32% slower** at equal concurrency, likely due to per-token quant/dequant cost in attention. Most importantly, output quality drifted against the bf16 baseline on most sampled documents.

---

## Additional Details

Hardware: one NVIDIA A100-SXM4-80GB (compute capability 8.0).

Model: `openai/gpt-oss-120b`, native MXFP4, ~61-65 GB of weights. vLLM v0.23.0.

Workload: tuned on 100 sample documents, prompts mean 11,347 tokens
(p90 14,108, longest ~45k), outputs 108-2,075 tokens of structured JSON.
