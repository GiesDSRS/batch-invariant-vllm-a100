#!/usr/bin/env bash
# Deterministic, batch-invariant gpt-oss-120b on 1 A100 80GB (sm80).
# Flags are grouped and ordered exactly as in FLAGS.md.
#
# NOTE: nothing may interrupt the backslash continuations below, not a blank
# line and not a `#` comment. That is why the group labels are written as
# `# ...` command substitutions, which expand to nothing.

export VLLM_BATCH_INVARIANT=1

exec vllm serve openai/gpt-oss-120b \
  `# ---- Correctness / determinism ----` \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  `# deliberately NOT passing --enforce-eager` \
  `# ---- Speed and memory only ----` \
  --moe-backend=marlin \
  --gpu-memory-utilization=0.98 \
  --max-model-len=65536 \
  --max-num-seqs=32 \
  --max-num-batched-tokens=8192 \
  --no-scheduler-reserve-full-isl \
  --speculative-config='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  `# deliberately NOT passing --kv-cache-dtype; leave it bf16`
