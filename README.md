# Batch Invariant vLLM for openai/gpt-oss-120b in nvidia A100 GPUs

This repo contains all the code for a bitwise-reproducible, batch-invariant `openai/gpt-oss-120b` on a **single A100 80GB (sm80)** served vLLM with the faster marlin kernels (as compared to the MoE XXXX...)

The most important thing in this repo:

> **`VLLM_BATCH_INVARIANT=1` on its own is not enough.** It pins the *kernels*, not the *scheduler*. vLLM's async scheduling is on by default and composes the next step's batch before the current one retires, so batch membership depends on wall-clock timing and nondeterminism gets back in with no warning and no error.
> You also need `--no-async-scheduling`. See [Upstream](#upstream).



## Contents

| path                                                    | what                                                                  |
| ------------------------------------------------------- | --------------------------------------------------------------------- |
| [serve/vllm-serve.sh](serve/vllm-serve.sh)               | The serve command used to spin DSRS LLM server up                     |
| [serve/FLAGS.md](serve/FLAGS.md)                         | every flag, why it is there, what it was measured against             |
| [test/batch_invariance.py](test/batch_invariance.py)     | self-contained batch-invariance test (OpenAI SDK)                     |
| [test/README.md](test/README.md)                         | how to run it, and why naive determinism tests pass on broken configs |
| [image/](image/README.md)                                | scripts to build the sm80 Marlin MoE kernel and the serving image     |
| [image/patches/](image/patches)                          | upstream vLLM PR#46639, plus our two-line delta                       |

## Serve

With an image that has vLLM's batch-invariant Marlin MoE kernel compiled for sm80 (see [image/](image/README.md), you likely have to build it, as it is not in any released vLLM at the time of writing), run:

```Shell
VLLM_BATCH_INVARIANT=1 \
vllm serve openai/gpt-oss-120b \
  --moe-backend=marlin \
  --gpu-memory-utilization=0.98 \
  --max-model-len=65536 \
  --max-num-seqs=32 \
  --max-num-batched-tokens=8192 \
  --no-scheduler-reserve-full-isl \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --speculative-config='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

Clients then use `temperature=0` and must not set a seed-dependent sampler. 

## Test it

You can use the script below to start a test that sends the same long probe prompt at several concurrency levels and checks that every completion hashes identically.

```python
import hashlib, concurrent.futures as cf
from openai import OpenAI

client = OpenAI(base_url="...", 
                api_key="...")

probe = "Summarize the following text.\n\n" + ("The quick brown fox. " * 2000)

def ask(_):
    m = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
          {
            "role": "user", 
            "content": probe
          }
        ],
        temperature=0,
    ).choices[0].message

  # Hash the reasoning trace too. gpt-oss can spend the whole token budget
    # reasoning and leave `content` empty -- and hashing "" passes every test.
    text = (getattr(m, "reasoning_content", None) or "") + (m.content or "")
    assert text.strip(), "empty completion"
    return hashlib.sha256(text.encode()).hexdigest()[:12]

for n in (1, 5, 10, 50):
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        hashes = set(ex.map(ask, range(n)))
    print(f"batch {n:>2}: {hashes}")
    # every line must print the SAME single hash
```

> *Note that:* **The probe must be long**. Short prompts are reproducible even on "non-deterministic" configs

The full version, with staggered-length filler traffic that actually stresses the scheduler, is [test/batch_invariance.py](test/batch_invariance.py). To run it:

```sh
BASE_URL=https://your-server/v1 API_KEY=... uv run test/batch_invariance.py
```

`uv` reads [pyproject.toml](pyproject.toml), which pins `openai` and `tqdm`. With pip instead: `pip install .` then the same command without `uv run`.

## Upstream

We sent the async-scheduling fix to vLLM: [vllm-project/vllm#51287](https://github.com/vllm-project/vllm/pull/51287). It makes vLLM disable async scheduling on its own under `VLLM_BATCH_INVARIANT=1`, so nobody else has to find this the hard way.

## Status and credit

The batch-invariant Marlin MoE kernel is **[vLLM PR #46639](https://github.com/vllm-project/vllm/pull/46639), authored upstream (not by us)**, but used as a stepping stone. This repo adds:

1. compiling it for **sm80** against vLLM **v0.23.0** (the PR targets `csrc/libtorch_stable/moe/`; v0.23.0 has `csrc/moe/`),
2. `block_size_m = 16` instead of the PR's 64, bitwise-identical output, measured faster ([why](image/README.md#block_size_m)),
3. the serving configuration around it, above all: `--no-async-scheduling`.


**Hopefully: If PR #46639 merges, most of this repo becomes unnecessary** ...; Released vLLM images already ship compiled kernels for every architecture, so you would just install vLLM and pass something like `--moe-backend=marlin`. Worth checking the PR before building.





-----

Built at [Data Science Research Services](https://giesbusiness.illinois.edu/),
Gies College of Business, University of Illinois Urbana-Champaign, 

for a research workload requiring reproducible LLM extraction over ~17k documents.

Licensed [Apache-2.0](LICENSE) (matching vLLM).
