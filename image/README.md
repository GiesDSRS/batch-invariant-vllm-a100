# vLLM Docker image

Scripts and notes to build vLLM v0.23.0:

- `patches/pr46639.diff`: upstream PR#46639
- `patches/block_size_m-16.diff`: our delta on top
- `build_moe.sh`: builds both patches, on CPU in ~6 min. Produces:
  - `_moe_C.abi3.so`
  - `marlin_moe.py`
- `Dockerfile`: stock `vllm/vllm-openai:v0.23.0` with those two files copied in

To build the image, run:

```bash
# 1. build the two files. no GPU, ~6 min, needs ~40 GB at /scratch/moe
docker run --rm -v "$PWD/patches:/work/patches" -v "$PWD/build_moe.sh:/build.sh" \
  -v /scratch/moe:/work nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /build.sh

# 2. build the image from them
docker build -f Dockerfile -t vllm-mxfp4-marlin-batch-invariant:v0.23.0-bm16 /scratch/moe/out
```

---

## `patches/pr46639.diff`

The batch-invariant Marlin MoE kernel is **[vLLM PR #46639](https://github.com/vllm-project/vllm/pull/46639), authored upstream.** We applied it unmodified and then build on top.

It adds the CUDA kernel, and flips `MarlinExperts._supports_batch_invariance()` to `True`. This makes `--moe-backend=marlin` selectable under `VLLM_BATCH_INVARIANT=1`. Its 3 csrc files compile into the `.so` and its 1 Python file becomes `marlin_moe.py`.

The PR targets `csrc/libtorch_stable/moe/`; v0.23.0 has `csrc/moe/`, and `build_moe.sh` rewrites the paths.

> **Check the PR's status before building anything.** If/when it merges, released vLLM would carry the kernel for every architecture and none of this is needed, you would just install vLLM and pass `--moe-backend=marlin`. Still open as of 2026-08-10.

## `patches/block_size_m-16.diff`

`block_size_m = 64` to `16`, at the two batch-invariant sites in `marlin_moe.py`. Python only. Bitwise-identical output, measurably faster on sm80.

## `build_moe.sh`

Rebuilds only `_moe_C` (~24 translation units at sm80).

Build inside **Ubuntu 22.04, dont do it inside 24.04**. A 24.04 build links `GLIBC_2.38` and dies at import against the runtime image's glibc 2.35.

`torch` must match the version in the runtime image you will copy the `.so` into.

It aborts on its own if any of three things is wrong:

- the `batch_invariant.hpp` include landed inside `namespace MARLIN_NAMESPACE_NAME` (`--fuzz=5` can do this, and it fails to compile),
- `block_size_m` is not 16 at **both** sites, or
- the glibc floor is too high.

## `Dockerfile`

Stock base plus the two build outputs. Nothing else is installed.

### Verify the image

Inside the built image:

```bash
L=/usr/local/lib/python3.12/dist-packages/vllm

python3 -c "import vllm._moe_C, torch; print([o for o in dir(torch.ops._moe_C) if 'marlin' in o])"
objdump -T $L/_moe_C.abi3.so | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1
grep -c 'block_size_m = 16' $L/model_executor/layers/fused_moe/experts/marlin_moe.py
```

Marlin ops must resolve, max `GLIBC_` must not exceed the runtime's 2.35 (a 22.04 build reads 2.34), and `block_size_m = 16` must show up twice.
