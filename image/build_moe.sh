#!/usr/bin/env bash
# Build ONLY vLLM's _moe_C extension, from v0.23.0 with PR #46639 applied.
#
# Why this is small: PR #46639 touches csrc/.../marlin_moe_wna16/, which compiles into
# the _moe_C target (~24 translation units at sm80), NOT the _C target (335 TUs).
# vLLM supports single-target builds and the kernel loads via torch.ops, so a rebuilt
# _moe_C.abi3.so drops into an existing install. ~6 minutes instead of all day.
#
# Run inside nvidia/cuda:13.0.1-devel-ubuntu22.04 -- 22.04, NOT 24.04. See README.md.
# Needs no GPU. ~40 GB of scratch at /work.
#
# Output: /work/out/_moe_C.abi3.so
set -euo pipefail

export TMPDIR=/work/tmp PIP_CACHE_DIR=/work/pipcache CCACHE_DIR=/work/ccache
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # 8.0 = A100. Keeps it small.
export MAX_JOBS=${MAX_JOBS:-12}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

PATCH_DIR=${PATCH_DIR:-/work/patches}
mkdir -p /work/{tmp,pipcache,ccache,src,out}

echo "### toolchain"
nvcc --version | tail -1
command -v git   >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends git ccache patch python3-pip; }
# jammy's apt cmake is 3.22; vLLM needs >= 3.26
command -v cmake >/dev/null || python3 -m pip install -q "cmake>=3.26,<4" ninja
# torch must match the version in the runtime image you will COPY the .so into
python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda)"

echo "### source: vLLM v0.23.0"
cd /work/src
[ -d vllm/.git ] || git clone --filter=blob:none --branch v0.23.0 \
  https://github.com/vllm-project/vllm.git vllm
cd vllm && git checkout -f v0.23.0 && git clean -fdx csrc vllm >/dev/null 2>&1 || true
git log -1 --oneline

echo "### apply PR #46639"
# The PR is against csrc/libtorch_stable/moe/...; v0.23.0 has csrc/moe/...
sed -e 's|csrc/libtorch_stable/moe/|csrc/moe/|g' \
    -e 's|"libtorch_stable/quantization/|"quantization/|g' \
    "$PATCH_DIR/pr46639.diff" > /work/pr_local.diff

# CUDA hunks only. The Python-side change is applied to marlin_moe.py separately (it
# ships in the serving image, not in the .so), and the buildkite yaml is irrelevant here.
python3 - <<'PY'
import re
d = open('/work/pr_local.diff').read()
parts = re.split(r'(?m)^diff --git ', d)
keep = [p for p in parts if p.startswith('a/csrc/')]
open('/work/pr_cuda.diff', 'w').write(''.join('diff --git ' + p for p in keep))
print('cuda hunks kept:', len(keep))
PY

patch -p1 --fuzz=5 -l -i /work/pr_cuda.diff || {
  echo "!!! patch failed - see .rej"; find csrc -name '*.rej' | head; exit 1; }

# GOTCHA: --fuzz=5 can place `#include "core/batch_invariant.hpp"` INSIDE
# `namespace MARLIN_NAMESPACE_NAME { ... }`, which fails to compile with
#   marlin_moe_wna16::vllm has no member kFloat16
# The include must sit above the namespace opening. Check it.
if awk '/^namespace MARLIN_NAMESPACE_NAME/{ns=1} /batch_invariant.hpp/{if(ns) exit 1}' \
     csrc/moe/marlin_moe_wna16/ops.cu; then
  echo "include placement OK"
else
  echo "!!! batch_invariant.hpp landed inside the namespace - move it above" >&2; exit 1
fi
grep -n "batch_invariant" csrc/moe/marlin_moe_wna16/ops.cu | head -3

echo "### configure (target: _moe_C only)"
cmake -G Ninja -B build -S . \
  -DVLLM_PYTHON_EXECUTABLE="$(which python3)" \
  -DVLLM_TARGET_DEVICE=cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  -DCMAKE_INSTALL_PREFIX=. 2>&1 | tail -15

echo "### build _moe_C"
time cmake --build build --target _moe_C -j "$MAX_JOBS" 2>&1 | tail -25

echo "### artifact"
find build -name "_moe_C*.so" -exec ls -lh {} \; -exec cp {} /work/out/ \;

echo "### sanity: glibc floor (must be <= the runtime image's; 22.04 gives 2.34)"
objdump -T /work/out/_moe_C.abi3.so | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1

echo "### Python side: marlin_moe.py"
# PR #46639 also flips MarlinExperts._supports_batch_invariance() to True (this is what
# lets --moe-backend=marlin be selected under VLLM_BATCH_INVARIANT at all) and pins
# block_size_m at the two batch-invariant sites. Then our delta changes that pin from
# the PR's 64 to 16 -- bitwise-identical output, measurably faster. See README.md.
python3 - <<'PY'
import re
d = open('/work/pr_local.diff').read()
parts = re.split(r'(?m)^diff --git ', d)
keep = [p for p in parts if p.startswith('a/vllm/')]
open('/work/pr_python.diff', 'w').write(''.join('diff --git ' + p for p in keep))
print('python hunks kept:', len(keep))
PY
patch -p1 --fuzz=5 -l -i /work/pr_python.diff
patch -p1 --fuzz=5 -l -i "$PATCH_DIR/block_size_m-16.diff"

MM=vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
grep -A1 'if envs.VLLM_BATCH_INVARIANT' "$MM" | grep -c 'block_size_m = 16' \
  | grep -qx 2 || { echo "!!! expected block_size_m = 16 at BOTH sites" >&2; exit 1; }
cp "$MM" /work/out/marlin_moe.py

echo "### done -- build context is /work/out"
ls -l /work/out
echo "BUILD_DONE"
