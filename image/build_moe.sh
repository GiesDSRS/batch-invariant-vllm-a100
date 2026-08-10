#!/usr/bin/env bash
# Builds the two files the Dockerfile copies. See README.md.
#
#   run in   nvidia/cuda:13.0.1-devel-ubuntu22.04, no GPU, ~40 GB at /work
#   input    $PATCH_DIR (default /work/patches)
#   output   /work/out/{_moe_C.abi3.so, marlin_moe.py}
#
# pipefail is load-bearing: steps pipe into `tail`, and without it a failed
# build returns tail's exit code and the script copies out a stale .so.
set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TMPDIR=/work/tmp PIP_CACHE_DIR=/work/pipcache CCACHE_DIR=/work/ccache
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # 8.0 = A100
export MAX_JOBS=${MAX_JOBS:-12}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

PATCH_DIR=${PATCH_DIR:-/work/patches}
mkdir -p /work/{tmp,pipcache,ccache,src,out}

command -v git   >/dev/null || { apt-get update -qq; apt-get install -y -qq --no-install-recommends git ccache patch python3-pip; }
command -v cmake >/dev/null || python3 -m pip install -q "cmake>=3.26,<4" ninja   # jammy apt has 3.22

cd /work/src
[ -d vllm/.git ] || git clone --filter=blob:none --branch v0.23.0 \
  https://github.com/vllm-project/vllm.git vllm
cd vllm && git checkout -f v0.23.0 && git clean -fdx csrc vllm >/dev/null 2>&1 || true

# PR #46639 targets csrc/libtorch_stable/moe/; v0.23.0 has csrc/moe/
sed -e 's|csrc/libtorch_stable/moe/|csrc/moe/|g' \
    -e 's|"libtorch_stable/quantization/|"quantization/|g' \
    "$PATCH_DIR/pr46639.diff" > /work/pr.diff

# a/csrc/ hunks compile into the .so, a/vllm/ hunks patch marlin_moe.py
split_diff() {
python3 - "$1" "$2" <<'PY'
import re, sys
parts = re.split(r'(?m)^diff --git ', open('/work/pr.diff').read())
keep = [p for p in parts if p.startswith(sys.argv[1])]
open(sys.argv[2], 'w').write(''.join('diff --git ' + p for p in keep))
PY
}

split_diff a/csrc/ /work/cuda.diff
patch -p1 --fuzz=5 -l -i /work/cuda.diff

# --fuzz=5 can land the batch_invariant.hpp include INSIDE
# `namespace MARLIN_NAMESPACE_NAME`, which fails to compile. It must sit above.
awk '/^namespace MARLIN_NAMESPACE_NAME/{ns=1} /batch_invariant.hpp/{if(ns) exit 1}' \
  csrc/moe/marlin_moe_wna16/ops.cu \
  || { echo "batch_invariant.hpp is inside the namespace, move it above" >&2; exit 1; }

cmake -G Ninja -B build -S . \
  -DVLLM_PYTHON_EXECUTABLE="$(which python3)" \
  -DVLLM_TARGET_DEVICE=cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  -DCMAKE_INSTALL_PREFIX=.
cmake --build build --target _moe_C -j "$MAX_JOBS"
find build -name "_moe_C*.so" -exec cp {} /work/out/ \;

split_diff a/vllm/ /work/python.diff
patch -p1 --fuzz=5 -l -i /work/python.diff
patch -p1 --fuzz=5 -l -i "$PATCH_DIR/block_size_m-16.diff"

MM=vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
[ "$(grep -A1 'if envs.VLLM_BATCH_INVARIANT' "$MM" | grep -c 'block_size_m = 16')" = 2 ] \
  || { echo "expected block_size_m = 16 at BOTH sites" >&2; exit 1; }
cp "$MM" /work/out/marlin_moe.py

# glibc floor must be <= the runtime image's (22.04 gives 2.34)
objdump -T /work/out/_moe_C.abi3.so | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1
ls -l /work/out
