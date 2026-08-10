# Serving image, exactly as used in production for the run this repo documents.
#
# Stock vLLM v0.23.0, plus the batch-invariant Marlin MoE kernel (upstream PR #46639)
# compiled for sm80 and its Python-side counterpart.
#
# Only two files differ from stock vLLM. The kernel loads through torch.ops, so a
# rebuilt _moe_C.abi3.so drops straight into an existing install -- see README.md for
# how to produce it (~6 minutes on a CPU node; you do NOT need to build all of vLLM).
#
# Both must sit next to this Dockerfile in the build context:
#   _moe_C.abi3.so   built by build_moe.sh, from /work/out/
#   marlin_moe.py    also from /work/out/ -- vLLM's copy with pr46639.diff and then
#                    patches/block_size_m-16.diff applied
#
# Verify the built IMAGE, not just the build -- see README.md, "Verify the image".
FROM vllm/vllm-openai:v0.23.0

# `quark` is imported by the MXFP4 *emulation* MoE backend's per-forward dequant
# (mxfp4_utils.py::_dequant_mxfp4), which the stock image does not ship. We serve with
# --moe-backend=marlin, which does not reach that code path, so this is very likely
# unnecessary -- but it was present for every measurement in this repo and we have not
# tested its removal, so it stays. Dropping it is the first thing to try if you want a
# leaner image; report back if it works.
#
# Pulled from PyPI at build time. It brings scipy/onnx/pandas and friends, but NOT
# torch/triton/numpy -- the base already has numpy 2.x.
RUN pip install --no-cache-dir amd-quark

# Warm-import so quark's C++ emulation kernel is JIT-compiled into this layer rather
# than on first request. Only matters if you actually use the emulation backend.
RUN python3 -c "import quark" || true

COPY _moe_C.abi3.so /usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so
COPY marlin_moe.py  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
