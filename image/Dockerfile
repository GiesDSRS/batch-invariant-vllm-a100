# Stock vLLM v0.23.0 + PR#46639's batch-invariant Marlin MoE kernel for sm80.
# Both files are build outputs of build_moe.sh and must sit next to this Dockerfile. See README.md for more details

FROM vllm/vllm-openai:v0.23.0

COPY _moe_C.abi3.so /usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so
COPY marlin_moe.py  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
