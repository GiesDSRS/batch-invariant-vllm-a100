# Building the image


| file                             | what                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `patches/pr46639.diff`         | upstream PR#46639, unmodified                                                              |
| `patches/block_size_m-16.diff` | our two-line delta on top of it                                                            |
| `build_moe.sh`                 | builds`_moe_C.abi3.so` and `marlin_moe.py` from both patches. CPU only, no GPU, ~6 min |
| `Dockerfile`                   | the serving image: stock`vllm/vllm-openai:v0.23.0` with those two files copied in        |

Scripts carry their own notes in the header comments on how to run them and what is needed for the build.

>  Note: files change relative to stock vLLM v0.23.0:
>
> * `_moe_C.abi3.so`: rebuilt with PR#46639's batch-invariant Marlin MoE CUDA kernel, compiled for sm80
> * `marlin_moe.py`: PR#46639's Python side, plus our  `block_size_m = 16` delta



The batch-invariant Marlin MoE kernel is **[vLLM PR #46639](https://github.com/vllm-project/vllm/pull/46639), authored upstream.** `patches/pr46639.diff` applies the changes in the PR unmodified.

Our contribution is three things:

1. **Compiling it for sm80 against v0.23.0.** The PR targets `csrc/libtorch_stable/moe/`; v0.23.0 has `csrc/moe/`, so the paths are rewritten (`build_moe.sh` does this).
2. **`block_size_m = 16`** instead of the PR's 64 (`patches/block_size_m-16.diff`).
3. The serving configuration around it (see [../serve/FLAGS.md](../serve/FLAGS.md)).

**Check the PR's status before building anything.** If it merges, released vLLM images will carry the kernel for every architecture and none of this is needed, you would just install vLLM and pass `--moe-backend=marlin`.
