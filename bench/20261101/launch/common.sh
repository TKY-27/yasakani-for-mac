# Shared settings for the 2026-10-18 same-PC comparison servers (WSL2 Ubuntu 24.04, run as root). Offline: never fetch at serve time.
# Set MODELS to your model folder and SGLANG_VENV / VLLM_VENV / TRTLLM_VENV to the virtual environments (SGLang 0.5.22, vLLM 0.31.0,
# TensorRT-LLM 1.3.0rc29). Folder names under $MODELS -> Hugging Face repos:
#   nvidia_NVFP4                  nvidia/Qwen3.8-27B-NVFP4
#   gittensor_NVFP4_LMHead4       gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 (LMHead4)
#   gittensor_DSpark_NVFP4        gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4
#   redhat_speculator_dspark      RedHatAI/Qwen3.8-27B-speculator.dspark
#   z-lab_Qwen3.8-27B-DFlash2     z-lab/Qwen3.8-27B-DFlash2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
M=$MODELS
DF2=$MODELS/z-lab_Qwen3.8-27B-DFlash2
SG=$SGLANG_VENV/bin/python; [ -x "$SG" ] || SG=$SGLANG_VENV/bin/python
VL=$VLLM_VENV/bin
TR=$TRTLLM_VENV/bin
LOGD=./logs; mkdir -p "$LOGD"
PORT=30000
# FlashInfer (vLLM's fast sm_120 decode path, SGLang's attention backend) compiles its kernels with nvcc on first use: vLLM refuses
# FlashInfer when no nvcc is on PATH and no prebuilt cubin package is installed (2026-10-18: "FlashInfer backend is not available").
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH"
# 2026-10-18: 16 parallel nvcc / cicc jobs building FlashInfer's CUTLASS sm120 FP4 GEMM took the whole WSL VM down (uptime reset twice);
# FlashInfer passes MAX_JOBS to ninja -j. 4 jobs keep the build inside the 80 GB VM. The build is cached, so only the first start pays.
export MAX_JOBS=4
