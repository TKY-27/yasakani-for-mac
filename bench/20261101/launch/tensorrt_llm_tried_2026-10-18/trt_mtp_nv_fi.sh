#!/usr/bin/env bash
# trt_mtp_nv_fi (2026-10-18): TensorRT-LLM with the FlashInfer attention backend; the default TRTLLM backend failed in its XQA NVRTC JIT on the RTX 5090
source "$(dirname "$0")/common.sh"
exec $TR/trtllm-serve $M/nvidia_NVFP4 --host 127.0.0.1 --port $PORT --reasoning_parser qwen3_5 --config "$(dirname "$0")/trt_mtp_fi.yaml" 2>&1 | tee -a $LOGD/trt_mtp_nv_fi.log
