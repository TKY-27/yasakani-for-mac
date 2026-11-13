#!/usr/bin/env bash
# trt_mtp_nv (generated 2026-10-18)
source "$(dirname "$0")/common.sh"
exec $TR/trtllm-serve $M/nvidia_NVFP4 --host 127.0.0.1 --port $PORT --reasoning_parser qwen3_5 --config "$(dirname "$0")/trt_mtp.yaml" 2>&1 | tee -a $LOGD/trt_mtp_nv.log
