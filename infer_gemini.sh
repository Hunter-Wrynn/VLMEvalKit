#!/bin/bash
# Gemini 2.5 Pro MathVista评测脚本

export LMUData="/data/xyc/mhx/route/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=6
echo $CUDA_VISIBLE_DEVICES


# 运行评测
python /data/xyc/mhx/route/VLMEvalKit/run.py \
  --config /data/xyc/mhx/route/VLMEvalKit/config/MathVerse/gpt5nano.json \
  --work-dir /data/xyc/mhx/route/VLMEvalKit/outputs/MathVerse \
  --verbose \
  --mode all

