#!/bin/bash
# InternVL 3.5系列 MathVista评测脚本

export LMUData="/data/xyc/mhx/route/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=0,1
echo $CUDA_VISIBLE_DEVICES


# 运行评测
python run.py \
    --config /data/xyc/mhx/route/VLMEvalKit/config/MathVision/GPT5.json \
    --work-dir ./outputs/MathVision \
    --mode all \
    --verbose \
    --reuse



