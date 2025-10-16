#!/bin/bash
# InternVL 3.5系列 MathVista评测脚本

export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

cd /data/xyc/mhx/rtbench/VLMEvalKit

# 运行评测
python run.py \
    --config config/Mathvista/Qwen2.5vl_7B.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode infer \
    --verbose



