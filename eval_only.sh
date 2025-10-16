#!/bin/bash
# 只运行评估（eval）的脚本
# 前提：已经完成了 infer 步骤，有推理结果文件

export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

cd /data/xyc/mhx/rtbench/VLMEvalKit

# 只运行评估，不进行推理
# 需要先运行过 --mode infer，生成推理结果文件
python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse

# 注意：
# 1. --mode eval: 只运行评估，跳过推理
# 2. --reuse: 重用已有的推理结果文件
# 3. 确保 work-dir 和 config 与 infer 时使用的一致

