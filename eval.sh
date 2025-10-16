#!/bin/bash
# 对已有的 InternVL3.5-1B MathVista 推理结果进行评估

export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"

cd /data/xyc/mhx/rtbench/VLMEvalKit

# 只运行评估，不进行推理
# --reuse: 自动重用最新的推理结果文件
python run.py \
    --config config/Mathvista/GPT5.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse

echo ""
echo "========================================"
echo "评估完成！"
echo "评估结果保存在最新的时间戳目录中"
echo "======================================"

