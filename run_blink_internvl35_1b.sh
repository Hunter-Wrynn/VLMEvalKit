#!/bin/bash
# InternVL3.5-1B BLINK 评测脚本

export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=4

cd /data/xyc/mhx/rtbench/VLMEvalKit

echo "========================================"
echo "BLINK 数据集评测"
echo "模型: InternVL3.5-1B"
echo "========================================"
echo ""
echo "数据集信息："
echo "- 类型: 图像多选题 (MCQ)"
echo "- 评估: 精确匹配 (不需要 OpenAI API)"
echo ""
echo "========================================"

# 运行评测
# --mode all: 推理 + 评估
# BLINK 使用 exact_matching，不需要 OpenAI API key
python run.py \
    --config config/BLINK/internvl35_1b.json \
    --work-dir ./outputs/BLINK \
    --mode all \
    --verbose

echo ""
echo "========================================"
echo "评测完成！"
echo "结果保存在: outputs/BLINK/InternVL3_5-1B/"
echo "查看准确率: outputs/BLINK/InternVL3_5-1B/InternVL3_5-1B_BLINK_acc.csv"
echo "========================================"


