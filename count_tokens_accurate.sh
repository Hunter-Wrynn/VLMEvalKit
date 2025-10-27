#!/bin/bash
# 精确Token计数脚本 - 真实计算图像token数量

export LMUData="/data/xyc/mhx/route/VLMEvalKit/dataset"

# 配置参数
CONFIG_FILE="config/Mathvista/Qwen2_5_VL_72B.json"
MODEL_PATH="Qwen/Qwen2.5-VL-72B-Instruct"  # 使用HuggingFace模型路径
MODEL_TYPE="qwen2.5vl"
WORK_DIR="./token_stats_accurate/Qwen72B_MathVista_MINI"

echo "=========================================="
echo "精确Token计数工具 - 真实计算图像token"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "模型路径: $MODEL_PATH"
echo "模型类型: $MODEL_TYPE"
echo "输出目录: $WORK_DIR"
echo "=========================================="
echo ""

# 运行精确token计数
python count_tokens_accurate.py \
    --config "$CONFIG_FILE" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --work-dir "$WORK_DIR"

echo ""
echo "=========================================="
echo "精确Token计数完成！"
echo "结果保存在: $WORK_DIR"
echo "=========================================="
echo ""
echo "输出文件说明:"
echo "  - *.pkl: 完整的统计结果（推荐使用）"
echo "  - *.csv: CSV格式的统计结果"
echo "  - *.xlsx: Excel格式的统计结果"
echo "  - *_summary.json: 统计摘要"
echo ""
echo "主要改进:"
echo "  ✓ 真实计算图像token数量（基于Qwen VL的patch机制）"
echo "  ✓ 考虑图像尺寸调整和patch分割"
echo "  ✓ 提供详细的图像处理信息"
echo "  ✓ 更准确的token统计"