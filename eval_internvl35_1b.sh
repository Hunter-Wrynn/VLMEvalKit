#!/bin/bash
# 对已有的 InternVL3.5-1B MathVista 推理结果进行评估

export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"

# MathVista 评估需要 OpenAI API key（使用 gpt-4o-mini 作为 judge）
# 请设置您的 OpenAI API key：
# export OPENAI_API_KEY="sk-your-api-key-here"

# 如果已经在 .env 文件中配置了 API key，则无需手动 export
# 如果没有，请取消下面这行的注释并填入您的 key：
# export OPENAI_API_KEY="sk-your-api-key-here"

# eval 模式通常不需要 GPU，但如果评估过程需要可以设置
# export CUDA_VISIBLE_DEVICES=7

cd /data/xyc/mhx/rtbench/VLMEvalKit

echo "========================================"
echo "对 InternVL3.5-1B 的推理结果进行评估"
echo ""
echo "工作原理："
echo "1. 系统会自动从 outputs/MathVista_MINI/InternVL3_5-1B/ 下"
echo "   查找最新的推理结果（T20251015_G6f67cef5/）"
echo "2. 创建今天的新目录（例如 T20251016_Gxxxxxxxx/）"
echo "3. 自动复制推理结果到新目录"
echo "4. 在新目录中进行评估"
echo ""
echo "无需手动指定旧的时间戳目录！"
echo "========================================"

# 只运行评估，不进行推理
# --reuse: 自动重用最新的推理结果文件
python run.py \
    --config config/Mathvista/internvl35_1b_only_config.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse

echo ""
echo "========================================"
echo "评估完成！"
echo "查看新生成的目录: outputs/MathVista_MINI/InternVL3_5-1B/"
echo "评估结果保存在最新的时间戳目录中"
echo "======================================"

