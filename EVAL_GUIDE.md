# VLMEvalKit 评估指南

## 三种运行模式

VLMEvalKit 支持三种运行模式：

### 1. `--mode all`（默认模式）
同时运行推理（infer）和评估（eval）
```bash
python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode all \
    --verbose
```

### 2. `--mode infer`
只运行推理，不运行评估
```bash
python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode infer \
    --verbose
```

### 3. `--mode eval`
只运行评估，不运行推理（需要已有推理结果）
```bash
python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse
```

## 从 infer 结果进行 eval 的完整流程

### 步骤 1：运行推理（infer）

```bash
#!/bin/bash
export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"
export CUDA_VISIBLE_DEVICES=4

cd /data/xyc/mhx/rtbench/VLMEvalKit

python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode infer \
    --verbose
```

推理完成后，结果会保存在：
- `./outputs/MathVista_MINI/Ovis2-1B/T<日期>_G<commit>/`
- 文件格式：`Ovis2-1B_MathVista_MINI.xlsx` 或 `.tsv`

### 步骤 2：运行评估（eval）

**重要参数说明：**
- `--mode eval`: 只运行评估，跳过推理步骤
- `--reuse`: 重用最新的推理结果文件（必须添加）
- `--work-dir`: 必须与 infer 时使用的一致
- `--config`: 必须与 infer 时使用的一致

```bash
#!/bin/bash
export LMUData="/data/xyc/mhx/rtbench/VLMEvalKit/dataset"

cd /data/xyc/mhx/rtbench/VLMEvalKit

python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse
```

## 输出文件结构

```
outputs/MathVista_MINI/
└── Ovis2-1B/
    ├── T20251015_G6f67cef5/          # 时间戳_Git哈希命名的目录
    │   ├── Ovis2-1B_MathVista_MINI.xlsx      # 推理结果
    │   ├── Ovis2-1B_MathVista_MINI_time.pkl  # 推理时间
    │   └── Ovis2-1B_MathVista_MINI_acc.json  # 评估结果（eval后生成）
    └── Ovis2-1B_MathVista_MINI.xlsx  # 软链接到最新结果
```

## 常见问题

### Q1: eval 模式找不到推理结果？
**解决方法：**
1. 确保 `--work-dir` 和 `--config` 与 infer 时完全一致
2. 必须添加 `--reuse` 参数
3. 检查输出目录下是否有推理结果文件

### Q2: 如何只评估特定的已有结果？
如果你已经有推理结果文件，可以：
1. 确保文件在正确的目录结构下
2. 使用 `--mode eval --reuse` 参数
3. 系统会自动找到最新的推理结果进行评估

### Q3: 如何重新评估？
如果要重新评估已有的推理结果：
```bash
python run.py \
    --config config/Mathvista/Ovis2_1b.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --verbose \
    --reuse \
    --reuse-aux 0  # 不重用辅助文件，强制重新评估
```

## 其他有用参数

- `--api-nproc 4`: 并行 API 调用数（用于需要 GPT 评估的数据集，如 MathVista）
- `--retry 3`: API 调用失败时的重试次数
- `--judge gpt-4o-mini`: 指定评估用的 judge 模型
- `--ignore`: 忽略失败的推理样本

## MathVista 特殊说明

MathVista 数据集会使用 GPT-4o-mini 作为 judge 进行评估（见 run.py 第 382 行）。
确保设置了正确的 OpenAI API key：
```bash
export OPENAI_API_KEY="your-api-key"
```

或在 `.env` 文件中配置：
```
OPENAI_API_KEY=your-api-key
```

