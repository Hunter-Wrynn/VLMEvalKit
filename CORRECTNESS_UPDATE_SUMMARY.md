# MathVista 评估结果添加 `correct` 列 - 完整方案

## ✅ 修改完成

现在 MathVista 评估会在 **两个文件** 中都添加 `correct` 列：

1. **评估详细结果**：`InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx`
2. **原始推理结果**：`InternVL3_5-1B_MathVista_MINI.xlsx` ⭐

---

## 修改内容

### 文件：`vlmeval/dataset/image_vqa.py`

修改了 `MathVista.evaluate_heuristic` 方法（第 312-372 行）

#### 修改 1：导入 `post_check` 函数

```python
# 第 313 行
from .utils.mathvista import MathVista_auxeval, MathVista_acc, post_check
```

#### 修改 2：在评估结果中添加 `correct` 列

```python
# 第 353-356 行
# Add correct column to mark whether each question is answered correctly
data['correct'] = [post_check(data.iloc[i], prefetch=False) for i in range(len(data))]

dump(data, storage)  # 保存到 _gpt-4o-mini.xlsx
```

#### 修改 3：将 `correct` 信息更新回原始文件 ⭐ 新增

```python
# 第 358-367 行
# Also update the original eval_file with correct column
data_with_correct = load(storage)
if 'correct' in data_with_correct.columns:
    original_data = load(eval_file)
    if 'correct' not in original_data.columns or not all(original_data['correct'] == data_with_correct['correct']):
        # Add/update res, log, and correct columns to original file
        original_data['res'] = data_with_correct['res']
        original_data['log'] = data_with_correct['log']
        original_data['correct'] = data_with_correct['correct']
        dump(original_data, eval_file)  # 保存回原始文件
```

---

## 文件说明

### 评估前

推理完成后，只有一个文件：

```
outputs/MathVista_MINI/InternVL3_5-1B/T20251016_Gxxxxxxxx/
└── InternVL3_5-1B_MathVista_MINI.xlsx  # 只有 prediction, answer
```

**包含的列**：
- `index`, `question`, `image`, `prediction`, `answer`, `task`, `skills`, ...
- ❌ 没有 `res`, `log`, `correct`

### 评估后（新版本）

```
outputs/MathVista_MINI/InternVL3_5-1B/T20251016_Gxxxxxxxx/
├── InternVL3_5-1B_MathVista_MINI.xlsx              # ✅ 更新了！包含 correct
├── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx  # ✅ 包含 correct
├── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.pkl
└── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini_score.csv
```

**两个 xlsx 文件都包含**：
- 原有列：`index`, `question`, `prediction`, `answer`, `task`, `skills`, ...
- ✅ 新增：`res` - GPT 提取的答案
- ✅ 新增：`log` - 提取日志
- ✅ 新增：`correct` - 是否回答正确（True/False）

---

## 文件内容示例

### InternVL3_5-1B_MathVista_MINI.xlsx

| index | question | prediction | answer | res | log | correct |
|-------|----------|------------|--------|-----|-----|---------|
| 1 | What is 2+2? | The answer is 4 | 4 | 4 | Succeed | True |
| 2 | What is 3*5? | I think it's 16 | 15 | 16 | Succeed | False |
| 3 | Choose A or B | I choose A | A | A | Prefetch succeed | True |

### InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx

完全相同的内容！

---

## 使用方法

### 1. 重新评估（推荐）

如果您想为已有的推理结果添加 `correct` 列：

```bash
cd /data/xyc/mhx/rtbench/VLMEvalKit

# 删除旧的评估结果（保留推理结果）
rm outputs/MathVista_MINI/InternVL3_5-1B/T20251016_Gxxxxxxxx/*_gpt-4o-mini*

# 重新评估
bash eval.sh
```

### 2. 评估新模型

对新的推理结果进行评估，会自动包含 `correct` 列：

```bash
# 推理
python run.py \
    --config config/Mathvista/internvl35_4b_only_config.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode infer

# 评估（会自动添加 correct 到两个文件）
python run.py \
    --config config/Mathvista/internvl35_4b_only_config.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode eval \
    --reuse
```

### 3. 完整流程（推理 + 评估）

```bash
# 一次性完成推理和评估
python run.py \
    --config config/Mathvista/internvl35_1b_only_config.json \
    --work-dir ./outputs/MathVista_MINI \
    --mode all \
    --verbose
```

---

## 验证修改

### 检查原始文件

```python
import pandas as pd

# 加载原始推理结果文件
df = pd.read_excel('outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI.xlsx')

# 检查是否包含 correct 列
print("Columns:", df.columns.tolist())
assert 'correct' in df.columns, "❌ correct column not found!"
print("✅ correct column exists!")

# 查看前几行
print(df[['index', 'question', 'answer', 'res', 'correct']].head(10))

# 统计准确率
accuracy = df['correct'].sum() / len(df) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}%")
```

### 检查详细结果文件

```python
# 加载详细结果文件
df_detail = pd.read_excel('outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx')

# 应该和原始文件内容一致
assert 'correct' in df_detail.columns
print("✅ Both files have correct column!")

# 验证两个文件的 correct 列一致
if len(df) == len(df_detail):
    assert all(df['correct'] == df_detail['correct'])
    print("✅ correct columns match in both files!")
```

---

## 数据分析示例

现在您可以直接从原始文件进行分析，无需加载 `_gpt-4o-mini` 文件：

```python
import pandas as pd

# 直接加载原始文件即可
df = pd.read_excel('outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI.xlsx')

# 1. 总体准确率
print(f"Overall Accuracy: {df['correct'].mean() * 100:.2f}%")

# 2. 按任务类型统计
task_stats = df.groupby('task').agg({
    'correct': ['sum', 'count', 'mean']
}).round(4)
task_stats.columns = ['Correct', 'Total', 'Accuracy']
task_stats['Accuracy'] = task_stats['Accuracy'] * 100
print("\nAccuracy by Task:")
print(task_stats.sort_values('Accuracy', ascending=False))

# 3. 查看错误的题目
wrong_df = df[df['correct'] == False]
print(f"\nWrong Answers: {len(wrong_df)}/{len(df)}")
print(wrong_df[['index', 'question', 'answer', 'res', 'prediction']].head())

# 4. Prefetch 成功率和准确率
prefetch_success = df[df['log'] == 'Prefetch succeed']
print(f"\nPrefetch Success Rate: {len(prefetch_success)/len(df)*100:.2f}%")
print(f"Prefetch Accuracy: {prefetch_success['correct'].mean()*100:.2f}%")

# 5. 按技能统计（如果有多个技能，需要展开）
import ast
df['skills_list'] = df['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [x])
df_exploded = df.explode('skills_list')
skill_stats = df_exploded.groupby('skills_list')['correct'].agg(['sum', 'count', 'mean'])
skill_stats['accuracy'] = skill_stats['mean'] * 100
print("\nAccuracy by Skill:")
print(skill_stats.sort_values('accuracy', ascending=False))
```

---

## 与其他数据集的对比

### VQA 类数据集（如 TextVQA, GQA）

这些数据集已经有类似的字段：

```python
# ImageVQADataset 评估后包含
data['eval_match'] = [...]  # 类似 correct
data['eval_score'] = [...]  # 评分
```

### MCQ 类数据集（如 MMBench, BLINK）

circular 模式下包含：

```python
data['hit'] = [...]  # 对错标记
```

### 现在 MathVista 保持一致！

```python
data['correct'] = [...]  # 对错标记 ✅
data['res'] = [...]      # 提取的答案
data['log'] = [...]      # 提取日志
```

---

## 优势

### 1. 两个文件都有完整信息
- ✅ 原始文件：方便直接查看和分析
- ✅ 详细文件：保留评估的完整信息

### 2. 无需额外处理
- ✅ 评估时自动更新两个文件
- ✅ 无需手动合并数据

### 3. 向后兼容
- ✅ 如果文件已有 `correct` 列，会检查并更新
- ✅ 不会重复添加或破坏数据

### 4. 便于分析
- ✅ 直接加载原始文件即可分析
- ✅ 所有信息都在一个地方

---

## 注意事项

### 1. 文件更新时机

只在评估时更新，推理时不会添加这些列：

| 模式 | 原始文件内容 |
|------|------------|
| `--mode infer` | 只有 `prediction`, `answer` |
| `--mode eval` | 添加 `res`, `log`, `correct` ✅ |
| `--mode all` | 添加 `res`, `log`, `correct` ✅ |

### 2. 重新评估

如果重新评估，会更新这三列：
- `res` - 可能因为 API 结果不同而变化
- `log` - 更新为新的提取日志
- `correct` - 根据新的 `res` 重新计算

### 3. 文件一致性

评估完成后，确保两个文件的 `correct` 列一致：

```python
df1 = pd.read_excel('InternVL3_5-1B_MathVista_MINI.xlsx')
df2 = pd.read_excel('InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx')
assert all(df1['correct'] == df2['correct']), "Files mismatch!"
```

---

## 总结

### ✅ 修改完成

1. **评估详细文件**（`_gpt-4o-mini.xlsx`）包含 `correct` 列
2. **原始推理文件**（`.xlsx`）也会更新 `correct` 列 ⭐
3. 两个文件内容一致，都包含完整的评估信息

### 📝 使用建议

- **分析时**：直接使用原始文件（`InternVL3_5-1B_MathVista_MINI.xlsx`）
- **调试时**：两个文件都可以，内容相同
- **共享时**：共享原始文件即可，信息完整

### 🎯 下次运行

下次运行 `eval.sh` 时，会自动在两个文件中添加 `correct` 列！

```bash
bash eval.sh
```

完成！✅

