# 为 MathVista 评估结果添加每道题的对错标记

## 现状分析

### 当前保存的文件

评估完成后，MathVista 会生成以下文件：

```
outputs/MathVista_MINI/InternVL3_5-1B/T20251016_Gxxxxxxxx/
├── InternVL3_5-1B_MathVista_MINI.xlsx              # 原始推理结果
├── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx  # 评估后的结果（包含提取的答案）
├── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.pkl   # pkl 格式（与 xlsx 相同）
└── InternVL3_5-1B_MathVista_MINI_gpt-4o-mini_score.csv  # 汇总的准确率
```

### 当前 xlsx 文件包含的列

**InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx** 包含：

| 列名 | 说明 |
|------|------|
| `index` | 问题索引 |
| `question` | 问题文本 |
| `prediction` | 模型原始回答 |
| `answer` | 标准答案 |
| `res` | GPT 提取的答案 |
| `log` | 提取日志（Prefetch succeed / Succeed / Failed） |
| `task` | 任务类型 |
| `skills` | 技能列表 |
| ... | 其他元数据 |

### ❌ 缺少什么？

**缺少 `correct` 或 `hit` 列，标记每道题是否回答正确！**

---

## 解决方案

### 方案 1: 修改代码添加 `correct` 列（推荐）

修改 `vlmeval/dataset/image_vqa.py` 中的 `MathVista.evaluate_heuristic` 方法。

**文件位置**: `vlmeval/dataset/image_vqa.py` 第 312-357 行

**修改方案**:

```python
@classmethod
def evaluate_heuristic(self, eval_file, **judge_kwargs):
    from .utils.mathvista import MathVista_auxeval, MathVista_acc, post_check  # 导入 post_check
    
    model = judge_kwargs['model']
    storage = get_intermediate_file_path(eval_file, f'_{model}')
    tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
    nproc = judge_kwargs.pop('nproc', 4)
    
    if not osp.exists(storage):
        data = load(eval_file)
        model = build_judge(max_tokens=128, **judge_kwargs)
        assert model.working(), 'MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        tups = [(model, line) for line in lines]
        indices = [line['index'] for line in lines]
        
        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)
        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]
        
        if len(indices):
            new_results = track_progress_rich(
                MathVista_auxeval,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=tmp_file,
            )
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans
                assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']
        
        data['res'] = [ans[idx]['res'] for idx in data['index']]
        data['log'] = [ans[idx]['log'] for idx in data['index']]
        
        # ✅ 新增：添加 correct 列
        data['correct'] = [post_check(data.iloc[i], prefetch=False) for i in range(len(data))]
        
        dump(data, storage)
    
    score = MathVista_acc(storage)
    score_pth = get_intermediate_file_path(storage, '_score', 'csv')
    dump(score, score_pth)
    return score
```

**修改内容**：
1. 导入 `post_check` 函数
2. 在保存前添加一行：计算每道题的 `correct` 状态
3. `correct` 列值为 `True`/`False` 或布尔值

---

### 方案 2: 使用脚本后处理（无需修改代码）

如果不想修改源代码，可以创建一个脚本来添加 `correct` 列。

**脚本**: `add_correctness_flag.py`

```python
#!/usr/bin/env python3
"""
为 MathVista 评估结果添加 correct 列
用法: python add_correctness_flag.py <result_file>
"""

import sys
import pandas as pd
from vlmeval.smp import load, dump
from vlmeval.dataset.utils.mathvista import post_check

def add_correctness_flag(result_file):
    """
    为 MathVista 结果文件添加 correct 列
    
    Args:
        result_file: 评估结果文件路径（pkl 或 xlsx）
    """
    # 加载数据
    print(f"Loading {result_file}...")
    data = load(result_file)
    
    # 检查是否已有 correct 列
    if 'correct' in data.columns:
        print("✓ 'correct' column already exists!")
        return
    
    # 添加 correct 列
    print("Adding 'correct' column...")
    correct_list = []
    for i in range(len(data)):
        item = data.iloc[i]
        is_correct = post_check(item, prefetch=False)
        correct_list.append(is_correct)
    
    data['correct'] = correct_list
    
    # 保存
    print(f"Saving to {result_file}...")
    dump(data, result_file)
    
    # 统计
    total = len(data)
    correct_count = sum(correct_list)
    accuracy = correct_count / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 50)
    print("✓ Successfully added 'correct' column!")
    print("=" * 50)
    print(f"Total questions: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_correctness_flag.py <result_file>")
        print("Example: python add_correctness_flag.py outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx")
        sys.exit(1)
    
    result_file = sys.argv[1]
    add_correctness_flag(result_file)
```

**使用方法**:

```bash
cd /data/xyc/mhx/rtbench/VLMEvalKit

# 为评估结果添加 correct 列
python add_correctness_flag.py outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx
```

---

### 方案 3: 批量处理所有结果文件

创建批量处理脚本：

**脚本**: `batch_add_correctness.sh`

```bash
#!/bin/bash
# 批量为所有 MathVista 评估结果添加 correct 列

cd /data/xyc/mhx/rtbench/VLMEvalKit

echo "========================================"
echo "批量添加 correct 列到 MathVista 评估结果"
echo "========================================"

# 查找所有 gpt-4o-mini.xlsx 文件
files=$(find outputs/MathVista_MINI -name "*_gpt-4o-mini.xlsx" -type f)

count=0
for file in $files; do
    echo ""
    echo "Processing: $file"
    python add_correctness_flag.py "$file"
    count=$((count + 1))
done

echo ""
echo "========================================"
echo "完成！处理了 $count 个文件"
echo "========================================"
```

---

## 实现步骤

### 推荐方案：修改代码（一劳永逸）

#### 步骤 1: 修改代码

编辑 `/data/xyc/mhx/rtbench/VLMEvalKit/vlmeval/dataset/image_vqa.py`:

```bash
vim /data/xyc/mhx/rtbench/VLMEvalKit/vlmeval/dataset/image_vqa.py
```

在第 313 行附近，找到：
```python
def evaluate_heuristic(self, eval_file, **judge_kwargs):
    from .utils.mathvista import MathVista_auxeval, MathVista_acc
```

修改为：
```python
def evaluate_heuristic(self, eval_file, **judge_kwargs):
    from .utils.mathvista import MathVista_auxeval, MathVista_acc, post_check
```

在第 351 行附近，找到：
```python
        data['res'] = [ans[idx]['res'] for idx in data['index']]
        data['log'] = [ans[idx]['log'] for idx in data['index']]
        dump(data, storage)
```

修改为：
```python
        data['res'] = [ans[idx]['res'] for idx in data['index']]
        data['log'] = [ans[idx]['log'] for idx in data['index']]
        
        # 添加 correct 列
        data['correct'] = [post_check(data.iloc[i], prefetch=False) for i in range(len(data))]
        
        dump(data, storage)
```

#### 步骤 2: 重新评估

修改代码后，重新运行评估：

```bash
bash eval.sh
```

现在生成的 xlsx 文件会包含 `correct` 列！

#### 步骤 3: 验证

```python
import pandas as pd

# 加载结果
df = pd.read_excel('outputs/MathVista_MINI/InternVL3_5-1B/InternVL3_5-1B_MathVista_MINI_gpt-4o-mini.xlsx')

# 检查 correct 列
print(df.columns)
print(df[['index', 'prediction', 'answer', 'res', 'correct']].head())

# 统计准确率
accuracy = df['correct'].mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
```

---

## 输出示例

修改后的 xlsx 文件将包含：

| index | question | prediction | answer | res | log | correct |
|-------|----------|------------|--------|-----|-----|---------|
| 1 | What is 2+2? | The answer is 4 | 4 | 4 | Succeed | True |
| 2 | What is 3*5? | I think it's 16 | 15 | 16 | Succeed | False |
| 3 | What color? | The color is red | red | red | Prefetch succeed | True |

---

## 其他数据集参考

### VQA 类数据集（已经有 correct 信息）

**ImageVQADataset** (如 TextVQA, GQA) 已经保存了每道题的对错：

```python
# vlmeval/dataset/image_vqa.py 第 88-92 行
data['eval_gt'] = [r['gt'] for r in res]
data['eval_pred'] = [r['pred'] for r in res]
data['eval_match'] = [r['match'] for r in res]      # ← 对错信息
data['eval_score'] = [np.mean(r['match']) for r in res]
```

### MCQ 类数据集（已经有 hit 信息）

**ImageMCQDataset** (如 MMBench, BLINK) 在 circular 模式下也保存了 hit：

```python
# vlmeval/dataset/utils/multiple_choice.py
data['hit'] = [x['opt'] == x['answer'] for x in data]
```

---

## 总结

### 当前状态
- ❌ MathVista 的 `_gpt-4o-mini.xlsx` **没有** `correct` 列
- ✅ 但数据中有 `res`（提取的答案）和 `answer`（标准答案）
- ✅ 可以通过 `post_check()` 函数计算是否正确

### 推荐方案
1. **修改代码**（推荐）：在 `evaluate_heuristic` 中添加 3 行代码
2. **后处理脚本**：评估完后运行脚本添加 `correct` 列

### 修改后的优势
- ✅ 每道题都能看到是否正确
- ✅ 方便分析错误case
- ✅ 可以按 task/skill 统计正确率
- ✅ 与其他数据集保持一致

需要我帮您实现这个修改吗？

