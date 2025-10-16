#!/usr/bin/env python3
"""
OCRBench结果分析脚本
为每个样本添加正确性判断，并生成详细的评估结果
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

def evaluate_sample_correctness(row):
    """
    判断单个样本是否正确
    """
    predict = str(row['prediction'])
    answers = eval(row['answer']) if isinstance(row['answer'], str) else row['answer']
    category = row['category']
    
    # 对于手写数学表达式识别，去除空格和换行符进行匹配
    if category == 'Handwritten Mathematical Expression Recognition':
        for answer in answers:
            answer_clean = answer.strip().replace('\n', ' ').replace(' ', '')
            predict_clean = predict.strip().replace('\n', ' ').replace(' ', '')
            if answer_clean in predict_clean:
                return True
    else:
        # 对于其他类别，转换为小写并去除换行符进行匹配
        for answer in answers:
            answer_clean = answer.lower().strip().replace('\n', ' ')
            predict_clean = predict.lower().strip().replace('\n', ' ')
            if answer_clean in predict_clean:
                return True
    
    return False

def analyze_ocrbench_results(model_name, input_file, output_dir):
    """
    分析OCRBench结果并生成详细报告
    """
    print(f"正在分析 {model_name} 的结果...")
    
    # 读取预测结果
    df = pd.read_excel(input_file)
    print(f"总样本数: {len(df)}")
    
    # 添加正确性判断
    print("正在计算每个样本的正确性...")
    df['is_correct'] = df.apply(evaluate_sample_correctness, axis=1)
    
    # 计算各类别的统计信息
    category_stats = {}
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        correct_count = category_df['is_correct'].sum()
        total_count = len(category_df)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        category_stats[category] = {
            'correct_count': int(correct_count),
            'total_count': int(total_count),
            'accuracy': float(accuracy)
        }
    
    # 计算总体统计
    total_correct = df['is_correct'].sum()
    total_samples = len(df)
    overall_accuracy = total_correct / total_samples
    
    # 创建详细结果DataFrame（不包含统计列）
    detailed_df = df.copy()
    
    # 保存详细结果
    os.makedirs(output_dir, exist_ok=True)
    detailed_output = os.path.join(output_dir, f'{model_name}_OCRBench_detailed.xlsx')
    detailed_df.to_excel(detailed_output, index=False)
    print(f"详细结果已保存到: {detailed_output}")
    
    # 保存错误样本
    wrong_samples = detailed_df[~detailed_df['is_correct']]
    wrong_output = os.path.join(output_dir, f'{model_name}_OCRBench_wrong_samples.xlsx')
    wrong_samples.to_excel(wrong_output, index=False)
    print(f"错误样本已保存到: {wrong_output}")
    print(f"错误样本数量: {len(wrong_samples)}")
    
    # 保存统计结果
    stats_output = os.path.join(output_dir, f'{model_name}_OCRBench_stats.json')
    stats = {
        'model_name': model_name,
        'overall_accuracy': float(overall_accuracy),
        'total_correct': int(total_correct),
        'total_samples': int(total_samples),
        'category_stats': category_stats
    }
    
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存到: {stats_output}")
    
    # 打印统计信息
    print(f"\n=== {model_name} 评估结果 ===")
    print(f"总体准确率: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    print("\n各类别准确率:")
    for category, cat_stats in category_stats.items():
        print(f"  {category}: {cat_stats['accuracy']:.4f} ({cat_stats['correct_count']}/{cat_stats['total_count']})")
    
    return detailed_df, stats

def main():
    """
    主函数：分析两个模型的结果
    """
    # 模型结果文件路径
    models = [
        {
            'name': 'InternVL2_5-8B',
            'file': 'outputs/InternVL2_5-8B/InternVL2_5-8B_OCRBench.xlsx'
        },
        {
            'name': 'GLM4_1VThinking-9b',
            'file': 'outputs/GLM4_1VThinking-9b/GLM4_1VThinking-9b_OCRBench.xlsx'
        },
        {
            'name': 'InternVL3_5-1B',
            'file': 'outputs/InternVL3_5-1B/InternVL3_5-1B_OCRBench.xlsx'
        },       
        {
            'name': 'InternVL3_5-2B',
            'file': 'outputs/InternVL3_5-2B/InternVL3_5-2B_OCRBench.xlsx'
        },     
        {
            'name': 'InternVL3_5-4B',
            'file': 'outputs/InternVL3_5-4B/InternVL3_5-4B_OCRBench.xlsx'
        },     
        {
            'name': 'InternVL3_5-8B',
            'file': 'outputs/InternVL3_5-8B/InternVL3_5-8B_OCRBench.xlsx'
        },     
        {
            'name': 'InternVL3_5-14B',
            'file': 'outputs/InternVL3_5-14B/InternVL3_5-14B_OCRBench.xlsx'
        },     
        {
            'name': 'InternVL3_5-38B',
            'file': 'outputs/InternVL3_5-38B/InternVL3_5-38B_OCRBench.xlsx'
        },     
        {
            'name': 'Ovis2-1B',
            'file': 'outputs/Ovis2-1B/Ovis2-1B_OCRBench.xlsx'
        },    
        {
            'name': 'Ovis2-2B',
            'file': 'outputs/Ovis2-2B/Ovis2-2B_OCRBench.xlsx'
        },    
        {
            'name': 'Ovis2-4B',
            'file': 'outputs/Ovis2-4B/Ovis2-4B_OCRBench.xlsx'
        },    
        {
            'name': 'Ovis2-8B',
            'file': 'outputs/Ovis2-8B/Ovis2-8B_OCRBench.xlsx'
        },    
        {
            'name': 'GPT5',
            'file': 'outputs/GPT5_OCRBench/GPT5_OCRBench_OCRBench.xlsx'
        },    
    ]
    
    output_dir = 'outputs/analysis'
    
    all_results = {}
    all_detailed_dfs = {}
    
    for model in models:
        if os.path.exists(model['file']):
            print(f"\n{'='*50}")
            print(f"分析模型: {model['name']}")
            print(f"{'='*50}")
            
            detailed_df, stats = analyze_ocrbench_results(
                model['name'], 
                model['file'], 
                output_dir
            )
            all_results[model['name']] = stats
            all_detailed_dfs[model['name']] = detailed_df
        else:
            print(f"文件不存在: {model['file']}")
    
    # 生成对比报告
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("模型对比报告")
        print(f"{'='*50}")
        
        comparison_data = []
        for model_name, stats in all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Overall Accuracy': f"{stats['overall_accuracy']:.4f}",
                'Correct/Total': f"{stats['total_correct']}/{stats['total_samples']}"
            })
            
            # 添加各类别准确率
            for category, cat_stats in stats['category_stats'].items():
                comparison_data[-1][f'{category}'] = f"{cat_stats['accuracy']:.4f}"
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_output = os.path.join(output_dir, 'model_comparison.xlsx')
        comparison_df.to_excel(comparison_output, index=False)
        print(f"对比报告已保存到: {comparison_output}")
        
        print("\n模型对比:")
        print(comparison_df.to_string(index=False))
    
    # 创建包含所有模型预测和正确性的综合CSV
    if len(all_detailed_dfs) > 0:
        print(f"\n{'='*50}")
        print("创建综合对比CSV")
        print(f"{'='*50}")
        
        # 获取第一个模型的基础列（index, question, answer, category）
        first_model = list(all_detailed_dfs.keys())[0]
        combined_df = all_detailed_dfs[first_model][['index', 'question', 'answer', 'category']].copy()
        
        # 先添加所有模型的prediction列
        for model_name, detailed_df in all_detailed_dfs.items():
            combined_df[f'{model_name}_prediction'] = detailed_df['prediction']
        
        # 再添加所有模型的is_correct列
        for model_name, detailed_df in all_detailed_dfs.items():
            combined_df[f'{model_name}_is_correct'] = detailed_df['is_correct']
        
        # 保存综合对比CSV
        combined_output = os.path.join(output_dir, 'all_models_comparison.csv')
        combined_df.to_csv(combined_output, index=False, encoding='utf-8-sig')
        print(f"综合对比CSV已保存到: {combined_output}")
        print(f"包含 {len(combined_df)} 个样本和 {len(all_detailed_dfs)} 个模型的结果")

if __name__ == "__main__":
    main()
