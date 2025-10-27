"""
精确Token计数脚本 - 真实计算图像token数量
"""
import json
import os
import subprocess
from functools import partial
from PIL import Image
import numpy as np


def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 1))

from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.smp import *


class AccurateTokenCounter:
    """精确计算token数量的计数器"""
    
    def __init__(self, model_path, model_type='qwen2.5vl'):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.dump_image_func = None
        
        print(f"[AccurateTokenCounter] 初始化 {model_type} tokenizer...")
        print(f"[AccurateTokenCounter] 模型路径: {model_path}")
        
        if 'qwen2.5' in self.model_type or 'qwen2_5' in self.model_type:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
            self.tokenizer = self.processor.tokenizer
            self.use_processor = True
            print("[AccurateTokenCounter] 使用 Qwen2.5-VL processor")
        elif 'qwen2' in self.model_type:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            self.use_processor = True
            print("[AccurateTokenCounter] 使用 Qwen2-VL processor")
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.processor = None
            self.use_processor = False
            print("[AccurateTokenCounter] 使用基础 tokenizer")
        
        print("[AccurateTokenCounter] Tokenizer加载完成!")
    
    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func
    
    def dump_image(self, line, dataset):
        return self.dump_image_func(line)
    
    def use_custom_prompt(self, dataset):
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)
        
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:
            return True
        return False
    
    def build_prompt(self, line, dataset):
        from vlmeval.dataset import DATASET_TYPE
        
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')
    
    def _build_mmmu_prompt(self, line, dataset):
        import string
        import pandas as pd
        
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs
    
    def _build_mcq_prompt(self, line, dataset):
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'
        
        import string
        import pandas as pd
        import re
        
        def cn_string(s):
            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False
        
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs
    
    def _build_yorn_prompt(self, line, dataset):
        YORN_PROMPT = ' Please answer yes or no.'
        
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs
    
    def _build_vqa_prompt(self, line, dataset):
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'
        
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs
    
    def _prepare_content(self, inputs, dataset=None):
        def ensure_image_url(image):
            prefixes = ['http://', 'https://', 'file://', 'data:image;']
            if any(image.startswith(prefix) for prefix in prefixes):
                return image
            if os.path.exists(image):
                return 'file://' + image
            return image
        
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
    
    def calculate_image_tokens_accurate(self, image_path, min_pixels=1280, max_pixels=16384):
        """
        精确计算图像token数量
        基于Qwen VL的官方规则
        """
        try:
            import math
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            width, height = original_size
            print(f"[DEBUG] 原始图像尺寸: {original_size}")
            
            # 根据官方规则计算
            # 对于Qwen2.5-VL: 每28x28像素对应一个Token
            patch_size = 28
            
            # 将宽高都调整为28的整数倍
            h_bar = round(height / patch_size) * patch_size
            w_bar = round(width / patch_size) * patch_size
            
            # 图像的Token下限：4个Token (28*28*4 = 3136像素)
            min_pixels = patch_size * patch_size * 4
            # 图像的Token上限：1280个Token (28*28*1280 = 1003520像素)
            max_pixels = 1280 * patch_size * patch_size
            
            print(f"[DEBUG] 初始调整后尺寸: {w_bar}x{h_bar}")
            print(f"[DEBUG] 像素范围: [{min_pixels}, {max_pixels}]")
            
            # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
            if h_bar * w_bar > max_pixels:
                # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
                beta = math.sqrt((height * width) / max_pixels)
                # 重新计算调整后的宽高，确保为28的整数倍
                h_bar = math.floor(height / beta / patch_size) * patch_size
                w_bar = math.floor(width / beta / patch_size) * patch_size
                print(f"[DEBUG] 缩放因子: {beta:.4f} (缩小)")
            elif h_bar * w_bar < min_pixels:
                # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
                beta = math.sqrt(min_pixels / (height * width))
                # 重新计算调整后的高度，确保为28的整数倍
                h_bar = math.ceil(height * beta / patch_size) * patch_size
                w_bar = math.ceil(width * beta / patch_size) * patch_size
                print(f"[DEBUG] 缩放因子: {beta:.4f} (放大)")
            
            print(f"[DEBUG] 最终尺寸: {w_bar}x{h_bar}")
            
            # 计算patch数量
            patches_h = h_bar // patch_size
            patches_w = w_bar // patch_size
            total_patches = patches_h * patches_w
            
            print(f"[DEBUG] Patch数量: {patches_h}x{patches_w} = {total_patches}")
            
            # 每个patch对应一个token
            image_tokens = total_patches
            
            # 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各计1个Token）
            total_image_tokens = image_tokens + 2
            
            print(f"[DEBUG] 图像tokens: {image_tokens} + 2(特殊标记) = {total_image_tokens}")
            
            return {
                'original_size': original_size,
                'target_size': (w_bar, h_bar),
                'patches_h': patches_h,
                'patches_w': patches_w,
                'total_patches': total_patches,
                'image_tokens': image_tokens,
                'total_image_tokens': total_image_tokens,
                'patch_size': patch_size
            }
            
        except Exception as e:
            print(f"[ERROR] 图像处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'original_size': (0, 0),
                'target_size': (0, 0),
                'patches_h': 0,
                'patches_w': 0,
                'total_patches': 0,
                'image_tokens': 0,
                'total_image_tokens': 2,  # 至少包含特殊标记
                'error': str(e)
            }
    
    def _calculate_target_size(self, original_size, min_pixels, max_pixels):
        """
        计算目标尺寸，模拟Qwen VL的smart_resize逻辑
        """
        width, height = original_size
        original_pixels = width * height
        
        # 如果像素数在范围内，直接调整到28的倍数
        if min_pixels <= original_pixels <= max_pixels:
            # 调整到28的倍数
            new_width = ((width + 27) // 28) * 28
            new_height = ((height + 27) // 28) * 28
            return (new_width, new_height)
        
        # 如果像素数太小，放大
        if original_pixels < min_pixels:
            scale_factor = (min_pixels / original_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            # 如果像素数太大，缩小
            scale_factor = (max_pixels / original_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        
        # 调整到28的倍数
        new_width = ((new_width + 27) // 28) * 28
        new_height = ((new_height + 27) // 28) * 28
        
        return (new_width, new_height)
    
    def count_tokens_accurate(self, message, dataset=None):
        """
        精确计算token数量
        """
        try:
            # 准备内容
            messages = []
            messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
            
            if self.use_processor:
                # 使用processor处理
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # 计算文本token数量
                text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                text_token_count = len(text_tokens)
                
                # 精确计算图像token数量
                image_tokens_total = 0
                image_details = []
                
                for s in message:
                    if s['type'] == 'image':
                        image_path = s['value']
                        if image_path.startswith('file://'):
                            image_path = image_path[7:]  # 移除file://前缀
                        
                        # 精确计算这张图像的token数量
                        image_info = self.calculate_image_tokens_accurate(image_path)
                        image_tokens_total += image_info['total_image_tokens']  # 使用包含特殊标记的总数
                        image_details.append(image_info)
                        
                        print(f"[DEBUG] 图像 {image_path}: {image_info['original_size']} -> {image_info['target_size']} -> {image_info['total_patches']} patches -> {image_info['total_image_tokens']} tokens")
                
                total_tokens = text_token_count + image_tokens_total
                
                return {
                    'text_token_count': text_token_count,
                    'image_count': len([s for s in message if s['type'] == 'image']),
                    'image_tokens_total': image_tokens_total,
                    'total_tokens': total_tokens,
                    'image_details': image_details,
                    'text_tokens_list': text_tokens[:50],  # 只保存前50个token用于调试
                }
            else:
                # 基础tokenizer
                prompt = ''
                for s in message:
                    if s['type'] == 'image':
                        prompt += f'<img>{s["value"]}</img>'
                    elif s['type'] == 'text':
                        prompt += s['value']
                
                tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                return {
                    'text_token_count': len(tokens),
                    'image_count': len([s for s in message if s['type'] == 'image']),
                    'total_tokens': len(tokens),
                    'tokens_list': tokens[:50],
                }
        except Exception as e:
            print(f"[ERROR] Token计数失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'text_token_count': 0,
                'total_tokens': 0
            }


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def count_tokens_for_dataset_accurate(model, dataset, dataset_name, output_file):
    """为数据集的每条样本精确计算token数量"""
    logger = get_logger('AccurateTokenCounter')
    
    # 设置dump_image函数
    model.set_dump_image(dataset.dump_image)
    
    data = dataset.data
    lt = len(data)
    
    logger.info(f"开始精确计算 {dataset_name} 数据集的token数量...")
    logger.info(f"总样本数: {lt}")
    
    results = []
    
    for i in tqdm(range(lt), desc=f'精确计算Token数量'):
        item = data.iloc[i]
        idx = item['index']
        
        try:
            # 构建提示词
            if model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(item, dataset=dataset_name)
            else:
                struct = dataset.build_prompt(item)
            
            # 精确计算token数量
            token_stats = model.count_tokens_accurate(message=struct, dataset=dataset_name)
            
            # 保存结果
            result = {
                'index': idx,
                'dataset': dataset_name,
                **token_stats
            }
            
            # 添加问题文本（完整保存）
            if 'question' in item:
                result['question'] = str(item['question'])
            
            results.append(result)
            
            # 每100条打印一次进度
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i+1}/{lt} 条样本")
                logger.info(f"最新样本token统计: {token_stats}")
        
        except Exception as e:
            logger.error(f"样本 {idx} 处理失败: {e}")
            results.append({
                'index': idx,
                'dataset': dataset_name,
                'error': str(e),
                'text_token_count': 0,
                'total_tokens': 0
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    
    # 计算统计信息
    total_text_tokens = df['text_token_count'].sum()
    avg_text_tokens = df['text_token_count'].mean()
    max_text_tokens = df['text_token_count'].max()
    min_text_tokens = df['text_token_count'].min()
    
    total_image_tokens = df['image_tokens_total'].sum() if 'image_tokens_total' in df.columns else 0
    avg_image_tokens = df['image_tokens_total'].mean() if 'image_tokens_total' in df.columns else 0
    
    total_tokens = df['total_tokens'].sum()
    avg_tokens = df['total_tokens'].mean()
    
    logger.info("\n" + "="*60)
    logger.info("精确Token统计摘要:")
    logger.info(f"  总样本数: {len(df)}")
    logger.info(f"  总文本Token数: {total_text_tokens:,}")
    logger.info(f"  平均文本Token数: {avg_text_tokens:.2f}")
    logger.info(f"  最大文本Token数: {max_text_tokens}")
    logger.info(f"  最小文本Token数: {min_text_tokens}")
    logger.info(f"  总图像Token数: {total_image_tokens:,}")
    logger.info(f"  平均图像Token数: {avg_image_tokens:.2f}")
    logger.info(f"  总Token数: {total_tokens:,}")
    logger.info(f"  平均Token数: {avg_tokens:.2f}")
    logger.info("="*60 + "\n")
    
    # 保存为多种格式
    base_path = output_file.rsplit('.', 1)[0]
    
    # 保存为pkl
    dump(df, base_path + '.pkl')
    logger.info(f"精确Token统计已保存到: {base_path}.pkl")
    
    # 保存为CSV
    df.to_csv(base_path + '.csv', index=False)
    logger.info(f"精确Token统计已保存到: {base_path}.csv")
    
    # 保存为Excel
    try:
        df.to_excel(base_path + '.xlsx', index=False)
        logger.info(f"精确Token统计已保存到: {base_path}.xlsx")
    except:
        logger.warning("无法保存为Excel格式，请确保安装了openpyxl")
    
    # 保存统计摘要
    summary = {
        'dataset': dataset_name,
        'total_samples': len(df),
        'total_text_tokens': int(total_text_tokens),
        'avg_text_tokens': float(avg_text_tokens),
        'max_text_tokens': int(max_text_tokens),
        'min_text_tokens': int(min_text_tokens),
        'total_image_tokens': int(total_image_tokens),
        'avg_image_tokens': float(avg_image_tokens),
        'total_tokens': int(total_tokens),
        'avg_tokens': float(avg_tokens),
    }
    
    with open(base_path + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"统计摘要已保存到: {base_path}_summary.json")
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='精确计算数据集每条样本的token数量（不加载模型）')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径（仅加载tokenizer）')
    parser.add_argument('--model-type', type=str, default='qwen2.5vl', 
                        choices=['qwen2.5vl', 'qwen2vl', 'qwenvl'],
                        help='模型类型')
    parser.add_argument('--work-dir', type=str, default='./token_stats_accurate', help='输出目录')
    parser.add_argument('--dataset', type=str, default=None, help='指定要处理的数据集（可选）')
    
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('AccurateTokenCounter')
    args = parse_args()
    
    # 加载配置
    cfg = load(args.config)
    
    # 创建输出目录
    os.makedirs(args.work_dir, exist_ok=True)
    
    # 初始化精确token计数器（只加载tokenizer）
    logger.info(f"初始化精确Token计数器...")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"模型类型: {args.model_type}")
    
    token_counter = AccurateTokenCounter(args.model_path, args.model_type)
    
    # 获取要处理的数据集列表
    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(cfg['data'].keys())
    
    logger.info(f"要处理的数据集: {dataset_names}")
    
    # 处理每个数据集
    for dataset_name in dataset_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理数据集: {dataset_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            # 构建数据集
            dataset = build_dataset_from_config(cfg['data'], dataset_name)
            if dataset is None:
                logger.error(f'数据集 {dataset_name} 无效，跳过')
                continue
            
            # 输出文件路径
            output_file = osp.join(args.work_dir, f'token_stats_accurate_{dataset_name}.pkl')
            
            # 精确计算token数量
            count_tokens_for_dataset_accurate(token_counter, dataset, dataset_name, output_file)
            
        except Exception as e:
            logger.exception(f'数据集 {dataset_name} 处理失败: {e}')
            continue
    
    logger.info("\n" + "="*60)
    logger.info("所有数据集处理完成！")
    logger.info(f"结果保存在: {args.work_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    load_env()
    main()
