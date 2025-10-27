#!/usr/bin/env python3
"""
测试图像token计算 - 验证官方规则实现
"""
import math
from PIL import Image

def token_calculate_official(image_path):
    """
    官方提供的token计算方法
    """
    # 打开指定的图片文件
    image = Image.open(image_path)
    
    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    
    # 对于Qwen2.5-VL: 将宽高都调整为28的整数倍
    h_bar = round(height / 28) * 28 
    w_bar = round(width / 28) * 28
    
    # 图像的Token下限：4个Token
    min_pixels = 28 * 28 * 4
    # 图像的Token上限：1280个Token
    max_pixels = 1280 * 28 * 28
        
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 重新计算调整后的宽高，确保为28的整数倍
        h_bar = math.floor(height / beta / 28) * 28
        w_bar = math.floor(width / beta / 28) * 28
    elif h_bar * w_bar < min_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 重新计算调整后的高度，确保为28的整数倍
        h_bar = math.ceil(height * beta / 28) * 28
        w_bar = math.ceil(width * beta / 28) * 28
    
    # 计算图像的Token数：Token数 = 总像素除以28 * 28
    token = int((h_bar * w_bar) / (28 * 28))
    
    # 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各计1个Token）
    total_tokens = token + 2
    
    return {
        'original_size': (width, height),
        'target_size': (w_bar, h_bar),
        'tokens': token,
        'total_tokens': total_tokens
    }

def test_with_sample_images():
    """
    使用样本图像测试
    """
    print("="*60)
    print("测试图像Token计算")
    print("="*60)
    
    # 测试一些常见的图像尺寸
    test_cases = [
        (1024, 768),   # 4:3 标准分辨率
        (1920, 1080),  # 16:9 高清
        (512, 512),    # 正方形
        (256, 256),    # 小图像
        (2048, 1536),  # 大图像
    ]
    
    for width, height in test_cases:
        print(f"\n测试图像尺寸: {width}x{height}")
        
        # 模拟官方计算
        h_bar = round(height / 28) * 28 
        w_bar = round(width / 28) * 28
        
        min_pixels = 28 * 28 * 4
        max_pixels = 1280 * 28 * 28
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / 28) * 28
            w_bar = math.floor(width / beta / 28) * 28
            print(f"  需要缩小: 缩放因子 {beta:.4f}")
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / 28) * 28
            w_bar = math.ceil(width * beta / 28) * 28
            print(f"  需要放大: 缩放因子 {beta:.4f}")
        else:
            print(f"  无需缩放")
        
        token = int((h_bar * w_bar) / (28 * 28))
        total_tokens = token + 2
        
        print(f"  原始尺寸: {width}x{height}")
        print(f"  目标尺寸: {w_bar}x{h_bar}")
        print(f"  图像tokens: {token}")
        print(f"  总tokens: {total_tokens} (包含特殊标记)")

if __name__ == '__main__':
    test_with_sample_images()

