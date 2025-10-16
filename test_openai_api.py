#!/usr/bin/env python3
"""
测试 OpenAI API 是否可用
运行方法：python test_openai_api.py
"""

import os
import sys

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 OpenAI API 连接")
print("=" * 60)

# 检查环境变量
api_key = os.environ.get('OPENAI_API_KEY', None)
if api_key:
    print(f"✓ 找到 OPENAI_API_KEY: {api_key[:10]}...{api_key[-4:]}")
else:
    print("✗ 未设置 OPENAI_API_KEY 环境变量")
    print("\n请先设置 API key:")
    print('  export OPENAI_API_KEY="sk-your-api-key-here"')
    sys.exit(1)

print("\n正在测试 API 连接...")
print("-" * 60)

try:
    from vlmeval.api import OpenAIWrapper
    
    # 测试 gpt-4o-mini（MathVista 使用的模型）
    print("测试模型: gpt-4o-mini")
    model = OpenAIWrapper('gpt-4o-mini', verbose=True)
    
    # 检查是否可用
    if model.working():
        print("\n✓ API 连接成功！模型可用。")
    else:
        print("\n✗ API 连接失败！模型不可用。")
        sys.exit(1)
    
    # 尝试一个简单的调用
    print("\n正在发送测试消息...")
    msgs = [dict(type='text', value='Hello! Please reply with "OK".')]
    code, answer, resp = model.generate_inner(msgs)
    
    print(f"\n返回代码: {code}")
    print(f"返回答案: {answer}")
    
    if code == 0:
        print("\n" + "=" * 60)
        print("✓✓✓ OpenAI API 完全正常，可以开始评估！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗✗✗ API 调用返回错误")
        print("=" * 60)
        print(f"错误信息: {resp}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ 发生错误: {type(e).__name__}")
    print(f"错误详情: {str(e)}")
    print("\n可能的原因:")
    print("1. API key 无效或已过期")
    print("2. 网络无法访问 OpenAI API（可能需要代理）")
    print("3. API 配额用尽")
    print("\n如果在中国大陆，可能需要设置代理:")
    print('  export HTTP_PROXY="http://your-proxy:port"')
    print('  export HTTPS_PROXY="http://your-proxy:port"')
    sys.exit(1)


