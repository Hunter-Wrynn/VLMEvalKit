"""
自定义GPT4V类，添加时间记录功能
"""

import time
import json
import os
from datetime import datetime
from vlmeval.api.gpt import GPT4V

class GPT4VWithTiming(GPT4V):
    """带时间记录的GPT4V类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_log = []
        self.timing_file = os.path.join(os.getcwd(), "gpt5_timing_detailed.json")
    
    def generate_inner(self, inputs, **kwargs):
        """重写generate_inner方法，添加时间记录"""
        start_time = time.time()
        
        # 调用原始的generate_inner方法
        result = super().generate_inner(inputs, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 记录时间信息
        timing_record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": result[0] == 0,  # 检查返回码
            "input_length": len(str(inputs)),
            "output_length": len(result[1]) if len(result) > 1 else 0
        }
        
        self.timing_log.append(timing_record)
        self.save_timing_log()
        
        # 打印时间信息（如果verbose=True）
        if self.verbose:
            print(f"⏱️  推理时间: {duration:.2f}秒")
        
        return result
    
    def save_timing_log(self):
        """保存时间记录到文件"""
        try:
            with open(self.timing_file, 'w', encoding='utf-8') as f:
                json.dump(self.timing_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存时间记录失败: {e}")
    
    def get_timing_stats(self):
        """获取时间统计信息"""
        if not self.timing_log:
            return None
        
        durations = [record["duration"] for record in self.timing_log]
        successful = [record for record in self.timing_log if record["success"]]
        
        return {
            "total_calls": len(self.timing_log),
            "successful_calls": len(successful),
            "success_rate": len(successful) / len(self.timing_log),
            "total_time": sum(durations),
            "average_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "median_time": sorted(durations)[len(durations) // 2]
        }
    
    def print_timing_summary(self):
        """打印时间统计摘要"""
        stats = self.get_timing_stats()
        if not stats:
            print("没有时间记录数据")
            return
        
        print("\n" + "="*50)
        print(f"GPT-5 ({self.model}) 推理时间统计")
        print("="*50)
        print(f"总调用次数: {stats['total_calls']}")
        print(f"成功调用次数: {stats['successful_calls']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"总推理时间: {stats['total_time']:.2f} 秒")
        print(f"平均推理时间: {stats['average_time']:.2f} 秒")
        print(f"最短推理时间: {stats['min_time']:.2f} 秒")
        print(f"最长推理时间: {stats['max_time']:.2f} 秒")
        print(f"中位数推理时间: {stats['median_time']:.2f} 秒")
        print("="*50)




