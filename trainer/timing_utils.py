# Copyright (c) 2024
# Time measurement utilities for CF-3DGS training
# Compatible with EndoGSLAM timing format

import time
import os
import json
from typing import Dict, List, Optional
from collections import defaultdict


class TimingStats:
    """
    时间统计类，用于记录和输出训练过程中的各项时间指标
    与 EndoGSLAM 的时间统计格式兼容
    """
    
    def __init__(self, output_dir: str = None):
        """
        初始化时间统计器
        
        Args:
            output_dir: 输出目录，用于保存时间统计结果
        """
        self.output_dir = output_dir
        
        # 时间统计变量 - 与 EndoGSLAM 对齐
        # Tracking Time（位姿估计时间）
        self.tracking_iter_time_sum = 0.0
        self.tracking_iter_time_count = 0
        self.tracking_frame_time_sum = 0.0
        self.tracking_frame_time_count = 0
        
        # Reconstruction/Mapping Time（重建/映射时间）
        self.mapping_iter_time_sum = 0.0
        self.mapping_iter_time_count = 0
        self.mapping_frame_time_sum = 0.0
        self.mapping_frame_time_count = 0
        
        # Rendering Time（渲染时间）
        self.rendering_time_sum = 0.0
        self.rendering_time_count = 0
        
        # 初始化时间
        self.init_time = 0.0
        
        # 其他时间统计
        self.total_time = 0.0
        self.frame_times: List[Dict] = []  # 每帧的详细时间
        
        # 计时器状态
        self._start_times: Dict[str, float] = {}
        self._frame_start_time: Optional[float] = None
        self._current_frame_stats: Dict[str, float] = {}
        
        # 全局开始时间
        self._global_start_time = time.perf_counter()
    
    def start(self, name: str):
        """开始计时"""
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """
        结束计时并返回经过的时间
        
        Args:
            name: 计时器名称
            
        Returns:
            经过的时间（秒）
        """
        if name not in self._start_times:
            print(f"Warning: Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.perf_counter() - self._start_times[name]
        del self._start_times[name]
        return elapsed
    
    def start_frame(self, frame_idx: int):
        """开始帧计时"""
        self._frame_start_time = time.perf_counter()
        self._current_frame_stats = {
            'frame_idx': frame_idx,
            'tracking_time': 0.0,
            'mapping_time': 0.0,
            'rendering_time': 0.0,
        }
    
    def end_frame(self) -> Dict:
        """
        结束帧计时
        
        Returns:
            当前帧的时间统计
        """
        if self._frame_start_time is None:
            return {}
        
        self._current_frame_stats['total_time'] = time.perf_counter() - self._frame_start_time
        self.frame_times.append(self._current_frame_stats.copy())
        
        result = self._current_frame_stats.copy()
        self._frame_start_time = None
        self._current_frame_stats = {}
        
        return result
    
    # ==================== Tracking Time ====================
    
    def start_tracking_iter(self):
        """开始追踪迭代计时"""
        self.start('tracking_iter')
    
    def stop_tracking_iter(self):
        """结束追踪迭代计时"""
        elapsed = self.stop('tracking_iter')
        self.tracking_iter_time_sum += elapsed
        self.tracking_iter_time_count += 1
        return elapsed
    
    def start_tracking_frame(self):
        """开始追踪帧计时"""
        self.start('tracking_frame')
    
    def stop_tracking_frame(self):
        """结束追踪帧计时"""
        elapsed = self.stop('tracking_frame')
        self.tracking_frame_time_sum += elapsed
        self.tracking_frame_time_count += 1
        if self._current_frame_stats:
            self._current_frame_stats['tracking_time'] = elapsed
        return elapsed
    
    # ==================== Mapping/Reconstruction Time ====================
    
    def start_mapping_iter(self):
        """开始映射迭代计时"""
        self.start('mapping_iter')
    
    def stop_mapping_iter(self):
        """结束映射迭代计时"""
        elapsed = self.stop('mapping_iter')
        self.mapping_iter_time_sum += elapsed
        self.mapping_iter_time_count += 1
        return elapsed
    
    def start_mapping_frame(self):
        """开始映射帧计时"""
        self.start('mapping_frame')
    
    def stop_mapping_frame(self):
        """结束映射帧计时"""
        elapsed = self.stop('mapping_frame')
        self.mapping_frame_time_sum += elapsed
        self.mapping_frame_time_count += 1
        if self._current_frame_stats:
            self._current_frame_stats['mapping_time'] = elapsed
        return elapsed
    
    # ==================== Rendering Time ====================
    
    def start_rendering(self):
        """开始渲染计时"""
        self.start('rendering')
    
    def stop_rendering(self):
        """结束渲染计时"""
        elapsed = self.stop('rendering')
        self.rendering_time_sum += elapsed
        self.rendering_time_count += 1
        if self._current_frame_stats:
            self._current_frame_stats['rendering_time'] += elapsed
        return elapsed
    
    # ==================== Init Time ====================
    
    def start_init(self):
        """开始初始化计时"""
        self.start('init')
    
    def stop_init(self):
        """结束初始化计时"""
        self.init_time = self.stop('init')
        return self.init_time
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict:
        """
        获取时间统计信息
        
        Returns:
            包含所有时间统计的字典
        """
        # 防止除零
        tracking_frame_count = max(1, self.tracking_frame_time_count)
        mapping_frame_count = max(1, self.mapping_frame_time_count)
        rendering_count = max(1, self.rendering_time_count)
        
        # 计算平均值
        tracking_frame_avg = self.tracking_frame_time_sum / tracking_frame_count
        mapping_frame_avg = self.mapping_frame_time_sum / mapping_frame_count
        rendering_avg = self.rendering_time_sum / rendering_count
        
        # 总时间
        self.total_time = time.perf_counter() - self._global_start_time
        
        # 计算百分比（注意：渲染时间已包含在 tracking 和 mapping 中）
        total_frame_time = tracking_frame_avg + mapping_frame_avg
        if total_frame_time > 0:
            tracking_pct = 100 * tracking_frame_avg / total_frame_time
            mapping_pct = 100 * mapping_frame_avg / total_frame_time
        else:
            tracking_pct = mapping_pct = 0.0
        
        return {
            # Tracking
            'tracking_frame_time_avg_s': tracking_frame_avg,
            'tracking_frame_count': self.tracking_frame_time_count,
            
            # Mapping
            'mapping_frame_time_avg_s': mapping_frame_avg,
            'mapping_frame_count': self.mapping_frame_time_count,
            
            # Rendering (仅作为参考，已包含在 tracking/mapping 中)
            'rendering_time_avg_ms': rendering_avg * 1000,
            'rendering_count': self.rendering_time_count,
            'rendering_time_total_s': self.rendering_time_sum,
            
            # Init
            'init_time_s': self.init_time,
            
            # Total
            'total_time_s': self.total_time,
            'total_frame_time_s': total_frame_time,
            
            # Percentages
            'tracking_pct': tracking_pct,
            'mapping_pct': mapping_pct,
        }
    
    def print_statistics(self):
        """打印时间统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("Time Statistics (Compatible with EndoGSLAM)")
        print("="*70)
        
        print(f"\nAverage Tracking/Frame Time: {stats['tracking_frame_time_avg_s']:.4f} s")
        print(f"Average Mapping/Frame Time: {stats['mapping_frame_time_avg_s']:.4f} s")
        print(f"\nTotal Frames Processed:")
        print(f"  Tracking frames: {stats['tracking_frame_count']}")
        print(f"  Mapping frames: {stats['mapping_frame_count']}")
        
        print(f"\nRendering Statistics (reference only, included in above times):")
        print(f"  Average per call: {stats['rendering_time_avg_ms']:.2f} ms")
        print(f"  Total calls: {stats['rendering_count']}")
        print(f"  Total time: {stats['rendering_time_total_s']:.2f} s")
        
        print(f"\nInit Time: {stats['init_time_s']:.2f} s")
        print(f"Total Training Time: {stats['total_time_s']:.2f} s")
        
        print(f"\nFrame Time Breakdown:")
        print(f"  Tracking:   {stats['tracking_pct']:.1f}%")
        print(f"  Mapping:    {stats['mapping_pct']:.1f}%")
        
        print("="*70 + "\n")
    
    def save_statistics(self, filepath: str = None):
        """
        保存时间统计信息到文件
        
        Args:
            filepath: 保存路径，默认为 output_dir/runtimes.txt
        """
        if filepath is None:
            if self.output_dir is None:
                print("Warning: No output directory specified, skipping save")
                return
            filepath = os.path.join(self.output_dir, "runtimes.txt")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        stats = self.get_statistics()
        
        # 保存与 EndoGSLAM 兼容的格式
        with open(filepath, 'w') as f:
            f.write("# CF-3DGS Time Statistics (EndoGSLAM Compatible Format)\n")
            f.write("#" + "="*68 + "\n\n")
            
            f.write(f"Average Tracking/Frame Time: {stats['tracking_frame_time_avg_s']:.6f} s\n")
            f.write(f"Average Mapping/Frame Time: {stats['mapping_frame_time_avg_s']:.6f} s\n")
            f.write(f"Frame Time: {stats['total_frame_time_s']:.6f} s\n")
            
            f.write(f"\n# Rendering Statistics (reference, included in above)\n")
            f.write(f"Average Rendering Time per call: {stats['rendering_time_avg_ms']:.4f} ms\n")
            f.write(f"Total Rendering Time: {stats['rendering_time_total_s']:.2f} s\n")
            f.write(f"Rendering Calls: {stats['rendering_count']}\n")
            
            f.write(f"\n# Additional Statistics\n")
            f.write(f"Init Time: {stats['init_time_s']:.2f} s\n")
            f.write(f"Total Training Time: {stats['total_time_s']:.2f} s\n")
            
            f.write(f"\n# Frame Counts\n")
            f.write(f"Tracking Frames: {stats['tracking_frame_count']}\n")
            f.write(f"Mapping Frames: {stats['mapping_frame_count']}\n")
            
            f.write(f"\n# Time Breakdown\n")
            f.write(f"Tracking: {stats['tracking_pct']:.1f}%\n")
            f.write(f"Mapping: {stats['mapping_pct']:.1f}%\n")
        
        print(f"Time statistics saved to: {filepath}")
        
        # 同时保存 JSON 格式，便于后续分析
        json_filepath = filepath.replace('.txt', '.json')
        with open(json_filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Time statistics (JSON) saved to: {json_filepath}")
    
    def save_frame_details(self, filepath: str = None):
        """
        保存每帧的详细时间信息
        
        Args:
            filepath: 保存路径
        """
        if filepath is None:
            if self.output_dir is None:
                print("Warning: No output directory specified, skipping save")
                return
            filepath = os.path.join(self.output_dir, "frame_times.json")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.frame_times, f, indent=2)
        
        print(f"Frame details saved to: {filepath}")


# 全局计时器实例（可选使用）
_global_timer: Optional[TimingStats] = None


def get_global_timer() -> TimingStats:
    """获取全局计时器实例"""
    global _global_timer
    if _global_timer is None:
        _global_timer = TimingStats()
    return _global_timer


def reset_global_timer(output_dir: str = None) -> TimingStats:
    """重置全局计时器"""
    global _global_timer
    _global_timer = TimingStats(output_dir)
    return _global_timer


# 上下文管理器，方便使用
class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, timer: TimingStats, name: str, category: str = 'iter'):
        """
        Args:
            timer: TimingStats 实例
            name: 计时器名称 ('tracking', 'mapping', 'rendering')
            category: 类别 ('iter' 或 'frame')
        """
        self.timer = timer
        self.name = name
        self.category = category
    
    def __enter__(self):
        if self.name == 'tracking':
            if self.category == 'iter':
                self.timer.start_tracking_iter()
            else:
                self.timer.start_tracking_frame()
        elif self.name == 'mapping':
            if self.category == 'iter':
                self.timer.start_mapping_iter()
            else:
                self.timer.start_mapping_frame()
        elif self.name == 'rendering':
            self.timer.start_rendering()
        elif self.name == 'init':
            self.timer.start_init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name == 'tracking':
            if self.category == 'iter':
                self.timer.stop_tracking_iter()
            else:
                self.timer.stop_tracking_frame()
        elif self.name == 'mapping':
            if self.category == 'iter':
                self.timer.stop_mapping_iter()
            else:
                self.timer.stop_mapping_frame()
        elif self.name == 'rendering':
            self.timer.stop_rendering()
        elif self.name == 'init':
            self.timer.stop_init()
        return False


if __name__ == "__main__":
    # 测试代码
    import time as t
    
    timer = TimingStats(output_dir="./test_output")
    
    # 模拟初始化
    with Timer(timer, 'init'):
        t.sleep(0.1)
    
    # 模拟帧处理
    for frame in range(3):
        timer.start_frame(frame)
        
        # 模拟 Tracking
        timer.start_tracking_frame()
        for i in range(5):
            timer.start_tracking_iter()
            t.sleep(0.01)
            timer.stop_tracking_iter()
        timer.stop_tracking_frame()
        
        # 模拟 Mapping
        timer.start_mapping_frame()
        for i in range(10):
            timer.start_mapping_iter()
            t.sleep(0.02)
            timer.stop_mapping_iter()
        timer.stop_mapping_frame()
        
        # 模拟 Rendering
        timer.start_rendering()
        t.sleep(0.005)
        timer.stop_rendering()
        
        timer.end_frame()
    
    # 打印和保存统计
    timer.print_statistics()
    timer.save_statistics()
    timer.save_frame_details()
