"""
测试结果保存和加载模块
支持保存反向求解和正向求解的所有测试结果
"""

import json
import gzip
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def serialize_distribution(dist):
    """序列化分布对象 (ForwardDiscreteDistribution 或 AlphaDiscreteDistribution)"""
    if dist is None:
        return None
    
    # 如果已经是字典
    if isinstance(dist, dict):
        if 'values' in dist and 'L1' in dist:
            return dist
        return None
    
    # 处理分布对象
    try: 
        # 检查是否有必需的属性
        if not (hasattr(dist, 'values') and hasattr(dist, 'L1')):
            return None
        
        values = dist.values
        L1 = dist.L1
        
        # 转换 values 为列表
        if isinstance(values, list):
            values_list = values
        elif isinstance(values, np.ndarray):
            values_list = values.tolist()
        elif hasattr(values, '__iter__'):
            values_list = list(values)
        else:
            print(f"⚠ 无法转换 values 类型: {type(values)}")
            return None
        
        return {
            'values': values_list,
            'L1':  int(L1)
        }
    
    except Exception as e:
        print(f"⚠ 序列化分布失败: {type(dist).__name__}, 错误: {e}")
        return None


def serialize_object_to_dict(obj, depth=0, max_depth=10):
    """将对象转换为字典"""
    if depth > max_depth:
        return str(obj)
    
    if obj is None:
        return None
    
    # 基本类型直接返回
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 字典类型
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # 确保key是字符串
            if not isinstance(key, str):
                key = str(key)
            
            # 跳过私有属性和缓存
            if key.startswith('_') or key in ['label', 'quantile_cache', 'mean_cache', 'variance_cache']:
                continue
            
            if key == 'distribution':
                result[key] = serialize_distribution(value)
            else:
                result[key] = serialize_object_to_dict(value, depth + 1, max_depth)
        return result
    
    # 列表或元组
    if isinstance(obj, (list, tuple)):
        return [serialize_object_to_dict(item, depth + 1, max_depth) for item in obj]
    
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # 对象类型（有 __dict__）
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            # 确保key是字符串
            if not isinstance(key, str):
                key = str(key)
            
            # 跳过私有属性和缓存
            if key.startswith('_') or key in ['label', 'quantile_cache', 'mean_cache', 'variance_cache']: 
                continue
            
            if key == 'distribution':  
                result[key] = serialize_distribution(value)
            else:
                result[key] = serialize_object_to_dict(value, depth + 1, max_depth)
        return result
    
    # 其他情况转为字符串
    try:
        return str(obj)
    except:
        return None


def serialize_result(result:  Dict[str, Any], solver_type: str) -> Dict[str, Any]:
    """
    序列化测试结果
    
    Args:
        result: 测试结果字典
        solver_type: 'reverse' 或 'forward'
    
    Returns:
        可序列化的结果字典
    """
    return serialize_object_to_dict(result)


def save_results(results:  Dict[str, Any],
                solver_type: str,
                output_dir: str = 'results',
                compress:  bool = False):
    """
    保存测试结果到文件
    
    Args: 
        results: 测试结果字典
        solver_type: 'reverse' 或 'forward'
        output_dir: 输出目录
        compress: 是否压缩
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{solver_type}_results_{timestamp}"
    
    # 序列化结果
    print(f"\n序列化 {solver_type} 测试结果...")
    try:
        serialized = serialize_result(results, solver_type)
    except Exception as e:
        print(f"❌ 序列化失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 添加元数据
    serialized['_metadata'] = {
        'solver_type': solver_type,
        'timestamp': timestamp,
        'datetime':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0'
    }
    
    try:
        if compress:
            # 保存为压缩JSON
            filename = os.path.join(output_dir, f"{base_filename}.json.gz")
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"✓ 压缩结果已保存: {filename}")
        else:
            # 保存为普通JSON
            filename = os.path.join(output_dir, f"{base_filename}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"✓ 结果已保存: {filename}")
        
        # 同时保存最新版本
        latest_filename = os.path.join(output_dir, f"{solver_type}_results_latest.json")
        with open(latest_filename, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"✓ 最新版本已保存: {latest_filename}")
        
        # 统计信息
        file_size = os.path.getsize(filename) / 1024  # KB
        print(f"  文件大小: {file_size:.2f} KB")
        
        return filename
    
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_results(filename: str) -> Dict[str, Any]:
    """
    从文件加载测试结果
    
    Args:
        filename: 文件路径
    
    Returns:
        结果字典
    """
    print(f"\n加载测试结果:  {filename}")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    try:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                results = json.load(f)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        # 显示元数据
        if '_metadata' in results:
            meta = results['_metadata']
            print(f"  求解器类型: {meta.get('solver_type', 'unknown')}")
            print(f"  生成时间: {meta.get('datetime', 'unknown')}")
            print(f"  版本: {meta.get('version', 'unknown')}")
        
        print(f"✓ 结果加载成功")
        
        return results
    
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise


def list_saved_results(output_dir: str = 'results') -> Dict[str, List[str]]:
    """
    列出所有保存的结果文件
    
    Args:
        output_dir: 结果目录
    
    Returns: 
        {'reverse': [...], 'forward': [...]}
    """
    if not os.path.exists(output_dir):
        return {'reverse': [], 'forward': []}
    
    files = os.listdir(output_dir)
    
    reverse_files = sorted([
        f for f in files 
        if f.startswith('reverse_results_') and not f.endswith('_latest.json')
    ])
    
    forward_files = sorted([
        f for f in files 
        if f.startswith('forward_results_') and not f.endswith('_latest.json')
    ])
    
    return {
        'reverse': reverse_files,
        'forward': forward_files
    }


def get_latest_results(solver_type: str, output_dir: str = 'results') -> str:
    """
    获取最新的结果文件路径
    
    Args:
        solver_type: 'reverse' 或 'forward'
        output_dir: 结果目录
    
    Returns:
        文件路径
    """
    latest_file = os.path.join(output_dir, f"{solver_type}_results_latest.json")
    if os.path.exists(latest_file):
        return latest_file
    
    # 如果没有 latest 文件，找最新的时间戳文件
    files = list_saved_results(output_dir)[solver_type]
    if files:
        return os.path.join(output_dir, files[-1])
    
    raise FileNotFoundError(f"未找到 {solver_type} 的结果文件")


def print_results_summary(output_dir: str = 'results'):
    """打印保存的结果摘要"""
    print(f"\n{'='*70}")
    print(f"保存的结果文件摘要")
    print(f"{'='*70}")
    print(f"目录: {output_dir}\n")
    
    if not os.path.exists(output_dir):
        print(f"⚠ 目录不存在\n")
        return
    
    results = list_saved_results(output_dir)
    
    print(f"反向求解结果: {len(results['reverse'])} 个文件")
    for f in results['reverse'][-5:]:  # 最新5个
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {f} ({size:.1f} KB)")
    if len(results['reverse']) > 5:
        print(f"  ...(还有 {len(results['reverse'])-5} 个)")
    
    print(f"\n正向求解结果: {len(results['forward'])} 个文件")
    for f in results['forward'][-5:]: 
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {f} ({size:.1f} KB)")
    if len(results['forward']) > 5:
        print(f"  ...(还有 {len(results['forward'])-5} 个)")
    
    print(f"\n{'='*70}\n")


def clean_old_results(output_dir: str = 'results', keep_last: int = 5):
    """
    清理旧的结果文件，只保留最新的几个
    
    Args: 
        output_dir: 结果目录
        keep_last:  保留最新的文件数量
    """
    results = list_saved_results(output_dir)
    
    deleted_count = 0
    
    for solver_type in ['reverse', 'forward']:
        files = results[solver_type]
        if len(files) > keep_last:
            # 删除旧文件
            for f in files[:-keep_last]:
                filepath = os.path.join(output_dir, f)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"✓ 删除:  {f}")
                except Exception as e: 
                    print(f"⚠ 删除失败 {f}: {e}")
    
    print(f"\n✓ 清理完成，删除了 {deleted_count} 个文件")


if __name__ == "__main__": 
    """使用示例"""
    print("结果管理器")
    print("=" * 70)
    
    # 列出现有结果
    print_results_summary()
    
    print("\n使用方式：")
    print("\n1.保存结果：")
    print("   from result_manager import save_results")
    print("   save_results(results, solver_type='reverse', output_dir='results')")
    
    print("\n2.加载结果：")
    print("   from result_manager import load_results, get_latest_results")
    print("   filename = get_latest_results('reverse')")
    print("   results = load_results(filename)")
    
    print("\n3.列出所有结果：")
    print("   from result_manager import list_saved_results")
    print("   files = list_saved_results()")
    
    print("\n4.清理旧结果：")
    print("   from result_manager import clean_old_results")
    print("   clean_old_results(keep_last=5)")
    
    print("\n" + "=" * 70)