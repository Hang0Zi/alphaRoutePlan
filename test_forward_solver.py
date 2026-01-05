"""
正向求解器测试文件（使用config配置，支持K-Paths多路径）
"""

import sys
import os
from datetime import datetime
import numpy as np
import time as time_module
from typing import Dict
from forward_solver import ForwardLabelSettingSolver
from result_manager import save_results,get_latest_results, list_saved_results, print_results_summary
import config as config
# 添加路径
sys.path.insert(0, os.path.dirname(__file__))
import time
# 使用run_solver.py中的全局数据加载函数
from run_solver import load_data_once, get_data, select_od_pair, time_to_string
from visualization_generator import generate_html_from_files


def test_forward_basic():
    """测试正向求解基本功能（K=1，单路径）"""
    print(f"\n{'='*70}")
    print(f"测试:  正向求解基本功能")
    print(f"{'='*70}\n")
    
    # 获取数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 使用config中的参数初始化求解器
    mode = config.get_mode_config('standard')
    
    solver = ForwardLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=config.FORWARD_L1,
        L2=config.FORWARD_L2,
        K=config.FORWARD_K,
        verbose=config.FORWARD_VERBOSE
    )
    
    # 选择OD对
    origin, destination = select_od_pair(node_to_index)
    print(f"  测试OD对: {origin} → {destination}")
    
    # 使用config中的默认出发时间
    departure_time = (config.DEFAULT_DEPARTURE_HOUR * 60 + 
                     config.DEFAULT_DEPARTURE_MINUTE) * 10
    alpha = config.FORWARD_ALPHA_DEFAULT
    
    print(f"  出发时间:  {time_to_string(departure_time)} "
          f"(配置: {config.DEFAULT_DEPARTURE_HOUR}:{config.DEFAULT_DEPARTURE_MINUTE: 02d})")
    print(f"  可靠性要求: α={alpha} (配置:  FORWARD_ALPHA_DEFAULT)")
    print(f"  参数: L1={config.FORWARD_L1}, L2={config.FORWARD_L2}, K={config.FORWARD_K}\n")
    
    # ✅ 调用 solve_k_paths，K=1（单路径）
    result = solver.solve_k_paths(
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        alpha=alpha,
        K=1,  # 单路径
        max_labels=mode['max_labels']
    )
    
    # 验证结果
    print(f"\n{'─'*70}")
    print(f"结果验证")
    print(f"{'─'*70}")
    
    assert result['success'], "❌ 求解失败"
    print(f"  ✓ 求解成功")
    
    assert result['path'] is not None, "❌ 路径为空"
    print(f"  ✓ 路径非空 (长度: {len(result['path'])})")
    
    assert result['path'][0] == origin, "❌ 起点不匹配"
    print(f"  ✓ 起点正确:  {origin}")
    
    assert result['path'][-1] == destination, "❌ 终点不匹配"
    print(f"  ✓ 终点正确: {destination}")
    
    assert result['earliest_arrival_time'] > departure_time, "❌ 到达早于出发"
    print(f"  ✓ 时间逻辑正确")
    
    print(f"  ✓ 最早到达时间: {time_to_string(result['earliest_arrival_time'])}")
    print(f"  ✓ 旅行时间: {result['travel_time']/10:.1f}分钟")
    
    print(f"\n  🎉 测试通过！")
    print(f"{'='*70}\n")
    
    return result


def test_forward_alpha_sensitivity():
    """
    测试α敏感性分析（K-Paths版本，每个α值找K=5条候选路径）
    
    Returns:
        包含所有α值的多路径结果
    """
    print(f"\n{'='*70}")
    print(f"测试:  α敏感性分析（正向，K-Paths版本）")
    print(f"{'='*70}\n")
    
    # 获取数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 使用config中的fast模式
    mode = config.get_mode_config('fast')
    
    solver = ForwardLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        K=50,  # 用于内部卷积
        verbose=config.FORWARD_VERBOSE
    )
    
    # 选择OD对（固定使用同一对以便对比）
    origin, destination = select_od_pair(node_to_index)
    
    # 使用config中的默认出发时间
    departure_time = (config.DEFAULT_DEPARTURE_HOUR * 60 + 
                     config.DEFAULT_DEPARTURE_MINUTE) * 10
    
    print(f"  测试OD对: {origin} → {destination}")
    print(f"  出发时间:  {time_to_string(departure_time)}")
    print(f"  配置模式: {mode['description']}")
    print(f"  每个α值寻找K=5条候选路径")
    print(f"  参数: L1={mode['L1']}, L2={mode['L2']}\n")
    
    # 使用config中的α敏感性分析测试值
    alphas = config.ALPHA_SENSITIVITY_VALUES
    print(f"  测试α值数量: {len(alphas)} (来自 ALPHA_SENSITIVITY_VALUES)")
    
    # ✅ 存储结果的数据结构
    alpha_results = []  # 存储每个α值的结果
    
    print(f"\n  开始测试:")
    for i, alpha in enumerate(alphas, 1):
        print(f"    [{i}/{len(alphas)}] α={alpha:.2f}...", end='', flush=True)
        
        # ✅ 调用 solve_k_paths，每个α值找K=5条候选路径
        result = solver.solve_k_paths(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            alpha=alpha,
            K=5,  # 每个α值找5条候选路径
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            # ✅ 提取最优路径信息
            best_info = {
                'alpha': alpha,
                'best_path': result['path'],
                'best_path_coords': result.get('path_coords', []),
                'earliest_arrival': result['earliest_arrival_time'],
                'expected_arrival':  result['expected_arrival_time'],
                'std_arrival': result['std_arrival_time'],
                'travel_time': result['travel_time'],
                'best_distribution': result['distribution'],  # 已经是字典格式
                'candidates': [],  # ✅ 存储所有候选路径
                'all_paths': []    # ✅ 用于分布对比图
            }
            
            # ✅ 提取所有候选路径信息
            for candidate in result.get('top_k_candidates', []):
                candidate_info = {
                    'rank': candidate['rank'],
                    'path': candidate['path'],
                    'path_coords': candidate.get('path_coords', []),
                    'earliest_arrival':  candidate['earliest_arrival'],
                    'expected_arrival': candidate['expected_arrival'],
                    'std_arrival': candidate['std_arrival'],
                    'travel_time': candidate['travel_time'],
                    'variance': candidate['variance'],
                    'is_best': candidate['is_best'],
                    'distribution': candidate['distribution']  # 已经是字典格式
                }
                best_info['candidates'].append(candidate_info)
                
                # ✅ 为分布对比图准备数据
                best_info['all_paths'].append({
                    'values': candidate['distribution']['values'],
                    'is_best': candidate['is_best'],
                    'path_length': len(candidate['path']),
                    'earliest_arrival': candidate['earliest_arrival'],
                    'expected_arrival': candidate['expected_arrival']
                })
            
            alpha_results.append(best_info)
            
            print(f" ✓ 最早={time_to_string(result['earliest_arrival_time'])}, "
                  f"候选路径数={len(result.get('top_k_candidates', []))}")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"结果验证")
    print(f"{'─'*70}")
    
    success_rate = len(alpha_results) / len(alphas) * 100
    print(f"  成功率: {len(alpha_results)}/{len(alphas)} ({success_rate:.1f}%)")
    
    if alpha_results:
        print(f"\n  详细结果（显示前10个α值）:")
        print(f"  {'α值':<8} {'最早到达':<12} {'期望到达':<12} {'候选数':<10} {'最优路径长度':<15}")
        print(f"  {'-'*70}")
        
        for r in alpha_results[:10]:  # 只显示前10个
            print(f"  {r['alpha']:<8.2f} "
                  f"{time_to_string(r['earliest_arrival']):<12} "
                  f"{time_to_string(r['expected_arrival']):<12} "
                  f"{len(r['candidates']):<10} "
                  f"{len(r['best_path']):<15}")
        
        if len(alpha_results) > 10:
            print(f"  ...(还有 {len(alpha_results) - 10} 个结果)")
    
    print(f"\n  🎉 测试通过！")
    print(f"{'='*70}\n")
    
    # ✅ 返回包含元信息的完整结果
    return {
        'alpha_results': alpha_results,
        'origin': origin,
        'destination':  destination,
        'departure_time': departure_time,
        'num_alphas': len(alphas),
        'success_count': len(alpha_results)
    }


def test_forward_multiple_od():
    """测试多OD对（K=3，每对找3条候选路径）"""
    print(f"\n{'='*70}")
    print(f"测试: 多OD对（正向，K-Paths版本）")
    print(f"{'='*70}\n")
    
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 使用config中的fast模式
    mode = config.get_mode_config('fast')
    
    solver = ForwardLabelSettingSolver(
        G, sparse_data, node_to_index, scenario_dates,
        scenario_probs, time_intervals_per_day,
        L1=mode['L1'],
        L2=mode['L2'],
        K=50,
        verbose=config.FORWARD_VERBOSE
    )
    
    # 使用config中的默认出发时间和α值
    departure_time = (config.DEFAULT_DEPARTURE_HOUR * 60 + 
                     config.DEFAULT_DEPARTURE_MINUTE) * 10
    alpha = config.FORWARD_ALPHA_DEFAULT
    
    print(f"  出发时间: {time_to_string(departure_time)}")
    print(f"  可靠性:  α={alpha}")
    print(f"  每对找K=3条候选路径")
    print(f"  配置模式: {mode['description']}\n")
    
    # 测试5个不同的OD对
    num_tests = config.NUM_TESTS
    results = []
    
    print(f"  测试 {num_tests} 个不同的OD对:")
    
    for i in range(num_tests):
        origin, destination = select_od_pair(node_to_index)
        
        print(f"    [{i+1}/{num_tests}] {origin}→{destination}...", end='', flush=True)
        
        # ✅ 调用 solve_k_paths，每对找K=3条候选路径
        result = solver.solve_k_paths(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            alpha=alpha,
            K=3,  # 每对找3条候选路径
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            od_result = {
                'origin': origin,
                'destination': destination,
                'departure_time': departure_time,
                'alpha': alpha,
                'best_path': result['path'],
                'best_path_coords':  result.get('path_coords', []),
                'earliest_arrival': result['earliest_arrival_time'],
                'expected_arrival':  result['expected_arrival_time'],
                'travel_time': result['travel_time'],
                'path_length': len(result['path']),
                'best_distribution': result['distribution'],
                'candidates': []
            }
            
            # ✅ 提取候选路径信息
            for candidate in result.get('top_k_candidates', []):
                candidate_info = {
                    'rank': candidate['rank'],
                    'path': candidate['path'],
                    'path_coords': candidate.get('path_coords', []),
                    'earliest_arrival': candidate['earliest_arrival'],
                    'expected_arrival': candidate['expected_arrival'],
                    'std_arrival': candidate['std_arrival'],
                    'travel_time': candidate['travel_time'],
                    'is_best': candidate['is_best'],
                    'distribution': candidate['distribution']
                }
                od_result['candidates'].append(candidate_info)
            
            results.append(od_result)
            print(f" ✓ 旅行={result['travel_time']/10:.1f}分, 候选数={len(result.get('top_k_candidates', []))}")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"结果验证")
    print(f"{'─'*70}")
    
    success_rate = len(results) / num_tests * 100
    print(f"  成功率: {len(results)}/{num_tests} ({success_rate:.1f}%)")
    
    if results: 
        travel_times = [r['travel_time']/10 for r in results]
        path_lengths = [r['path_length'] for r in results]
        print(f"\n  统计信息:")
        print(f"    旅行时间: 均值={np.mean(travel_times):.1f}分, "
              f"标准差={np.std(travel_times):.1f}分")
        print(f"    路径长度: 均值={np.mean(path_lengths):.1f}, "
              f"范围=[{min(path_lengths)}, {max(path_lengths)}]")
    
    print(f"\n  🎉 测试通过！")
    print(f"{'='*70}\n")
    
    return {
        'od_results': results,
        'num_tests': num_tests,
        'success_count': len(results)
    }


def clean_forward_results(results:  Dict) -> Dict:
    """
    清理结果，确保可序列化
    
    移除不可序列化的对象（如label对象），保留必要的数据
    """
    cleaned = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            cleaned[key] = clean_result_item(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_result_item(item) if isinstance(item, dict) else item for item in value]
        else:
            # 基本类型直接保留
            cleaned[key] = value
    
    return cleaned


def clean_result_item(item: Dict) -> Dict:
    """清理单个结果项"""
    if not isinstance(item, dict):
        if hasattr(item, '__dict__'):
            item = item.__dict__
        else:
            return item
    
    cleaned = {}
    
    for key, value in item.items():
        # ✅ 跳过不可序列化的字段
        if key in ['label', 'quantile_cache', 'mean_cache', 'variance_cache', 'std_cache']:   
            continue
        
        # ✅ 处理分布对象
        elif key == 'distribution':
            if hasattr(value, 'values') and hasattr(value, 'L1'):
                # 如果是分布对象，转换为字典
                cleaned[key] = {
                    'values': value.values.tolist() if hasattr(value.values, 'tolist') else list(value.values),
                    'weights': value.weights.tolist() if hasattr(value.weights, 'tolist') else list(value.weights),
                    'L1': int(value.L1)
                }
            elif isinstance(value, dict):
                # 已经是字典格式
                cleaned[key] = value
        
        # ✅ 处理列表
        elif isinstance(value, list):
            cleaned[key] = [
                clean_result_item(v) if isinstance(v, dict) or hasattr(v, '__dict__') else v 
                for v in value
            ]
        
        # ✅ 处理嵌套字典
        elif isinstance(value, dict):
            cleaned[key] = clean_result_item(value)
        
        # ✅ 处理对象
        elif hasattr(value, '__dict__'):
            cleaned[key] = clean_result_item(value.__dict__)
        
        # ✅ 基本类型
        else: 
            cleaned[key] = value
    
    return cleaned


def run_forward_tests_with_save(testname: str):
    """运行正向测试并保存结果"""
    print(f"\n{'='*70}")
    print(f"正向求解器测试套件（K-Paths版本）")
    print(f"{'='*70}")
    
    # 打印配置信息
    print(f"\n使用的配置参数:")
    print(f"  FORWARD_L1: {config.FORWARD_L1}")
    print(f"  FORWARD_L2: {config.FORWARD_L2}")
    print(f"  FORWARD_K: {config.FORWARD_K}")
    print(f"  FORWARD_ALPHA_DEFAULT: {config.FORWARD_ALPHA_DEFAULT}")
    print(f"  DEFAULT_DEPARTURE_HOUR: {config.DEFAULT_DEPARTURE_HOUR}")
    print(f"  DEFAULT_DEPARTURE_MINUTE: {config.DEFAULT_DEPARTURE_MINUTE}")
    print(f"  ALPHA_SENSITIVITY_VALUES: {len(config.ALPHA_SENSITIVITY_VALUES)} 个值")
    print()
    
    # 加载数据
    load_data_once()
    G, _, _, _, _, _ = get_data()
    
    results_all = {}
    
    try:
        print("运行正向测试1:  基本求解（K=1）...")
        results_all['test1'] = test_forward_basic()
        
        print("运行正向测试2: α敏感性分析（K=5）...")
        results_all['test2'] = test_forward_alpha_sensitivity()
        
        print("运行正向测试3: 多OD对稳定性（K=3）...")
        results_all['test3'] = test_forward_multiple_od()
        
        # 清理结果
        print("\n清理结果数据...")
        results_all = clean_forward_results(results_all)
        print("✓ 数据清理完成")
        
        # 保存结果
        print("\n保存正向测试结果...")
        save_results(results_all, solver_type='forward', output_dir=f'results/{testname}')
        
        print(f"\n{'='*70}")
        print(f"✓ 所有正向测试完成并已保存")
        print(f"  - 测试1: 单路径基本求解")
        print(f"  - 测试2: {len(results_all['test2']['alpha_results'])} 个α值的多路径分析")
        print(f"  - 测试3: {results_all['test3']['num_tests']} 个OD对的多路径测试")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 测试失败")
        print(f"{'='*70}")
        print(f"错误: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__": 
    print(f"\n{'='*70}")
    print(f"正向求解器测试程序（K-Paths版本）")
    print(f"{'='*70}")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 验证配置
    print("验证配置文件...")
    if config.validate_config():
        print("✓ 配置验证通过\n")
    else:
        print("❌ 配置验证失败，请检查 config.py\n")
        sys.exit(1)
    
    # 预加载数据
    print("预加载数据...")
    load_data_once()
    G, _, _, _, _, _ = get_data()
    print(f"✓ 路网图加载完成:  {len(G.nodes())} 个节点\n")
    print("✓ 数据加载完成\n")
    
    # 运行测试
    all_success = True
    for testname in config.TESTNAME: 
        success = run_forward_tests_with_save(f'{testname}')
        if not success:
            all_success = False

        # 为每个测试名创建对应的文件路径
        # reverse_file = f'results/{testname}/reverse_results_latest.json'
        # forward_file = f'results/{testname}/forward_results_latest.json'
        
        output_file = f'results/{testname}/{testname}.html'
        try:
            reverse_file = get_latest_results(f'reverse',output_dir =f'results/{testname}')
            print(f"✓ 找到反向结果: {reverse_file}")
        except FileNotFoundError: 
            print(f"⚠ 未找到反向结果文件")
        
        try:
            forward_file = get_latest_results(f'forward',output_dir =f'results/{testname}')
            print(f"✓ 找到正向结果: {forward_file}")
        except FileNotFoundError: 
            print(f"⚠ 未找到正向结果文件")
        
        if not reverse_file and not forward_file:
            print(f"\n❌ 错误:  未找到任何结果文件")
            print(f"请先运行测试生成结果:")
            print(f"  python run_solver.py")
            print(f"  python test_forward_solver.py")
            sys.exit(1)

        generate_html_from_files(
            G=G,
            reverse_file=reverse_file,
            forward_file=forward_file,
            output_file=output_file
        )
        
        print(f"\n{'='*70}")
        print(f"✓ 可视化生成完成！")
        print(f"{'='*70}")
        print(f"\n请在浏览器中打开: {output_file}")
        print(f"\n功能特性:")
        print(f"  ✓ 反向/正向模式切换")
        print(f"  ✓ 交互式地图展示")
        print(f"  ✓ CDF分布对比图")
        print(f"  ✓ SVG导出功能")
        print(f"  ✓ 详细数据表格")
        print(f"\n{'='*70}\n")

    
    # 退出
    sys.exit(0 if all_success else 1)