"""
统一可视化生成器（支持反向+正向，读取保存的结果文件）
功能：
1.从JSON文件读取反向/正向测试结果
2.生成统一的HTML可视化界面
3.支持切换反向/正向模式
4.支持SVG导出
5.Leaflet地图可视化
"""

import json
import gzip
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import os


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

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


def time_to_string(time_01min):
    """时间格式转换"""
    if time_01min is None:
        return "N/A"
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours: 02d}:{minutes:02d}"


def get_path_coords(G, path):
    """获取路径坐标"""
    if not path:
        return []
    coords = []
    for node in path:
        if node in G.nodes:
            node_data = G.nodes[node]
            if 'y' in node_data and 'x' in node_data:
                coords.append([node_data['y'], node_data['x']])
    return coords


def load_result_file(filename: str) -> Dict:
    """加载结果文件"""
    if not filename or not os.path.exists(filename):
        return {}
    
    print(f"  加载:  {filename}")
    
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)


# ════════════════════════════════════════════════════════════════
# 数据处理函数
# ════════════════════════════════════════════════════════════════

def process_reverse_data(G, reverse_results: Dict) -> Dict:
    """处理反向求解结果"""
    data = {
        'test1': {},
        'test2': {'summary': [], 'detailed': {}},
        'test5': []
    }
    
    # 测试1
    test1 = reverse_results.get('test1', {})
    if test1.get('success'):
        path_coords = get_path_coords(G, test1.get('path', []))
        if path_coords:
            data['test1'] = {
                'success': True,
                'origin': test1['path'][0],
                'destination':  test1['path'][-1],
                'arrival_time': time_to_string(test1.get('target_arrival_time')),
                'departure_time':  time_to_string(test1.get('latest_departure_time')),
                'expected_departure_time': time_to_string(test1.get('expected_departure_time')),
                'reserved_time': test1.get('reserved_time', 0) / 10,
                'path_length': len(test1['path']),
                'path':  test1['path'],
                'path_coords': path_coords
            }
    
        # 测试2
        test2_data = reverse_results.get('test2', {})
        if isinstance(test2_data, dict):
            # 汇总数据
            for r in test2_data.get('all_results', []):
                path_coords = get_path_coords(G, r.get('path', []))
                data['test2']['summary'].append({
                    'alpha': float(r['alpha']),
                    'latest_departure':   float(r['latest_departure']),
                    'latest_departure_str': time_to_string(r['latest_departure']),
                    'expected_departure': float(r.get('expected_departure', 0)),
                    'expected_departure_str': time_to_string(r.get('expected_departure', 0)),  # ✅ 添加这行
                    'reserved_time': float(r.get('reserved_time', 0)) / 10,
                    'path_length': int(r.get('path_length', 0)),
                    'path':  r.get('path', []),
                    'path_coords': path_coords
                })
        
        # 详细数据
        for alpha_key, detailed in test2_data.get('detailed_results', {}).items():
            alpha = float(alpha_key)
            all_paths_data = []
            
            # 兼容多种数据格式
            candidates = detailed.get('all_paths') or detailed.get('all_candidates', [])
            
            for path_info in candidates:
                dist = path_info.get('distribution', {})
                all_paths_data.append({
                    'values': dist.get('values', []),
                    'is_best': path_info.get('is_best', False) or path_info.get('rank') == 1,
                    'path_length': len(path_info.get('path', [])),
                    'latest_departure': float(path_info.get('latest_departure', 0)),
                    'expected_departure': float(path_info.get('expected_departure', 0))
                })
            
            data['test2']['detailed'][str(alpha)] = {
                'alpha': alpha,
                'num_candidates': len(all_paths_data),
                'all_paths': all_paths_data,
                'best_path_coords': get_path_coords(G, detailed.get('path', []))
            }
    
    # 测试5
    for r in reverse_results.get('test5', []):
        path_coords = get_path_coords(G, r.get('path', []))
        data['test5'].append({
            'origin': r['origin'],
            'destination': r['destination'],
            'alpha': float(r.get('alpha', 0)),
            'target_arrival_str': time_to_string(r.get('target_arrival')),
            'latest_departure_str': time_to_string(r.get('latest_dep')),
            'expected_departure_str': time_to_string(r.get('expected_dep')),
            'reserved_time':  float(r.get('reserved', 0)) / 10,
            'path_length': int(r.get('path_length', 0)),
            'path': r.get('path', []),
            'path_coords':  path_coords
        })
    
    return data


def process_forward_data(G, forward_results:  Dict) -> Dict:
    """
    处理正向求解结果(K-Paths版本 - 支持多路径分布可视化)
    
    与反向测试相同的可视化效果
    """
    data = {
        'test1': {},
        'test2': {'summary': [], 'detailed': {}},
        'test3': []
    }
    
    print(f"\n{'='*60}")
    print(f"处理正向数据(K-Paths)")
    print(f"{'='*60}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试1: 基本求解
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    test1 = forward_results. get('test1', {})
    if isinstance(test1, dict) and test1.get('success'):
        path = test1.get('path', [])
        if path:
            path_coords = get_path_coords(G, path)
            if path_coords:
                data['test1'] = {
                    'success': True,
                    'origin': path[0],
                    'destination': path[-1],
                    'departure_time': time_to_string(test1.get('departure_time')),
                    'earliest_arrival': time_to_string(test1.get('earliest_arrival_time')),
                    'expected_arrival': time_to_string(test1.get('expected_arrival_time')),
                    'travel_time': test1.get('travel_time', 0) / 10,
                    'path_length': len(path),
                    'path':  path,
                    'path_coords':  path_coords
                }
                print(f"  ✓ test1 处理成功")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试2: α敏感性分析(K-Paths格式)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    test2 = forward_results.get('test2', {})
    
    if isinstance(test2, dict) and 'alpha_results' in test2:
        alpha_results = test2['alpha_results']
        print(f"  检测到 K-Paths 格式:  {len(alpha_results)} 个α值")
        
        for alpha_result in alpha_results:
            alpha = alpha_result['alpha']
            
            # 提取最优路径信息
            best_path = alpha_result['best_path']
            best_path_coords = alpha_result. get('best_path_coords', [])
            
            # 汇总数据
            data['test2']['summary'].append({
                'alpha': float(alpha),
                'earliest_arrival': float(alpha_result['earliest_arrival']),
                'earliest_arrival_str': time_to_string(alpha_result['earliest_arrival']),
                'expected_arrival': float(alpha_result['expected_arrival']),
                'expected_arrival_str': time_to_string(alpha_result['expected_arrival']),
                'travel_time': float(alpha_result['travel_time']) / 10,
                'path_length': len(best_path),
                'path': best_path,
                'path_coords': best_path_coords
            })
            
            # ✅ 详细数据:  处理所有候选路径(与反向相同的格式)
            all_paths_data = []
            all_paths_coords = []
            
            candidates = alpha_result.get('candidates', [])
            
            for candidate in candidates:
                # ✅ 提取分布数据(与反向相同的格式)
                dist = candidate['distribution']
                all_paths_data.append({
                    'values': dist['values'],  # 用于CDF曲线
                    'is_best': candidate['is_best'],  # 标记最优路径
                    'path_length': len(candidate['path']),
                    'earliest_arrival': float(candidate['earliest_arrival']),
                    'expected_arrival': float(candidate['expected_arrival']),
                    'rank': candidate['rank']
                })
                
                # ✅ 路径坐标(用于地图显示)
                path_coords = candidate. get('path_coords', [])
                if path_coords:
                    all_paths_coords.append({
                        'coords': path_coords,
                        'is_best': candidate['is_best'],
                        'rank': candidate['rank']
                    })
            
            # ✅ 保存详细数据(与反向相同的结构)
            data['test2']['detailed'][str(alpha)] = {
                'alpha': alpha,
                'num_candidates': len(candidates),
                'all_paths':  all_paths_data,  # ✅ 关键:  与反向相同的字段名
                'best_path_coords': best_path_coords,
                'all_path_coords': all_paths_coords,
                'best_distribution': alpha_result['best_distribution'],
                'earliest_arrival':  float(alpha_result['earliest_arrival']),
                'expected_arrival': float(alpha_result['expected_arrival'])
            }
        
        print(f"  ✓ 处理了 {len(data['test2']['summary'])} 个α值")
        print(f"  ✓ 详细数据: {len(data['test2']['detailed'])} 个α值,每个包含多条候选路径")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试3: 多OD对
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    test3 = forward_results.get('test3', {})
    
    if isinstance(test3, dict):
        od_results = test3.get('od_results', [])
    elif isinstance(test3, list):
        od_results = test3
    else:
        od_results = []
    
    for od in od_results:
        if isinstance(od, dict):
            path = od.get('path', od.get('best_path', []))
            path_coords = od.get('path_coords', od.get('best_path_coords', []))
            
            data['test3'].append({
                'origin': od. get('origin'),
                'destination': od.get('destination'),
                'departure_time_str': time_to_string(od.get('departure_time')),
                'earliest_arrival_str': time_to_string(od.get('earliest_arrival')),
                'expected_arrival_str': time_to_string(od.get('expected_arrival')),
                'travel_time':  float(od.get('travel_time', 0)) / 10,
                'path_length': len(path),
                'path': path,
                'path_coords': path_coords
            })
    
    if od_results:
        print(f"  ✓ 处理了 {len(data['test3'])} 个OD对")
    
    print(f"{'='*60}\n")
    
    return data

# ════════════════════════════════════════════════════════════════
# HTML生成主函数
# ════════════════════════════════════════════════════════════════

def generate_html_from_files(G,
                             reverse_file: Optional[str] = None,
                             forward_file: Optional[str] = None,
                             output_file: str = 'solver_visualization.html'):
    """
    从保存的结果文件生成HTML可视化
    
    Args:
        G: 路网图
        reverse_file: 反向求解结果文件路径
        forward_file: 正向求解结果文件路径
        output_file: 输出HTML文件路径
    """
    
    print(f"\n{'='*70}")
    print(f"生成统一HTML可视化（从文件）")
    print(f"{'='*70}")
    
    # 加载结果
    reverse_results = {}
    forward_results = {}
    
    if reverse_file:
        reverse_results = load_result_file(reverse_file)
        print(f"  ✓ 反向结果:  {len(reverse_results)} 个测试")
    
    if forward_file:
        forward_results = load_result_file(forward_file)
        print(f"  ✓ 正向结果: {len(forward_results)} 个测试")
    
    # 处理数据
    reverse_data = process_reverse_data(G, reverse_results) if reverse_results else {}
    forward_data = process_forward_data(G, forward_results) if forward_results else {}
    
    # 构建数据JSON
    data_json = {
        'reverse': reverse_data,
        'forward': forward_data,
        'has_reverse': bool(reverse_results),
        'has_forward':  bool(forward_results)
    }
    
    # 生成HTML
    html_content = _generate_complete_html(data_json)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n  ✓ HTML可视化文件已生成: {output_file}")
    print(f"  ✓ 支持反向/正向模式切换")
    print(f"  ✓ 支持SVG导出")
    print(f"\n  请在浏览器中打开查看")
    print(f"{'='*70}\n")


def generate_html_with_svg(G, results_all_tests, output_file='reverse_solver_visualization.html'):
    """
    兼容旧接口：直接从内存中的测试结果生成HTML
    
    Args:
        G: 路网图
        results_all_tests: 测试结果字典
        output_file: 输出文件路径
    """
    print(f"\n{'='*70}")
    print(f"生成HTML+SVG可视化（兼容模式）")
    print(f"{'='*70}")
    
    # 处理数据
    reverse_data = process_reverse_data(G, results_all_tests)
    
    data_json = {
        'reverse': reverse_data,
        'forward': {},
        'has_reverse': True,
        'has_forward':  False
    }
    
    html_content = _generate_complete_html(data_json)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n  ✓ HTML可视化文件已生成:  {output_file}")
    print(f"{'='*70}\n")


def _generate_complete_html(data_json:  Dict) -> str:
    """生成完整的HTML内容"""
    
    has_reverse = data_json['has_reverse']
    has_forward = data_json['has_forward']
    
    # 模式切换按钮
    mode_buttons = ""
    if has_reverse and has_forward:
        mode_buttons = '''
        <div class="mode-selector">
            <button class="mode-button active" onclick="switchMode('reverse')">🔙 反向求解 (Reverse)</button>
            <button class="mode-button" onclick="switchMode('forward')">▶️ 正向求解 (Forward)</button>
        </div>
        '''
    elif has_reverse: 
        mode_buttons = '<div class="mode-selector"><span class="mode-label">🔙 反向求解模式</span></div>'
    elif has_forward:
        mode_buttons = '<div class="mode-selector"><span class="mode-label">▶️ 正向求解模式</span></div>'
    
    # 初始显示的模式
    initial_mode = 'reverse' if has_reverse else 'forward'
    reverse_display = 'block' if has_reverse else 'none'
    forward_display = 'none' if has_reverse else 'block'
    
    html = f'''<! DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>路径规划求解器可视化 - 反向/正向</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom:  20px;
            box-shadow:  0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 1.1em; margin-top: 5px; }}
        
        /* 模式切换器 */
        .mode-selector {{
            display: flex;
            gap: 10px;
            margin:  20px 0;
            justify-content: center;
        }}
        .mode-button {{
            padding: 15px 40px;
            border: 3px solid #667eea;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s;
            color: #667eea;
        }}
        .mode-button:hover {{ transform: scale(1.05); }}
        .mode-button.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .mode-label {{
            font-size: 1.3em;
            font-weight: 600;
            color: #667eea;
        }}
        
        /* 标签页 */
        .nav-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-button {{
            padding: 15px 30px;
            border: none;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .tab-button: hover {{ transform: translateY(-2px); }}
        .tab-button.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom:  20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .card-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin:  20px 0;
        }}
        .info-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .info-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .info-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        
        .map-container {{
            height: 500px;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        
        .selector-group {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .selector-group label {{
            font-weight: 600;
            margin-right: 10px;
        }}
        .selector-group select {{
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #667eea;
            font-size: 1em;
            min-width: 200px;
        }}
        
        .svg-container {{
            width: 100%;
            overflow-x: auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .export-button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border:  none;
            border-radius:  5px;
            cursor:  pointer;
            font-size:  1em;
            margin:  10px 5px;
        }}
        .export-button:hover {{ background: #5568d3; }}
        
        .explanation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .mode-content {{ display: none; }}
        .mode-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚗 路径规划求解器可视化</h1>
            <p class="subtitle">Reverse & Forward Label-Setting Algorithms</p>
            <p class="subtitle">生成时间:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {mode_buttons}
        </header>
        
        <!-- 反向模式内容 -->
        <div id="reverse-content" class="mode-content" style="display: {reverse_display};">
            <div class="nav-tabs">
                <button class="tab-button active" onclick="showTab('reverse', 'overview')">📊 总览</button>
                <button class="tab-button" onclick="showTab('reverse', 'test1')">🎯 测试1</button>
                <button class="tab-button" onclick="showTab('reverse', 'test2')">📈 测试2 (α敏感性)</button>
                <button class="tab-button" onclick="showTab('reverse', 'test5')">🔄 测试5 (多OD)</button>
            </div>
            
            <div id="reverse-overview" class="tab-content active">
                <div class="card">
                    <h2 class="card-title">反向求解总览</h2>
                    <div class="explanation">
                        <strong>问题: </strong> 给定<strong>目标到达时间</strong>和可靠性要求α，求解<strong>最晚出发时间</strong>和最优路径
                    </div>
                    <div id="reverseOverviewInfo"></div>
                </div>
            </div>
            
            <div id="reverse-test1" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试1: 基本求解</h2>
                    <div id="reverseTest1Info"></div>
                    <div class="map-container" id="reverseTest1Map"></div>
                </div>
            </div>
            
            <div id="reverse-test2" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试2: α敏感性分析</h2>
                    <div class="explanation">
                        <strong>说明:</strong> 选择α值查看该可靠性要求下所有候选路径的出发时间CDF分布对比
                    </div>
                    <div class="selector-group">
                        <label>选择α值:</label>
                        <select id="reverseAlphaSelect" onchange="updateReverseAlphaView()">
                            <option value="">-- 选择 --</option>
                        </select>
                    </div>
                    <div id="reverseAlphaInfo"></div>
                    <div class="svg-container" id="reverseAlphaChart"></div>
                    <button class="export-button" onclick="exportSVG('reverseAlphaChart', 'reverse_alpha_distribution')">💾 导出SVG</button>
                    <div class="map-container" id="reverseTest2Map"></div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">α敏感性汇总表</h2>
                    <table id="reverseAlphaSummaryTable">
                        <thead>
                            <tr>
                                <th>α值</th>
                                <th>最晚出发</th>
                                <th>期望出发</th>
                                <th>预留时间(分)</th>
                                <th>路径长度</th>
                            </tr>
                        </thead>
                        <tbody id="reverseAlphaSummaryBody"></tbody>
                    </table>
                </div>
            </div>
            
            <div id="reverse-test5" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试5: 多OD对稳定性</h2>
                    <div class="selector-group">
                        <label>选择OD对: </label>
                        <select id="reverseODSelect" onchange="updateReverseODView()">
                            <option value="">-- 选择 --</option>
                        </select>
                    </div>
                    <div id="reverseODInfo"></div>
                    <div class="map-container" id="reverseTest5Map"></div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">多OD对汇总表</h2>
                    <table id="reverseODSummaryTable">
                        <thead>
                            <tr>
                                <th>编号</th>
                                <th>起点</th>
                                <th>终点</th>
                                <th>α值</th>
                                <th>目标到达</th>
                                <th>最晚出发</th>
                                <th>预留(分)</th>
                                <th>路径长度</th>
                            </tr>
                        </thead>
                        <tbody id="reverseODSummaryBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- 正向模式内容 -->
        <div id="forward-content" class="mode-content" style="display: {forward_display};">
            <div class="nav-tabs">
                <button class="tab-button active" onclick="showTab('forward', 'overview')">📊 总览</button>
                <button class="tab-button" onclick="showTab('forward', 'test1')">🎯 测试1</button>
                <button class="tab-button" onclick="showTab('forward', 'test2')">📈 测试2 (α敏感性)</button>
                <button class="tab-button" onclick="showTab('forward', 'test3')">🔄 测试3 (多OD)</button>
            </div>
            
            <div id="forward-overview" class="tab-content active">
                <div class="card">
                    <h2 class="card-title">正向求解总览</h2>
                    <div class="explanation">
                        <strong>问题:</strong> 给定<strong>出发时间</strong>和可靠性要求α，求解α概率下<strong>最早到达时间</strong>和最优路径
                    </div>
                    <div id="forwardOverviewInfo"></div>
                </div>
            </div>
            
            <div id="forward-test1" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试1: 基本求解</h2>
                    <div id="forwardTest1Info"></div>
                    <div class="map-container" id="forwardTest1Map"></div>
                </div>
            </div>
            
                        <div id="forward-test2" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试2: α敏感性分析</h2>
                    
                    <div class="explanation">
                        <strong>说明:</strong> 选择α值查看该可靠性要求下的到达时间CDF分布曲线
                    </div>
                    
                    <div class="selector-group">
                        <label>选择α值（查看分布）:</label>
                        <select id="forwardAlphaSelect" onchange="updateForwardAlphaView()">
                            <option value="">-- 选择 --</option>
                        </select>
                    </div>
                    
                    <div id="forwardAlphaInfo"></div>
                    <div class="svg-container" id="forwardAlphaChart"></div>
                    <button class="export-button" onclick="exportSVG('forwardAlphaChart', 'forward_alpha_distribution')">💾 导出SVG</button>
                    <div class="map-container" id="forwardTest2Map"></div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">α敏感性汇总表</h2>
                    <table id="forwardAlphaSummaryTable">
                        <thead>
                            <tr>
                                <th>α值</th>
                                <th>最早到达</th>
                                <th>期望到达</th>
                                <th>旅行时间(分)</th>
                                <th>路径长度</th>
                            </tr>
                        </thead>
                        <tbody id="forwardAlphaSummaryBody"></tbody>
                    </table>
                </div>
            </div>
            
            <div id="forward-test3" class="tab-content">
                <div class="card">
                    <h2 class="card-title">测试3: 多OD对稳定性</h2>
                    <div class="selector-group">
                        <label>选择OD对:</label>
                        <select id="forwardODSelect" onchange="updateForwardODView()">
                            <option value="">-- 选择 --</option>
                        </select>
                    </div>
                    <div id="forwardODInfo"></div>
                    <div class="map-container" id="forwardTest3Map"></div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">多OD对汇总表</h2>
                    <table id="forwardODSummaryTable">
                        <thead>
                            <tr>
                                <th>编号</th>
                                <th>起点</th>
                                <th>终点</th>
                                <th>出发时间</th>
                                <th>最早到达</th>
                                <th>旅行时间(分)</th>
                                <th>路径长度</th>
                            </tr>
                        </thead>
                        <tbody id="forwardODSummaryBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        // 数据
        const data = {json.dumps(data_json, ensure_ascii=False, cls=NumpyEncoder)};
        
        // 地图对象
        let maps = {{}};
        
        // 当前模式
        let currentMode = '{initial_mode}';
        function formatTime(time_01min) {{
            if (!time_01min || time_01min === 0) return 'N/A';
            const totalMinutes = time_01min / 10;
            const hours = Math.floor(totalMinutes / 60);
            const minutes = Math.floor(totalMinutes % 60);
            return `${{hours.toString().padStart(2, '0')}}:${{minutes.toString().padStart(2, '0')}}`;
        }}
        // ═════════════════════════════════════════════════════════════
        // 初始化
        // ═════════════════════════════════════════════════════════════
        
        window.onload = function() {{
            if (currentMode === 'reverse' && data.has_reverse) {{
                initReverseMode();
            }} else if (currentMode === 'forward' && data.has_forward) {{
                initForwardMode();
            }}
        }};
        
        // ═════════════════════════════════════════════════════════════
        // 模式切换
        // ═════════════════════════════════════════════════════════════
        
        function switchMode(mode) {{
            currentMode = mode;
            
            // 切换按钮状态
            document.querySelectorAll('.mode-button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            // 切换内容
            document.getElementById('reverse-content').style.display = mode === 'reverse' ? 'block' : 'none';
            document.getElementById('forward-content').style.display = mode === 'forward' ? 'block' : 'none';
            
            // 初始化相应模式
            if (mode === 'reverse') {{
                initReverseMode();
            }} else {{
                initForwardMode();
            }}
        }}
        
        // ═════════════════════════════════════════════════════════════
        // 标签页切换
        // ═════════════════════════════════════════════════════════════
        
        function showTab(mode, tabName) {{
            const prefix = mode + '-';
            document.querySelectorAll(`#${{mode}}-content .tab-content`).forEach(t => t.classList.remove('active'));
            document.querySelectorAll(`#${{mode}}-content .tab-button`).forEach(b => b.classList.remove('active'));
            document.getElementById(prefix + tabName).classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'test1' && mode === 'reverse') initReverseTest1();
            if (tabName === 'test1' && mode === 'forward') initForwardTest1();
        }}
        
        // ═════════════════════════════════════════════════════════════
        // 反向模式初始化
        // ═════════════════════════════════════════════════════════════
        
        function initReverseMode() {{
            if (!data.has_reverse) return;
            
            // 总览信息
            const test1Success = data.reverse.test1 && data.reverse.test1.success;
            const overview = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">测试1状态</div>
                        <div class="info-value">${{test1Success ? '✓' : '✗'}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试2 α点数</div>
                        <div class="info-value">${{data.reverse.test2.summary.length}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试5 OD对数</div>
                        <div class="info-value">${{data.reverse.test5.length}}</div>
                    </div>
                </div>
            `;
            document.getElementById('reverseOverviewInfo').innerHTML = overview;
            
            // 填充选择器和表格
            populateReverseSelectors();
            populateReverseAlphaSummary();
            populateReverseODSummary();
        }}
        
        function populateReverseSelectors() {{
            // α选择器
            const alphaSelect = document.getElementById('reverseAlphaSelect');
            alphaSelect.innerHTML = '<option value="">-- 选择 --</option>';
            Object.keys(data.reverse.test2.detailed).forEach(alpha => {{
                alphaSelect.innerHTML += `<option value="${{alpha}}">${{parseFloat(alpha).toFixed(2)}}</option>`;
            }});
            
            // OD选择器
            const odSelect = document.getElementById('reverseODSelect');
            odSelect.innerHTML = '<option value="">-- 选择 --</option>';
            data.reverse.test5.forEach((od, i) => {{
                odSelect.innerHTML += `<option value="${{i}}">OD${{i+1}}:  ${{od.origin}} → ${{od.destination}}</option>`;
            }});
        }}
        
        function populateReverseAlphaSummary() {{
            const tbody = document.getElementById('reverseAlphaSummaryBody');
            tbody.innerHTML = '';
            data.reverse.test2.summary.forEach(r => {{
                const expectedDep = r.expected_departure_str || 
                                   (r.expected_departure ?  formatTime(r.expected_departure) : 'N/A');
                const row = `<tr>
                    <td>${{r.alpha.toFixed(2)}}</td>
                    <td>${{r.latest_departure_str}}</td>
                    <td>${{expectedDep}}</td>
                    <td>${{r.reserved_time.toFixed(1)}}</td>
                    <td>${{r.path_length}}</td>
                </tr>`;
                tbody.innerHTML += row;
            }});
        }}
        
        function populateReverseODSummary() {{
            const tbody = document.getElementById('reverseODSummaryBody');
            tbody.innerHTML = '';
            data.reverse.test5.forEach((od, i) => {{
                const row = `<tr>
                    <td>${{i+1}}</td>
                    <td>${{od.origin}}</td>
                    <td>${{od.destination}}</td>
                    <td>${{od.alpha.toFixed(2)}}</td>
                    <td>${{od.target_arrival_str}}</td>
                    <td>${{od.latest_departure_str}}</td>
                    <td>${{od.reserved_time.toFixed(1)}}</td>
                    <td>${{od.path_length}}</td>
                </tr>`;
                tbody.innerHTML += row;
            }});
        }}
        
        function initReverseTest1() {{
            if (!data.reverse.test1 || !data.reverse.test1.success) return;
            
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{data.reverse.test1.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{data.reverse.test1.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">目标到达</div>
                        <div class="info-value">${{data.reverse.test1.arrival_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最晚出发</div>
                        <div class="info-value">${{data.reverse.test1.departure_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">预留时间</div>
                        <div class="info-value">${{data.reverse.test1.reserved_time.toFixed(1)}}分</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">路径长度</div>
                        <div class="info-value">${{data.reverse.test1.path_length}}节点</div>
                    </div>
                </div>
            `;
            document.getElementById('reverseTest1Info').innerHTML = info;
            
            // 初始化地图
            if (! maps.reverseTest1 && data.reverse.test1.path_coords) {{
                initMap('reverseTest1Map', data.reverse.test1.path_coords);
            }}
        }}
        
                function updateReverseAlphaView() {{
            const alpha = document.getElementById('reverseAlphaSelect').value;
            
            if (! alpha) {{
                document.getElementById('reverseAlphaInfo').innerHTML = '';
                document.getElementById('reverseAlphaChart').innerHTML = '';
                return;
            }}
            
            const detailed = data.reverse. test2.detailed[alpha];
            if (!detailed) return;
            
            // 显示信息
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">α值</div>
                        <div class="info-value">${{parseFloat(alpha).toFixed(2)}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">候选路径数</div>
                        <div class="info-value">${{detailed.num_candidates}}</div>
                    </div>
                </div>
                <div class="explanation" style="margin-top: 15px;">
                    <strong>图表说明: </strong> 红色粗线是最优路径（在α=${{parseFloat(alpha).toFixed(2)}}分位数处最优），其他颜色是候选路径
                </div>
            `;
            document.getElementById('reverseAlphaInfo').innerHTML = info;
            
            // 生成分布图
            const svg = createDistributionSVG(detailed. all_paths, parseFloat(alpha), 'reverse');
            document.getElementById('reverseAlphaChart').innerHTML = svg;
            
            // ✅ 重建地图
            if (detailed.best_path_coords && detailed.best_path_coords.length > 0) {{
                // 销毁旧地图
                if (maps.reverseTest2) {{
                    maps.reverseTest2.remove();
                    maps.reverseTest2 = null;
                }}
                
                // 清空容器
                const mapContainer = document.getElementById('reverseTest2Map');
                mapContainer.innerHTML = '';
                
                // 重建地图
                setTimeout(() => {{
                    const center = [
                        detailed. best_path_coords.reduce((s, c) => s + c[0], 0) / detailed.best_path_coords.length,
                        detailed.best_path_coords.reduce((s, c) => s + c[1], 0) / detailed.best_path_coords.length
                    ];
                    
                    maps.reverseTest2 = L.map('reverseTest2Map').setView(center, 13);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution:  '© OpenStreetMap contributors'
                    }}).addTo(maps.reverseTest2);
                    
                    const polyline = L.polyline(detailed. best_path_coords, {{
                        color: '#FF0000',
                        weight: 5,
                        opacity: 0.7
                    }}).addTo(maps.reverseTest2);
                    
                    maps.reverseTest2.fitBounds(polyline.getBounds());
                    
                    L. circleMarker(detailed.best_path_coords[0], {{
                        radius: 10,
                        fillColor: '#00ff00',
                        color:  '#006600',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps.reverseTest2).bindPopup('起点');
                    
                    L.circleMarker(detailed.best_path_coords[detailed.best_path_coords.length - 1], {{
                        radius: 10,
                        fillColor: '#ff0000',
                        color:  '#660000',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps.reverseTest2).bindPopup('终点');
                }}, 50);
            }}
        }}
        
        function updateReverseODView() {{
            const idx = parseInt(document.getElementById('reverseODSelect').value);
            if (isNaN(idx)) return;
            
            const od = data.reverse.test5[idx];
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{od.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{od.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">α值</div>
                        <div class="info-value">${{od.alpha. toFixed(2)}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">目标到达</div>
                        <div class="info-value">${{od.target_arrival_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最晚出发</div>
                        <div class="info-value">${{od.latest_departure_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">预留时间</div>
                        <div class="info-value">${{od.reserved_time. toFixed(1)}}分</div>
                    </div>
                </div>
            `;
            document.getElementById('reverseODInfo').innerHTML = info;
            
            // ✅ 重建地图
            if (od.path_coords && od.path_coords.length > 0) {{
                // 销毁旧地图
                if (maps. reverseTest5) {{
                    maps.reverseTest5.remove();
                    maps.reverseTest5 = null;
                }}
                
                // 清空容器
                const mapContainer = document.getElementById('reverseTest5Map');
                mapContainer.innerHTML = '';
                
                // 重建地图
                setTimeout(() => {{
                    const center = [
                        od. path_coords.reduce((s, c) => s + c[0], 0) / od.path_coords.length,
                        od.path_coords.reduce((s, c) => s + c[1], 0) / od.path_coords.length
                    ];
                    
                    maps.reverseTest5 = L.map('reverseTest5Map').setView(center, 13);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(maps.reverseTest5);
                    
                    const polyline = L.polyline(od.path_coords, {{
                        color: '#FF5722',
                        weight: 5,
                        opacity: 0.7
                    }}).addTo(maps.reverseTest5);
                    
                    maps.reverseTest5.fitBounds(polyline.getBounds());
                    
                    L.circleMarker(od. path_coords[0], {{
                        radius: 10,
                        fillColor: '#00ff00',
                        color: '#006600',
                        weight:  2,
                        fillOpacity: 0.8
                    }}).addTo(maps.reverseTest5).bindPopup('起点');
                    
                    L.circleMarker(od.path_coords[od.path_coords.length - 1], {{
                        radius:  10,
                        fillColor: '#ff0000',
                        color: '#660000',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps.reverseTest5).bindPopup('终点');
                }}, 50);
            }}
        }}
        
        // ═════════════════════════════════════════════════════════════
        // 正向模式初始化
        // ═════════════════════════════════════════════════════════════
        
        function initForwardMode() {{
            if (!data.has_forward) return;
            
            const test1Success = data.forward.test1 && data.forward.test1.success;
            const overview = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">测试1状态</div>
                        <div class="info-value">${{test1Success ? '✓' :  '✗'}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试2 α点数</div>
                        <div class="info-value">${{data.forward.test2.summary.length}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">测试3 OD对数</div>
                        <div class="info-value">${{data.forward.test3.length}}</div>
                    </div>
                </div>
            `;
            document.getElementById('forwardOverviewInfo').innerHTML = overview;
            
            populateForwardSelectors();
            populateForwardAlphaSummary();
            populateForwardODSummary();
        }}
        
        function populateForwardSelectors() {{
            // ✅ α选择器（新增）
            const alphaSelect = document.getElementById('forwardAlphaSelect');
            if (alphaSelect) {{
                alphaSelect.innerHTML = '<option value="">-- 选择 --</option>';
                const alphas = Object.keys(data.forward.test2.detailed || {{}});
                console.log('Forward alpha values:', alphas);
                alphas.forEach(alpha => {{
                    alphaSelect.innerHTML += `<option value="${{alpha}}">${{parseFloat(alpha).toFixed(2)}}</option>`;
                }});
            }}
            
            // OD选择器
            const odSelect = document.getElementById('forwardODSelect');
            odSelect.innerHTML = '<option value="">-- 选择 --</option>';
            data.forward.test3.forEach((od, i) => {{
                odSelect.innerHTML += `<option value="${{i}}">OD${{i+1}}:  ${{od.origin}} → ${{od.destination}}</option>`;
            }});
        }}
        
        function populateForwardAlphaSummary() {{
            const tbody = document.getElementById('forwardAlphaSummaryBody');
            tbody.innerHTML = '';
            data.forward.test2.summary.forEach(r => {{
                const row = `<tr>
                    <td>${{r.alpha.toFixed(2)}}</td>
                    <td>${{r.earliest_arrival_str}}</td>
                    <td>${{r.expected_arrival_str || 'N/A'}}</td>
                    <td>${{r.travel_time.toFixed(1)}}</td>
                    <td>${{r.path_length}}</td>
                </tr>`;
                tbody.innerHTML += row;
            }});
        }}
        
        function populateForwardODSummary() {{
            const tbody = document.getElementById('forwardODSummaryBody');
            tbody.innerHTML = '';
            data.forward.test3.forEach((od, i) => {{
                const row = `<tr>
                    <td>${{i+1}}</td>
                    <td>${{od.origin}}</td>
                    <td>${{od.destination}}</td>
                    <td>${{od.departure_time_str}}</td>
                    <td>${{od.earliest_arrival_str}}</td>
                    <td>${{od.travel_time.toFixed(1)}}</td>
                    <td>${{od.path_length}}</td>
                </tr>`;
                tbody.innerHTML += row;
            }});
        }}
        
        function initForwardTest1() {{
            if (!data.forward.test1 || !data.forward.test1.success) return;
            
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{data.forward.test1.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{data.forward.test1.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">出发时间</div>
                        <div class="info-value">${{data.forward.test1.departure_time}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最早到达</div>
                        <div class="info-value">${{data.forward.test1.earliest_arrival}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">期望到达</div>
                        <div class="info-value">${{data.forward.test1.expected_arrival || 'N/A'}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">旅行时间</div>
                        <div class="info-value">${{data.forward.test1.travel_time.toFixed(1)}}分</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">路径长度</div>
                        <div class="info-value">${{data.forward.test1.path_length}}节点</div>
                    </div>
                </div>
            `;
            document.getElementById('forwardTest1Info').innerHTML = info;
            
            if (!maps.forwardTest1 && data.forward.test1.path_coords) {{
                initMap('forwardTest1Map', data.forward.test1.path_coords);
            }}
        }}
        
            function updateForwardAlphaView() {{
                const alpha = document.getElementById('forwardAlphaSelect').value;
                
                if (! alpha) {{
                    document.getElementById('forwardAlphaInfo').innerHTML = '';
                    document.getElementById('forwardAlphaChart').innerHTML = '';
                    return;
                }}
                
                const detailed = data. forward. test2.detailed[alpha];
                if (!detailed) {{
                    document. getElementById('forwardAlphaInfo').innerHTML = '<p>该α值没有分布数据</p>';
                    document.getElementById('forwardAlphaChart').innerHTML = '';
                    return;
                }}
                
                // ✅ 显示信息(包含候选路径数)
                const earliestArrival = formatTime(detailed.earliest_arrival);
                const expectedArrival = formatTime(detailed.expected_arrival);
                
                const info = `
                    <div class="info-grid">
                        <div class="info-box">
                            <div class="info-label">α值</div>
                            <div class="info-value">${{parseFloat(alpha).toFixed(2)}}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">候选路径数</div>
                            <div class="info-value">${{detailed.num_candidates}}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">最早到达时间</div>
                            <div class="info-value">${{earliestArrival}}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">期望到达时间</div>
                            <div class="info-value">${{expectedArrival}}</div>
                        </div>
                    </div>
                    <div class="explanation" style="margin-top: 15px;">
                        <strong>图表说明:</strong> 红色粗线是最优路径(在α=${{parseFloat(alpha).toFixed(2)}}分位数处最优),其他颜色是候选路径
                    </div>
                `;
                document.getElementById('forwardAlphaInfo').innerHTML = info;
                
                // ✅ 生成多路径分布对比图(与反向相同的函数)
                const svg = createDistributionSVG(detailed.all_paths, parseFloat(alpha), 'forward');
                document.getElementById('forwardAlphaChart').innerHTML = svg;
                
                // ✅ 地图显示多条路径
                if (detailed.all_path_coords && detailed.all_path_coords. length > 0) {{
                    updateMultiPathMap(detailed.all_path_coords, 'forwardTest2Map', parseFloat(alpha));
                }} else if (detailed.best_path_coords && detailed.best_path_coords.length > 0) {{
                    updateSinglePathMap(detailed.best_path_coords, 'forwardTest2Map');
                }}
            }}

            // ═════════════════════════════════════════════════════════════
            // 多路径地图显示(正向/反向通用)
            // ═════════════════════════════════════════════════════════════

            function updateMultiPathMap(allPathCoords, mapId, alpha) {{
                // 销毁旧地图
                if (maps[mapId]) {{
                    maps[mapId].remove();
                    maps[mapId] = null;
                }}
                
                const mapContainer = document.getElementById(mapId);
                mapContainer.innerHTML = '';
                
                setTimeout(() => {{
                    // 找到最优路径用于计算中心
                    const bestPath = allPathCoords.find(p => p.is_best);
                    if (! bestPath || !bestPath.coords || bestPath.coords.length === 0) {{
                        console.error('No valid best path coordinates');
                        return;
                    }}
                    
                    const center = [
                        bestPath.coords. reduce((s, c) => s + c[0], 0) / bestPath.coords.length,
                        bestPath.coords.reduce((s, c) => s + c[1], 0) / bestPath.coords.length
                    ];
                    
                    maps[mapId] = L.map(mapId).setView(center, 13);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(maps[mapId]);
                    
                    // 颜色方案
                    const colors = ['#4444FF', '#44FF44', '#FF44FF', '#FFAA44', '#44AAFF'];
                    
                    // 绘制所有候选路径
                    allPathCoords.forEach((pathInfo, idx) => {{
                        if (! pathInfo.coords || pathInfo. coords.length === 0) return;
                        
                        const color = pathInfo.is_best ? '#FF0000' : colors[idx % colors.length];
                        const weight = pathInfo.is_best ?  5 : 3;
                        const opacity = pathInfo.is_best ?  0.9 : 0.5;
                        
                        const polyline = L.polyline(pathInfo.coords, {{
                            color: color,
                            weight: weight,
                            opacity: opacity
                        }}).addTo(maps[mapId]);
                        
                        const label = pathInfo.is_best ? '最优路径' : `候选路径 #${{pathInfo.rank || idx+1}}`;
                        polyline.bindPopup(`<strong>${{label}}</strong>`);
                    }});
                    
                    // 添加起终点标记
                    const start = bestPath.coords[0];
                    const end = bestPath.coords[bestPath.coords.length - 1];
                    
                    L.circleMarker(start, {{
                        radius: 10,
                        fillColor: '#00ff00',
                        color: '#006600',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps[mapId]).bindPopup('起点');
                    
                    L.circleMarker(end, {{
                        radius: 10,
                        fillColor: '#ff0000',
                        color: '#660000',
                        weight: 2,
                        fillOpacity:  0.8
                    }}).addTo(maps[mapId]).bindPopup('终点');
                    
                    // 自适应缩放
                    const allCoords = allPathCoords. flatMap(p => p.coords || []);
                    if (allCoords.length > 0) {{
                        maps[mapId].fitBounds(allCoords);
                    }}
                }}, 50);
            }}

            function updateSinglePathMap(pathCoords, mapId) {{
                if (maps[mapId]) {{
                    maps[mapId].remove();
                    maps[mapId] = null;
                }}
                
                const mapContainer = document.getElementById(mapId);
                mapContainer.innerHTML = '';
                
                setTimeout(() => {{
                    const center = [
                        pathCoords. reduce((s, c) => s + c[0], 0) / pathCoords.length,
                        pathCoords.reduce((s, c) => s + c[1], 0) / pathCoords.length
                    ];
                    
                    maps[mapId] = L.map(mapId).setView(center, 13);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(maps[mapId]);
                    
                    const polyline = L.polyline(pathCoords, {{
                        color:  '#667eea',
                        weight: 5,
                        opacity: 0.7
                    }}).addTo(maps[mapId]);
                    
                    maps[mapId].fitBounds(polyline.getBounds());
                    
                    L.circleMarker(pathCoords[0], {{
                        radius: 10,
                        fillColor: '#00ff00',
                        color: '#006600',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps[mapId]).bindPopup('起点');
                    
                    L.circleMarker(pathCoords[pathCoords.length - 1], {{
                        radius:  10,
                        fillColor:  '#ff0000',
                        color: '#660000',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps[mapId]).bindPopup('终点');
                }}, 50);
            }}

                function updateForwardODView() {{
            const idx = parseInt(document.getElementById('forwardODSelect').value);
            if (isNaN(idx)) return;
            
            const od = data.forward.test3[idx];
            const info = `
                <div class="info-grid">
                    <div class="info-box">
                        <div class="info-label">起点</div>
                        <div class="info-value">${{od.origin}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">终点</div>
                        <div class="info-value">${{od.destination}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">出发时间</div>
                        <div class="info-value">${{od.departure_time_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">最早到达</div>
                        <div class="info-value">${{od.earliest_arrival_str}}</div>
                    </div>
                    <div class="info-box">
                        <div class="info-label">旅行时间</div>
                        <div class="info-value">${{od.travel_time. toFixed(1)}}分</div>
                    </div>
                </div>
            `;
            document.getElementById('forwardODInfo').innerHTML = info;
            
            // ✅ 重建地图
            if (od. path_coords && od.path_coords.length > 0) {{
                // 销毁旧地图
                if (maps.forwardTest3) {{
                    maps.forwardTest3.remove();
                    maps. forwardTest3 = null;
                }}
                
                // 清空容器
                const mapContainer = document.getElementById('forwardTest3Map');
                mapContainer.innerHTML = '';
                
                // 重建地图
                setTimeout(() => {{
                    const center = [
                        od.path_coords.reduce((s, c) => s + c[0], 0) / od.path_coords.length,
                        od.path_coords.reduce((s, c) => s + c[1], 0) / od.path_coords.length
                    ];
                    
                    maps.forwardTest3 = L.map('forwardTest3Map').setView(center, 13);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(maps.forwardTest3);
                    
                    const polyline = L.polyline(od.path_coords, {{
                        color: '#667eea',
                        weight: 5,
                        opacity: 0.7
                    }}).addTo(maps.forwardTest3);
                    
                    maps.forwardTest3.fitBounds(polyline.getBounds());
                    
                    L. circleMarker(od.path_coords[0], {{
                        radius: 10,
                        fillColor: '#00ff00',
                        color: '#006600',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps.forwardTest3).bindPopup('起点');
                    
                    L.circleMarker(od.path_coords[od. path_coords.length - 1], {{
                        radius: 10,
                        fillColor:  '#ff0000',
                        color: '#660000',
                        weight: 2,
                        fillOpacity: 0.8
                    }}).addTo(maps.forwardTest3).bindPopup('终点');
                }}, 50);
            }}
        }}
        
        // ═════════════════════════════════════════════════════════════
        // 地图初始化
        // ═════════════════════════════════════════════════════════════
        
        function initMap(mapId, pathCoords) {{
            if (!pathCoords || pathCoords.length === 0) return;
            
            const center = [
                pathCoords.reduce((s, c) => s + c[0], 0) / pathCoords.length,
                pathCoords.reduce((s, c) => s + c[1], 0) / pathCoords.length
            ];
            
            maps[mapId] = L.map(mapId).setView(center, 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(maps[mapId]);
            
            const polyline = L.polyline(pathCoords, {{color: '#667eea', weight: 5}}).addTo(maps[mapId]);
            maps[mapId].fitBounds(polyline.getBounds());
            
            L.circleMarker(pathCoords[0], {{
                radius: 10, fillColor: '#00ff00', color: '#006600',
                weight: 2, fillOpacity: 0.8
            }}).addTo(maps[mapId]).bindPopup('起点');
            
            L.circleMarker(pathCoords[pathCoords.length - 1], {{
                radius:  10, fillColor: '#ff0000', color: '#660000',
                weight: 2, fillOpacity: 0.8
            }}).addTo(maps[mapId]).bindPopup('终点');
        }}
        
        // ═════════════════════════════════════════════════════════════
        // SVG生成 - 分布对比图
        // ═════════════════════════════════════════════════════════════
        
        function createDistributionSVG(allPaths, alpha, mode) {{
            if (!allPaths || allPaths.length === 0) return '<p>无数据</p>';
            
            const width = 1200, height = 500;
            const margin = {{top: 60, right: 50, bottom: 80, left: 80}};
            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;
            
            // 计算值域
            let allValues = [];
            allPaths.forEach(p => allValues = allValues.concat(p.values));
            const minVal = Math.min(...allValues) / 10;
            const maxVal = Math.max(...allValues) / 10;
            const valRange = maxVal - minVal;
            
            let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" id="distributionSVG">`;
            
            const title = mode === 'reverse' ?  '出发时间分布对比' : '到达时间分布对比';
            svg += `<text x="${{width/2}}" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">${{title}} (α=${{alpha.toFixed(2)}})</text>`;
            
            const chartX = margin.left;
            const chartY = margin.top;
            
            // 坐标轴
            svg += `<line x1="${{chartX}}" y1="${{chartY + chartHeight}}" x2="${{chartX + chartWidth}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            svg += `<line x1="${{chartX}}" y1="${{chartY}}" x2="${{chartX}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            
            // Y轴刻度 (CDF:  0-1)
            for (let i = 0; i <= 5; i++) {{
                const yVal = i / 5;
                const py = chartY + chartHeight - (i / 5) * chartHeight;
                svg += `<text x="${{chartX - 10}}" y="${{py + 5}}" text-anchor="end" font-size="11">${{yVal.toFixed(1)}}</text>`;
                svg += `<line x1="${{chartX}}" y1="${{py}}" x2="${{chartX + chartWidth}}" y2="${{py}}" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>`;
            }}
            
            // X轴刻度
            for (let i = 0; i <= 5; i++) {{
                const xVal = minVal + (i / 5) * valRange;
                const px = chartX + (i / 5) * chartWidth;
                svg += `<text x="${{px}}" y="${{chartY + chartHeight + 25}}" text-anchor="middle" font-size="11">${{xVal.toFixed(0)}}</text>`;
            }}
            
            // 绘制路径
            const colors = ['#4444FF', '#44FF44', '#FF44FF', '#FFAA44', '#44AAFF', '#AA44FF'];
            
            allPaths.forEach((pathInfo, idx) => {{
                const values = pathInfo.values.slice().sort((a, b) => a - b);
                const n = values.length;
                
                let pathData = 'M';
                values.forEach((val, i) => {{
                    const xNorm = (val/10 - minVal) / valRange;
                    const px = chartX + xNorm * chartWidth;
                    const py = chartY + chartHeight - ((i+1)/n) * chartHeight;
                    pathData += ` ${{px}},${{py}}`;
                }});
                
                const color = pathInfo.is_best ? '#FF0000' : colors[idx % colors.length];
                const strokeWidth = pathInfo.is_best ? 4 : 1.5;
                const opacity = pathInfo.is_best ? 1.0 : 0.4;
                
                svg += `<path d="${{pathData}}" fill="none" stroke="${{color}}" stroke-width="${{strokeWidth}}" opacity="${{opacity}}"/>`;
            }});
            
            // α分位数线
            const quantileY = mode === 'reverse' ? 
                chartY + chartHeight - (1-alpha) * chartHeight : 
                chartY + chartHeight - alpha * chartHeight;
            svg += `<line x1="${{chartX}}" y1="${{quantileY}}" x2="${{chartX + chartWidth}}" y2="${{quantileY}}" `;
            svg += `stroke="orange" stroke-width="2" stroke-dasharray="8,4"/>`;
            svg += `<text x="${{chartX + chartWidth - 5}}" y="${{quantileY - 5}}" text-anchor="end" font-size="12" fill="orange" font-weight="bold">`;
            svg += `α=${{alpha.toFixed(2)}} 分位数</text>`;
            
            // 轴标签
            const xLabel = mode === 'reverse' ? '出发时间 (分钟)' : '到达时间 (分钟)';
            svg += `<text x="${{width/2}}" y="${{height - 10}}" text-anchor="middle" font-size="14" font-weight="bold">${{xLabel}}</text>`;
            svg += `<text x="20" y="${{chartY + chartHeight/2}}" text-anchor="middle" font-size="14" font-weight="bold" `;
            svg += `transform="rotate(-90 20 ${{chartY + chartHeight/2}})">累积概率 (CDF)</text>`;
            
            // 图例
            const legendX = chartX + 20;
            const legendY = chartY + 20;
            let legendItems = Math.min(allPaths.length, 6);
            let legendHeight = 25 * legendItems + 10;
            svg += `<rect x="${{legendX - 10}}" y="${{legendY - 15}}" width="200" height="${{legendHeight}}" `;
            svg += `fill="white" stroke="#ccc" stroke-width="1" opacity="0.9"/>`;
            
            let legendCount = 0;
            allPaths.forEach((pathInfo, idx) => {{
                if (legendCount >= 6) return;
                
                const color = pathInfo.is_best ?  '#FF0000' : colors[idx % colors.length];
                const label = pathInfo.is_best ?  `最优路径 (长度${{pathInfo.path_length}})` : `候选${{idx+1}}`;
                
                const ly = legendY + legendCount * 25;
                svg += `<line x1="${{legendX}}" y1="${{ly}}" x2="${{legendX + 30}}" y2="${{ly}}" stroke="${{color}}" stroke-width="3"/>`;
                svg += `<text x="${{legendX + 40}}" y="${{ly + 5}}" font-size="11">${{label}}</text>`;
                
                legendCount++;
            }});
            
            svg += '</svg>';
            return svg;
        }}
        
                // ═════════════════════════════════════════════════════════════
        // SVG生成 - 单条分布曲线（正向用）
        // ═════════════════════════════════════════════════════════════
        
        function createSingleDistributionSVG(distribution, alpha, mode) {{
            if (!distribution || !distribution.values || ! Array.isArray(distribution.values) || distribution.values.length === 0) {{
                console.log('No valid distribution data');
                return '<p>无分布数据</p>';
            }}
            
            console.log('Creating single distribution SVG with', distribution.values.length, 'values');
            
            const width = 1200, height = 500;
            const margin = {{top: 60, right: 50, bottom: 80, left: 80}};
            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;
            
            const values = distribution.values.slice().sort((a, b) => a - b);
            const minVal = Math.min(...values) / 10;
            const maxVal = Math.max(...values) / 10;
            const valRange = maxVal - minVal;
            
            let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" id="distributionSVG">`;
            
            const title = mode === 'forward' ? '到达时间分布 (CDF)' : '出发时间分布 (CDF)';
            svg += `<text x="${{width/2}}" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">${{title}} (α=${{alpha.toFixed(2)}})</text>`;
            
            const chartX = margin.left;
            const chartY = margin.top;
            
            // 坐标轴
            svg += `<line x1="${{chartX}}" y1="${{chartY + chartHeight}}" x2="${{chartX + chartWidth}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            svg += `<line x1="${{chartX}}" y1="${{chartY}}" x2="${{chartX}}" y2="${{chartY + chartHeight}}" stroke="#333" stroke-width="2"/>`;
            
            // Y轴刻度
            for (let i = 0; i <= 10; i++) {{
                const yVal = i / 10;
                const py = chartY + chartHeight - (i / 10) * chartHeight;
                svg += `<text x="${{chartX - 10}}" y="${{py + 5}}" text-anchor="end" font-size="11">${{yVal.toFixed(1)}}</text>`;
                svg += `<line x1="${{chartX}}" y1="${{py}}" x2="${{chartX + chartWidth}}" y2="${{py}}" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5"/>`;
            }}
            
            // X轴刻度
            for (let i = 0; i <= 10; i++) {{
                const xVal = minVal + (i / 10) * valRange;
                const px = chartX + (i / 10) * chartWidth;
                svg += `<text x="${{px}}" y="${{chartY + chartHeight + 25}}" text-anchor="middle" font-size="11">${{xVal.toFixed(0)}}</text>`;
                if (i % 2 === 0) {{
                    svg += `<line x1="${{px}}" y1="${{chartY + chartHeight}}" x2="${{px}}" y2="${{chartY + chartHeight + 5}}" stroke="#333" stroke-width="1"/>`;
                }}
            }}
            
            // 绘制CDF曲线
            const n = values.length;
            let pathData = 'M';
            values.forEach((val, i) => {{
                const xNorm = (val/10 - minVal) / valRange;
                const px = chartX + xNorm * chartWidth;
                const py = chartY + chartHeight - ((i+1)/n) * chartHeight;
                if (i === 0) {{
                    pathData += `${{px}},${{py}}`;
                }} else {{
                    pathData += ` L${{px}},${{py}}`;
                }}
            }});
            
            svg += `<path d="${{pathData}}" fill="none" stroke="#2E7BB4" stroke-width="3"/>`;
            
            // α分位数标记
            const quantileIdx = Math.floor(alpha * (n - 1));
            const quantileVal = values[quantileIdx] / 10;
            const quantileX = chartX + ((quantileVal - minVal) / valRange) * chartWidth;
            const quantileY = chartY + chartHeight - alpha * chartHeight;
            
            // 垂直线
            svg += `<line x1="${{quantileX}}" y1="${{chartY + chartHeight}}" x2="${{quantileX}}" y2="${{quantileY}}" `;
            svg += `stroke="#FF4444" stroke-width="2" stroke-dasharray="5,5"/>`;
            
            // 水平线
            svg += `<line x1="${{chartX}}" y1="${{quantileY}}" x2="${{quantileX}}" y2="${{quantileY}}" `;
            svg += `stroke="#FF4444" stroke-width="2" stroke-dasharray="5,5"/>`;
            
            // 标记点
            svg += `<circle cx="${{quantileX}}" cy="${{quantileY}}" r="5" fill="#FF4444" stroke="white" stroke-width="2"/>`;
            
            // 标注文本
            svg += `<text x="${{quantileX}}" y="${{chartY + chartHeight + 45}}" text-anchor="middle" font-size="12" fill="#FF4444" font-weight="bold">`;
            svg += `${{quantileVal.toFixed(1)}}分</text>`;
            
            svg += `<text x="${{chartX - 35}}" y="${{quantileY + 5}}" text-anchor="end" font-size="12" fill="#FF4444" font-weight="bold">`;
            svg += `${{alpha.toFixed(2)}}</text>`;
            
            // 轴标签
            const xLabel = mode === 'forward' ? '到达时间 (分钟)' : '出发时间 (分钟)';
            svg += `<text x="${{width/2}}" y="${{height - 10}}" text-anchor="middle" font-size="14" font-weight="bold">${{xLabel}}</text>`;
            svg += `<text x="20" y="${{chartY + chartHeight/2}}" text-anchor="middle" font-size="14" font-weight="bold" `;
            svg += `transform="rotate(-90 20 ${{chartY + chartHeight/2}})">累积概率 (CDF)</text>`;
            
            // 图例
            const legendX = chartX + chartWidth - 250;
            const legendY = chartY + 20;
            svg += `<rect x="${{legendX - 10}}" y="${{legendY - 15}}" width="240" height="80" `;
            svg += `fill="white" stroke="#ccc" stroke-width="1" opacity="0.9"/>`;
            
            svg += `<line x1="${{legendX}}" y1="${{legendY}}" x2="${{legendX + 40}}" y2="${{legendY}}" stroke="#2E7BB4" stroke-width="3"/>`;
            svg += `<text x="${{legendX + 50}}" y="${{legendY + 5}}" font-size="12">到达时间分布 (CDF)</text>`;
            
            svg += `<line x1="${{legendX}}" y1="${{legendY + 25}}" x2="${{legendX + 40}}" y2="${{legendY + 25}}" stroke="#FF4444" stroke-width="2" stroke-dasharray="5,5"/>`;
            svg += `<text x="${{legendX + 50}}" y="${{legendY + 30}}" font-size="12">α=${{alpha.toFixed(2)}} 分位数</text>`;
            
            svg += `<text x="${{legendX}}" y="${{legendY + 55}}" font-size="11" fill="#666">`;
            svg += `最早到达时间: ${{quantileVal.toFixed(1)}}分</text>`;
            
            svg += '</svg>';
            return svg;
        }}

        // ═════════════════════════════════════════════════════════════
        // SVG导出功能
        // ═════════════════════════════════════════════════════════════
        
        function exportSVG(containerId, filename) {{
            const container = document.getElementById(containerId);
            if (!container) {{
                alert('找不到SVG容器');
                return;
            }}
            
            const svgElement = container.querySelector('svg');
            if (!svgElement) {{
                alert('没有可导出的SVG图表');
                return;
            }}
            
            // 序列化SVG
            const serializer = new XMLSerializer();
            let svgString = serializer.serializeToString(svgElement);
            
            // 添加XML声明
            svgString = '<?xml version="1.0" encoding="UTF-8"?>\\n' + svgString;
            
            // 创建Blob并下载
            const blob = new Blob([svgString], {{type: 'image/svg+xml;charset=utf-8'}});
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `${{filename}}.svg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''
    
    return html


# ════════════════════════════════════════════════════════════════
# 便捷函数 - 兼容旧接口
# ════════════════════════════════════════════════════════════════

def generate_html_with_svg(G, results_all_tests, output_file='reverse_solver_visualization.html'):
    """
    兼容旧接口：直接从内存中的测试结果生成HTML
    
    Args:
        G: 路网图
        results_all_tests: 测试结果字典
        output_file: 输出文件路径
    """
    print(f"\n{'='*70}")
    print(f"生成HTML+SVG可视化（兼容模式）")
    print(f"{'='*70}")
    
    # 处理数据
    reverse_data = process_reverse_data(G, results_all_tests)
    
    data_json = {
        'reverse':  reverse_data,
        'forward': {},
        'has_reverse': True,
        'has_forward':  False
    }
    
    html_content = _generate_complete_html(data_json)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n  ✓ HTML可视化文件已生成:  {output_file}")
    print(f"{'='*70}\n")


# ════════════════════════════════════════════════════════════════
# 主程序示例
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__": 
    """
    使用示例
    """
    print("可视化生成器")
    print("=" * 70)
    print("\n使用方式：")
    print("\n1.从结果文件生成可视化：")
    print("   from visualization_generator import generate_html_from_files")
    print("   generate_html_from_files(")
    print("       G=G,")
    print("       reverse_file='results/reverse_results_latest.json',")
    print("       forward_file='results/forward_results_latest.json',")
    print("       output_file='solver_visualization.html'")
    print("   )")
    print("\n2.兼容旧接口（直接从内存）：")
    print("   from visualization_generator import generate_html_with_svg")
    print("   generate_html_with_svg(G, results_all_tests, 'output.html')")
    print("\n" + "=" * 70)