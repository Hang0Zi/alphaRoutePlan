"""
修复版Flask API - 正确处理文件上传
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import sys
import os
import time
import json
from pathlib import Path
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(__file__))

from run_solver import load_data_once, get_data, get_precomputed_data
from forward_solver import ForwardLabelSettingSolver
from reverse_solver_pseudocode import ReverseLabelSettingSolver

app = Flask(__name__, static_folder='.')
CORS(app)

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'temp'

# 全局变量
G = None
sparse_data = None
node_to_index = None
scenario_dates = None
scenario_probs = None
time_intervals_per_day = None
adj_list_forward = None
adj_list_backward = None
link_distributions = None
link_dists_backward =None
data_loaded = False

@app.route('/')
def index():
    """返回主页"""
    return send_from_directory('.', 'integrated_solver.html')

@app.route('/api/status', methods=['GET'])
def status():
    """检查服务状态"""
    return jsonify({
        'status': 'running',
        'data_loaded': data_loaded,
        'timestamp': time.time()
    })

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """加载路网数据"""
    global G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day
    global adj_list_forward, adj_list_backward, link_distributions,link_dists_backward , data_loaded
    
    try:
        print("\n" + "="*70)
        print("开始加载路网数据（优化版）...")
        print("="*70)
        
        load_data_once()
        
        G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
        adj_list_forward, adj_list_backward, link_distributions,link_dists_backward  = get_precomputed_data()
        
        data_loaded = True
        
        print("\n✓ 数据加载完成！")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': '数据加载成功（含预计算邻接表和链路分布）',
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'num_forward_adj': len(adj_list_forward),
            'num_backward_adj': len(adj_list_backward),
            'num_distributions': len(link_distributions)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message':  f'数据加载失败: {str(e)}'
        }), 500

@app.route('/api/solve', methods=['POST'])
def solve():
    """运行求解算法"""
    global G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day
    global adj_list_forward, adj_list_backward, link_distributions,link_dists_backward , data_loaded
    
    if not data_loaded:
        return jsonify({
            'success':  False,
            'message': '数据未加载，请先加载数据'
        }), 400
    
    try:
        params = request.json
        
        origin = int(params['origin'])
        destination = int(params['destination'])
        mode = params['mode']
        alpha = float(params['alpha'])
        K = int(params.get('K', 10))
        max_labels = int(params.get('max_labels', 100000))
        
        if origin not in G.nodes():
            return jsonify({'success': False, 'message':  f'起点 {origin} 不在路网中'}), 400
        if destination not in G.nodes():
            return jsonify({'success': False, 'message': f'终点 {destination} 不在路网中'}), 400
        
        print(f"\n{'='*60}")
        print(f"求解请求:  {mode} | O={origin} D={destination} α={alpha}")
        print(f"{'='*60}")
        
        result = None
        
        if mode == 'forward':
            departure_time = int(params.get('departure_time', 4800))
            
            solver = ForwardLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                adj_list=adj_list_forward,
                link_distributions=link_distributions,
                L1=50, L2=10, K=K, verbose=False
            )
            
            result = solver.solve_k_paths(
                origin=origin,
                destination=destination,
                departure_time=departure_time,
                alpha=alpha,
                K=K,
                max_labels=max_labels
            )
            
        elif mode == 'backward':

            print(max_labels)
            target_arrival_time = int(params.get('target_arrival_time', 5400))
            
            solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                L1=50, L2=10, K=K, verbose=False
            )
            
            result = solver.solve(
                origin=origin,
                destination=destination,
                target_arrival_time=target_arrival_time,
                alpha=alpha,
                max_labels=max_labels,
                K=K
            )
            print('求解成功')

        print(f"求解结果: {result is not None}")
        if result: 
            print(f"  success: {result.get('success', False)}")
            if result.get('success'):
                print(f"  path长度: {len(result.get('path', []))}")
            else:
                print(f"  失败原因: {result.get('message', '未知')}")
        
        if result and result.get('success'):
            print(f"\n[清理前验证]")
            print(f"  result的类型: {type(result)}")
            print(f"  result的键: {list(result.keys())}")
            print(f"  result['success']: {result.get('success')}")
            print(f"  result['path']存在: {'path' in result}")
            if 'path' in result: 
                print(f"  result['path']类型: {type(result['path'])}")
                print(f"  result['path']长度:  {len(result['path'])}")
                print(f"  result['path']内容: {result['path']}")
            
            cleaned_result = clean_result_for_json(result)
            
            print(f"\n[清理后验证]")
            print(f"  cleaned_result的键: {list(cleaned_result.keys())}")
            print(f"  cleaned_result['success']: {cleaned_result.get('success')}")
            print(f"  cleaned_result['path']存在: {'path' in cleaned_result}")
            if 'path' in cleaned_result: 
                print(f"  cleaned_result['path']类型: {type(cleaned_result['path'])}")
                print(f"  cleaned_result['path']长度: {len(cleaned_result['path'])}")
                print(f"  cleaned_result['path']内容: {cleaned_result['path']}")
            
            print(f"\n✓ 求解成功:  {len(cleaned_result.get('path', []))} 个节点")
            print(f"✓ 求解耗时: {cleaned_result.get('total_time', 0):.2f}秒")
            print(f"{'='*60}\n")
            
            return jsonify(cleaned_result)
        else:
            error_msg = '求解失败，未找到路径'
            if result and 'message' in result:
                error_msg = result['message']
            
            print(f"✗ {error_msg}")
            print(f"{'='*60}\n")
            
            return jsonify({
                'success': False,
                'message': error_msg
            }), 404
    
    except Exception as e:  
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ 求解异常:\n{error_trace}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'message': f'求解出错: {str(e)}'
        }), 500


@app.route('/api/generate-visualization', methods=['POST'])
def generate_visualization():
    """✨ 生成批量可视化HTML"""
    global G, data_loaded
    
    print("\n" + "="*70)
    print("收到可视化生成请求")
    print("="*70)
    
    # ✨ 调试信息
    print(f"Content-Type: {request.content_type}")
    print(f"Files: {request.files}")
    print(f"Form:  {request.form}")
    
    if not data_loaded or not G: 
        print("✗ 数据未加载")
        return jsonify({
            'success': False,
            'message': '数据未加载，请先加载数据'
        }), 400
    
    try:
        # ✨ 检查文件
        reverse_file = request.files.get('reverse_file')
        forward_file = request.files.get('forward_file')
        
        print(f"Reverse file: {reverse_file}")
        print(f"Forward file: {forward_file}")
        
        if not reverse_file and not forward_file:
            print("✗ 未找到文件")
            return jsonify({
                'success': False,
                'message': '请至少上传一个结果文件'
            }), 400
        
        # 创建临时目录
        temp_dir = Path(app.config['UPLOAD_FOLDER'])
        temp_dir.mkdir(exist_ok=True)
        
        reverse_path = None
        forward_path = None
        
        # 保存上传的文件
        if reverse_file and reverse_file.filename:
            filename = secure_filename(reverse_file.filename)
            reverse_path = temp_dir / f'reverse_{int(time.time())}_{filename}'
            reverse_file.save(str(reverse_path))
            print(f"  ✓ 保存反向结果:  {reverse_path}")
        
        if forward_file and forward_file.filename:
            filename = secure_filename(forward_file.filename)
            forward_path = temp_dir / f'forward_{int(time.time())}_{filename}'
            forward_file.save(str(forward_path))
            print(f"  ✓ 保存正向结果: {forward_path}")
        
        # 生成可视化
        print("\n开始生成可视化...")
        
        from visualization_generator import generate_html_from_files
        
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'viz_{int(time.time())}.html'
        
        generate_html_from_files(
            G=G,
            reverse_file=str(reverse_path) if reverse_path else None,
            forward_file=str(forward_path) if forward_path else None,
            output_file=str(output_file)
        )
        
        print(f"  ✓ 可视化已生成: {output_file}")
        print("="*70 + "\n")
        
        # 清理临时文件
        if reverse_path and reverse_path.exists():
            reverse_path.unlink()
            print(f"  ✓ 清理临时文件: {reverse_path}")
        if forward_path and forward_path.exists():
            forward_path.unlink()
            print(f"  ✓ 清理临时文件: {forward_path}")
        
        # 返回结果
        view_url = f'/view-visualization/{output_file.name}'
        
        return jsonify({
            'success': True,
            'message': '可视化生成成功',
            'output_file': str(output_file),
            'view_url': view_url
        })
    
    except Exception as e: 
        import traceback
        traceback.print_exc()
        return jsonify({
            'success':  False,
            'message': f'生成可视化失败: {str(e)}'
        }), 500


@app.route('/view-visualization/<filename>')
def view_visualization(filename):
    """✨ 查看生成的可视化"""
    try:
        viz_path = Path('visualizations') / filename
        if not viz_path.exists():
            return f"文件未找到: {filename}", 404
        return send_file(viz_path)
    except Exception as e: 
        return f"错误:  {e}", 500


@app.route('/api/list-results', methods=['GET'])
def list_results():
    """列出所有保存的结果文件"""
    try:
        results_dir = Path('results')
        if not results_dir.exists():
            return jsonify({
                'success': True,
                'files': []
            })
        
        files = []
        for test_dir in results_dir.iterdir():
            if test_dir.is_dir():
                for file in test_dir.glob('*.json'):
                    files.append({
                        'path': str(file),
                        'name': file.name,
                        'test_name': test_dir.name,
                        'size': file.stat().st_size,
                        'modified': file.stat().st_mtime
                    })
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message':  str(e)
        }), 500
"""
修改后端view_result - 返回简要摘要而不是原始JSON
"""

@app.route('/api/view-result', methods=['POST'])
def view_result():
    """✨ 查看结果文件内容 - 返回简要摘要"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        print(f"\n{'='*60}")
        print(f"查看结果文件: {file_path}")
        print(f"{'='*60}")
        
        if not file_path:  
            return jsonify({
                'success': False,
                'message':  '未提供文件路径'
            }), 400
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            print(f"  ✗ 文件不存在")
            return jsonify({
                'success': False,
                'message': f'文件不存在: {file_path}'
            }), 404
        
        # 读取JSON文件
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        print(f"  ✓ 文件读取成功")
        print(f"  顶层键: {list(result_data.keys())}")
        
        # ✨ 解析和提取简要信息
        parsed = parse_result_summary(result_data)
        
        print(f"  识别到的测试:  {parsed['test_names']}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'parsed': parsed,  # 只返回解析后的摘要
            'file_info': {
                'name': file_path_obj.name,
                'size': file_path_obj.stat().st_size,
                'modified': file_path_obj.stat().st_mtime
            }
        })
    
    except Exception as e:  
        import traceback
        error_trace = traceback.format_exc()
        print(f"  ✗ 错误:\n{error_trace}")
        
        return jsonify({
            'success': False,
            'message': f'读取文件失败: {str(e)}'
        }), 500
"""
根据实际数据结构重写解析函数
"""

def parse_result_summary(data):
    """
    解析结果数据，提取关键摘要信息
    """
    
    parsed = {
        'test_names': [],
        'tests':  {}
    }
    
    print(f"  开始解析，顶层键: {list(data.keys())}")
    
    # 遍历顶层键，识别测试
    for key, value in data.items():
        # 跳过元数据
        if key.startswith('_') or not isinstance(value, dict):
            continue
        
        print(f"  处理:  {key}, 类型: {type(value)}, 键: {list(value.keys()) if isinstance(value, dict) else 'N/A'}")
        
        test_info = None
        
        # 识别测试类型
        if key == 'test1':
            # test1是基础单次求解测试
            test_info = extract_test1_summary(value)
            
        elif key == 'test2':
            # test2是Alpha敏感性分析 - 新结构：alpha_results
            if 'alpha_results' in value: 
                test_info = extract_alpha_results_summary(value)
            elif 'all_results' in value:
                # 旧结构兼容
                test_info = extract_alpha_sensitivity_summary(value)
            else:
                test_info = extract_generic_test_summary(value, key)
                
        elif key == 'test3':
            # test3是多OD对测试 - 新结构：od_results
            if 'od_results' in value:
                test_info = extract_od_results_summary(value)
            else:
                test_info = extract_performance_summary(value)
            
        elif key.startswith('test'):
            # 其他测试
            test_info = extract_generic_test_summary(value, key)
        
        if test_info:
            print(f"  ✓ {key} 解析成功:  {test_info['type']}")
            parsed['test_names'].append(key)
            parsed['tests'][key] = test_info
        else:
            print(f"  ✗ {key} 解析失败")
    
    print(f"  最终解析结果: {len(parsed['tests'])} 个测试")
    return parsed


def extract_test1_summary(test_data):
    """提取test1基础测试的摘要"""
    
    summary = {
        'type': 'basic_test',
        'name': '基础求解测试',
        'overview': {},
        'result': {}
    }
    
    print(f"    [test1] 数据键: {list(test_data.keys())}")
    
    # test1直接包含求解结果
    if test_data.get('success', True):  # 默认认为成功
        summary['overview']['success'] = True
        
        # 提取关键信息
        summary['result'] = {
            'origin': test_data.get('origin'),
            'destination': test_data.get('destination'),
            'path_length': len(test_data.get('path', [])),
            'total_time': test_data.get('total_time', 0),
            'iterations': test_data.get('iterations', 0),
            'alpha': test_data.get('alpha', 0),
            'num_candidates': test_data.get('num_candidates', 0),
        }
        
        # Forward solver特有字段
        if 'departure_time' in test_data: 
            summary['result']['solver_type'] = 'forward'
            summary['result']['departure_time'] = test_data.get('departure_time', 0) / 10
            summary['result']['earliest_arrival'] = test_data.get('earliest_arrival_time', 0) / 10
            summary['result']['expected_arrival'] = test_data.get('expected_arrival_time', 0) / 10
            summary['result']['median_arrival'] = test_data.get('median_arrival_time', 0) / 10
            summary['result']['std_arrival'] = test_data.get('std_arrival_time', 0) / 10
            summary['result']['travel_time'] = test_data.get('travel_time', 0) / 10
        
        # Backward solver特有字段
        if 'target_arrival_time' in test_data: 
            summary['result']['solver_type'] = 'backward'
            summary['result']['target_arrival'] = test_data.get('target_arrival_time', 0) / 10
            summary['result']['latest_departure'] = test_data.get('latest_departure_time', 0) / 10
            summary['result']['expected_departure'] = test_data.get('expected_departure_time', 0) / 10
            summary['result']['reserved_time'] = test_data.get('reserved_time', 0) / 10
        
        print(f"    [test1] ✓ 成功解析，类型: {summary['result'].get('solver_type', 'unknown')}")
    else:
        summary['overview']['success'] = False
        summary['result']['error'] = test_data.get('error', '未知错误')
        print(f"    [test1] ✗ 求解失败")
    
    return summary


def extract_alpha_results_summary(test_data):
    """提取test2 Alpha敏感性分析的摘要 - 新结构"""
    
    summary = {
        'type': 'alpha_sensitivity',
        'name': 'Alpha敏感性分析',
        'overview': {},
        'key_results': [],
        'statistics': {},
        'full_results': []
    }
    
    print(f"    [test2] 数据键: {list(test_data.keys())}")
    
    # 新结构：alpha_results
    if 'alpha_results' in test_data:
        alpha_results = test_data['alpha_results']
        print(f"    [test2] alpha_results数量: {len(alpha_results)}")
        
        summary['overview'] = {
            'total_tests': len(alpha_results),
            'success_count': test_data.get('success_count', len(alpha_results)),
            'origin': test_data.get('origin'),
            'destination': test_data.get('destination'),
            'departure_time': test_data.get('departure_time', 0) / 10,
            'num_alphas': test_data.get('num_alphas', len(alpha_results)),
        }
        
        if alpha_results:
            # 提取有效结果（有best_path的）
            valid_results = [r for r in alpha_results if r.get('best_path')]
            
            if valid_results:
                alphas = [r['alpha'] for r in valid_results]
                travel_times = [r.get('travel_time', 0) / 10 for r in valid_results]
                path_lengths = [len(r.get('best_path', [])) for r in valid_results]
                
                summary['statistics'] = {
                    'alpha_range': [min(alphas), max(alphas)],
                    'avg_travel_time': sum(travel_times) / len(travel_times),
                    'min_travel_time': min(travel_times),
                    'max_travel_time': max(travel_times),
                    'avg_path_length': sum(path_lengths) / len(path_lengths),
                }
                
                # 提取关键结果点
                key_indices = []
                if len(valid_results) > 0:
                    key_indices.append(0)
                if len(valid_results) > 4:
                    key_indices.append(len(valid_results) // 4)
                if len(valid_results) > 2:
                    key_indices.append(len(valid_results) // 2)
                if len(valid_results) > 4:
                    key_indices.append(3 * len(valid_results) // 4)
                if len(valid_results) > 1:
                    key_indices.append(len(valid_results) - 1)
                
                summary['key_results'] = [
                    {
                        'alpha': valid_results[i]['alpha'],
                        'earliest_arrival': valid_results[i].get('earliest_arrival', 0) / 10,
                        'expected_arrival': valid_results[i].get('expected_arrival', 0) / 10,
                        'travel_time': valid_results[i].get('travel_time', 0) / 10,
                        'path_length': len(valid_results[i].get('best_path', [])),
                        'std_arrival': valid_results[i].get('std_arrival', 0) / 10,
                    }
                    for i in key_indices if i < len(valid_results)
                ]
                
                # 完整结果
                summary['full_results'] = [
                    {
                        'alpha': r['alpha'],
                        'earliest_arrival': r.get('earliest_arrival', 0) / 10,
                        'expected_arrival': r.get('expected_arrival', 0) / 10,
                        'travel_time': r.get('travel_time', 0) / 10,
                        'path_length': len(r.get('best_path', [])),
                        'std_arrival': r.get('std_arrival', 0) / 10,
                    }
                    for r in valid_results
                ]
                
                print(f"    [test2] ✓ 成功解析 {len(valid_results)}/{len(alpha_results)} 个有效结果")
            else:
                print(f"    [test2] ⚠ 没有有效结果（无best_path）")
    
    return summary


def extract_od_results_summary(test_data):
    """提取test3 多OD对测试的摘要 - 新结构"""
    
    summary = {
        'type': 'multi_od_test',
        'name': '多OD对测试',
        'overview': {},
        'key_results': [],
        'statistics': {},
        'full_results': []
    }
    
    print(f"    [test3] 数据键: {list(test_data.keys())}")
    
    # 新结构：od_results
    if 'od_results' in test_data: 
        od_results = test_data['od_results']
        print(f"    [test3] od_results数量: {len(od_results)}")
        
        summary['overview'] = {
            'total_tests':  len(od_results),
            'success_count': test_data.get('success_count', len(od_results)),
            'num_tests': test_data.get('num_tests', len(od_results)),
        }
        
        if od_results:
            # 提取有效结果
            valid_results = [r for r in od_results if r.get('best_path')]
            
            if valid_results: 
                travel_times = [r.get('travel_time', 0) / 10 for r in valid_results]
                path_lengths = [r.get('path_length', 0) for r in valid_results]
                earliest_arrivals = [r.get('earliest_arrival', 0) / 10 for r in valid_results]
                
                summary['statistics'] = {
                    'avg_travel_time': sum(travel_times) / len(travel_times),
                    'min_travel_time': min(travel_times),
                    'max_travel_time': max(travel_times),
                    'avg_path_length': sum(path_lengths) / len(path_lengths),
                    'min_path_length': min(path_lengths),
                    'max_path_length': max(path_lengths),
                }
                
                # 提取关键结果点（前5个）
                summary['key_results'] = [
                    {
                        'origin': r.get('origin'),
                        'destination': r.get('destination'),
                        'departure_time': r.get('departure_time', 0) / 10,
                        'alpha': r.get('alpha', 0),
                        'earliest_arrival': r.get('earliest_arrival', 0) / 10,
                        'expected_arrival': r.get('expected_arrival', 0) / 10,
                        'travel_time':  r.get('travel_time', 0) / 10,
                        'path_length': r.get('path_length', 0),
                    }
                    for r in valid_results[:5]  # 只取前5个
                ]
                
                # 完整结果（简化版）
                summary['full_results'] = [
                    {
                        'origin': r.get('origin'),
                        'destination': r.get('destination'),
                        'travel_time': r.get('travel_time', 0) / 10,
                        'path_length': r.get('path_length', 0),
                        'earliest_arrival': r.get('earliest_arrival', 0) / 10,
                        'expected_arrival': r.get('expected_arrival', 0) / 10,
                    }
                    for r in valid_results
                ]
                
                print(f"    [test3] ✓ 成功解析 {len(valid_results)}/{len(od_results)} 个有效结果")
            else:
                print(f"    [test3] ⚠ 没有有效结果")
    
    return summary


def extract_alpha_sensitivity_summary(test_data):
    """提取Alpha敏感性分析的摘要 - 旧结构兼容"""
    
    summary = {
        'type': 'alpha_sensitivity',
        'name': 'Alpha敏感性分析',
        'overview': {},
        'key_results': [],
        'statistics': {},
        'full_results': []
    }
    
    print(f"    [test2-旧] 数据键: {list(test_data.keys())}")
    
    # 旧结构：all_results
    if 'all_results' in test_data: 
        all_results = test_data['all_results']
        print(f"    [test2-旧] all_results数量: {len(all_results)}")
        
        summary['overview'] = {
            'total_tests': len(all_results),
            'origin': test_data.get('origin'),
            'destination': test_data.get('destination'),
            'target_arrival': test_data.get('target_arrival_time', 0) / 10,
        }
        
        if all_results:
            alphas = [r['alpha'] for r in all_results]
            reserved_times = [r['reserved_time'] / 10 for r in all_results]
            path_lengths = [len(r['best_path']) for r in all_results]
            
            summary['statistics'] = {
                'alpha_range': [min(alphas), max(alphas)],
                'avg_reserved_time': sum(reserved_times) / len(reserved_times),
                'min_reserved_time': min(reserved_times),
                'max_reserved_time':  max(reserved_times),
                'avg_path_length': sum(path_lengths) / len(path_lengths),
            }
            
            # 提取关键结果点
            key_indices = []
            if len(all_results) > 0:
                key_indices.append(0)
            if len(all_results) > 4:
                key_indices.append(len(all_results) // 4)
            if len(all_results) > 2:
                key_indices.append(len(all_results) // 2)
            if len(all_results) > 4:
                key_indices.append(3 * len(all_results) // 4)
            if len(all_results) > 1:
                key_indices.append(len(all_results) - 1)
            
            summary['key_results'] = [
                {
                    'alpha': all_results[i]['alpha'],
                    'latest_departure': all_results[i]['latest_departure'] / 10,
                    'expected_departure':  all_results[i]['expected_departure'] / 10,
                    'reserved_time': all_results[i]['reserved_time'] / 10,
                    'path_length': all_results[i]['path_length'],
                    'target_arrival': all_results[i]['target_arrival'] / 10,
                }
                for i in key_indices if i < len(all_results)
            ]
            
            # 完整结果
            summary['full_results'] = [
                {
                    'alpha': r['alpha'],
                    'latest_departure': r['latest_departure'] / 10,
                    'expected_departure': r['expected_departure'] / 10,
                    'reserved_time': r['reserved_time'] / 10,
                    'path_length': r['path_length'],
                }
                for r in all_results
            ]
            
            print(f"    [test2-旧] ✓ 成功解析 {len(all_results)} 个结果点")
    
    return summary


def extract_performance_summary(test_data):
    """提取性能测试摘要 - 通用版"""
    
    summary = {
        'type': 'performance',
        'name':  '性能测试',
        'overview': {},
        'results': []
    }
    
    print(f"    [test3-旧] 数据键: {list(test_data.keys())}")
    
    # 简单处理
    if 'success' in test_data:
        if test_data['success']:
            summary['overview'] = {
                'success': True,
                'total_time': test_data.get('total_time', 0),
                'iterations': test_data.get('iterations', 0),
            }
            print(f"    [test3-旧] ✓ 测试成功")
        else:
            summary['overview'] = {
                'success': False,
                'error': test_data.get('error', '未知错误')
            }
            print(f"    [test3-旧] ✗ 测试失败")
    else:
        summary['overview'] = {
            'note': '复杂结构，请查看完整JSON'
        }
        print(f"    [test3-旧] ⚠ 复杂结构，建议查看原始数据")
    
    return summary


def extract_generic_test_summary(test_data, test_name):
    """提取通用测试摘要"""
    
    summary = {
        'type': 'generic',
        'name': test_name,
        'overview': {}
    }
    
    print(f"    [{test_name}] 通用解析，数据键: {list(test_data.keys())}")
    
    # 检查是否有success字段
    if 'success' in test_data:
        summary['overview']['success'] = test_data['success']
        
        if test_data['success']:
            # 提取所有可能的关键字段
            key_fields = {
                'origin': '起点',
                'destination':  '终点',
                'path': '路径',
                'path_length': '路径长度',
                'total_time': '求解时间',
                'iterations': '迭代次数',
                'alpha': '可靠性',
                'departure_time': '出发时间',
                'earliest_arrival_time': '最早到达',
                'expected_arrival_time': '期望到达',
                'target_arrival_time': '目标到达',
                'latest_departure_time': '最晚出发',
                'expected_departure_time': '期望出发',
                'travel_time': '旅行时间',
                'reserved_time': '预留时间',
            }
            
            for field, label in key_fields.items():
                if field in test_data:
                    value = test_data[field]
                    
                    # 时间字段转换
                    if 'time' in field and isinstance(value, (int, float)) and value > 100:
                        summary['overview'][field] = value / 10
                    elif field == 'path': 
                        summary['overview']['path_length'] = len(value)
                    else:
                        summary['overview'][field] = value
            
            print(f"    [{test_name}] ✓ 提取了 {len(summary['overview'])} 个字段")
        else:
            summary['overview']['error'] = test_data.get('error', '未知错误')
            print(f"    [{test_name}] ✗ 求解失败")
    else:
        # 没有success字段，可能是其他结构
        summary['overview']['note'] = f'无法识别的 {test_name} 结构'
        summary['overview']['keys'] = list(test_data.keys())
        print(f"    [{test_name}] ⚠ 未识别的结构")
    
    return summary


@app.route('/api/download-result/<path:filepath>')
def download_result(filepath):
    """✨ 下载结果文件"""
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=file_path.name
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'下载失败: {str(e)}'
        }), 500

def clean_result_for_json(result):
    """清理结果，确保可JSON序列化"""
    import numpy as np
    
    cleaned = {}
    
    keep_fields = [
        'success', 'path', 'path_coords',
        'earliest_arrival_time', 'expected_arrival_time', 
        'median_arrival_time', 'std_arrival_time',
        'latest_departure_time', 'expected_departure_time',
        'median_departure_time', 'std_departure_time',
        'travel_time', 'reserved_time',
        'departure_time', 'target_arrival_time',
        'total_time', 'iterations', 'alpha', 'K',
        'origin', 'destination', 'num_candidates'
    ]
    
    for field in keep_fields:
        if field in result:
            value = result[field]
            
            # 🔍 调试输出
            if field in ['success', 'path']: 
                print(f"[clean] 处理字段 {field}: 类型={type(value)}, 值={value if field != 'path' else ('列表长度' + str(len(value)) if isinstance(value, list) else '?')}")
            
            if isinstance(value, bool):
                cleaned[field] = value
            elif isinstance(value, (np.integer, np.int32, np.int64)):
                cleaned[field] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                cleaned[field] = float(value)
            elif isinstance(value, np.ndarray):
                cleaned[field] = value.tolist()
            elif isinstance(value, list):
                cleaned[field] = convert_list(value)
            elif isinstance(value, dict):
                cleaned[field] = convert_dict(value)
            else:
                cleaned[field] = value
    
    # 🔍 验证清理后的关键字段
    print(f"[clean] 清理后 success存在: {'success' in cleaned}, 值:  {cleaned.get('success')}")
    print(f"[clean] 清理后 path存在: {'path' in cleaned}, 类型: {type(cleaned.get('path'))}")
    if 'path' in cleaned: 
        print(f"[clean] path长度: {len(cleaned['path']) if isinstance(cleaned['path'], list) else '不是列表!'}")
        if isinstance(cleaned['path'], list) and len(cleaned['path']) > 0:
            print(f"[clean] path前3个元素: {cleaned['path'][:3]}")
    
    if 'distribution' in result:
        dist = result['distribution']
        if isinstance(dist, dict):
            cleaned['distribution'] = {
                'values': convert_to_list(dist.get('values', [])),
                'weights': convert_to_list(dist.get('weights', [])),
                'L1': int(dist.get('L1', 0)) if dist.get('L1') is not None else 0
            }
    
    if 'path' in result: 
        pa = result['path']
        cleaned['length'] = len(pa) if isinstance(pa, list) else 0
    
    if 'top_k_candidates' in result:
        candidates = result['top_k_candidates']
        cleaned['top_k_candidates'] = []
        
        for i, candidate in enumerate(candidates[: 10]):
            cleaned_candidate = {
                'rank': i + 1,
                'path_length': len(candidate.get('path', [])),
                'is_best':  bool(candidate.get('is_best', False))  # 确保是布尔值
            }
            
            for time_field in ['earliest_arrival', 'expected_arrival', 'latest_departure', 'expected_departure']:
                if time_field in candidate: 
                    value = candidate[time_field]
                    if isinstance(value, (np.integer, np.int32, np.int64)):
                        cleaned_candidate[time_field] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        cleaned_candidate[time_field] = float(value)
                    else:
                        cleaned_candidate[time_field] = value
            
            cleaned['top_k_candidates'].append(cleaned_candidate)
    
    if 'stats' in result: 
        cleaned['stats'] = convert_dict(result['stats'])
    
    return cleaned


def convert_to_list(value):
    """转换为列表"""
    import numpy as np
    
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [convert_to_list(v) if isinstance(v, (np.ndarray, list)) else v for v in value]
    else:
        return value


def convert_list(lst):
    """递归转换列表 - 修复版"""
    import numpy as np
    
    if not isinstance(lst, list):
        print(f"[convert_list] 警告: 输入不是列表，类型: {type(lst)}")
        return lst
    
    result = []
    for item in lst:
        if isinstance(item, (np.integer, np.int32, np.int64)):
            result.append(int(item))
        elif isinstance(item, (np.floating, np.float32, np.float64)):
            result.append(float(item))
        elif isinstance(item, np.ndarray):
            result.append(item.tolist())
        elif isinstance(item, list):
            result.append(convert_list(item))
        elif isinstance(item, dict):
            result.append(convert_dict(item))
        elif isinstance(item, bool):
            result.append(item)
        elif isinstance(item, str):
            result.append(item)
        elif item is None:
            result.append(None)
        else:
            # 🔍 对于其他类型，尝试转换
            try: 
                result.append(int(item))
            except: 
                result.append(str(item))
    
    return result


def convert_dict(dct):
    """递归转换字典 - 修复版"""
    import numpy as np
    
    if not isinstance(dct, dict):
        print(f"[convert_dict] 警告: 输入不是字典，类型: {type(dct)}")
        return dct
    
    result = {}
    for key, value in dct.items():
        if isinstance(value, (np.integer, np.int32, np.int64)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, list):
            result[key] = convert_list(value)
        elif isinstance(value, dict):
            result[key] = convert_dict(value)
        elif isinstance(value, bool):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value
        elif value is None:
            result[key] = None
        else:
            result[key] = value
    
    return result


# def clean_result_for_json(result):
#     """清理结果"""
#     import numpy as np
    
#     cleaned = {}
    
#     keep_fields = [
#         'success', 'path', 'path_coords',
#         'earliest_arrival_time', 'expected_arrival_time', 
#         'median_arrival_time', 'std_arrival_time',
#         'latest_departure_time', 'expected_departure_time',
#         'median_departure_time', 'std_departure_time',
#         'travel_time', 'reserved_time',
#         'departure_time', 'target_arrival_time',
#         'total_time', 'iterations', 'alpha', 'K',
#         'origin', 'destination', 'num_candidates'
#     ]
    
#     for field in keep_fields:
#         if field in result:
#             value = result[field]
#             if isinstance(value, bool):
#                 cleaned[field] = value  # Python的True会自动转为JSON的true
#             elif isinstance(value, (np.integer, np.int32, np.int64)):
#                 cleaned[field] = int(value)
#             elif isinstance(value, (np.floating, np.float32, np.float64)):
#                 cleaned[field] = float(value)
#             elif isinstance(value, np.ndarray):
#                 cleaned[field] = value.tolist()
#             elif isinstance(value, list):
#                 cleaned[field] = convert_list(value)
#             elif isinstance(value, dict):
#                 cleaned[field] = convert_dict(value)
#             else:
#                 cleaned[field] = value
    
#     if 'distribution' in result:
#         dist = result['distribution']
#         if isinstance(dist, dict):
#             cleaned['distribution'] = {
#                 'values': convert_to_list(dist.get('values', [])),
#                 'weights': convert_to_list(dist.get('weights', [])),
#                 'L1': int(dist.get('L1', 0)) if dist.get('L1') is not None else 0
#             }
    
#     if 'path' in result:
#         pa = result['path']
#         cleaned['length'] = len(pa)

#     if 'top_k_candidates' in result:
#         candidates = result['top_k_candidates']
#         cleaned['top_k_candidates'] = []
        
#         for i, candidate in enumerate(candidates[: 10]):
#             cleaned_candidate = {
#                 'rank': i + 1,
#                 'path_length': len(candidate.get('path', [])),
#                 'is_best': candidate.get('is_best', False)
#             }
            
#             for time_field in ['earliest_arrival', 'expected_arrival', 'latest_departure', 'expected_departure']:
#                 if time_field in candidate: 
#                     value = candidate[time_field]
#                     if isinstance(value, (np.integer, np.int32, np.int64)):
#                         cleaned_candidate[time_field] = int(value)
#                     elif isinstance(value, (np.floating, np.float32, np.float64)):
#                         cleaned_candidate[time_field] = float(value)
#                     else:
#                         cleaned_candidate[time_field] = value
            
#             cleaned['top_k_candidates'].append(cleaned_candidate)
    
#     if 'stats' in result:
#         cleaned['stats'] = convert_dict(result['stats'])
    
#     return cleaned


# def convert_to_list(value):
#     """转换为列表"""
#     import numpy as np
    
#     if isinstance(value, np.ndarray):
#         return value.tolist()
#     elif isinstance(value, list):
#         return [convert_to_list(v) for v in value]
#     else:
#         return value


# def convert_list(lst):
#     """递归转换列表"""
#     import numpy as np
    
#     result = []
#     for item in lst:
#         if isinstance(item, (np.integer, np.int32, np.int64)):
#             result.append(int(item))
#         elif isinstance(item, (np.floating, np.float32, np.float64)):
#             result.append(float(item))
#         elif isinstance(item, np.ndarray):
#             result.append(item.tolist())
#         elif isinstance(item, list):
#             result.append(convert_list(item))
#         elif isinstance(item, dict):
#             result.append(convert_dict(item))
#         else:
#             result.append(item)
#     return result


def convert_dict(dct):
    """递归转换字典"""
    import numpy as np
    
    result = {}
    for key, value in dct.items():
        if isinstance(value, (np.integer, np.int32, np.int64)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, list):
            result[key] = convert_list(value)
        elif isinstance(value, dict):
            result[key] = convert_dict(value)
        else:
            result[key] = value
    return result


if __name__ == '__main__': 
    print("\n" + "="*70)
    print("集成版Flask API服务启动（完整版 + 调试）")
    print("="*70)
    print("\n💡 功能:")
    print("  ✓ 交互式求解")
    print("  ✓ 批量可视化生成（增强调试）")
    print("  ✓ 历史结果查看和下载")
    print("  ✓ 邻接表和链路分布预计算")
    print("\n📂 访问:  http://127.0.0.1:6601")
    print("\n" + "="*70 + "\n")
    
    # 确保必要的目录存在
    Path('temp').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=6601, debug=True, threaded=True)