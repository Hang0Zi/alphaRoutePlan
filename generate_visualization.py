"""
独立的可视化生成脚本
从保存的结果文件生成HTML可视化，无需重新运行求解器
"""

import sys
import os
import time
import config as config

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from visualization_generator import generate_html_from_files
from result_manager import get_latest_results, list_saved_results, print_results_summary
from run_solver import load_data_once, get_data


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print(f"独立可视化生成器")
    print(f"{'='*70}\n")
    
    # 显示可用的结果文件
    print_results_summary()
    
    # 加载路网图数据
    print("加载路网图数据...")
    load_data_once()
    G, _, _, _, _, _ = get_data()
    print(f"✓ 路网图加载完成:  {len(G.nodes())} 个节点\n")
    
    # 获取最新的结果文件
    reverse_file = None
    forward_file = None
    use_update=True
    # 生成可视化
    # 使用Unix时间戳
    timestamp = int(time.time())
    
    if use_update:
        for testname in config.TESTNAME:
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
    else:

        reverse_file = f'results/v6_4step/reverse_results_latest.json'
        forward_file = f'results/v6_4step/forward_results_latest.json'
        output_file = f'results/v6_4step/solver_visualization_{timestamp}.html'
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


if __name__ == "__main__":
    main()