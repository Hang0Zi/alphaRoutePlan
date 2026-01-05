"""
正向Label-Setting求解器（修复 α 敏感性）
给定出发时间和起终点，推导到达时间分布和α概率下最早到达时间的路径
"""

import numpy as np
import heapq
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════════
# 数据结构定义
# ════════════════════════════════════════════════════════════════


from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ForwardDiscreteDistribution:
    """
    正向离散分布类(到达时间分布)
    
    支持带权重的离散分布,用于更准确地表示到达时间的概率分布
    """
    values: np.ndarray  # 改为 numpy 数组
    L1: int
    weights: Optional[np.ndarray] = None  # ✅ 添加权重参数
    
    def __init__(self, values, L1, weights=None):
        """
        初始化正向离散分布
        
        Args:
            values: 离散值列表或数组
            L1: 离散化级别
            weights: 权重数组(可选),如果为None则使用均匀权重
        """
        # 转换为 numpy 数组
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        if len(values) != L1:
            raise ValueError(f"期望{L1}个值,实际得到{len(values)}个")
        
        self.L1 = L1
        
        # ✅ 处理权重
        if weights is None:
            # 默认均匀权重
            self.weights = np.ones(L1) / L1
        else:
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            
            if len(weights) != L1:
                raise ValueError(f"权重数量({len(weights)})与值数量({L1})不匹配")
            
            # 归一化权重
            if weights.sum() > 0:
                self.weights = weights / weights.sum()
            else:
                self.weights = np.ones(L1) / L1
        
        # 排序值和对应的权重
        sorted_indices = np.argsort(values)
        self.values = values[sorted_indices]
        self.weights = self.weights[sorted_indices]
    
    def get_quantile(self, alpha: float) -> float:
        """
        获取α分位数(使用权重的累积分布)
        
        Args:
            alpha: 分位数(0-1)
            
        Returns:
            对应的分位数值
        """
        if alpha <= 0:
            return float(self.values[0])
        if alpha >= 1:
            return float(self.values[-1])
        
        # ✅ 使用加权累积分布
        cumsum = np.cumsum(self.weights)
        
        # 线性插值
        quantile_value = np.interp(alpha, cumsum, self.values)
        
        return float(quantile_value)
    
    def get_mean(self) -> float:
        """计算加权平均值"""
        return float(np.sum(self.values * self.weights))
    
    def get_expected(self) -> float:
        """计算期望值(与 get_mean 相同)"""
        return self.get_mean()
    
    def get_variance(self) -> float:
        """计算加权方差"""
        mu = self.get_mean()
        return float(np.sum(self.weights * (self.values - mu) ** 2))
    
    def get_std(self) -> float:
        """计算加权标准差"""
        return float(np.sqrt(self.get_variance()))
    
    def get_median(self) -> float:
        """计算中位数(0.5分位数)"""
        return self.get_quantile(0.5)
    
    def to_dict(self) -> dict:
        """转换为字典格式(用于JSON序列化)"""
        return {
            'values':  self.values.tolist() if isinstance(self.values, np.ndarray) else list(self.values),
            'weights': self.weights.tolist() if isinstance(self.weights, np.ndarray) else list(self.weights),
            'L1': self.L1
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ForwardDiscreteDistribution':
        """从字典创建分布"""
        values = np.array(data['values'])
        L1 = data.get('L1', len(values))
        weights = np.array(data['weights']) if 'weights' in data else None
        return cls(values, L1, weights)
    
    def forward_convolve(self,
                        get_link_dist_func,
                        current: int,
                        successor: int,
                        time_intervals_per_day: int,
                        L2: int,
                        K:  int) -> 'ForwardDiscreteDistribution':
        """
        正向卷积: 从当前节点的出发时间分布推导到后继节点的到达时间分布
        
        核心思想:
        1.遍历所有可能的出发时间 t_dep(从self.values)
        2.对每个 t_dep,计算其对应的时间片 slot_dep
        3.从路段分布 D(current, successor, slot_dep) 采样L2个旅行时间
        4.计算 L1*L2 个到达时间: t_arr = t_dep + travel_time
        5.取前K个作为新的到达时间分布
        
        Args:
            get_link_dist_func: 获取链路分布的函数
            current: 当前节点
            successor: 后继节点
            time_intervals_per_day: 每天时间片数
            L2: 每个出发时间采样的旅行时间数
            K: 取前K个到达时间
            
        Returns:
            到达时间分布
        """
        
        # 获取可用时间片
        available_slots = self._get_available_slots(get_link_dist_func, current, successor)
        
        if not available_slots:
            raise ValueError(f"边({current}, {successor})没有链路分布数据")
        
        # ✅ 存储所有候选到达时间及其权重
        all_arrival_times = []
        all_arrival_weights = []
        
        # 遍历所有出发时间
        for i, t_dep in enumerate(self.values):
            # 该出发时间的权重
            dep_weight = self.weights[i]
            
            # 计算出发时间片
            slot_dep = int(t_dep / 10) % time_intervals_per_day
            
            # 获取该时间片的路段分布
            D_slot = get_link_dist_func(current, successor, slot_dep)
            
            if D_slot is None:
                # 如果精确时间片没有分布,尝试最近的时间片
                nearest_slot = self._find_nearest_slot(
                    slot_dep, available_slots, time_intervals_per_day
                )
                D_slot = get_link_dist_func(current, successor, nearest_slot)
            
            if D_slot is None:
                continue
            
            # 从路段分布采样L2个旅行时间
            sampled_travel_times = D_slot.sample_L2_times(t_dep, L2)
            
            # ✅ 计算对应的到达时间和权重
            for travel_time in sampled_travel_times: 
                t_arr = t_dep + travel_time
                all_arrival_times.append(t_arr)
                # 权重按出发时间权重和采样数平均分配
                all_arrival_weights.append(dep_weight / L2)
        
        if not all_arrival_times: 
            raise ValueError(f"正向卷积失败: 无有效到达时间")
        
        # ✅ 转换为 numpy 数组并排序
        all_arrival_times = np.array(all_arrival_times)
        all_arrival_weights = np.array(all_arrival_weights)
        
        # 按到达时间排序
        sorted_indices = np.argsort(all_arrival_times)
        all_arrival_times = all_arrival_times[sorted_indices]
        all_arrival_weights = all_arrival_weights[sorted_indices]
        
        # ✅ 降采样到K个值(保留权重信息)
        if len(all_arrival_times) > K:
            # 使用分位数方法选择代表性的K个点
            cumsum = np.cumsum(all_arrival_weights)
            cumsum = cumsum / cumsum[-1]  # 归一化
            
            # 选择K个均匀分布的分位数
            target_quantiles = np.linspace(1/(K+1), K/(K+1), K)
            
            # 通过插值找到对应的到达时间
            selected_times = np.interp(target_quantiles, cumsum, all_arrival_times)
            selected_weights = np.ones(K) / K  # 降采样后使用均匀权重
        else: 
            selected_times = all_arrival_times
            selected_weights = all_arrival_weights
        
        # ✅ 调整到L1个值
        if len(selected_times) < self.L1:
            # 插值增加到L1个
            cumsum = np.cumsum(selected_weights)
            cumsum = cumsum / cumsum[-1]
            target_quantiles = np.linspace(1/(self.L1+1), self.L1/(self.L1+1), self.L1)
            final_times = np.interp(target_quantiles, cumsum, selected_times)
            final_weights = np.ones(self.L1) / self.L1
        elif len(selected_times) > self.L1:
            # 降采样到L1个
            cumsum = np.cumsum(selected_weights)
            cumsum = cumsum / cumsum[-1]
            target_quantiles = np.linspace(1/(self.L1+1), self.L1/(self.L1+1), self.L1)
            final_times = np.interp(target_quantiles, cumsum, selected_times)
            final_weights = np.ones(self.L1) / self.L1
        else:
            # 大小刚好
            final_times = selected_times
            final_weights = selected_weights
        
        # ✅ 确保排序
        sorted_indices = np.argsort(final_times)
        final_times = final_times[sorted_indices]
        final_weights = final_weights[sorted_indices]
        
        # ✅ 创建新分布(传入权重)
        return ForwardDiscreteDistribution(final_times, self.L1, final_weights)
    
    def _find_nearest_slot(self, target_slot: int, available_slots: List[int],
                          time_intervals_per_day:  int) -> int:
        """找到最近的时间片"""
        min_dist = float('inf')
        best_slot = available_slots[0]
        
        for slot in available_slots: 
            dist = abs(slot - target_slot)
            cyclic_dist = min(dist, time_intervals_per_day - dist)
            
            if cyclic_dist < min_dist:
                min_dist = cyclic_dist
                best_slot = slot
        
        return best_slot
    
    def _get_available_slots(self, get_link_dist_func, u: int, v: int) -> List[int]:
        """获取可用时间片(带缓存)"""
        if not hasattr(ForwardDiscreteDistribution, '_slot_cache'):
            ForwardDiscreteDistribution._slot_cache = {}
        
        cache_key = (u, v)
        if cache_key in ForwardDiscreteDistribution._slot_cache:
            return ForwardDiscreteDistribution._slot_cache[cache_key]
        
        available = []
        try:
            link_distributions = get_link_dist_func.__self__.link_distributions
            for (link_u, link_v, slot) in link_distributions.keys():
                if link_u == u and link_v == v:
                    available.append(slot)
        except AttributeError:
            raise ValueError("无法访问链路分布数据")
        
        result = sorted(set(available))
        ForwardDiscreteDistribution._slot_cache[cache_key] = result
        return result
    
    def __len__(self) -> int:
        """返回分布大小"""
        return self.L1
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ForwardDiscreteDistribution(L1={self.L1}, "
                f"mean={self.get_mean():.2f}, "
                f"std={self.get_std():.2f}, "
                f"range=[{self.values[0]:.1f}, {self.values[-1]:.1f}])")

@dataclass
class LinkTimeDistribution:
    """路段旅行时间分布"""
    time_prob: Dict[int, float]
    times: List[int]
    cdf: List[float]
    time_slot: int
    
    def __init__(self, time_prob_dict: Dict[int, float], time_slot: int = None):
        if not time_prob_dict:
            raise ValueError("链路分布不能为空")
        
        total_prob = sum(time_prob_dict.values())
        self.time_prob = {t: p/total_prob for t, p in time_prob_dict.items()}
        self.time_slot = time_slot
        
        sorted_times = sorted(self.time_prob.keys())
        self.times = sorted_times
        
        cumulative = 0.0
        self.cdf = []
        for t in sorted_times:
            cumulative += self.time_prob[t]
            self.cdf.append(cumulative)
    
    def sample_L2_times(self, reference_time: int, L2: int) -> List[int]:
        """采样L2个旅行时间（逆CDF方法）"""
        samples = []
        for i in range(1, L2 + 1):
            quantile = i / (L2 + 1)
            sample = self._inverse_cdf(quantile)
            samples.append(sample)
        return sorted(samples)
    
    def _inverse_cdf(self, quantile: float) -> int:
        """逆CDF（线性插值）"""
        if quantile <= 0:
            return self.times[0]
        if quantile >= 1:
            return self.times[-1]
        
        for i, cdf_val in enumerate(self.cdf):
            if cdf_val >= quantile:
                if i == 0:
                    return self.times[0]
                
                lower_cdf = self.cdf[i-1] if i > 0 else 0
                upper_cdf = cdf_val
                lower_time = self.times[i-1] if i > 0 else self.times[0]
                upper_time = self.times[i]
                
                if upper_cdf > lower_cdf:
                    weight = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
                else:
                    weight = 0.5
                
                return int(round(lower_time + weight * (upper_time - lower_time)))
        
        return self.times[-1]


from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ForwardLabel:
    """正向标签类"""
    node_id: int
    distribution:  'ForwardDiscreteDistribution'
    path: List[int]
    cost: float
    alpha: float = 0.95  # 可靠性参数
    
    # ✅ 添加缓存字段
    mean_cache: float = field(default=0.0, init=False, repr=False)
    variance_cache: float = field(default=0.0, init=False, repr=False)
    std_cache: float = field(default=0.0, init=False, repr=False)
    
    def __post_init__(self):
        """后初始化: 预计算统计量"""
        # ✅ 预计算并缓存统计量
        self.mean_cache = self.distribution.get_mean()
        self.variance_cache = self.distribution.get_variance()
        self.std_cache = self.distribution.get_std()
    
    def __lt__(self, other:  'ForwardLabel') -> bool:
        """优先队列排序: cost越小越优"""
        return self.cost < other.cost
    
    # ✅ 添加属性访问器
    @property
    def expected_value(self) -> float:
        """期望值(均值)"""
        return self.mean_cache
    
    @property
    def std_value(self) -> float:
        """标准差"""
        return self.std_cache
    
    @property
    def variance_value(self) -> float:
        """方差"""
        return self.variance_cache
    
    def get_quantile(self, alpha: float) -> float:
        """获取α分位数"""
        return self.distribution.get_quantile(alpha)
    
    def dominates_weak(self, other: 'ForwardLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """
        支配规则(正向: 越小越好)
        
        支配条件(按优先级):
        1.α分位数严格更小 → 支配
        2.α分位数相等 + 期望值更小 → 支配
        3.α分位数和期望值都相等 + 方差更小 → 支配
        
        Args:
            other: 另一个标签
            alpha: 可靠性参数
            epsilon: 数值容差
            
        Returns:
            bool:  是否支配
        """
        # 必须在同一节点
        if self.node_id != other.node_id:
            return False
        
        # 主目标: α分位数(越小越好)
        Q_self = self.distribution.get_quantile(alpha)
        Q_other = other.distribution.get_quantile(alpha)
        
        # 策略1:主目标严格更优
        if Q_self < Q_other - epsilon:
            return True
        
        # 策略2:主目标相等,比较次要目标
        if abs(Q_self - Q_other) <= epsilon:
            mu_self = self.expected_value
            mu_other = other.expected_value
            
            # 期望值更优
            if mu_self < mu_other - epsilon:
                return True
            
            # 期望值也相等,比较方差
            if abs(mu_self - mu_other) <= epsilon:
                sigma2_self = self.variance_value
                sigma2_other = other.variance_value
                
                # 方差更小(更稳定)
                if sigma2_self < sigma2_other - epsilon:
                    return True
        
        return False
    
    def dominates(self, other: 'ForwardLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """统一接口"""
        return self.dominates_weak(other, alpha, epsilon)
    
    def __eq__(self, other: object) -> bool:
        """相等性判断"""
        if not isinstance(other, ForwardLabel):
            return False
        return (self.node_id == other.node_id and 
                abs(self.cost - other.cost) < 1e-9)
    
    def __hash__(self) -> int:
        """哈希值"""
        return hash((self.node_id, round(self.cost, 6)))
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ForwardLabel(node={self.node_id}, "
                f"Q={self.cost:.2f}, "
                f"E={self.expected_value:.2f}, "
                f"σ={self.std_value:.2f})")


# ════════════════════════════════════════════════════════════════
# 正向求解器主类
# ════════════════════════════════════════════════════════════════

class ForwardLabelSettingSolver:
    """正向Label-Setting求解器"""
    
    def __init__(self, G, sparse_data, node_to_index, scenario_dates,
                 scenario_probs, time_intervals_per_day,
                 L1:  int = 50, L2: int = 10, K: int = 100,
                 verbose: bool = False,
                 max_labels_per_node: int = 20):
        """初始化"""
        self.G = G
        self.sparse_data = sparse_data
        self.node_to_index = node_to_index
        self.index_to_node = {v:  k for k, v in node_to_index.items()}
        self.scenario_dates = scenario_dates
        self.scenario_probs = scenario_probs
        self.time_intervals_per_day = time_intervals_per_day
        self.n_scenarios = len(scenario_dates)
        
        self.L1 = L1
        self.L2 = L2
        self.K = K
        self.verbose = verbose
        self.max_labels_per_node = max_labels_per_node
        
        print(f"\n{'='*70}")
        print(f"初始化正向Label-Setting求解器")
        print(f"{'='*70}")
        print(f"  算法:  正向Label-Setting (α敏感性修复版)")
        print(f"  问题:  给定出发时间，求解到达时间分布")
        print(f"  参数: L1={L1}, L2={L2}, K={K}")
        print(f"  详细输出: {'开启' if verbose else '关闭'}")
        
        # 构建邻接表
        self.adj_list = defaultdict(list)
        self._build_adjacency_lists()
        
        # 预计算链路分布
        self.link_distributions = {}
        self._precompute_link_distributions()
        
        # 统计信息
        self.stats = defaultdict(int)
        
        print(f"\n✓ 初始化完成")
        print(f"{'='*70}\n")
    
    def _build_adjacency_lists(self):
        """构建邻接表"""
        print(f"  [1/2] 构建邻接表...")
        start_time = time.time()
        
        edges_set = set()
        for (scenario_idx, time_idx, from_idx, to_idx) in self.sparse_data.keys():
            if scenario_idx < self.n_scenarios:
                from_node = self.index_to_node[from_idx]
                to_node = self.index_to_node[to_idx]
                edges_set.add((from_node, to_node))
        
        for from_node, to_node in edges_set:
            self.adj_list[from_node].append(to_node)
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {len(edges_set):,} 条边")
    
    def _precompute_link_distributions(self):
        """预计算链路分布"""
        print(f"  [2/2] 预计算链路分布...")
        start_time = time.time()
        
        link_time_data = defaultdict(list)
        
        for (scenario_idx, time_idx, from_idx, to_idx), travel_time_minutes in self.sparse_data.items():
            if scenario_idx >= self.n_scenarios:
                continue
            
            from_node = self.index_to_node[from_idx]
            to_node = self.index_to_node[to_idx]
            travel_time_01min = int(travel_time_minutes * 10)
            
            link_time_data[(from_node, to_node, time_idx)].append(travel_time_01min)
        
        distribution_count = 0
        for (u, v, t), times in link_time_data.items():
            time_counts = defaultdict(int)
            for time_val in times:
                time_counts[time_val] += 1
            
            total = len(times)
            time_prob = {time_val: count/total for time_val, count in time_counts.items()}
            
            try:
                self.link_distributions[(u, v, t)] = LinkTimeDistribution(time_prob, time_slot=t)
                distribution_count += 1
            except ValueError:
                continue
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {distribution_count:,} 个分布")
    
    def _get_link_distribution_at_slot(self, u: int, v: int, slot: int) -> Optional[LinkTimeDistribution]:
        """获取指定出发时间片的链路分布"""
        if (u, v, slot) in self.link_distributions:
            return self.link_distributions[(u, v, slot)]
        
        # 容差匹配
        tolerance = 5
        candidates = []
        
        for (link_u, link_v, link_t) in self.link_distributions.keys():
            if link_u == u and link_v == v:
                diff = abs(link_t - slot)
                cyclic_diff = min(diff, self.time_intervals_per_day - diff)
                
                if cyclic_diff <= tolerance:  
                    candidates.append((link_t, cyclic_diff))
        
        if candidates:
            best_slot = min(candidates, key=lambda x: x[1])[0]
            return self.link_distributions[(u, v, best_slot)]
        
        return None
    
    def solve_k_paths(self, origin:  int, destination: int, departure_time: int,
                    alpha: float, K: int = 10, max_labels: int = 100000,
                    print_interval: int = 100) -> Dict:
        """
        正向K-Paths求解：给定出发时间，找到K条候选路径，选出α分位数到达时间最早的路径
        
        Args:  
            origin:  起点
            destination: 终点
            departure_time:  出发时间（0.1分钟单位）
            alpha: 可靠性参数
            K: 候选路径数量
            max_labels: 最大标签数
            print_interval: 打印间隔
        
        Returns:
            包含K条候选路径和最优路径的结果字典
        """
        
        print(f"\n{'='*70}")
        print(f"正向Label-Setting求解（K-Paths版本）")
        print(f"{'='*70}")
        print(f"  起点: {origin}")
        print(f"  终点: {destination}")
        print(f"  出发时间: {departure_time/10:.1f}分 ({self._time_to_string(departure_time)})")
        print(f"  可靠性:  α={alpha*100:.1f}%")
        print(f"  候选路径数: K={K}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤1：搜索K条候选路径
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 初始化
        open_labels = []
        node_labels = defaultdict(list)
        destination_candidates = []  # ✅ 存储所有到达终点的候选标签
        
        # ✅ 初始标签
        init_dist = ForwardDiscreteDistribution(
            values=np.array([departure_time] * self.L1),
            L1=self.L1,
            weights=np.ones(self.L1) / self.L1
        )
        init_label = ForwardLabel(origin, init_dist, [origin], departure_time)
        
        heapq.heappush(open_labels, init_label)
        node_labels[origin].append(init_label)
        self.stats = defaultdict(int)
        self.stats['labels_generated'] = 1
        
        print(f"开始搜索 K={K} 条候选路径...\n")
        
        iteration = 0
        
        # 主循环：找到K条到达终点的路径
        while open_labels and self.stats['labels_generated'] < max_labels: 
            iteration += 1
            current_label = heapq.heappop(open_labels)
            
            if self.verbose and (iteration % print_interval == 0 or iteration <= 5):
                print(f"  迭代#{iteration}:  节点{current_label.node_id}, "
                    f"cost={current_label.cost/10:.1f}分, "
                    f"候选数={len(destination_candidates)}")
            
            # ✅ 到达终点：保存为候选路径
            if current_label.node_id == destination:
                earliest_arrival = current_label.distribution.get_quantile(alpha)
                expected_arrival = current_label.expected_value
                
                # ✅ 获取路径坐标（用于地图显示）
                path_coords = self._get_path_coordinates(current_label.path)
                
                # 保存候选路径信息
                candidate_info = {
                    'iteration': iteration,
                    'path':  current_label.path,
                    'path_coords': path_coords,  # ✅ 添加地图坐标
                    'distribution': {
                        'values': current_label.distribution.values.tolist(),
                        'weights': current_label.distribution.weights.tolist(),
                        'L1': current_label.distribution.L1
                    },
                    'earliest_arrival': earliest_arrival,
                    'expected_arrival': expected_arrival,
                    'median_arrival': current_label.distribution.get_median(),
                    'std_arrival':  current_label.std_value,
                    'variance': current_label.variance_value,
                    'travel_time':  earliest_arrival - departure_time,
                    'label': current_label,
                    'alpha': alpha,
                    'rank': None,
                    'is_best': False  # ✅ 初始化为 False
                }
                
                destination_candidates.append(candidate_info)
                
                print(f"  🎯 找到候选路径#{len(destination_candidates)} 迭代#{iteration}, "
                    f"Q_α={earliest_arrival/10:.1f}分, "
                    f"Mean={expected_arrival/10:.1f}分, "
                    f"路径长度={len(current_label.path)}")
                
                # ✅ 找到K条路径后继续搜索（确保探索充分）
                if len(destination_candidates) >= K:
                    # 继续搜索，但有上限
                    if len(destination_candidates) >= K * 2:  # 找到2K条后停止
                        print(f"\n  ✓ 已找到 {len(destination_candidates)} 条候选路径，停止搜索\n")
                        break
                
                # 继续搜索其他路径
                continue
            
            # 支配性检查（较宽松，保留多样性）
            if self._is_dominated(current_label, node_labels[current_label.node_id], alpha):
                self.stats['labels_dominated'] += 1
                continue
            
            self.stats['labels_extended'] += 1
            
            # 正向扩展
            if current_label.node_id not in self.adj_list:
                continue
            
            for successor in self.adj_list[current_label.node_id]: 
                if successor in current_label.path:
                    continue
                
                # 正向卷积
                try:
                    def get_link_dist(u, v, slot):
                        return self._get_link_distribution_at_slot(u, v, slot)
                    
                    get_link_dist.__self__ = self
                    
                    new_dist = current_label.distribution.forward_convolve(
                        get_link_dist_func=get_link_dist,
                        current=current_label.node_id,
                        successor=successor,
                        time_intervals_per_day=self.time_intervals_per_day,
                        L2=self.L2,
                        K=self.K
                    )
                    
                    self.stats['convolutions'] += 1
                    
                except Exception as e:
                    if self.verbose and iteration <= 10:
                        print(f"      ⚠ 卷积失败: {e}")
                    continue
                
                new_cost = new_dist.get_quantile(alpha)
                new_label = ForwardLabel(
                    successor, 
                    new_dist,
                    current_label.path + [successor],
                    new_cost
                )
                
                self.stats['labels_generated'] += 1
                
                # 支配性剪枝
                if self._is_dominated(new_label, node_labels[successor], alpha):
                    self.stats['labels_dominated'] += 1
                    continue
                
                # 反向剪枝
                original_count = len(node_labels[successor])
                node_labels[successor] = [
                    old for old in node_labels[successor]
                    if not new_label.dominates_weak(old, alpha)
                ]
                self.stats['labels_dominated'] += (original_count - len(node_labels[successor]))
                
                node_labels[successor].append(new_label)
                node_labels[successor] = self._prune_labels(node_labels[successor], alpha)
                heapq.heappush(open_labels, new_label)
            
            # 进度显示
            if not self.verbose and iteration % 100 == 0:
                print(f"  进度: 迭代#{iteration}, 生成{self.stats['labels_generated']: ,}, "
                    f"候选{len(destination_candidates)}, "
                    f"剪枝{self.stats['labels_dominated']: ,}", end='\r')
        
        total_time = time.time() - start_time
        
        print(f"\n\n{'='*70}")
        print(f"搜索完成")
        print(f"{'='*70}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2：对K条候选路径排序
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if not destination_candidates:
            print(f"✗ 未找到到达终点的路径")
            return {
                'success': False,
                'total_time': total_time,
                'iterations': iteration,
                'stats': dict(self.stats),
                'num_candidates': 0
            }
        
        print(f"\n找到 {len(destination_candidates)} 条候选路径")
        print(f"开始排序和比较...\n")
        
        # ✅ 多目标排序：主要Q_α（越小越好），次要Mean，再次要Var
        def rank_score(candidate):
            return (
                candidate['earliest_arrival'],      # 主目标：Q_α（越小越好）
                candidate['expected_arrival'],      # 次要：均值（越小越好）
                candidate['variance']               # 再次要：方差（越小越好）
            )
        
        # 排序：从最优到最差
        sorted_candidates = sorted(destination_candidates, key=rank_score)
        
        # ✅ 设置排名和最优标记
        for rank, candidate in enumerate(sorted_candidates, 1):
            candidate['rank'] = rank
            candidate['is_best'] = (rank == 1)  # 排名第1的标记为最优
        
        # 取前K条
        top_k_candidates = sorted_candidates[:K]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3：输出结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        best_candidate = top_k_candidates[0]
        
        print(f"{'='*70}")
        print(f"Top-{len(top_k_candidates)} 候选路径对比")
        print(f"{'='*70}\n")
        
        print(f"{'排名':<6} {'Q_α(分)':<15} {'Mean(分)':<15} {'Std(分)':<12} {'旅行时间(分)':<15} {'路径长度': <10}")
        print(f"{'-'*70}")
        
        for candidate in top_k_candidates: 
            print(f"{candidate['rank']:<6} "
                f"{candidate['earliest_arrival']/10:<15.1f} "
                f"{candidate['expected_arrival']/10:<15.1f} "
                f"{candidate['std_arrival']/10:<12.2f} "
                f"{candidate['travel_time']/10:<15.1f} "
                f"{len(candidate['path']):<10}")
        
        print(f"\n{'='*70}")
        print(f"✓ 最优路径（排名#1）")
        print(f"{'='*70}")
        print(f"\n  路径:  {self._format_path(best_candidate['path'])}")
        print(f"  长度: {len(best_candidate['path'])} 个节点")
        print(f"\n  时间:")
        print(f"    出发时间: {self._time_to_string(departure_time)}")
        print(f"    最早到达 (α={alpha}): {self._time_to_string(best_candidate['earliest_arrival'])}")
        print(f"    期望到达:  {self._time_to_string(best_candidate['expected_arrival'])}")
        print(f"    旅行时间: {best_candidate['travel_time']/10:.1f}分")
        print(f"    标准差: {best_candidate['std_arrival']/10:.2f}分")
        print(f"\n  性能:")
        print(f"    总耗时: {total_time:.2f}秒")
        print(f"    迭代次数: {iteration}")
        print(f"    候选路径数: {len(destination_candidates)}")
        print(f"    生成标签:  {self.stats['labels_generated']: ,}")
        print(f"    剪枝率: {self.stats['labels_dominated']/self.stats['labels_generated']*100:.1f}%")
        print(f"{'='*70}\n")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 构建返回结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        result = {
            'success': True,
            # 最优路径信息
            'path': best_candidate['path'],
            'path_coords': best_candidate['path_coords'],  # ✅ 添加坐标
            'earliest_arrival_time': best_candidate['earliest_arrival'],
            'expected_arrival_time': best_candidate['expected_arrival'],
            'median_arrival_time': best_candidate['median_arrival'],
            'std_arrival_time': best_candidate['std_arrival'],
            'travel_time': best_candidate['travel_time'],
            'distribution': best_candidate['distribution'],  # ✅ 已经是字典格式
            'departure_time': departure_time,
            
            # Top-K候选路径
            'top_k_candidates':  top_k_candidates,
            'num_candidates': len(destination_candidates),
            'all_candidates': sorted_candidates,
            
            # 元信息
            'total_time':  total_time,
            'iterations': iteration,
            'alpha': alpha,
            'K': K,
            'origin': origin,
            'destination':  destination,
            'stats': dict(self.stats)
        }
        
        return result


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 辅助方法：获取路径坐标
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_path_coordinates(self, path):
        """
        获取路径的地理坐标
        
        Args:
            path: 节点列表
            
        Returns: 
            坐标列表 [[lat1, lon1], [lat2, lon2], ...]
        """
        coords = []
        for node in path:
            if node in self.G.nodes:
                node_data = self.G.nodes[node]
                if 'y' in node_data and 'x' in node_data:
                    # Leaflet 使用 [lat, lon] 格式
                    coords.append([node_data['y'], node_data['x']])
        
        return coords


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 保留原有的 solve() 方法（单路径版本）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def solve(self, origin:  int, destination: int, departure_time: int,
            alpha: float, max_labels: int = 100000,
            print_interval: int = 100) -> Dict: 
        """
        正向求解：给定出发时间，求解到达时间分布和路径（单路径版本）
        
        如果需要K条候选路径，请使用 solve_k_paths() 方法
        
        Args:
            origin: 起点
            destination: 终点
            departure_time: 出发时间（0.1分钟单位）
            alpha: 可靠性参数
            max_labels: 最大标签数
            print_interval: 打印间隔
        
        Returns:
            包含到达时间分布和路径的结果字典
        """
        
        # 调用 K-Paths 版本，K=1（只找一条最优路径）
        result = self.solve_k_paths(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            alpha=alpha,
            K=1,  # 只找1条路径
            max_labels=max_labels,
            print_interval=print_interval
        )
        
        # 简化返回结果（移除K-Paths相关字段）
        if result['success']:
            result.pop('top_k_candidates', None)
            result.pop('all_candidates', None)
            result.pop('num_candidates', None)
            result.pop('K', None)
        
        return result

    def _is_dominated(self, label: ForwardLabel, existing_labels: List[ForwardLabel],
                     alpha: float) -> bool:
        """支配性检查"""
        if len(existing_labels) < self.max_labels_per_node:
            domination_count = 0
            for existing in existing_labels:
                if existing.dominates_weak(label, alpha):
                    domination_count += 1
            return domination_count >= 2
        
        for existing in existing_labels:
            if existing.dominates_weak(label, alpha):
                return True
        
        return False
    
    def _prune_labels(self, labels: List[ForwardLabel], alpha: float) -> List[ForwardLabel]:
        """标签剪枝"""
        if len(labels) <= self.max_labels_per_node:
            return labels
        
        def label_score(label):
            q = label.distribution.get_quantile(alpha)
            return (q, label.mean_cache, label.variance_cache)
        
        sorted_labels = sorted(labels, key=label_score)
        return sorted_labels[:self.max_labels_per_node]
    
    def _format_path(self, path: List[int]) -> str:
        if len(path) <= 10:
            return ' → '.join(map(str, path))
        return f"{' → '.join(map(str, path[: 5]))} → ...→ {' → '.join(map(str, path[-3:]))}"
    
    def _time_to_string(self, time_01min):
        """时间格式转换"""
        total_minutes = time_01min / 10
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        return f"{hours:02d}:{minutes:02d}"