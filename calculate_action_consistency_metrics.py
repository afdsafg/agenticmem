"""
计算动作一致性指标的脚本

使用方法:
python calculate_action_consistency_metrics.py --result_dir /path/to/results

注意：此脚本需要修改后的logger保存每步的详细信息
"""

import os
import json
import pickle
import numpy as np
import argparse
from typing import Dict, List, Tuple

# 设置matplotlib后端，避免GUI相关问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理numpy数据类型
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def calculate_direction_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三个连续点之间的方向变化角度
    
    Args:
        p1, p2, p3: 连续三个位置点 (x, y)
    
    Returns:
        方向变化角度（度数，0-180）
    """
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p2
    
    # 处理零向量的情况
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # 计算夹角（弧度）
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    angle_rad = np.arccos(cos_angle)
    
    # 转换为度数
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_direction_changes(trajectory: np.ndarray) -> List[float]:
    """
    计算整个轨迹的方向变化角度列表
    
    Args:
        trajectory: (N, 2) 数组，N个位置点
    
    Returns:
        长度为N-2的角度列表
    """
    if len(trajectory) < 3:
        return []
    
    angles = []
    for i in range(len(trajectory) - 2):
        angle = calculate_direction_angle(
            trajectory[i], 
            trajectory[i+1], 
            trajectory[i+2]
        )
        angles.append(angle)
    
    return angles


def calculate_stability_ratio(angles: List[float], threshold_deg: float = 30.0) -> float:
    """
    计算方向稳定性比例（角度变化小于阈值的比例）
    
    Args:
        angles: 方向变化角度列表
        threshold_deg: 角度阈值（度）
    
    Returns:
        稳定性比例 [0, 1]
    """
    if len(angles) == 0:
        return 0.0
    
    stable_count = sum(1 for angle in angles if angle <= threshold_deg)
    return stable_count / len(angles)


def calculate_frontier_reselection_rate(frontier_choices: List[str]) -> Tuple[float, Dict]:
    """
    计算frontier重复选择率
    
    Args:
        frontier_choices: 每步选择的frontier ID列表
    
    Returns:
        (重复选择率, 选择次数统计字典)
    """
    if len(frontier_choices) == 0:
        return 0.0, {}
    
    # 统计每个frontier被选择的次数
    choice_counts = {}
    for choice in frontier_choices:
        choice_counts[choice] = choice_counts.get(choice, 0) + 1
    
    # 计算被重复选择的frontier比例
    reselected = sum(1 for count in choice_counts.values() if count > 1)
    reselection_rate = reselected / len(choice_counts) if len(choice_counts) > 0 else 0.0
    
    return reselection_rate, choice_counts


def calculate_position_revisit_rate(trajectory: np.ndarray, distance_threshold: float = 0.5) -> Tuple[float, List[int]]:
    """
    计算位置重访率（检测agent是否重复访问相同或相近的位置）
    
    Args:
        trajectory: (N, 2) 数组，N个位置点
        distance_threshold: 距离阈值（米），默认0.5m
    
    Returns:
        (重访率, 每步的重访次数列表)
    """
    if len(trajectory) < 2:
        return 0.0, []
    
    revisit_counts = []
    
    # 对于每个位置（从第2个开始），检查它与之前所有位置的距离
    for i in range(1, len(trajectory)):
        current_pos = trajectory[i]
        previous_positions = trajectory[:i]
        
        # 计算当前位置与所有之前位置的距离
        distances = np.linalg.norm(previous_positions - current_pos, axis=1)
        
        # 统计距离小于阈值的位置数量（即重访次数）
        revisit_count = np.sum(distances <= distance_threshold)
        revisit_counts.append(revisit_count)
    
    # 计算重访率：有重访的步骤占比
    revisit_rate = np.sum(np.array(revisit_counts) > 0) / len(revisit_counts) if len(revisit_counts) > 0 else 0.0
    
    return revisit_rate, revisit_counts


def calculate_position_clustering(trajectory: np.ndarray, distance_threshold: float = 0.5) -> Dict:
    """
    计算位置聚类统计（分析agent在多少个不同区域活动）
    
    Args:
        trajectory: (N, 2) 数组，N个位置点
        distance_threshold: 距离阈值（米）
    
    Returns:
        聚类统计字典
    """
    if len(trajectory) < 2:
        return {"n_clusters": 1, "mean_cluster_size": 1, "max_cluster_size": 1}
    
    # 简单的聚类：如果两个点距离小于阈值，认为在同一区域
    visited_clusters = []  # 每个元素是一个聚类中心
    cluster_sizes = []  # 每个聚类的大小
    
    for pos in trajectory:
        # 检查是否属于已有聚类
        found_cluster = False
        for i, cluster_center in enumerate(visited_clusters):
            if np.linalg.norm(pos - cluster_center) <= distance_threshold:
                cluster_sizes[i] += 1
                found_cluster = True
                break
        
        # 如果不属于任何已有聚类，创建新聚类
        if not found_cluster:
            visited_clusters.append(pos)
            cluster_sizes.append(1)
    
    return {
        "n_clusters": len(visited_clusters),
        "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "cluster_distribution": cluster_sizes
    }


def analyze_single_episode(episode_data: Dict, distance_threshold: float = 0.5) -> Dict:
    """
    分析单个episode的动作一致性指标
    
    Args:
        episode_data: 包含轨迹和选择信息的字典
            - "trajectory": (N, 2) 位置序列
            - "frontier_choices": frontier选择序列（可选）
        distance_threshold: 位置重访的距离阈值（米）
    
    Returns:
        指标字典
    """
    trajectory = np.array(episode_data["trajectory"])
    
    # 指标1: 方向变化角度
    angles = calculate_direction_changes(trajectory)
    
    metrics = {
        "n_steps": len(trajectory),
        "direction_changes": angles,
        "mean_angle_change": np.mean(angles) if len(angles) > 0 else 0.0,
        "std_angle_change": np.std(angles) if len(angles) > 0 else 0.0,
        "max_angle_change": np.max(angles) if len(angles) > 0 else 0.0,
    }
    
    # 指标2: 方向稳定性（多个阈值）
    for threshold in [30, 45, 60, 90]:
        stability = calculate_stability_ratio(angles, threshold)
        metrics[f"stability_ratio_{threshold}deg"] = stability
    
    # 指标3: frontier重复选择率（如果有数据）
    if "frontier_choices" in episode_data and episode_data["frontier_choices"]:
        reselection_rate, choice_counts = calculate_frontier_reselection_rate(
            episode_data["frontier_choices"]
        )
        metrics["frontier_reselection_rate"] = reselection_rate
        metrics["unique_frontiers_selected"] = len(choice_counts)
        metrics["total_frontier_selections"] = len(episode_data["frontier_choices"])
    
    # 指标4: 位置重访率（新增）
    revisit_rate, revisit_counts = calculate_position_revisit_rate(trajectory, distance_threshold)
    metrics["position_revisit_rate"] = revisit_rate
    metrics["mean_revisit_count"] = np.mean(revisit_counts) if len(revisit_counts) > 0 else 0.0
    metrics["max_revisit_count"] = max(revisit_counts) if len(revisit_counts) > 0 else 0
    metrics["total_revisits"] = sum(revisit_counts) if len(revisit_counts) > 0 else 0
    
    # 指标5: 位置聚类统计（分析探索区域的分布）
    clustering_stats = calculate_position_clustering(trajectory, distance_threshold)
    metrics["n_position_clusters"] = clustering_stats["n_clusters"]
    metrics["mean_cluster_size"] = clustering_stats["mean_cluster_size"]
    metrics["max_cluster_size"] = clustering_stats["max_cluster_size"]
    
    return metrics


def load_episode_data(episode_dir: str) -> Dict:
    """
    从episode目录加载数据
    
    需要的文件格式：
    - trajectory.json: {"positions": [[x1, y1], [x2, y2], ...]}
    - frontier_choices.json: {"choices": ["frontier_0", "frontier_1", ...]}
    """
    trajectory_path = os.path.join(episode_dir, "trajectory.json")
    frontier_path = os.path.join(episode_dir, "frontier_choices.json")
    
    episode_data = {}
    
    # 加载轨迹数据
    if os.path.exists(trajectory_path):
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
            episode_data["trajectory"] = data["positions"]
    else:
        print(f"Warning: {trajectory_path} not found")
        return None
    
    # 加载frontier选择数据（可选）
    if os.path.exists(frontier_path):
        with open(frontier_path, 'r') as f:
            data = json.load(f)
            episode_data["frontier_choices"] = data.get("choices", [])
    
    return episode_data


def analyze_all_episodes(result_dir: str, distance_threshold: float = 0.5) -> Dict:
    """
    分析所有episode并汇总统计
    """
    exp_dir = os.path.join(result_dir, "exp_eval_aeqa")
    
    if not os.path.exists(exp_dir):
        print(f"Error: {exp_dir} not found")
        return None
    
    # 获取所有episode目录
    episode_dirs = [
        os.path.join(exp_dir, d) 
        for d in os.listdir(exp_dir) 
        if os.path.isdir(os.path.join(exp_dir, d))
    ]
    
    all_metrics = []
    failed_episodes = []
    
    for episode_dir in episode_dirs:
        episode_id = os.path.basename(episode_dir)
        episode_data = load_episode_data(episode_dir)
        
        if episode_data is None:
            failed_episodes.append(episode_id)
            continue
        
        metrics = analyze_single_episode(episode_data, distance_threshold)
        metrics["episode_id"] = episode_id
        all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print("Error: No valid episode data found")
        print("请确保已经修改代码保存了trajectory.json和frontier_choices.json")
        return None
    
    # 汇总统计
    summary = {
        "n_episodes": len(all_metrics),
        "n_failed": len(failed_episodes),
        "mean_angle_change": np.mean([m["mean_angle_change"] for m in all_metrics]),
        "std_angle_change": np.std([m["mean_angle_change"] for m in all_metrics]),
    }
    
    # 各阈值下的平均稳定性
    for threshold in [30, 45, 60, 90]:
        key = f"stability_ratio_{threshold}deg"
        values = [m[key] for m in all_metrics]
        summary[f"mean_{key}"] = np.mean(values)
        summary[f"std_{key}"] = np.std(values)
    
    # 位置重访率统计（新增）
    summary["mean_position_revisit_rate"] = np.mean([m["position_revisit_rate"] for m in all_metrics])
    summary["std_position_revisit_rate"] = np.std([m["position_revisit_rate"] for m in all_metrics])
    summary["mean_revisit_count"] = np.mean([m["mean_revisit_count"] for m in all_metrics])
    summary["mean_total_revisits"] = np.mean([m["total_revisits"] for m in all_metrics])
    
    # 位置聚类统计
    summary["mean_n_position_clusters"] = np.mean([m["n_position_clusters"] for m in all_metrics])
    summary["mean_cluster_size"] = np.mean([m["mean_cluster_size"] for m in all_metrics])
    
    # frontier相关统计（如果有数据）
    frontier_metrics = [m for m in all_metrics if "frontier_reselection_rate" in m]
    if len(frontier_metrics) > 0:
        summary["mean_frontier_reselection_rate"] = np.mean(
            [m["frontier_reselection_rate"] for m in frontier_metrics]
        )
        summary["n_episodes_with_frontier_data"] = len(frontier_metrics)
        summary["frontier_data_coverage"] = len(frontier_metrics) / len(all_metrics)
    
    return {
        "summary": summary,
        "per_episode": all_metrics,
        "failed_episodes": failed_episodes
    }


def visualize_results(results: Dict, output_dir: str):
    """
    可视化结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    per_episode = results["per_episode"]
    
    # 1. 方向变化角度分布
    all_angles = []
    for episode in per_episode:
        all_angles.extend(episode["direction_changes"])
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_angles, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Direction Change Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Direction Changes')
    plt.axvline(np.mean(all_angles), color='r', linestyle='--', 
                label=f'Mean: {np.mean(all_angles):.2f}°')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'angle_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 每个episode的平均角度变化
    mean_angles = [m["mean_angle_change"] for m in per_episode]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_angles)), mean_angles, alpha=0.7)
    plt.xlabel('Episode Index')
    plt.ylabel('Mean Direction Change (degrees)')
    plt.title('Mean Direction Change per Episode')
    plt.axhline(np.mean(mean_angles), color='r', linestyle='--', 
                label=f'Overall Mean: {np.mean(mean_angles):.2f}°')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'per_episode_angles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 稳定性比例对比（不同阈值）
    thresholds = [30, 45, 60, 90]
    stability_means = [
        results["summary"][f"mean_stability_ratio_{t}deg"] for t in thresholds
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, stability_means, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Angle Threshold (degrees)')
    plt.ylabel('Stability Ratio')
    plt.title('Direction Stability at Different Thresholds')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    for t, s in zip(thresholds, stability_means):
        plt.text(t, s + 0.02, f'{s:.3f}', ha='center')
    plt.savefig(os.path.join(output_dir, 'stability_thresholds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 位置重访率分析（新增）
    revisit_rates = [m["position_revisit_rate"] for m in per_episode]
    revisit_counts = [m["mean_revisit_count"] for m in per_episode]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 4.1 重访率分布
    ax1.hist(revisit_rates, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax1.set_xlabel('Position Revisit Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Position Revisit Rates')
    ax1.axvline(np.mean(revisit_rates), color='r', linestyle='--', 
                label=f'Mean: {np.mean(revisit_rates):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 4.2 每个episode的重访率
    ax2.bar(range(len(revisit_rates)), revisit_rates, alpha=0.7, color='coral')
    ax2.set_xlabel('Episode Index')
    ax2.set_ylabel('Position Revisit Rate')
    ax2.set_title('Position Revisit Rate per Episode')
    ax2.axhline(np.mean(revisit_rates), color='r', linestyle='--', 
                label=f'Mean: {np.mean(revisit_rates):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_revisit_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 探索效率分析：聚类数量 vs 步数
    n_steps = [m["n_steps"] for m in per_episode]
    n_clusters = [m["n_position_clusters"] for m in per_episode]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(n_steps, n_clusters, alpha=0.6, s=100)
    plt.xlabel('Number of Steps')
    plt.ylabel('Number of Position Clusters')
    plt.title('Exploration Efficiency: Unique Areas Visited vs Total Steps')
    
    # 添加趋势线
    if len(n_steps) > 1:
        z = np.polyfit(n_steps, n_clusters, 1)
        p = np.poly1d(z)
        plt.plot(sorted(n_steps), p(sorted(n_steps)), "r--", alpha=0.8, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exploration_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {output_dir}")


def compare_methods(baseline_results: Dict, method_results: Dict) -> Dict:
    """
    比较baseline和你的方法
    """
    comparison = {
        "metric": [],
        "baseline": [],
        "our_method": [],
        "improvement": []
    }
    
    # 比较平均方向变化角度（越小越好）
    baseline_angle = baseline_results["summary"]["mean_angle_change"]
    method_angle = method_results["summary"]["mean_angle_change"]
    improvement = (baseline_angle - method_angle) / baseline_angle * 100
    
    comparison["metric"].append("Mean Direction Change (degrees)")
    comparison["baseline"].append(f"{baseline_angle:.2f}")
    comparison["our_method"].append(f"{method_angle:.2f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    # 比较稳定性比例（越大越好）
    for threshold in [30, 45, 60, 90]:
        key = f"mean_stability_ratio_{threshold}deg"
        baseline_stab = baseline_results["summary"][key]
        method_stab = method_results["summary"][key]
        improvement = (method_stab - baseline_stab) / baseline_stab * 100
        
        comparison["metric"].append(f"Stability Ratio (<{threshold}°)")
        comparison["baseline"].append(f"{baseline_stab:.3f}")
        comparison["our_method"].append(f"{method_stab:.3f}")
        comparison["improvement"].append(f"{improvement:.2f}%")
    
    # 比较frontier重复选择率（如果两者都有数据）
    if "mean_frontier_reselection_rate" in baseline_results["summary"] and \
       "mean_frontier_reselection_rate" in method_results["summary"]:
        baseline_resel = baseline_results["summary"]["mean_frontier_reselection_rate"]
        method_resel = method_results["summary"]["mean_frontier_reselection_rate"]
        improvement = (baseline_resel - method_resel) / baseline_resel * 100
        
        comparison["metric"].append("Frontier Reselection Rate")
        comparison["baseline"].append(f"{baseline_resel:.3f}")
        comparison["our_method"].append(f"{method_resel:.3f}")
        comparison["improvement"].append(f"{improvement:.2f}%")
    
    # 比较位置重访率（新增，越小越好）
    baseline_revisit = baseline_results["summary"]["mean_position_revisit_rate"]
    method_revisit = method_results["summary"]["mean_position_revisit_rate"]
    improvement = (baseline_revisit - method_revisit) / baseline_revisit * 100
    
    comparison["metric"].append("Position Revisit Rate")
    comparison["baseline"].append(f"{baseline_revisit:.3f}")
    comparison["our_method"].append(f"{method_revisit:.3f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    # 比较位置聚类数量（越多表示探索了更多不同区域）
    baseline_clusters = baseline_results["summary"]["mean_n_position_clusters"]
    method_clusters = method_results["summary"]["mean_n_position_clusters"]
    improvement = (method_clusters - baseline_clusters) / baseline_clusters * 100
    
    comparison["metric"].append("Number of Unique Areas (Clusters)")
    comparison["baseline"].append(f"{baseline_clusters:.2f}")
    comparison["our_method"].append(f"{method_clusters:.2f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="计算动作一致性指标")
    parser.add_argument("--result_dir", type=str, required=True,
                      help="结果目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="输出目录（默认为result_dir/consistency_metrics）")
    parser.add_argument("--baseline_dir", type=str, default=None,
                      help="baseline结果目录（用于对比）")
    parser.add_argument("--distance_threshold", type=float, default=0.5,
                      help="位置重访的距离阈值（米），默认0.5m")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, "consistency_metrics")
    
    print("="*60)
    print("动作一致性指标计算")
    print("="*60)
    print(f"结果目录: {args.result_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"距离阈值: {args.distance_threshold}m")
    
    # 分析当前方法
    print("\n分析当前方法的结果...")
    results = analyze_all_episodes(args.result_dir, args.distance_threshold)
    
    if results is None:
        print("\n" + "="*60)
        print("错误：无法加载数据")
        print("="*60)
        print("\n请确保每个episode目录下包含以下文件：")
        print("  - trajectory.json: 保存每步的位置")
        print("  - frontier_choices.json: 保存每步选择的frontier（可选）")
        return
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # 打印摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    summary = results["summary"]
    print(f"成功分析的episode数量: {summary['n_episodes']}")
    print(f"失败的episode数量: {summary['n_failed']}")
    print(f"\n平均方向变化角度: {summary['mean_angle_change']:.2f}° ± {summary['std_angle_change']:.2f}°")
    
    print("\n方向稳定性比例（不同阈值）:")
    for threshold in [30, 45, 60, 90]:
        mean_key = f"mean_stability_ratio_{threshold}deg"
        std_key = f"std_stability_ratio_{threshold}deg"
        print(f"  <{threshold}°: {summary[mean_key]:.3f} ± {summary[std_key]:.3f}")
    
    if "mean_frontier_reselection_rate" in summary:
        print(f"\nFrontier重复选择率:")
        print(f"  平均值: {summary['mean_frontier_reselection_rate']:.3f}")
        print(f"  有效episode数: {summary['n_episodes_with_frontier_data']}/{summary['n_episodes']}")
        print(f"  数据覆盖率: {summary['frontier_data_coverage']*100:.1f}%")
    else:
        print(f"\n⚠️ 注意：没有episode包含frontier选择数据")
    
    # 位置重访率统计（新增）
    print(f"\n位置重访率（距离阈值0.5m）:")
    print(f"  平均重访率: {summary['mean_position_revisit_rate']:.3f} ({summary['mean_position_revisit_rate']*100:.1f}%)")
    print(f"  标准差: {summary['std_position_revisit_rate']:.3f}")
    print(f"  平均每步重访次数: {summary['mean_revisit_count']:.2f}")
    print(f"  平均总重访次数: {summary['mean_total_revisits']:.2f}")
    
    print(f"\n位置聚类统计:")
    print(f"  平均探索区域数: {summary['mean_n_position_clusters']:.2f}")
    print(f"  平均每区域停留步数: {summary['mean_cluster_size']:.2f}")
    
    # 可视化
    print("\n生成可视化图表...")
    visualize_results(results, args.output_dir)
    
    # 如果提供了baseline，进行对比
    if args.baseline_dir:
        print("\n分析baseline结果...")
        baseline_results = analyze_all_episodes(args.baseline_dir, args.distance_threshold)
        if baseline_results:
            comparison = compare_methods(baseline_results, results)
            
            print("\n" + "="*60)
            print("方法对比")
            print("="*60)
            for i in range(len(comparison["metric"])):
                print(f"{comparison['metric'][i]}")
                print(f"  Baseline: {comparison['baseline'][i]}")
                print(f"  Our Method: {comparison['our_method'][i]}")
                print(f"  Improvement: {comparison['improvement'][i]}")
                print()
            
            with open(os.path.join(args.output_dir, "comparison.json"), 'w') as f:
                json.dump(comparison, f, indent=4, cls=NumpyEncoder)
    
    print("\n" + "="*60)
    print(f"分析完成！结果已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
