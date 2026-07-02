"""
计算EXPRESS-Bench的最终分数
根据EXPRESS-Bench中定义的评分机制计算C_avg, C_star_avg, E_path, d_T_avg
"""

import json
import numpy as np
import argparse

def score(results):
    """
    根据EXPRESS-Bench的评分机制计算分数
    
    Args:
        results: 包含评估结果的列表，每个元素包含:
            - EAC: "Your mark: {image_score}, {accuracy_score}"
            - path_len: 实际路径长度
            - geodesic_distance: 最短路径长度（地理距离）
            - goal_dis: 到目标点的距离
    
    Returns:
        C_avg: 平均准确度分数（考虑image对齐）
        C_star_avg: 平均准确度分数（不考虑image对齐）
        E_path: 考虑路径效率的平均分数
        d_T_avg: 平均目标距离
    """
    C, C_star, p_path, l_path, d_T = [], [], [], [], []
    
    for result in results:
        if result.get("path_len") != float("inf") and "accuracy_score" in result:

            # "Your mark: {image_score}, {accuracy_score}"
            image_score = result.get("image_alignment_score", 0)
            accuracy_score = result.get("accuracy_score", 1)
            
            # C = image_score * accuracy_score
            # C_star = accuracy_score
            C.append(image_score * accuracy_score)
            C_star.append(accuracy_score)
            p_path.append(result["path_len"])
            l_path.append(result["geodesic_distance"])
        
        if result.get("goal_dis") != float("inf"):
            d_T.append(result["goal_dis"])
    
    C = np.array(C)
    C_star = np.array(C_star)
    p_path = np.array(p_path)
    l_path = np.array(l_path)
    d_T = np.array(d_T) if d_T else np.array([])
    
    # 计算路径权重：l_path / max(p_path, l_path)
    weight_path = l_path / np.maximum(p_path, l_path)
    
    # 计算各项指标（分数范围0-5，转换为0-100）
    C_avg = np.mean(100.0 * (np.clip(C, 0, 5) / 5))
    C_star_avg = np.mean(100.0 * (np.clip(C_star, 0, 5) / 5))
    E_path = np.mean(100.0 * (np.clip(C, 0, 5) / 5) * weight_path)
    d_T_avg = np.mean(d_T) if len(d_T) > 0 else float('inf')
    
    return C_avg, C_star_avg, E_path, d_T_avg

def calculate_scores_by_category(results):
    """按类别计算分数"""
    categories = {}
    
    for result in results:
        if "accuracy_score" not in result:
            continue
        
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    category_scores = {}
    for category, cat_results in categories.items():
        if cat_results:
            C_avg, C_star_avg, E_path, d_T_avg = score(cat_results)
            category_scores[category] = {
                "C_avg": C_avg,
                "C_star_avg": C_star_avg,
                "E_path": E_path,
                "d_T_avg": d_T_avg,
                "n_questions": len(cat_results)
            }
    
    return category_scores

def main(evaluation_file, output_file=None):
    """
    主函数：读取评估结果文件，计算分数
    
    Args:
        evaluation_file: evaluate_express_bench.py生成的评估结果文件
        output_file: 输出的分数文件（可选）
    """
    print(f"正在加载评估结果: {evaluation_file}")
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"评估结果加载完成，共 {len(results)} 个结果")
    
    # 统计有效结果
    valid_results = [r for r in results if "accuracy_score" in r]
    print(f"结果: {len(valid_results)}/{len(results)}")
    
    if not valid_results:
        print("错误：没有有效的评估结果")
        return
    
    # 计算总体分数
    print("\n" + "="*50)
    print("总体分数:")
    print("="*50)
    C_avg, C_star_avg, E_path, d_T_avg = score(valid_results)
    
    print(f"C_avg (Image对齐 × 准确度): {C_avg:.2f}")
    print(f"C_star_avg (仅准确度): {C_star_avg:.2f}")
    print(f"E_path (考虑路径效率): {E_path:.2f}")
    print(f"d_T_avg (平均目标距离): {d_T_avg:.2f}")
    
    # 按类别计算分数
    print("\n" + "="*50)
    print("按类别的分数:")
    print("="*50)
    category_scores = calculate_scores_by_category(valid_results)
    
    for category, scores in sorted(category_scores.items()):
        print(f"\n类别: {category} (问题数: {scores['n_questions']})")
        print(f"  C_avg: {scores['C_avg']:.2f}")
        print(f"  C_star_avg: {scores['C_star_avg']:.2f}")
        print(f"  E_path: {scores['E_path']:.2f}")
        print(f"  d_T_avg: {scores['d_T_avg']:.2f}")
    
    # 保存结果
    output_data = {
        "total": {
            "C_avg": float(C_avg),
            "C_star_avg": float(C_star_avg),
            "E_path": float(E_path),
            "d_T_avg": float(d_T_avg),
            "n_questions": len(valid_results),
            "n_total_results": len(results)
        },
        "by_category": {
            cat: {
                "C_avg": float(scores['C_avg']),
                "C_star_avg": float(scores['C_star_avg']),
                "E_path": float(scores['E_path']),
                "d_T_avg": float(scores['d_T_avg']),
                "n_questions": scores['n_questions']
            }
            for cat, scores in category_scores.items()
        }
    }
    
    if output_file:
        print(f"\n保存分数结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    else:
        # 默认保存在同一目录
        import os
        output_file = evaluation_file.replace('.json', '_scores.json')
        print(f"\n保存分数结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print("计算完成！")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算EXPRESS-Bench分数')
    parser.add_argument(
        "--evaluation-file",
        type=str,
        required=True,
        help="评估结果文件（由evaluate_express_bench.py生成）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出的分数文件（可选，默认为evaluation-file的同名_scores.json文件）"
    )
    
    args = parser.parse_args()
    
    main(args.evaluation_file, args.output_file)

