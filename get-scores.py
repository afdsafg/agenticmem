import json
import numpy as np
import pickle
import os
import argparse

skip_scenes = []
skip_scene_ids = [scene.split("-")[1] for scene in skip_scenes]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result-path",
    type=str,
)
parser.add_argument(
    "--dataset",
    default="open-eqa-184",
    type=str,
)
parser.add_argument(
    "--only-evaluated",
    action="store_true",
    help="Only calculate scores for evaluated questions (ignore baseline for missing questions)",
)
parser.add_argument(
    "--start-ratio",
    type=float,
    default=0.0,
    help="Start ratio for data subset (default: 0.0)",
)
parser.add_argument(
    "--end-ratio",
    type=float,
    default=1.0,
    help="End ratio for data subset (default: 1.0)",
)
args = parser.parse_args()


data_path = args.result_path
path_length_name = "/Pred-EQA/path_length_list.pkl"
path_length_path = data_path + path_length_name

gt_path = f'/.../Pred-EQA/data/{args.dataset}.json'
pred_path = f'{data_path}/metrics/gpt_answer-metrics.json'

# Use Blind LLM as the baseline for unsuccessful episodes
baseline_path = f'/.../Pred-EQA/data/{args.dataset}-gpt-4o-1234-metrics.json'
# Use path length in GT trajectories for SPL
gt_path_length_path = '/.../Pred-EQA/data/gt_path_length.json'

with open(gt_path_length_path, 'rb') as f:
    gt_path_length_map = json.load(f)
with open(path_length_path, 'rb') as f:
    path_length_map = pickle.load(f)

baseline_path_length_map = {k: float('inf') for k, v in gt_path_length_map.items()}

def spl(path_length, gt_path_length):
    return gt_path_length / max(gt_path_length, path_length)

separate_spl = {}
separate_scores = {}
gt = json.load(open(gt_path))
pred = json.load(open(pred_path))
baseline = json.load(open(baseline_path))

# Decide which questions to process based on parameters
if args.start_ratio != 0.0 or args.end_ratio != 1.0:
    # Use data subset based on start_ratio and end_ratio
    # Sort ground truth questions by question_id to ensure consistent ordering
    gt_sorted = sorted(gt, key=lambda x: x["question_id"])
    total_questions = len(gt_sorted)
    start_idx = int(args.start_ratio * total_questions)
    end_idx = int(args.end_ratio * total_questions)
    gt_subset = gt_sorted[start_idx:end_idx]
    question_ids_to_process = [q["question_id"] for q in gt_subset]
    print(f"[INFO] Evaluating subset [{args.start_ratio:.2f}, {args.end_ratio:.2f}]: {len(question_ids_to_process)} questions (using baseline for missing)")
elif args.only_evaluated:
    # Only process questions that were actually evaluated
    question_ids_to_process = list(pred.keys())
    print(f"[INFO] Only evaluating {len(question_ids_to_process)} questions that were actually run")
else:
    # Process all questions in baseline (original behavior)
    question_ids_to_process = list(baseline.keys())
    print(f"[INFO] Evaluating all {len(question_ids_to_process)} questions (using baseline for missing)")

# Track statistics
num_predicted = 0
num_baseline_used = 0

for question_id in question_ids_to_process:
    # Get baseline score (if needed)
    score = baseline.get(question_id, 1)  # Default to 1 (lowest score) if not in baseline
    
    # Find question in ground truth
    question_list = [q for q in gt if q['question_id'] == question_id]
    if not question_list:
        print(f"[WARNING] question_id {question_id} not found in ground truth, skipping")
        continue
    question = question_list[0]
    
    if question['episode_history'].split("-")[-1] in skip_scene_ids:
        continue
    
    # Get ground truth path length
    if question_id not in gt_path_length_map:
        print(f"[WARNING] No ground truth path length for {question_id}, skipping")
        continue
    gt_path_length = gt_path_length_map[question_id]
    
    # Determine path length and score
    if question_id not in pred.keys():
        path_length = baseline_path_length_map[question_id]
        # score already set from baseline above
        num_baseline_used += 1
    else:
        try:
            path_length = path_length_map[question_id]
        except:
            print(f"[WARNING] {question_id} not in path_length_map, using infinite path length")
            path_length = baseline_path_length_map[question_id]
        score = pred[question_id]
        num_predicted += 1
    
    category = question['category']
    if category not in separate_scores:
        separate_scores[category] = []
    if category not in separate_spl:
        separate_spl[category] = []
    separate_scores[category].append(score)
    separate_spl[category].append(spl(path_length, gt_path_length))

total_scores = []
total_spl = []
for category, scores in separate_scores.items():
    spl_coeffs = separate_spl[category]
    total_scores.extend(scores)
    scores = np.array(scores)
    spl_scores = np.array(spl_coeffs)
    scores = 100.0 * (scores - 1.0) / 4.0
    spl_scores = scores * spl_coeffs
    total_spl.extend(spl_scores)
    scores = np.mean(scores)
    spl_scores = np.mean(spl_scores)
    print(f'{category}: {scores:.2f}')
    print(f'{category} SPL: {spl_scores:.2f}')

total_scores = np.array(total_scores)
total_scores = 100.0 * (total_scores - 1.0) / 4.0
total_scores_mean = np.mean(total_scores)
total_spl = np.array(total_spl)
total_spl_mean = np.mean(total_spl)

print(f'\n{"="*50}')
print(f'Total: {total_scores_mean:.2f}')
print(f'Total SPL: {total_spl_mean:.2f}')
print(f'Number of questions evaluated: {len(total_scores)}')
print(f'  - Questions with predictions: {num_predicted}')
print(f'  - Questions using baseline: {num_baseline_used}')
print(f'{"="*50}')
