"""
评估EXPRESS-Bench结果
读取run_express_bench_evaluation_vlm_only.py生成的结果，使用GPT-4o进行评估并生成EAC分数
"""

import json
import os
import time
import pickle
import argparse
import numpy as np
from openai import OpenAI
from typing import Optional
import re
from tqdm import tqdm

# 初始化OpenAI客户端
client = OpenAI(
    api_key='',
    base_url=''
)

def load_prompt(prompt_file):
    """加载prompt文件"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def call_gpt4o(system_prompt, user_prompt, image_path=None, max_retries=5):
    """调用GPT-4o API"""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # 构建用户消息
            user_content = []
            user_content.append({"type": "text", "text": user_prompt})
            
            # 如果有图片，添加图片
            if image_path and os.path.exists(image_path):
                import base64
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            messages.append({"role": "user", "content": user_content})
            
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # "gpt-4o"
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API调用出错 (尝试 {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(3)
            else:
                print(f"达到最大重试次数，跳过该问题")
                return None
    
    return None

def extract_system_and_user_prompt(prompt_content):
    """从prompt文件中提取system和user部分"""
    lines = prompt_content.strip().split('\n')
    
    system_prompt = ""
    user_prompt = ""
    current_section = None
    
    for line in lines:
        if line.strip() == "[system]:":
            current_section = "system"
            continue
        elif line.strip() == "[user]:":
            current_section = "user"
            continue
        
        if current_section == "system":
            system_prompt += line + "\n"
        elif current_section == "user":
            user_prompt += line + "\n"
    
    return system_prompt.strip(), user_prompt.strip()

def parse_eac_scores(response):
    """
    从GPT-4o的回答中提取EAC两个分数
    期望格式: "Your mark: 0.5, 3" 或 "1, 5" 等
    返回: (image_alignment_score, accuracy_score)
    """
    if not response:
        return None, None
    
    # 尝试多种正则表达式模式来提取分数
    patterns = [
        r'Your mark:\s*(\d+\.?\d*)\s*,\s*(\d+)',  # "Your mark: 0.5, 3"
        r'(\d+\.?\d*)\s*,\s*(\d+)',  # "0.5, 3"
        r'mark:\s*(\d+\.?\d*)\s*,\s*(\d+)',  # "mark: 1, 5"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                score1 = float(matches[0][0])
                score2 = int(matches[0][1])
                # 验证分数范围
                if 0 <= score1 <= 1 and 1 <= score2 <= 5:
                    return score1, score2
            except (ValueError, IndexError):
                continue
    
    print(f"警告：无法从响应中解析分数: {response}")
    return None, None

def aggregate_split_results(output_dir):
    """聚合所有split的结果文件（V2版本 - 支持express_bench_info）"""
    # 查找所有的gpt_answer文件
    gpt_answer_files = []
    for file in os.listdir(output_dir):
        if file.startswith("gpt_answer_") and file.endswith(".json"):
            gpt_answer_files.append(file)
    
    if not gpt_answer_files:
        print(f"未找到gpt_answer文件在目录: {output_dir}")
        return None, None, None
    
    # 聚合所有结果
    all_gpt_answers = {}
    all_path_lengths = {}
    all_express_info = {}
    
    for gpt_file in gpt_answer_files:
        # 提取ratio信息
        parts = gpt_file.replace("gpt_answer_", "").replace(".json", "").split("_")
        start_ratio = float(parts[0])
        end_ratio = float(parts[1])
        
        # 读取gpt_answer
        with open(os.path.join(output_dir, gpt_file), 'r') as f:
            gpt_answers = json.load(f)
            
            # 处理两种格式：字典格式和列表格式
            if isinstance(gpt_answers, list):
                # 列表格式: [{"question_id": "xxx", "answer": "yyy"}, ...]
                for item in gpt_answers:
                    all_gpt_answers[item['question_id']] = item['answer']
            elif isinstance(gpt_answers, dict):
                # 字典格式: {"question_id": "answer", ...}
                all_gpt_answers.update(gpt_answers)
            else:
                print(f"警告: 未知的gpt_answer格式: {type(gpt_answers)}")
        
        # 读取对应的path_length文件
        path_file = f"path_length_list_{start_ratio}_{end_ratio}.pkl"
        if os.path.exists(os.path.join(output_dir, path_file)):
            with open(os.path.join(output_dir, path_file), 'rb') as f:
                path_lengths = pickle.load(f)
                all_path_lengths.update(path_lengths)
        
        # ⭐ 新增：读取express_bench_info文件
        express_info_file = f"express_bench_info_{start_ratio}_{end_ratio}.json"
        if os.path.exists(os.path.join(output_dir, express_info_file)):
            with open(os.path.join(output_dir, express_info_file), 'r') as f:
                express_info = json.load(f)
                all_express_info.update(express_info)
            print(f"✓ 找到EXPRESS-Bench信息文件: {express_info_file}")
        else:
            print(f"⚠️  未找到EXPRESS-Bench信息文件: {express_info_file}")
    
    print(f"聚合了 {len(gpt_answer_files)} 个split文件")
    print(f"总共 {len(all_gpt_answers)} 个答案")
    print(f"EXPRESS-Bench信息: {len(all_express_info)} 个question")
    
    return all_gpt_answers, all_path_lengths, all_express_info

def evaluate_express_bench(
    result_dir,
    questions_file,
    prompt_file,
    output_file,
    episode_image_dir=None,
    random_answer_prompt_file=None
):
    """
    评估EXPRESS-Bench结果（V2版本）
    """
    
    print("正在加载结果数据...")
    # 聚合所有split的结果
    gpt_answers, path_lengths, express_info = aggregate_split_results(result_dir)
    
    if gpt_answers is None:
        print("无法加载结果数据")
        return
    
    print(f"答案数据加载完成，共 {len(gpt_answers)} 个答案")
    
    print("正在加载问题数据...")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    print(f"问题数据加载完成，共 {len(questions_data)} 个问题")
    
    print("正在加载评估prompt...")
    prompt_content = load_prompt(prompt_file)
    system_prompt, user_prompt_template = extract_system_and_user_prompt(prompt_content)
    print("Prompt加载完成")
    
    # 加载random_answer prompt（用于"蒙眼"猜答案）
    random_answer_system_prompt = None
    random_answer_user_prompt = None
    if random_answer_prompt_file and os.path.exists(random_answer_prompt_file):
        print(f"正在加载random_answer prompt: {random_answer_prompt_file}")
        random_answer_content = load_prompt(random_answer_prompt_file)
        random_answer_system_prompt, random_answer_user_prompt = extract_system_and_user_prompt(random_answer_content)
        print("Random answer prompt加载完成")
    else:
        print("⚠️  未提供random_answer prompt文件，将跳过空答案的重新生成")
    
    # 创建question_id到question_data的映射
    question_map = {q['question_id']: q for q in questions_data}
    
    results = []
    evaluated_count = 0
    
    # 统计空答案数量
    empty_answer_count = 0
    regenerated_answer_count = 0
    
    # 遍历所有有答案的问题
    for question_id, model_answer in tqdm(gpt_answers.items(), desc="评估进度"):
        if question_id not in question_map:
            print(f"\n警告: question_id {question_id} 不在问题数据集中")
            continue
        
        question_data = question_map[question_id]
        question = question_data['question']
        ground_truth_answer = question_data['answer']
        
        print(f"\n评估问题 {question_id}")
        print(f"问题: {question}")
        print(f"正确答案: {ground_truth_answer}")
        print(f"原始模型回答: {model_answer}")
        
        # ⭐ 新增：检查模型回答是否为空，如果是则用GPT-4o-mini"蒙眼"猜答案
        if (model_answer is None or model_answer == "" or (isinstance(model_answer, str) and model_answer.strip() == "")):
            empty_answer_count += 1
            print(f"⚠️  检测到空答案！")
            
            if random_answer_system_prompt is not None and random_answer_user_prompt is not None:
                print(f"🔄 正在使用GPT-4o-mini重新生成答案...")
                
                # 构造random_answer的user prompt（只包含问题，不包含图片）
                random_user_prompt = f"{random_answer_user_prompt}\n\nQ: {question}\nA: "
                
                # 调用GPT-4o-mini进行"蒙眼"猜测
                regenerated_answer = call_gpt4o(
                    random_answer_system_prompt, 
                    random_user_prompt, 
                    image_path=None,  # 不使用图片
                    max_retries=5
                )
                
                if regenerated_answer:
                    # 清理答案格式
                    regenerated_answer = regenerated_answer.replace("A:", "").strip()
                    model_answer = regenerated_answer
                    regenerated_answer_count += 1
                    print(f"✓ 重新生成的答案: {model_answer}")
                else:
                    print(f"❌ 重新生成答案失败，使用空字符串")
                    model_answer = ""
            else:
                print(f"⚠️  未加载random_answer prompt，跳过重新生成")
                model_answer = ""
        
        print(f"最终模型回答: {model_answer}")
        
        # 构造评估的user prompt
        eval_user_prompt = f"{user_prompt_template}\n"
        eval_user_prompt += f"Question: {question}\n"
        eval_user_prompt += f"Answer: {ground_truth_answer}\n"
        eval_user_prompt += f"Response: {model_answer}\n"
        
        # 尝试查找对应的图片（修改为从chosen_snapshot目录查找）
        image_path = None
        chosen_snapshot_dir = f"{episode_image_dir}/{question_id}/chosen_snapshot"
        
        if os.path.exists(chosen_snapshot_dir):
            # 获取目录中的所有图片文件
            image_files = [f for f in os.listdir(chosen_snapshot_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                # 如果有多个图片，选择第一个
                image_path = os.path.join(chosen_snapshot_dir, image_files[0])
                print(f"✓ 找到图片: {image_path}")
                print(f"📷 本次评估使用图片作为输入 (共{len(image_files)}张图片)")
            else:
                print(f"⚠️  目录存在但未找到图片: {chosen_snapshot_dir}")
                print(f"📷 本次评估无图片输入")
        else:
            print(f"⚠️  未找到chosen_snapshot目录: {chosen_snapshot_dir}")
            print(f"📷 本次评估无图片输入")
        
        # 调用GPT-4o进行评估
        eval_response = call_gpt4o(system_prompt, eval_user_prompt, image_path)
        
        if eval_response:
            print(f"评估结果: {eval_response}")
            
            # 解析EAC分数
            image_score, accuracy_score = parse_eac_scores(eval_response)
            
            if image_score is not None and accuracy_score is not None:
                print(f"Image对齐分数: {image_score}, 准确度分数: {accuracy_score}")
                
                # 获取path_length
                path_length = path_lengths.get(question_id, float('inf'))
                
                # ⭐ 获取EXPRESS-Bench额外信息
                if question_id in express_info:
                    geodesic_distance = express_info[question_id].get('geodesic_distance', float('inf'))
                    goal_dis = express_info[question_id].get('goal_dis', float('inf'))
                    print(f"✓ 使用EXPRESS-Bench信息: geodesic_dist={geodesic_distance:.3f}, goal_dis={goal_dis:.3f}")
                else:
                    # 回退到数据集中的信息
                    geodesic_distance = question_data.get("geodesic_distance", float('inf'))
                    goal_dis = float('inf')
                    print(f"⚠️  使用数据集信息: geodesic_dist={geodesic_distance:.3f}")
                
                # 构建结果条目
                result = {
                    "question_id": question_id,
                    "question": question,
                    "answer": ground_truth_answer,
                    "gen_answer": model_answer,
                    "category": question_data.get("category", "unknown"),
                    "EAC": f"Your mark: {image_score}, {accuracy_score}",
                    "image_alignment_score": image_score,
                    "accuracy_score": accuracy_score,
                    "path_len": path_length,
                    "geodesic_distance": geodesic_distance,
                    "goal_dis": goal_dis,
                    "evaluation_response": eval_response
                }
                results.append(result)
                evaluated_count += 1
            else:
                print("分数解析失败")
                result = {
                    "question_id": question_id,
                    "question": question,
                    "answer": ground_truth_answer,
                    "gen_answer": model_answer,
                    "category": question_data.get("category", "unknown"),
                    "error": "Score parsing failed",
                    "evaluation_response": eval_response
                }
                results.append(result)
        else:
            print("评估失败")
            result = {
                "question_id": question_id,
                "question": question,
                "answer": ground_truth_answer,
                "gen_answer": model_answer,
                "category": question_data.get("category", "unknown"),
                "error": "Evaluation API call failed"
            }
            results.append(result)
        
        # 每处理10个问题保存一次
        if (evaluated_count + 1) % 10 == 0:
            print(f"\n保存中间结果...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    print(f"\n保存最终结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算统计信息
    valid_results = [r for r in results if 'accuracy_score' in r]
    if valid_results:
        avg_image_score = np.mean([r['image_alignment_score'] for r in valid_results])
        avg_accuracy = np.mean([r['accuracy_score'] for r in valid_results])
        print(f"\n评估完成！")
        print(f"成功评估: {len(valid_results)}/{len(gpt_answers)}")
        print(f"平均Image对齐分数: {avg_image_score:.2f}")
        print(f"平均准确度分数: {avg_accuracy:.2f}")
        print(f"\n空答案统计:")
        print(f"  检测到空答案数量: {empty_answer_count}")
        print(f"  成功重新生成答案: {regenerated_answer_count}")
        print(f"  重新生成成功率: {regenerated_answer_count}/{empty_answer_count if empty_answer_count > 0 else 0}")
        print(f"\n评估结果已保存到: {output_file}")
    else:
        print("\n警告：没有成功评估的结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估EXPRESS-Bench结果')
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="结果目录，包含gpt_answer、path_length和express_bench_info文件"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="/.../Pred-EQA/data/express-bench.json",
        help="问题数据集文件"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="/.../Pred-EQA/prompts/EXPRESS-Bench-prompt/evaluation.txt",
        help="评估prompt文件"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出的评估结果文件（默认保存在result-dir/express_bench_evaluation_results.json）"
    )
    parser.add_argument(
        "--episode-image-dir",
        type=str,
        default=None,
        help="episode图片目录（可选）"
    )
    parser.add_argument(
        "--random-answer-prompt",
        type=str,
        default="/.../Pred-EQA/prompts/EXPRESS-Bench-prompt/random_answer.txt",
        help="用于空答案重新生成的prompt文件"
    )
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = os.path.join(args.result_dir, "express_bench_evaluation_results.json")
    
    if args.episode_image_dir is None:
        args.episode_image_dir = args.result_dir
    
    # 评估
    evaluate_express_bench(
        result_dir=args.result_dir,
        questions_file=args.questions_file,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        episode_image_dir=args.episode_image_dir,
        random_answer_prompt_file=args.random_answer_prompt
    )

