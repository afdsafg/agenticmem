"""
第二步：使用GPT-4o对Blind LLM的回答进行评估打分
读取第一步生成的回答，使用evaluation.txt的prompt，生成评分
"""

import json
import os
import time
from openai import OpenAI
from typing import Optional
import re

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

def load_answers(answers_file):
    """加载回答数据"""
    with open(answers_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def call_gpt4o(system_prompt, user_prompt, max_retries=5):
    """调用GPT-4o API"""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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

def parse_scores(response):
    """
    从GPT-4o的回答中提取两个分数
    期望格式: "0.5, 3" 或 "1, 5" 等
    返回: (image_alignment_score, accuracy_score)
    """
    if not response:
        return None, None
    
    # 尝试多种正则表达式模式来提取分数
    patterns = [
        r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # 匹配 "0.5, 3" 或 "1, 5"
        r'(\d+\.?\d*)\s+(\d+\.?\d*)',      # 匹配 "0.5 3"
        r'(\d+\.?\d*).*?(\d+\.?\d*)',      # 宽松匹配
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                score1 = float(matches[0][0])
                score2 = float(matches[0][1])
                return score1, score2
            except (ValueError, IndexError):
                continue
    
    print(f"警告：无法从响应中解析分数: {response}")
    return None, None

def evaluate_blind_llm_answers(answers_file, prompt_file, output_metrics_file):
    """评估Blind LLM的回答"""
    
    print("正在加载回答数据...")
    answers = load_answers(answers_file)
    print(f"回答数据加载完成，共 {len(answers)} 个回答")
    
    print("正在加载评估prompt...")
    prompt_content = load_prompt(prompt_file)
    system_prompt, user_prompt_template = extract_system_and_user_prompt(prompt_content)
    print("Prompt加载完成")
    
    metrics = {}
    detailed_results = []
    
    for idx, item in enumerate(answers):
        question = item['question']
        answer = item['answer']
        response = item['blind_llm_response']
        question_id = item['question_id']
        
        print(f"\n评估问题 {idx + 1}/{len(answers)} (ID: {question_id})")
        print(f"问题: {question}")
        print(f"正确答案: {answer}")
        print(f"模型回答: {response}")
        
        # 构造评估的user prompt
        # 注意：由于是Blind LLM，没有图片，所以Image alignment score应该是0
        # 但我们仍然按照原始prompt的格式来评估
        eval_user_prompt = f"{user_prompt_template}\n"
        eval_user_prompt += f"Question: {question}\n"
        eval_user_prompt += f"Answer: {answer}\n"
        eval_user_prompt += f"Response: {response}\n"
        eval_user_prompt += f"Image: [No image provided - Blind LLM]\n"
        
        # 调用GPT-4o进行评估
        eval_response = call_gpt4o(system_prompt, eval_user_prompt)
        
        if eval_response:
            print(f"评估结果: {eval_response}")
            
            # 解析分数
            image_score, accuracy_score = parse_scores(eval_response)
            
            if image_score is not None and accuracy_score is not None:
                print(f"Image对齐分数: {image_score}, 准确度分数: {accuracy_score}")
                
                # 保存到metrics字典（使用question_id作为key）
                metrics[str(question_id)] = accuracy_score
                
                # 保存详细结果
                detailed_results.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "image_alignment_score": image_score,
                    "accuracy_score": accuracy_score,
                    "evaluation_response": eval_response
                })
            else:
                print("分数解析失败")
                detailed_results.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "image_alignment_score": None,
                    "accuracy_score": None,
                    "evaluation_response": eval_response,
                    "error": "Score parsing failed"
                })
        else:
            print("评估失败")
            detailed_results.append({
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "response": response,
                "error": "Evaluation API call failed"
            })
        
        # 每处理10个问题保存一次
        if (idx + 1) % 10 == 0:
            print(f"\n保存中间结果...")
            with open(output_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            detailed_output = output_metrics_file.replace('.json', '-detailed.json')
            with open(detailed_output, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    print(f"\n保存最终结果到 {output_metrics_file}...")
    with open(output_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    detailed_output = output_metrics_file.replace('.json', '-detailed.json')
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # 计算统计信息
    valid_scores = [r['accuracy_score'] for r in detailed_results if r.get('accuracy_score') is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n评估完成！")
        print(f"成功评估: {len(valid_scores)}/{len(answers)}")
        print(f"平均准确度分数: {avg_score:.2f}")
        print(f"评分结果已保存到: {output_metrics_file}")
        print(f"详细结果已保存到: {detailed_output}")
    else:
        print("\n警告：没有成功评估的结果")

