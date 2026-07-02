import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src.const import *
from qwen_vl_utils import process_vision_info
import re
import torch
import numpy as np
from src.long_term_memory import TextLongTermMemory
from src.plan_extraction_utils import extract_predictive_plan

def safe_findall(pattern, string):
    """安全的正则表达式查找函数,处理字符串可能为None的情况"""
    if string is None:
        return []
    return re.findall(pattern, string)

def safe_strip(string):
    """安全的字符串strip函数,处理字符串可能为None的情况"""
    if string is None:
        return ""
    return string.strip()

client = OpenAI(
    api_key="local",
    base_url="http://127.0.0.1:12221/v1",
    timeout=3600
)


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                    },
                }
            )
    return formated_content


# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
                messages=message_text,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None

def call_openai_api_text(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
                messages=message_text,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.8,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None


def parse_number_list(numbers_str: str) -> list:
    """解析数字列表字符串，如 '0, 1, 2' 或 '0,1,2'"""
    if not numbers_str:
        return []

    # 特殊处理：如果字符串是"."，表示保留全部
    if numbers_str.strip() == ".":
        return ["ALL"]  # 返回包含"ALL"的列表，而不是字符串本身

    # 清理字符串
    clean_str = re.sub(r'[{}()\[\]]', '', numbers_str)
    clean_str = clean_str.replace(' ', '')
    # 去除末尾的标点符号，如句点
    clean_str = clean_str.rstrip('.,;:!?')

    # 如果清理后是空字符串，返回空列表表示保留空集合
    if not clean_str:
        return []

    # 分割并转换为整数
    number_strs = clean_str.split(',')
    numbers = []

    for num_str in number_strs:
        num_str = num_str.strip()
        if num_str.isdigit():
            numbers.append(int(num_str))

    return numbers


def parse_retain_response(response: str, prefix: str = "Retain Snapshots") -> list:
    """
    通用的解析函数，从模型响应中解析 "Retain X:" 后面的数字列表
    
    Args:
        response: 模型响应字符串
        prefix: 前缀，如 "Retain Snapshots" 或 "Retain Frontiers"
    
    处理多种格式：
    1. "Retain X: {0, 1, 2}."
    2. "Retain X: 0, 1, 2."
    3. "Retain X: 0, 1, 2, 3, 4, 5, 6, 7, 8."
    
    策略：
    - 查找所有匹配项，使用最后一个（避免匹配到格式说明）
    - 支持带花括号和不带花括号的格式
    - 处理句号、换行等结尾符号
    """
    if not response:
        return []
    
    select_id = []
    
    # 策略1：匹配带花括号的格式 "Retain X: {0, 1, 2}."
    # 使用更严格的模式，确保匹配实际数字列表而不是格式说明
    pattern_with_braces = rf'{prefix}:\s*{{([0-9,\s]+)}}'
    matches_with_braces = safe_findall(pattern_with_braces, response)
    
    if matches_with_braces:
        # 取最后一个匹配（最可能是实际答案）
        final_match = matches_with_braces[-1].strip()
        if final_match:
            select_id = parse_number_list(final_match)
            if select_id:
                return select_id
    
    # 策略2：匹配不带花括号的格式 "Retain X: 0, 1, 2."
    # 改进正则：要求至少有一个数字，并且不是格式说明中的占位符
    # 排除格式说明中的 "{i, ...}" 这样的模式
    pattern_without_braces = rf'{prefix}:\s*([0-9]+(?:\s*,\s*[0-9]+)*)'
    matches_without_braces = safe_findall(pattern_without_braces, response)
    
    if matches_without_braces:
        # 取最后一个匹配（最可能是实际答案）
        # 但我们需要检查整行，因为可能包含被文本分隔的数字
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if f'{prefix}:' in line:
                # 提取冒号后的整行内容
                after_colon = line.split(f'{prefix}:', 1)[-1].strip()
                # 提取所有数字（包括被文本分隔的）
                numbers = re.findall(r'\d+', after_colon)
                if numbers:
                    try:
                        select_id = [int(n) for n in numbers]
                        return select_id
                    except ValueError:
                        continue
        # 如果上面的反向查找失败，使用原来的方法作为备选
        final_match = matches_without_braces[-1].strip()
        if final_match:
            select_id = parse_number_list(final_match)
            if select_id:
                return select_id
    
    # 策略3：如果上面都没匹配到，尝试更宽松的模式
    # 查找包含实际数字（不是占位符）的行
    lines = response.split('\n')
    for line in reversed(lines):  # 从后往前查找
        line = line.strip()
        # 确保这一行包含 "Retain X:" 且后面有数字
        if f'{prefix}:' in line:
            # 提取冒号后的部分
            after_colon = line.split(f'{prefix}:', 1)[-1].strip()
            # 去除可能的引号和格式说明
            after_colon = re.sub(r'[{}()\[\]"]', '', after_colon)
            # 尝试提取所有数字（包括被文本分隔的数字）
            numbers = re.findall(r'\d+', after_colon)
            if numbers:
                try:
                    select_id = [int(n) for n in numbers]
                    return select_id
                except ValueError:
                    continue
    
    # 如果所有策略都失败，返回空列表
    return []


def parse_retain_snapshots_response(response: str) -> list:
    """解析 Retain Snapshots 响应"""
    return parse_retain_response(response, "Retain Snapshots")


def parse_retain_frontiers_response(response: str) -> list:
    """解析 Retain Frontiers 响应"""
    return parse_retain_response(response, "Retain Frontiers")

def remove_digits(text: str) -> str:
    """将字符串中所有数字替换为空格"""
    return re.sub(r'\d', ' ', text)


def generate_step_summary(step_num, agent_outputs, question, step=None):
    """
    为一个step生成综合总结，涵盖所有agent的输出
    优化：增强总结质量，添加更多上下文信息
    Args:
        step_num: step编号
        agent_outputs: 该step内所有agent的输出列表
        question: 当前问题
        step: 当前step对象，用于获取memory信息
    Returns:
        step的综合总结
    """
    if not agent_outputs:
        return "No agent outputs for this step."
    agent_summaries = []
    for output in agent_outputs:
        agent_type = output['agent_type']
        content = output['content']
        agent_summaries.append(f"{agent_type}: {content}")

    combined_content = "\n".join(agent_summaries)

    sys_prompt = f"""You are a concise assistant that summarizes a step in an exploration process for long-term textual memory.

Task: Summarize the step's key information useful for future reasoning. Focus on:
1. Critical environmental or spatial observations (e.g., room layout, connectivity, notable objects).
2. Progress and current status.

Constraints:
Keep under 150 words.
Be specific and forward-looking—prioritize details that won't be available in future steps (e.g., visual or spatial context).
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
If no meaningful activity or observation occurred, return "No significant activity in this step."
"""

    # 添加memory信息到prompt中
    memory_info_str = ""
    if step is not None:
        try:
            memory_info_str = format_memory_info(step, max_steps=20, outside=False)
            memory_info_str = f"\nRelevant Memory Information:\n{memory_info_str}\n"
        except Exception as e:
            memory_info_str = "\nMemory information unavailable.\n"

    contents = [(f"Step {step_num} Agent Outputs:\n{combined_content}{memory_info_str}",)]

    summary = call_openai_api_text(sys_prompt, contents)

    if summary is None:
        return combined_content[:200] + "..." if len(combined_content) > 200 else combined_content
    summary = remove_digits(summary)

    return summary.strip()


def generate_response_summary(response: str, response_type: str, question: Optional[str] = None, step: Optional[dict] = None) -> str:
    """
    使用VLM生成响应总结，基于问题过滤相关信息
    Args:
        response: VLM的完整响应
        response_type: 响应类型
        question: 当前的问题，用于过滤相关信息
        step: 当前step对象，用于获取memory信息
    Returns:
        响应的总结文本
    """
    if not response or response.strip() == "":
        return "No response to summarize"

    # 如果没有提供问题，使用原有逻辑
    if not question:
        question_context = ""
    else:
        question_context = f"Question Context: {question}\n"

    sys_prompt = f"""You are a concise and helpful assistant that converts visual observations into textual long-term memory.

Task: Summarize the current visual scene clearly and briefly for future reference. Focus ONLY on:
1. Environmental details (e.g., room layout, objects, walls, doors).
2. Spatial connectivity (e.g., how rooms or areas link to each other).

Constraints:
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
Describe only observable environment and connectivity.
Keep under 100 words."""
    memory_info_str = ""
    if step is not None:
        try:
            memory_info_str = format_memory_info(step)
            memory_info_str = f"\nRelevant Memory Information:\n{memory_info_str}\n"
        except Exception as e:
            memory_info_str = "\nMemory information unavailable.\n"
    contents = [(f"{response_type.upper()} Response to convert:\n{response}{memory_info_str}",)]
    summary = call_openai_api_text(sys_prompt, contents)

    if summary is None:
        return response[:200] + "..." if len(response) > 200 else response
    summary = remove_digits(summary)
    return summary.strip()


def _parse_answerer_response(response: str) -> dict:
    """从 answerer 响应中解析答案或 Continue Exploration 决策。
    期望格式:
      - "Answer: [your concise answer] (Evidence: Snapshot [index])"
      - "Continue Exploration"
    """
    if not response:
        return {"action": "unknown", "answer_text": "", "evidence_snapshot": None}

    text = response.strip()
    lowered = text.lower()

    # Continue Exploration
    if "continue exploration" in lowered:
        return {"action": "continue_exploration", "answer_text": "", "evidence_snapshot": None}

    # Answer: ... (Evidence: Snapshot i)
    # 取最后一个 "Answer:" 匹配，避免命中格式说明
    answer_matches = re.findall(
        r'Answer:\s*(.+?)(?:\s*\(Evidence:\s*Snapshot\s*(\d+)\s*\))?\s*$',
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if answer_matches:
        answer_text, snap_idx = answer_matches[-1]
        answer_text = answer_text.strip().strip('.').strip()
        # 去掉可能残留的句尾标点/引号
        answer_text = answer_text.strip('"').strip("'").strip()
        snap = int(snap_idx) if snap_idx and snap_idx.isdigit() else None
        if answer_text:
            return {
                "action": "answer",
                "answer_text": answer_text,
                "evidence_snapshot": snap,
            }

    # 退化：整段当作 answer_text（至少有内容可记）
    snippet = text[:200].strip()
    return {"action": "answer", "answer_text": snippet, "evidence_snapshot": None}


def _parse_planner_response(response: str, parsed_action: Optional[str] = None,
                            parsed_reason: Optional[str] = None) -> dict:
    """从 planner/explore 响应中解析 "Next Step: Frontier i" 或 "Stop Exploration"。
    若主流程已解析出 action（如 "frontier 2" / "stop exploration"），直接复用。"""
    if parsed_action:
        action = parsed_action.lower().strip()
        if action.startswith("frontier"):
            # "frontier 2" -> target_id=2
            nums = re.findall(r'\d+', action)
            target_id = int(nums[0]) if nums else None
            return {
                "action": "frontier",
                "target_type": "frontier",
                "target_id": target_id,
                "reason": parsed_reason or "",
            }
        if "stop" in action:
            return {"action": "stop_exploration", "target_type": None, "target_id": None,
                    "reason": parsed_reason or ""}

    # 从原始响应里兜底解析
    if response:
        text = response.strip()
        m = re.search(r'next\s+step\s*:\s*frontier\s+(\d+)', text, re.IGNORECASE)
        if m:
            return {"action": "frontier", "target_type": "frontier",
                    "target_id": int(m.group(1)), "reason": parsed_reason or ""}
        if re.search(r'stop\s+exploration', text, re.IGNORECASE):
            return {"action": "stop_exploration", "target_type": None, "target_id": None,
                    "reason": parsed_reason or ""}

    return {"action": "unknown", "target_type": None, "target_id": None,
            "reason": parsed_reason or ""}


def extract_structured_output_from_response(response: str, response_type: str,
                                            question: Optional[str] = None,
                                            step: Optional[dict] = None,
                                            parsed_action: Optional[str] = None,
                                            parsed_reason: Optional[str] = None) -> dict:
    """
    从VLM响应中提取结构化输出信息，基于问题过滤相关信息。

    返回的字典包含：
      - raw_response:       VLM 完整原始响应
      - reasoning:          由 VLM 生成的场景摘要
      - response_type:      agent 类型
      - parsed_decision:    可读的简短决策字符串（用于 long_term_memory 历史展示）
      - structured_output:  按 agent 类型组织的结构化字段（frontier_ids/snapshot_ids/answer_text/action...）
    """
    response_summary = generate_response_summary(response, response_type, question, step)

    structured_output: Dict = {}
    parsed_decision = "Unknown"

    try:
        if response_type == "frontier_manager":
            ids = parse_retain_frontiers_response(response or "")
            structured_output = {"frontier_ids": ids}
            parsed_decision = f"Retain Frontiers: {ids}" if ids else "Retain Frontiers: []"

        elif response_type == "snapshot_manager":
            ids = parse_retain_snapshots_response(response or "")
            structured_output = {"snapshot_ids": ids}
            parsed_decision = f"Retain Snapshots: {ids}" if ids else "Retain Snapshots: []"

        elif response_type == "answerer":
            parsed = _parse_answerer_response(response or "")
            structured_output = parsed
            if parsed["action"] == "answer":
                ans = parsed["answer_text"]
                parsed_decision = f"Answer: {ans[:80]}"
            elif parsed["action"] == "continue_exploration":
                parsed_decision = "Continue Exploration"
            else:
                parsed_decision = "Unknown"

        elif response_type == "planner":
            parsed = _parse_planner_response(response or "", parsed_action, parsed_reason)
            structured_output = parsed
            if parsed["action"] == "frontier":
                parsed_decision = f"Next Step: Frontier {parsed.get('target_id')}"
            elif parsed["action"] == "stop_exploration":
                parsed_decision = "Stop Exploration"
            else:
                parsed_decision = "Unknown"

        else:
            structured_output = {}
            parsed_decision = (response or "")[:80].strip() or "Unknown"
    except Exception as e:
        logging.warning(f"extract_structured_output_from_response parse failed for "
                        f"{response_type}: {e}")
        structured_output = {}
        parsed_decision = "Unknown"

    return {
        "raw_response": response,
        "reasoning": response_summary,
        "response_type": response_type,
        "parsed_decision": parsed_decision,
        "structured_output": structured_output,
    }



def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.3 get snapshots
    snapshot_imgs, snapshot_classes = [], []
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]
        snapshot_imgs.append(encode_tensor2base64(snapshot_img))
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        # remove duplicates
        snapshot_class = sorted(list(set(snapshot_class)))
        seen_classes.update(snapshot_class)
        snapshot_classes.append(snapshot_class)


    keep_index = list(range(len(snapshot_imgs)))
    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        keep_index,
    )


def format_manage_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):
    sys_prompt = """Task: You are an indoor MEMORY MANAGEMENT AGENT responsible for CURATING and PRESERVING visual snapshots and spatial information collected by the embodied agent during its navigation, working in tandem with your existing TEXTUAL MEMORY and high-level plan. 

Instructions:
1. CAREFULLY analyze the information needed to answer the question, paying special attention to location details, objectives, object relationships, and any mentioned or implied attributes.
2. Review all available snapshots thoroughly and cross-reference them with your TEXTUAL MEMORY. When deciding whether to retain a snapshot, adopt a conservative approach - if there is ANY potential visual relevance to the current question or its context, it should be preserved. Specifically, retain snapshots that include:
   - Any room types or spaces that may be related to the question's context, even indirectly.
   - Adjacent or connected areas that could provide spatial clues or lead to relevant locations.
   - Partial views or incomplete perspectives of objects, appliances, or features that might be useful in reasoning.
   - Environmental or contextual cues (e.g., lighting, layout, orientation) that help establish spatial understanding or support inference.
   - Objects or categories explicitly mentioned in the question, as well as those that are semantically or functionally associated.
   - Any image that provides visual background or situational information not fully captured by text, which could aid in answering the question or reconstructing the environment.

3. MEMORY COMPACTION (Textual Redundancy Filter): To prevent critical visual clues from being overwhelmed by redundant trajectory images, you may DISCARD a snapshot ONLY IF it meets BOTH of the following conditions:
   - It is completely irrelevant to the primary question or objective (contains no target objects or contextual clues).
   - Its environmental content, spatial relationships, or navigational cues are already adequately and comprehensively described in your existing textual memory.

4. When in doubt—especially if you are unsure whether the textual memory fully captures the visual nuances of the scene—err on the side of retention. Even seemingly minor or indirect visual clues can become valuable during later stages of reasoning or path reconstruction.
"""
    content = []
    # 1. Question
    content.append((f"Question: {question}\n",))

    # 2. Memory information 
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            # memory_info = format_memory_info(step, max_steps=3)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # 3. Snapshots display
    content.append(("Available Snapshots:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append((f"Snapshot {i}: ", img))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Retain Snapshots: {i, ...}.\""
    content.append((text,))
    

    return sys_prompt, content

def format_answer_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):

    sys_prompt = """Task: You are an indoor agent that needs to determine if the current collected information is sufficient to answer the question.
 
Instructions:   
1. CAREFULLY analyze the information needed to answer the question, especially location, objectives, relationships, and attributes.
2. CAREFULLY analyze ALL available snapshots (total observed clues).
3. If ANY snapshot contains information needed to answer the question, output Answer.
4. If NO snapshot provides sufficient information, output Continue Exploration.
"""

    content = []
    # 1. Question
    content.append((f"Question: {question}\n",))

    # 2. Memory information 
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # 3. Snapshots display
    content.append(("Available Snapshots:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append((f"Snapshot {i}: ", img))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. If answerable, provide your final answer in the exact format: \"Answer: [your concise answer] (Evidence: Snapshot [index])\"\n"
    text += "If not, use format: \"Continue Exploration\""
    content.append((text,))


    return sys_prompt, content


def format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None, 
):
    sys_prompt = """Task: You are an indoor agent that needs to PHYSICALLY NAVIGATE through sequential frontier selections to finally find information needed for answering the question.

Instructions:
1. Analyze the question's information requirements, especially locations, objectives, relationships, and attributes. Identify target objects and their typical locations based on common sense.
2. Assess the previously observed clues to determine already explored areas and objects.
3. Given question needs and current exploration progress, choose a frontier based on the following Core Principles and constraints:
principle 1: Use common room-object relationships to infer possible locations of the target object (e.g., "refrigerator" in kitchen, "bed" in bedroom). Use typical room connections to prioritize exploration directions (e.g., kitchen is often adjacent to living room or dining room).
principle 2: If you are in an unrelated area, choose the frontier leading to a potentially relevant area.  If previously observed clues do not suggest that the relevant area has already been explored, continue exploring without stopping until you reach the relevant area.
principle 3: Balance proximity with strategic long-range exploration when clues suggest distant frontiers.
constraint 1: If you find that you are still in an irrelevant area, you can only choose a frontier and continue walking in order to reach the relevant area.
constraint 2: You can only access to unvisited areas by selecting a frontier step-by-step.
constraint 3: Keep selecting a frontier for moving until you find conclusive evidence enough to answer the question. Note that the objects mentioned in all questions are definitely available.

"""

    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))
    
    # 2. Memory information (add before images)
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))


    content.append(("Previously Observed Clues:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append(("\n", img))
        content.append(("\n",))

    # 2. Frontiers display
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))  # TODO
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",))

    # # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Next Step: Frontier i\" or \"Stop Exploration\", where i is the index of the frontier you choose."
    # text += "Ensure that your answer includes at least 3 snapshot indices.\n\n"
    content.append((text,))


    return sys_prompt, content

def format_high_level_plan_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None,  # 添加step参数用于memory信息
):
    sys_prompt = """Task: You are a HIGH-LEVEL EXPLORATION PLANNER AGENT responsible for devising a long-term navigation and search plan to answer the user's question. Based on the question, you must break down the goal into a sequence of high-level tasks (e.g., go to a room, find an object, observe an attribute) and output them as an ordered to-do list. This plan will guide the low-level agents in subsequent steps.

Instructions:
1. Analyze the user's question and identify its type (object recognition, attribute recognition, spatial relationship, object state, functional reasoning, world knowledge, or object localization).
2. Decompose the question into subgoals. For example:
   - Object recognition: Determine which object to find and where it is likely located.
   - Attribute recognition: Identify the object and which attribute to check.
   - Spatial understanding: Decide which locations or objects need exploration to understand their spatial arrangement.
   - Object state recognition: Determine which object's state to verify and how to observe it.
   - Functional reasoning: Identify relevant objects that demonstrate the function in question.
   - World knowledge: Use typical associations (e.g., kitchen contains a fridge) to infer where to search.
   - Object localization: Plan a search sequence for locating the object in different rooms.
3. For each subgoal, create a clear task (e.g., “Go to the kitchen”, “Find the refrigerator”, “Check the microwave's door status”).
4. Create Parallel Prediction-Based Branches for the immediate next step. For the most immediate unresolved navigation or search task (e.g., figuring out how to get to the kitchen), do not output a single generic task. Instead, generate multiple parallel prediction-based exploration branches. Formulate these as testable hypotheses based on current observations and world knowledge (e.g., instead of [ ] Find the kitchen, create [ ] Explore the frontier leading to the hallway since it may lead to the kitchen and [ ] Explore the frontier leading to the living area since it may lead to the kitchen).
5. Combine these immediate predictive branches and the remaining downstream high-level tasks into a single, cohesive, ordered to-do list. Place the parallel predictive branches at the very top as the active starting point, followed by the subsequent tasks.
6. Use the updateable checklist format for output. Mark tasks as [ ] pending, [-] in progress, or [x] completed based on what has been done so far. When agents investigate and eliminate predictive branches, mark the incorrect or dead-end branches as completed [x] with a brief inline explanation. Add new tasks immediately when they become apparent. Do not remove unfinished tasks unless they are truly irrelevant to the goal.
Core Principles:
- Before updating, always confirm which todos have been completed or invalidated since the last update.
- You may update multiple statuses in a single update (e.g., mark the previous as completed, invalidate obsolete tasks, and mark the new one as in progress).
- Dynamic Replanning: Because the environment is partially observable, new observations may completely invalidate your previous assumptions or downstream plans. If this happens, you MUST actively overhaul the plan. You are allowed to restructure, replace, or pivot away from previously planned unfinished tasks to align with the newly discovered ground truth.
- When a prediction-based branch proves incorrect (a dead-end), OR when a downstream task becomes obsolete due to a plan overhaul, mark it as [x] AND append a brief inline comment explaining why it failed or was discarded (e.g., [x] Search the kitchen ).
- Once ONE predictive branch successfully locates the target, immediately mark all other parallel predictive branches for that same goal as [x] with an explanation.
- When a completely new actionable path is discovered that pivots the entire strategy, add the new tasks immediately into the sequence and mark the old, now-irrelevant tasks as [x] with a brief explanation of the pivot.
- For regular tasks that remain relevant to the current valid strategy, only mark them as completed [x] when fully accomplished successfully (no partials, no unresolved dependencies).
Content Constraints:
STRICTLY NEVER mention ANY snapshot or frontier identifiers (e.g., "Snapshot 2", "Frontier 0") - these labels are step-specific and will cause confusion in later steps when the current image is no longer available.
AVOID relative directional references tied to transient views (e.g., “left of the snapshot”, “right of the frontier”). Instead, describe spatial relationships using observable objects (e.g., “the chair is next to the table”).
Example:
[ ] Go through the doorway into the kitchen.
[x] Explore the frontier leading to the hallway to check if it leads to the kitchen. <!-- Irrelevant; kitchen is confirmed via doorway -->
[x] Explore the frontier leading to the living area to check if it leads to the kitchen. <!-- Also irrelevant; kitchen is already identified -->
[x] Retain the view through the kitchen doorway as it leads to the target location. <!-- Already completed; serves as navigation anchor -->

""" 
    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))
    
    # 2. Memory information (add before images)
    if step is not None:
        try:
            memory_info = format_memory_info(step)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    # # 3. Format specification
    text = '''Output Format:
1. First, think step by step and explain your reasoning clearly.
2. Always output your tasks in the following XML checklist format:
<update_todo_list>
<todos>
[ ] Pending task description
[-] In progress task description <!-- status; rationale -->
[x] Completed or pruned task description <!-- status; rationale -->
</todos>
</update_todo_list>
'''
    content.append((text,))


    return sys_prompt, content



def format_plan_manager_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        step=None,  
):
    sys_prompt = '''Task: You are an EXPLORATION DIRECTION MANAGEMENT AGENT responsible for STRATEGICALLY SELECTING and PRUNING potential frontiers based on observed visual snapshots.  Your goal is to eliminate directions that have BOTH OBVIOUSLY BEEN EXPLORED AND ARE IRRELEVANT to answering the question.

Instructions:
1.   CAREFULLY analyze the provided visual snapshots to identify areas that have already been explored.
2.   Determine which frontiers (exploration directions) can be safely removed because they MEET BOTH CRITERIA:
- They lead to areas ALREADY CONFIRMED AS VISITED with high certainty.
- The area or objects within them are CLEARLY UNRELATED TO THE QUESTION or its context.
3.   ONLY remove such frontiers if BOTH conditions above are MET.  If ANY DOUBT exists about either exploration status or relevance, KEEP THE FRONTIER.
4.   Retain all other frontiers, including those where there is ANY UNCERTAINTY regarding their exploration status or their relevance to the question.
5.   Maintain spatial awareness: even partially visible rooms or ambiguous paths should be preserved unless you are ABSOLUTELY CERTAIN about their irrelevance.
6.   REMEMBER, the key is to avoid deleting potentially useful information.  When in doubt, err on the side of caution and retain the frontier.
'''
    content = []
    # 1. Context reminder
    content.append((f"Target Question: {question}\n",))

    # # 2. Memory information (add before images)
    if step is not None:
        try:
            # memory_info = format_memory_info(step)
            memory_info = format_memory_info(step, only_high_level_plan=True)
            content.append((memory_info,))
        except Exception as e:
            content.append(("Memory information unavailable.\n",))

    content.append(("Previously Observed Clues:\n",))
    if not snapshot_imgs:
        content.append(("No snapshots available\n",))
    else:
        for i, img in enumerate(snapshot_imgs):
            content.append(("\n", img))
        content.append(("\n",))

    # 2. Frontiers display
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))  # TODO
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",))

    # 3. Format specification
    text = "Output Format:\n"
    text += "1. First, think step by step and explain your reasoning clearly.\n"
    text += "2. Then, provide your final answer in the exact format: \"Retain Frontiers: {i, ...}\" (retain at least 1 frontiers)."
    content.append((text,))

    return sys_prompt, content


def format_force_answer_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    step=None,  
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings. To answer the question, you are required to choose a Snapshot as the answer.\n"
    sys_prompt += "Definitions:\n"
    sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "If you choose a Snapshot, you need to directly give an answer to the question. Your answer is mandatory and you must select one of the available Snapshots.\n"

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            content.append(("\n",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i\n[Answer]', where i is the index of the snapshot you choose. "
    text += "Your answer is mandatory and you must select one of the available Snapshots. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
    text += "(2) you can also utilize other snapshots and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
    content.append((text,))

    return sys_prompt, content





def get_agent_outputs_by_step_and_type(step, step_num, agent_types=None):
    """
    优化的记忆检索函数：一次性获取指定step中指定类型agent的输出
    Args:
        step: 当前step对象
        step_num: 要检索的step编号
        agent_types: 要检索的agent类型列表，如果为None则检索所有类型
    Returns:
        字典，键为agent类型，值为该类型的所有输出
    """
    if 'scene' not in step or step['scene'] is None:
        return {}

    # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
    agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]
    
    # 如果没有指定agent类型，则使用默认顺序
    if agent_types is None:
        agent_types = agent_execution_order
    
    # 一次性检索所有需要的agent输出
    agent_outputs = {}
    try:
        all_current_step_outputs = []
        for agent_type in agent_types:
            outputs = step['scene'].long_term_memory.retrieve_by_type(f"{agent_type}_output", top_k=50)
            for output in outputs:
                if output.step == step_num:
                    all_current_step_outputs.append((agent_type, output))
        
        # 按agent类型组织输出
        for agent_type, output in all_current_step_outputs:
            if agent_type not in agent_outputs:
                agent_outputs[agent_type] = []
            
            if (output.structured_decision and
                'raw_response_summary' in output.structured_decision and
                output.structured_decision['raw_response_summary']):
                agent_outputs[agent_type].append({
                    'content': output.structured_decision['raw_response_summary'],
                    'timestamp': getattr(output, 'timestamp', None),
                    'raw_output': output  # 保留原始输出以供需要时使用
                })
    except Exception as e:
        logging.warning(f"Error retrieving agent outputs for step {step_num}: {e}")
    
    return agent_outputs


def format_memory_info(step, max_steps=50, outside=True, only_high_level_plan=False):
    """
    格式化前N个step的memory信息，使用step-level的综合总结
    
    功能说明：
    - 优先检索预生成的总结，避免冗余生成
    - 添加当前step中已执行agent的详细记忆
    - 提高检索效率，添加缓存机制，改进信息组织
    - 添加high-level planner的结构化信息（优先使用当前step的，如果没有则使用上一个step的）
    
    Args:
        step: 当前step对象，包含scene信息
            必需字段: 'scene', 'current_step', 'question', 'current_position'
            
        max_steps: int, 默认50
            最大显示的历史step数量。控制"Previous Steps Summary"部分显示多少个历史步骤的总结。
            数值越大，显示的历史信息越多，但可能导致prompt过长。
            
        outside: bool, 默认True
            是否包含历史步骤总结。
            - True: 完整输出，包含"Previous Steps Summary"部分（显示前max_steps个历史步骤）
            - False: 不包含历史步骤总结，只显示当前步骤的信息和high-level plan
            主要用于当前步骤内部agent之间的信息传递，避免重复显示历史信息。
            
        only_high_level_plan: bool, 默认False
            是否只返回high-level plan信息。
            - True: 仅返回"High-Level Plan"部分，忽略当前步骤进度和历史总结
            - False: 返回完整的memory信息（根据outside参数决定是否包含历史）
            适用于只需要规划信息的场景，可以大幅减少prompt长度。
    
    Returns:
        str: 格式化的memory信息字符串，包含以下部分（根据参数控制）：
            1. High-Level Plan: 当前或上一步的任务规划列表（如果存在）
            2. Current Step Progress: 当前步骤中各agent的执行情况（仅当only_high_level_plan=False时）
            3. Previous Steps Summary: 历史步骤总结（仅当outside=True且only_high_level_plan=False时）
            
        如果没有可用信息，返回相应的提示信息。
    """
    if 'scene' not in step or step['scene'] is None:
        return "No scene information available for memory retrieval.\n"

    try:
        memory_info = []
        current_step = step.get('current_step', 0)
        question = step.get('question', '')

        # 定义agent执行顺序
        # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
        agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]

        # 1. 检索预生成的step总结（对于之前的steps和当前step如果已有）
        step_summaries = {}
        try:
            # 优化：一次检索所有需要的step_summary_output，减少多次调用
            summary_outputs = step['scene'].long_term_memory.retrieve_by_type("step_summary_output", top_k=50)
            if summary_outputs:
                for output in summary_outputs:
                    if (output.structured_decision and
                        'raw_response_summary' in output.structured_decision and
                        output.structured_decision['raw_response_summary']):
                        step_summaries[output.step] = output.structured_decision['raw_response_summary']
        except Exception as e:
            logging.warning(f"Error retrieving step summaries: {e}")

        # 2. 检索high-level planner的输出
        high_level_plan_info = None
        try:
            # 首先检查当前step是否有high-level plan
            current_planner_outputs = step['scene'].long_term_memory.retrieve_by_step_and_type(current_step, "high_level_planner_output")
            if current_planner_outputs:
                # 使用最新的high-level plan
                latest_planner_output = current_planner_outputs[-1]
                if (latest_planner_output.structured_decision and
                    'todo_list' in latest_planner_output.structured_decision):
                    high_level_plan_info = {
                        'step': current_step,
                        'todo_list': latest_planner_output.structured_decision['todo_list'],
                        'raw_output': latest_planner_output.structured_decision.get('raw_response', '')
                    }
            else:
                # 如果当前step没有，检查上一个step
                if current_step > 0:
                    previous_planner_outputs = step['scene'].long_term_memory.retrieve_by_step_and_type(current_step - 1, "high_level_planner_output")
                    if previous_planner_outputs:
                        # 使用上一个step的最新high-level plan
                        latest_planner_output = previous_planner_outputs[-1]
                        if (latest_planner_output.structured_decision and
                            'todo_list' in latest_planner_output.structured_decision):
                            high_level_plan_info = {
                                'step': current_step - 1,
                                'todo_list': latest_planner_output.structured_decision['todo_list'],
                                'raw_output': latest_planner_output.structured_decision.get('raw_response', '')
                            }
        except Exception as e:
            logging.warning(f"Error retrieving high-level planner info: {e}")

        # 3. 优化：使用专门的函数检索当前step的所有agent输出
        current_step_agents = {}
        try:
            # 使用优化的记忆检索函数
            agent_outputs_by_type = get_agent_outputs_by_step_and_type(step, current_step, agent_execution_order)
            
            # 将每个agent类型的最新输出添加到current_step_agents
            for agent_type in agent_execution_order:
                if agent_type in agent_outputs_by_type and agent_outputs_by_type[agent_type]:
                    # 使用最新的输出
                    latest_output = agent_outputs_by_type[agent_type][-1]
                    current_step_agents[agent_type] = latest_output
        except Exception as e:
            logging.warning(f"Error retrieving current step agents: {e}")

        # 4. 格式化输出 - 优化信息组织，使最重要的信息更突出
        if not step_summaries and not current_step_agents and not high_level_plan_info:
            return "No memory available.\n"

        # 优先显示high-level plan信息，如果存在
        if high_level_plan_info:
            memory_info.append("High-Level Plan:\n")
            memory_info.append(f"- Plan from Step {high_level_plan_info['step']}:\n")
            for task in high_level_plan_info['todo_list']:
                status = task.get('status', 'unknown')
                task_desc = task.get('task', '')
                memory_info.append(f" * [{status}] {task_desc}\n")
            memory_info.append("\n")
        
        # 如果只需要high-level plan信息，直接返回
        if only_high_level_plan:
            if high_level_plan_info:
                return "".join(memory_info)
            else:
                return "No high-level plan available.\n"

        # 然后显示当前step的agent信息，因为这是最相关的
        if current_step_agents:
            memory_info.append("Current Step Progress:\n")
            for agent_type in agent_execution_order:
                if agent_type in current_step_agents:
                    agent_info = current_step_agents[agent_type]
                    agent_name = agent_type.replace('_', ' ').title()
                    memory_info.append(f"- {agent_name}: {agent_info['content']}\n")
            memory_info.append("\n")

        # 然后显示之前的steps（使用预生成总结）
        if outside:
            memory_info.append("Previous Steps Summary:\n")
            previous_steps = sorted([s for s in step_summaries.keys() if s < current_step], reverse=True)
            recent_steps = previous_steps[:max_steps]
            if recent_steps:
                for step_num in recent_steps:
                    summary = step_summaries[step_num]
                    memory_info.append(f"Step {step_num}: {summary}\n\n")
            else:
                memory_info.append("(No previous step summaries available)\n\n")

        return "".join(memory_info)

    except Exception as e:
        logging.error(f"Error in format_memory_info: {str(e)}")
        return f"Error retrieving memory information: {str(e)}\n"





def explore_step(step, cfg, verbose=False):
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        snapshot_id_mapping,
    ) = get_step_info(step, verbose)
    ##################################################
    ### -1  Snapshot Manager: manage the snapshots ###
    ##################################################
    snapshot_select_id = list(range(len(snapshot_imgs)))
    numbers_int = None  
    if len(snapshot_imgs) > 3:
        sys_prompt, content = format_manage_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )

        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"MANAGER: {full_response}")
        snapshot_select_id = parse_retain_snapshots_response(full_response)
        if snapshot_select_id:
            snapshot_select_id = [i for i in snapshot_select_id if 0 <= i < len(snapshot_imgs)]
            if not snapshot_select_id:
                logging.warning(f"All parsed snapshot indices were out of bounds, keeping no snapshots")
        else:
            logging.info(f"No snapshots selected from response (empty result), keeping no snapshots")
        
        logging.info(f"Snapshot selection parsed: keep {snapshot_select_id} out of {len(snapshot_imgs)}")
                
        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(full_response or "", "snapshot_manager", question)

                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)

                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"

                    actual_step = step.get('current_step', 0)

                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="snapshot_manager",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                
                    all_snapshot_outputs = step['scene'].long_term_memory.retrieve_by_type("snapshot_manager_output", top_k=50)
                    if all_snapshot_outputs and len(all_snapshot_outputs) > 1: 
                        logging.info(f"Snapshot Manager History (All Steps):")
                        for output in all_snapshot_outputs[-5:]:  
                            logging.info(f" - Step {output.step}: {output.content[:100]}...") 
                                
                    logging.info(f"=== End Snapshot Manager Update ===")
            except Exception as e:
                logging.error(f"Error recording snapshot manager decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for snapshot manager. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
        numbers_int = set(range(len(snapshot_imgs))) - set(snapshot_select_id)

        logging.info(numbers_int)
        try:
            filtered_imgs = [img for i, img in enumerate(snapshot_imgs) if i not in numbers_int]
            filtered_cls = [cls for i, cls in enumerate(snapshot_classes) if i not in numbers_int]
            snapshot_id_mapping = [snapshot_id_mapping[i] for i in snapshot_select_id]
            snapshot_imgs = filtered_imgs  
            snapshot_classes = filtered_cls
        except Exception as e:
            logging.info(f"Error in exclude snapshots: {numbers_int}")
            logging.info(e)
    else:
        logging.info("could not find {}")
        # filtered_imgs = []
        # filtered_cls = []
        logging.info('there is no useful snapshots')
        full_response = 'continue exploration'


    ##################################################
    ### 0  Frontier Manager: manage the frontiers ###
    ##################################################
    frontier_select_id = list(range(len(frontier_imgs)))
    if len(frontier_imgs) > 1:
        sys_prompt, content = format_plan_manager_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"PLAN MANAGER: {full_response}")
        
        if full_response is None:
            logging.info("VLM response is None, using default frontier selection")
            frontier_select_id = list(range(len(frontier_imgs)))
        else:
            frontier_select_id = parse_retain_frontiers_response(full_response)
            if frontier_select_id:
                frontier_select_id = [i for i in frontier_select_id if 0 <= i < len(frontier_imgs)]
                if not frontier_select_id:
                    frontier_select_id = list(range(len(frontier_imgs)))
                    logging.warning(f"All parsed frontier indices were out of bounds, falling back to keep all {len(frontier_imgs)} frontiers")
            else:
                frontier_select_id = list(range(len(frontier_imgs)))
                logging.warning(f"Failed to parse frontier selection from response or empty result, falling back to keep all {len(frontier_imgs)} frontiers")
                    
            if 'scene' in step and step['scene'] is not None:
                try:
                    structured_output = extract_structured_output_from_response(full_response or "", "frontier_manager", question)
                    
                    current_position = step.get('current_position', [0, 0])
                    position = np.array(current_position)
                    
                    if hasattr(step['scene'], 'text_memory_system'):
                        frontier_descriptions = {}
                        for i in range(len(frontier_imgs)):
                            frontier_descriptions[i] = f"Frontier {i} at step {step.get('current_step', 0)}"
                        
                        actual_step = step.get('current_step', 0)
                        
                        step['scene'].text_memory_system.record_structured_agent_output(
                            step=actual_step,
                            agent_type="frontier_manager",
                            structured_output=structured_output,
                            raw_response=full_response or "",
                            position=position
                        )
                        
                        all_frontier_outputs = step['scene'].long_term_memory.retrieve_by_type("frontier_manager_output", top_k=50)
                        if all_frontier_outputs and len(all_frontier_outputs) > 1:  
                            logging.info(f"Frontier Manager History (All Steps):")
                            for output in all_frontier_outputs[-5:]:  
                                logging.info(f" - Step {output.step}: {output.content[:100]}...") 
                                    
                        logging.info(f"=== End Frontier Manager Update ===")
                except Exception as e:
                    logging.error(f"Error recording frontier manager decision to memory: {e}")
            else:
                logging.warning(f"Step does not have scene attribute, skipping memory recording for frontier manager. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")

        # TODO if retain
        frontier_out_id = set(range(len(frontier_imgs))) - set(frontier_select_id)
        logging.info(frontier_select_id)
        logging.info(frontier_out_id)
        try:
            filtered_imgs = [img for i, img in enumerate(frontier_imgs) if i not in frontier_out_id]
            frontier_imgs = filtered_imgs  # TODO
    
        except Exception as e:
            logging.info(f"Error in exclude frontier: {frontier_out_id}")
            logging.info(e)



    ##################################################
    ### 1  Answerer: check enough info to answer? ###
    ##################################################
    if snapshot_imgs:
        sys_prompt, content = format_answer_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,  
            snapshot_classes,  
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )

        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        full_response = call_openai_api(sys_prompt, content)
        logging.info(f"ANSWERER: {full_response}")
        
        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(full_response or "", "answerer", question)
                
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                
                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"
                    
                    actual_step = step.get('current_step', 0)
                    
                    # 记录结构化输出，这会自动处理内部逻辑
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="answerer",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                
                    all_answerer_outputs = step['scene'].long_term_memory.retrieve_by_type("answerer_output", top_k=50)
                    if all_answerer_outputs and len(all_answerer_outputs) > 1: 
                        logging.info(f"Answerer History (All Steps):")
                        for output in all_answerer_outputs[-5:]:  
                            logging.info(f" - Step {output.step}: {output.content[:100]}...") 

                    logging.info(f"=== End Answerer Update ===")
            except Exception as e:
                logging.error(f"Error recording answerer decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for answerer. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
    else:
        logging.info('there is no useful snapshots')
        full_response = 'continue exploration'

    ##################################################
    ### 2  Planner: if [can not answer] & [frontier_imgs] is not none --> continue explore ###
    ##################################################
    if 'continue exploration' in safe_strip(full_response).lower():
        logging.info('###### high-level plan ######')
        sys_prompt, content = format_high_level_plan_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        full_response = call_openai_api_text(sys_prompt, content)
        full_response = safe_strip(full_response)
        logging.info(f"{full_response}")
        
        try:
            todo_list = extract_predictive_plan(full_response)
            if todo_list:
                logging.info(f"Extracted {len(todo_list)} tasks from high-level planner:")
                for i, task in enumerate(todo_list):
                    status = task.get('status', 'unknown')
                    task_desc = task.get('task', '')
                    logging.info(f"  {i+1}. [{status}] {task_desc}")
                
                if 'scene' in step and step['scene'] is not None and hasattr(step['scene'], 'text_memory_system'):
                    current_position = step.get('current_position', [0, 0])
                    position = np.array(current_position)
                    actual_step = step.get('current_step', 0)
                    
                    structured_output = {
                        "raw_response": full_response,
                        "reasoning": f"High-level plan with {len(todo_list)} tasks extracted",
                        "response_type": "high_level_plan",
                        "structured_output": {
                            "todo_list": todo_list
                        }
                    }
                    
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="high_level_planner",
                        structured_output=structured_output,
                        raw_response=full_response,
                        position=position
                    )
                    logging.info(f"High-level plan with todo list recorded to memory for step {actual_step}")
        except Exception as e:
            logging.error(f"Error extracting todo list from high-level planner output: {e}")
        
        logging.info('###### continue exploration ######')
        sys_prompt, content = format_explore_prompt(
            question,
            egocentric_imgs,
            frontier_imgs,
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            step=step,
        )
  
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        full_response = call_openai_api(sys_prompt, content)
        full_response = safe_strip(full_response)
        if full_response is None:
            response = "stop exploration"
            reason = "No response from model"
        else:
            lines = full_response.strip().split('\n')
            answer_pattern = re.compile(
                r'(?:next\s+step\s*:\s*frontier\s+(\d+)|stop\s+exploration)',
                re.IGNORECASE
            )
            
            answer_line_index = None
            answer_match = None
            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                match = answer_pattern.search(line)  
                if match:
                    answer_line_index = i
                    answer_match = match
                    break

            if answer_match:
                if answer_match.group(1):  # Frontier case
                    frontier_index = answer_match.group(1)
                    response = f"frontier {frontier_index}"
                else:  # Stop Exploration case
                    response = "stop exploration"
                reason = '\n'.join(lines[:answer_line_index]).strip() or "No reasoning provided."
            else:
                found_response = False
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if 'next step: frontier' in line_lower:
                        # 提取frontier后的数字
                        frontier_match = re.search(r'frontier\s+(\d+)', line_lower)
                        if frontier_match:
                            response = f"frontier {frontier_match.group(1)}"
                            reason = '\n'.join(lines[:i]).strip() or "No reasoning provided."
                            found_response = True
                            break
                    elif 'stop exploration' in line_lower:
                        response = "stop exploration"
                        reason = '\n'.join(lines[:i]).strip() or "No reasoning provided."
                        found_response = True
                        break
                
                if not found_response:
                    response = "stop exploration"
                    reason = full_response  # 或记录错误

            response = response.lower().strip()

        if 'scene' in step and step['scene'] is not None:
            try:
                structured_output = extract_structured_output_from_response(
                    full_response or "", "planner", question,
                    parsed_action=response, parsed_reason=reason,
                )
                
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                
                if hasattr(step['scene'], 'text_memory_system'):
                    snapshot_descriptions = {}
                    for i in range(len(snapshot_imgs)):
                        snapshot_descriptions[f"snapshot_{i}"] = f"Snapshot {i} from step {step.get('current_step', 0)}"
                    
                    frontier_descriptions = {}
                    for i in range(len(frontier_imgs)):
                        frontier_descriptions[i] = f"Frontier {i} at step {step.get('current_step', 0)}"
                    
                    actual_step = step.get('current_step', 0)
                    
                    step['scene'].text_memory_system.record_structured_agent_output(
                        step=actual_step,
                        agent_type="planner",
                        structured_output=structured_output,
                        raw_response=full_response or "",
                        position=position
                    )
                    
                    all_planner_outputs = step['scene'].long_term_memory.retrieve_by_type("planner_output", top_k=50)
                    if all_planner_outputs and len(all_planner_outputs) > 1:  
                        logging.info(f"Planner History (All Steps):")
                        for output in all_planner_outputs[-5:]: 
                            logging.info(f"  - Step {output.step}: {output.content[:100]}...")  # 限制长度
                    
                    logging.info(f"=== End Planner Update ===")
            except Exception as e:
                logging.error(f"Error recording planner decision to memory: {e}")
        else:
            logging.warning(f"Step does not have scene attribute, skipping memory recording for planner. Scene in step: {'scene' in step}, scene value: {step.get('scene', 'Not found')}")
        
        if 'scene' in step and step['scene'] is not None:
            try:
                current_position = step.get('current_position', [0, 0])
                position = np.array(current_position)
                actual_step = step.get('current_step', 0)
                question = step.get('question', '')

                # 检查是否已经记录过这个step的总结，避免重复记录
                existing_summaries = step['scene'].long_term_memory.retrieve_by_type("step_summary_output", top_k=10)
                already_recorded = any(output.step == actual_step for output in existing_summaries)

                if already_recorded:
                    logging.info(f"Step {actual_step} summary already recorded, skipping...")
                else:
                    # 收集当前step所有agent的输出
                    step_agents = []
                    # agent_execution_order = ["frontier_manager", "snapshot_manager", "answerer", "planner", "forced_answerer"]
                    agent_execution_order = ["snapshot_manager", "frontier_manager", "answerer", "planner", "forced_answerer"]

                    try:
                        agent_outputs_by_type = get_agent_outputs_by_step_and_type(step, actual_step, agent_execution_order)
                        
                        # 将所有agent的输出添加到step_agents列表
                        for agent_type in agent_execution_order:
                            if agent_type in agent_outputs_by_type:
                                for output in agent_outputs_by_type[agent_type]:
                                    step_agents.append({
                                        'agent_type': agent_type,
                                        'content': output['content']
                                    })
                    except Exception as e:
                        logging.warning(f"Error retrieving {agent_type} outputs for step {actual_step}: {e}")

                    if step_agents:
                        logging.info(f"Generating step summary for step {actual_step} with {len(step_agents)} agent outputs")

                        # 生成step总结
                        step_summary = generate_step_summary(actual_step, step_agents, question)

                        if step_summary and len(step_summary.strip()) > 10:  # 确保总结不为空且有意义
                            # 记录到长期记忆
                            structured_output = {
                                "raw_response": step_summary,
                                "reasoning": step_summary,
                                "response_type": "step_summary"
                            }

                            step['scene'].text_memory_system.record_structured_agent_output(
                                step=actual_step,
                                agent_type="step_summary",
                                structured_output=structured_output,
                                raw_response=step_summary,
                                position=position
                            )

                            logging.info(f"Successfully recorded step {actual_step} summary: {step_summary[:100]}...")
                        else:
                            logging.warning(f"Generated step summary is too short or empty for step {actual_step}")
                    else:
                        logging.info(f"No agent outputs found for step {actual_step}, skipping summary generation")

            except Exception as e:
                logging.error(f"Error generating and recording step summary for step {step.get('current_step', 'unknown')}: {e}")


            logging.info(f"response: {response}")
            logging.info(f"reason: {reason}")

            try:
                response_parts = response.split(" ")
                choice_type = response_parts[0]
                choice_id = response_parts[1] if len(response_parts) > 1 else None
                response_valid = False

                if (
                        choice_type == "frontier"
                        and choice_id is not None
                        and choice_id.isdigit()
                        and 0 <= int(choice_id) < len(frontier_imgs)
                ):
                    try:
                        choice_id = frontier_select_id[int(choice_id)]
                        response = choice_type + ' ' + str(choice_id)
                        response_valid = True
                    except (ValueError, IndexError):
                        logging.info(f"Error in frontier selection: choice_id={choice_id}, frontier_select_id={frontier_select_id}")
                        response_valid = False

                if 'stop' in response.lower():
                # if 'stop' in reason.lower() or 'stop' in response.lower():
                    logging.info(f"######  stop exploring  #######")
                    response_valid = False

                if response_valid:
                    final_response = response
                    final_reason = reason
                    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs), numbers_int if numbers_int else None

            except Exception as e:
                logging.info(f"Error in splitting response: {response}")
                logging.info(e)
                if 'stop exploration' in response.lower():
                    response_valid = False
                else:
                    try:
                        response_parts = response.split(" ")
                        if len(response_parts) >= 2 and response_parts[0] == "frontier":
                            choice_type = response_parts[0]
                            choice_id = response_parts[1]
                            if choice_id.isdigit() and 0 <= int(choice_id) < len(frontier_imgs):
                                choice_id = frontier_select_id[int(choice_id)]
                                response = choice_type + ' ' + str(choice_id)
                                response_valid = True
                    except (ValueError, IndexError):
                        response_valid = False

    ##################################################
    ### 3  Forced Answerer: if [no need to explore] or [no frontier_imgs to explore]
    ##################################################
    logging.info(f"######  force a answer  #######")
    sys_prompt, content = format_force_answer_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs, # filtered_imgs,  # snapshot_imgs,
        snapshot_classes, # filtered_cls,  # snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        step=step,
    )

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    full_response = call_openai_api(sys_prompt, content)
    full_response = safe_strip(full_response)
    if "\n" in full_response:
        full_response_list = full_response.split("\n")
        response, reason = full_response_list[0], full_response_list[-1]
        response, reason = safe_strip(response), safe_strip(reason)
    else:
        response = full_response
        reason = ""
    response = response.lower()

    logging.info(f"response: {response}")
    logging.info(f"reason: {reason}")

    if 'stop' in safe_strip(response).lower() and len(response.split(" ")) > 1:
        try:
            choice_type, choice_id = response.split(" ")
            choice_id = snapshot_select_id[int(choice_id)]
        except (ValueError, IndexError):
            logging.info(f"Error in snapshot selection: response={response}, snapshot_select_id={snapshot_select_id}")
            choice_type = "stop"
            choice_id = ""
        response = choice_type + ' ' + str(choice_id)

    final_response = response
    final_reason = reason

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs), numbers_int if numbers_int else None

