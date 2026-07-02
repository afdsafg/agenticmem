"""GOAT-Bench KSS retrieval entry + exploration hint generation.

P4 of the MSGNav + Pred-EQA fusion plan.
"""

import logging
from typing import Optional

from src.kss_retrieval import Key_Subgraph_Selection
from src.pred_eqa import call_openai_api
from src.goal_types import GoalInfo


def kss_retrieve(scene, goal_info: GoalInfo, cfg) -> tuple:
    """KSS retrieval entry point.

    Returns: (hit: bool, target_obj: Optional[dict], hint: Optional[str])
    """
    if len(scene.objects) == 0:
        # 无记忆直接探索
        return False, None, None

    step = {
        "question": goal_info.text,
        "task_type": goal_info.type,  # image 子任务时 format_question 会读 step["image"]
        "image": goal_info.image_goal,  # ponytail: image 子任务参考图路径; format_question 会 open() 它
        "objects": scene.objects,
        "edges": scene.edges,
        "all_imgs": scene.all_observations,
        "image_to_edges": scene.img_to_edge,
        "top_k_categories": cfg.top_k,
        "prompt_h": cfg.prompt_img_size[0],
        "prompt_w": cfg.prompt_img_size[1],
        "frontier_imgs": [],  # ponytail: KSS 检索阶段无 frontier 图像
        "use_egocentric_views": False,
    }

    # Key_Subgraph_Selection 返回 7-tuple; 第 4 个元素是 selected_objs (dict)
    result = Key_Subgraph_Selection(step, use_room_filter=cfg.use_room_det)
    selected_objs = result[3]

    if len(selected_objs) == 0:
        hint = generate_exploration_hint(scene, goal_info, cfg, selected_objs=None)
        return False, None, hint

    target_obj = find_target_in_selected(selected_objs, goal_info, scene)
    if target_obj is not None:
        return True, target_obj, None

    hint = generate_exploration_hint(scene, goal_info, cfg, selected_objs=selected_objs)
    return False, None, hint


def find_target_in_selected(selected_objs, goal_info: GoalInfo, scene) -> Optional[dict]:
    """在 KSS 选出的对象中找目标。

    object 子任务: 精确类别匹配。
    description/image 子任务: KSS 阶段不精确匹配,返回 None 走探索路径。
    """
    if goal_info.type == "object":
        target_class = (goal_info.category or "").lower()
        for obj_id, obj in selected_objs.items():
            class_name = obj.get("class_name", "") if isinstance(obj, dict) else getattr(obj, "class_name", "")
            if class_name.lower() == target_class:
                return obj
    return None


def generate_exploration_hint(
    scene,
    goal_info: GoalInfo,
    cfg,
    selected_objs=None,
) -> str:
    """LLM 读场景图生成探索摘要。

    selected_objs 非空时,prompt 里标注哪些对象已被 KSS 选中但不完全是目标。
    """
    # 序列化场景图: {id}: {class_name}, {room_label}, [neighbor_ids]
    graph_lines = []
    for obj_id, obj in scene.objects.items():
        class_name = obj.get("class_name", "") if isinstance(obj, dict) else getattr(obj, "class_name", "")
        room_label = obj.get("room_label", "unknown") if isinstance(obj, dict) else getattr(obj, "room_label", "unknown")
        # 邻居: 从 edges 里找以 obj_id 为起点的边
        neighbors = []
        for (a, b) in scene.edges.keys():
            if a == obj_id and b != obj_id:
                neighbors.append(str(b))
        graph_lines.append(f"{obj_id}: {class_name}, {room_label}, [{','.join(neighbors)}]")
    graph_text = "\n".join(graph_lines)

    # 目标描述
    if goal_info.type == "object":
        target_desc = f"a {goal_info.category}"
    elif goal_info.type == "description":
        target_desc = goal_info.lang_desc or "the described object"
    else:  # image
        target_desc = "the object shown in the reference image"

    selected_note = ""
    if selected_objs:
        selected_ids = list(selected_objs.keys()) if hasattr(selected_objs, "keys") else list(selected_objs)
        selected_note = (
            f"\nNote: KSS selected objects {selected_ids} as potentially relevant "
            f"but none is an exact match for the target. Use this as a starting point.\n"
        )

    sys_prompt = "You are an AI agent exploring a 3D indoor scene for navigation."
    user_prompt = (
        f"Target: {target_desc}\n"
        f"Already explored scene graph:\n"
        f"{graph_text}\n"
        f"{selected_note}"
        f"\nYou have NOT found the target. Generate a brief exploration hint:\n"
        f"1. List explored room types\n"
        f"2. List detected object categories\n"
        f"3. Suggest which unexplored direction is most promising (based on semantic association)\n"
        f"4. Keep under 100 words\n"
        f"\nHint:"
    )

    response = call_openai_api(sys_prompt, [(user_prompt, None)])
    return response if response is not None else ""
