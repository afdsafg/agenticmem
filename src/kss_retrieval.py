"""
KSS (Key Subgraph Selection) retrieval module.

Ported from MSGNav src/explore_utils.py. Logic preserved verbatim;
only import sources adapted for Pred-EQA:
- call_openai_api / format_content -> from src.pred_eqa (Pred-EQA cloud API)
- encode_tensor2base64 / resize_image -> inlined here (from MSGNav utils.py / explore_utils.py)

Exposes:
  - format_prefiltering_prompt
  - get_prefiltering_objs
  - related_object_KSS
  - edge_pruning_KSS
  - Key_Subgraph_Selection  (optional; kept for parity, uses format_question inlined)
"""

import base64
import heapq
import logging
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

# Pred-EQA cloud VLM API (same signature as MSGNav's call_openai_api)
from src.pred_eqa import call_openai_api, format_content


# ---------------------------------------------------------------------------
# Image helpers (ported verbatim from MSGNav src/explore_utils.py / src/utils.py)
# ---------------------------------------------------------------------------
def encode_tensor2base64(img, min_size=16):
    """Ported from MSGNav explore_utils.py L133."""
    if min_size is not None:
        if (type(img) == np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        if min(width, height) < min_size:
            scale = min_size / min(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def resize_image(image, target_h, target_w):
    """Ported from MSGNav src/utils.py L80."""
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


# ---------------------------------------------------------------------------
# format_question (inlined from MSGNav explore_utils.py L149) for Key_Subgraph_Selection
# ---------------------------------------------------------------------------
def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")
    return question, image_goal


# ---------------------------------------------------------------------------
# KSS core functions (ported verbatim from MSGNav explore_utils.py L588-780)
# ---------------------------------------------------------------------------
def format_prefiltering_prompt(question, scene_graph, top_k=10, image_goal=None, room_label=False):
    """Ported from MSGNav explore_utils.py L588."""
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"

    prompt = """To efficiently solve the problem,  you should identify key objects that are most helpful for guiding exploration toward the target.
Please follow these strict instructions:
1. Read and understand the full 3D scene graph. Each object includes its id, class, room, and nearby objects (i.e., its neighbors in the graph).
2. Rank objects by how helpful they are for locating the target, based on:
  Semantic relevance to the target; Co-occurrence with the target in typical environments; Presence in the same room as the target.
3. Choose only the most informative and strategically diverse objects for exploration. To maximize coverage: Avoid choosing objects that are directly connected (i.e., neighbors) in the scene graph."""
    content.append((prompt,))
    # ------------------format an 3D scene graph-------------------------
    prompt = "Here is is the format for input 3D scene graph:\n"
    prompt += "Object ID: Class"
    if room_label:
        prompt += ", Located room"
    prompt += ", nearby objects ID\n"
    content.append((prompt,))

    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects in 3D scene graph:\n"
    prompt += "Question: \nWhat can I use to watch my favorite shows and movies?\n"
    if not room_label:
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "1: tv, (1.46, 1.71, 1.00), [2, 3]\n"
        prompt += "2: speaker, (3.36, 1.42, -1.10), [1, 3]\n"
        prompt += "3: sofa, (0.48, 1.91, 2.59), [1, 2]\n"
        prompt += "4: bed, (2.42, 1.04, 3.89), [5]\n"
        prompt += "5: lamp, (1.15, 1.66, 1.37), [4]\n"
        prompt += "6: box, (0.30, 1.36, 0.41), [7]\n"
        prompt += "7: cabinet, (2.01, 1.52, 2.08), [6]\n"
        prompt += "Answer:\n1\n5\n"
    else:
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "1: tv, living room, [2, 3]\n"
        prompt += "2: speaker, living room, [1, 3]\n"
        prompt += "3: sofa, living room, [1, 2]\n"
        prompt += "4: bed, bedroom, [5]\n"
        prompt += "5: lamp, bedroom, [4]\n"
        prompt += "6: box, kitchen, [7]\n"
        prompt += "7: cabinet, kitchen, [6]\n"
        prompt += "Answer:\n1\n5\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful key objects in order.\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is the 3D scene graph based on the above input format\n"
    )
    for id in scene_graph.keys():
        obj = scene_graph[id]
        prompt += f"{obj['id']}: {obj['class']}"
        if room_label:
            prompt += f", {obj['room']}"
        prompt += f", [{', '.join(map(str, obj['related_objects_id']))}]\n"
    if len(scene_graph) == 0:
        prompt += "    No items in the 3D scene graph.\n"
    prompt += f"Do not print any object that are not included in the 3D scene graph or include any additional information other than the ID in your response:\n"
    prompt += "Answer: \n"

    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_objs(question, obj_infos, top_k=10, image_goal=None, use_room_filter=False):
    """Ported from MSGNav explore_utils.py L666."""
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, obj_infos, top_k=top_k, image_goal=image_goal, room_label=use_room_filter
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image [{c[1][:10]}...]"

    response = call_openai_api(prefiltering_sys, prefiltering_content)
    logging.info(message)
    logging.info(response)
    if response is None:
        return []
    # parse the response and return the top_k objects
    obj_id_set = set(obj_infos.keys())
    selected_objs = response.strip().split("\n")
    selected_objs = [int(id.strip()) for id in selected_objs if id.strip().isdigit()]
    selected_objs = [id for id in selected_objs if id in obj_id_set]
    selected_objs = selected_objs[:top_k]
    return selected_objs


def related_object_KSS(
    question,
    objs,
    edges,
    top_k=10,
    image_goal=None,
    verbose=False,
    use_ollama=False,  # ponytail: kept for signature parity; Pred-EQA uses cloud API only
    use_room_filter=False,
):
    """Ported from MSGNav explore_utils.py L693.

    Returns: list[int] of selected object ids.
    """
    obj_infos = {}
    for obj_id in objs.keys():
        obj_infos[obj_id] = {
            "id": obj_id,
            "pos": objs[obj_id]["bbox"].center,
            "class": objs[obj_id]["class_name"],
            "room": objs[obj_id]["room_label"],
            "related_objects_id": [],
        }
    for node in edges.keys():
        obj_infos[node[0]]["related_objects_id"].append(node[1])
    selected_objs = get_prefiltering_objs(
        question, obj_infos, top_k, image_goal, use_room_filter
    )
    if verbose:
        logging.info(f"Prefiltering selected objects: {selected_objs}")

    return selected_objs


def edge_pruning_KSS(edges, objs, images, selected_obj_id, image_to_edges, prompt_h, prompt_w):
    """Ported from MSGNav explore_utils.py L724.

    Returns: (connected_objs: dict, selected_edges: dict, processed_images: dict)
    """
    selected_objs = {obj_id: objs[obj_id] for obj_id in selected_obj_id}
    connected_objs = {}
    connected_objs.update(selected_objs)
    for node in edges.keys():  # node=(<object1>,<object2>), a tuple
        if node[0] in selected_objs and node[1] not in selected_objs:
            connected_objs.update({node[1]: objs[node[1]]})
        elif node[1] in selected_objs and node[0] not in selected_objs:
            connected_objs.update({node[0]: objs[node[0]]})  # drag in nodes not added yet

    processed_images = {}
    selected_edges = {}

    if len(selected_objs) == 0:
        logging.info("No selected objects after prefiltering, returning empty dicts")
        return {}, {}, {}
    for a_obj_id in sorted(list(selected_objs.keys())):
        for b_obj_id in sorted(list(connected_objs.keys())):
            if (a_obj_id, b_obj_id) in edges and (b_obj_id, a_obj_id) not in selected_edges:
                selected_edges[(a_obj_id, b_obj_id)] = edges[(a_obj_id, b_obj_id)]
    selected_image_to_edges = {}
    for img in image_to_edges.keys():
        selected_image_to_edges[img] = list(set(image_to_edges[img]) & set(selected_edges.keys()))
    uncovered = {e: True for e in list(selected_edges.keys())}
    uncovered_cnt = len(uncovered)
    gain = {img: len(edges) for img, edges in selected_image_to_edges.items()}
    order = {img: i for i, img in enumerate(sorted(selected_image_to_edges.keys(), key=lambda x: str(x)))}
    heap = [(-gain[img], order[img], img) for img in selected_image_to_edges]
    heapq.heapify(heap)
    while uncovered_cnt > 0 and heap:  # pseudocode: while Uncovered edges(U) =/= Empty
        neg_g, _, img = heapq.heappop(heap)
        g = -neg_g
        if g != gain.get(img, 0):
            continue
        if g <= 0:
            logging.info("Error in Greedy Image Allocation!!!")
            break

        image = images[img]
        resized_rgb = resize_image(
            image, prompt_h, prompt_w
        )
        processed_images[img] = encode_tensor2base64(resized_rgb)
        for e in list(selected_image_to_edges[img]):
            if uncovered[e]:
                uncovered[e] = False
                uncovered_cnt -= 1

                for other in selected_edges[e].rel_img:
                    if other == img:
                        continue
                    if gain.get(other, 0) > 0:
                        gain[other] -= 1

                        heapq.heappush(heap, (-gain[other], order[other], other))

    return connected_objs, selected_edges, processed_images


# ---------------------------------------------------------------------------
# Key_Subgraph_Selection (ported from MSGNav explore_utils.py L157)
# Kept for top-level parity; not required by tests.
# ---------------------------------------------------------------------------
def Key_Subgraph_Selection(step, verbose=False, use_ollama=False, use_room_filter=False):
    """Ported from MSGNav explore_utils.py L157."""
    question, image_goal = format_question(step)

    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    objs = step['objects']
    edges = step['edges']
    images = step['all_imgs']
    prompt_h = step['prompt_h']
    prompt_w = step['prompt_w']
    image_to_edges = step['image_to_edges']
    selected_obj_id = related_object_KSS(
        question,
        objs,
        edges,
        step["top_k_categories"],
        image_goal,
        verbose=verbose,
        use_ollama=use_ollama,
        use_room_filter=use_room_filter,
    )

    selected_objs, selected_edges, processed_images = edge_pruning_KSS(
        edges, objs, images, selected_obj_id, image_to_edges, prompt_h, prompt_w
    )

    return (
        question,
        image_goal,
        egocentric_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        frontier_imgs,
    )
