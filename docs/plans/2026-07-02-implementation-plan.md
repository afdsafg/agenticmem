# 实现计划：MSGNav + Pred-EQA 融合 → GOAT-Bench 导航

设计文档: `docs/plans/2026-07-02-msgnav-pred-eqa-fusion-design.md`

## 阶段概览

| 阶段 | 内容 | 依赖 |
|------|------|------|
| P1 | 基础设施移植（KSS + VDD + 工具函数） | 无 |
| P2 | GOAT-Bench 数据加载 + habitat-sim 直跑 runner | P1 |
| P3 | Scene 适配（跨子任务记忆 + 子任务切换钩子） | P2 |
| P4 | 子任务开始：KSS 检索 + 探索摘要生成 | P1, P3 |
| P5 | 探索循环：Pred-EQA planner + VLM 目标发现判定器 | P3, P4 |
| P6 | 导航：VDD 视点决策 + 到达判定 + subtask_stop | P1, P5 |
| P7 | 集成测试 + 冒烟跑通 | P1-P6 |

---

## P1: 基础设施移植

### P1.1 移植 KSS 检索模块 → `src/kss_retrieval.py`

从 MSGNav `src/explore_utils.py` 移植，适配 Pred-EQA 的 `call_openai_api`（已有 cloud API）。

移植函数：
- `format_prefiltering_prompt(question, scene_graph, top_k, image_goal, room_label)` (L588-663)
- `get_prefiltering_objs(question, obj_infos, top_k, image_goal, use_room_filter)` (L666-690)
- `related_object_KSS(question, objs, edges, top_k, image_goal, use_room_filter)` (L693-722)
- `edge_pruning_KSS(edges, objs, images, selected_obj_id, image_to_edges, prompt_h, prompt_w)` (L724-780)
- `Key_Subgraph_Selection(step, ...)` (L157-202) — 入口

适配点：
- `call_openai_api` → 用 Pred-EQA 的 `src/pred_eqa.py:66` 版本（cloud API）
- `objs` 结构兼容：Pred-EQA 检测栈已用 MSGNav 风格 `MapObjectDict`，`class_name`/`room_label`/`bbox.center` 字段一致
- `images` / `image_to_edges` → 从 `scene.all_observations` / `scene.img_to_edge` 取
- `encode_tensor2base64` → 移植 MSGNav `utils.py` 的图像编码

验证：单元测试——构造 mock objects/edges，跑 `related_object_KSS` 返回 obj ID 列表。

### P1.2 移植 VDD 视点决策模块 → `src/vdd.py`

从 MSGNav `src/utils.py` 移植：
- `generate_candidate_viewpoints(bbox_center, radius, pts, num_points=20)` (L9-18)
- `is_point_visible(viewpoint, target_point, scene_points_tree, threshold=0.05)` (L22-37)
- `compute_visibility(viewpoint, target_points, scene_points_tree)` (L39-48)
- `Visibility_based_Viewpoint_Decision(target_points, scene_points, pts, tsdf_planner, radius_factor)` (L51-76)

依赖：`scipy.spatial.KDTree`, `open3d`, `habitat_sim`, `numpy`（3dmem env 已装）。

VDD 还依赖 `tsdf_planner.mask_true_point` / `get_near_true_point`（MSGNav `tsdf_planner.py:806-824`）。Pred-EQA TSDFPlanner 有底层（`self.unoccupied`/`habitat2voxel`/`normal2habitat` 都在），缺这两个方法。一并移植：
- `mask_true_point(viewpoints)` (MSGNav `tsdf_planner.py:806-815`)
- `get_near_true_point(viewpoints)` (MSGNav `tsdf_planner.py:817-824`)
- `get_nearest_true_point(point, bool_map)` (MSGNav `geom.py:69`)

加到 Pred-EQA `src/tsdf_planner.py`（或 `src/vdd.py` 内实现，接收 tsdf_planner 参数）。

### P1.3 移植 `select_navigation_corner` → `src/vdd.py`

从 MSGNav `src/query_vlm.py:39-106` 移植。VDD fallback 用。

验证：mock 点云 + tsdf_planner，跑 VDD 返回视点坐标。

---

## P2: GOAT-Bench 数据加载 + Runner

### P2.1 下载 GOAT-Bench 数据

```bash
# 在服务器或本地
# val split
wget <goat-bench val.json.gz 路径> → data/datasets/goat_bench/v1/val/val.json.gz
# content per scene
wget <content/*.json.gz> → data/datasets/goat_bench/v1/val/content/*.json.gz
```

先确认数据源 URL（goat-bench README 或 habitat 数据下载脚本）。

### P2.2 GOAT-Bench episode 加载器 → `src/goat_dataset_loader.py`

直接读 JSON，不用 habitat dataset 框架：

```python
def load_goat_episodes(split_path: str) -> List[dict]:
    """Load episodes from goat-bench JSON.gz.
    Returns list of episode dicts with:
    - episode_id, scene_id, start_position, start_rotation
    - tasks: List[[category, type, instance_id]]
    - goals: dict keyed by "{scene}_{category}" → List[goal_dict]
    """
```

目标解析（参考 `goat_dataset.py:161-219`）：
- `goal[1]=="object"` → category 文本，goal dict 含 `position`/`view_points`
- `goal[1]=="description"` → goal dict `lang_desc` 字段
- `goal[1]=="image"` → goal dict `image_goals`（相机参数），需 habitat-sim 渲染参考图

### P2.3 GOAT-Bench Runner 骨架 → `src/goat_runner.py`

参考 Pred-EQA `run_aeqa_evaluation_vlm_only.py` + MSGNav `run_goatbench_evaluation.py` 结构：

```python
def main():
    # 1. 加载模型（YOLOWorld/SAM/CLIP，复用 Pred-EQA 加载代码）
    # 2. 加载 episodes
    for episode in episodes:
        # 3. 初始化 Scene + TSDFPlanner（habitat-sim 在 Scene 内）
        scene = Scene(scene_id, cfg, cfg_cg, detection_model=..., ...)
        tsdf_planner = TSDFPlanner(...)
        
        for subtask_idx, (category, goal_type, instance_id) in enumerate(episode["tasks"]):
            # 4. 解析子任务目标
            goal_info = resolve_goal(episode, subtask_idx)  # → text/image/position
            
            # 5. 子任务开始：KSS 检索 (P4)
            kss_result = kss_retrieve(scene, goal_info)
            if kss_result.hit:
                # 直接导航 (P6)
                navigate_to_object(kss_result.target_obj, ...)
            else:
                # 生成探索摘要 (P4)
                hint = generate_exploration_hint(scene, goal_info)
                # 探索循环 (P5)
                run_exploration_loop(scene, tsdf_planner, goal_info, hint)
            
            # 6. 成功判定：测地线距离 < 1m
            success = check_success(pts, goal_info.viewpoints, scene.pathfinder)
        
        scene.close()
```

### P2.4 配置文件 → `cfg/eval_goat.yaml`

基于 `cfg/eval_pred_eqa.yaml` 改：
- `questions_list_path` → `goat_data_path`
- 加 `success_distance: 1.0`
- 加 `dicision_radius`（VDD 参数，参考 MSGNav cfg）
- 加 `clear_up_memory_every_subtask: false`
- 保留检测栈配置（yolo/sam/clip/room_det/edge_dist_threshold）

验证：能加载 1 个 episode，打印 subtask 列表，habitat-sim 能加载场景。

---

## P3: Scene 适配

### P3.1 跨子任务记忆保留接口

`src/scene_vlm_only.py` 加方法：
```python
def reset_for_new_subtask(self):
    """Clear per-subtask state, keep objects/edges/img_to_edge (cross-subtask memory)."""
    self.snapshots = {}
    self.frames = {}
    self.all_observations = {}
    self.filtered_snapshots = set()
    # 长期记忆（planner 输出）每子任务重置
    self.text_memory_system = SceneIntegration(self)
    self.long_term_memory = []
    # objects/edges/img_to_edge 保留 — 跨子任务记忆
```

参考 MSGNav `clear_up_detections`（L174-184）但**不删 objects/edges**。

### P3.2 子任务切换检测

Runner 层监听 `subtask_idx` 变化 → 调 `scene.reset_for_new_subtask()`。

验证：跑 2 个子任务，确认 objects/edges 保留，snapshots/frames 清空。

---

## P4: 子任务开始 — KSS 检索 + 探索摘要

### P4.1 KSS 检索入口 → `src/goat_runner.py`

子任务开始时调用 P1.1 移植的 `Key_Subgraph_Selection`：

```python
def kss_retrieve(scene, goal_info, cfg):
    """Returns (hit: bool, target_obj: Optional[dict], hint: Optional[str])."""
    if len(scene.objects) == 0:
        return False, None, None  # 无记忆，直接探索
    
    step = {
        "question": goal_info.text,       # object/description 文本
        "image": goal_info.image_goal,     # image 子任务的参考图
        "objects": scene.objects,
        "edges": scene.edges,
        "all_imgs": scene.all_observations,
        "image_to_edges": scene.img_to_edge,
        "top_k_categories": cfg.top_k,
        "prompt_h": cfg.prompt_img_size[0],
        "prompt_w": cfg.prompt_img_size[1],
    }
    
    question, image_goal, ego_imgs, selected_objs, selected_edges, processed_images, frontier_imgs = \
        Key_Subgraph_Selection(step, use_room_filter=cfg.use_room_det)
    
    if len(selected_objs) == 0:
        # Miss → 生成探索摘要
        hint = generate_exploration_hint(scene, goal_info, cfg)
        return False, None, hint
    
    # 命中 → 检查是否真有目标类别对象
    target_obj = find_target_in_selected(selected_objs, goal_info, scene)
    if target_obj:
        return True, target_obj, None
    else:
        hint = generate_exploration_hint(scene, goal_info, cfg, selected_objs)
        return False, None, hint
```

### P4.2 目标对象匹配 → `find_target_in_selected`

```python
def find_target_in_selected(selected_objs, goal_info, scene):
    """在 KSS 选出的对象中找目标。"""
    if goal_info.type == "object":
        for obj_id, obj in selected_objs.items():
            if obj["class_name"].lower() == goal_info.category.lower():
                return obj
    # description/image 靠 VLM 判定（P5），KSS 阶段不精确匹配
    return None  # description/image 走探索路径
```

注：object 子任务可在 KSS 阶段精确命中；description/image 需探索中 VLM 判定。

### P4.3 探索摘要生成 → `generate_exploration_hint`

```python
def generate_exploration_hint(scene, goal_info, cfg, selected_objs=None):
    """LLM 读场景图生成探索摘要。"""
    # 序列化场景图
    graph_text = ""
    for obj_id, obj in scene.objects.items():
        neighbors = [str(n[1]) for n in scene.edges if n[0] == obj_id]
        graph_text += f"{obj_id}: {obj['class_name']}, {obj.get('room_label','unknown')}, [{','.join(neighbors)}]\n"
    
    # 目标描述
    if goal_info.type == "object":
        target_desc = f"a {goal_info.category}"
    elif goal_info.type == "description":
        target_desc = goal_info.lang_desc
    else:  # image
        target_desc = "the object shown in the reference image"
    
    sys_prompt = "You are an AI agent exploring a 3D indoor scene for navigation."
    user_prompt = f"""Target: {target_desc}
Already explored scene graph:
{graph_text}

You have NOT found the target. Generate a brief exploration hint:
1. List explored room types
2. List detected object categories
3. Suggest which unexplored direction is most promising (based on semantic association)
4. Keep under 100 words

Hint:"""
    
    return call_openai_api(sys_prompt, [(user_prompt, None)])
```

验证：mock 场景图 + 3 种 goal_type，确认摘要输出合理。

---

## P5: 探索循环 — Pred-EQA planner + VLM 目标发现判定

### P5.1 探索循环主体 → `src/goat_runner.py`

```python
def run_exploration_loop(scene, tsdf_planner, goal_info, hint, cfg, max_steps=500):
    pts = scene.init_pts
    angle = scene.init_angle
    exploration_hint_injected = False
    
    for step in range(max_steps):
        # 1. 多视角观测
        for view_angle in get_view_angles(angle, cfg.total_views):
            obs = scene.get_observation(pts, view_angle)
            scene.update_scene_graph(obs["color_sensor"], obs["depth_sensor"], ...)
            tsdf_planner.integrate(...)
        
        # 2. 更新快照 + frontier
        scene.update_snapshots()
        tsdf_planner.update_frontier_map(...)
        
        # 3. VLM 目标发现判定（每步检查）
        current_view = scene.get_observation(pts, angle)["color_sensor"]
        if vlm_check_target_found(current_view, goal_info, cfg):
            return True  # 发现目标，交 P6 导航/停止
        
        # 4. query_vlm_for_response（注入 hint 仅首步）
        if not exploration_hint_injected and hint:
            step_dict["exploration_hint"] = hint
            exploration_hint_injected = True
        
        choice, reason, n_filtered = query_vlm_for_response(
            question=goal_info.text, scene=scene, tsdf_planner=tsdf_planner, ...)
        
        if choice is None:  # stop
            return False
        
        # 5. 导航
        tsdf_planner.set_next_navigation_point(choice=choice, pts=pts, ...)
        pts, angle, _, _, _, target_arrived = tsdf_planner.agent_step(...)
    
    return False
```

### P5.2 VLM 目标发现判定器 → `src/goat_runner.py`

替代 Pred-EQA answerer，yes/no 判定：

```python
def vlm_check_target_found(rgb, goal_info, cfg):
    """VLM 判定当前视角是否发现目标。"""
    img_b64 = encode_image_base64(rgb)
    
    if goal_info.type == "object":
        prompt = f"Is there a {goal_info.category} in this image? Answer yes or no only."
    elif goal_info.type == "description":
        prompt = f"Is there an object matching this description in this image? Description: {goal_info.lang_desc}. Answer yes or no only."
    else:  # image
        # 参考图 + 当前视角
        prompt = "Is this the same object/category as the reference image? Answer yes or no only."
        content = [(prompt, goal_info.image_b64), (None, img_b64)]
    
    response = call_openai_api("You are a navigation assistant.", content)
    return response and "yes" in response.lower()
```

### P5.3 Pred-EQA `explore_step` 适配

`src/pred_eqa.py`:
- `explore_step` 入口（L1174 后）加 `exploration_hint` 注入到 `format_memory_info`
- answerer 块（L1357-1420）改为调 `vlm_check_target_found`（或用 flag 跳过，runner 层自管判定）
- forced_answerer 块（L1717-1769）同上

最小改动：用 `cfg.navigation_mode=True` flag，`explore_step` 内 answerer/forced_answerer 跳过问答逻辑，直接返回 `continue exploration`，让 runner 层的 `vlm_check_target_found` 接管目标判定。

### P5.4 `exploration_hint` 注入 `format_memory_info`

`src/pred_eqa.py:1001` `format_memory_info` 加段：
```python
if step.get("exploration_hint"):
    memory_info += f"\n[Exploration Hint]\n{step['exploration_hint']}\n"
```

验证：1 个 episode 跑探索循环，确认 hint 注入 planner prompt，VLM 判定器正常返回 yes/no。

---

## P6: 导航 — VDD + 到达判定 + subtask_stop

### P6.1 KSS 命中 → VDD 导航

```python
def navigate_to_object(target_obj, scene, tsdf_planner, cfg):
    """KSS 命中后用 VDD 导航到目标对象。"""
    target_points = np.array(target_obj["pcd"].points)
    all_scene_points = np.concatenate([
        scene.objects[idx]["pcd"].points for idx in scene.objects
    ])
    
    obj_pos = Visibility_based_Viewpoint_Decision(
        target_points, all_scene_points, scene.init_pts, tsdf_planner, cfg.dicision_radius)
    
    if obj_pos is None:
        obj_pos = select_navigation_corner(target_obj["bbox"], scene.init_pts)
    
    # 导航到 obj_pos
    tsdf_planner.set_next_navigation_point(...)
    # 步进直到到达
    while not target_arrived:
        pts, angle, ..., target_arrived = tsdf_planner.agent_step(...)
```

### P6.2 探索中发现目标 → 导航 + subtask_stop

P5.2 `vlm_check_target_found` 返回 True 后：
- 检测栈可能已有该对象的 pcd → 用 VDD 导航过去
- 或直接当前位置 subtask_stop（如果 VLM 确认当前视角已看到）

### P6.3 成功判定

```python
def check_success(pts, goal_viewpoints, pathfinder, success_distance=1.0):
    """测地线距离 < success_distance。"""
    min_dist = min(
        geodesic_distance(pts, vp, pathfinder)
        for vp in goal_viewpoints
    )
    return min_dist < success_distance
```

参考 MSGNav `run_goatbench_evaluation.py:564-577`。

验证：1 个 object 子任务，KSS 命中 → VDD 导航 → 到达 → 成功判定。

---

## P7: 测试与评估

### 环境分工

- **本机**（conda `3dmem`）：pytest 单元测试（P1-P6 各模块）。habitat_sim/open3d/scipy/torch 已验证可用。
- **服务器**（`8.157.94.238:57249`，conda `3dmem`）：GOAT-Bench 数据在此，跑评估。本机不跑评估（算力不足）。

### P7.1 本机 pytest（每阶段随行）

- P1: KSS mock objects/edges → 返回 obj ID 列表；VDD mock 点云+tsdf_planner → 返回视点坐标
- P4: mock 场景图 + 3 种 goal_type → 摘要输出合理
- P5: mock RGB + goal_info → VLM 判定 yes/no（mock call_openai_api）
- P6: mock 目标对象 + pathfinder → 成功判定

测试文件：`tests/test_kss.py`, `tests/test_vdd.py`, `tests/test_goat_runner.py`

### P7.2 服务器冒烟测试

- 1 个 episode（含 3 种子任务类型各 1 个）
- 确认：场景图跨子任务保留、KSS 检索、探索摘要、VLM 判定、VDD 导航、成功判定全链路跑通
- 无 crash，日志清晰

### P7.3 服务器小规模评估

- 5-10 个 episode
- 统计 success rate per subtask type (object/description/image)
- 确认跨子任务记忆有效（第 2+ 子任务 KSS 命中率 > 第 1 个）

### P7.4 服务器全量评估

- 推到服务器 `8.157.94.238:57249`
- 跑全量 val split
- 对比 MSGNav baseline

---

## 实现顺序（关键路径）

```
P1.1 KSS 移植 ─┐
P1.2 VDD 移植 ─┤
P1.3 corner  ──┤
               ├→ P4 KSS 检索+摘要 ─┐
P2 数据+runner ─┤                    ├→ P7 集成测试
P3 Scene 适配 ─┤                    │
               └→ P5 探索循环 ──────┤
                  P6 VDD 导航 ──────┘
```

P1 三个子任务可并行。P2/P3 可并行。P4 依赖 P1+P3。P5/P6 依赖 P3+P4。

## 文件清单

| 文件 | 动作 | 阶段 |
|------|------|------|
| `src/kss_retrieval.py` | 新建（移植 MSGNav） | P1.1 |
| `src/vdd.py` | 新建（移植 MSGNav） | P1.2-P1.3 |
| `src/goat_dataset_loader.py` | 新建 | P2.2 |
| `src/goat_runner.py` | 新建 | P2.3, P4-P6 |
| `cfg/eval_goat.yaml` | 新建 | P2.4 |
| `src/scene_vlm_only.py` | 改（加 reset_for_new_subtask） | P3.1 |
| `src/pred_eqa.py` | 改（exploration_hint 注入 + nav mode） | P5.3-P5.4 |
| `src/query_vlm.py` | 改（step_dict 加 exploration_hint） | P5.4 |
