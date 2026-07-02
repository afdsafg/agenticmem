# MSGNav + Pred-EQA 融合设计：GOAT-Bench 导航

## 目标

GOAT-Bench 纯导航（object/description/image nav），用 MSGNav 场景图记忆 + KSS 检索 + Pred-EQA 探索循环。跨子任务保留记忆。

## 三方现状

### MSGNav（场景图记忆 + KSS）
- 场景图：`objects`（class_name, room_label, bbox.center, clip_ft, image_path）+ `edges`（共现，rel_img）
- KSS 流程（`explore_utils.py`）：
  1. `related_object_KSS` → `get_prefiltering_objs` → `format_prefiltering_prompt`：LLM 读文本化场景图 `{id}: {class}[, room], [neighbor_ids]`，对目标排序返回 top-k obj ID
  2. `edge_pruning_KSS`：贪心集合覆盖选图片，返回压缩子图 (objs, edges, images)
  3. 下游 `explore_two_step`：子图 + frontier 喂 VLM，选 `Object i`（导航）/ `Image i`（重检测）/ `Continue Exploration`（探索）
- Miss case：空子图 → explore-only prompt，无专门 fallback

### Pred-EQA（探索循环 + planner）
- 主循环（`run_aeqa_evaluation_vlm_only.py`）：observe → update_snapshots → update_frontier_map → `query_vlm_for_response` → navigate
- `explore_step`（`pred_eqa.py:1165`）：多 agent 串行
  - snapshot_mgr → frontier_mgr → answerer → (planner) → (forced_answerer)
  - planner 生成 todo list 存 `long_term_memory`，`format_memory_info` 每步组装上下文
- `long_term_memory`（`SceneIntegration`）：跨步维护 planner 输出
- 检测栈已移植：每步更新 `scene.objects`/`scene.edges`（MSGNav 风格）

### GOAT-Bench（任务定义）
- 3 种子任务：object（类别导航）/ description（语言描述导航）/ image（图像目标导航）
- episode 含多子任务，`SubtaskStopAction` 切换，跨子任务期望记忆保留
- 目标传 1024d CLIP 嵌入（`GoatGoalSensor`），无文本问题
- 每 episode 5000 步，每子任务 500 步
- 成功：subtask_stop 时距目标 < 1m

## 架构设计

### 总体：三阶段子任务处理

```
子任务开始
  │
  ├─ ① KSS 检索（读场景图）
  │     命中目标对象？
  │       ├─ 是 → 直接导航到对象位置 → ②
  │       └─ 否 → 生成探索摘要 → ③
  │
  ├─ ② 导航模式（记忆命中）
  │     直奔对象 bbox.center，到点后 subtask_stop
  │
  └─ ③ 探索模式（KSS miss）
        摘要注入 planner 一次（先验种子）
        Pred-EQA 探索循环（snapshot_mgr/frontier_mgr/planner）
        planner 自维护 long_term_memory + todo list
        检测栈每步更新场景图
        发现目标 → 切 ② 或直接 subtask_stop
```

### 记忆范围
- 同 episode 跨子任务保留 `scene.objects`/`scene.edges`/`scene.img_to_edge`
- episode 间（scene 切换）清空
- 对齐 GOAT-Bench 默认期望（`ablate_memory=False`）

### 砍掉
- Pred-EQA 的 answerer / forced_answerer（纯导航无问答）

## 关键设计细节

### 1. KSS 检索（子任务开始时，复用 MSGNav 代码）

**输入**: 子任务目标 + 当前 `scene.objects` + `scene.edges`
- object 子任务：目标 = 类别文本（从 CLIP 嵌入反查或 goat-bench dataset 直接给 category）
- description 子任务：目标 = 语言描述文本
- image 子任务：目标 = 图像 CLIP 嵌入（KSS 支持 `image_goal` 参数）

**流程**（直接移植 `explore_utils.py`）:
```
related_object_KSS(question/goal, objs, edges, top_k=10, image_goal)
  → format_prefiltering_prompt: 序列化 "{id}: {class}[, room], [neighbor_ids]"
  → call_openai_api(text-only)  # LLM 排序
  → 返回 top-k obj ID 列表

edge_pruning_KSS(edges, objs, images, selected_obj_id, ...)
  → 贪心集合覆盖选图片
  → 返回 (selected_objs, selected_edges, processed_images)
```

**命中判定**: `selected_objs` 非空且含目标类别的对象 → 命中
**Miss 判定**: `selected_objs` 为空，或不含目标类别

### 2. 命中 → 导航

取 `selected_objs` 中目标对象的 `bbox.center`（X,Z 平面），设为导航目标点。
用 Pred-EQA 的 `set_next_navigation_point` 导航。
到点后调 `SubtaskStopAction`。

### 3. Miss → 探索摘要生成

LLM 读场景图后生成文本摘要，包含：
- 已探索房间类型列表（从 `objects` 的 `room_label` 去重）
- 已检测对象类别清单（`class_name` 计数）
- 未探索方向提示（frontier 位置 + 哪些方向离已知相关对象最近）
- 语义关联提示（如目标"bed"未找到，但有"pillow"+"blanket"在 room_X → 建议 explore 那个方向）

**Prompt 模板**（新增）:
```
System: You are an AI agent exploring a 3D indoor scene for navigation.

User:
Target: {goal_description}
Already explored scene graph:
{serialized_scene_graph}  # "{id}: {class}, {room}, [neighbors]"

You have NOT found the target object. Generate a brief exploration hint:
1. List explored room types
2. List detected object categories
3. Suggest which unexplored direction is most promising (based on semantic association)
4. Keep under 100 words

Hint:
```

**注入点**: Pred-EQA `explore_step` 入口（`pred_eqa.py:1174`，`get_step_info` 之后，snapshot_mgr 之前）。
摘要作为 `step["exploration_hint"]` 存入，`format_memory_info`（`pred_eqa.py:1001`）把它加进 planner prompt。
**只在子任务第一步注入**，后续步 planner 通过 long_term_memory 自维护。

### 4. 探索循环（Pred-EQA 原版，砍 answerer）

每步：
1. observe 多视角 → `scene.update_scene_graph`（检测栈更新 objects/edges）
2. `scene.update_snapshots`
3. `tsdf_planner.update_frontier_map`
4. `query_vlm_for_response`:
   - snapshot_mgr（保留快照）
   - frontier_mgr（剪枝 frontier）
   - **planner**（生成 todo list + 选 frontier）← 摘要已注入 memory
   - ~~answerer~~（砍）
   - ~~forced_answerer~~（砍）
5. 选 frontier → navigate
6. **目标检测判断**（新增）：每步检测栈发现新对象时，检查是否匹配子任务目标
   - 匹配 → 导航到该对象 → subtask_stop
   - 不匹配 → 继续

### 5. 子任务目标获取

GOAT-Bench 通过 `GoatGoalSensor` 传 1024d CLIP 嵌入，无文本。需反查：
- **object 子任务**: `goat_dataset.py` 解析 `goal[0]=category`，直接有类别文本
- **description 子任务**: `goal[1]="description"` + instance_id，dataset 有语言描述
- **image 子任务**: `goal[1]="image"` + instance_id，dataset 有实例图像路径

→ 需在 runner 层从 goat-bench dataset 直接取目标文本/图像，不走 CLIP 嵌入反查（避免信息损失）。

### 6. 子任务切换检测

GOAT-Bench 无显式"子任务开始"事件，靠 `GoatCurrentSubtaskSensor`（int 0/1/2）变化检测。
Runner 监听 `current_subtask` 值变化 → 触发 KSS 检索 + 摘要生成流程。

## 文件改动清单

### 新建
- `src/goat_runner.py` — GOAT-Bench 评估 runner（habitat-sim 直接跑，对接 goat-bench dataset，替换 `run_aeqa_evaluation_vlm_only.py`）
- `src/kss_retrieval.py` — KSS 检索 + 探索摘要生成（从 MSGNav `explore_utils.py` 移植）
- `src/vdd.py` — VDD 视点决策模块（从 MSGNav `utils.py` 移植：`generate_candidate_viewpoints`, `is_point_visible`, `compute_visibility`, `Visibility_based_Viewpoint_Decision`, `select_navigation_corner`）
- `src/goat_scene.py` — 适配 GOAT-Bench 的 Scene 子类（跨子任务保留记忆，episode 间清空）
- `cfg/eval_goat.yaml` — GOAT-Bench 评估配置

### 修改
- `src/pred_eqa.py` — `explore_step` 加 `exploration_hint` 注入点；answerer/forced_answerer 改造成"目标发现判定器"（VLM yes/no 判定，替代原问答）
- `src/scene_vlm_only.py` — 加跨子任务记忆保留/重置接口
- `src/query_vlm.py` — `step_dict` 加 `exploration_hint` 字段；导航目标点改用 VDD 选点（移植 `query_vlm.py:247,296` 调用方式）

### 移植来源（MSGNav）
- `explore_utils.py`: `Key_Subgraph_Selection`, `related_object_KSS`, `get_prefiltering_objs`, `format_prefiltering_prompt`, `edge_pruning_KSS`
- `utils.py`: `generate_candidate_viewpoints`, `is_point_visible`, `compute_visibility`, `Visibility_based_Viewpoint_Decision`（VDD 模块）
- `query_vlm.py`: `call_openai_api`, `format_content`, `select_navigation_corner`

## 开放问题（已解决）

### 1. GOAT-Bench 环境对接 → habitat-sim 直接跑

不走 GOAT-Bench 的 RL env + gym wrapper。直接用 habitat-sim + goat task 数据：
- 从 goat-bench dataset（`goat_dataset.py`）直接读 episode/subtask 定义
- 用 habitat-sim 加载 HM3D 场景（Pred-EQA 已有此能力）
- runner 自己管 episode/subtask 循环，调 habitat-sim API 观测+移动
- 参考 Pred-EQA 的 `run_aeqa_evaluation_vlm_only.py` 结构，替换任务源为 goat-bench dataset

### 2. 探索中目标匹配判定 → VLM 判定

不用 GT semantic mask（MSGNav 的 IoU 方式是评测 oracle，实际部署没有）。
不用 class_name 精确匹配（description/image 子任务无类别文本）。

**统一用 VLM 判定**（把 Pred-EQA 的 answerer 改造成"目标发现判定器"）：
- 每步探索后，把当前视角 RGB + 子任务目标描述喂 VLM
- VLM 回答"是否发现目标"
- object 子任务：prompt = "Is there a {category} in this image? Answer yes/no."
- description 子任务：prompt = "Is there an object matching this description in this image? Description: {description_text}. Answer yes/no."
- image 子任务：prompt = [目标图像] + [当前视角] + "Is this the same object/category as the reference image? Answer yes/no."
- VLM 说 yes → 命中，调 subtask_stop
- VLM 说 no → 继续探索

这取代了 Pred-EQA 原 answerer 的"回答问题"功能，改成"判定目标发现"。

### 3. 导航精度 → 复用 MSGNav VDD 模块

KSS 命中后不只直奔 bbox.center，用 MSGNav 的 `Visibility_based_Viewpoint_Decision`（`utils.py:51`）选最佳可视视点：

**VDD 流程**:
1. `generate_candidate_viewpoints(bbox_center, radius, pts)`（`utils.py:9`）：在目标 bbox center 周围圆形生成 20 个候选视点
2. `tsdf_planner.mask_true_point(candidate_viewpoints)`：过滤掉不可导航的候选点
3. 对每个候选视点，`compute_visibility(vp, target_points, scene_points_tree)`（`utils.py:39`）：
   - 用 KDTree 建场景点云索引
   - 沿视点→目标点连线采样，检查遮挡
   - 返回可见目标点比例
4. 选 visibility_score 最高的视点作为导航目标
5. fallback：`select_navigation_corner(aabb, robot_position)`（无可见视点时用 bbox 角点）

**调用方式**（参考 `query_vlm.py:247,296`）:
```python
target_points = np.array(obj["pcd"].points)
all_scene_points = np.concatenate([scene.objects[idx]["pcd"].points for idx in scene.objects.keys()])
obj_pos = Visibility_based_Viewpoint_Decision(target_points, all_scene_points, pts, tsdf_planner, cfg.dicision_radius)
if obj_pos is None:
    obj_pos = select_navigation_corner(aabb=obj["bbox"], robot_position=pts)
```

需要从 MSGNav 移植：`utils.py` 的 `generate_candidate_viewpoints`, `is_point_visible`, `compute_visibility`, `Visibility_based_Viewpoint_Decision`，以及 `select_navigation_corner`。

### 4. description/image 子任务目标对齐 → 原始任务输入给 VLM

不通过 CLIP 嵌入反查，直接从 goat-bench dataset 取原始任务输入：
- **object 子任务**: `goal[0]=category` → 类别文本直接给 VLM
- **description 子任务**: dataset 有语言描述文本 → 告诉 VLM "找到匹配此描述的物体"
- **image 子任务**: dataset 有实例图像路径 → 告诉 VLM "找到与这张参考图像相同的物体/类别"

**KSS 检索时的目标表示**:
- object: 类别文本作为 `question` 参数
- description: 描述文本作为 `question` 参数
- image: 实例图像作为 `image_goal` 参数（KSS 原生支持）

**VLM 目标发现判定时的 prompt**（见开放问题2）:
- 明确告诉 VLM 要找什么类型的对象（"description 对应的物体" / "image 中的物体"）
- VLM 基于视觉语义判断当前视角是否命中

**成功判定**: 用 1m 测地线距离阈值（由 goat-bench 环境/数据集提供 GT 目标位置）。我们只负责把 agent 导航到目标附近，不负责精确判定成功——成功由评测脚本基于 GT 位置算。
