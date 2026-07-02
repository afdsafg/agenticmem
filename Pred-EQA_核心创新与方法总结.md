# Pred-EQA: 核心创新点、方法与工作流程详细总结

> 论文: *Predict Before You Explore: Predictive Planning with Specialized Memory for Embodied Question Answering*
> 发表: CVPR 2026
> 作者: Bowen Yuan, Sisi You (南京邮电大学), Bing-Kun Bao (合肥工业大学)

---

## 一、背景与动机

Embodied Question Answering (EQA) 要求 agent 在3D环境中主动导航、收集视觉证据、并回答问题。现有方法存在两个核心问题：

1. **反应式规划 (Reactive Planning)**: 现有 VLM-based agent 每收到一个新观测才做决策，缺乏跨步骤的长远规划意图，导致动作不一致、轨迹不连贯。
2. **单一记忆瓶颈 (Monolithic Memory)**: 所有观测不分轻重都塞进一个记忆结构中，导致稀疏但关键的证据被大量无关信息淹没，检索困难。

**核心洞察**: 论文将 EQA 重新定义为**预测性处理 (Predictive Processing)** 问题——人类不是被动响应传感器输入，而是主动预测未来观测，当预测失败时修正内部先验。这一预测-修正循环能够维持长期连贯意图，并自然地将观测分离为"问题无关的稳定先验"和"问题相关的稀疏证据"。

---

## 二、核心创新点总结

### 创新1: 预测-修正-更新循环 (Prediction-Correction-Update Loop)

受神经科学中的 **活跃推理 (Active Inference)** 和**自由能原理 (Free Energy Principle)** 启发，将 EQA 建模为一个预测-修正-更新循环：

$$\min F = D_{KL}[Q(\psi|s,q,a) \| P(\psi,s|q)] - \mathbb{E}_Q[\log P(a|\psi,s,q)]$$

- **KL散度项**: 鼓励 agent 形成稳定的空间/语义先验，避免目标漂移
- **似然项**: 驱动 agent 选择最能减少不确定性的动作

这转化为三个具体的循环操作：**Prediction → Correction → Update**。

### 创新2: 预测式分层规划 (Predictive Hierarchical Planning)

| 组件 | 功能 | 对应代码 |
|------|------|----------|
| **High-Level Planner** | 根据问题和记忆，预测"哪些语义区域可能包含证据"，生成多个**并行探索分支** (predictive exploration branches)，每条分支是可验证的假设 (如 "门廊可能通向厨房") | `src/pred_eqa.py:768-840` (`format_high_level_plan_prompt`) |
| **Low-Level Executor** | 在分支范围内执行局部探索动作，遵循三个原则:<br/>1. 最大化信息增益（选择最能减少不确定性的frontier）<br/>2. 常识引导（利用典型房间-物体关联）<br/>3. 一致性驱动（遵循高层计划维持连贯轨迹） | `src/pred_eqa.py:697-766` (`format_explore_prompt`) |

**与传统方法的区别**:
- 传统方法: "我看到什么 → 我该去哪" (反应式)
- Pred-EQA: "我预测证据可能在厨房 → 验证'门廊通向厨房'这个假设 → 如果不通则修正" (预测式)

高层规划器输出的结构化 TODO list 格式 (`src/plan_extraction_utils.py:4-60`):

```
<update_todo_list>
<todos>
[-] Go through the doorway into the kitchen.
[x] Explore the frontier leading to the hallway <!-- Irrelevant; kitchen confirmed via doorway -->
[x] Explore the frontier leading to the living area <!-- Also irrelevant -->
[ ] Locate the oven in the kitchen.
[ ] Inspect the oven handle.
</todos>
</update_todo_list>
```

每个任务有三个状态: `[ ]` pending, `[-]` in_progress, `[x]` completed，Completed 的任务会附带失败/成功原因的注释。

### 创新3: 功能专业化双记忆系统 (Functionally Specialized Memory)

| 记忆类型 | 存储内容 | 实现 | 作用 |
|----------|----------|------|------|
| **Textual Structural Memory** (文本结构化记忆) | 环境布局、房间连接关系、轨迹历史等**稳定先验**（与问题无关，持续积累） | `src/long_term_memory.py:25-171` <br/> `src/scene_integration.py:7-30` | 为预测式规划提供持久的空间/语义锚点，维持长期规划一致性 |
| **Visual Evidence Memory** (视觉证据记忆) | 仅保留**与问题相关**的视觉快照（极度稀疏，平均每步 < 3 帧） | 由 Snapshot Manager agent 动态过滤 (`src/pred_eqa.py:586-638`) | 确保答题时有干净、无冗余的证据，避免检索干扰 |

关键设计:
- **Textual Memory** 在每一步由 Recorder agent 将环境观测转换成结构化文本条目，再汇总为 prior 摘要 (`src/pred_eqa.py:274-327`)。
- **Visual Memory** 由 Snapshot Manager agent 根据以下准则过滤快照: 1) 直接描述答案相关信息；2) 提供视觉-空间线索帮助减少预测不确定性；3) 提供对不确定性区域的非冗余视角。

### 创新4: 多Agent协作框架 (Multi-Agent Framework)

Pred-EQA 将系统解耦为多个职责清晰的 VLM-agent，每个 agent 有独立且精心管理的 prompt context：

| Agent | 职责 | 对应代码 |
|-------|------|----------|
| **Snapshot Manager** | 筛选与问题相关的视觉快照，丢弃冗余和无关帧 | `format_manage_prompt` (l.586-638) |
| **Frontier Manager** | 筛选和修剪探索方向，移除已探索且无关的 frontier | `format_plan_manager_prompt` (l.844-907) |
| **Answerer** | 判断当前收集的证据是否足以回答问题 | `format_answer_prompt` (l.640-694) |
| **High-Level Planner** | 生成/更新预测式探索分支和 TODO list | `format_high_level_plan_prompt` (l.768-840) |
| **Low-Level Executor (Planner)** | 根据高层计划选择具体的下一步 frontier | `format_explore_prompt` (l.697-766) |
| **Forced Answerer** | 当探索终止或无可用 frontier 时强制基于已有快照回答问题 | `format_force_answer_prompt` (l.910-956) |

所有 agent 共享同一个本地部署的 VLM (通过 vLLM)，通过 OpenAI 兼容 API 调用 (`src/pred_eqa.py:30-34`)。

---

## 三、整体工作流程 (Workflow)

### 3.1 系统初始化

```
Scene 初始化 (src/scene_vlm_only.py:36-143)
  ├── 加载 HM3D 3D场景
  ├── 初始化 Habitat-Sim 模拟器
  ├── 初始化 TSDF 体积融合规划器 (src/tsdf_planner.py:80-164)
  ├── 初始化 TextLongTermMemory (文本长期记忆)
  └── 初始化 SceneIntegration (记忆系统整合)
```

### 3.2 单步探索循环 (Per-Step Exploration Loop)

每步的核心循环在 `src/pred_eqa.py:1175-1779` 的 `explore_step()` 函数中实现，执行顺序如下：

```
Step t 开始
  │
  ├── Step 0: Snapshot Manager
  │   ├── 当快照数 > 3 时触发
  │   ├── VLM 分析每个快照与问题的相关性
  │   ├── 输出 "Retain Snapshots: {i, j, ...}"
  │   ├── 过滤: 只保留与问题相关的快照
  │   └── 记录决策到 Textual Structural Memory
  │
  ├── Step 1: Frontier Manager
  │   ├── 当 frontier 数 > 1 时触发
  │   ├── VLM 分析哪些 frontier 已被探索且与问题无关
  │   ├── 输出 "Retain Frontiers: {i, j, ...}"
  │   ├── 过滤: 移除已访问且无关的探索方向
  │   └── 记录决策到 Textual Structural Memory
  │
  ├── Step 2: Answerer
  │   ├── 如果有有效快照，VLM 判断是否足以回答问题
  │   ├── 足够 → 输出 "Answer: [答案] (Evidence: Snapshot i)"
  │   ├── 不足 → 输出 "Continue Exploration"
  │   └── 记录决策到 Textual Structural Memory
  │
  ├── Step 3: [IF Continue Exploration]
  │   ├── High-Level Planner
  │   │   ├── 分析问题类型（识别/属性/空间/状态/功能/常识/定位）
  │   │   ├── 分解为子目标
  │   │   ├── 生成多个并行预测式探索分支 (testable hypotheses)
  │   │   ├── 输出 XML 格式的 <update_todo_list>
  │   │   ├── 支持动态重新规划（新观测可能推翻旧假设）
  │   │   └── 记录到 Textual Structural Memory (类型: high_level_planner_output)
  │   │
  │   └── Low-Level Executor (Planner)
  │       ├── 接收问题、高层TODO list、快照、frontiers
  │       ├── 按三原则选择具体 frontier:
  │       │   ├── 常识: 冰箱→厨房, 床→卧室
  │       │   ├── 信息增益: 选择最能验证预测的分支
  │       │   └── 一致性: 遵循高层计划
  │       ├── 输出 "Next Step: Frontier i" 或 "Stop Exploration"
  │       └── 记录决策到 Textual Structural Memory
  │
  ├── Step 4: Step Summary Generation
  │   ├── 收集当前步所有 agent 的输出
  │   ├── 生成综合摘要 (~150词)
  │   ├── 摘要聚焦于: 环境布局、空间连接性、当前进度
  │   └── 记录到 Textual Structural Memory (类型: step_summary_output)
  │
  └── Step 5: [IF 无有效响应 或 无 frontier 或 探索终止]
      └── Forced Answerer
          ├── 强制从已有的快照中选择答案
          ├── 输出 "Snapshot i\n[答案]"
          └── 返回最终答案
```

### 3.3 环境交互循环

```
TSDFPlanner 执行导航 (src/tsdf_planner.py:612-956)
  ├── 更新 TSDF 体积地图 (RGB-D 融合)
  ├── 提取/更新 Frontiers (未探索边界)
  │   ├── 基于 TSDF 占用的连通区域分析
  │   ├── DBSCAN 聚类 frontier 区域
  │   └── KMeans 拆分过大的 frontier 角度范围
  ├── 路径规划 (NavMesh-based)
  │   ├── 计算到目标 frontier/snapshot 的路径
  │   └── 沿路径移动 max_dist_from_cur 距离
  └── 采集新观测
      ├── 6 张 360° egocentric 视图 (初始化)
      ├── 3 张 120° heading-aligned 视图 (每步)
      └── 更新 snapshot 和 frame 缓冲区
```

### 3.4 记忆管理循环

```
Textual Structural Memory (TextLongTermMemory)
  ├── 每步记录所有 agent 的结构化输出
  │   ├── agent_output (带 entry_type 索引)
  │   ├── step_summary (压缩历史)
  │   └── high_level_plan (TODO list)
  ├── 检索接口:
  │   ├── retrieve_by_type()    - 按 agent 类型检索
  │   ├── retrieve_by_step()    - 按步骤检索
  │   ├── retrieve_by_type_and_step() - 组合检索
  │   └── retrieve_by_time()    - 按时间检索
  └── format_memory_info() 格式化记忆为 prompt 上下文
      ├── only_high_level_plan=True  → 仅返回 TODO list
      ├── outside=True  → 包含历史步骤摘要
      └── outside=False → 仅当前步骤 + TODO list

Visual Evidence Memory (Snapshot 管理)
  ├── Scene.snapshots: 当前保留的视觉快照
  ├── Scene.filtered_snapshots: 已被过滤的 (防止重新加入)
  ├── Snapshot Manager 每步动态筛选
  └── 平均每步保留 < 3 帧 (极度稀疏)
```

### 3.5 完整 EQA Episode 流程

```
run_aeqa_evaluation_vlm_only.py (A-EQA) / run_express_bench_evaluation_vlm_only.py (Express-Bench)
  │
  ├── 1. 加载问题 {question_id, episode_history (场景ID), question}
  ├── 2. 初始化 Scene (加载 3D 场景)
  ├── 3. 设置初始位置和角度 (从 GT 路径)
  ├── 4. 采集初始 egocentric 视图 (360°)
  │
  └── 5. 探索循环 (最多 50 步):
      ├── a. 更新 TSDF 地图和 frontiers
      ├── b. query_vlm_for_response() → explore_step()
      │   └── 执行 Snapshot Manager → Frontier Manager → Answerer → (High-Level Planner + Executor) → 决策
      ├── c. 如果 Answerer 返回答案 → 终止探索
      ├── d. 如果 Executor 选择 frontier → 导航到目标
      │   └── TSDFPlanner.set_next_navigation_point() 计算导航点
      │   └── TSDFPlanner.agent_step() 执行一步移动
      ├── e. 采集新观测，更新 snapshots/frames
      └── f. 如果无有效响应或达到步数上限 → Forced Answerer 强制回答

  ├── 6. 记录轨迹、快照、答案
  └── 7. 评估
      ├── A-EQA: evaluate-predictions.py (LLM-Match) + get-scores.py (LLM-SPL)
      └── Express-Bench: evaluate_express_bench.py + get_scores_express_bench.py (C, C*, E_path, d_T)
```

---

## 四、关键技术细节

### 4.1 纯 VLM Pipeline (No Detectors)

Pred-EQA 是**纯 VLM 驱动**的，不依赖任何目标检测器 (如 DETR)、场景图 (如 ConceptGraphs)、或语义地图。所有感知和推理都由 VLM 直接完成：
- RGB-D 图像直接送入 VLM (不经过特征提取)
- 不维护 3D 物体表征
- 不进行特征匹配或 CLIP 编码
- TSDF 仅用于几何层面的 frontier 提取（不涉及语义）

这使得系统极其简洁，性能增益完全来自架构设计而非外部模块。

### 4.2 记忆的预测式分离

传统方法将所有观测混在一起 → 检索困难。Pred-EQA 利用预测式处理的视角自然分离：
- 预测需要**稳定的先验** (文本记忆) —— 空间布局、房间关系
- 答题需要**稀疏的证据** (视觉记忆) —— 只保留与问题相关的图

两个记忆模块相互补充：文本记忆为预测提供锚点，视觉记忆为验证和答题提供证据。

### 4.3 探索分支的动态管理

High-Level Planner 生成的不是固定的计划，而是**可验证的假设分支**：
- 当一条分支被证实有效 → 其他同目标的并行分支立即标记为 `[x]` 并注释原因
- 当一条分支被证伪 → 标记为 `[x]` 并解释失败原因
- 当环境揭示新信息 → 支持动态重新规划，废弃旧的、添加新的

这种设计避免了 TODO-list 规划在部分可观测环境中的脆弱性——固定子目标列表在新观测下很快过时，而预测式分支始终与当前信念保持同步。

### 4.4 信息增益驱动的行动选择

Low-Level Executor 的行动选择量化为最小化自由能：
1. **信息增益最大化**: 选择最能减少预测分支不确定性的 frontier
2. **常识引导**: 当无视觉线索时，利用 "冰箱通常在厨房" 等常识排名
3. **一致性驱动**: 优先验证正在进行的预测分支，维持轨迹连贯性

---

## 五、代码与论文对应关系

| 论文概念 | 代码位置 | 说明 |
|----------|----------|------|
| 预测-修正-更新循环 | `src/pred_eqa.py:1175-1779` | `explore_step()` 函数实现完整循环 |
| Hierarchical Planner | `src/pred_eqa.py:768-840` (High-Level) <br/> `src/pred_eqa.py:697-766` (Executor) | Prompt 工程实现 |
| High-Level 预测分支 | `src/plan_extraction_utils.py:4-60` | `extract_predictive_plan()` 解析 XML TODO list |
| Textual Structural Memory | `src/long_term_memory.py:25-171` | `TextLongTermMemory` 类，带类型/步骤/时间索引 |
| Visual Evidence Memory | `src/scene_vlm_only.py:128-133` | `Scene.snapshots` + `Scene.filtered_snapshots` |
| Snapshot Manager | `src/pred_eqa.py:586-638` | `format_manage_prompt()` + 解析 |
| Frontier Manager | `src/pred_eqa.py:844-907` | `format_plan_manager_prompt()` + 解析 |
| Answerer | `src/pred_eqa.py:640-694` | `format_answer_prompt()` + 解析 |
| Memory 格式化 | `src/pred_eqa.py:1011-1169` | `format_memory_info()` 将记忆注入 prompt |
| Scene 初始化 | `src/scene_vlm_only.py:36-143` | `Scene.__init__()` 加载场景+记忆系统 |
| TSDF 几何探索 | `src/tsdf_planner.py:80-1078` | `TSDFPlanner` 前线提取与导航 |
| 场景整合 | `src/scene_integration.py:7-30` | `SceneIntegration` 桥接 Scene 和 Memory |
| 评估流水线 | `run_aeqa_evaluation_vlm_only.py` <br/> `run_express_bench_evaluation_vlm_only.py` | Episode 级别的评估入口 |
| 自由能公式 | 论文 Eq.(1) | 理论框架，不直接编码但指导了规划器的信息增益设计 |

---

## 六、实验结果亮点

| 数据集 | 指标 | Pred-EQA (Qwen3-VL 8B) | 对比 |
|--------|------|------------------------|------|
| **A-EQA** | LLM-Match ↑ | **53.3** | 超过 GPT-4o 最优方法 3D-Mem (52.6) |
| **A-EQA** | LLM-SPL ↑ | **48.5** | 超过所有 proprietary agent，探索效率最高 |
| **Express-Bench** | C ↑ | **52.58** | 超过 ToolEQA (42.21) +10.37% |
| **Express-Bench** | C* ↑ | **70.54** | 超过 ToolEQA (65.77) +4.77% |
| **Express-Bench** | E_path ↑ | **47.66** | 相对 SOTA 提升 >20% |

**消融实验关键发现**:
- 纯预测式规划 (+5.1% LLM-Match / +5.6% LLM-SPL)
- 纯专业化记忆 (+1.5% LLM-Match / +4.1% LLM-SPL)
- 两者联合 (+7.6% LLM-Match / +9.0% LLM-SPL) — 验证了规划与记忆的协同效应

**可扩展性**: Pred-EQA 在 Qwen2.5-VL 3B→7B→32B 和 Qwen3-VL 4B→8B→30B→32B 上均显示稳定增长，证明性能增益来自架构而非模型规模。

---

## 七、系统架构图 (文字版)

```
┌─────────────────────────────────────────────────────────────┐
│                      Pred-EQA System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────┐   │
│  │   Habitat   │    │         VLM (vLLM API)           │   │
│  │  Simulator  │    │    ┌─────────────────────────┐   │   │
│  │  (HM3D)    │    │    │  Snapshot Manager       │   │   │
│  │             │    │    │  Frontier Manager       │   │   │
│  │  RGB-D ────┼────┼───▶│  Answerer               │   │   │
│  │  Obs       │    │    │  High-Level Planner     │   │   │
│  │             │    │    │  Low-Level Executor     │   │   │
│  └──────┬──────┘    │    └──────────┬──────────────┘   │   │
│         │           └──────────────┼───────────────────┘   │
│         ▼                          │                       │
│  ┌──────────────┐                  │                       │
│  │ TSDF Planner │                  ▼                       │
│  │  ├Frontiers  │    ┌──────────────────────────┐         │
│  │  ├Occupancy  │    │   Specialized Memory      │         │
│  │  └Navigation │    │  ┌────────────────────┐   │         │
│  └──────────────┘    │  │ Textual Structural │   │         │
│                       │  │   Memory (prior)   │◀──┼─────────┤
│                       │  └────────────────────┘   │         │
│                       │  ┌────────────────────┐   │         │
│                       │  │ Visual Evidence    │   │         │
│                       │  │   Memory (sparse)  │◀──┼─────────┤
│                       │  └────────────────────┘   │         │
│                       └──────────────────────────┘         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Prediction → Correction → Update Loop               │  │
│  │  - High-Level Planner 预测证据可能出现的位置         │  │
│  │  - Executor 验证/否证预测                            │  │
│  │  - Memory 更新先验和证据                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```
