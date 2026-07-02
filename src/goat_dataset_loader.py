"""GOAT-Bench dataset loader.

Reads GOAT-Bench JSON.gz episode/goal data directly, without the habitat
dataset framework. Mirrors the parsing logic in goat_bench/dataset/goat_dataset.py
L140-221 but returns plain dataclasses.

Data layout (server):
    {data_path}/{split}/{split}.json.gz          # episodes + goals (per-scene files)
    {data_path}/{split}/content/{scene}.json.gz  # per-scene episode splits

The val.json.gz top-level structure:
    {
        "episodes": [ {episode_id, scene_id, start_position, start_rotation, tasks}, ... ],
        "goals": { "{scene_basename}_{category}": [ {goal_dict}, ... ], ... }
    }

episode["tasks"] = List[[category:str, type:str, instance_id:int]]
goal_dict fields: object_category, object_id, position, view_points,
                  lang_desc (description tasks), image_goals (image tasks),
                  children_object_categories.
"""

import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# goat_dataset.py:12 — habitat-lab default scene path prefix
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@dataclass
class SubtaskGoal:
    """Resolved subtask goal parsed from episode.tasks + goals dict."""
    category: str            # goal[0]
    goal_type: str           # goal[1]: "object" | "description" | "image"
    instance_id: int         # goal[2]
    # Resolved from goals dict (list of matching goal dicts):
    target_positions: list = field(default_factory=list)  # goal_dict["position"]
    viewpoints: list = field(default_factory=list)        # goal_dict["view_points"]
    lang_desc: Optional[str] = None                       # goal_dict["lang_desc"]
    image_goal_params: Optional[list] = None              # goal_dict["image_goals"]
    # Full list of goal dicts for this subtask (object tasks may have multiple)
    goal_dicts: list = field(default_factory=list)


@dataclass
class GoatEpisode:
    """GOAT-Bench composite episode."""
    episode_id: str
    scene_id: str            # raw scene_id from json (may have DEFAULT_SCENE_PATH_PREFIX)
    start_position: list
    start_rotation: list
    subtasks: List[SubtaskGoal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Goal resolution (mirrors goat_dataset.py:160-219 / MSGNav dataset_utils.py:1-65)
# ---------------------------------------------------------------------------
def _resolve_subtask_goal(
    goal_spec: list,
    goals: Dict[str, list],
    scene_basename: str,
) -> Optional[SubtaskGoal]:
    """Resolve one task tuple [category, type, instance_id] against the goals dict.

    Returns None if the goal category is missing from goals (filtered out).
    """
    category, goal_type, instance_id = goal_spec[0], goal_spec[1], goal_spec[2]

    # Find the goal list whose first entry matches this category.
    # goals key format: "{scene_basename}_{category}" but categories with children
    # are merged (goat_dataset.py:194-205). Match by object_category of first entry.
    matching_goal_lists = [
        glist for glist in goals.values()
        if glist and glist[0].get("object_category") == category
    ]
    if not matching_goal_lists:
        logger.warning(
            "Goal category %r not found in goals dict for scene %s",
            category, scene_basename,
        )
        return None

    dset_same_cat_goals = matching_goal_lists[0]

    # Merge children categories (goat_dataset.py:197-205)
    children_categories = dset_same_cat_goals[0].get("children_object_categories", [])
    for child_category in children_categories:
        child_key = f"{scene_basename}_{child_category}"
        if child_key in goals:
            dset_same_cat_goals = dset_same_cat_goals + goals[child_key]

    # object task: all goals of the category; description/image: filter by instance_id
    if goal_type == "object":
        resolved = dset_same_cat_goals
    else:
        resolved = [
            g for g in dset_same_cat_goals
            if g.get("object_id") == instance_id
        ]
        if not resolved:
            logger.warning(
                "No goal instance %d for category %r (type=%s) in scene %s",
                instance_id, category, goal_type, scene_basename,
            )
            return None

    # description filter: skip if lang_desc > 55 words (goat_dataset.py:178)
    if goal_type == "description":
        lang_desc = resolved[0].get("lang_desc", "")
        if len(lang_desc.split(" ")) > 55:
            return None

    # Aggregate fields
    target_positions = [g.get("position") for g in resolved if g.get("position") is not None]
    viewpoints: list = []
    for g in resolved:
        viewpoints.extend(g.get("view_points", []))
    lang_desc = resolved[0].get("lang_desc") if goal_type == "description" else None
    image_goal_params = resolved[0].get("image_goals") if goal_type == "image" else None

    return SubtaskGoal(
        category=category,
        goal_type=goal_type,
        instance_id=instance_id,
        target_positions=target_positions,
        viewpoints=viewpoints,
        lang_desc=lang_desc,
        image_goal_params=image_goal_params,
        goal_dicts=resolved,
    )


def _scene_basename(scene_id: str) -> str:
    """GOAT-Bench goals key uses scene basename without path and without .glb/.basis."""
    base = os.path.basename(scene_id)
    # Strip compound suffixes first (.semantic.glb, .basis.glb) then single .glb
    for ext in (".semantic.glb", ".basis.glb", ".glb"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_goat_episodes(
    data_path: str,
    split: str = "val",
) -> List[GoatEpisode]:
    """Load GOAT-Bench episodes from a split json.gz.

    data_path: directory containing {split}/{split}.json.gz and {split}/content/.
    Returns List[GoatEpisode], each with a subtasks list (filtered).
    """
    split_dir = os.path.join(data_path, split)
    split_file = os.path.join(split_dir, f"{split}.json.gz")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"GOAT-Bench split file not found: {split_file}")

    logger.info("Loading GOAT-Bench episodes from %s", split_file)
    with gzip.open(split_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes_raw = data.get("episodes", [])
    goals = data.get("goals", {})
    if not isinstance(goals, dict):
        goals = {}

    episodes: List[GoatEpisode] = []
    for i, ep_raw in enumerate(episodes_raw):
        scene_id = ep_raw["scene_id"]
        # Strip DEFAULT_SCENE_PATH_PREFIX if present (goat_dataset.py:147-156)
        if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
            scene_id = scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]

        scene_base = _scene_basename(scene_id)

        subtasks: List[SubtaskGoal] = []
        for goal_spec in ep_raw.get("tasks", []):
            sg = _resolve_subtask_goal(goal_spec, goals, scene_base)
            if sg is not None:
                subtasks.append(sg)

        if not subtasks:
            continue

        episodes.append(GoatEpisode(
            episode_id=str(ep_raw.get("episode_id", i)),
            scene_id=scene_id,
            start_position=list(ep_raw.get("start_position", [])),
            start_rotation=list(ep_raw.get("start_rotation", [])),
            subtasks=subtasks,
        ))

    logger.info("Loaded %d GOAT-Bench episodes (%d raw)", len(episodes), len(episodes_raw))
    return episodes


def get_scene_path(scene_id: str, scene_data_path: str) -> str:
    """Build a habitat-sim-loadable scene path from a GOAT-Bench scene_id.

    scene_id may be bare (e.g. "00800/00800.basis.glb") or already absolute.
    scene_data_path is the HM3D dataset root (e.g. /root/hm3d/).
    """
    # If already absolute and exists, return as-is
    if os.path.isabs(scene_id) and os.path.exists(scene_id):
        return scene_id
    # Strip prefix if present
    sid = scene_id
    if sid.startswith(DEFAULT_SCENE_PATH_PREFIX):
        sid = sid[len(DEFAULT_SCENE_PATH_PREFIX):]
    return os.path.join(scene_data_path, sid)


def render_image_goal(simulator, image_goal_params: Any) -> Optional[np.ndarray]:
    """Render the reference image for an image subtask.

    image_goal_params: InstanceImageParameters-like (has pose/hfov/dimensions).
    Uses habitat_sim to render at the goal camera pose.

    ponytail: returns None placeholder; full impl needs habitat_sim camera setup
    matching InstanceImageParameters. Add when wiring image subtasks (P5+).
    """
    # TODO: extract pose (position+rotation), hfov, dimensions from image_goal_params
    # TODO: simulator.get_observations_at(position, rotation) -> rgb
    logger.warning("render_image_goal not implemented; returning None placeholder")
    return None
