"""Tests for src/goat_dataset_loader.py.

No real GOAT-Bench data required — all inputs are mocked via tmp_path + gzip.
"""

import gzip
import json
import os

import pytest

from src.goat_dataset_loader import (
    GoatEpisode,
    SubtaskGoal,
    _scene_basename,
    get_scene_path,
    load_goat_episodes,
    render_image_goal,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------
def test_subtask_goal_defaults():
    sg = SubtaskGoal(category="chair", goal_type="object", instance_id=0)
    assert sg.category == "chair"
    assert sg.goal_type == "object"
    assert sg.instance_id == 0
    assert sg.target_positions == []
    assert sg.viewpoints == []
    assert sg.lang_desc is None
    assert sg.image_goal_params is None
    assert sg.goal_dicts == []


def test_goat_episode_defaults():
    ep = GoatEpisode(
        episode_id="ep1",
        scene_id="00800/00800.glb",
        start_position=[1.0, 0.5, 2.0],
        start_rotation=[0, 0, 0, 1],
    )
    assert ep.episode_id == "ep1"
    assert ep.scene_id == "00800/00800.glb"
    assert ep.start_position == [1.0, 0.5, 2.0]
    assert ep.start_rotation == [0, 0, 0, 1]
    assert ep.subtasks == []


# ---------------------------------------------------------------------------
# _scene_basename
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scene_id,expected", [
    ("00800/00800.glb", "00800"),
    ("00800/00800.basis.glb", "00800"),
    ("00800/00800.semantic.glb", "00800"),
    ("data/scene_datasets/hm3d/00800/00800.glb", "00800"),
])
def test_scene_basename(scene_id, expected):
    assert _scene_basename(scene_id) == expected


# ---------------------------------------------------------------------------
# get_scene_path
# ---------------------------------------------------------------------------
def test_get_scene_path_relative():
    p = get_scene_path("00800/00800.glb", "/root/hm3d")
    assert p == os.path.join("/root/hm3d", "00800/00800.glb")


def test_get_scene_path_strips_prefix():
    # strips DEFAULT_SCENE_PATH_PREFIX + "hm3d/" → searches train/val splits
    p = get_scene_path("data/scene_datasets/hm3d/train/00800/00800.glb", "/root/hm3d")
    # Non-existent scene → fallback guess
    assert "00800" in p


def test_get_scene_path_absolute_existing(tmp_path):
    # absolute existing path: strip prefix, search splits, find under train/
    scene_dir = tmp_path / "train" / "00800-SCENE"
    scene_dir.mkdir(parents=True)
    glb = scene_dir / "00800-SCENE.basis.glb"
    glb.write_text("dummy")
    p = get_scene_path("hm3d/val//00800-SCENE/00800-SCENE.basis.glb", str(tmp_path))
    assert p == str(glb)


# ---------------------------------------------------------------------------
# load_goat_episodes with mocked gzip
# ---------------------------------------------------------------------------
def _make_goal_dict(
    object_category="chair",
    object_id=1,
    position=None,
    view_points=None,
    lang_desc=None,
    image_goals=None,
    children_object_categories=None,
):
    g = {
        "object_category": object_category,
        "object_id": object_id,
        "position": position or [1.0, 0.5, 1.0],
        "view_points": view_points or [
            {"agent_state": {"position": [1.1, 0.5, 1.1], "rotation": [0, 0, 0, 1]},
             "view_position": [1.1, 0.5, 1.1]},
        ],
        "children_object_categories": children_object_categories or [],
    }
    if lang_desc is not None:
        g["lang_desc"] = lang_desc
    if image_goals is not None:
        g["image_goals"] = image_goals
    return g


def _write_split(data_path, split, payload):
    split_dir = os.path.join(data_path, split)
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f"{split}.json.gz")
    with gzip.open(split_file, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return split_file


def test_load_goat_episodes_object_task(tmp_path):
    scene_basename = "00800"
    goals_key = f"{scene_basename}_chair"
    payload = {
        "episodes": [
            {
                "episode_id": 0,
                "scene_id": "data/scene_datasets/hm3d/00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [["chair", "object", 0]],
            },
        ],
        "goals": {
            goals_key: [_make_goal_dict(object_category="chair", object_id=1)],
        },
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.episode_id == "0"
    assert ep.scene_id == "hm3d/00800/00800.glb"  # prefix stripped
    assert len(ep.subtasks) == 1
    st = ep.subtasks[0]
    assert st.category == "chair"
    assert st.goal_type == "object"
    assert len(st.target_positions) == 1
    assert len(st.viewpoints) == 1


def test_load_goat_episodes_description_task(tmp_path):
    goals_key = "00800_chair"
    lang_desc = "a wooden chair near the desk"
    payload = {
        "episodes": [
            {
                "episode_id": 1,
                "scene_id": "00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [["chair", "description", 5]],
            },
        ],
        "goals": {
            goals_key: [_make_goal_dict(
                object_category="chair", object_id=5, lang_desc=lang_desc,
            )],
        },
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    assert len(episodes) == 1
    st = episodes[0].subtasks[0]
    assert st.goal_type == "description"
    assert st.lang_desc == lang_desc


def test_load_goat_episodes_description_filtered_long(tmp_path):
    """lang_desc > 55 words should be filtered out (goat_dataset.py:178)."""
    long_desc = " ".join(["word"] * 60)
    goals_key = "00800_chair"
    payload = {
        "episodes": [
            {
                "episode_id": 2,
                "scene_id": "00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [["chair", "description", 5]],
            },
        ],
        "goals": {
            goals_key: [_make_goal_dict(
                object_category="chair", object_id=5, lang_desc=long_desc,
            )],
        },
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    # subtask filtered out -> episode has no subtasks -> skipped
    assert len(episodes) == 0


def test_load_goat_episodes_image_task(tmp_path):
    goals_key = "00800_chair"
    image_goals = [{"pose": [0, 0, 0, 1], "hfov": 90, "dimensions": [256, 256]}]
    payload = {
        "episodes": [
            {
                "episode_id": 3,
                "scene_id": "00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [["chair", "image", 7]],
            },
        ],
        "goals": {
            goals_key: [_make_goal_dict(
                object_category="chair", object_id=7, image_goals=image_goals,
            )],
        },
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    assert len(episodes) == 1
    st = episodes[0].subtasks[0]
    assert st.goal_type == "image"
    assert st.image_goal_params == image_goals


def test_load_goat_episodes_missing_category_filtered(tmp_path):
    """If goal category not in goals dict, subtask is skipped."""
    payload = {
        "episodes": [
            {
                "episode_id": 4,
                "scene_id": "00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [["nonexistent", "object", 0]],
            },
        ],
        "goals": {},
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    assert len(episodes) == 0


def test_load_goat_episodes_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_goat_episodes(str(tmp_path), "val")


def test_load_goat_episodes_multiple_subtasks(tmp_path):
    goals = {
        "00800_chair": [_make_goal_dict(object_category="chair", object_id=1)],
        "00800_table": [_make_goal_dict(object_category="table", object_id=2)],
    }
    payload = {
        "episodes": [
            {
                "episode_id": 5,
                "scene_id": "00800/00800.glb",
                "start_position": [0.0, 0.5, 0.0],
                "start_rotation": [0, 0, 0, 1],
                "tasks": [
                    ["chair", "object", 0],
                    ["table", "object", 0],
                ],
            },
        ],
        "goals": goals,
    }
    _write_split(str(tmp_path), "val", payload)

    episodes = load_goat_episodes(str(tmp_path), "val")
    assert len(episodes) == 1
    assert len(episodes[0].subtasks) == 2
    assert episodes[0].subtasks[0].category == "chair"
    assert episodes[0].subtasks[1].category == "table"


# ---------------------------------------------------------------------------
# render_image_goal placeholder
# ---------------------------------------------------------------------------
def test_render_image_goal_returns_none():
    assert render_image_goal(simulator=None, image_goal_params=None) is None
