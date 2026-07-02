"""Unit tests for src/goat_retrieval.py (KSS retrieve + exploration hint).

Mocks Key_Subgraph_Selection and call_openai_api to avoid real VLM / habitat calls.
"""

import os

# Clear proxy env vars before importing src.pred_eqa (which constructs OpenAI
# client at module load). socks5h scheme is unsupported by httpx.
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        del os.environ[_k]

import pytest

from src import goat_retrieval
from src.goal_types import GoalInfo


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class FakeObj(dict):
    """Mimics MapObjectDict: supports both obj['key'] and obj.key access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


class FakeEdge:
    """Mimics MSGNav edge object."""
    def __init__(self, rel_img):
        self.rel_img = rel_img


def make_obj(class_name, room_label="unknown"):
    obj = FakeObj()
    obj.class_name = class_name
    obj.room_label = room_label
    return obj


class FakeScene:
    """Minimal scene mock with objects/edges/img_to_edge/all_observations."""
    def __init__(self, objects=None, edges=None, img_to_edge=None, all_observations=None):
        self.objects = objects if objects is not None else {}
        self.edges = edges if edges is not None else {}
        self.img_to_edge = img_to_edge if img_to_edge is not None else {}
        self.all_observations = all_observations if all_observations is not None else {}


class FakeCfg:
    """Minimal cfg mock."""
    def __init__(self, top_k=10, use_room_det=False, prompt_img_size=(32, 32)):
        self.top_k = top_k
        self.use_room_det = use_room_det
        self.prompt_img_size = prompt_img_size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def empty_scene():
    return FakeScene(objects={})


@pytest.fixture
def populated_scene():
    objs = {
        1: make_obj("tv", "living room"),
        2: make_obj("sofa", "living room"),
        3: make_obj("bed", "bedroom"),
    }
    edges = {
        (1, 2): FakeEdge(rel_img=["img_a"]),
        (1, 3): FakeEdge(rel_img=["img_b"]),
    }
    img_to_edge = {
        "img_a": [(1, 2)],
        "img_b": [(1, 3)],
    }
    return FakeScene(objects=objs, edges=edges, img_to_edge=img_to_edge,
                     all_observations={"img_a": "mock", "img_b": "mock"})


@pytest.fixture
def cfg():
    return FakeCfg()


def _kss_returns(selected_objs):
    """Build a fake Key_Subgraph_Selection return value (7-tuple)."""
    return ("question", None, [], selected_objs, {}, {}, [])


# ---------------------------------------------------------------------------
# Tests: kss_retrieve
# ---------------------------------------------------------------------------
class TestKssRetrieve:
    def test_empty_objects_returns_miss_no_hint(self, empty_scene, cfg):
        """Empty scene.objects → (False, None, None) without calling KSS/VLM."""
        hit, target, hint = goat_retrieval.kss_retrieve(empty_scene, GoalInfo(type="object", category="tv"), cfg)
        assert hit is False
        assert target is None
        assert hint is None

    def test_kss_miss_empty_selected_calls_hint(self, populated_scene, cfg, monkeypatch):
        """KSS returns empty selected_objs → miss + hint generated."""
        monkeypatch.setattr(
            goat_retrieval, "Key_Subgraph_Selection",
            lambda step, **kw: _kss_returns({}),
        )
        monkeypatch.setattr(
            goat_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "Explore the bedroom.",
        )
        goal = GoalInfo(type="object", category="tv")
        hit, target, hint = goat_retrieval.kss_retrieve(populated_scene, goal, cfg)
        assert hit is False
        assert target is None
        assert hint == "Explore the bedroom."

    def test_kss_hit_with_target(self, populated_scene, cfg, monkeypatch):
        """KSS returns selected_objs containing target class → hit."""
        selected = {1: make_obj("tv", "living room")}
        monkeypatch.setattr(
            goat_retrieval, "Key_Subgraph_Selection",
            lambda step, **kw: _kss_returns(selected),
        )
        goal = GoalInfo(type="object", category="tv")
        hit, target, hint = goat_retrieval.kss_retrieve(populated_scene, goal, cfg)
        assert hit is True
        assert target is not None
        assert target["class_name"] == "tv"
        assert hint is None

    def test_kss_selected_without_target_miss_hint(self, populated_scene, cfg, monkeypatch):
        """KSS returns selected_objs without target class → miss + hint."""
        selected = {3: make_obj("bed", "bedroom")}
        monkeypatch.setattr(
            goat_retrieval, "Key_Subgraph_Selection",
            lambda step, **kw: _kss_returns(selected),
        )
        monkeypatch.setattr(
            goat_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "Go to living room.",
        )
        goal = GoalInfo(type="object", category="tv")
        hit, target, hint = goat_retrieval.kss_retrieve(populated_scene, goal, cfg)
        assert hit is False
        assert target is None
        assert hint == "Go to living room."

    def test_kss_hit_case_insensitive(self, populated_scene, cfg, monkeypatch):
        """Class matching should be case-insensitive."""
        selected = {1: make_obj("TV", "living room")}
        monkeypatch.setattr(
            goat_retrieval, "Key_Subgraph_Selection",
            lambda step, **kw: _kss_returns(selected),
        )
        goal = GoalInfo(type="object", category="tv")
        hit, target, hint = goat_retrieval.kss_retrieve(populated_scene, goal, cfg)
        assert hit is True


# ---------------------------------------------------------------------------
# Tests: find_target_in_selected
# ---------------------------------------------------------------------------
class TestFindTargetInSelected:
    def test_object_type_match(self):
        selected = {1: make_obj("tv"), 2: make_obj("sofa")}
        goal = GoalInfo(type="object", category="tv")
        result = goat_retrieval.find_target_in_selected(selected, goal, FakeScene())
        assert result is not None
        assert result["class_name"] == "tv"

    def test_object_type_no_match(self):
        selected = {1: make_obj("bed"), 2: make_obj("sofa")}
        goal = GoalInfo(type="object", category="tv")
        result = goat_retrieval.find_target_in_selected(selected, goal, FakeScene())
        assert result is None

    def test_description_type_returns_none(self):
        selected = {1: make_obj("tv"), 2: make_obj("sofa")}
        goal = GoalInfo(type="description", lang_desc="a comfy seat")
        result = goat_retrieval.find_target_in_selected(selected, goal, FakeScene())
        assert result is None

    def test_image_type_returns_none(self):
        selected = {1: make_obj("tv"), 2: make_obj("sofa")}
        goal = GoalInfo(type="image", image_goal="base64data")
        result = goat_retrieval.find_target_in_selected(selected, goal, FakeScene())
        assert result is None

    def test_object_case_insensitive(self):
        selected = {1: make_obj("TV")}
        goal = GoalInfo(type="object", category="tv")
        result = goat_retrieval.find_target_in_selected(selected, goal, FakeScene())
        assert result is not None

    def test_empty_selected(self):
        goal = GoalInfo(type="object", category="tv")
        result = goat_retrieval.find_target_in_selected({}, goal, FakeScene())
        assert result is None


# ---------------------------------------------------------------------------
# Tests: generate_exploration_hint
# ---------------------------------------------------------------------------
class TestGenerateExplorationHint:
    def test_returns_string(self, populated_scene, cfg, monkeypatch):
        """Hint should be a string from call_openai_api."""
        monkeypatch.setattr(
            goat_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "A hint text.",
        )
        goal = GoalInfo(type="object", category="tv")
        hint = goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        assert hint == "A hint text."

    def test_none_response_returns_empty(self, populated_scene, cfg, monkeypatch):
        monkeypatch.setattr(
            goat_retrieval, "call_openai_api",
            lambda sys_prompt, contents: None,
        )
        goal = GoalInfo(type="object", category="tv")
        hint = goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        assert hint == ""

    def test_sys_prompt_correct(self, populated_scene, cfg, monkeypatch):
        """Verify sys_prompt passed to call_openai_api."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["sys"] = sys_prompt
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        goal = GoalInfo(type="object", category="tv")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        assert captured["sys"] == "You are an AI agent exploring a 3D indoor scene for navigation."

    def test_prompt_includes_goal_text(self, populated_scene, cfg, monkeypatch):
        """user_prompt should contain target description."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        goal = GoalInfo(type="description", lang_desc="a red chair")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        user_text = captured["contents"][0][0]
        assert "a red chair" in user_text

    def test_prompt_includes_graph(self, populated_scene, cfg, monkeypatch):
        """user_prompt should contain serialized scene graph."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        goal = GoalInfo(type="object", category="tv")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        user_text = captured["contents"][0][0]
        assert "1: tv" in user_text
        assert "3: bed" in user_text

    def test_selected_objs_note_in_prompt(self, populated_scene, cfg, monkeypatch):
        """When selected_objs provided, prompt should mention KSS selected."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        selected = {2: make_obj("sofa")}
        goal = GoalInfo(type="object", category="tv")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg, selected_objs=selected)
        user_text = captured["contents"][0][0]
        assert "KSS selected" in user_text

    def test_no_selected_note_when_none(self, populated_scene, cfg, monkeypatch):
        """When selected_objs is None, no KSS note in prompt."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        goal = GoalInfo(type="object", category="tv")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg, selected_objs=None)
        user_text = captured["contents"][0][0]
        assert "KSS selected" not in user_text

    def test_image_goal_description(self, populated_scene, cfg, monkeypatch):
        """Image goal → target_desc mentions reference image."""
        captured = {}

        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "hint"

        monkeypatch.setattr(goat_retrieval, "call_openai_api", fake_api)
        goal = GoalInfo(type="image", image_goal="base64")
        goat_retrieval.generate_exploration_hint(populated_scene, goal, cfg)
        user_text = captured["contents"][0][0]
        assert "reference image" in user_text
