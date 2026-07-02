"""Unit tests for src/goat_runner.py (navigate_to_target, run_exploration,
vlm_check_target_found, check_success).

All dependencies mocked — no habitat_sim / real VLM / real detection.
"""

import os

# Clear proxy env vars before importing src.pred_eqa (OpenAI client at load).
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        del os.environ[_k]

import numpy as np
import pytest
import types


class FakeObj(dict):
    """Mimics MapObjectDict: supports obj['key'] and obj.key access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


class FakePCD:
    def __init__(self, points):
        self.points = points


class FakeBBox:
    def __init__(self, center):
        self._c = np.asarray(center, dtype=float)
    def center(self):
        return self._c


def make_obj(class_name="chair", points=None, bbox_center=None):
    obj = FakeObj()
    obj["class_name"] = class_name
    if points is None:
        points = np.array([[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]])
    obj["pcd"] = FakePCD(points)
    if bbox_center is not None:
        obj["bbox"] = FakeBBox(bbox_center)
    return obj


class FakeCfg:
    def __init__(self, **kw):
        self.prompt_h = 32
        self.prompt_w = 32
        self.hfov = 90
        self.img_height = 64
        self.img_width = 64
        self.dicision_radius = 1.0
        self.success_distance = 1.0
        self.max_steps_per_subtask = 3
        self.extra_view_phase_1 = 2
        self.extra_view_angle_deg_phase_1 = 60
        self.extra_view_phase_2 = 6
        self.extra_view_angle_deg_phase_2 = 40
        self.margin_h_ratio = 0.6
        self.margin_w_ratio = 0.25
        self.explored_depth = 1.7
        self.egocentric_views = True
        self.planner = FakeObj()
        for k, v in kw.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class FakePlanner:
    """Minimal TSDFPlanner mock."""
    def __init__(self):
        self.max_point = None
        self.target_point = None
        self.frontiers = []
        self.call_count = 0

    def habitat2voxel(self, pts):
        return np.zeros(3, dtype=int)

    def normal2voxel(self, pts):
        return np.zeros(2, dtype=int)

    def integrate(self, **kw):
        pass

    def update_frontier_map(self, **kw):
        return True

    def set_next_navigation_point(self, **kw):
        self.max_point = kw.get("choice")
        self.target_point = np.zeros(2, dtype=int)
        return True

    def agent_step(self, **kw):
        # Return 6-tuple: pts, angle, voxel, fig, path_points, target_arrived
        self.call_count += 1
        pts = kw.get("pts")
        angle = kw.get("angle")
        arrived = self.call_count >= 2
        return (pts, angle, np.zeros(2, dtype=int), None, None, arrived)


class FakeScene:
    """Minimal Scene mock."""
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else {}
        self.snapshots = {}
        self.all_observations = {}
        self.pathfinder = None

    def get_observation(self, pts, angle):
        rgb = np.zeros((64, 64, 4), dtype=np.uint8)
        depth = np.ones((64, 64), dtype=np.float32)
        return ({"color_sensor": rgb, "depth_sensor": depth},
                np.eye(4))

    def update_scene_graph(self, **kw):
        return (None, [], None)

    def update_snapshots(self):
        pass

    def sanity_check(self, cfg=None):
        pass

    def reset_for_new_subtask(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg():
    return FakeCfg()


@pytest.fixture
def goal_object():
    from src.goal_types import GoalInfo
    return GoalInfo(type="object", category="chair", viewpoints=[])


@pytest.fixture
def goal_desc():
    from src.goal_types import GoalInfo
    return GoalInfo(type="description", lang_desc="a red sofa", viewpoints=[])


@pytest.fixture
def goal_image():
    from src.goal_types import GoalInfo
    return GoalInfo(type="image", image_goal="base64str", viewpoints=[])


# ---------------------------------------------------------------------------
# vlm_check_target_found
# ---------------------------------------------------------------------------
class TestVlmCheckTargetFound:
    def test_object_yes(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        monkeypatch.setattr(goat_runner, "call_openai_api", lambda *a, **k: "yes")
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, goal_object, cfg) is True

    def test_object_no(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        monkeypatch.setattr(goat_runner, "call_openai_api", lambda *a, **k: "no")
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, goal_object, cfg) is False

    def test_none_response(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        monkeypatch.setattr(goat_runner, "call_openai_api", lambda *a, **k: None)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, goal_object, cfg) is False

    def test_description_type(self, monkeypatch, cfg, goal_desc):
        from src import goat_runner
        captured = {}
        def fake_api(sys_prompt, contents):
            captured["contents"] = contents
            return "yes"
        monkeypatch.setattr(goat_runner, "call_openai_api", fake_api)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, goal_desc, cfg) is True
        # prompt contains description text
        assert "red sofa" in captured["contents"][0][0]

    def test_image_with_ref(self, monkeypatch, cfg, goal_image):
        from src import goat_runner
        captured = {}
        def fake_api(sys_prompt, contents):
            captured["n"] = len(contents)
            return "yes"
        monkeypatch.setattr(goat_runner, "call_openai_api", fake_api)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, goal_image, cfg) is True
        assert captured["n"] == 2  # reference + current

    def test_image_no_ref_fallback(self, monkeypatch, cfg):
        from src.goal_types import GoalInfo
        from src import goat_runner
        gi = GoalInfo(type="image", image_goal=None)
        monkeypatch.setattr(goat_runner, "call_openai_api", lambda *a, **k: "yes")
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        assert goat_runner.vlm_check_target_found(rgb, gi, cfg) is True


# ---------------------------------------------------------------------------
# check_success
# ---------------------------------------------------------------------------
class TestCheckSuccess:
    def test_no_viewpoints(self):
        from src.goat_runner import check_success
        assert check_success(np.array([0, 0, 0]), [], None, 1.0) is False

    def test_close_euclidean(self, monkeypatch):
        from src import goat_runner
        # force euclidean fallback by making habitat_sim import fail
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name == "habitat_sim":
                raise ImportError("stub")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        pts = np.array([0.0, 0.0, 0.0])
        vps = [np.array([0.5, 0.0, 0.0])]
        assert goat_runner.check_success(pts, vps, None, 1.0) is True

    def test_far_euclidean(self, monkeypatch):
        from src import goat_runner
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name == "habitat_sim":
                raise ImportError("stub")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        pts = np.array([0.0, 0.0, 0.0])
        vps = [np.array([10.0, 0.0, 0.0])]
        assert goat_runner.check_success(pts, vps, None, 1.0) is False


# ---------------------------------------------------------------------------
# navigate_to_target
# ---------------------------------------------------------------------------
class TestNavigateToTarget:
    def test_success(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        scene = FakeScene(objects={1: make_obj("chair", bbox_center=[1, 1, 1])})
        planner = FakePlanner()

        # VDD returns a viewpoint
        monkeypatch.setattr(
            goat_runner, "Visibility_based_Viewpoint_Decision",
            lambda *a, **k: np.array([0.5, 0.5, 0.5]),
        )
        monkeypatch.setattr(
            goat_runner, "check_success", lambda *a, **k: True,
        )

        pts = np.array([0.0, 1.5, 0.0])
        angle = 0.0
        success, pts2, angle2 = goat_runner.navigate_to_target(
            scene, planner, scene.objects[1], goal_object, cfg, pts, angle,
        )
        assert success is True
        # planner.agent_step was called → target_point set
        assert planner.target_point is not None

    def test_vdd_none_no_bbox(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        obj = FakeObj()
        obj["pcd"] = FakePCD(np.array([[1.0, 1.0, 1.0]]))
        # no bbox key
        scene = FakeScene(objects={})
        planner = FakePlanner()
        monkeypatch.setattr(
            goat_runner, "Visibility_based_Viewpoint_Decision",
            lambda *a, **k: None,
        )
        pts = np.array([0.0, 1.5, 0.0])
        success, pts2, angle2 = goat_runner.navigate_to_target(
            scene, planner, obj, goal_object, cfg, pts, 0.0,
        )
        assert success is False

    def test_vdd_none_bbox_fallback(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        obj = make_obj("chair", bbox_center=[1, 1, 1])
        scene = FakeScene(objects={})
        planner = FakePlanner()
        monkeypatch.setattr(
            goat_runner, "Visibility_based_Viewpoint_Decision",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            goat_runner, "select_navigation_corner",
            lambda *a, **k: np.array([0.5, 0.5, 0.5]),
        )
        monkeypatch.setattr(
            goat_runner, "check_success", lambda *a, **k: True,
        )
        pts = np.array([0.0, 1.5, 0.0])
        success, _, _ = goat_runner.navigate_to_target(
            scene, planner, obj, goal_object, cfg, pts, 0.0,
        )
        assert success is True


# ---------------------------------------------------------------------------
# run_exploration
# ---------------------------------------------------------------------------
class TestRunExploration:
    def test_vlm_finds_target_immediately(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        scene = FakeScene(objects={})
        planner = FakePlanner()

        # vlm_check_target_found → True on step 0, check_success → True
        monkeypatch.setattr(goat_runner, "vlm_check_target_found", lambda *a, **k: True)
        monkeypatch.setattr(goat_runner, "check_success", lambda *a, **k: True)

        pts = np.array([0.0, 1.5, 0.0])
        success, pts2, angle2 = goat_runner.run_exploration(
            scene, planner, goal_object, hint="go left", cfg=cfg,
            pts=pts, angle=0.0,
        )
        assert success is True

    def test_max_steps_no_find(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        scene = FakeScene(objects={})
        planner = FakePlanner()
        # planner needs max_point handling for query_vlm path
        planner.max_point = None
        planner.target_point = None

        monkeypatch.setattr(goat_runner, "vlm_check_target_found", lambda *a, **k: False)
        monkeypatch.setattr(goat_runner, "check_success", lambda *a, **k: False)

        # query_vlm returns a Frontier choice
        from src.tsdf_planner import Frontier
        ft = types.SimpleNamespace()
        # use SnapShot type to trigger early return path? No—return Frontier
        class FakeFrontier:
            pass
        fake_ft = FakeFrontier()
        monkeypatch.setattr(
            goat_runner, "query_vlm_for_response",
            lambda **k: (fake_ft, "answer", 1),
        )
        # set_next_navigation_point + agent_step succeed; loop exhausts steps
        monkeypatch.setattr(
            goat_runner, "SnapShot", type("SnapShot", (), {}),
        )

        pts = np.array([0.0, 1.5, 0.0])
        success, pts2, angle2 = goat_runner.run_exploration(
            scene, planner, goal_object, hint="", cfg=cfg,
            pts=pts, angle=0.0,
        )
        assert success is False

    def test_snapshot_choice_returns_true(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        scene = FakeScene(objects={})
        planner = FakePlanner()

        monkeypatch.setattr(goat_runner, "vlm_check_target_found", lambda *a, **k: False)

        # query_vlm returns a SnapShot → immediate success
        class FakeSnapShot:
            pass
        snap = FakeSnapShot()
        monkeypatch.setattr(
            goat_runner, "query_vlm_for_response",
            lambda **k: (snap, "answer", 1),
        )
        # make type() check match: patch SnapShot in goat_runner namespace
        monkeypatch.setattr(goat_runner, "SnapShot", FakeSnapShot)

        pts = np.array([0.0, 1.5, 0.0])
        success, _, _ = goat_runner.run_exploration(
            scene, planner, goal_object, hint="hint", cfg=cfg,
            pts=pts, angle=0.0,
        )
        assert success is True

    def test_object_detected_navigates(self, monkeypatch, cfg, goal_object):
        from src import goat_runner
        # scene has a matching object → triggers navigate_to_target
        scene = FakeScene(objects={1: make_obj("chair", bbox_center=[1, 1, 1])})
        planner = FakePlanner()

        monkeypatch.setattr(goat_runner, "vlm_check_target_found", lambda *a, **k: False)

        class FakeFrontier:
            pass
        fake_ft = FakeFrontier()
        monkeypatch.setattr(
            goat_runner, "query_vlm_for_response",
            lambda **k: (fake_ft, "answer", 1),
        )
        monkeypatch.setattr(
            goat_runner, "SnapShot", type("SnapShot", (), {}),
        )
        # navigate_to_target mocked to succeed
        monkeypatch.setattr(
            goat_runner, "navigate_to_target",
            lambda *a, **k: (True, k.get("pts", a[5]), k.get("angle", a[6])),
        )

        pts = np.array([0.0, 1.5, 0.0])
        success, _, _ = goat_runner.run_exploration(
            scene, planner, goal_object, hint="", cfg=cfg,
            pts=pts, angle=0.0,
        )
        assert success is True
