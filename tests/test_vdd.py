import sys
import os
import numpy as np
from unittest.mock import MagicMock
from scipy.spatial import KDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vdd import (
    generate_candidate_viewpoints,
    is_point_visible,
    compute_visibility,
    Visibility_based_Viewpoint_Decision,
    select_navigation_corner,
    get_aabb_corner_points,
)


def test_generate_candidate_viewpoints():
    bbox_center = np.array([1.0, 2.0, 3.0])
    radius = 1.5
    pts = np.array([0.0, 0.5, 0.0])  # only pts[1] used (height)
    vps = generate_candidate_viewpoints(bbox_center, radius, pts, num_points=20)
    assert vps.shape == (20, 3)
    # height preserved
    assert np.allclose(vps[:, 1], 0.5)
    # first point at angle 0: x = bbox_center[0]+r, z = bbox_center[2]
    assert np.isclose(vps[0, 0], 1.0 + 1.5)
    assert np.isclose(vps[0, 2], 3.0)


def test_is_point_visible_true():
    # scene points far from the ray -> visible
    scene = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    tree = KDTree(scene)
    viewpoint = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    assert is_point_visible(viewpoint, target, tree, threshold=0.05) is True


def test_is_point_visible_false():
    # scene point blocking the ray -> not visible
    viewpoint = np.array([0.0, 0.0, 0.0])
    target = np.array([2.0, 0.0, 0.0])
    blocker = np.array([1.0, 0.0, 0.0])
    tree = KDTree(blocker.reshape(1, 3))
    assert is_point_visible(viewpoint, target, tree, threshold=0.05) is False


def test_compute_visibility():
    # half visible: one target blocked, one clear
    scene = np.array([[1.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    tree = KDTree(scene)
    viewpoint = np.array([0.0, 0.0, 0.0])
    targets = np.array([
        [2.0, 0.0, 0.0],   # blocked by [1,0,0]
        [0.0, 2.0, 0.0],   # clear
    ])
    vis = compute_visibility(viewpoint, targets, tree)
    assert 0.0 <= vis <= 1.0
    assert np.isclose(vis, 0.5)


def test_visibility_based_viewpoint_decision_found():
    # mock tsdf_planner: mask_true_point returns all, get_near_true_point returns all
    planner = MagicMock()
    planner.mask_true_point.side_effect = lambda vps: vps
    planner.get_near_true_point.side_effect = lambda vps: vps

    # scene points far away so all viewpoints are visible
    scene_points = np.array([[100.0, 100.0, 100.0]])
    target_points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ])
    pts = np.array([0.0, 0.0, 0.0])
    result = Visibility_based_Viewpoint_Decision(target_points, scene_points, pts, planner, radius_factor=1.0)
    assert result is not None
    assert result.shape == (3,)


def test_visibility_based_viewpoint_decision_none():
    # mask_true_point returns empty -> no candidates evaluated, and
    # get_near_true_point also returns empty -> best_viewpoint stays None
    planner = MagicMock()
    planner.mask_true_point.side_effect = lambda vps: np.array([]).reshape(0, 3)
    planner.get_near_true_point.side_effect = lambda vps: np.array([]).reshape(0, 3)

    scene_points = np.array([[100.0, 100.0, 100.0]])
    target_points = np.array([[0.0, 0.0, 0.0]])
    pts = np.array([0.0, 0.0, 0.0])
    result = Visibility_based_Viewpoint_Decision(target_points, scene_points, pts, planner, radius_factor=1.0)
    assert result is None


def test_get_aabb_corner_points():
    aabb = MagicMock()
    aabb.get_min_bound.return_value = np.array([0.0, 0.0, 0.0])
    aabb.get_max_bound.return_value = np.array([2.0, 2.0, 2.0])
    corners = get_aabb_corner_points(aabb)
    assert corners.shape == (4, 3)


def test_select_navigation_corner_closest():
    aabb = MagicMock()
    aabb.get_min_bound.return_value = np.array([0.0, 0.0, 0.0])
    aabb.get_max_bound.return_value = np.array([2.0, 2.0, 2.0])
    robot = np.array([5.0, 0.0, 5.0])
    pt = select_navigation_corner(aabb, "closest_to_robot", robot)
    assert pt.shape == (3,)
