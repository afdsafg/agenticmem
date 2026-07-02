"""VDD (Viewpoint Decision) module ported from MSGNav.

Source: /home/afdsafg/下载/new/MSGNav/src/utils.py, query_vlm.py, tsdf_planner.py
Logic preserved; mask_true_point/get_near_true_point exposed both as
standalone functions (taking tsdf_planner) and as TSDFPlanner methods.
"""
import numpy as np
from scipy.spatial import KDTree

from src.geom import get_nearest_true_point


def generate_candidate_viewpoints(bbox_center, radius, pts, num_points=20):
    """Generate candidate viewpoints around the target bounding box center on a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    viewpoints = []
    for angle in angles:
        x = bbox_center[0] + radius * np.cos(angle)
        y = bbox_center[2] + radius * np.sin(angle)
        z = pts[1]  # Z remains constant
        viewpoints.append(np.array([x, z, y]))
    return np.array(viewpoints)


def is_point_visible(viewpoint, target_point, scene_points_tree, threshold=0.05):
    """Check if a target point is visible from a viewpoint considering scene occlusion."""
    direction = target_point - viewpoint
    view_distance = np.linalg.norm(direction)
    direction /= view_distance

    num_samples = min(1000, int(view_distance / threshold) + 1)
    sample_points = np.array([
        viewpoint + t * direction
        for t in np.linspace(3 * threshold, view_distance - 3 * threshold, num_samples)
    ])

    distances, indices = scene_points_tree.query(sample_points, k=1)
    return not np.any(distances < threshold)


def compute_visibility(viewpoint, target_points, scene_points_tree):
    """Compute visibility of target points from a given viewpoint considering scene occlusion."""
    visible_count = 0
    for target_point in target_points:
        visible = is_point_visible(viewpoint, target_point, scene_points_tree)
        if visible:
            visible_count += 1
    return visible_count / target_points.shape[0]


def mask_true_point(tsdf_planner, viewpoints):
    """Standalone wrapper for TSDFPlanner.mask_true_point.

    ponytail: kept as a function so callers can pass a mock planner in tests
    without monkeypatching the class. Upgrade path: drop this and call
    tsdf_planner.mask_true_point() directly once tests use real planners.
    """
    return tsdf_planner.mask_true_point(viewpoints)


def get_near_true_point(tsdf_planner, viewpoints):
    """Standalone wrapper for TSDFPlanner.get_near_true_point. See mask_true_point."""
    return tsdf_planner.get_near_true_point(viewpoints)


def Visibility_based_Viewpoint_Decision(target_points, scene_points, pts, tsdf_planner, radius_factor):
    """Calculate the best viewpoint from a set of candidate viewpoints."""
    target_points = target_points[np.random.choice(target_points.shape[0], min(1000, target_points.shape[0]), replace=False)]
    scene_points_tree = KDTree(scene_points)
    bbox_center = target_points.mean(axis=0)
    best_visibility = 0
    best_viewpoint = None
    candidate_viewpoints = generate_candidate_viewpoints(bbox_center, radius_factor, pts)
    filtered_viewpoints = tsdf_planner.mask_true_point(candidate_viewpoints)
    for vp in filtered_viewpoints:
        vp[1] += 1.5  # camera height
        visibility_score = compute_visibility(vp, target_points, scene_points_tree)
        vp[1] -= 1.5
        if visibility_score > best_visibility:
            best_visibility = visibility_score
            best_viewpoint = vp
    if best_viewpoint is None:
        near_viewpoints = tsdf_planner.get_near_true_point(candidate_viewpoints)
        for vp in near_viewpoints:
            vp[1] += 1.5
            visibility_score = compute_visibility(vp, target_points, scene_points_tree)
            vp[1] -= 1.5
            if visibility_score > best_visibility:
                best_visibility = visibility_score
                best_viewpoint = vp
    print("Best viewpoint:", best_viewpoint)
    return best_viewpoint


def get_aabb_corner_points(aabb):
    """Get 4 navigation-relevant corner points from an Open3D AABB.

    ponytail: MSGNav returns 4 midpoints (not the 8 raw corners). Preserved
    as-is for select_navigation_corner compatibility.
    """
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    return np.array([
        [min_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [max_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], min_bound[2]],
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], max_bound[2]],
    ])


def select_navigation_corner(aabb, selection_strategy="closest_to_robot", robot_position=None):
    """Select a bounding box corner point as the navigation target.

    Parameters:
    aabb: Open3D Axis-Aligned Bounding Box object
    selection_strategy: "closest_to_robot", "lowest", "front_center"
    robot_position: (3,) array [x, y, z] (required for robot-dependent strategies)
    """
    corners = get_aabb_corner_points(aabb)

    if selection_strategy == "closest_to_robot" and robot_position is not None:
        distances = np.linalg.norm(corners[:, [0, 2]] - robot_position[[0, 2]], axis=1)
        return corners[np.argmin(distances)]

    elif selection_strategy == "lowest":
        return corners[np.argmin(corners[:, 1] - robot_position[1])]

    elif selection_strategy == "front_center" and robot_position is not None:
        center = aabb.get_center()
        to_robot = robot_position - center
        to_robot[2] = 0
        to_robot /= np.linalg.norm(to_robot)

        face_centers = [
            np.mean(corners[[0, 1, 2, 3]], axis=0),
            np.mean(corners[[4, 5, 6, 7]], axis=0),
            np.mean(corners[[0, 1, 4, 5]], axis=0),
            np.mean(corners[[2, 3, 6, 7]], axis=0),
        ]
        face_directions = [
            np.array([-1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([0, 1, 0]),
        ]
        dot_products = [np.dot(d, to_robot) for d in face_directions]
        selected_face = np.argmax(dot_products)

        face_corners = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [0, 1, 4, 5],
            3: [2, 3, 6, 7],
        }[selected_face]
        face_points = corners[face_corners]
        return face_points[np.argmin(face_points[:, 2])]

    else:
        return corners[np.argmin(corners[:, 2])]
