import os
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
from quaternion import as_float_array
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union, Any
import copy

from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from src.habitat import (
    make_semantic_cfg,
    make_simple_cfg,
    get_quaternion,
    get_navigable_point_to,
)
from src.geom import get_cam_intr, IoU
from src.utils import rgba2rgb
from src.tsdf_planner import SnapShot

# Pure VLM imports
from src.conceptgraph.slam.slam_classes import MapObjectDict

# New text-based long-term memory and planning imports
from src.scene_integration import SceneIntegration


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
    ):
        self.cfg = cfg
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # about the loading the scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(
            cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".basis.navmesh",
        )
        semantic_texture_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.glb",
        )
        scene_semantic_annotation_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.txt",
        )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        if not os.path.exists(semantic_texture_path) or not os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.warning(
                f"semantic_texture_path: {semantic_texture_path} or scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"
            )

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            sim_cfg = make_semantic_cfg(sim_settings)
        else:
            sim_cfg = make_simple_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # Object detection module deprecated for the simplified VLM-only approach,
        # so we no longer load object classes here.

        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.info(f"Load scene {scene_id} successfully with semantic texture")
        else:
            logging.info(f"Load scene {scene_id} successfully without semantic texture")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.img_width, cfg.img_height, cfg.hfov)

        # about scene graph - for simplified VLM, we store only image observations
        # Still kept for compatibility with TSDF planner but will be mostly empty
        # This is needed for the TSDF planner's agent_step method
        self.objects = (
            MapObjectDict()
        )  # Kept for compatibility with TSDF planner, but mostly unused
        self.object_id_counter = 1

        self.snapshots: Dict[str, SnapShot] = {}  # image_path -> snapshot
        self.frames: Dict[str, SnapShot] = {}  # image_path -> all frames
        self.all_observations: Dict[str, np.ndarray] = (
            {}
        )  # image_path -> image, stores all actual observations at each step
        self.filtered_snapshots: set = set() # keep track of snapshots that were filtered out by VLM

        # Initialize new text-based memory and planning system
        self.text_memory_system = SceneIntegration(self)

        # Initialize new text-based memory and planning components
        self.long_term_memory = self.text_memory_system.long_term_memory

        
        # No detection, segmentation, or CLIP models in simplified VLM approach
        logging.info("Initialized Simplified VLM Scene - images go directly to VLM")

    def close(self):
        """显式关闭模拟器资源"""
        try:
            if hasattr(self, 'simulator') and self.simulator is not None:
                self.simulator.close()
                self.simulator = None
        except Exception as e:
            logging.warning(f"Error closing simulator: {e}")
    
    def __del__(self):
        self.close()

    def clear_up_detections(self):
        self.objects = MapObjectDict()
        self.object_id_counter = 1

        self.snapshots = {}
        self.frames = {}
        self.all_observations = {}

    def get_observation(self, pts, angle):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0)
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs

    def get_frontier_observation_and_detect_target(
        self,
        pts,
        view_dir,
        target_obj_id,
        target_obj_class,
        camera_tilt=0.0,
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object - for pure VLM, we'll rely on VLM to identify objects
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        target_detected = target_obj_id in np.unique(semantic_obs)
        
        return obs, target_detected

    def get_navigable_point_to(
        self,
        target_position,
        max_search=1000,
        min_dist=6,
        max_dist=999,
        prev_start_positions=None,
    ):
        self.pathfinder.seed(random.randint(0, 10000))
        return get_navigable_point_to(
            target_position,
            self.pathfinder,
            max_search,
            min_dist,
            max_dist,
            prev_start_positions,
        )

    def update_scene_graph_with_vlm(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path,
        frame_idx,
        target_obj_mask=None,  # the boolean mask of target object generated from the semantic sensor. If given, return the object id of the target object
        vlm_model=None,  # Pure VLM model for scene understanding
        vlm_processor=None,
    ) -> Tuple[np.ndarray, List[int], Optional[int]]:
        """
        Simplified approach: Store the image directly and pass to VLM without any preprocessing
        """
        # Return the original image; no object IDs since we're not doing detection
        added_obj_ids = []
        
        # Create a frame for this observation
        frame = SnapShot(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
        )
        
        # Simply store the raw image for VLM processing
        self.all_observations[img_path] = image_rgb
        self.frames[img_path] = frame
        
        # No complex processing - just pass image directly to VLM
        # In a real implementation, the VLM would be called here or later during decision making
        target_obj_id = None  # We don't detect specific objects anymore
            
        return image_rgb, added_obj_ids, target_obj_id

    def cleanup_empty_frames_snapshots(self):
        """
        For pure VLM approach, we handle snapshots differently
        Instead of removing frames with empty detected objects, we keep all frames
        since we're not doing object detection
        """
        # In pure VLM approach, we don't remove frames based on object detection
        # We keep all frames since we're storing images for VLM processing
        pass

    def update_snapshots(self):
        """
        Simplified approach: Add new frames to snapshots for VLM to evaluate
        In pure VLM approach, we don't cluster objects or perform complex processing
        All frames should be added as snapshots so VLM can evaluate and filter them
        However, we should respect VLM's previous filtering decisions
        """
        # Add all frames that are not already in snapshots and were not previously filtered out
        # This ensures VLM gets to see all new observations for filtering decisions
        # while respecting previous filtering decisions
        for filename, frame in self.frames.items():
            if filename not in self.snapshots and filename not in self.filtered_snapshots:
                self.snapshots[filename] = frame

    def periodic_cleanup_objects(self, frame_idx, pts):
        """
        Simplified approach: No complex object cleanup needed
        since we're not maintaining 3D object representations
        """
        # In simplified approach, minimal cleanup is needed
        # since we're just storing images and basic frame info
        pass

    def sanity_check(self, cfg):
        """
        For simplified VLM approach, the sanity checks are adapted
        """
        # Basic sanity check for simplified VLM approach
        assert len(self.snapshots) <= len(self.frames), f"Snapshots ({len(self.snapshots)}) should not exceed frames ({len(self.frames)})"
        
        # All snapshots should also be in frames
        for snapshot_id in self.snapshots.keys():
            assert snapshot_id in self.frames, f"Snapshot {snapshot_id} not in frames"
        
        # In simplified approach, we don't have complex object relationships to check
        # The objects dict might be mostly empty, which is expected

    def print_scene_graph(self):
        """
        For simplified VLM approach, print a different representation of the scene
        """
        logging.info(f"Simplified VLM Scene Graph - Total snapshots: {len(self.snapshots)}")
        for snapshot_id in self.snapshots.keys():
            logging.info(f"Snapshot: {snapshot_id}")
