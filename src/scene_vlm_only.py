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
from src.conceptgraph.slam.slam_classes import MapObjectDict, MapEdge

# New text-based long-term memory and planning imports
from src.scene_integration import SceneIntegration


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
        detection_model=None,
        sam_predictor=None,
        clip_model=None,
        clip_preprocess=None,
        clip_tokenizer=None,
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

        # Object detection module - optional detection stack (MSGNav-style)
        # If detection_model provided, runs YOLO+SAM+CLIP in parallel with VLM
        self.detection_model = detection_model
        self.sam_predictor = sam_predictor
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.obj_classes = None
        if detection_model is not None and os.path.exists(scene_semantic_annotation_path):
            from src.conceptgraph.utils.general_utils import ObjectClasses
            self.obj_classes = ObjectClasses(
                classes_file_path=scene_semantic_annotation_path,
                bg_classes=self.cfg_cg["bg_classes"],
                skip_bg=self.cfg_cg["skip_bg"],
                class_set=cfg.class_set,
            )
            self.detection_model.set_classes(self.obj_classes.get_classes_arr())
            self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen room', 'laundry room', 'others']
            self.edges: Dict[tuple, Any] = {}
            self.img_to_edge: Dict[str, list] = {}
            logging.info(f"Detection stack enabled: {len(self.obj_classes.get_classes_arr())} classes loaded")

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

    def reset_for_new_subtask(self):
        """子任务切换：清 per-subtask 状态，保留 objects/edges/img_to_edge（跨子任务记忆）。"""
        self.snapshots = {}
        self.frames = {}
        self.all_observations = {}
        self.filtered_snapshots = set()
        self.text_memory_system = SceneIntegration(self)
        self.long_term_memory = self.text_memory_system.long_term_memory
        # objects/edges/img_to_edge 保留 — 跨子任务场景图记忆

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

    def update_scene_graph(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path,
        frame_idx,
        target_obj_mask=None,
        vlm_model=None,
        vlm_processor=None,
    ) -> Tuple[np.ndarray, List[int], Optional[int]]:
        """Update scene graph. Two parallel paths:
        1. Detection stack (if models loaded): YOLO+SAM+CLIP → self.objects with image_path
        2. VLM path (always): store raw image in all_observations + frames for snapshot
        Snapshot generation is NOT affected by detection stack.
        """
        added_obj_ids = []

        # --- Detection stack path (parallel, does not touch snapshots) ---
        if self.detection_model is not None and self.obj_classes is not None:
            try:
                added_obj_ids = self._run_detection_stack(
                    image_rgb, depth, intrinsics, cam_pos, pts, img_path
                )
            except Exception as e:
                logging.warning(f"Detection stack failed (non-fatal, VLM path continues): {e}")
                added_obj_ids = []

        # --- VLM path (always runs, untouched from original) ---
        frame = SnapShot(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
        )
        self.all_observations[img_path] = image_rgb
        self.frames[img_path] = frame

        target_obj_id = None
        return image_rgb, added_obj_ids, target_obj_id

    def _run_detection_stack(
        self, image_rgb, depth, intrinsics, cam_pos, pts, img_path
    ) -> List[int]:
        """MSGNav-style detection stack. Fills self.objects with image_path tracking.
        Does NOT touch self.snapshots or self.frames."""
        import supervision as sv
        from src.conceptgraph.utils.ious import mask_subtract_contained
        from src.conceptgraph.utils.general_utils import filter_detections
        from src.conceptgraph.utils.model_utils import compute_clip_features_batched
        from src.conceptgraph.slam.utils import (
            resize_gobs, filter_gobs, init_process_pcd, get_bounding_box,
            detections_to_obj_pcd_and_bbox,
        )
        from src.conceptgraph.utils.general_utils import measure_time
        from src.conceptgraph.slam.mapping import (
            compute_spatial_similarities, compute_visual_similarities,
            aggregate_similarities, match_detections_to_objects,
        )

        obj_classes = self.obj_classes
        cfg_cg = self.cfg_cg

        # 0. Room detection (CLIP-based)
        use_room_det = self.cfg.get("use_room_det", False)
        if use_room_det:
            from src.conceptgraph.utils.model_utils import compute_clip_features_batched_check
            sim = compute_clip_features_batched_check(
                image_list=[image_rgb],
                clip_model=self.clip_model.to("cuda"),
                clip_tokenizer=self.clip_tokenizer,
                clip_preprocess=self.clip_preprocess,
                text_goal=np.array(self.rooms),
                image_goal=None,
                extra_text=False
            ).detach().cpu().numpy()
            room_label = self.rooms[np.argmax(sim)]
            room_conf = np.max(sim).item()
        else:
            room_label = "unknown"
            room_conf = 0.0

        # 1. YOLO detection
        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # 2. SAM segmentation
        if xyxy_tensor.numel() != 0:
            sam_out = self.sam_predictor.predict(image_rgb, bboxes=xyxy_tensor, verbose=False)
            masks_np = sam_out[0].masks.data.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        curr_det = sv.Detections(
            xyxy=xyxy_np, confidence=confidences,
            class_id=detection_class_ids, mask=masks_np,
        )
        if len(curr_det) == 0:
            return []

        # 3. filter detections
        curr_det, labels = filter_detections(
            image=image_rgb, detections=curr_det, classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=cfg_cg.min_mask_size_ratio,
            confidence_threshold=cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            return []

        # 4. CLIP features
        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb, curr_det, self.clip_model, self.clip_preprocess,
            self.clip_tokenizer, obj_classes.get_classes_arr(), self.device,
        )

        raw_gobs = {
            "xyxy": curr_det.xyxy, "confidence": curr_det.confidence,
            "class_id": curr_det.class_id, "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops, "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
            "room_label": room_label, "room_conf": room_conf,
        }

        # 5. resize + filter gobs
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        filtered_gobs = filter_gobs(
            resized_gobs, image_rgb,
            skip_bg=cfg_cg["skip_bg"],
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=cfg_cg.mask_conf_threshold,
        )
        gobs = filtered_gobs
        if len(gobs["mask"]) == 0:
            return []

        # 6. mask subtract + point cloud backprojection
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])
        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth, masks=gobs["mask"], cam_K=intrinsics[:3, :3],
            image_rgb=image_rgb, trans_pose=cam_pos,
            min_points_threshold=cfg_cg.min_points_threshold,
            spatial_sim_type=cfg_cg["spatial_sim_type"],
            obj_pcd_max_points=cfg_cg.obj_pcd_max_points, device=self.device,
        )
        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=cfg_cg["dbscan_eps"],
                    dbscan_min_points=cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg_cg["spatial_sim_type"], pcd=obj["pcd"],
                )

        if all([obj is None for obj in obj_pcds_and_bboxes]):
            return []

        gobs["bbox"] = [obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes]
        gobs["pcd"] = [obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes]

        # 7. filter by distance
        gobs = self.filter_gobs_with_distance(pts, gobs)

        # 8. make detection list (with image_path / image_path_list)
        detection_list = self.make_detection_list_from_pcd_and_gobs(gobs, img_path, obj_classes)
        if len(detection_list) == 0:
            return []

        # 9. match + merge (or add all if first frame)
        if len(self.objects) == 0:
            self.objects.update(detection_list)
            added_obj_ids = list(detection_list.keys())
        else:
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg_cg["spatial_sim_type"],
                detection_list=detection_list, objects=self.objects,
                downsample_voxel_size=cfg_cg["downsample_voxel_size"],
            )
            visual_sim = compute_visual_similarities(detection_list, self.objects)
            agg_sim = aggregate_similarities(
                match_method=cfg_cg["match_method"], phys_bias=cfg_cg["phys_bias"],
                spatial_sim=spatial_sim, visual_sim=visual_sim,
            )
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim, detection_threshold=cfg_cg["sim_threshold"],
                existing_obj_ids=list(self.objects.keys()),
                detected_obj_ids=list(detection_list.keys()),
            )
            added_obj_ids = self.merge_obj_matches(
                detection_list=detection_list, match_indices=match_indices,
                obj_classes=obj_classes,
            )

        # 10. update scene graph edges (MapEdge co-occurrence)
        try:
            self.update_scene_graph_edges(added_obj_ids, img_path)
        except Exception as e:
            print(f"[detection] edge update failed: {e}")

        return added_obj_ids

    def update_scene_graph_edges(self, frame_obj_ids, img_path: str):
        """Build co-occurrence edges between objects in same frame. (From MSGNav)"""
        self.img_to_edge[img_path] = []
        for a_obj_id in frame_obj_ids:
            for b_obj_id in frame_obj_ids:
                if a_obj_id > b_obj_id:
                    continue
                obj1 = self.objects.get(a_obj_id)
                obj2 = self.objects.get(b_obj_id)
                if obj1 is None or obj2 is None:
                    continue
                if obj1["bbox"] is None or obj2["bbox"] is None:
                    continue
                obj1_center = obj1["bbox"].center[[0, 2]]
                obj2_center = obj2["bbox"].center[[0, 2]]
                if np.linalg.norm(obj1_center - obj2_center) > self.cfg_cg["edge_dist_threshold"]:
                    continue
                if (a_obj_id, b_obj_id) in self.edges:
                    self.edges[(a_obj_id, b_obj_id)].rel_img.append(img_path)
                    self.edges[(a_obj_id, b_obj_id)].num_detections += 1
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))
                else:
                    edge = MapEdge(obj1_idx=a_obj_id, obj2_idx=b_obj_id, rel_img=img_path, num_detections=1)
                    self.edges[(a_obj_id, b_obj_id)] = edge
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:
                        self.edges[(b_obj_id, a_obj_id)] = edge
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))

    def del_unused_scene_graph_edges(self):
        """Remove edges referencing deleted objects. (From MSGNav)"""
        valid_ids = set(self.objects.keys())
        to_del = []
        for (a, b) in list(self.edges.keys()):
            if a not in valid_ids or b not in valid_ids:
                to_del.append((a, b))
        for k in to_del:
            del self.edges[k]
        for img_path in list(self.img_to_edge.keys()):
            self.img_to_edge[img_path] = [
                (a, b) for (a, b) in self.img_to_edge[img_path]
                if a in valid_ids and b in valid_ids
            ]
            if len(self.img_to_edge[img_path]) == 0:
                del self.img_to_edge[img_path]

    def filter_gobs_with_distance(self, pts, gobs):
        """Filter out objects too far from observation point. (From MSGNav)"""
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:
                continue
            if np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]]) > self.cfg.scene_graph.obj_include_dist:
                continue
            idx_to_keep.append(idx)
        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or isinstance(gobs[attribute], float) or attribute == "classes":
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")
        return gobs

    def make_detection_list_from_pcd_and_gobs(self, gobs, image_path, obj_classes):
        """Build detection list with image_path / image_path_list tracking. (From MSGNav)"""
        from src.conceptgraph.slam.slam_classes import DetectionDict, to_tensor
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:
                continue
            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)
            detected_object = {
                "id": self.object_id_counter,
                "class_name": curr_class_name,
                "class_id": [curr_class_idx],
                "num_detections": 1,
                "conf": gobs["confidence"][mask_idx],
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "mask_num": gobs["mask"][mask_idx].sum(),
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                "image": None,
                "room_label": gobs["room_label"],
                "room_conf": gobs["room_conf"],
                "image_path": image_path,
                "image_path_list": [image_path],
            }
            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1
        return detection_list

    def merge_obj_matches(self, detection_list, match_indices, obj_classes):
        """Merge detected objects into existing objects. (Simplified from MSGNav)"""
        from src.conceptgraph.slam.utils import merge_obj2_into_obj1
        from collections import Counter
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                added_obj_ids.append(detected_obj_id)
            else:
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]
                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj, obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    device=self.device, run_dbscan=False,
                )
                class_id_counter = Counter(merged_obj["class_id"])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                merged_obj["class_name"] = obj_classes.get_classes_arr()[most_common_class_id]
                self.objects[existing_obj_match_id] = merged_obj
        return added_obj_ids

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
