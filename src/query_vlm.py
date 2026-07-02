import logging
from typing import Tuple, Optional, Union

from src.pred_eqa import explore_step
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from src.scene_vlm_only import Scene

def query_vlm_for_response(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    agent_position_history,
    rgb_egocentric_views: list,
    cfg,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
    step_dict = {}

    step_dict["obj_map"] = {}
    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = []  # No detected objects
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]

    # prepare frontier
    step_dict["frontier_imgs"] = [frontier.feature for frontier in tsdf_planner.frontiers]
    step_dict["frontier_pos"] = [frontier.position for frontier in tsdf_planner.frontiers]
    step_dict["agent_position_history"] = agent_position_history

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare question
    step_dict["question"] = question

    
    # Add current step and position information for memory recording
    step_dict["current_step"] = len(agent_position_history) 
    step_dict["current_position"] = agent_position_history[-1] if agent_position_history else [0, 0, 0]
    step_dict["scene"] = scene 
    step_dict["tsdf_planner"] = tsdf_planner 
    step_dict["snapshots"] = scene.snapshots 
    
    outputs, snapshot_id_mapping, reason, n_filtered_snapshots, eliminate_list = explore_step(  # TODO
        step_dict, cfg, verbose=verbose
    )

    if eliminate_list is not None:
        eliminate_set = set(eliminate_list)
        filtered_snapshots = {}
        for i, (k, v) in enumerate(scene.snapshots.items()):
            if i not in eliminate_set:
                filtered_snapshots[k] = v
            else:
                scene.filtered_snapshots.add(k)
        
        scene.snapshots = filtered_snapshots
        logging.info(f"After filtering: {len(scene.snapshots)} snapshots remain")
        if not scene.snapshots:
            scene.clear_up_detections()

    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None

    # parse returned results
    try:
        parts = outputs.split(" ")
        target_type = parts[0]
        if target_type == "stop":
            # Special case: "stop exploration" response
            logging.info(f"Received stop exploration command: {outputs}")
            return None
        target_index = parts[1] if len(parts) > 1 else None
        if target_index is None:
            logging.info(f"No target index provided in response: {outputs}")
            return None
        logging.info(f"Prediction: {target_type}, {target_index}")
    except Exception as e:
        logging.info(f"Wrong output format, failed! Error: {e}")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    # Handle the case where the agent selects a snapshot - in pure VLM approach, this should terminate exploration
    if target_type == "snapshot":
        try:
            vlm_snapshot_index = int(target_index)
        except ValueError:
            logging.info(f"Invalid snapshot target index: {target_index}, failed!")
            return None
        
        if vlm_snapshot_index < 0 or vlm_snapshot_index >= len(list(scene.snapshots.keys())):
            logging.info(
                f"Predicted snapshot target index out of range: {vlm_snapshot_index}, failed! Available snapshots: {len(list(scene.snapshots.keys()))}"
            )
            return None

        logging.info(f"The index of target snapshot {vlm_snapshot_index}")

        # get the target snapshot
        pred_target_snapshot = list(scene.snapshots.values())[vlm_snapshot_index]
        logging.info(f"Selected snapshot {pred_target_snapshot.image} as answer, exploration should stop")
        return pred_target_snapshot, reason, n_filtered_snapshots
    else:  # target_type == "frontier"
        if len(tsdf_planner.frontiers) == 0:
            logging.info(
                f"No frontiers available, but VLM selected a frontier. Available frontiers: {len(tsdf_planner.frontiers)}"
            )
            return None
        
        try:
            target_index = int(target_index)
        except ValueError:
            logging.info(f"Invalid frontier target index: {target_index}, failed!")
            return None
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed! Available frontiers: {len(tsdf_planner.frontiers)}"
            )
            return None
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        # TODO
        reason = ''

        return pred_target_frontier, reason, n_filtered_snapshots