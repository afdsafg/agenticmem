import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use('Agg')  # 确保使用非交互式后端


import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt


from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_vlm_only import Scene  
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm import query_vlm_for_response 
from src.logger import Logger
from src.const import *
from src.habitat import pos_normal_to_habitat, pos_habitat_to_normal

import base64
import torch

def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    # only process a subset of the questions
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")


    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
        if any([invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        question = question_data["question"]
        answer = question_data["answer"]
        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
        )

        try:
            if 'scene' in locals():
                scene.close() 
        except:
            pass
        scene = Scene(
            scene_id,
            cfg,
            cfg_cg,
        )

        # initialize the TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=get_scene_bnds(scene.pathfinder, floor_height=pts[1])[0],
            voxel_size=cfg.tsdf_grid_size,
            floor_height=pts[1],
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
            save_visualization=cfg.save_visualization,
        )

        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],
            )
        )

        logging.info(f"\n\nQuestion id {question_id} initialization successful!")

        # run steps
        task_success = False
        cnt_step = -1

        gpt_answer = None
        n_filtered_snapshots = 0
        agent_position_history = [] 
        while cnt_step < cfg.num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")
            
            try:
                # (1) Observe the surroundings, update the scene graph and occupancy map
                # Determine the viewing angles for the current step
                if cnt_step == 0:
                    angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                    total_views = 1 + cfg.extra_view_phase_2
                else:
                    angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                    total_views = 1 + cfg.extra_view_phase_1
                all_angles = [
                    angle + angle_increment * (i - total_views // 2)
                    for i in range(total_views)
                ]
                # Let the main viewing angle be the last one to avoid potential overwriting problems
                main_angle = all_angles.pop(total_views // 2)
                all_angles.append(main_angle)

                rgb_egocentric_views = []  # TODO
                all_added_obj_ids = []  # No object IDs in simplified approach
                for view_idx, ang in enumerate(all_angles):
                    # For each view
                    obs, cam_pose = scene.get_observation(pts, ang)
                    rgb = obs["color_sensor"]
                    depth = obs["depth_sensor"]

                    obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                    scene.all_observations[obs_file_name] = rgb[..., :3]
                    
                    from src.tsdf_planner import SnapShot
                    frame = SnapShot(
                        image=obs_file_name,
                        color=(random.random(), random.random(), random.random()),
                        obs_point=tsdf_planner.habitat2voxel(pts),
                    )
                    scene.frames[obs_file_name] = frame
                    
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    rgb_egocentric_views.append(resized_rgb)

                    if cfg.save_visualization:
                        plt.imsave(
                            os.path.join(eps_snapshot_dir, obs_file_name), rgb
                        )
                    else:
                        plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)

                    scene.periodic_cleanup_objects(
                        frame_idx=cnt_step * total_views + view_idx, pts=pts
                    )

                    # Update depth map, occupancy map
                    tsdf_planner.integrate(
                        color_im=rgb,
                        depth_im=depth,
                        cam_intr=cam_intr,
                        cam_pose=pose_habitat_to_tsdf(cam_pose),
                        obs_weight=1.0,
                        margin_h=int(cfg.margin_h_ratio * img_height),
                        margin_w=int(cfg.margin_w_ratio * img_width),
                        explored_depth=cfg.explored_depth,
                    )

                # (2) Update Memory Snapshots - add new frames to snapshots for VLM evaluation
                scene.update_snapshots()
                logging.info(
                    f"Step {cnt_step}, current snapshots count: {len(scene.snapshots)} snapshots"
                )

                # (3) Update the Frontier Snapshots
                update_success = tsdf_planner.update_frontier_map(
                    pts=pts,
                    cfg=cfg.planner,
                    scene=scene,
                    cnt_step=cnt_step,
                    save_frontier_image=cfg.save_visualization,  # flag to save image
                    eps_frontier_dir=eps_frontier_dir,  # path to save image
                    prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                )


                if not update_success:
                    logging.info("Warning! Update frontier map failed!")
                    if cnt_step == 0 and len(scene.snapshots) == 0:  # if the first step fails and no snapshots exist, we might need to stop
                        logging.info(
                            f"Question id {question_id} has no snapshots and frontier update failed!"
                        )
                    else:
                        logging.info("Continuing exploration despite frontier map update failure")

                # (4) Choose the next navigation point by querying the VLM
                if cfg.choose_every_step:
                    if (
                        tsdf_planner.max_point is not None  
                        and type(tsdf_planner.max_point) == Frontier
                    ):
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None

                if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                    agent_position_history.append(tsdf_planner.normal2voxel(pos_habitat_to_normal(pts)))

                    vlm_response = query_vlm_for_response(
                        question=question,
                        scene=scene,
                        tsdf_planner=tsdf_planner,
                        agent_position_history=agent_position_history,
                        rgb_egocentric_views=rgb_egocentric_views,
                        cfg=cfg,
                        verbose=True,
                    )
                    if vlm_response is None:
                        logging.info(
                            f"Question id {question_id} invalid: query_vlm_for_response failed and no snapshots available!"
                        )
                        break

                    max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response

                    # Check if the VLM chose a snapshot as the answer - if so, we should stop exploration
                    if type(max_point_choice) == SnapShot:
                        # when the VLM selects a snapshot as the answer, we consider the question is finished
                        # and save the chosen target snapshot without navigating to its location
                        if max_point_choice.image is not None:
                            snapshot_filename = max_point_choice.image.split(".")[0]
                            os.system(
                                f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                            )

                        task_success = True
                        logging.info(
                            f"Question id {question_id} finished with snapshot selection!"
                        )
                        break

                    # set the vlm choice as the navigation target (only for frontiers)
                    update_success = tsdf_planner.set_next_navigation_point(
                        choice=max_point_choice,
                        pts=pts,
                        objects=scene.objects,  # This might be minimal in pure VLM approach
                        cfg=cfg.planner,
                        pathfinder=scene.pathfinder,
                        random_position=False,
                    )

                    if type(max_point_choice) == Frontier:
                        frontier_id = max_point_choice.image if hasattr(max_point_choice, 'image') else f"frontier_{cnt_step}"
                        frontier_pos = max_point_choice.position if hasattr(max_point_choice, 'position') else None
                        logger.log_frontier_choice(frontier_id, frontier_pos)
                    
                    if not update_success:
                        logging.info(
                            f"Question id {question_id} invalid: set_next_navigation_point failed!"
                        )
                        break

                # (5) Agent navigate to the target point for one step
                return_values = tsdf_planner.agent_step(
                    pts=pts,
                    angle=angle,
                    objects=scene.objects,  # This might be empty in pure VLM approach
                    snapshots=scene.snapshots,
                    pathfinder=scene.pathfinder,
                    cfg=cfg.planner,
                    path_points=None,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_id} invalid: agent_step failed!")
                    break


                # update agent's position and rotation
                pts, angle, pts_voxel, fig, _, target_arrived = return_values
                logger.log_step(pts_voxel=pts_voxel)
                logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

                # sanity check about objects, scene graph, snapshots, ...
                scene.sanity_check(cfg=cfg)

                if cfg.save_visualization:
                    # save the top-down visualization
                    logger.save_topdown_visualization(
                        cnt_step=cnt_step,
                        fig=fig,
                        tsdf_planner=tsdf_planner,
                    )
                    # save the visualization of vlm's choice at each step
                    logger.save_frontier_visualization(
                        cnt_step=cnt_step,
                        tsdf_planner=tsdf_planner,
                        max_point_choice=max_point_choice,
                        global_caption=f"{question}\n{answer}",
                    )

                # (6) Check if the agent has arrived at the target to finish the question
                if type(max_point_choice) == SnapShot:
                    if max_point_choice.image is not None:
                        snapshot_filename = max_point_choice.image.split(".")[0]
                        os.system(
                            f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                        )

                    task_success = True
                    logging.info(
                        f"Question id {question_id} finished with snapshot selection!"
                    )
                    break
            except Exception as e:
                logging.error(f"Error in step {cnt_step}: {e}")
                if "GL::Context::current(): no current context" in str(e) or "core dumped" in str(e).lower():
                    logging.error("OpenGL context error detected, attempting to continue to next episode...")
                    break
                else:
                    raise  

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=logger.explore_dist,
            gpt_answer=gpt_answer,
            n_filtered_snapshots=n_filtered_snapshots,
            n_total_snapshots=len(scene.snapshots),  # Update this to reflect pure VLM
            n_total_frames=len(scene.frames),
        )

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        scene.print_scene_graph()

        # update the saved results after each episode
        logger.save_results()

        if not cfg.save_visualization:
            os.system(f"rm -r {episode_dir}") 
        try:
            scene.close()
        except:
            pass

    logger.save_results()
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="/.../Pred-EQA/cfg/eval_pred_eqa.yaml", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--qwen", help="qwen version", default="Qwen2.5-VL-3B-Instruct", type=str)
    args = parser.parse_args()
    # resolve placeholder "/.../" in cfg path to project root (script dir's parent)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.cfg_file.startswith("/.../"):
        args.cfg_file = args.cfg_file.replace("/.../", _project_root + "/", 1)
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    # resolve any remaining "/.../" placeholders in cfg string values
    _placeholder = "/.../"
    _replacement = _project_root + "/"
    for _k in cfg:
        if isinstance(cfg[_k], str) and _placeholder in cfg[_k]:
            cfg[_k] = cfg[_k].replace(_placeholder, _replacement)
    # resolve hm3d dataset path placeholder
    _hm3d_root = os.environ.get("HM3D_DATA_PATH", "/root")
    for _k in cfg:
        if isinstance(cfg[_k], str) and "/path-to-your-hm3d-dataset/" in cfg[_k]:
            cfg[_k] = cfg[_k].replace("/path-to-your-hm3d-dataset/", _hm3d_root + "/")

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    cfg.qwen = args.qwen 

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio)