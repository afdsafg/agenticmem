import os
import json
import pickle
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union

from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot


class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        n_total_questions,
        voxel_size,  # used for calculating the moving distance
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size

        if os.path.exists(
            os.path.join(output_dir, f"success_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(output_dir, f"success_list_{start_ratio}_{end_ratio}.pkl"),
                "rb",
            ) as f:
                self.success_list = pickle.load(f)
        else:
            self.success_list = []

        if os.path.exists(
            os.path.join(output_dir, f"path_length_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(
                    output_dir, f"path_length_list_{start_ratio}_{end_ratio}.pkl"
                ),
                "rb",
            ) as f:
                self.path_length_list = pickle.load(f)
        else:
            self.path_length_list = {}

        if os.path.exists(
            os.path.join(output_dir, f"fail_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(output_dir, f"fail_list_{start_ratio}_{end_ratio}.pkl"),
                "rb",
            ) as f:
                self.fail_list = pickle.load(f)
        else:
            self.fail_list = []

        if os.path.exists(
            os.path.join(output_dir, f"gpt_answer_{start_ratio}_{end_ratio}.json")
        ):
            with open(
                os.path.join(output_dir, f"gpt_answer_{start_ratio}_{end_ratio}.json"),
                "r",
            ) as f:
                self.gpt_answer_list = json.load(f)
        else:
            self.gpt_answer_list = []

        if os.path.exists(
            os.path.join(
                output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_filtered_snapshots_list = json.load(f)
        else:
            self.n_filtered_snapshots_list = {}

        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_total_snapshots_list = json.load(f)
        else:
            self.n_total_snapshots_list = {}

        if os.path.exists(
            os.path.join(output_dir, f"n_total_frames_{start_ratio}_{end_ratio}.json")
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_frames_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        self.n_total_questions = n_total_questions
        n_success = len(self.success_list)
        n_fail = len(self.fail_list)
        self.n_total = n_success + n_fail
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        # some sanity check
        assert n_success == len(
            self.path_length_list
        ), f"{n_success} != {len(self.path_length_list)}"
        # assert n_success == len(
        #     self.gpt_answer_list
        # ), f"{n_success} != {len(self.gpt_answer_list)}"
        assert self.n_total == len(
            self.n_filtered_snapshots_list
        ), f"{self.n_total} != {len(self.n_filtered_snapshots_list)}"
        assert self.n_total == len(
            self.n_total_snapshots_list
        ), f"{self.n_total} != {len(self.n_total_snapshots_list)}"
        assert self.n_total == len(
            self.n_total_frames_list
        ), f"{self.n_total} != {len(self.n_total_frames_list)}"

        # logging for episode
        self.episode_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.explore_dist = 0
        
        # 添加：用于记录轨迹和frontier选择，以计算动作一致性指标
        self.trajectory_positions = []
        self.frontier_choices = []
        self.frontier_positions = []  # 记录每步选择的frontier的位置

    def save_results(self):
        # sanity check
        assert len(self.success_list) == len(self.path_length_list)
        # assert len(self.success_list) == len(self.gpt_answer_list)
        assert self.n_total == len(self.n_filtered_snapshots_list)
        assert self.n_total == len(self.n_total_snapshots_list)
        assert self.n_total == len(self.n_total_frames_list)

        with open(
            os.path.join(
                self.output_dir, f"success_list_{self.start_ratio}_{self.end_ratio}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_list, f)
        with open(
            os.path.join(
                self.output_dir,
                f"path_length_list_{self.start_ratio}_{self.end_ratio}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.path_length_list, f)
        with open(
            os.path.join(
                self.output_dir, f"fail_list_{self.start_ratio}_{self.end_ratio}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(self.fail_list, f)
        with open(
            os.path.join(
                self.output_dir, f"gpt_answer_{self.start_ratio}_{self.end_ratio}.json"
            ),
            "w",
        ) as f:
            json.dump(self.gpt_answer_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_filtered_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_frames_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)

    def aggregate_results(self):
        # aggregate the results from different splits into a single file
        success_list = []
        path_length_list = {}
        all_success_list_paths = glob.glob(
            os.path.join(self.output_dir, "success_list_*.pkl")
        )
        all_path_length_list_paths = glob.glob(
            os.path.join(self.output_dir, "path_length_list_*.pkl")
        )
        for success_list_path in all_success_list_paths:
            with open(success_list_path, "rb") as f:
                success_list += pickle.load(f)
        for path_length_list_path in all_path_length_list_paths:
            with open(path_length_list_path, "rb") as f:
                path_length_list.update(pickle.load(f))

        with open(os.path.join(self.output_dir, "success_list.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(self.output_dir, "path_length_list.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)

        gpt_answer_list = []
        all_gpt_answer_list_paths = glob.glob(
            os.path.join(self.output_dir, "gpt_answer_*.json")
        )
        for gpt_answer_list_path in all_gpt_answer_list_paths:
            with open(gpt_answer_list_path, "r") as f:
                gpt_answer_list += json.load(f)

        with open(os.path.join(self.output_dir, "gpt_answer.json"), "w") as f:
            json.dump(gpt_answer_list, f, indent=4)

        n_filtered_snapshots_list = {}
        all_n_filtered_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_snapshots_*.json")
        )
        for n_filtered_snapshots_list_path in all_n_filtered_snapshots_list_paths:
            with open(n_filtered_snapshots_list_path, "r") as f:
                n_filtered_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_snapshots.json"), "w") as f:
            json.dump(n_filtered_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of filtered snapshots: {np.mean(list(n_filtered_snapshots_list.values()))}"
        )

        n_total_snapshots_list = {}
        all_n_total_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_snapshots_*.json")
        )
        for n_total_snapshots_list_path in all_n_total_snapshots_list_paths:
            with open(n_total_snapshots_list_path, "r") as f:
                n_total_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_snapshots.json"), "w") as f:
            json.dump(n_total_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of total snapshots: {np.mean(list(n_total_snapshots_list.values()))}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}"
        )

    def log_episode_result(
        self,
        success: bool,
        question_id,
        explore_dist,
        gpt_answer,
        n_filtered_snapshots,
        n_total_snapshots,
        n_total_frames,
    ):
        if success:
            if question_id not in self.success_list:
                self.success_list.append(question_id)
            self.path_length_list[question_id] = explore_dist
            logging.info(
                f"Question id {question_id} finish successfully, {explore_dist} length"
            )
        else:
            if question_id not in self.fail_list:
                self.fail_list.append(question_id)
            logging.info(f"Question id {question_id} failed, {explore_dist} length")

        logging.info(
            f"{self.n_total + 1}/{self.n_total_questions}: Success rate: {len(self.success_list)}/{self.n_total + 1}"
        )
        logging.info(
            f"Mean path length for success exploration: {np.mean(list(self.path_length_list.values()))}"
        )
        logging.info(
            f"Filtered snapshots/Total snapshots/Total frames: {n_filtered_snapshots}/{n_total_snapshots}/{n_total_frames}"
        )

        self.gpt_answer_list.append({"question_id": question_id, "answer": gpt_answer})

        self.n_filtered_snapshots_list[question_id] = n_filtered_snapshots
        self.n_total_snapshots_list[question_id] = n_total_snapshots
        self.n_total_frames_list[question_id] = n_total_frames

        self.n_total += 1
        
        # 添加：保存轨迹和frontier选择信息到文件（用于计算动作一致性指标）
        if self.episode_dir is not None:
            # 保存轨迹数据
            if hasattr(self, 'trajectory_positions') and len(self.trajectory_positions) > 0:
                trajectory_data = {
                    "positions": self.trajectory_positions,
                    "question_id": question_id,
                    "success": success,
                    "path_length": explore_dist
                }
                trajectory_path = os.path.join(self.episode_dir, "trajectory.json")
                with open(trajectory_path, 'w') as f:
                    json.dump(trajectory_data, f, indent=4)
            
            # 保存frontier选择数据
            if hasattr(self, 'frontier_choices') and len(self.frontier_choices) > 0:
                frontier_data = {
                    "choices": self.frontier_choices,
                    "positions": self.frontier_positions if hasattr(self, 'frontier_positions') else [],
                    "question_id": question_id,
                    "n_choices": len(self.frontier_choices)
                }
                frontier_path = os.path.join(self.episode_dir, "frontier_choices.json")
                with open(frontier_path, 'w') as f:
                    json.dump(frontier_data, f, indent=4)

        # clear up the episode log
        self.episode_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.explore_dist = 0
        self.trajectory_positions = []
        self.frontier_choices = []
        self.frontier_positions = []

    def init_episode(
        self,
        question_id,
        init_pts_voxel,
    ):
        self.episode_dir = os.path.join(self.output_dir, question_id)
        eps_chosen_snapshot_dir = os.path.join(self.episode_dir, "chosen_snapshot")
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_chosen_snapshot_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack([self.pts_voxels, init_pts_voxel])

        self.explore_dist = 0
        
        # 添加：初始化轨迹记录
        self.trajectory_positions = [init_pts_voxel.tolist()]
        self.frontier_choices = []
        self.frontier_positions = []

        return (
            self.episode_dir,
            eps_chosen_snapshot_dir,
            eps_frontier_dir,
            eps_snapshot_dir,
        )

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )
        
        # 添加：记录当前位置到轨迹
        if hasattr(self, 'trajectory_positions'):
            self.trajectory_positions.append(pts_voxel.tolist())

    def log_frontier_choice(self, frontier_id, frontier_position=None):
        """
        记录选择的frontier
        
        Args:
            frontier_id: frontier的标识符（如图片名称）
            frontier_position: frontier的位置坐标（可选）
        """
        if not hasattr(self, 'frontier_choices'):
            self.frontier_choices = []
            self.frontier_positions = []
        
        self.frontier_choices.append(frontier_id)
        if frontier_position is not None:
            self.frontier_positions.append(frontier_position.tolist() if isinstance(frontier_position, np.ndarray) else frontier_position)
    
    def save_topdown_visualization(self, cnt_step, fig, tsdf_planner=None):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        # 检查fig是否为None，如果是则跳过可视化保存
        if fig is None:
            logging.warning(f"Visualization figure is None for step {cnt_step}, skipping save")
            return

        # 获取matplotlib图中的轴 - 保持与原代码一致的处理方式
        if not hasattr(fig, 'axes') or len(fig.axes) == 0:
            logging.warning(f"Figure has no axes for step {cnt_step}, skipping save")
            return

        ax1 = fig.axes[0]
        
        # 绘制探索路径 - 只画到当前位置，不画到下一步计划位置
        # pts_voxels[-1] 是下一步计划去的位置（还没到达）
        # pts_voxels[-2] 是当前实际位置（已到达）
        # 所以轨迹只画到倒数第二个点
        if len(self.pts_voxels) > 1:
            # 只绘制到当前位置的路径（不包含最后一个点，因为那是计划去的位置）
            current_pts = self.pts_voxels[:-1] if len(self.pts_voxels) > 1 else self.pts_voxels
            
            if len(current_pts) > 1:
                ax1.plot(
                    current_pts[:, 1],
                    current_pts[:, 0],
                    linewidth=2,
                    color="red",
                    alpha=0.7,
                    label="Exploration Path"
                )
            
            # 标记起始点
            ax1.scatter(
                self.pts_voxels[0, 1],
                self.pts_voxels[0, 0],
                color="green",
                s=100,
                marker='o',
                label="Start",
                zorder=5
            )
            
            # 注意：当前位置的标记已经在agent_step的fig中绘制了（青蓝色圆圈）
            # 这里不需要重复标记
            
            # 如果有多个路径点，标记一些中间路径点以显示探索进度
            if len(current_pts) > 2:
                # 每隔几个点标记一个，避免过于密集
                step_size = max(1, len(current_pts) // 10)  # 最多标记10个中间点
                for i in range(0, len(current_pts), step_size):
                    if i != 0 and i != len(current_pts) - 1:  # 跳过起始点和终点
                        ax1.scatter(
                            current_pts[i, 1],
                            current_pts[i, 0],
                            color="orange",
                            s=30,
                            marker='.',
                            alpha=0.6
                        )

        # 注意：frontier的紫色箭头已经在agent_step中的fig上绘制过了
        # 这里不需要重复绘制，避免箭头重叠
        # 只需要添加图例说明即可
        if tsdf_planner is not None:
            # 不再重复绘制frontiers和箭头，因为agent_step中已经绘制
            # 只在必要时添加特殊标记
            pass

        # 添加标题和图例
        ax1.set_title(f"Exploration Map - Step {cnt_step}")
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, "{}_map.png".format(cnt_step)), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_frontier_visualization(
        self,
        cnt_step,
        tsdf_planner: TSDFPlanner,
        max_point_choice: Union[SnapShot, Frontier],
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        episode_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        if type(max_point_choice) == SnapShot:
            num_images += 1
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
                elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                    img_path = os.path.join(
                        episode_snapshot_dir, max_point_choice.image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    axs[h_idx, w_idx].set_title("Snapshot Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(os.path.join(frontier_video_path, f"{cnt_step}.png"))
        plt.close()
