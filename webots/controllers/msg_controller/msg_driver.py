import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import messagebox
from typing import Optional, List, Dict, Any, Tuple
import torch
from controller import Supervisor
import random
import math
import copy
import time
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import deque
import heapq

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.multiscale_pcn import PlaceCellLayer
from core.layers.multiscale_pcn_with_gcn_v13 import MultiscalePlaceCellWithGrid
from core.layers.grid_cell_layer_v13 import GridCellLayer
from core.layers.reward_cell_layer_v11 import RewardCellLayerTest, C_LAMBDA
from core.layers.unified_multiscale_pcn import UnifiedMultiScalePCN
from core.layers.unified_multiscale_rcn import UnifiedMultiScaleRCN
from core.robot.robot_mode import RobotMode
from core.robot.webots_worlds import get_world_obstacles, get_world_config
from visualizations.vis_utils import get_env_checkpoint_positions, get_env_goal_pos
from analysis.stats.stats_collector import stats_collector as StatsCollector

# Replay step budget is proportional to lambda_s to normalize spread per time constant
STEPS_PER_LAMBDA = 8  # Adjust to push farther (higher) or be more local (lower)
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
PICKLE_WRITE_BUFFER_SIZE = 8 * 1024 * 1024


def _dump_pickle_file(path: str, payload) -> None:
    """Write one pickle payload with faster protocol/buffering."""
    with open(path, "wb", buffering=PICKLE_WRITE_BUFFER_SIZE) as f:
        pickle.dump(payload, f, protocol=PICKLE_PROTOCOL)

def _steps_for_scale(scale_def: Dict[str, Any]) -> int:
    """Derive custom replay timesteps proportional to lambda_s for this scale."""
    sigma_pc_s = scale_def.get("sigma_pc_s", scale_def.get("sigma_r", 1.0))
    return int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA))

# --- PyTorch seeds / random ---
# torch.manual_seed(5)
# np.random.seed(5)
# rng = default_rng(5)  # or keep it as is
np.set_printoptions(precision=2)


class Driver(Supervisor):
    """Controls robot navigation and learning using neural networks for place and reward cells.

    This class manages the robot's sensory inputs, motor outputs, and neural network layers
    to enable autonomous navigation and learning in an environment. It coordinates between
    place cells for spatial representation and reward cells for goal-directed behavior.

    Attributes:
        max_speed (float): Maximum wheel rotation speed in rad/s.
        left_speed (float): Current left wheel speed in rad/s.
        right_speed (float): Current right wheel speed in rad/s.
        timestep (int): Duration of each simulation step in ms.
        wheel_radius (float): Radius of robot wheels in meters.
        axle_length (float): Distance between wheels in meters.
        num_steps (int): Total number of simulation steps.
        hmap_loc (ndarray): History of xzy-coordinates.
        hmap_pcn (ndarray): History of place cell activations.
        hmap_hdn (ndarray): History of head direction activations.
        hmap_g (ndarray): History of goal estimates.
        robot (Robot): Main robot controller instance.
        keyboard (Keyboard): Keyboard input device.
        compass (Compass): Compass sensor device.
        range_finder (RangeFinder): LiDAR sensor device.
        left_bumper (TouchSensor): Left collision sensor.
        right_bumper (TouchSensor): Right collision sensor.
        rotation_field (Field): Robot rotation field.
        left_motor (Motor): Left wheel motor controller.
        right_motor (Motor): Right wheel motor controller.
        left_position_sensor (PositionSensor): Left wheel encoder.
        right_position_sensor (PositionSensor): Right wheel encoder.
        pcn (PlaceCellLayer): Place cell neural network.
        rcn (RewardCellLayer): Reward cell neural network.
        boundary_data (Tensor): Current LiDAR readings.
        goal_location (List[float]): Target [x,y] coordinates.
        expected_reward (float): Predicted reward at current state.
    """
    def initialization(
        self,
        mode: RobotMode = RobotMode.PLOTTING,
        run_time_hours: int = 2,
        randomize_start_loc: bool = True,
        start_loc: Optional[List[float]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        scales: Optional[List[Dict[str, Any]]] = None,
        rcn_learning_rates: Optional[List[float]] = None,
        stats_collector: Optional[StatsCollector] = None,
        trial_id: Optional[str] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        goal_config: Optional[Dict[str, Any]] = None,
        trial_config: Optional[Dict[str, Any]] = None,
        max_dist: Optional[float] = None,
        plot_bvc: Optional[bool] = False,
        td_learning: Optional[bool] = False,
        use_prox_mod: Optional[bool] = False,
        use_unified_multiscale: Optional[bool] = False,
        environment_size: Optional[List[float]] = None,
        grid_size: Optional[float] = None,
        coverage_percentage: Optional[float] = None,
        minimum_learning_duration_hours: float = 4.0,
        min_goal_visits: int = 3,
        min_checkpoint_visits: int = 3,
        checkpoint_visit_radius: float = 0.8,
        goal_assoc_require_all_scales: bool = False,
        defer_goal_association_until_training_end: bool = True,
        goal_visit_cooldown_seconds: float = 6.0,
        goal_exit_hysteresis: float = 0.1,
        proximity_mode: str = "min",
        proximity_trimmed_sigma: float = 2.5,
        proximity_pair_percentile: float = 25.0,
        optimal_path_distance: Optional[float] = None,
        path_failure_ratio: Optional[float] = None,
        paths_folder: Optional[str] = None,
        hmaps_folder: Optional[str] = None,
        auto_trial_name: Optional[str] = None,
        num_auto_trials: int = 5,
        current_auto_trial: int = 1,
        lightweight_hmaps: bool = False,
        hmap_sample_stride: int = 10,
        hmap_topk: int = 8,
        gcn_scale_invariant: bool = True,
        gcn_use_adaptive_minmax: bool = False,
        goal_assoc_unique_topk: int = 16,
        goal_assoc_max_activation_drop: float = 0.08,
        goal_assoc_locality_radius: float = 2.0,
        goal_assoc_locality_max_activation_drop: float = 0.65,
        goal_map_use_compact_pc_gate: bool = True,
        goal_map_pc_compactness_peak_fraction: float = 0.50,
        goal_map_pc_compactness_min_peak: float = 0.05,
        goal_map_pc_compactness_max_radius: float = 4.0,
        goal_map_fill_source_envelope_tau: float = 6.0,
        goal_map_fill_source_envelope_floor: float = 0.10,
        record_experience_transitions: bool = True,
        two_phase_learning: bool = False,
        phase1_min_steps: int = 2500,
        phase1_max_steps: int = 20000,
        phase1_bin_size: float = 0.5,
        phase1_min_revisit_bins: int = 15,
        phase1_revisit_cosine_threshold: float = 0.90,
        phase1_revisit_window: int = 200,
        defer_experience_build_until_phase2_end: bool = True,
        goal_map_replay_timesteps: int = 10,
        goal_map_paper_replay_tau: float = 4.0,
        goal_map_goal_replay_timesteps: Optional[int] = 8,
        goal_map_goal_replay_tau: Optional[float] = 4.0,
        goal_map_goal_replay_event_count: int = 4,
        goal_map_checkpoint_replay_event_count: int = 3,
        goal_map_replay_event_merge_gap_steps: int = 3,
        goal_map_checkpoint_room_support_min_fraction: float = 0.98,
        hybrid_path_replay_weight: float = 0.8,
        hybrid_diffusion_replay_weight: float = 0.2,
        goal_map_path_replay_weight: Optional[float] = None,
        goal_map_diffusion_replay_weight: Optional[float] = None,
        goal_map_experience_transition_topk: Optional[int] = None,
        unified_reward_normalization_mode: str = "input_l1",
        unified_goal_map_mode: str = "paper_room_local_replay",
        goal_map_room_normalization_mode: str = "per_room_peak",
        goal_map_path_topk: int = 16,
        goal_map_path_decay: float = 0.97,
        goal_map_path_frontier_only: bool = True,
        goal_map_neighbor_topk: int = 12,
        goal_map_neighbor_steps: int = 6,
        goal_map_neighbor_decay: float = 0.84,
        goal_map_neighbor_frontier_only: bool = True,
        goal_map_neighbor_seed_scale: float = 0.60,
        goal_map_goal_seed_mode: str = "contact_ensemble",
        goal_map_goal_replay_retention_mode: str = "baseline",
        goal_map_goal_seed_post_contact_steps: int = 2,
        goal_map_goal_seed_within_visit_aggregation: str = "max",
        goal_map_goal_seed_across_visit_aggregation: str = "mean",
        goal_map_goal_seed_state_normalization: str = "peak",
        goal_map_goal_seed_self_retention_alpha: float = 0.15,
        goal_map_path_constraint_sigma: float = 4.0,
        goal_map_path_constraint_topk: int = 128,
        goal_map_local_constraint_sigma: float = 2.6,
        goal_map_local_constraint_topk: int = 36,
        goal_map_goal_seed_center_radius_factor: float = 0.0,
        goal_map_goal_seed_center_min_radius: float = 0.0,
        goal_map_checkpoint_seed_center_radius_factor: float = 0.0,
        goal_map_checkpoint_seed_center_min_radius: float = 0.0,
        goal_map_spatial_obstacle_block: bool = True,
        goal_map_obstacle_margin: float = 0.05,
        goal_map_use_checkpoint_relays: bool = False,
        goal_map_checkpoint_mode: str = "directed_route_memory_tree",
        goal_map_checkpoint_scale: float = 0.25,
        goal_map_checkpoint_support_mode: str = "threshold_normalized",
        goal_map_checkpoint_stop_parent_replay: bool = True,
        checkpoint_boost_threshold: float = 0.01,
        checkpoint_target_arrival: float = 0.15,
        checkpoint_min_relay_weight: float = 1.0,
        goal_map_route_selection_support_fraction: float = 0.15,
        goal_map_route_selection_overlap_threshold: float = 0.08,
        goal_map_use_denoised_experience_graph: bool = True,
        goal_map_use_segmented_checkpoint_histories: bool = True,
        goal_map_smoothing_sigma: float = 1.0,
        goal_map_smoothing_topk: int = 24,
        goal_map_smoothing_passes: int = 0,
        goal_map_smoothing_mix: float = 0.45,
        prune_experience_loops: bool = True,
        loop_prune_min_top1: float = 0.08,
        loop_prune_min_top1_to_top2_ratio: float = 1.15,
        pcn_gate_mode: str = "normal",
        pcn_soft_scale_overlap: bool = True,
        pcn_soft_scale_overlap_in_learning: bool = True,
        pcn_learning_adaptation_mode: str = "gaussian_post_competition_expression",
        pcn_learning_stdp_start_steps: int = 8000,
        pcn_learning_stop_ojas_on_stabilization: bool = False,
        pcn_learning_cross_scale_coupling_start_steps: int = 8000,
        pcn_learning_cross_scale_coupling_ramp_steps: int = 12000,
        pcn_learning_cross_scale_coupling_min: float = 0.0,
        pcn_soft_scale_gate_floor: float = 0.20,
        pcn_soft_scale_gate_floor_in_learning: float = 0.12,
        pcn_soft_cross_inhibition_scale: float = 0.35,
        pcn_soft_cross_inhibition_scale_in_learning: float = 0.25,
        pcn_soft_cross_inhibition_cap: float = 0.75,
        pcn_cross_scale_inhibition_base_enabled: bool = True,
        unified_preplay_scale_arbitration: bool = True,
        unified_preplay_scale_prior_mix: float = 0.0,
        unified_preplay_mode: str = "sampling",
        unified_preplay_commit_threshold: float = 0.30,
        preplay_heading_selection_mode: str = "trajectory_vector",
        unified_exploit_action_mode: str = "preplay_primary",
        unified_transition_preplay_enabled: bool = False,
        unified_spatial_fallback_enabled: bool = True,
        unified_spatial_override_margin: float = 0.0,
        unified_spatial_lookahead_distance: float = 1.0,
        unified_spatial_lookahead_sigma: float = 0.9,
        unified_exploit_step_fraction: float = 0.25,
        unified_graph_preplay_cluster_count: int = 1024,
        unified_graph_preplay_projection_dim: int = 64,
        unified_graph_preplay_kmeans_iters: int = 6,
        unified_graph_preplay_edge_source: str = "hdn",
        unified_graph_preplay_direction_tolerance: int = 0,
        unified_graph_preplay_reward_weight: float = 0.5,
        unified_graph_preplay_edge_weight: float = 0.0,
        exploit_direction_clearance_percentile: float = 20.0,
        use_bvc_context_modulation: bool = True,
        bvc_context_gain_floor: float = 0.0,
        bvc_context_gain_strength: float = 1.0,
        ):
        """
        Initializes the Driver class, setting up the robot's sensors and neural networks.
        """
        # Supervisor basics
        self.robot = self.getFromDef("agent")
        self.robot_mode = mode
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"[DRIVER] Using device: {self.device}")
        self.dtype = torch.float32

        # Determine or set world name
        if world_name is None:
            world_path = self.getWorldPath()
            world_name = os.path.splitext(os.path.basename(world_path))[0]
        self.world_name = world_name

        # BVC Context Modulation Params
        self.use_bvc_context_modulation = bool(use_bvc_context_modulation)
        self.bvc_context_gain_floor = float(min(0.95, max(0.0, bvc_context_gain_floor)))
        self.bvc_context_gain_strength = float(max(0.0, bvc_context_gain_strength))

        # Store auto trial parameters
        self.auto_trial_name = auto_trial_name
        self.num_auto_trials = num_auto_trials
        self.current_auto_trial = current_auto_trial

        # Directories for saving/loading data
        if mode in {RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
                    RobotMode.PLOTTING_AUTO, RobotMode.PLOTTING_COVERAGE_AUTO} and auto_trial_name:
            # Auto trial mode: pkl_{trial_name}/{world_name}_{trial_num}/
            base_folder = f"pkl_{auto_trial_name}"
            trial_folder = f"{world_name}_{current_auto_trial}"
            if hmaps_folder:
                self.hmap_dir = hmaps_folder
            else:
                self.hmap_dir = os.path.join(base_folder, trial_folder, "hmaps")
            self.network_dir = os.path.join(base_folder, trial_folder, "networks")
            self.trial_base_dir = os.path.join(base_folder, trial_folder)
        else:
            # Standard mode: pkl/{world_name}/
            if hmaps_folder:
                self.hmap_dir = hmaps_folder
            else:
                self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
            self.network_dir = os.path.join("pkl", self.world_name, "networks")
            self.trial_base_dir = None

        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Stats / trial info
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        # Head direction layer size
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10
        # Webots worlds in this repo use two different ground-plane conventions:
        # - environment_1..6: planar=(x, y), height=z
        # - 20x20_*_multi_goal: planar=(x, z), height=y
        self.planar_axis_indices = (
            (0, 2)
            if self.world_name in {"20x20_multi_goal", "20x20_cross_multi_goal", "20x20_maze_multi_goal"}
            else (0, 1)
        )
        self.height_axis_idx = 1 if self.planar_axis_indices == (0, 2) else 2
        # Goal-visit debounce settings:
        # - Cooldown prevents rapid re-counting from jitter near goal boundary.
        # - Exit hysteresis requires moving farther out before "leaving" the goal.
        self.goal_visit_cooldown_seconds = max(0.0, float(goal_visit_cooldown_seconds))
        self.goal_visit_cooldown_steps = max(
            1, int(round(self.goal_visit_cooldown_seconds / (self.timestep / 1000.0)))
        )
        self.goal_exit_hysteresis = max(0.0, float(goal_exit_hysteresis))
        self.minimum_learning_duration_seconds = max(
            0.0, float(minimum_learning_duration_hours) * 3600.0
        )
        self.min_checkpoint_visits = int(max(0, min_checkpoint_visits))
        self.checkpoint_visit_radius = float(max(0.0, checkpoint_visit_radius))
        self.goal_assoc_require_all_scales = bool(goal_assoc_require_all_scales)
        self.pcn_gate_mode = str(pcn_gate_mode).strip().lower()
        if self.pcn_gate_mode not in {"normal", "no_gate_no_inhibition", "no_gate_with_inhibition"}:
            print(f"[DRIVER] Unknown pcn_gate_mode='{self.pcn_gate_mode}', falling back to 'normal'")
            self.pcn_gate_mode = "normal"
        self.pcn_soft_scale_overlap = bool(pcn_soft_scale_overlap)
        self.pcn_soft_scale_overlap_in_learning = bool(
            pcn_soft_scale_overlap_in_learning
        )
        self.pcn_learning_adaptation_mode = str(
            pcn_learning_adaptation_mode
        ).strip().lower()
        if self.pcn_learning_adaptation_mode != "gaussian_post_competition_expression":
            print(
                "[DRIVER] Unknown pcn_learning_adaptation_mode="
                f"'{self.pcn_learning_adaptation_mode}', falling back to "
                "'gaussian_post_competition_expression'"
            )
            self.pcn_learning_adaptation_mode = "gaussian_post_competition_expression"
        self.pcn_learning_stdp_start_steps = int(
            max(0, pcn_learning_stdp_start_steps)
        )
        self.pcn_learning_stop_ojas_on_stabilization = bool(
            pcn_learning_stop_ojas_on_stabilization
        )
        self.pcn_learning_cross_scale_coupling_start_steps = int(
            max(0, pcn_learning_cross_scale_coupling_start_steps)
        )
        self.pcn_learning_cross_scale_coupling_ramp_steps = int(
            max(0, pcn_learning_cross_scale_coupling_ramp_steps)
        )
        self.pcn_learning_cross_scale_coupling_min = float(
            min(1.0, max(0.0, pcn_learning_cross_scale_coupling_min))
        )
        self.pcn_soft_scale_gate_floor = float(min(0.95, max(0.0, pcn_soft_scale_gate_floor)))
        self.pcn_soft_scale_gate_floor_in_learning = float(
            min(0.95, max(0.0, pcn_soft_scale_gate_floor_in_learning))
        )
        self.pcn_soft_cross_inhibition_scale = float(max(0.0, pcn_soft_cross_inhibition_scale))
        self.pcn_soft_cross_inhibition_scale_in_learning = float(
            max(0.0, pcn_soft_cross_inhibition_scale_in_learning)
        )
        self.pcn_soft_cross_inhibition_cap = float(max(0.0, pcn_soft_cross_inhibition_cap))
        self.pcn_cross_scale_inhibition_base_enabled = bool(
            pcn_cross_scale_inhibition_base_enabled
        )
        self.proximity_mode = str(proximity_mode).strip().lower()
        if self.proximity_mode not in {"min", "trimmed_mean", "opposite_pair_percentile", "local_minima", "raw_local_minima"}:
            print(f"[DRIVER] Unknown proximity_mode='{self.proximity_mode}', falling back to 'min'")
            self.proximity_mode = "min"
        self.proximity_trimmed_sigma = max(0.1, float(proximity_trimmed_sigma))
        self.proximity_pair_percentile = float(
            min(100.0, max(0.0, proximity_pair_percentile))
        )
        self.gcn_scale_invariant = bool(gcn_scale_invariant)
        self.gcn_use_adaptive_minmax = bool(gcn_use_adaptive_minmax)
        self.goal_assoc_unique_topk = int(max(1, goal_assoc_unique_topk))
        self.goal_assoc_max_activation_drop = float(
            min(0.5, max(0.0, goal_assoc_max_activation_drop))
        )
        self.goal_assoc_locality_radius = float(max(0.0, goal_assoc_locality_radius))
        self.goal_assoc_locality_max_activation_drop = float(
            min(0.95, max(0.0, goal_assoc_locality_max_activation_drop))
        )
        self.defer_goal_association_until_training_end = bool(
            defer_goal_association_until_training_end
        )
        self.goal_map_use_compact_pc_gate = bool(goal_map_use_compact_pc_gate)
        self.goal_map_pc_compactness_peak_fraction = float(
            min(0.95, max(0.05, goal_map_pc_compactness_peak_fraction))
        )
        self.goal_map_pc_compactness_min_peak = float(
            max(1e-6, goal_map_pc_compactness_min_peak)
        )
        self.goal_map_pc_compactness_max_radius = float(
            max(0.0, goal_map_pc_compactness_max_radius)
        )
        self.goal_map_fill_source_envelope_tau = float(
            max(1e-6, goal_map_fill_source_envelope_tau)
        )
        self.goal_map_fill_source_envelope_floor = float(
            min(0.95, max(0.0, goal_map_fill_source_envelope_floor))
        )
        self.record_experience_transitions = bool(record_experience_transitions)
        self._prev_unified_pcn_activations = None
        self.two_phase_learning = bool(two_phase_learning)
        if mode == RobotMode.LEARN_LOCATIONS_TWO_PHASE and not self.two_phase_learning:
            print("[DRIVER] Forcing two_phase_learning=True for LEARN_LOCATIONS_TWO_PHASE mode.")
            self.two_phase_learning = True
        self.two_phase_split_learning = bool(
            self.two_phase_learning and mode == RobotMode.LEARN_LOCATIONS_TWO_PHASE
        )
        self.phase1_min_steps = int(max(100, phase1_min_steps))
        self.phase1_max_steps = int(max(self.phase1_min_steps + 100, phase1_max_steps))
        self.phase1_bin_size = float(max(0.1, phase1_bin_size))
        self.phase1_min_revisit_bins = int(max(1, phase1_min_revisit_bins))
        self.phase1_revisit_cosine_threshold = float(min(1.0, max(0.0, phase1_revisit_cosine_threshold)))
        self.phase1_revisit_window = int(max(10, phase1_revisit_window))
        # Two-phase switch should require low drift across a broad revisited slice
        # of the explored environment, not just a good global mean cosine.
        self.phase1_revisit_fraction_threshold = 0.40
        self.phase1_revisit_low_percentile = 10.0
        self.phase1_revisit_low_percentile_threshold = float(
            max(0.0, self.phase1_revisit_cosine_threshold - 0.03)
        )
        self.phase1_stability_required_streak = int(
            max(25, min(100, self.phase1_revisit_window // 8))
        )
        self.defer_experience_build_until_phase2_end = bool(
            defer_experience_build_until_phase2_end
        )
        self.goal_map_replay_timesteps = int(max(1, goal_map_replay_timesteps))
        self.goal_map_paper_replay_tau = float(max(1e-6, goal_map_paper_replay_tau))
        self.goal_map_goal_replay_timesteps = (
            None
            if goal_map_goal_replay_timesteps is None
            else int(max(1, goal_map_goal_replay_timesteps))
        )
        self.goal_map_goal_replay_tau = (
            None
            if goal_map_goal_replay_tau is None
            else float(max(1e-6, goal_map_goal_replay_tau))
        )
        self.goal_map_goal_replay_event_count = int(
            max(1, goal_map_goal_replay_event_count)
        )
        self.goal_map_checkpoint_replay_event_count = int(
            max(1, goal_map_checkpoint_replay_event_count)
        )
        self.goal_map_replay_event_merge_gap_steps = int(
            max(0, goal_map_replay_event_merge_gap_steps)
        )
        self.hybrid_path_replay_weight = float(max(0.0, hybrid_path_replay_weight))
        self.hybrid_diffusion_replay_weight = float(max(0.0, hybrid_diffusion_replay_weight))
        self.goal_map_path_replay_weight = (
            None if goal_map_path_replay_weight is None else float(max(0.0, goal_map_path_replay_weight))
        )
        self.goal_map_diffusion_replay_weight = (
            None if goal_map_diffusion_replay_weight is None else float(max(0.0, goal_map_diffusion_replay_weight))
        )
        self.goal_map_experience_transition_topk = (
            None
            if goal_map_experience_transition_topk is None
            else int(max(1, goal_map_experience_transition_topk))
        )
        goal_map_mode = str(unified_goal_map_mode).strip().lower()
        if goal_map_mode not in {
            "paper_room_local_replay",
            "competitive_path_neighbor",
            "additive_replay",
            "trajectory_state_backbone",
            "room_masked_experience_replay",
        }:
            print(
                f"[DRIVER] Unknown unified_goal_map_mode='{unified_goal_map_mode}', "
                "falling back to 'paper_room_local_replay'"
            )
            goal_map_mode = "paper_room_local_replay"
        self.unified_goal_map_mode = goal_map_mode
        reward_mode = str(unified_reward_normalization_mode).strip().lower()
        if reward_mode not in {"input_l1", "weight_mass"}:
            print(
                f"[DRIVER] Unknown unified_reward_normalization_mode='{unified_reward_normalization_mode}', "
                "falling back to 'input_l1'"
            )
            reward_mode = "input_l1"
        if goal_map_mode == "paper_room_local_replay" and reward_mode != "input_l1":
            print(
                "[DRIVER] paper_room_local_replay forces "
                "unified_reward_normalization_mode='input_l1'"
            )
            reward_mode = "input_l1"
        self.unified_reward_normalization_mode = reward_mode
        room_norm_mode = str(goal_map_room_normalization_mode).strip().lower()
        if room_norm_mode not in {"per_room_peak", "none"}:
            print(
                f"[DRIVER] Unknown goal_map_room_normalization_mode='{goal_map_room_normalization_mode}', "
                "falling back to 'per_room_peak'"
            )
            room_norm_mode = "per_room_peak"
        self.goal_map_room_normalization_mode = room_norm_mode
        self.goal_map_checkpoint_room_support_min_fraction = float(
            min(1.0, max(0.50, goal_map_checkpoint_room_support_min_fraction))
        )
        self.goal_map_path_topk = int(max(1, goal_map_path_topk))
        self.goal_map_path_decay = float(min(0.9999, max(0.0, goal_map_path_decay)))
        self.goal_map_path_frontier_only = bool(goal_map_path_frontier_only)
        self.goal_map_neighbor_topk = int(max(1, goal_map_neighbor_topk))
        self.goal_map_neighbor_steps = int(max(0, goal_map_neighbor_steps))
        self.goal_map_neighbor_decay = float(min(0.9999, max(0.0, goal_map_neighbor_decay)))
        self.goal_map_neighbor_frontier_only = bool(goal_map_neighbor_frontier_only)
        self.goal_map_neighbor_seed_scale = float(
            min(1.0, max(0.0, goal_map_neighbor_seed_scale))
        )
        goal_seed_mode = str(goal_map_goal_seed_mode).strip().lower()
        if goal_seed_mode not in {"exact_contact", "contact_ensemble"}:
            goal_seed_mode = "contact_ensemble"
        self.goal_map_goal_seed_mode = goal_seed_mode
        goal_retention_mode = str(goal_map_goal_replay_retention_mode).strip().lower()
        if goal_retention_mode not in {"baseline", "boosted"}:
            goal_retention_mode = "baseline"
        self.goal_map_goal_replay_retention_mode = goal_retention_mode
        self.goal_map_goal_seed_post_contact_steps = int(
            max(0, goal_map_goal_seed_post_contact_steps)
        )
        within_visit_aggregation = (
            str(goal_map_goal_seed_within_visit_aggregation).strip().lower()
        )
        if within_visit_aggregation not in {"max", "mean"}:
            within_visit_aggregation = "max"
        self.goal_map_goal_seed_within_visit_aggregation = within_visit_aggregation
        across_visit_aggregation = (
            str(goal_map_goal_seed_across_visit_aggregation).strip().lower()
        )
        if across_visit_aggregation not in {"max", "mean"}:
            across_visit_aggregation = "mean"
        self.goal_map_goal_seed_across_visit_aggregation = across_visit_aggregation
        state_normalization = str(goal_map_goal_seed_state_normalization).strip().lower()
        if state_normalization not in {"peak"}:
            state_normalization = "peak"
        self.goal_map_goal_seed_state_normalization = state_normalization
        self.goal_map_goal_seed_self_retention_alpha = float(
            min(1.0, max(0.0, goal_map_goal_seed_self_retention_alpha))
        )
        self.goal_map_path_constraint_sigma = float(max(0.1, goal_map_path_constraint_sigma))
        self.goal_map_path_constraint_topk = int(max(1, goal_map_path_constraint_topk))
        self.goal_map_local_constraint_sigma = float(max(0.1, goal_map_local_constraint_sigma))
        self.goal_map_local_constraint_topk = int(max(1, goal_map_local_constraint_topk))
        self.goal_map_spatial_obstacle_block = bool(goal_map_spatial_obstacle_block)
        self.goal_map_obstacle_margin = float(max(0.0, goal_map_obstacle_margin))
        self.goal_map_goal_seed_center_radius_factor = float(
            max(0.0, goal_map_goal_seed_center_radius_factor)
        )
        self.goal_map_goal_seed_center_min_radius = float(
            max(0.0, goal_map_goal_seed_center_min_radius)
        )
        self.goal_map_checkpoint_seed_center_radius_factor = float(
            max(0.0, goal_map_checkpoint_seed_center_radius_factor)
        )
        self.goal_map_checkpoint_seed_center_min_radius = float(
            max(0.0, goal_map_checkpoint_seed_center_min_radius)
        )
        self.goal_map_use_checkpoint_relays = bool(goal_map_use_checkpoint_relays)
        checkpoint_mode = str(goal_map_checkpoint_mode).strip().lower()
        if checkpoint_mode not in {
            "implicit_cascade",
            "topology_schedule",
            "route_selected_tree",
            "directed_route_memory_tree",
            "multi_route_memory_graph",
        }:
            checkpoint_mode = "implicit_cascade"
        self.goal_map_checkpoint_mode = checkpoint_mode
        self.goal_map_checkpoint_scale = float(max(0.0, goal_map_checkpoint_scale))
        checkpoint_support_mode = str(goal_map_checkpoint_support_mode).strip().lower()
        if checkpoint_support_mode not in {"threshold_normalized", "raw_support"}:
            checkpoint_support_mode = "threshold_normalized"
        self.goal_map_checkpoint_support_mode = checkpoint_support_mode
        self.goal_map_checkpoint_stop_parent_replay = bool(
            goal_map_checkpoint_stop_parent_replay
        )
        self.checkpoint_boost_threshold = float(max(0.0, checkpoint_boost_threshold))
        self.checkpoint_target_arrival = float(max(0.0, checkpoint_target_arrival))
        self.checkpoint_min_relay_weight = float(max(0.0, checkpoint_min_relay_weight))
        self.goal_map_route_selection_support_fraction = float(
            min(0.95, max(1e-4, goal_map_route_selection_support_fraction))
        )
        self.goal_map_route_selection_overlap_threshold = float(
            min(1.0, max(0.0, goal_map_route_selection_overlap_threshold))
        )
        self.goal_map_use_denoised_experience_graph = bool(
            goal_map_use_denoised_experience_graph
        )
        self.goal_map_use_segmented_checkpoint_histories = bool(
            goal_map_use_segmented_checkpoint_histories
        )
        self.goal_map_smoothing_sigma = float(max(0.1, goal_map_smoothing_sigma))
        self.goal_map_smoothing_topk = int(max(1, goal_map_smoothing_topk))
        self.goal_map_smoothing_passes = int(max(0, goal_map_smoothing_passes))
        self.goal_map_smoothing_mix = float(min(1.0, max(0.0, goal_map_smoothing_mix)))
        self._unified_reward_spatial_cache = None
        self._unified_pc_spatial_cache = None
        self._unified_visibility_mask_cache = None
        if not hasattr(self, "_frozen_unified_dense_hmaps_cache"):
            self._frozen_unified_dense_hmaps_cache = None
        if not hasattr(self, "_frozen_unified_dense_hdn_cache"):
            self._frozen_unified_dense_hdn_cache = None
        if not hasattr(self, "_goal_specific_unified_rcn_cache"):
            self._goal_specific_unified_rcn_cache = {}
        self.unified_preplay_scale_arbitration = bool(unified_preplay_scale_arbitration)
        self.unified_preplay_scale_prior_mix = float(
            min(1.0, max(0.0, unified_preplay_scale_prior_mix))
        )
        requested_unified_preplay_mode = str(unified_preplay_mode).strip().lower()
        if requested_unified_preplay_mode != "sampling":
            print(
                "[DRIVER] Ignoring deprecated unified_preplay_mode="
                f"'{requested_unified_preplay_mode}' and using 'sampling'"
            )
        self.unified_preplay_mode = "sampling"
        self.unified_preplay_commit_threshold = float(
            min(1.0, max(0.0, unified_preplay_commit_threshold))
        )
        requested_heading_selection_mode = str(
            preplay_heading_selection_mode
        ).strip().lower()
        if requested_heading_selection_mode != "trajectory_vector":
            print(
                "[DRIVER] Ignoring deprecated preplay_heading_selection_mode="
                f"'{requested_heading_selection_mode}' and using 'trajectory_vector'"
            )
        self.preplay_heading_selection_mode = "trajectory_vector"
        requested_unified_action_mode = str(
            unified_exploit_action_mode
        ).strip().lower()
        if requested_unified_action_mode != "preplay_primary":
            print(
                "[DRIVER] Ignoring deprecated unified_exploit_action_mode="
                f"'{requested_unified_action_mode}' and using 'preplay_primary'"
            )
        self.unified_exploit_action_mode = "preplay_primary"
        self.unified_transition_preplay_enabled = False
        self.unified_spatial_fallback_enabled = False
        self.unified_spatial_override_margin = float(
            max(0.0, unified_spatial_override_margin)
        )
        self.unified_spatial_lookahead_distance = float(
            max(0.05, unified_spatial_lookahead_distance)
        )
        self.unified_spatial_lookahead_sigma = float(
            max(1e-3, unified_spatial_lookahead_sigma)
        )
        self.unified_exploit_step_fraction = float(
            min(1.0, max(0.05, unified_exploit_step_fraction))
        )
        self.unified_graph_preplay_cluster_count = int(
            max(64, unified_graph_preplay_cluster_count)
        )
        self.unified_graph_preplay_projection_dim = int(
            max(8, unified_graph_preplay_projection_dim)
        )
        self.unified_graph_preplay_kmeans_iters = int(
            max(1, unified_graph_preplay_kmeans_iters)
        )
        self.unified_graph_preplay_edge_source = str(
            unified_graph_preplay_edge_source
        ).strip().lower()
        self.unified_graph_preplay_direction_tolerance = int(
            max(0, min(int(self.n_hd // 2), unified_graph_preplay_direction_tolerance))
        )
        self.unified_graph_preplay_reward_weight = float(
            max(0.0, unified_graph_preplay_reward_weight)
        )
        self.unified_graph_preplay_edge_weight = float(
            max(0.0, unified_graph_preplay_edge_weight)
        )
        self.exploit_direction_clearance_percentile = float(
            min(100.0, max(0.0, exploit_direction_clearance_percentile))
        )
        self.prune_experience_loops = bool(prune_experience_loops)
        self.loop_prune_min_top1 = float(max(0.0, loop_prune_min_top1))
        self.loop_prune_min_top1_to_top2_ratio = float(
            max(1.0, loop_prune_min_top1_to_top2_ratio)
        )

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 8
        self.max_dist = max_dist
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756

        # Simulation run time
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000)) + 1

        # Exploration/exploitation radius
        self.goal_r = {"explore": 0.3, "exploit": 0.5}
        self.goal_location = goal_location if goal_location else [-3, 3]
        self.start_loc = start_loc
        # Height used when explicitly resetting the robot pose.
        initial_translation = self.robot.getField("translation").getSFVec3f()
        initial_rotation = self.robot.getField("rotation").getSFRotation()
        self.spawn_ground_y = float(initial_translation[self.height_axis_idx])
        # Use the world-authored stable pose as the recovery target. Cross-worlds are
        # not authored with an identity rotation, so forcing [0, 0, 1, 0] makes the
        # robot look "tipped" forever even when the spawn position is correct.
        self.upright_rotation = [float(v) for v in initial_rotation]

        # Default single scale if none provided
        if not scales:
            print("[DRIVER] No scales provided, using default scale")
            scales = [{
                "name": "default_scale",
                "num_pc": 500,
                "sigma_r": 0.5,
                "sigma_theta": 1.0
            }]
        self.scales = scales
        
        # Store learning rates for later use in RCN initialization
        self.rcn_learning_rates = rcn_learning_rates if rcn_learning_rates is not None else [0.1] * len(scales)
        self.td_learning = td_learning
        self.use_prox_mod = use_prox_mod
        self.use_unified_multiscale = use_unified_multiscale

        # Store coverage parameters
        self.environment_size = environment_size
        self.grid_size = grid_size
        self.coverage_percentage = coverage_percentage
        self.min_goal_visits = min_goal_visits

        # Store random spawn parameters
        self.optimal_path_distance = optimal_path_distance
        self.path_failure_ratio = path_failure_ratio
        self.paths_folder = paths_folder
        self.lightweight_hmaps = bool(lightweight_hmaps)
        self.hmap_sample_stride = max(1, int(hmap_sample_stride))
        self.hmap_topk = max(1, int(hmap_topk))

        # Initialize distance tracking for random spawn mode
        self.total_distance_traveled = 0.0
        self.last_position = None

        # Setup unified goal system
        self._setup_goals(goal_config, goal_location)

        # Setup trial configuration
        self._setup_trials(trial_config)

        self.start_loc = start_loc

        # Random or fixed start
        if randomize_start_loc:
            while True:
                candidate = [
                    random.uniform(-2.3, 2.3),
                    random.uniform(-2.3, 2.3),
                ]
                # Check against all goals in unified system
                min_dist = float('inf')
                for goal in self.goals:
                    dist = np.sqrt(
                        (candidate[0] - goal["location"][0]) ** 2 +
                        (candidate[1] - goal["location"][1]) ** 2
                    )
                    min_dist = min(min_dist, dist)
                if min_dist >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(
                self._compose_translation(candidate[0], candidate[1])
            )
            self.robot.getField("rotation").setSFRotation(self.upright_rotation)
            self.robot.resetPhysics()
        else:
            if self.start_loc is not None:
                self.robot.getField("translation").setSFVec3f(
                    self._compose_translation(self.start_loc[0], self.start_loc[1])
                )
                self.robot.getField("rotation").setSFRotation(self.upright_rotation)
                self.robot.resetPhysics()

        # Initialize sensors
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)
        self.range_finder = self.getDevice("range-finder")
        self.range_finder.enable(self.timestep)
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)

        # Bumpers
        self.collided = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.left_bumper = self.getDevice("bumper_left")
        self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice("bumper_right")
        self.right_bumper.enable(self.timestep)

        # Motors / position sensors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        self.step_count = 0

        # Clear if in Oja's mode
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Load or init Grid Cell Networks / PCNs / RCNs.
        # Random exploit trials run many respawns back-to-back on the same Driver
        # instance. Re-reading the same networks from disk each trial causes visible
        # sim-speed pulsing after the first successful trial, so reuse the already
        # loaded models when possible and only reset dynamic activations/traces.
        reuse_loaded_networks = bool(
            mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}
            and (
                (
                    bool(use_unified_multiscale)
                    and getattr(self, "unified_pcn", None) is not None
                    and getattr(self, "unified_rcn", None) is not None
                )
                or (
                    not bool(use_unified_multiscale)
                    and bool(getattr(self, "pcns", None))
                    and bool(getattr(self, "rcns", None))
                )
            )
        )
        if reuse_loaded_networks:
            print("[DRIVER] Reusing loaded networks for random exploit trial reset")
            if self.use_unified_multiscale:
                self.unified_pcn.reset_activations()
                self.pcns = [self.unified_pcn]
                if hasattr(self.unified_rcn, "reward_cell_activations"):
                    self.unified_rcn.reward_cell_activations = torch.zeros_like(
                        self.unified_rcn.reward_cell_activations
                    )
                self.rcns = [self.unified_rcn]
            else:
                for pcn in self.pcns:
                    if hasattr(pcn, "reset_activations"):
                        pcn.reset_activations()
                for rcn in self.rcns:
                    if hasattr(rcn, "reward_cell_activations"):
                        rcn.reward_cell_activations = torch.zeros_like(
                            rcn.reward_cell_activations
                        )
            if not hasattr(self, "gcns") or self.gcns is None:
                self.gcns = []
        else:
            self.gcns = []
            self.init_grid_cell_networks()
            self.pcns = []
            self.rcns = []
            self.load_pcns(enable_ojas, enable_stdp)
            self.load_rcns()
        if self.use_unified_multiscale:
            print("[DRIVER] Using PCNs: ['unified_pcn.pkl']")
            print("[DRIVER] Using RCNs: ['unified_rcn.pkl']")
        else:
            pcn_files = [f"pcn_scale_{scale_def['scale_index']}.pkl" for scale_def in self.scales]
            rcn_files = [f"rcn_scale_{scale_def['scale_index']}.pkl" for scale_def in self.scales]
            print(f"[DRIVER] Using PCNs: {pcn_files}")
            print(f"[DRIVER] Using RCNs: {rcn_files}")

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Initialize alpha as a tensor of zeros with the same length as the number of scales
        self.alpha = torch.zeros(len(self.scales), dtype=self.dtype, device=self.device)

        self.hmap_scale_priority = torch.zeros(self.num_steps, device=self.device, dtype=torch.float32)

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        self.hmap_prox = torch.zeros((self.num_steps,), device=self.device, dtype=torch.float32)
        self.prox = 0.0
        self.hmap_scale_priority = torch.zeros(
            (self.num_steps, len(self.scales)),  # row per step, col per scale
            device=self.device,
            dtype=torch.float32
        )

        # For multi-scale place/grid logs
        self.hmap_pcn_activities = []
        self.hmap_gcn_activities = []
        self.hmap_compact_stats = {"pcn": {}, "gcn": {}, "sample_stride": self.hmap_sample_stride}
        for scale_def in self.scales:
            scale_idx = scale_def["scale_index"]
            self.hmap_compact_stats["pcn"][scale_idx] = []
            self.hmap_compact_stats["gcn"][scale_idx] = []
        self.scale_history_index_by_scale_idx = {
            int(scale_def["scale_index"]): i for i, scale_def in enumerate(self.scales)
        }
        # Diagnostics for post-run analysis of scale gating and recruitment behavior.
        self.diag_scale_indices = [int(s["scale_index"]) for s in self.scales]
        self.diag_sample_steps = []
        self.diag_prox_values = []
        self.diag_scale_pref_values = []
        self.diag_active_counts = []
        self.diag_active_fractions = []
        self.diag_activation_sums = []
        self.diag_activation_means = []
        self.diag_activation_peaks = []
        self.diag_cross_scale_inhibition_means = []
        self.diag_cross_scale_inhibition_peaks = []
        self.diag_other_scale_activity_sums = []
        self.diag_effective_cross_scale_factors = []
        self.afferent_diag_attr_by_metric = {
            "raw_bvc_abs_mean": "last_raw_bvc_abs_mean_per_scale",
            "raw_grid_abs_mean": "last_raw_grid_abs_mean_per_scale",
            "raw_grid_share": "last_raw_grid_share_per_scale",
            "balanced_bvc_abs_mean": "last_balanced_bvc_abs_mean_per_scale",
            "balanced_grid_abs_mean": "last_balanced_grid_abs_mean_per_scale",
            "balanced_grid_share": "last_balanced_grid_share_per_scale",
            "mixed_bvc_abs_mean": "last_mixed_bvc_abs_mean_per_scale",
            "mixed_grid_abs_mean": "last_mixed_grid_abs_mean_per_scale",
            "mixed_grid_share": "last_mixed_grid_share_per_scale",
            "bvc_gain": "last_bvc_gain_value_per_scale",
            "grid_gain": "last_grid_gain_value_per_scale",
            "effective_grid_influence": "last_effective_grid_influence_mean_per_scale",
            "bvc_context_gain": "last_bvc_context_gain_per_scale",
            "bvc_afferent_source_sum": "last_bvc_afferent_source_sum_per_scale",
            "grid_afferent_source_sum": "last_grid_afferent_source_sum_per_scale",
            "bvc_afferent_inhibition_mean": "last_bvc_afferent_inhibition_mean_per_scale",
            "grid_afferent_inhibition_mean": "last_grid_afferent_inhibition_mean_per_scale",
            "afferent_inhibition_mean": "last_afferent_inhibition_mean_per_scale",
        }
        self.diag_afferent_metric_rows = {
            metric_name: [] for metric_name in self.afferent_diag_attr_by_metric
        }
        # Sample diagnostics on the same cadence as compact hmaps to avoid per-step
        # host transfers during long training runs.
        self.scale_diag_sample_stride = max(1, self.hmap_sample_stride)

        if not self.lightweight_hmaps:
            for scale_def in self.scales:
                n_pc = scale_def["num_pc"]
                self.hmap_pcn_activities.append(
                    torch.zeros((self.num_steps, n_pc), device=self.device, dtype=torch.float32)
                )

            for scale_def in self.scales:
                n_gc = scale_def.get("num_grid_cells", 0)
                self.hmap_gcn_activities.append(
                    torch.zeros((self.num_steps, n_gc), device=self.device, dtype=torch.float32)
                )

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device) 

        # Rotation tracking for excessive loop detection
        self.rotation_accumulator = 0.0
        self.rotation_loop_count = 0
        self.steps_since_last_loop = 0
        self.last_heading_deg = None
        self.done = False
        self._trial_terminal_handled = False
        # Clear stale room-mask caches so exploit recomputes from fresh spawn position.
        self._last_unified_exploit_room_reward_mask = None
        self._unified_reward_hmap_cache = None
        self.last_unified_committed_hd_bin = None
        self._last_unified_commit_target_signature = None
        self._last_unified_active_checkpoint_target = None
        self._last_unified_commit_reset_reason = None
        self._reset_unified_heading_commit_state(reason="trial_start")
        
        # Number of steps to force exploration (init to 0)
        self.force_explore_count = 0
        self.training_log_interval_steps = 100
        self.grid_diag_interval_steps = 100
        self.coverage_incomplete_log_interval_steps = 500
        self.last_coverage_incomplete_log_step = -10**9

        # Per-scale reliability tracking
        # Reset to None at start of each trial - will be re-initialized to 1.0 in exploit_v12
        self.scale_reliability = None  # Shape: [num_scales], values in [0, 1]
        self.last_scale_weights = None  # Store scale weights for credit assignment
        self.loop_scale_contributions = None  # Track scale contributions to rotation loops
        print(f"[DRIVER] Scale reliability reset for new trial (will initialize to 1.0 on first exploit_v12 call)")

        # Optionally keep a single-scale reference
        self.pcn = self.pcns[0] if self.pcns else None
        self.rcn = self.rcns[0] if self.rcns else None

        # Hierarchical sub-goal navigation via scene-placed checkpoints
        self.detected_doorways = []        # [(x, y)] read from checkpoint_* scene nodes
        self.checkpoint_beta = 0.03        # topological relay decay rate with graph distance to goal

        self.checkpoint_boost_gamma = 0.5        # base relay amplitude before adaptive gap compensation
        self.checkpoint_target_arrival = float(max(0.0, self.checkpoint_target_arrival))
        self.checkpoint_min_relay_weight = float(max(0.0, self.checkpoint_min_relay_weight))
        self.checkpoint_relay_margin = 1.15     # extra margin to reach the next upstream checkpoint
        self.checkpoint_max_relay_weight = 4.0  # cap for adaptive checkpoint relay weights
        self.goal_map_max_initial_backbone_gain = 2.0  # cap for goal seed amplification
        self._valid_checkpoints = None           # set after _compute_valid_checkpoints() — all checkpoints
        self._load_checkpoints_from_scene()
        self.checkpoint_visit_counts = {}
        self.checkpoint_crossing_counts = {}
        self.checkpoint_crossing_steps = {}
        self.checkpoint_currently_in = {}
        self.checkpoint_last_count_step = {}
        self.checkpoint_entry_side = {}
        self.checkpoint_last_clear_side = {}
        self.checkpoint_crossed_centerline = {}
        self.checkpoint_crossing_step_candidate = {}
        if self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_ADAPTIVE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
            RobotMode.LEARN_HEBB,
        } and self.detected_doorways:
            self.checkpoint_visit_counts = {
                idx: 0 for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_crossing_counts = {
                idx: {"neg_to_pos": 0, "pos_to_neg": 0}
                for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_crossing_steps = {
                idx: {"neg_to_pos": [], "pos_to_neg": []}
                for idx in range(len(self.detected_doorways))
            }
            # Multi-route memory graph: online edge potentiation between checkpoints/goals.
            # Key: (src_cp_idx, src_direction, dst_cp_idx, dst_direction), Value: count
            self.checkpoint_to_checkpoint_edges: Dict[Tuple[int, str, int, str], int] = {}
            # Key: (src_cp_idx, src_direction, goal_name), Value: count
            self.checkpoint_to_goal_edges: Dict[Tuple[int, str, str], int] = {}
            # Track most recently crossed checkpoint for successive-crossing potentiation.
            self.checkpoint_last_crossed: Optional[Tuple[int, str]] = None
            self.checkpoint_currently_in = {
                idx: False for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_last_count_step = {
                idx: -10**9 for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_entry_side = {
                idx: 0 for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_last_clear_side = {
                idx: 0 for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_crossed_centerline = {
                idx: False for idx in range(len(self.detected_doorways))
            }
            self.checkpoint_crossing_step_candidate = {
                idx: -1 for idx in range(len(self.detected_doorways))
            }
            if self.min_checkpoint_visits > 0:
                print(
                    f"[DRIVER] Checkpoint crossing gate enabled: "
                    f"{len(self.detected_doorways)} checkpoints, "
                    f"min_crossings_per_direction={self.min_checkpoint_visits}, "
                    f"radius={self.checkpoint_visit_radius:.2f}m"
                )

        # Enforce per-goal maps for unified exploit when multi-goal target is available.
        # `_ensure_goal_specific_rcn_loaded_for_exploit()` only reloads when the goal
        # actually changes, which avoids expensive redundant unpickles across respawns.
        self._ensure_goal_specific_rcn_loaded_for_exploit()

        self.plot_bvc = plot_bvc

        # Two-phase learning state (for coverage-learning modes).
        self.two_phase_phase = "single"
        self.phase2_start_step = None
        self.adaptive_stdp_start_step = None
        # Revisit-consistency stability tracking:
        # _phase1_bin_activations: bin_key -> last recorded activation vector
        # _phase1_revisit_cosines: deque of cosine similarities from revisited bins
        # _phase1_bins_with_revisits: set of bin keys visited more than once
        self._phase1_bin_activations = {}
        self._phase1_bin_last_cosines = {}
        self._phase1_revisit_cosines = deque(maxlen=self.phase1_revisit_window)
        self._phase1_bins_with_revisits = set()
        self._phase1_last_stability_eval_step = -1
        self._phase1_stability_streak = 0
        self._phase1_last_stability_status = None
        self.ojas_stopped_after_stabilization = False
        self.ojas_stop_step = None
        if self.two_phase_learning and self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            self.two_phase_phase = (
                "phase1_ojas" if self.two_phase_split_learning else "phase1_gate"
            )
            if self.two_phase_split_learning:
                # Strict two-phase mode: field formation first, recurrent STDP later.
                self._set_learning_flags(enable_ojas=True, enable_stdp=False)
            if self.defer_experience_build_until_phase2_end:
                self.record_experience_transitions = False
            phase1_label = (
                "OJAS only"
                if self.two_phase_split_learning
                else "coverage on, goal/checkpoint counting deferred"
            )
            print(
                f"[TRAIN-2P] Phase 1 start ({phase1_label}). "
                f"min_steps={self.phase1_min_steps}, bin_size={self.phase1_bin_size}m, "
                f"min_revisit_bins={self.phase1_min_revisit_bins}, "
                f"revisit_frac>={self.phase1_revisit_fraction_threshold:.2f}, "
                f"revisit_cos>={self.phase1_revisit_cosine_threshold:.3f}, "
                f"p{int(self.phase1_revisit_low_percentile)}"
                f">={self.phase1_revisit_low_percentile_threshold:.3f}, "
                f"streak>={self.phase1_stability_required_streak}"
            )
        elif self.robot_mode == RobotMode.LEARN_LOCATIONS_ADAPTIVE:
            self.two_phase_phase = "adaptive_ojas"
            self._set_learning_flags(enable_ojas=True, enable_stdp=False)
            self.record_experience_transitions = False
            print(
                f"[ADAPTIVE] OJAS-only start. "
                f"min_steps={self.phase1_min_steps}, bin_size={self.phase1_bin_size}m, "
                f"min_revisit_bins={self.phase1_min_revisit_bins}, "
                f"revisit_frac>={self.phase1_revisit_fraction_threshold:.2f}, "
                f"revisit_cos>={self.phase1_revisit_cosine_threshold:.3f}, "
                f"p{int(self.phase1_revisit_low_percentile)}"
                f">={self.phase1_revisit_low_percentile_threshold:.3f}, "
                f"streak>={self.phase1_stability_required_streak}"
            )

        if self.use_unified_multiscale:
            print("[DRIVER] *** UNIFIED MULTI-SCALE MODE ENABLED ***")
            print("[DRIVER] Using adaptive cross-scale inhibition and unified replay")

        # Coverage-learning specific startup diagnostics.
        if self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_ADAPTIVE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            ojas_enabled, stdp_enabled = self._get_learning_flags()
            print(
                f"[TRAIN] Coverage learning start | unified={self.use_unified_multiscale} "
                f"| OJAS={ojas_enabled} | STDP={stdp_enabled}"
            )
            if self.robot_mode == RobotMode.LEARN_LOCATIONS_ADAPTIVE:
                print(
                    f"[TRAIN] Adaptive extras | stop_ojas_on_stability="
                    f"{int(self.pcn_learning_stop_ojas_on_stabilization)}"
                )
            if hasattr(self, "target_coverage_percentage"):
                print(
                    f"[TRAIN] Targets | coverage>={self.target_coverage_percentage*100:.1f}% "
                    f"| min_goal_visits={self.min_goal_visits}"
                )
            if hasattr(self, "goal_visit_counts"):
                goal_names = ", ".join(list(self.goal_visit_counts.keys()))
                print(f"[TRAIN] Goals: {goal_names}")
            if self.two_phase_learning:
                print("[TRAIN] Two-phase mode enabled: OJAS stabilization -> STDP coverage/goal learning")

        # Step once
        self.step(self.timestep)

        # Track trial start time for per-trial timeouts
        self.trial_start_time = self.getTime()
        print(f"[DRIVER] Trial started at simulation time: {self.trial_start_time:.1f}s")

    def _translation_to_planar_xy(self, translation) -> List[float]:
        """Convert Webots translation [x, y, z] to this world's planar [x, y]."""
        ax0, ax1 = self.planar_axis_indices
        return [float(translation[ax0]), float(translation[ax1])]

    def _translation_height(self, translation) -> float:
        """Return the vertical coordinate for the active world convention."""
        return float(translation[self.height_axis_idx])

    def _compose_translation(
        self, planar_x: float, planar_y: float, height: Optional[float] = None
    ) -> List[float]:
        """Build a Webots translation vector from planar coordinates plus height."""
        if height is None:
            height = float(self.spawn_ground_y)
        vec = [0.0, 0.0, 0.0]
        ax0, ax1 = self.planar_axis_indices
        vec[ax0] = float(planar_x)
        vec[ax1] = float(planar_y)
        vec[self.height_axis_idx] = float(height)
        return vec

    def _current_planar_xy(self) -> List[float]:
        """Read the robot's current planar position in the controller's canonical frame."""
        return self._translation_to_planar_xy(
            self.robot.getField("translation").getSFVec3f()
        )

    def _clockwise_heading_deg_to_standard_rad(self, heading_deg: float) -> float:
        """Convert this controller's clockwise heading degrees into a standard math angle."""
        return -np.deg2rad(float(heading_deg))

    def _planar_delta_to_clockwise_heading_deg(
        self,
        delta_x: float,
        delta_y: float,
    ) -> Optional[float]:
        """Convert a planar displacement into the controller's clockwise heading convention."""
        if math.hypot(float(delta_x), float(delta_y)) <= 1e-9:
            return None
        angle_deg = float(np.degrees(np.arctan2(delta_y, delta_x)))
        return (-angle_deg) % 360.0

    def _current_planar_tensor(self, device=None, dtype=None) -> torch.Tensor:
        """Current planar position as a tensor."""
        planar = self._current_planar_xy()
        return torch.tensor(planar, dtype=dtype or self.dtype, device=device or self.device)

    def _configure_unified_replay_settings(self, unified_rcn, goal_map_mode: bool = False):
        """Apply controller-level replay and reward-readout knobs to a unified RCN."""
        if unified_rcn is None:
            return
        active_goal_map_mode = str(
            getattr(self, "unified_goal_map_mode", "paper_room_local_replay")
        ).strip().lower()
        if goal_map_mode:
            if active_goal_map_mode == "paper_room_local_replay":
                path_w, diff_w = 1.0, 0.0
            else:
                goal_path = getattr(self, "goal_map_path_replay_weight", None)
                goal_diff = getattr(self, "goal_map_diffusion_replay_weight", None)
                path_w = float(max(0.0, goal_path if goal_path is not None else 1.0))
                diff_w = float(max(0.0, goal_diff if goal_diff is not None else 0.0))
            goal_exp_topk = getattr(self, "goal_map_experience_transition_topk", None)
        else:
            path_w = float(max(0.0, getattr(self, "hybrid_path_replay_weight", 0.8)))
            diff_w = float(max(0.0, getattr(self, "hybrid_diffusion_replay_weight", 0.2)))
            goal_exp_topk = None
        if path_w + diff_w <= 1e-12:
            path_w, diff_w = (1.0, 0.0) if goal_map_mode else (0.8, 0.2)
        unified_rcn.enable_hybrid_replay = bool(diff_w > 1e-12)
        unified_rcn.path_replay_weight = path_w
        unified_rcn.diffusion_replay_weight = diff_w
        if goal_exp_topk is not None:
            unified_rcn.experience_transition_topk = int(max(1, goal_exp_topk))
        unified_rcn.reward_normalization_mode = str(
            getattr(self, "unified_reward_normalization_mode", "input_l1")
        ).strip().lower()
        unified_rcn.replay_residual_mix = 0.0
        unified_rcn.goal_map_generation_mode = active_goal_map_mode
        unified_rcn.reward_denominator_override = None
        if goal_map_mode and unified_rcn.goal_map_generation_mode == "room_masked_experience_replay":
            unified_rcn.use_global_decay = False
            unified_rcn.lambda_per_scale = [15.0, 25.0, 45.0]
            unified_rcn.lambda_s = float(unified_rcn.lambda_per_scale[1])
            unified_rcn.lambda_global = float(
                sum(unified_rcn.lambda_per_scale) / float(len(unified_rcn.lambda_per_scale))
            )
            if hasattr(unified_rcn, "_build_lambda_per_pc"):
                unified_rcn.lambda_per_pc = unified_rcn._build_lambda_per_pc()
        unified_rcn.goal_map_path_topk = int(max(1, getattr(self, "goal_map_path_topk", 16)))
        unified_rcn.goal_map_path_decay = float(
            min(0.9999, max(0.0, getattr(self, "goal_map_path_decay", 0.97)))
        )
        unified_rcn.goal_map_path_frontier_only = bool(
            getattr(self, "goal_map_path_frontier_only", True)
        )
        unified_rcn.goal_map_neighbor_topk = int(
            max(1, getattr(self, "goal_map_neighbor_topk", 12))
        )
        unified_rcn.goal_map_neighbor_steps = int(
            max(0, getattr(self, "goal_map_neighbor_steps", 6))
        )
        unified_rcn.goal_map_neighbor_decay = float(
            min(0.9999, max(0.0, getattr(self, "goal_map_neighbor_decay", 0.84)))
        )
        unified_rcn.goal_map_neighbor_frontier_only = bool(
            getattr(self, "goal_map_neighbor_frontier_only", True)
        )
        unified_rcn.goal_map_neighbor_seed_scale = float(
            min(1.0, max(0.0, getattr(self, "goal_map_neighbor_seed_scale", 0.60)))
        )
        unified_rcn.goal_map_checkpoint_support_mode = str(
            getattr(self, "goal_map_checkpoint_support_mode", "threshold_normalized")
        ).strip().lower()
        unified_rcn.goal_map_checkpoint_stop_parent_replay = bool(
            getattr(self, "goal_map_checkpoint_stop_parent_replay", True)
        )
        # Legacy fallback path if hybrid mode is disabled in future.
        unified_rcn.experience_mix_eta = diff_w / max(path_w + diff_w, 1e-12)

    ##########################################################################
    #                           PCN / RCN LOADING                            #
    ##########################################################################
    def load_pcns(self, enable_ojas: Optional[bool], enable_stdp: Optional[bool]):
        if self.use_unified_multiscale:
            self._load_unified_pcn(enable_ojas, enable_stdp)
        else:
            self.pcns = []
            for i, scale_def in enumerate(self.scales):
                scale_idx = scale_def["scale_index"]
                fname = f"pcn_scale_{scale_idx}.pkl"
                path = os.path.join(self.network_dir, fname)

                # Get corresponding grid cell network (may be None)
                gcn = self.gcns[i]
                num_grid_cells = int(getattr(gcn, "total_grid_cells", scale_def.get("num_grid_cells", 0))) if gcn else 0

                pcn = self._load_or_init_pcn_for_scale(
                    path,
                    scale_def,
                    num_grid_cells,
                    enable_ojas if enable_ojas else None,
                    enable_stdp if enable_stdp else None,
                )

                self.pcns.append(pcn)

    def _load_unified_pcn(self, enable_ojas, enable_stdp):
        """Load or initialize the unified multi-scale PCN."""
        path = os.path.join(self.network_dir, "unified_pcn.pkl")
        scale_configs = []
        for scale_def, gcn in zip(self.scales, self.gcns):
            cfg = dict(scale_def)
            if gcn is not None:
                actual_gc = int(getattr(gcn, "total_grid_cells", cfg.get("num_grid_cells", 0)))
                if int(cfg.get("num_grid_cells", actual_gc)) != actual_gc:
                    print(
                        f"[DRIVER] WARNING: scale {cfg.get('scale_index', '?')} num_grid_cells config "
                        f"({cfg.get('num_grid_cells')}) != GCN total ({actual_gc}); using GCN total."
                    )
                cfg["num_grid_cells"] = actual_gc
            else:
                cfg["num_grid_cells"] = 0
            scale_configs.append(cfg)

        # Unified parameters are derived from scale configs (no driver-side hardcoded defaults).
        gamma_pp_values = [float(s["gamma_pp"]) for s in scale_configs]
        gamma_pb_values = [float(s["gamma_pb"]) for s in scale_configs]
        gamma_pg_values = [float(s["gamma_pg"]) for s in scale_configs]
        grid_influences = [float(s["grid_influence"]) for s in scale_configs]
        gamma_cross_values = [float(s["gamma_cross"]) for s in scale_configs]
        gamma_cross = float(np.mean(gamma_cross_values))
        sigma_tune_values = []
        sigma_tune_k_values = []
        for s in scale_configs:
            sigma_r = float(s.get("sigma_r", 1.0))
            sigma_tune_k = float(s.get("sigma_tune_k", 1.0))
            sigma_tune_fallback = float(s.get("sigma_tune", sigma_r * sigma_tune_k))
            sigma_tune_values.append(max(1e-3, sigma_tune_fallback))
            sigma_tune_k_values.append(sigma_tune_k)
        sigma_tune = float(np.mean(sigma_tune_values))
        d_opt_values = [float(s["d_opt"]) for s in scale_configs]
        largest_cfg = scale_configs[-1] if len(scale_configs) > 0 else {}
        large_scale_one_sided = bool(largest_cfg.get("large_scale_one_sided", False))
        large_scale_plateau = float(largest_cfg.get("large_scale_plateau", 1.0))
        large_scale_plateau_onset_sigma = float(
            max(0.0, largest_cfg.get("large_scale_plateau_onset_sigma", 1.0))
        )
        large_scale_plateau_full_sigma = float(
            max(
                large_scale_plateau_onset_sigma,
                largest_cfg.get("large_scale_plateau_full_sigma", 2.0),
            )
        )
        def _uniform_scale_value(key, default, tol=1e-9):
            values = [s.get(key, default) for s in scale_configs]
            if len(values) == 0:
                return default
            first = values[0]
            if isinstance(first, bool):
                return first if all(bool(v) == bool(first) for v in values) else default
            if isinstance(first, str):
                first_str = str(first)
                return first_str if all(str(v) == first_str for v in values) else default
            try:
                first_float = float(first)
            except (TypeError, ValueError):
                return default
            for value in values[1:]:
                try:
                    current = float(value)
                except (TypeError, ValueError):
                    return default
                if abs(current - first_float) > tol:
                    return default
            return first

        correlation_weighting_enabled = bool(
            _uniform_scale_value("enable_correlation_weighting", False)
        )
        correlation_window = int(_uniform_scale_value("correlation_window", 100))
        correlation_update_freq = int(
            _uniform_scale_value("correlation_update_freq", 10)
        )
        correlation_scaling = float(_uniform_scale_value("correlation_scaling", 2.0))
        min_correlation_weight = float(
            _uniform_scale_value("min_correlation_weight", 0.1)
        )
        correlation_threshold = float(_uniform_scale_value("correlation_threshold", 0.01))
        adjacency_modes = [
            str(s.get("adjacency_learning_mode", "dense")).strip().lower()
            for s in scale_configs
        ]
        adjacency_learning_mode = str(
            _uniform_scale_value("adjacency_learning_mode", "dense")
        ).strip().lower()
        if adjacency_learning_mode not in {"dense", "topk"}:
            adjacency_learning_mode = "dense"
        adjacency_topk_values = [
            int(v)
            for v in (s.get("adjacency_topk", None) for s in scale_configs)
            if v is not None
        ]
        adjacency_topk = (
            int(_uniform_scale_value("adjacency_topk", adjacency_topk_values[0]))
            if len(adjacency_topk_values) == len(scale_configs) and len(adjacency_topk_values) > 0
            and len(set(adjacency_topk_values)) == 1
            else None
        )
        adjacency_activity_floor = float(
            _uniform_scale_value("adjacency_activity_floor", 0.0)
        )
        enable_adaptive_stdp = bool(
            _uniform_scale_value("enable_adaptive_stdp", False)
        )
        adaptive_initial_lr = float(
            _uniform_scale_value("adaptive_initial_lr", 0.1)
        )
        adaptive_final_lr = float(_uniform_scale_value("adaptive_final_lr", 0.03))
        adaptive_decay_rate = float(_uniform_scale_value("adaptive_decay_rate", 3.0))
        stdp_learning_rate = float(_uniform_scale_value("stdp_learning_rate", 0.01))
        tau_hd = float(_uniform_scale_value("tau_hd", 0.1))
        enable_connection_decay = bool(
            _uniform_scale_value("enable_connection_decay", True)
        )
        connection_decay_rate = float(
            _uniform_scale_value("connection_decay_rate", 1e-5)
        )
        w_in_init_ratio = float(np.mean([s["w_in_init_ratio"] for s in scale_configs]))

        try:
            with open(path, "rb") as f:
                unified_pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing unified PCN from {path}")
            # --- Ensure BVC modulation attributes exist (for old pickles) ---
            if not hasattr(unified_pcn, "use_bvc_context_modulation"):
                unified_pcn.use_bvc_context_modulation = self.use_bvc_context_modulation

            if not hasattr(unified_pcn, "bvc_context_gain_floor"):
                unified_pcn.bvc_context_gain_floor = self.bvc_context_gain_floor

            if not hasattr(unified_pcn, "bvc_context_gain_strength"):
                unified_pcn.bvc_context_gain_strength = self.bvc_context_gain_strength

            # Guard against older unified PCN pickles that do not include grid integration fields.
            required_attrs = ["grid_influence", "alpha_pg_per_pc", "num_grid_total"]
            if not all(hasattr(unified_pcn, attr) for attr in required_attrs):
                raise pickle.UnpicklingError("Legacy unified PCN format detected; reinitializing with grid support.")
            expected_num_pc = sum(s["num_pc"] for s in scale_configs)
            expected_num_gc = sum(s.get("num_grid_cells", 0) for s in scale_configs)
            if (
                getattr(unified_pcn, "num_pc_total", None) != expected_num_pc
                or getattr(unified_pcn, "num_grid_total", None) != expected_num_gc
                or getattr(unified_pcn, "d_opt", torch.empty(0)).numel() != len(scale_configs)
                or not hasattr(unified_pcn, "bvc_layers")
                or len(getattr(unified_pcn, "bvc_layers", [])) != len(scale_configs)
            ):
                raise pickle.UnpicklingError("Unified PCN shape/config mismatch; reinitializing.")

            unified_pcn.enable_ojas = bool(enable_ojas)
            unified_pcn.enable_stdp = bool(enable_stdp)
            unified_pcn.scale_configs = copy.deepcopy(scale_configs)
            if hasattr(unified_pcn, "_build_d_opt_per_pc"):
                unified_pcn._build_d_opt_per_pc()
            if getattr(unified_pcn, "n_hd", None) != self.n_hd:
                unified_pcn.n_hd = int(self.n_hd)
            unified_pcn.soft_scale_overlap = bool(self.pcn_soft_scale_overlap)
            unified_pcn.soft_scale_overlap_in_learning = bool(
                self.pcn_soft_scale_overlap_in_learning
            )
            unified_pcn.learning_adaptation_mode = str(
                self.pcn_learning_adaptation_mode
            )
            unified_pcn.learning_stdp_start_steps = int(
                self.pcn_learning_stdp_start_steps
            )
            unified_pcn.learning_cross_scale_coupling_start_steps = int(
                self.pcn_learning_cross_scale_coupling_start_steps
            )
            unified_pcn.learning_cross_scale_coupling_ramp_steps = int(
                self.pcn_learning_cross_scale_coupling_ramp_steps
            )
            unified_pcn.learning_cross_scale_coupling_min = float(
                self.pcn_learning_cross_scale_coupling_min
            )
            unified_pcn.soft_scale_gate_floor = float(self.pcn_soft_scale_gate_floor)
            unified_pcn.soft_scale_gate_floor_in_learning = float(
                self.pcn_soft_scale_gate_floor_in_learning
            )
            unified_pcn.soft_cross_inhibition_scale = float(self.pcn_soft_cross_inhibition_scale)
            unified_pcn.soft_cross_inhibition_scale_in_learning = float(
                self.pcn_soft_cross_inhibition_scale_in_learning
            )
            unified_pcn.soft_cross_inhibition_cap = float(self.pcn_soft_cross_inhibition_cap)
            unified_pcn.cross_scale_inhibition_base_enabled = bool(
                self.pcn_cross_scale_inhibition_base_enabled
            )
            unified_pcn.cross_scale_inhibition_lambda_base = float(
                getattr(unified_pcn, "cross_scale_inhibition_lambda_base", 0.20)
            )
            for legacy_attr in (
                "learning_plasticity_bias_rule",
                "learning_plasticity_bias_floor",
                "learning_plasticity_bias_power",
                "learning_cross_scale_inhibition_scale",
                "learning_stdp_use_plasticity_bias",
                "post_competition_scale_bias_base",
                "post_competition_scale_bias_prior_strength",
                "post_competition_scale_bias_cross_strength",
                "post_competition_scale_bias_relative_strength",
                "post_competition_scale_bias_self_strength",
                "post_competition_scale_bias_max",
                "score_routed_scale_evidence_topk",
                "score_routed_scale_evidence_ema_decay",
            ):
                if hasattr(unified_pcn, legacy_attr):
                    delattr(unified_pcn, legacy_attr)
            unified_pcn.enable_correlation_weighting = bool(
                correlation_weighting_enabled
            )
            unified_pcn.correlation_window = int(max(10, correlation_window))
            unified_pcn.correlation_update_freq = int(max(1, correlation_update_freq))
            unified_pcn.correlation_scaling = float(correlation_scaling)
            unified_pcn.min_correlation_weight = float(
                min(1.0, max(0.0, min_correlation_weight))
            )
            unified_pcn.correlation_threshold = float(max(0.0, correlation_threshold))
            unified_pcn.adjacency_learning_mode = str(adjacency_learning_mode)
            unified_pcn.adjacency_topk = (
                int(adjacency_topk) if adjacency_topk is not None else None
            )
            unified_pcn.adjacency_activity_floor = float(
                max(0.0, adjacency_activity_floor)
            )
            unified_pcn.enable_adaptive_stdp = bool(enable_adaptive_stdp)
            unified_pcn.adaptive_initial_lr = float(max(0.0, adaptive_initial_lr))
            unified_pcn.adaptive_final_lr = float(max(0.0, adaptive_final_lr))
            unified_pcn.adaptive_decay_rate = float(max(0.0, adaptive_decay_rate))
            unified_pcn.stdp_learning_rate = float(max(0.0, stdp_learning_rate))
            unified_pcn.tau_hd = float(max(1e-6, tau_hd))
            unified_pcn.enable_connection_decay = bool(enable_connection_decay)
            unified_pcn.connection_decay_rate = float(
                min(0.999999, max(0.0, connection_decay_rate))
            )
            if hasattr(unified_pcn, "_ensure_learning_caches"):
                unified_pcn._ensure_learning_caches()
            if unified_pcn.enable_stdp:
                # Loaded pickles may have STDP toggled off at save time and thus missing traces.
                if getattr(unified_pcn, "place_cell_trace", None) is None:
                    unified_pcn.place_cell_trace = torch.zeros(
                        unified_pcn.num_pc_total,
                        dtype=unified_pcn.dtype,
                        device=unified_pcn.device,
                    )
                if getattr(unified_pcn, "hd_cell_trace", None) is None:
                    unified_pcn.hd_cell_trace = torch.zeros(
                        (self.n_hd, 1, 1),
                        dtype=unified_pcn.dtype,
                        device=unified_pcn.device,
                    )
            else:
                unified_pcn.place_cell_trace = None
                unified_pcn.hd_cell_trace = None
            unified_pcn.gamma_pp_per_pc = torch.cat([
                torch.full((cfg["num_pc"],), gamma_pp_values[i], dtype=unified_pcn.dtype, device=unified_pcn.device)
                for i, cfg in enumerate(scale_configs)
            ])
            unified_pcn.gamma_pb_per_pc = torch.cat([
                torch.full((cfg["num_pc"],), gamma_pb_values[i], dtype=unified_pcn.dtype, device=unified_pcn.device)
                for i, cfg in enumerate(scale_configs)
            ])
            unified_pcn.gamma_pg_per_pc = torch.cat([
                torch.full((cfg["num_pc"],), gamma_pg_values[i], dtype=unified_pcn.dtype, device=unified_pcn.device)
                for i, cfg in enumerate(scale_configs)
            ])
            unified_pcn.grid_influence_per_pc = torch.cat([
                torch.full((cfg["num_pc"],), grid_influences[i], dtype=unified_pcn.dtype, device=unified_pcn.device)
                for i, cfg in enumerate(scale_configs)
            ])
            unified_pcn.gamma_pp = float(np.mean(gamma_pp_values))
            unified_pcn.gamma_pb = float(np.mean(gamma_pb_values))
            unified_pcn.gamma_pg = float(np.mean(gamma_pg_values))
            unified_pcn.grid_influence = float(np.mean(grid_influences))
            unified_pcn.gamma_cross_per_scale = torch.tensor(
                gamma_cross_values, dtype=unified_pcn.dtype, device=unified_pcn.device
            )
            unified_pcn.gamma_cross = gamma_cross
            unified_pcn.sigma_tune_k_per_scale = torch.tensor(
                sigma_tune_k_values, dtype=unified_pcn.dtype, device=unified_pcn.device
            )
            unified_pcn.sigma_tune_per_scale = torch.tensor(
                sigma_tune_values, dtype=unified_pcn.dtype, device=unified_pcn.device
            )
            unified_pcn.d_opt = torch.tensor(d_opt_values, dtype=unified_pcn.dtype, device=unified_pcn.device)
            unified_pcn.sigma_tune = float(np.mean(sigma_tune_values))
            unified_pcn.large_scale_one_sided = large_scale_one_sided
            unified_pcn.large_scale_plateau = min(1.0, max(0.0, large_scale_plateau))
            unified_pcn.large_scale_plateau_onset_sigma = float(
                large_scale_plateau_onset_sigma
            )
            unified_pcn.large_scale_plateau_full_sigma = float(
                large_scale_plateau_full_sigma
            )
            unified_pcn.scale_configs = scale_configs
            if hasattr(unified_pcn, "_build_d_opt_per_pc"):
                unified_pcn._build_d_opt_per_pc()
            if hasattr(unified_pcn, "_configure_grid_balance_from_scale_configs"):
                unified_pcn._configure_grid_balance_from_scale_configs()

            print(f"[DRIVER] Updated unified PCN - enable_ojas: {unified_pcn.enable_ojas}, enable_stdp: {unified_pcn.enable_stdp}")

        except (FileNotFoundError, pickle.UnpicklingError):
            print("[DRIVER] Initializing new unified PCN")

            # Build per-scale BVC layers so each scale only receives its own sensory basis.
            bvc_layers = []
            for scale_cfg in scale_configs:
                bvc_layers.append(
                    BoundaryVectorCellLayer(
                        max_dist=self.max_dist,
                        n_res=720,
                        n_hd=self.n_hd,
                        sigma_theta=scale_cfg["sigma_theta"],
                        sigma_r=scale_cfg["sigma_r"],
                        num_bvc_per_dir=int(scale_cfg["num_bvc_per_dir"]),
                        device=self.device,
                    )
                )

            unified_pcn = UnifiedMultiScalePCN(
                scale_configs=scale_configs,
                bvc_layers=bvc_layers,
                timestep=self.timestep,
                n_hd=self.n_hd,
                enable_ojas=bool(enable_ojas),
                enable_stdp=bool(enable_stdp),
                w_in_init_ratio=w_in_init_ratio,
                grid_influence=float(np.mean(grid_influences)),
                gamma_pp=float(np.mean(gamma_pp_values)),
                gamma_pb=float(np.mean(gamma_pb_values)),
                gamma_pg=float(np.mean(gamma_pg_values)),
                gamma_cross=gamma_cross_values,
                sigma_tune=sigma_tune,
                gate_mode=self.pcn_gate_mode,
                soft_scale_overlap=self.pcn_soft_scale_overlap,
                soft_scale_overlap_in_learning=self.pcn_soft_scale_overlap_in_learning,
                learning_adaptation_mode=self.pcn_learning_adaptation_mode,
                learning_stdp_start_steps=self.pcn_learning_stdp_start_steps,
                learning_cross_scale_coupling_start_steps=self.pcn_learning_cross_scale_coupling_start_steps,
                learning_cross_scale_coupling_ramp_steps=self.pcn_learning_cross_scale_coupling_ramp_steps,
                learning_cross_scale_coupling_min=self.pcn_learning_cross_scale_coupling_min,
                enable_correlation_weighting=correlation_weighting_enabled,
                correlation_window=correlation_window,
                correlation_update_freq=correlation_update_freq,
                correlation_scaling=correlation_scaling,
                min_correlation_weight=min_correlation_weight,
                correlation_threshold=correlation_threshold,
                adjacency_learning_mode=adjacency_learning_mode,
                adjacency_topk=adjacency_topk,
                adjacency_activity_floor=adjacency_activity_floor,
                enable_adaptive_stdp=enable_adaptive_stdp,
                adaptive_initial_lr=adaptive_initial_lr,
                adaptive_final_lr=adaptive_final_lr,
                adaptive_decay_rate=adaptive_decay_rate,
                stdp_learning_rate=stdp_learning_rate,
                tau_hd=tau_hd,
                enable_connection_decay=enable_connection_decay,
                connection_decay_rate=connection_decay_rate,
                soft_scale_gate_floor=self.pcn_soft_scale_gate_floor,
                soft_scale_gate_floor_in_learning=self.pcn_soft_scale_gate_floor_in_learning,
                soft_cross_inhibition_scale=self.pcn_soft_cross_inhibition_scale,
                soft_cross_inhibition_scale_in_learning=self.pcn_soft_cross_inhibition_scale_in_learning,
                soft_cross_inhibition_cap=self.pcn_soft_cross_inhibition_cap,
                cross_scale_inhibition_base_enabled=self.pcn_cross_scale_inhibition_base_enabled,
                use_bvc_context_modulation=self.use_bvc_context_modulation,
                bvc_context_gain_floor=self.bvc_context_gain_floor,
                bvc_context_gain_strength=self.bvc_context_gain_strength,
                device=self.device,
            )

        self.unified_pcn = unified_pcn
        self.pcns = [unified_pcn]

    def _load_or_init_pcn_for_scale(self, path, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        try:
            with open(path, "rb") as f:
                pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing PCN from {path}")

            # Check PCN version compatibility
            pcn_class_name = pcn.__class__.__name__

            if pcn_class_name == "MultiscalePlaceCellWithGrid":
                # Check if loaded PCN has matching n_hd (head direction cells)
                if pcn.n_hd != self.n_hd:
                    print(f"[DRIVER] WARNING: Loaded PCN has n_hd={pcn.n_hd} but driver expects n_hd={self.n_hd}")
                    print(f"[DRIVER] Reinitializing hd_cell_trace tensor to match current n_hd")
                    pcn.n_hd = self.n_hd
                    # Reinitialize hd_cell_trace with correct shape
                    pcn.hd_cell_trace = torch.zeros(
                        (self.n_hd, 1, 1), dtype=torch.float64, device=self.device
                    )
                    # Reinitialize w_rec_hd with correct shape if it exists
                    if hasattr(pcn, 'w_rec_hd'):
                        pcn.w_rec_hd = torch.zeros(
                            (self.n_hd, pcn.num_pc, pcn.num_pc),
                            dtype=pcn.dtype,
                            device=self.device
                        )

                # Update parameters if needed
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp

                # Update place cell inhibition parameters from scale definition
                pcn.gamma_pp = scale_def.get("gamma_pp", 0.5)
                pcn.gamma_pb = scale_def.get("gamma_pb", 0.3)

                # Update grid cell parameters from scale definition
                pcn.grid_influence = scale_def.get("grid_influence", 0.5)
                pcn.gamma_pg = scale_def.get("gamma_pg", 0.3)

                # Update correlation based weighting parameters from scale definition
                pcn.enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
                pcn.correlation_window = scale_def.get("correlation_window", 100)
                pcn.correlation_update_freq = scale_def.get("correlation_update_freq", 10)
                pcn.correlation_scaling = scale_def.get("correlation_scaling", 2.0)
                pcn.min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
                pcn.correlation_threshold = scale_def.get("correlation_threshold", 0.01)

                # Update Oja's learning normalization parameters from scale definition
                if scale_def.get("alpha_pb") is not None:
                    pcn.alpha_pb = scale_def.get("alpha_pb")
                if scale_def.get("alpha_pg") is not None:
                    pcn.alpha_pg = scale_def.get("alpha_pg")

                print(f"[DRIVER] Updated MultiscalePlaceCellWithGrid PCN - grid_influence: {pcn.grid_influence}, gamma_pp: {pcn.gamma_pp}, gamma_pb: {pcn.gamma_pb}, alpha_pb: {pcn.alpha_pb:.4f}, alpha_pg: {pcn.alpha_pg:.4f}")
            else:
                # Update legacy PCN parameters
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp

                # Update place cell inhibition parameters from scale definition
                pcn.gamma_pp = scale_def.get("gamma_pp", 0.5)
                pcn.gamma_pb = scale_def.get("gamma_pb", 0.3)

            print(f"[DRIVER] Updated PCN for {path} - enable_ojas: {pcn.enable_ojas}, enable_stdp: {pcn.enable_stdp}")

        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[DRIVER] Initializing new PCN for {path}")

            # Get BVC parameters from scale definition
            num_bvc_per_dir = scale_def.get("num_bvc_per_dir", 50)

            bvc = BoundaryVectorCellLayer(
                max_dist=self.max_dist,
                n_res=720,
                n_hd=self.n_hd,
                sigma_theta=scale_def.get("sigma_theta"),
                sigma_r=scale_def.get("sigma_r"),
                num_bvc_per_dir=num_bvc_per_dir,
                device=self.device,
            )

            print(f"[DRIVER] Created BVC layer with {num_bvc_per_dir} BVCs per direction (total: {num_bvc_per_dir * self.n_hd} BVCs)")

            # Check if grid cells are enabled for this scale
            if num_grid_cells > 0:
                # Get weight initialization parameters
                w_in_init_ratio = scale_def.get("w_in_init_ratio", 0.25)
                w_grid_init_ratio = scale_def.get("w_grid_init_ratio", 0.25)

                # Get correlation based weighting parameters
                enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
                correlation_window = scale_def.get("correlation_window", 100)
                correlation_update_freq = scale_def.get("correlation_update_freq", 10)
                correlation_scaling = scale_def.get("correlation_scaling", 2.0)
                min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
                correlation_threshold = scale_def.get("correlation_threshold", 0.01)

                # Get Oja's learning normalization parameters
                alpha_pb = scale_def.get("alpha_pb", None)  # None will use default sqrt(0.5)
                alpha_pg = scale_def.get("alpha_pg", None)  # None will use default sqrt(0.5)

                # Use MultiscalePlaceCellWithGrid
                # Note: Proximity suppression is now handled in the grid cell layer
                pcn = MultiscalePlaceCellWithGrid(
                    bvc_layer=bvc,
                    num_pc=scale_def["num_pc"],
                    num_grid_cells=num_grid_cells,
                    timestep=self.timestep,
                    n_hd=self.n_hd,
                    enable_ojas=enable_ojas if enable_ojas is not None else False,
                    enable_stdp=enable_stdp if enable_stdp is not None else False,
                    w_in_init_ratio=w_in_init_ratio,
                    w_grid_init_ratio=w_grid_init_ratio,
                    w_grid_init_strategy=scale_def.get('w_grid_init_strategy', 'balanced_modules'),
                    gc_num_modules=scale_def.get('num_modules'),
                    gc_cells_per_module=scale_def.get('cells_per_module'),
                    grid_influence=scale_def.get("grid_influence", 0.5),
                    gamma_pp=scale_def.get("gamma_pp", 0.5),
                    gamma_pb=scale_def.get("gamma_pb", 0.25),
                    gamma_pg=scale_def.get("gamma_pg", 0.3),
                    alpha_pb=alpha_pb,
                    alpha_pg=alpha_pg,
                    enable_correlation_weighting = enable_correlation_weighting,
                    correlation_window = correlation_window,
                    correlation_update_freq = correlation_update_freq,
                    correlation_scaling = correlation_scaling,
                    min_correlation_weight = min_correlation_weight,
                    correlation_threshold = correlation_threshold,
                    device=self.device,
                )
                alpha_pb_val = alpha_pb if alpha_pb is not None else np.sqrt(0.5)
                alpha_pg_val = alpha_pg if alpha_pg is not None else np.sqrt(0.5)
                print(f"[DRIVER] Created MultiscalePlaceCellWithGrid with {num_grid_cells} grid cells, grid_influence={scale_def.get('grid_influence', 0.5)}, w_in_ratio={w_in_init_ratio}, w_grid_ratio={w_grid_init_ratio}, alpha_pb={alpha_pb_val:.4f}, alpha_pg={alpha_pg_val:.4f}")
            else:
                # Use standard PlaceCellLayer
                w_in_init_ratio = scale_def.get("w_in_init_ratio", 0.25)

                pcn = PlaceCellLayer(
                    bvc_layer=bvc,
                    num_pc=scale_def["num_pc"],
                    timestep=self.timestep,
                    n_hd=self.n_hd,
                    enable_ojas=enable_ojas,
                    enable_stdp=enable_stdp,
                    w_in_init_ratio=w_in_init_ratio,
                    gamma_pp=scale_def.get("gamma_pp", 0.5),
                    gamma_pb=scale_def.get("gamma_pb", 0.3),
                    device=self.device,
                )
                print(f"[DRIVER] Created standard PlaceCellLayer without grid cells, w_in_ratio={w_in_init_ratio}")

        return pcn

    def load_rcns(self):
        if self.use_unified_multiscale:
            self._load_unified_rcn()
        else:
            self.rcns = []
            for scale_def in self.scales:
                scale_idx = scale_def["scale_index"]
                fname = f"rcn_scale_{scale_idx}.pkl"
                path = os.path.join(self.network_dir, fname)
                learning_rate = scale_def["rcn_learning_rate"]
                rcn = self._load_or_init_rcn_for_scale(path, scale_def, learning_rate)
                self.rcns.append(rcn)

    def _load_or_init_rcn_for_scale(self, path, scale_def, learning_rate):
        try:
            with open(path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing RCN from {path}")
        except:
            print(f"[DRIVER] Initializing new RCN for {path} with learning rate {learning_rate}")
            # Get sigma_pc_s from scale definition (use sigma_pc_s if provided, otherwise sigma_r)
            sigma_pc_s = scale_def.get("sigma_pc_s", scale_def.get("sigma_r"))

            rcn = RewardCellLayerTest(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                replay_timesteps=scale_def.get("replay_timesteps", 20),
                replay_decay_factor=scale_def.get("replay_decay_factor", 6),
                custom_replay_timesteps=scale_def.get("custom_replay_timesteps", _steps_for_scale(scale_def)),
                initial_value_multiplier=scale_def.get("initial_value_multiplier", 1.0),
                sigma_pc_s=sigma_pc_s,
                device=self.device,
            )
        return rcn

    def _load_unified_rcn(self):
        """Load or initialize the unified multi-scale RCN."""
        path = os.path.join(self.network_dir, "unified_rcn.pkl")
        total_pc = sum(scale["num_pc"] for scale in self.scales)
        anchor_scale = next((s for s in self.scales if s.get("name") == "medium"), self.scales[0])
        learning_rate = float(anchor_scale["rcn_learning_rate"])
        replay_timesteps = int(anchor_scale["replay_timesteps"])

        try:
            with open(path, "rb") as f:
                unified_rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing unified RCN from {path}")
            if getattr(unified_rcn, "num_place_cells_total", None) != total_pc:
                raise pickle.UnpicklingError("Unified RCN shape mismatch; reinitializing.")
            unified_rcn.learning_rate = learning_rate
            unified_rcn.replay_timesteps = replay_timesteps
            if hasattr(unified_rcn, "reconfigure_from_scale_configs"):
                unified_rcn.reconfigure_from_scale_configs(self.scales)
            if hasattr(unified_rcn, "_ensure_experience_buffers"):
                unified_rcn._ensure_experience_buffers()
            if not hasattr(unified_rcn, "use_experience_replay"):
                unified_rcn.use_experience_replay = True
            if not hasattr(unified_rcn, "experience_topk"):
                unified_rcn.experience_topk = 8
            if not hasattr(unified_rcn, "enable_hybrid_replay"):
                unified_rcn.enable_hybrid_replay = True
            if not hasattr(unified_rcn, "path_replay_weight"):
                unified_rcn.path_replay_weight = 0.8
            if not hasattr(unified_rcn, "diffusion_replay_weight"):
                unified_rcn.diffusion_replay_weight = 0.2
            if not hasattr(unified_rcn, "experience_mix_eta"):
                unified_rcn.experience_mix_eta = 0.2
            if not hasattr(unified_rcn, "experience_transition_topk"):
                unified_rcn.experience_transition_topk = 12
            if not hasattr(unified_rcn, "use_global_decay"):
                unified_rcn.use_global_decay = True
            if not hasattr(unified_rcn, "lambda_global"):
                if hasattr(unified_rcn, "_build_global_lambda"):
                    unified_rcn.lambda_global = unified_rcn._build_global_lambda()
                else:
                    unified_rcn.lambda_global = float(getattr(unified_rcn, "lambda_s", 40.0))
            self._configure_unified_replay_settings(unified_rcn)
        except (FileNotFoundError, pickle.UnpicklingError):
            print("[DRIVER] Initializing new unified RCN")

            unified_rcn = UnifiedMultiScaleRCN(
                num_place_cells_total=total_pc,
                scale_configs=self.scales,
                num_replay=3,
                learning_rate=learning_rate,
                replay_timesteps=replay_timesteps,
                device=self.device,
            )
            self._configure_unified_replay_settings(unified_rcn)

        self.unified_rcn = unified_rcn
        self.rcns = [unified_rcn]
        self.loaded_goal_specific_rcn_goal = None
        self._loaded_goal_map_path = None
        self._loaded_goal_map_mtime = None
        self._loaded_goal_map_debug = "base_rcn"
        # Give the RCN scale boundary info so observe_transition can record per-scale top-k.
        if hasattr(self.unified_pcn, "scale_boundaries"):
            self.unified_rcn.scale_boundaries = self.unified_pcn.scale_boundaries

    def _load_goal_specific_rcns(self, goal_name):
        """Load goal-specific RCNs for EXPLOIT_LOCATIONS_RANDOM mode"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        if not os.path.exists(multi_goal_dir):
            print(f"[WARNING] Multi-goal rewards directory not found: {multi_goal_dir}")
            print(f"[WARNING] Make sure to run LEARN_HEBB, LEARN_LOCATIONS_COVERAGE, or LEARN_LOCATIONS_ADAPTIVE first!")
            return

        print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")

        if self.use_unified_multiscale:
            unified_goal_path = os.path.join(multi_goal_dir, f"unified_rcn_goal_{goal_name}.pkl")
            unified_goal_cache = getattr(self, "_goal_specific_unified_rcn_cache", None)
            if not isinstance(unified_goal_cache, dict):
                unified_goal_cache = {}
                self._goal_specific_unified_rcn_cache = unified_goal_cache
            try:
                cached_rcn = unified_goal_cache.get(goal_name)
                goal_map_mtime = None
                goal_map_stamp = "unknown"
                if os.path.exists(unified_goal_path):
                    try:
                        goal_map_mtime = os.path.getmtime(unified_goal_path)
                        goal_map_stamp = time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(goal_map_mtime),
                        )
                    except OSError:
                        goal_map_mtime = None
                if cached_rcn is not None:
                    self.unified_rcn = cached_rcn
                    print(f"[DRIVER] Reusing cached goal-specific unified RCN for goal: {goal_name}")
                else:
                    with open(unified_goal_path, "rb") as f:
                        self.unified_rcn = pickle.load(f)
                    unified_goal_cache[goal_name] = self.unified_rcn
                if hasattr(self.unified_rcn, "_ensure_experience_buffers"):
                    self.unified_rcn._ensure_experience_buffers()
                if not hasattr(self.unified_rcn, "use_experience_replay"):
                    self.unified_rcn.use_experience_replay = True
                if not hasattr(self.unified_rcn, "experience_topk"):
                    self.unified_rcn.experience_topk = 8
                if not hasattr(self.unified_rcn, "enable_hybrid_replay"):
                    self.unified_rcn.enable_hybrid_replay = True
                if not hasattr(self.unified_rcn, "path_replay_weight"):
                    self.unified_rcn.path_replay_weight = 0.8
                if not hasattr(self.unified_rcn, "diffusion_replay_weight"):
                    self.unified_rcn.diffusion_replay_weight = 0.2
                if not hasattr(self.unified_rcn, "experience_mix_eta"):
                    self.unified_rcn.experience_mix_eta = 0.2
                if not hasattr(self.unified_rcn, "experience_transition_topk"):
                    self.unified_rcn.experience_transition_topk = 12
                if not hasattr(self.unified_rcn, "use_global_decay"):
                    self.unified_rcn.use_global_decay = True
                if not hasattr(self.unified_rcn, "lambda_global"):
                    if hasattr(self.unified_rcn, "_build_global_lambda"):
                        self.unified_rcn.lambda_global = self.unified_rcn._build_global_lambda()
                    else:
                        self.unified_rcn.lambda_global = float(getattr(self.unified_rcn, "lambda_s", 40.0))
                if hasattr(self.unified_rcn, "reward_cell_activations"):
                    self.unified_rcn.reward_cell_activations = torch.zeros_like(
                        self.unified_rcn.reward_cell_activations
                    )
                self._configure_unified_replay_settings(self.unified_rcn)
                if hasattr(self.unified_pcn, "scale_boundaries"):
                    self.unified_rcn.scale_boundaries = self.unified_pcn.scale_boundaries
                self.rcns = [self.unified_rcn]
                self.rcn = self.unified_rcn
                self.loaded_goal_specific_rcn_goal = goal_name
                self._loaded_goal_map_path = unified_goal_path
                self._loaded_goal_map_mtime = goal_map_mtime
                self._loaded_goal_map_debug = (
                    f"{os.path.basename(unified_goal_path)}@{goal_map_stamp}"
                )
                print(f"[DRIVER] Loaded goal map artifact: {self._loaded_goal_map_debug}")
                self._unified_exploit_route_stage_cache = None
                # Restore checkpoint PC groups from associations pkl if not
                # already populated (cold-start exploit).
                if not getattr(self, "checkpoint_pc_groups", {}):
                    assoc_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
                    if os.path.exists(assoc_path):
                        try:
                            with open(assoc_path, "rb") as af:
                                assoc_data = pickle.load(af)
                            saved_groups = assoc_data.get("checkpoint_pc_groups", {})
                            if saved_groups:
                                self.checkpoint_pc_groups = {
                                    int(k): {
                                        side: t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)
                                        for side, t in v.items()
                                    }
                                    for k, v in saved_groups.items()
                                }
                        except Exception:
                            pass
                # Restore replay-derived room masks for exploit-time
                # room suppression (persisted on goal_rcn).
                _rcn_room_masks = getattr(
                    self.unified_rcn, "_replay_derived_room_masks", None
                )
                if _rcn_room_masks:
                    self._replay_derived_room_masks = dict(_rcn_room_masks)
                    print(
                        f"[DRIVER] Restored replay-derived room masks: "
                        f"{list(_rcn_room_masks.keys())}"
                    )
                if cached_rcn is None:
                    print(f"[DRIVER] Loaded goal-specific unified RCN: {unified_goal_path}")
                return
            except FileNotFoundError:
                print(f"[ERROR] Goal-specific unified RCN not found: {unified_goal_path}")
                print(f"[ERROR] Make sure LEARN_HEBB/LEARN_LOCATIONS_COVERAGE/LEARN_LOCATIONS_ADAPTIVE has produced unified goal maps.")
                raise

        self.rcns = []

        for scale_def in self.scales:
            scale_idx = scale_def["scale_index"]
            goal_rcn_path = os.path.join(
                multi_goal_dir, f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
            )

            try:
                with open(goal_rcn_path, "rb") as f:
                    rcn = pickle.load(f)
                print(f"[DRIVER] Loaded goal-specific RCN: {goal_rcn_path}")
                self.rcns.append(rcn)
            except FileNotFoundError:
                print(f"[ERROR] Goal-specific RCN not found: {goal_rcn_path}")
                print(f"[ERROR] Make sure LEARN_HEBB/LEARN_LOCATIONS_COVERAGE/LEARN_LOCATIONS_ADAPTIVE has been run for this goal!")
                raise

        # Update single-scale reference
        self.rcn = self.rcns[0] if self.rcns else None

    def _get_active_goal_name_for_exploit(self):
        """Return active goal name if one is selected for exploit-style modes."""
        if hasattr(self, "active_goal_name") and self.active_goal_name:
            return self.active_goal_name
        active_goals = [g for g in getattr(self, "goals", []) if g.get("active", False)]
        if active_goals:
            return active_goals[0]["name"]
        return None

    def _ensure_goal_specific_rcn_loaded_for_exploit(self):
        """
        Ensure exploit uses per-goal reward map when an active goal is defined.
        Applies to unified architecture in exploit-style modes.
        """
        if not self.use_unified_multiscale:
            return
        if self.robot_mode not in {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }:
            return
        if not getattr(self, "multi_goal_mode", False):
            return

        active_goal_name = self._get_active_goal_name_for_exploit()
        if not active_goal_name:
            return

        if getattr(self, "loaded_goal_specific_rcn_goal", None) == active_goal_name:
            return
        self._load_goal_specific_rcns(active_goal_name)

    ##########################################################################
    #                        GRID CELL NETWORK INITIALIZATION                #
    ##########################################################################

    def init_grid_cell_networks(self):
        """Initialize grid cell networks for each scale based on scale parameters."""
        self.gcns = []
        shared_cfg = None
        if self.gcn_scale_invariant:
            for scale_def in self.scales:
                if int(scale_def.get("num_grid_cells", 0)) > 0:
                    shared_cfg = {
                        "num_modules": int(scale_def.get("num_modules", 8)),
                        "cells_per_module": int(
                            scale_def.get(
                                "cells_per_module",
                                max(1, int(scale_def.get("num_grid_cells", 0)) // max(1, int(scale_def.get("num_modules", 8)))),
                            )
                        ),
                        "spread_range": scale_def.get("spread_range", (1.2, 1.2)),
                        "scale_multiplier": scale_def.get("scale_multiplier", 1.0),
                        "module_scale_ratio": float(scale_def.get("module_scale_ratio", 1.0)),
                        "translation_scale": scale_def.get("translation_scale", 1.0),
                        "mask_resolution": scale_def.get("mask_resolution", 128),
                        "smooth_sigma": scale_def.get("smooth_sigma", 1.5),
                        "activation_cache_size": int(scale_def.get("activation_cache_size", 0)),
                        "activation_cache_quantization": scale_def.get("activation_cache_quantization"),
                    }
                    break
            if shared_cfg is not None:
                print(
                    "[DRIVER] GCN scale-invariant mode ON "
                    f"(modules={shared_cfg['num_modules']}, "
                    f"cells/module={shared_cfg['cells_per_module']}, "
                    f"scale_multiplier={shared_cfg['scale_multiplier']}, "
                    f"module_scale_ratio={shared_cfg['module_scale_ratio']}, "
                    f"translation_scale={shared_cfg['translation_scale']})"
                )

        for scale_def in self.scales:
            scale_idx = scale_def["scale_index"]
            num_grid_cells = int(scale_def.get("num_grid_cells", 0))
            if self.gcn_scale_invariant and shared_cfg is not None:
                num_modules = int(shared_cfg["num_modules"])
                cells_per_module = int(shared_cfg["cells_per_module"])
                spread_range = shared_cfg["spread_range"]
                scale_multiplier = shared_cfg["scale_multiplier"]
                module_scale_ratio = shared_cfg["module_scale_ratio"]
                translation_scale = shared_cfg["translation_scale"]
                mask_resolution = shared_cfg["mask_resolution"]
                smooth_sigma = shared_cfg["smooth_sigma"]
                activation_cache_size = int(shared_cfg["activation_cache_size"])
                activation_cache_quantization = shared_cfg["activation_cache_quantization"]
            else:
                num_modules = int(scale_def.get("num_modules", 8))
                cells_per_module = int(
                    scale_def.get(
                        "cells_per_module",
                        max(1, num_grid_cells // max(1, num_modules)),
                    )
                )
                spread_range = scale_def.get("spread_range", (1.2, 1.2))
                scale_multiplier = scale_def.get("scale_multiplier", 1.0)
                module_scale_ratio = float(scale_def.get("module_scale_ratio", 1.0))
                translation_scale = scale_def.get("translation_scale", 1.0)
                mask_resolution = scale_def.get("mask_resolution", 128)
                smooth_sigma = scale_def.get("smooth_sigma", 1.5)
                activation_cache_size = int(scale_def.get("activation_cache_size", 0))
                activation_cache_quantization = scale_def.get("activation_cache_quantization")
            expected_total_cells = num_modules * cells_per_module

            # Explicitly disable GCN for this scale.
            if num_grid_cells == 0:
                self.gcns.append(None)
                continue

            fname = f"gcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)

            try:
                # Try to load existing grid cell network
                with open(path, "rb") as f:
                    gcn = pickle.load(f)
                loaded_total = int(getattr(gcn, "total_grid_cells", -1))
                if loaded_total != expected_total_cells:
                    raise pickle.UnpicklingError(
                        f"GCN size mismatch for scale {scale_idx}: "
                        f"loaded={loaded_total}, expected={expected_total_cells}"
                    )
                loaded_ratio = float(getattr(gcn, "module_scale_ratio", 1.0))
                if abs(loaded_ratio - float(module_scale_ratio)) > 1e-6:
                    raise pickle.UnpicklingError(
                        f"GCN module_scale_ratio mismatch for scale {scale_idx}: "
                        f"loaded={loaded_ratio}, expected={float(module_scale_ratio)}"
                    )
                # Refresh runtime placement for loaded objects.
                if hasattr(gcn, "to"):
                    gcn.to(device=self.device, dtype=self.dtype)
                else:
                    gcn.device = self.device
                    gcn.dtype = self.dtype
                if not hasattr(gcn, "activation_cache_size"):
                    gcn.activation_cache_size = 0
                if not hasattr(gcn, "activation_cache_quantization"):
                    gcn.activation_cache_quantization = None
                if not hasattr(gcn, "_activation_cache"):
                    gcn._activation_cache = OrderedDict()
                if getattr(gcn, "normalization", None) == "per-cell":
                    total_cells = int(getattr(gcn, "total_grid_cells", loaded_total))
                    if not hasattr(gcn, "cell_min") or int(gcn.cell_min.numel()) != total_cells:
                        gcn.cell_min = torch.full(
                            (total_cells,),
                            -1.0,
                            dtype=getattr(gcn, "dtype", self.dtype),
                            device=getattr(gcn, "device", self.device),
                        )
                    if not hasattr(gcn, "cell_max") or int(gcn.cell_max.numel()) != total_cells:
                        gcn.cell_max = torch.full(
                            (total_cells,),
                            1.0,
                            dtype=getattr(gcn, "dtype", self.dtype),
                            device=getattr(gcn, "device", self.device),
                        )
                    gcn.min_max_updated = not self.gcn_use_adaptive_minmax
                print(
                    f"[DRIVER] Loaded existing GCN from {path} "
                    f"(cells={loaded_total})"
                )
            except (FileNotFoundError, pickle.UnpicklingError):
                # Initialize new grid cell network with scale-specific parameters
                print(f"[DRIVER] Initializing new GCN for scale {scale_idx}")

                # Create new grid cell network (module + phase-based + world mask)
                gcn = GridCellLayer(
                    num_modules=num_modules,
                    cells_per_module=cells_per_module,
                    spread_range=spread_range,
                    scale_multiplier=scale_multiplier,
                    module_scale_ratio=module_scale_ratio,
                    translation_scale=translation_scale,
                    threshold=0.7,
                    threshold_type='soft',
                    normalization='per-cell',
                    world_name=self.world_name,
                    mask_resolution=mask_resolution,
                    smooth_sigma=smooth_sigma,
                    activation_cache_size=activation_cache_size,
                    activation_cache_quantization=activation_cache_quantization,
                    device=self.device.type,
                    dtype=self.dtype,
                )
                if getattr(gcn, "normalization", None) == "per-cell":
                    total_cells = int(getattr(gcn, "total_grid_cells", expected_total_cells))
                    gcn.cell_min = torch.full(
                        (total_cells,),
                        -1.0,
                        dtype=getattr(gcn, "dtype", self.dtype),
                        device=getattr(gcn, "device", self.device),
                    )
                    gcn.cell_max = torch.full(
                        (total_cells,),
                        1.0,
                        dtype=getattr(gcn, "dtype", self.dtype),
                        device=getattr(gcn, "device", self.device),
                    )
                    gcn.min_max_updated = not self.gcn_use_adaptive_minmax

            # Add to list of grid cell networks
            self.gcns.append(gcn)

    def _get_used_goal_pcs(self, scale_idx: int, exclude_goal: Optional[str] = None) -> set:
        """Collect already-associated PC indices for a given scale across goals."""
        used = set()
        if not hasattr(self, "goal_place_cell_associations"):
            return used
        for goal_name, per_scale in self.goal_place_cell_associations.items():
            if exclude_goal is not None and goal_name == exclude_goal:
                continue
            if scale_idx < len(per_scale):
                pc_idx = per_scale[scale_idx]
                if pc_idx is not None:
                    used.add(int(pc_idx))
        return used

    def _get_goal_location_by_name(self, goal_name: str) -> Optional[Tuple[float, float]]:
        """Return one goal's `(x, z)` location if it exists."""
        for goal in getattr(self, "goals", []):
            if str(goal.get("name")) != str(goal_name):
                continue
            loc = goal.get("location", [])
            if len(loc) >= 2:
                return float(loc[0]), float(loc[1])
        return None

    def _should_defer_unified_goal_associations(self) -> bool:
        """Return whether unified goal association selection should run offline."""
        return bool(
            getattr(self, "use_unified_multiscale", False)
            and getattr(self, "defer_goal_association_until_training_end", False)
        )

    def _record_goal_contact_step(
        self, goal_name: str, step: Optional[int] = None
    ) -> None:
        """Record a goal-contact step for deferred end-of-training association selection."""
        if not self._should_defer_unified_goal_associations():
            return
        contact_steps = getattr(self, "goal_contact_steps", None)
        if contact_steps is None or goal_name not in contact_steps:
            return
        chosen_step = int(self.step_count if step is None else step)
        history = contact_steps[goal_name]
        if history and int(history[-1]) == chosen_step:
            return
        history.append(chosen_step)
        self.goal_associations_finalized = False

    def _record_checkpoint_crossing_step(
        self,
        checkpoint_idx: int,
        direction_key: str,
        step: Optional[int] = None,
    ) -> None:
        """Record one directed checkpoint crossing step for replay-event selection."""
        crossing_steps = getattr(self, "checkpoint_crossing_steps", None)
        if crossing_steps is None:
            return
        dir_steps = crossing_steps.get(int(checkpoint_idx))
        if not isinstance(dir_steps, dict) or direction_key not in dir_steps:
            return
        chosen_step = int(self.step_count if step is None else step)
        history = dir_steps[direction_key]
        if history and int(history[-1]) == chosen_step:
            return
        history.append(chosen_step)
        if hasattr(self, "goal_associations_finalized"):
            self.goal_associations_finalized = False

    def _group_replay_history_steps_into_events(
        self,
        raw_steps: List[int],
    ) -> List[List[int]]:
        """Collapse dense step histories into replay events using the configured merge gap."""
        cleaned_steps = sorted(
            {
                int(step)
                for step in (raw_steps or [])
                if step is not None and int(step) >= 0
            }
        )
        if not cleaned_steps:
            return []
        merge_gap_steps = int(
            max(0, getattr(self, "goal_map_replay_event_merge_gap_steps", 3))
        )
        grouped = [[cleaned_steps[0]]]
        for step in cleaned_steps[1:]:
            if int(step) <= int(grouped[-1][-1]) + int(merge_gap_steps):
                grouped[-1].append(int(step))
            else:
                grouped.append([int(step)])
        return grouped

    def _get_recent_goal_contact_event_groups(
        self,
        goal_name: str,
        max_events: int = 1,
    ) -> List[List[int]]:
        """Return the most recent grouped goal-contact events for one goal."""
        if int(max_events) <= 0:
            return []
        contact_steps = getattr(self, "goal_contact_steps", {})
        history_start = int(max(0, self._get_reward_history_start_step()))
        filtered_steps = [
            int(step)
            for step in contact_steps.get(str(goal_name), [])
            if int(step) >= history_start
        ]
        grouped = self._group_replay_history_steps_into_events(
            filtered_steps
        )
        if not grouped:
            return []
        return grouped[-int(max(1, max_events)) :]

    def _get_recent_checkpoint_crossing_event_groups(
        self,
        checkpoint_idx: int,
        goal_name: Optional[str] = None,
        goal_location: Optional[Tuple[float, float]] = None,
        max_events: int = 1,
    ) -> List[List[int]]:
        """Return recent checkpoint crossing events for one doorway.

        Checkpoint replay is intentionally independent of goal-side handoff logic.
        We therefore gather recent crossings from both directions, then expand each
        event into a short temporal window so the seed chooser can pick a state
        that still lies inside the checkpoint's own replay mask.
        """
        if int(max_events) <= 0:
            return []
        crossing_steps = getattr(self, "checkpoint_crossing_steps", {})
        dir_steps = crossing_steps.get(int(checkpoint_idx), {})
        history_start = int(max(0, self._get_reward_history_start_step()))
        all_grouped = []
        for direction_key in ("neg_to_pos", "pos_to_neg"):
            filtered_steps = [
                int(step)
                for step in dir_steps.get(direction_key, [])
                if int(step) >= history_start
            ]
            grouped = self._group_replay_history_steps_into_events(
                filtered_steps
            )
            for event_group in grouped:
                if event_group:
                    all_grouped.append((int(event_group[-1]), list(event_group)))
        if not all_grouped:
            return []
        all_grouped.sort(key=lambda item: int(item[0]))

        stride = int(max(1, getattr(self, "hmap_sample_stride", 1)))
        merge_gap = int(
            max(0, getattr(self, "goal_map_replay_event_merge_gap_steps", 3))
        )
        window_radius = int(max(2, (2 * stride), merge_gap + 2))
        recent_groups = all_grouped[-int(max(1, max_events)) :]
        expanded_groups = []
        for _, event_group in recent_groups:
            center_step = int(event_group[-1])
            expanded_groups.append(
                [
                    int(step)
                    for step in range(
                        int(center_step - window_radius),
                        int(center_step + window_radius + 1),
                    )
                    if int(step) >= 0
                ]
            )
        return expanded_groups

    def _get_recent_checkpoint_crossing_event_groups_for_direction(
        self,
        checkpoint_idx: int,
        direction_key: str,
        max_events: int = 1,
    ) -> List[List[int]]:
        """Return the most recent grouped checkpoint crossings for one direction."""
        if int(max_events) <= 0:
            return []
        crossing_steps = getattr(self, "checkpoint_crossing_steps", {})
        dir_steps = crossing_steps.get(int(checkpoint_idx), {})
        history_start = int(max(0, self._get_reward_history_start_step()))
        filtered_steps = [
            int(step)
            for step in dir_steps.get(str(direction_key), [])
            if int(step) >= history_start
        ]
        grouped = self._group_replay_history_steps_into_events(
            filtered_steps
        )
        if not grouped:
            return []
        return grouped[-int(max(1, max_events)) :]

    def _goal_associations_ready_for_completion(self) -> bool:
        """Check whether goal-association state is sufficient for completion gating."""
        if (
            self._should_defer_unified_goal_associations()
            and not getattr(self, "goal_associations_finalized", False)
        ):
            contact_steps = getattr(self, "goal_contact_steps", {})
            for goal_name in self.goal_place_cell_associations.keys():
                if len(contact_steps.get(goal_name, [])) <= 0:
                    return False
            return True

        for goal_name, associations in self.goal_place_cell_associations.items():
            if self.goal_assoc_require_all_scales:
                if any(pc_idx is None for pc_idx in associations):
                    return False
            else:
                if all(pc_idx is None for pc_idx in associations):
                    return False
        return True

    def _finalize_deferred_goal_associations(self) -> bool:
        """
        Build unified goal associations once at training end from recorded goal-contact steps.

        This keeps dense-hmap locality/compactness scans out of the real-time control
        loop while preserving the same final association-selection logic for reward maps.
        """
        if not self._should_defer_unified_goal_associations():
            return True
        if getattr(self, "goal_associations_finalized", False):
            return self._goal_associations_ready_for_completion()
        if not hasattr(self, "goal_contact_steps") or not hasattr(
            self, "goal_place_cell_associations"
        ):
            return False
        if not getattr(self, "hmap_pcn_activities", None):
            print(
                "[LEARN_LOCATIONS] Deferred goal association finalization skipped: no hmap PC history."
            )
            return False

        print(
            "[LEARN_LOCATIONS] Finalizing deferred unified goal associations from recorded goal-contact steps..."
        )
        self._unified_pc_field_peak_cache = None
        for goal_name in self.goal_place_cell_associations.keys():
            self.goal_place_cell_associations[goal_name] = [None] * len(self.scales)
            self.goal_association_step[goal_name] = [None] * len(self.scales)
            self.goal_place_cell_activations[goal_name] = [None] * len(self.scales)

        finalized_count = 0
        for goal in self.goals:
            goal_name = goal["name"]
            raw_steps = self.goal_contact_steps.get(goal_name, [])
            if not raw_steps:
                continue
            valid_steps = sorted(
                {
                    int(step)
                    for step in raw_steps
                    if step is not None and int(step) >= 0
                }
            )
            for scale_idx in range(len(self.scales)):
                if scale_idx >= len(self.hmap_pcn_activities):
                    continue
                scale_history = self.hmap_pcn_activities[scale_idx]
                if scale_history is None or int(scale_history.shape[0]) <= 0:
                    continue
                for step in valid_steps:
                    if int(step) >= int(scale_history.shape[0]):
                        continue
                    scale_acts = scale_history[int(step)]
                    if hasattr(scale_acts, "detach"):
                        scale_acts = scale_acts.detach()
                    most_active_idx = self._select_goal_association_pc(
                        scale_acts=scale_acts,
                        goal_name=goal_name,
                        scale_idx=scale_idx,
                    )
                    if most_active_idx is None:
                        continue
                    activation_value = float(scale_acts[int(most_active_idx)].item())
                    if activation_value <= 0.01:
                        continue

                    stored_idx = self.goal_place_cell_associations[goal_name][scale_idx]
                    if stored_idx is None:
                        self.goal_place_cell_associations[goal_name][scale_idx] = int(
                            most_active_idx
                        )
                        self.goal_association_step[goal_name][scale_idx] = int(step)
                        self.goal_place_cell_activations[goal_name][scale_idx] = (
                            activation_value
                        )
                        finalized_count += 1
                        continue

                    if int(stored_idx) == int(most_active_idx):
                        stored_activation = float(
                            self.goal_place_cell_activations[goal_name][scale_idx]
                            or 0.0
                        )
                        if activation_value > stored_activation:
                            self.goal_association_step[goal_name][scale_idx] = int(step)
                            self.goal_place_cell_activations[goal_name][scale_idx] = (
                                activation_value
                            )
                        continue

                    stored_activation = float(
                        self.goal_place_cell_activations[goal_name][scale_idx] or 0.0
                    )
                    activation_ratio = activation_value / (stored_activation + 1e-6)
                    replace = False
                    if activation_ratio > (1.0 + self.ACTIVATION_SIMILARITY_THRESHOLD):
                        replace = True
                    elif activation_ratio < (1.0 - self.ACTIVATION_SIMILARITY_THRESHOLD):
                        replace = False
                    else:
                        old_strength = self._compute_place_cell_connection_strength(
                            self.unified_pcn, stored_idx, scale_idx
                        )
                        new_strength = self._compute_place_cell_connection_strength(
                            self.unified_pcn, most_active_idx, scale_idx
                        )
                        replace = new_strength >= old_strength

                    if replace:
                        self.goal_place_cell_associations[goal_name][scale_idx] = int(
                            most_active_idx
                        )
                        self.goal_association_step[goal_name][scale_idx] = int(step)
                        self.goal_place_cell_activations[goal_name][scale_idx] = (
                            activation_value
                        )

        self.goal_associations_finalized = True
        ready = self._goal_associations_ready_for_completion()
        print(
            "[LEARN_LOCATIONS] Deferred unified goal associations finalized: "
            f"{finalized_count} new bindings, ready={ready}"
        )
        return ready

    def _get_unified_pc_field_peak_data(self, target_device):
        """
        Compute dense-hmap field peaks and a compactness mask for unified PCs.

        Compactness is measured as the maximum spatial radius of the support
        above a fixed fraction of the cell's own peak activation. Cells with
        broad or split high-activation support are excluded from reward-map
        seeding and propagation because a single reward weight on such a cell
        creates detached spatial peaks that no later budget/smoothing step can
        separate.
        """
        if not getattr(self, "use_unified_multiscale", False):
            return None, None, None, None, "field_peaks=no_unified"
        source = self._get_unified_dense_hmap_source()
        if source is None:
            return None, None, None, None, "field_peaks=no_dense_hmaps"

        n_valid = int(source["n_valid"])
        peak_fraction = float(
            getattr(self, "goal_map_pc_compactness_peak_fraction", 0.50)
        )
        min_peak = float(getattr(self, "goal_map_pc_compactness_min_peak", 0.05))
        max_radius = float(getattr(self, "goal_map_pc_compactness_max_radius", 4.0))
        cache = getattr(self, "_unified_pc_field_peak_cache", None)
        if (
            cache is not None
            and cache.get("n_valid") == n_valid
            and cache.get("source") == source.get("source")
            and abs(float(cache.get("peak_fraction", -1.0)) - peak_fraction) <= 1e-9
            and abs(float(cache.get("min_peak", -1.0)) - min_peak) <= 1e-9
            and abs(float(cache.get("max_radius", -1.0)) - max_radius) <= 1e-9
        ):
            return (
                cache["peak_xy"].to(target_device),
                cache["peak_val"].to(target_device),
                cache["compact_mask"].to(target_device),
                cache["compact_radius"].to(target_device),
                cache["log"],
            )

        num_pc_total = int(self.unified_pcn.num_pc_total)
        peak_xy = np.zeros((num_pc_total, 2), dtype=np.float32)
        peak_val = np.zeros(num_pc_total, dtype=np.float32)
        compact_radius = np.full(num_pc_total, np.inf, dtype=np.float32)
        compact_mask = np.zeros(num_pc_total, dtype=bool)

        coords = np.asarray(source["hmap_loc"][:n_valid, :2], dtype=np.float32)
        scale_acts = source["scale_acts"]
        for scale_idx in range(len(self.scales)):
            if scale_idx >= len(scale_acts):
                continue
            acts = scale_acts[scale_idx][:n_valid]
            if hasattr(acts, "detach"):
                acts = acts.detach().cpu().numpy()
            else:
                acts = np.asarray(acts)
            if acts.ndim != 2 or acts.shape[0] <= 0:
                continue
            start = int(self.unified_pcn.scale_boundaries[scale_idx])
            end = int(self.unified_pcn.scale_boundaries[scale_idx + 1])
            peak_rows = np.argmax(acts, axis=0)
            peak_vals = np.max(acts, axis=0)
            peak_xy[start:end] = coords[peak_rows]
            peak_val[start:end] = peak_vals.astype(np.float32)
            for local_idx, pc_peak in enumerate(peak_vals.tolist()):
                global_idx = start + int(local_idx)
                pc_peak = float(pc_peak)
                if pc_peak <= min_peak:
                    continue
                support = acts[:, local_idx] >= (peak_fraction * pc_peak)
                if not np.any(support):
                    continue
                d = np.linalg.norm(
                    coords[support] - peak_xy[global_idx][None, :],
                    axis=1,
                )
                radius = float(np.max(d)) if d.size else 0.0
                compact_radius[global_idx] = radius
                if radius <= max_radius:
                    compact_mask[global_idx] = True

        compact_count = int(compact_mask.sum())
        active_count = int(np.count_nonzero(peak_val > min_peak))
        log = (
            "field_peaks("
            f"active={active_count},compact={compact_count},"
            f"peak_frac={peak_fraction:.2f},max_r={max_radius:.2f})"
        )
        self._unified_pc_field_peak_cache = {
            "n_valid": n_valid,
            "source": source.get("source"),
            "peak_fraction": peak_fraction,
            "min_peak": min_peak,
            "max_radius": max_radius,
            "peak_xy": torch.from_numpy(peak_xy),
            "peak_val": torch.from_numpy(peak_val),
            "compact_mask": torch.from_numpy(compact_mask),
            "compact_radius": torch.from_numpy(compact_radius),
            "log": log,
        }
        return (
            self._unified_pc_field_peak_cache["peak_xy"].to(target_device),
            self._unified_pc_field_peak_cache["peak_val"].to(target_device),
            self._unified_pc_field_peak_cache["compact_mask"].to(target_device),
            self._unified_pc_field_peak_cache["compact_radius"].to(target_device),
            log,
        )

    def _get_unified_scale_pc_peak_stats(
        self,
        scale_idx: int,
        candidate_indices: List[int],
        target_device=None,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Return dense-hmap field-peak `(x, z, act)` for selected local PC indices.

        This uses the actual dense activation history rather than center-of-mass
        estimates, so aliased PCs with a strong remote lobe can be rejected at
        the association stage before they contaminate reward replay.
        """
        if not getattr(self, "use_unified_multiscale", False) or not candidate_indices:
            return {}
        peak_xy, peak_val, _, _, _ = self._get_unified_pc_field_peak_data(
            target_device=target_device if target_device is not None else self.device
        )
        if peak_xy is None or peak_val is None:
            return {}
        scale_boundaries = getattr(self.unified_pcn, "scale_boundaries", None)
        if scale_boundaries is None or scale_idx < 0 or scale_idx + 1 >= len(scale_boundaries):
            return {}
        start = int(scale_boundaries[scale_idx])
        end = int(scale_boundaries[scale_idx + 1])
        valid_indices = sorted(
            {
                int(idx)
                for idx in candidate_indices
                if idx is not None and 0 <= int(idx) < (end - start)
            }
        )
        if not valid_indices:
            return {}

        peak_stats: Dict[int, Tuple[float, float, float]] = {}
        for local_idx in valid_indices:
            global_idx = start + int(local_idx)
            peak_pt = peak_xy[global_idx].detach().cpu().tolist()
            peak_value = float(peak_val[global_idx].item())
            peak_stats[int(local_idx)] = (
                float(peak_pt[0]),
                float(peak_pt[1]),
                peak_value,
            )
        return peak_stats

    def _select_goal_association_pc(
        self,
        scale_acts: torch.Tensor,
        goal_name: str,
        scale_idx: int,
    ) -> Optional[int]:
        """
        Select a goal association PC using uniqueness and source locality.

        In unified mode, dense hmaps let us reject aliased PCs whose field peaks
        are far from the actual goal. If no sufficiently local candidate exists,
        we skip this scale instead of storing a bad association that later
        creates detached reward peaks.
        """
        if scale_acts.numel() == 0:
            return None
        top_k = int(min(self.goal_assoc_unique_topk, int(scale_acts.numel())))
        vals, idx = torch.topk(scale_acts, k=top_k)
        idx = idx.detach().cpu().tolist()
        vals = vals.detach().cpu().tolist()

        used = self._get_used_goal_pcs(scale_idx=scale_idx, exclude_goal=goal_name)
        top_val = float(vals[0]) if vals else 0.0
        min_allowed = top_val * (1.0 - self.goal_assoc_max_activation_drop)
        goal_loc = self._get_goal_location_by_name(goal_name)
        locality_radius = float(getattr(self, "goal_assoc_locality_radius", 0.0))

        def _pick_best_local(candidates: List[Tuple[int, float, float]]) -> Optional[int]:
            if not candidates:
                return None
            candidates.sort(
                key=lambda item: (
                    int(item[0] in used),
                    -float(item[1]),
                    float(item[2]),
                    int(item[0]),
                )
            )
            return int(candidates[0][0])

        if (
            getattr(self, "use_unified_multiscale", False)
            and goal_loc is not None
            and locality_radius > 0.0
        ):
            peak_stats = self._get_unified_scale_pc_peak_stats(
                scale_idx=scale_idx,
                candidate_indices=[int(cand_idx) for cand_idx in idx],
                target_device=scale_acts.device,
            )
            scale_boundaries = getattr(self.unified_pcn, "scale_boundaries", None)
            compact_mask = None
            if (
                bool(getattr(self, "goal_map_use_compact_pc_gate", True))
                and scale_boundaries is not None
                and scale_idx + 1 < len(scale_boundaries)
            ):
                _, _, compact_mask_all, _, _ = self._get_unified_pc_field_peak_data(
                    target_device=scale_acts.device
                )
                if compact_mask_all is not None:
                    start = int(scale_boundaries[scale_idx])
                    end = int(scale_boundaries[scale_idx + 1])
                    compact_mask = compact_mask_all[start:end]
            if peak_stats:
                relaxed_drop = max(
                    float(self.goal_assoc_max_activation_drop),
                    float(getattr(self, "goal_assoc_locality_max_activation_drop", 0.0)),
                )
                relaxed_min_allowed = top_val * (1.0 - relaxed_drop)
                local_strict: List[Tuple[int, float, float]] = []
                local_relaxed: List[Tuple[int, float, float]] = []
                for cand_idx, cand_val in zip(idx, vals):
                    peak_info = peak_stats.get(int(cand_idx))
                    if peak_info is None:
                        continue
                    if compact_mask is not None and not bool(
                        compact_mask[int(cand_idx)].item()
                    ):
                        continue
                    peak_x, peak_z, _ = peak_info
                    peak_dist = math.hypot(
                        float(peak_x) - float(goal_loc[0]),
                        float(peak_z) - float(goal_loc[1]),
                    )
                    if peak_dist > locality_radius:
                        continue
                    cand_tuple = (int(cand_idx), float(cand_val), float(peak_dist))
                    if float(cand_val) >= min_allowed:
                        local_strict.append(cand_tuple)
                    if float(cand_val) >= relaxed_min_allowed:
                        local_relaxed.append(cand_tuple)
                selected_local = _pick_best_local(local_strict)
                if selected_local is not None:
                    return selected_local
                selected_local = _pick_best_local(local_relaxed)
                if selected_local is not None:
                    return selected_local
                return None

        for cand_idx, cand_val in zip(idx, vals):
            if float(cand_val) < min_allowed:
                continue
            if int(cand_idx) not in used:
                return int(cand_idx)
        return int(idx[0]) if idx else None

    ##########################################################################
    #                        GOAL / TRIAL / COVERAGE SETUP                   #
    ##########################################################################

    def _setup_goals(self, goal_config, goal_location):
        """Setup unified goal system supporting both single and multi-goal modes"""
        self.goals = []
        self.multi_goal_mode = False

        if goal_config is None:
            # Legacy single-goal mode
            self.goals.append({
                "name": "default",
                "location": goal_location if goal_location else [-3, 3],
                "radius": self.goal_r["explore"],
                "visited": False,
                "active": True
            })
            self.goal_location = self.goals[0]["location"]
        elif goal_config["type"] == "single":
            # Single goal configuration
            self.goals.append({
                "name": goal_config.get("name", "default"),
                "location": goal_config["location"],
                "radius": goal_config.get("radius", self.goal_r["explore"]),
                "visited": False,
                "active": True
            })
            self.goal_location = self.goals[0]["location"]
        elif goal_config["type"] == "multi":
            # Multi-goal configuration
            for goal in goal_config["goals"]:
                self.goals.append({
                    "name": goal["name"],
                    "location": goal["location"],
                    "radius": goal["radius"],
                    "visited": False,
                    "active": goal_config.get("target_goal") == goal["name"] if "target_goal" in goal_config else False
                })

            # Set target goal if specified (for EXPLOIT_LOCATIONS modes)
            if "target_goal" in goal_config:
                self.active_goal_name = goal_config["target_goal"]

            self.multi_goal_mode = True

            # Initialize goal tracking for learning modes
            if self.robot_mode in {
                RobotMode.LEARN_LOCATIONS_COVERAGE,
                RobotMode.LEARN_LOCATIONS_ADAPTIVE,
                RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                RobotMode.LEARN_LOCATIONS_TWO_PHASE,
                RobotMode.LEARN_HEBB,
            }:
                self.goal_place_cell_associations = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_association_step = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_place_cell_activations = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_contact_steps = {
                    goal["name"]: [] for goal in self.goals
                }
                self.goal_visit_counts = {
                    goal["name"]: 0 for goal in self.goals
                }
                self.goal_currently_in = {
                    goal["name"]: False for goal in self.goals
                }
                self.goal_last_count_step = {
                    goal["name"]: -10**9 for goal in self.goals
                }

                # Threshold for activation similarity (within 20% = use connections as tiebreaker)
                self.ACTIVATION_SIMILARITY_THRESHOLD = 0.20
                self.goal_associations_finalized = False
                print(
                    f"[DRIVER] Goal visit debounce enabled: cooldown={self.goal_visit_cooldown_seconds:.1f}s "
                    f"({self.goal_visit_cooldown_steps} steps), exit_hysteresis={self.goal_exit_hysteresis:.2f}m"
                )
                if self.use_unified_multiscale and self.defer_goal_association_until_training_end:
                    print(
                        "[DRIVER] Unified goal association finalization deferred to training end."
                    )

            # Initialize coverage tracking for modes that use it
            if self.robot_mode in {
                RobotMode.LEARN_LOCATIONS_COVERAGE,
                RobotMode.LEARN_LOCATIONS_ADAPTIVE,
                RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                RobotMode.LEARN_LOCATIONS_TWO_PHASE,
                RobotMode.LEARN_HEBB,
                RobotMode.PLOTTING_COVERAGE_AUTO,
            }:
                if self.environment_size and self.grid_size and self.coverage_percentage:
                    self._setup_coverage_tracking(self.environment_size, self.grid_size, self.coverage_percentage)

            print(f"[DRIVER] Multi-goal mode with {len(self.goals)} goals:")
            for goal in self.goals:
                status = "ACTIVE" if goal.get("active", False) else "inactive"
                print(f"  - {goal['name']}: {goal['location']} (radius: {goal['radius']}) [{status}]")

    def _setup_trials(self, trial_config):
        """Store trial configuration for reference"""
        if not trial_config:
            trial_config = {"type": "simple", "count": 1, "start_locations": [[0, 0]]}

        self.trial_config = trial_config

        # Handle different trial count keys
        trial_count = trial_config.get('count') or trial_config.get('trials_per_goal', 1)
        print(f"[DRIVER] Trial config: {trial_config['type']} - {trial_count} trials")

    def _setup_coverage_tracking(self, environment_size, grid_size, coverage_percentage):
        """Initialize coverage tracking for LEARN_LOCATIONS_COVERAGE mode"""
        if environment_size is None or grid_size is None or coverage_percentage is None:
            raise ValueError("Coverage parameters must be provided for LEARN_LOCATIONS_COVERAGE mode")

        self.environment_size = environment_size  # [width, height] in meters
        self.grid_size = grid_size  # Grid cell size in meters
        self.target_coverage_percentage = coverage_percentage  # Target coverage (0.0-1.0)

        # Calculate grid dimensions
        self.grid_width = int(environment_size[0] / grid_size)
        self.grid_height = int(environment_size[1] / grid_size)
        self.total_grid_cells = self.grid_width * self.grid_height

        # Initialize coverage grid (False = unvisited, True = visited)
        self.coverage_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        # Coverage tracking variables
        self.visited_cells = 0
        self.current_coverage_percentage = 0.0

        print(f"[COVERAGE] Environment: {environment_size[0]}x{environment_size[1]}m")
        print(f"[COVERAGE] Grid size: {grid_size}m")
        print(f"[COVERAGE] Grid dimensions: {self.grid_width}x{self.grid_height} ({self.total_grid_cells} total cells)")
        print(f"[COVERAGE] Target coverage: {coverage_percentage*100:.1f}%")

    def _update_coverage(self, robot_position):
        """Update coverage grid based on robot's current position"""
        # Convert world coordinates to grid coordinates
        # Assuming world center is at (0,0) and extends from -size/2 to +size/2
        world_x, world_z = robot_position[0], robot_position[1]

        # Convert to grid coordinates (0 to grid_width/height-1)
        grid_x = int((world_x + self.environment_size[0]/2) / self.grid_size)
        grid_z = int((world_z + self.environment_size[1]/2) / self.grid_size)

        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_z = max(0, min(self.grid_height - 1, grid_z))

        # Mark cell as visited if not already visited
        if not self.coverage_grid[grid_z][grid_x]:
            self.coverage_grid[grid_z][grid_x] = True
            self.visited_cells += 1
            self.current_coverage_percentage = self.visited_cells / self.total_grid_cells

            # Optional: Print coverage updates at intervals
            if self.visited_cells % 100 == 0:
                print(f"[COVERAGE] Visited {self.visited_cells}/{self.total_grid_cells} cells ({self.current_coverage_percentage*100:.1f}%)")

    def _check_coverage_complete(self):
        """Check if target coverage percentage has been reached"""
        return self.current_coverage_percentage >= self.target_coverage_percentage

    def _get_learning_flags(self):
        """Return whether Oja/STDP are enabled in current architecture."""
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            return bool(getattr(self.unified_pcn, "enable_ojas", False)), bool(getattr(self.unified_pcn, "enable_stdp", False))
        if self.pcns:
            pcn0 = self.pcns[0]
            return bool(getattr(pcn0, "enable_ojas", False)), bool(getattr(pcn0, "enable_stdp", False))
        return False, False

    def _set_learning_flags(self, enable_ojas: bool, enable_stdp: bool):
        """Set Oja/STDP flags on the active PCN architecture."""
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            self.unified_pcn.enable_ojas = bool(enable_ojas)
            self.unified_pcn.enable_stdp = bool(enable_stdp)
            if self.unified_pcn.enable_stdp:
                if getattr(self.unified_pcn, "place_cell_trace", None) is None:
                    self.unified_pcn.place_cell_trace = torch.zeros(
                        self.unified_pcn.num_pc_total,
                        dtype=self.unified_pcn.dtype,
                        device=self.unified_pcn.device,
                    )
                if getattr(self.unified_pcn, "hd_cell_trace", None) is None:
                    self.unified_pcn.hd_cell_trace = torch.zeros(
                        (self.n_hd, 1, 1),
                        dtype=self.unified_pcn.dtype,
                        device=self.unified_pcn.device,
                    )
            else:
                self.unified_pcn.place_cell_trace = None
                self.unified_pcn.hd_cell_trace = None
            return
        for pcn in self.pcns:
            pcn.enable_ojas = bool(enable_ojas)
            pcn.enable_stdp = bool(enable_stdp)

    def _stability_tracking_active(self) -> bool:
        """
        Return whether revisit-consistency stability tracking should run.
        """
        if self.two_phase_learning and self.phase2_start_step is None:
            return True
        if (
            self.robot_mode == RobotMode.LEARN_LOCATIONS_ADAPTIVE
            and self.pcn_learning_stop_ojas_on_stabilization
            and not self.ojas_stopped_after_stabilization
        ):
            ojas_enabled, _ = self._get_learning_flags()
            return bool(ojas_enabled)
        return False

    def _update_phase1_stability_metrics(self):
        """
        Track place field stability via revisit consistency.

        On each step, the robot's position is binned into a spatial grid.
        When a bin is revisited, the cosine similarity between the current
        activation vector and the activation stored from the previous visit
        to that bin is recorded. This measures whether the same location
        reliably produces the same place cell pattern — the biologically
        meaningful definition of a stable place code.
        """
        if not self._stability_tracking_active():
            return
        if not self.pcn_activations_list:
            return

        # Get current activation vector.
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            vec = self.unified_pcn.place_cell_activations.detach()
        else:
            vec = torch.cat([a.detach() for a in self.pcn_activations_list if a is not None], dim=0)

        if vec.numel() == 0:
            return

        # Get current spatial bin.
        raw_pos = self.robot.getField("translation").getSFVec3f()
        planar = self._translation_to_planar_xy(raw_pos)
        bx = int(planar[0] / self.phase1_bin_size)
        by = int(planar[1] / self.phase1_bin_size)
        bin_key = (bx, by)

        if bin_key in self._phase1_bin_activations:
            # Revisit: compare current activation to the stored one.
            stored = self._phase1_bin_activations[bin_key]
            if stored.numel() == vec.numel():
                denom = (torch.norm(vec) * torch.norm(stored)).item()
                if denom > 1e-12:
                    cos = float(torch.dot(vec, stored).item() / denom)
                    self._phase1_revisit_cosines.append(cos)
                    self._phase1_bins_with_revisits.add(bin_key)
                    self._phase1_bin_last_cosines[bin_key] = cos

        # Update stored activation for this bin (tracks latest visit).
        self._phase1_bin_activations[bin_key] = vec.clone()

    def _get_phase1_stability_status(self) -> Dict[str, float]:
        """Summarize current phase-1 drift metrics across revisited bins."""
        explored_bins = int(len(self._phase1_bin_activations))
        revisit_bins = int(len(self._phase1_bins_with_revisits))
        revisit_fraction = (
            float(revisit_bins) / float(max(1, explored_bins))
            if explored_bins > 0
            else 0.0
        )
        sample_count = int(len(self._phase1_revisit_cosines))
        recent_cos_mean = (
            float(np.mean(self._phase1_revisit_cosines))
            if self._phase1_revisit_cosines
            else 0.0
        )
        latest_bin_cosines = np.asarray(
            list(self._phase1_bin_last_cosines.values()), dtype=np.float32
        )
        bin_cos_mean = (
            float(np.mean(latest_bin_cosines))
            if latest_bin_cosines.size > 0
            else 0.0
        )
        bin_cos_low = (
            float(np.percentile(latest_bin_cosines, self.phase1_revisit_low_percentile))
            if latest_bin_cosines.size > 0
            else 0.0
        )
        enough_bins = revisit_bins >= self.phase1_min_revisit_bins
        enough_samples = sample_count >= max(10, self.phase1_min_revisit_bins)
        enough_fraction = revisit_fraction >= self.phase1_revisit_fraction_threshold
        enough_mean = bin_cos_mean >= self.phase1_revisit_cosine_threshold
        enough_low = bin_cos_low >= self.phase1_revisit_low_percentile_threshold
        stable_now = bool(
            enough_bins
            and enough_samples
            and enough_fraction
            and enough_mean
            and enough_low
        )
        return {
            "explored_bins": explored_bins,
            "revisit_bins": revisit_bins,
            "revisit_fraction": revisit_fraction,
            "sample_count": sample_count,
            "recent_cos_mean": recent_cos_mean,
            "bin_cos_mean": bin_cos_mean,
            "bin_cos_low": bin_cos_low,
            "stable_now": stable_now,
        }

    def _maybe_stop_ojas_on_stabilization(self):
        """
        In adaptive mode, switch from OJAS-only field formation to STDP-only
        recurrent learning once the place code is stable.
        """
        if (
            self.robot_mode != RobotMode.LEARN_LOCATIONS_ADAPTIVE
            or not self.pcn_learning_stop_ojas_on_stabilization
            or self.ojas_stopped_after_stabilization
        ):
            return

        ojas_enabled, stdp_enabled = self._get_learning_flags()
        if not ojas_enabled:
            return

        if not self._phase1_is_stable():
            return

        status = self._phase1_last_stability_status or self._get_phase1_stability_status()
        self._set_learning_flags(enable_ojas=False, enable_stdp=True)
        self.ojas_stopped_after_stabilization = True
        self.ojas_stop_step = int(self.step_count)
        self.adaptive_stdp_start_step = int(self.step_count)
        self.two_phase_phase = "adaptive_stdp"
        self._reset_learning_targets_for_adaptive_stdp_start()
        if hasattr(self, "unified_rcn") and hasattr(self.unified_rcn, "reset_experience_transitions"):
            self.unified_rcn.reset_experience_transitions()
        self.record_experience_transitions = True
        print(
            f"[ADAPTIVE][PHASE SWITCH] step={self.ojas_stop_step} "
            f"revisit_bins={int(status['revisit_bins'])}, "
            f"revisit_frac={float(status['revisit_fraction']):.2f}, "
            f"bin_cos={float(status['bin_cos_mean']):.4f}, "
            f"p{int(self.phase1_revisit_low_percentile)}="
            f"{float(status['bin_cos_low']):.4f} "
            f"| OJAS disabled, STDP enabled, counters/history reset."
        )

    def _two_phase_allows_counting(self) -> bool:
        """
        Return whether goal/checkpoint counting should be updated in current step.
        In two-phase coverage mode, path-related counting is phase-2 only.
        """
        if not self.two_phase_learning:
            return True
        return self.phase2_start_step is not None

    def _two_phase_allows_coverage_tracking(self) -> bool:
        """
        Return whether environment coverage should be tracked in current step.

        Gate-only coverage mode keeps coverage accumulation active in phase 1.
        Strict LEARN_LOCATIONS_TWO_PHASE keeps coverage gated until phase 2.
        """
        if not self.two_phase_learning:
            return True
        if self.two_phase_split_learning:
            return self.phase2_start_step is not None
        return True

    def _phase1_is_stable(self) -> bool:
        """
        Return True when place field coding is stable by revisit consistency.

        Stability requires:
        1. Minimum step count reached (safety floor).
        2. Enough distinct bins have been revisited at least once.
        3. A substantial fraction of explored bins have revisits.
        4. Per-bin revisit cosine stays high on average and in the lower tail.
        5. The condition holds for a short sustained streak, not one momentary spike.

        Fallback: if step_count >= phase1_max_steps, force-transition regardless
        of stability so goal counting is never permanently blocked.
        """
        if self.step_count < self.phase1_min_steps:
            self._phase1_stability_streak = 0
            return False
        status = self._get_phase1_stability_status()
        self._phase1_last_stability_status = status
        # Hard cap: force transition if phase-1 runs too long.
        if self.step_count >= self.phase1_max_steps:
            if self.two_phase_learning and self.phase2_start_step is None:
                label = "[TRAIN-2P][PHASE1]"
                action = "force-transitioning to phase 2"
            else:
                label = "[ADAPTIVE][STABILITY]"
                action = "treating coding as stabilized"
            print(
                f"{label} Max steps ({self.phase1_max_steps}) reached - "
                f"{action}. "
                f"revisit_bins={int(status['revisit_bins'])}, "
                f"revisit_frac={float(status['revisit_fraction']):.2f}, "
                f"bin_cos={float(status['bin_cos_mean']):.4f}, "
                f"p{int(self.phase1_revisit_low_percentile)}="
                f"{float(status['bin_cos_low']):.4f}"
            )
            return True
        if self._phase1_last_stability_eval_step != int(self.step_count):
            if bool(status["stable_now"]):
                self._phase1_stability_streak += 1
            else:
                self._phase1_stability_streak = 0
            self._phase1_last_stability_eval_step = int(self.step_count)
        return self._phase1_stability_streak >= self.phase1_stability_required_streak

    def _reset_learning_targets_for_phase2(self, reset_coverage: bool = True):
        """Reset phase-2 learning targets at the stability handoff."""
        if reset_coverage and hasattr(self, "coverage_grid"):
            self.coverage_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.visited_cells = 0
            self.current_coverage_percentage = 0.0
        self._reset_path_learning_targets()

    def _reset_path_learning_targets(self):
        """Reset goal/checkpoint counters and associations without touching coverage."""
        if hasattr(self, "goals"):
            for goal in self.goals:
                goal["visited"] = False
        if hasattr(self, "goal_visit_counts"):
            for k in self.goal_visit_counts.keys():
                self.goal_visit_counts[k] = 0
        if hasattr(self, "goal_currently_in"):
            for k in self.goal_currently_in.keys():
                self.goal_currently_in[k] = False
        if hasattr(self, "goal_last_count_step"):
            for k in self.goal_last_count_step.keys():
                self.goal_last_count_step[k] = -10**9
        if hasattr(self, "checkpoint_visit_counts"):
            for k in self.checkpoint_visit_counts.keys():
                self.checkpoint_visit_counts[k] = 0
        if hasattr(self, "checkpoint_crossing_counts"):
            for k in self.checkpoint_crossing_counts.keys():
                self.checkpoint_crossing_counts[k] = {"neg_to_pos": 0, "pos_to_neg": 0}
        if hasattr(self, "checkpoint_crossing_steps"):
            for k in self.checkpoint_crossing_steps.keys():
                self.checkpoint_crossing_steps[k] = {
                    "neg_to_pos": [],
                    "pos_to_neg": [],
                }
        if hasattr(self, "checkpoint_currently_in"):
            for k in self.checkpoint_currently_in.keys():
                self.checkpoint_currently_in[k] = False
        if hasattr(self, "checkpoint_last_count_step"):
            for k in self.checkpoint_last_count_step.keys():
                self.checkpoint_last_count_step[k] = -10**9
        if hasattr(self, "checkpoint_entry_side"):
            for k in self.checkpoint_entry_side.keys():
                self.checkpoint_entry_side[k] = 0
        if hasattr(self, "checkpoint_last_clear_side"):
            for k in self.checkpoint_last_clear_side.keys():
                self.checkpoint_last_clear_side[k] = 0
        if hasattr(self, "checkpoint_crossed_centerline"):
            for k in self.checkpoint_crossed_centerline.keys():
                self.checkpoint_crossed_centerline[k] = False
        if hasattr(self, "checkpoint_crossing_step_candidate"):
            for k in self.checkpoint_crossing_step_candidate.keys():
                self.checkpoint_crossing_step_candidate[k] = -1
        if hasattr(self, "checkpoint_last_crossed"):
            self.checkpoint_last_crossed = None
        if hasattr(self, "checkpoint_pc_groups"):
            for k in self.checkpoint_pc_groups.keys():
                for side in self.checkpoint_pc_groups[k]:
                    self.checkpoint_pc_groups[k][side].zero_()
        if hasattr(self, "checkpoint_proximity_pcs"):
            for k in self.checkpoint_proximity_pcs.keys():
                self.checkpoint_proximity_pcs[k].zero_()
        if hasattr(self, "goal_place_cell_associations"):
            for goal_name in self.goal_place_cell_associations.keys():
                self.goal_place_cell_associations[goal_name] = [None] * len(self.scales)
                self.goal_association_step[goal_name] = [None] * len(self.scales)
                self.goal_place_cell_activations[goal_name] = [None] * len(self.scales)
                if hasattr(self, "goal_contact_steps"):
                    self.goal_contact_steps[goal_name] = []
        if hasattr(self, "goal_associations_finalized"):
            self.goal_associations_finalized = False

    def _reset_learning_targets_for_adaptive_stdp_start(self):
        """Reset only path-learning state when adaptive mode hands off from OJAS to STDP."""
        self._reset_path_learning_targets()

    def _get_reward_history_start_step(self) -> int:
        """Return the first step whose history should contribute to reward-map replay graphs."""
        if self.two_phase_learning and self.phase2_start_step is not None:
            return int(max(0, self.phase2_start_step))
        if (
            self.robot_mode == RobotMode.LEARN_LOCATIONS_ADAPTIVE
            and self.adaptive_stdp_start_step is not None
        ):
            return int(max(0, self.adaptive_stdp_start_step))
        return 0

    def _start_phase2_stdp(self):
        """Open phase 2 after stability; strict mode also switches OJAS -> STDP."""
        if not (self.two_phase_learning and self.phase2_start_step is None):
            return
        status = self._phase1_last_stability_status or self._get_phase1_stability_status()
        self.two_phase_phase = (
            "phase2_stdp" if self.two_phase_split_learning else "phase2_counting"
        )
        self.phase2_start_step = int(self.step_count)
        if self.two_phase_split_learning:
            self._set_learning_flags(enable_ojas=False, enable_stdp=True)
        self._reset_learning_targets_for_phase2(
            reset_coverage=bool(self.two_phase_split_learning)
        )
        if hasattr(self, "unified_rcn") and hasattr(self.unified_rcn, "reset_experience_transitions"):
            self.unified_rcn.reset_experience_transitions()
        # Phase-2 transition handling:
        # - deferred mode: build once from history at the end (no per-step overhead)
        # - online mode: collect transitions each step during phase-2
        self.record_experience_transitions = not self.defer_experience_build_until_phase2_end
        switch_mode = (
            "PHASE 1 -> PHASE 2 (OJAS -> STDP)"
            if self.two_phase_split_learning
            else "PHASE 1 -> PHASE 2 (goal/checkpoint counting enabled)"
        )
        print("=" * 80)
        print(
            f"[TRAIN-2P][PHASE SWITCH] {switch_mode} at step={self.phase2_start_step} "
            f"(revisit_bins={int(status['revisit_bins'])}, "
            f"revisit_frac={float(status['revisit_fraction']):.2f}, "
            f"bin_cos={float(status['bin_cos_mean']):.4f}, "
            f"p{int(self.phase1_revisit_low_percentile)}="
            f"{float(status['bin_cos_low']):.4f}, "
            f"streak={self._phase1_stability_streak}/{self.phase1_stability_required_streak})."
        )
        if self.two_phase_split_learning:
            print(
                "[TRAIN-2P][PHASE SWITCH] Phase 2 uses STDP with coverage/goal counting enabled; "
                "coverage and path counters were reset at this transition."
            )
        else:
            print(
                "[TRAIN-2P][PHASE SWITCH] Learning flags were left unchanged; "
                "goal/checkpoint counting and reward seeding now start from this step."
            )
        print("=" * 80)

    def _build_experience_transitions_from_hmaps(self):
        """
        Deferred transition build: convert recorded phase-2 PC activity history into
        experienced transition counts in unified RCN.
        """
        if not self.use_unified_multiscale or not hasattr(self, "unified_rcn"):
            return
        if not hasattr(self.unified_rcn, "observe_transition"):
            return
        if self.lightweight_hmaps:
            print("[TRAIN-2P] WARNING: lightweight_hmaps=True; cannot rebuild dense transitions from history.")
            return
        if self.phase2_start_step is None:
            return
        if not self.hmap_pcn_activities:
            return

        start = int(max(0, self.phase2_start_step))
        end = int(min(self.step_count, min(h.shape[0] for h in self.hmap_pcn_activities)))
        counts, build_log = self._get_cached_unified_denoised_transition_counts(
            start_step=start,
            end_step=end,
        )
        if counts is None:
            return
        if hasattr(self.unified_rcn, "reset_experience_transitions"):
            self.unified_rcn.reset_experience_transitions()
        self.unified_rcn.experience_transition_counts = counts
        print(
            f"[TRAIN-2P] Built deferred experience transitions from history: "
            f"{build_log}"
        )

    def _extract_compact_unified_history(
        self,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        """Extract a compact denoised dominant-state trajectory from dense PC history."""
        if not self.use_unified_multiscale or not hasattr(self, "unified_rcn"):
            return None, "history=disabled"
        if not hasattr(self.unified_rcn, "observe_transition"):
            return None, "history=no_observer"
        if self.lightweight_hmaps:
            return None, "history=no_dense_hmaps"
        if not self.hmap_pcn_activities:
            return None, "history=no_hmaps"

        max_hmap_steps = min(h.shape[0] for h in self.hmap_pcn_activities)
        start = int(max(0, start_step))
        end = int(
            min(
                max_hmap_steps,
                self.step_count if end_step is None else end_step,
            )
        )
        if end - start < 2:
            return None, f"history=too_short(steps={max(0, end-start)})"

        scale_arrays = [h.detach().cpu() for h in self.hmap_pcn_activities]
        time_indices = list(range(start, end))
        num_dominant_valid = 0
        dominant_change_times = []
        dominant_change_ids = []

        prune_mode = "disabled"
        if self.prune_experience_loops:
            dominant_pairs = []
            for t in time_indices:
                vec_t = torch.cat([arr[t] for arr in scale_arrays], dim=0)
                if vec_t.numel() < 2:
                    continue
                top_vals, top_idx = torch.topk(vec_t, k=2)
                top1 = float(top_vals[0].item())
                top2 = float(max(1e-12, top_vals[1].item()))
                ratio = top1 / top2
                if (
                    top1 >= self.loop_prune_min_top1
                    and ratio >= self.loop_prune_min_top1_to_top2_ratio
                ):
                    dominant_pairs.append((t, int(top_idx[0].item())))

            num_dominant_valid = len(dominant_pairs)
            if dominant_pairs:
                # 1) Collapse consecutive repeats of the same dominant state.
                for t, dom in dominant_pairs:
                    if dominant_change_ids and dom == dominant_change_ids[-1]:
                        dominant_change_times[-1] = t
                    else:
                        dominant_change_times.append(t)
                        dominant_change_ids.append(dom)

                prune_mode = "changes_only"

                # 2) Remove only short loops/backtracks in the dominant-state path.
                # Long revisitations are part of global navigation history and
                # should remain available to the goal-map replay graph.
                max_loop_nodes = 6
                pruned_time_indices = []
                pruned_dom_ids = []
                for t, dom in zip(dominant_change_times, dominant_change_ids):
                    loop_start = None
                    search_start = max(0, len(pruned_dom_ids) - max_loop_nodes)
                    for pos in range(len(pruned_dom_ids) - 1, search_start - 1, -1):
                        if pruned_dom_ids[pos] == dom:
                            loop_start = pos
                            break

                    if loop_start is not None:
                        pruned_time_indices = pruned_time_indices[: loop_start + 1]
                        pruned_dom_ids = pruned_dom_ids[: loop_start + 1]
                        # Keep the latest revisit timestamp for the retained node.
                        pruned_time_indices[-1] = t
                    else:
                        pruned_time_indices.append(t)
                        pruned_dom_ids.append(dom)

                if len(pruned_time_indices) >= 2:
                    min_compact_steps = min(
                        len(dominant_change_times),
                        max(8, int(math.ceil(0.05 * len(dominant_change_times)))),
                    )
                    if len(pruned_time_indices) >= min_compact_steps:
                        time_indices = pruned_time_indices
                        prune_mode = f"short_window({max_loop_nodes})"
                    else:
                        time_indices = dominant_change_times
                        prune_mode = (
                            f"fallback_changes(compact={len(pruned_time_indices)}"
                            f"<min={min_compact_steps})"
                        )
                elif len(dominant_change_times) >= 2:
                    time_indices = dominant_change_times
                    prune_mode = "fallback_changes(compact<2)"

        change_steps = (
            len(dominant_change_times)
            if self.prune_experience_loops and num_dominant_valid > 0
            else len(time_indices)
        )
        history = {
            "scale_arrays": scale_arrays,
            "time_indices": time_indices,
            "start": start,
            "end": end,
            "prune_mode": prune_mode,
            "num_dominant_valid": num_dominant_valid,
            "change_steps": change_steps,
        }
        log = (
            f"steps={end-start}, compact_steps={len(time_indices)}, change_steps={change_steps}, "
            f"loop_prune={self.prune_experience_loops}, prune_mode={prune_mode}, "
            f"valid_dom={num_dominant_valid}"
        )
        return history, log

    def _build_unified_transition_counts_from_time_indices(
        self,
        scale_arrays,
        time_indices: List[int],
    ):
        """Build unified experience-transition counts from a compact list of time indices."""
        counts = torch.zeros_like(self.unified_rcn.experience_transition_counts)
        raw_counts = getattr(self.unified_rcn, "experience_transition_counts", None)
        self.unified_rcn.experience_transition_counts = counts
        num_edges = 0
        try:
            unified_history = torch.cat(scale_arrays, dim=1)
            for i in range(len(time_indices) - 1):
                t_prev = time_indices[i]
                t_curr = time_indices[i + 1]
                prev = unified_history[t_prev]
                curr = unified_history[t_curr]
                self.unified_rcn.observe_transition(prev, curr)
                num_edges += 1
            counts = self.unified_rcn.experience_transition_counts.detach().clone()
        finally:
            self.unified_rcn.experience_transition_counts = raw_counts
        return counts, num_edges

    def _build_unified_transition_counts_from_edge_pairs(
        self,
        scale_arrays,
        edge_pairs: List[tuple[int, int]],
    ):
        """Build unified experience-transition counts from explicit time-step edge pairs."""
        counts = torch.zeros_like(self.unified_rcn.experience_transition_counts)
        raw_counts = getattr(self.unified_rcn, "experience_transition_counts", None)
        self.unified_rcn.experience_transition_counts = counts
        num_edges = 0
        try:
            unified_history = torch.cat(scale_arrays, dim=1)
            for t_prev, t_curr in edge_pairs:
                prev = unified_history[t_prev]
                curr = unified_history[t_curr]
                self.unified_rcn.observe_transition(prev, curr)
                num_edges += 1
            counts = self.unified_rcn.experience_transition_counts.detach().clone()
        finally:
            self.unified_rcn.experience_transition_counts = raw_counts
        return counts, num_edges

    def _get_cached_unified_denoised_transition_counts(
        self,
        start_step: int,
        end_step: int,
    ):
        """
        Return cached denoised unified transition counts for the current history.

        Goal-map construction can request the same denoised trajectory graph
        several times at training end. Rebuilding it per goal is unnecessary.
        """
        cache_key = (
            int(start_step),
            int(end_step),
            int(self.step_count),
            bool(self.prune_experience_loops),
            float(self.loop_prune_min_top1),
            float(self.loop_prune_min_top1_to_top2_ratio),
            int(getattr(self.unified_rcn, "experience_topk", 0)),
        )
        cache = getattr(self, "_cached_unified_denoised_transition_counts", None)
        if isinstance(cache, dict) and cache.get("key") == cache_key:
            return cache.get("counts"), cache.get("log")

        counts, build_log = self._build_unified_experience_transition_counts_from_history(
            start_step=start_step,
            end_step=end_step,
        )
        self._cached_unified_denoised_transition_counts = {
            "key": cache_key,
            "counts": counts,
            "log": build_log,
        }
        return counts, build_log

    def _build_unified_experience_transition_counts_from_history(
        self,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        """
        Rebuild a denoised unified experience graph from dense PC history.

        This keeps the raw online learning dynamics untouched, but lets goal-map
        replay use a compact trajectory graph with short random-walk loops and
        backtracking cycles removed.
        """
        history, history_log = self._extract_compact_unified_history(
            start_step=start_step,
            end_step=end_step,
        )
        if history is None:
            return None, history_log

        counts, num_edges = self._build_unified_transition_counts_from_time_indices(
            history["scale_arrays"],
            history["time_indices"],
        )
        build_log = (
            f"{history_log}, edges={num_edges}"
        )
        return counts, build_log

    def _maybe_log_training_progress(self):
        """Periodic verbose diagnostics for coverage-learning runs."""
        if self.robot_mode not in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_ADAPTIVE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            return
        if self.step_count <= 0:
            return
        if self.step_count % self.training_log_interval_steps != 0:
            return

        elapsed = self.getTime() - self.trial_start_time
        ojas_enabled, stdp_enabled = self._get_learning_flags()
        coverage_pct = self.current_coverage_percentage * 100 if hasattr(self, "current_coverage_percentage") else 0.0
        coverage_target = self.target_coverage_percentage * 100 if hasattr(self, "target_coverage_percentage") else 0.0

        visit_summary = ""
        if hasattr(self, "goal_visit_counts") and self.goal_visit_counts:
            parts = []
            for goal_name, cnt in self.goal_visit_counts.items():
                parts.append(f"{goal_name}:{cnt}/{self.min_goal_visits}")
            visit_summary = " | visits=" + ", ".join(parts)
        if hasattr(self, "checkpoint_crossing_counts") and self.checkpoint_crossing_counts:
            parts = []
            for checkpoint_idx, counts in self.checkpoint_crossing_counts.items():
                neg_to_pos = int(counts.get("neg_to_pos", 0))
                pos_to_neg = int(counts.get("pos_to_neg", 0))
                parts.append(
                    f"cp{checkpoint_idx}:-+={neg_to_pos}/{self.min_checkpoint_visits},"
                    f"+-={pos_to_neg}/{self.min_checkpoint_visits}"
                )
            visit_summary += (" | " if visit_summary else " | ") + "checkpoints=" + ", ".join(parts)

        activity_parts = []
        for idx, act in enumerate(self.pcn_activations_list):
            if act is None or act.numel() == 0:
                continue
            active_frac = float((act > 0.01).float().mean().item() * 100.0)
            peak = float(torch.max(act).item())
            activity_parts.append(f"s{idx}:act%={active_frac:.1f},peak={peak:.3f}")
        activity_summary = " | " + " ; ".join(activity_parts) if activity_parts else ""

        print(
            f"[TRAIN] step={self.step_count} t={elapsed:.1f}s "
            f"| coverage={coverage_pct:.1f}/{coverage_target:.1f}% "
            f"| OJAS={ojas_enabled} STDP={stdp_enabled}"
            f"| phase={self.two_phase_phase}"
            f"{visit_summary}{activity_summary}"
        )
        if self.robot_mode == RobotMode.LEARN_LOCATIONS_ADAPTIVE:
            status = self._phase1_last_stability_status or self._get_phase1_stability_status()
            print(
                f"[TRAIN][ADAPTIVE] stability_bins={int(status['revisit_bins'])}/"
                f"{self.phase1_min_revisit_bins} "
                f"| stability_frac={float(status['revisit_fraction']):.2f}/"
                f"{self.phase1_revisit_fraction_threshold:.2f} "
                f"| stability_cos={float(status['bin_cos_mean']):.4f}/"
                f"{self.phase1_revisit_cosine_threshold:.3f} "
                f"| p{int(self.phase1_revisit_low_percentile)}="
                f"{float(status['bin_cos_low']):.4f}/"
                f"{self.phase1_revisit_low_percentile_threshold:.3f} "
                f"| streak={self._phase1_stability_streak}/"
                f"{self.phase1_stability_required_streak} "
                f"| ojas_stopped={int(self.ojas_stopped_after_stabilization)}"
            )

    def _maybe_log_grid_diagnostics(self):
        """Periodic unified-grid diagnostics to verify GC contribution."""
        if not self.use_unified_multiscale or not hasattr(self, "unified_pcn"):
            return
        if self.step_count <= 0 or self.step_count % self.grid_diag_interval_steps != 0:
            return

        diag = getattr(self.unified_pcn, "last_grid_diagnostics", None)
        if not diag:
            return

        per_scale = diag.get("per_scale", [])
        per_scale_str = " ".join(
            [
                f"s{item['scale_idx']}:raw={item.get('raw_grid_share', item['grid_share']):.2f}/"
                f"eff={item['grid_share']:.2f}"
                f"(g={item.get('grid_gain', 1.0):.2f},b={item.get('bvc_gain', 1.0):.2f})"
                for item in per_scale
            ]
        )
        print(
            f"[GRID-DIAG] step={self.step_count} "
            f"| raw_share={diag.get('raw_grid_share', diag.get('grid_share', 0.0)):.3f} "
            f"| eff_share={diag.get('grid_share', 0.0):.3f} "
            f"| eff_mix={diag.get('effective_grid_influence', 0.0):.3f} "
            f"| learn={int(bool(diag.get('learning_active', False)))} "
            f"| gc_active={diag.get('gc_active_frac', 0.0):.3f} "
            f"| gc_mean={diag.get('gc_mean', 0.0):.3f} "
            f"| gc_max={diag.get('gc_max', 0.0):.3f} "
            f"| {per_scale_str}"
        )

    def _update_distance_tracking(self):
        """Update total distance traveled for random spawn mode"""
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            current_position_2d = self._current_planar_xy()

            if self.last_position is not None:
                # Calculate distance moved since last update
                dx = current_position_2d[0] - self.last_position[0]
                dz = current_position_2d[1] - self.last_position[1]
                distance_moved = math.sqrt(dx*dx + dz*dz)
                self.total_distance_traveled += distance_moved

            self.last_position = current_position_2d

    def _is_robot_tipped(self, min_upright_cos: float = 0.25) -> bool:
        """
        Detect whether the robot is tipped over.

        Uses orientation matrix when available; falls back to axis-angle rotation field.
        """
        # Primary guard: if vertical height deviates too much from ground, robot is likely tipped or airborne.
        try:
            curr_pos = self.robot.getField("translation").getSFVec3f()
            if abs(self._translation_height(curr_pos) - float(self.spawn_ground_y)) > 0.12:
                return True
        except Exception:
            pass

        try:
            orientation = self.robot.getOrientation()
            # Conservative tilt heuristic from rotation matrix. The Create body uses
            # its local +Z as the vertical axis; the world's vertical axis depends on
            # the world convention, so use the matching row rather than hard-coding
            # orientation[8].
            if orientation and len(orientation) == 9:
                up_alignment_idx = 3 * int(self.height_axis_idx) + 2
                return float(orientation[up_alignment_idx]) < min_upright_cos
        except Exception:
            pass

        try:
            rot = self.robot.getField("rotation").getSFRotation()
            return abs(float(rot[0])) > 0.3 or abs(float(rot[1])) > 0.3
        except Exception:
            return False

    def _recover_upright_pose(self, reason: str = "tilt detected"):
        """Reset robot pose to upright at current x-z position."""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        planar = self._translation_to_planar_xy(curr_pos)
        self.stop()
        self.robot.getField("translation").setSFVec3f(
            self._compose_translation(planar[0], planar[1])
        )
        self.robot.getField("rotation").setSFRotation(self.upright_rotation)
        self.robot.resetPhysics()
        print(f"[RECOVERY] Upright reset performed ({reason}) at x={planar[0]:.2f}, z={planar[1]:.2f}")

    def _ensure_upright(self):
        """Check and recover robot pose if it has tipped over."""
        if self._is_robot_tipped():
            self._recover_upright_pose(reason="robot tipped")


    ########################################### RUN LOOP ###########################################

    def run(self):
        print(f"[DRIVER] Starting robot in {self.robot_mode}")
        if self.multi_goal_mode:
            print(f"[DRIVER] Multi-goal mode with {len(self.goals)} goals")
            for goal in self.goals:
                status = "ACTIVE" if goal.get("active", False) else "inactive"
                print(f"[DRIVER] Goal: {goal['name']} at {goal['location']} [{status}]")
        else:
            print(f"[DRIVER] Single goal at {self.goal_location}")

        while not self.done:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode == RobotMode.REBUILD_REWARD_MAP:
                self.rebuild_reward_maps_from_saved_state()
            elif self.robot_mode in (RobotMode.LEARN_OJAS,
                                      RobotMode.LEARN_HEBB,
                                      RobotMode.DMTP,
                                      RobotMode.PLOTTING,
                                      RobotMode.PLOTTING_AUTO,
                                      RobotMode.PLOTTING_COVERAGE_AUTO,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE,
                                      RobotMode.LEARN_LOCATIONS_ADAPTIVE,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                                      RobotMode.LEARN_LOCATIONS_TWO_PHASE):
                self.explore()
            elif self.robot_mode in (RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO):
                self.exploit()
            else:
                print("Unknown state. Exiting...")
                break

    def _reset_unified_heading_commit_state(
        self,
        reason: str,
        clear_target_signature: bool = True,
    ) -> None:
        """Clear unified exploit heading persistence with one explicit reason."""
        self.last_unified_committed_hd_bin = None
        self._last_unified_commit_reset_reason = str(reason)
        if clear_target_signature:
            self._last_unified_commit_target_signature = None

    def _get_unified_heading_commit_target_signature(self) -> tuple:
        """Return the current exploit target identity for persistence resets."""
        return (
            str(self._get_active_goal_name_for_exploit() or ""),
            str(getattr(self, "loaded_goal_specific_rcn_goal", "") or ""),
            str(getattr(self, "_loaded_goal_map_path", "") or ""),
            float(getattr(self, "_loaded_goal_map_mtime", 0.0) or 0.0),
            str(getattr(self, "_loaded_goal_map_debug", "") or ""),
            getattr(self, "_last_unified_active_checkpoint_target", None),
        )

    def _sync_unified_heading_commit_target(self) -> Optional[str]:
        """
        Reset persistence when the active goal/checkpoint target identity changes.
        """
        target_signature = self._get_unified_heading_commit_target_signature()
        prev_signature = getattr(
            self,
            "_last_unified_commit_target_signature",
            None,
        )
        if prev_signature is None:
            self._last_unified_commit_target_signature = target_signature
            return None
        if target_signature != prev_signature:
            self._reset_unified_heading_commit_state(
                reason="target_changed",
                clear_target_signature=False,
            )
            self._last_unified_commit_target_signature = target_signature
            return "target_changed"
        return None

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        """
        Handles exploration for multi-scale usage, calling compute_pcn_activations().
        """
        self._reset_unified_heading_commit_state(reason="explore_entry")
        for _ in range(self.tau_w):
            # 1) Sense environment
            self.sense()

            # 4) compute pcn_activations => fill self.pcn_activations_list
            self.compute_pcn_activations()

            # 5) If DMTP or EXPOIT => reward updates
            if self.robot_mode == RobotMode.DMTP or self.robot_mode == RobotMode.EXPLOIT:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)
                    # rcn.td_update(pcn.place_cell_activations, next_reward=actual_reward)
                # # Turn towards heading 225°
                # desired_heading_deg = 90
                # # Compute the minimal angular difference (normalized to [-180, 180])
                # angle_to_turn_deg = desired_heading_deg - self.current_heading_deg
                # angle_to_turn_deg = ((angle_to_turn_deg + 180) % 360) - 180
                # angle_to_turn = np.deg2rad(angle_to_turn_deg)
                # self.turn(angle_to_turn)
                # self.check_goal_reached()
                # self.update_hmaps()
                # self.forward()        
                # break 

            # 6) If collisions => turn away
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break

            # 7) Check goal, update hmaps, forward
            self.check_goal_reached()
            self._maybe_log_training_progress()
            self._maybe_log_grid_diagnostics()

            # Update coverage tracking.
            # In two-phase learning, coverage is counted only in phase-2.
            if self.robot_mode in {
                RobotMode.LEARN_LOCATIONS_COVERAGE,
                RobotMode.LEARN_LOCATIONS_ADAPTIVE,
                RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                RobotMode.LEARN_LOCATIONS_TWO_PHASE,
                RobotMode.LEARN_HEBB,
                RobotMode.PLOTTING_COVERAGE_AUTO,
            }:
                if self._two_phase_allows_coverage_tracking():
                    robot_pos = self._current_planar_xy()
                    self._update_coverage(robot_pos)


            # Update distance tracking if in EXPLOIT_LOCATIONS_RANDOM mode
            if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
                self._update_distance_tracking()

            self.update_hmaps(update_loc=True,
                              update_hdn=True,
                              update_pcn=True,
                              update_gcn=True,
                              update_scale_priority=True if self.robot_mode == RobotMode.EXPLOIT else False,
                              update_prox=True)
            self.forward()

        # A small random turn at the end
        self.turn(np.random.normal(0, np.deg2rad(30)))

    ########################################### EXPLOIT ###########################################
    def exploit(self):
        """Main exploit entrypoint. Delegates to the current exploit implementation."""
        return self.exploit_v12()

    def exploit_v12(self):
        """
        Hierarchical multi-scale preplay exploitation with entropy-based scale selection.

        Core philosophy:
        - Evaluate trajectories across all spatial scales (small, medium, large)
        - Evaluate all 8 head directions for each scale
        - Use TWO-LEVEL normalization instead of flat Boltzmann distribution:
          1. Within each scale: Compute Boltzmann over directions P(d|s)
          2. Across scales: Score by mean return and entropy, compute P(s)
        - Form joint distribution: P(s,d) = P(s) × P(d|s)
        - Apply safety filtering and combine direction vectors
        - Return final movement direction and expected value

        Key innovations compared to exploit_v10:
        1. Entropy-based scale quality: Low entropy = clear signal = trust this scale
        2. Scale scoring: Q_s = alpha*M_s - gamma*H_s (reward vs uncertainty tradeoff)
        3. Temporal stability: EMA over scale weights P(s) across timesteps
        4. Hierarchical structure: Prevents mediocre trajectories from diluting good ones
        5. Better scale selection: Explicitly favors confident, high-reward scales
        6. Dual-threshold loop detection: Early reliability penalty (2 loops), late forced exploration (5 loops)

        Spatial aliasing correction mechanisms (configurable):
        - Scale reliability: Long-term per-scale trust adjustment based on outcomes
        - Loop detection: Two-stage response to excessive rotation (reliability then exploration)

        Parameters are defined at the top of the function for clarity.
        """
        # In unified multi-goal exploit, always use goal-specific reward maps.
        self._ensure_goal_specific_rcn_loaded_for_exploit()

        #===================================================================
        # EXPLOIT V12 PARAMETERS
        #===================================================================

        # --- Random Seed for Reproducibility ---
        preplay_random_seed = None  # Set to integer (e.g., 42) for reproducible sampling, None for random

        # --- Hierarchical Preplay Configuration ---
        num_preplay_steps = 4              # Longer local lookahead for doorway approach
        discount_factor = 0.7              # 0.9 Temporal discount (gamma) 95
        within_scale_beta = 15.0          # Softer direction arbitration to reduce doorway bin-flipping
        scale_selection_beta = 3.0        # Legacy knob; unified preplay now uses the scale prior directly
        ema_lambda = 0.25                   # 0.1 EMA decay for scale entropies (0.1 = 10% new, 90% old)
        use_entropy_ema = True             # Toggle: True = apply EMA smoothing to entropies, False = use raw entropies
        entropy_exponent = 2.0             # Exponent for inverse entropy simplex (higher = more sensitive to entropy differences)
        reliability_exponent = 2.0         # Exponent for reliability scores (higher = more sensitive to reliability differences)
        variance_lambda = 1.0              # Weight for variance term in composite entropy (higher = more penalty for high variance)

        # --- Stochastic Sampling Preplay (Biologically Plausible Alternative) ---
        num_samples_per_direction = 24     # More samples reduces rollout variance near doorways
        sampling_strategy = "learned"      # "uniform" (random turns) or "learned" (W_rec-weighted turns)
        sampling_temperature = 0.20        # Lower -> less noisy turn sampling
        sample_aggregation = "mean"        # Average sampled rollouts per head direction to reduce volatility

        # --- Safety & Loop Detection ---
        min_safe_distance = 1                        # Minimum obstacle clearance
        loop_threshold_reliability = 1               # Loops before applying reliability penalty
        loop_threshold_forced_exploration = 10        # Loops before forcing exploration
        max_steps_between_loops = 10                 # Max steps between loops to count
        forced_exploration_steps = 10                # Steps to force exploration

        # --- Spatial Aliasing Correction Strategy ---
        use_loop_forced_exploration = False     # Original rotation-based exploration
        use_scale_reliability = True           # Mechanism 2: Per-scale reliability scoring

        # --- Scale Reliability Parameters (only used if use_scale_reliability=True) ---
        reliability_bad_factor = 0.6           # Multiplicative factor for bad outcomes (e.g., 0.8 = multiply by 0.8)
        reliability_good_factor = 1.03          # Multiplicative factor for good outcomes (e.g., 1.1 = multiply by 1.1, capped at 1.0)
        reliability_attribution = 'dominant'   # 'weighted' or 'dominant'
        reliability_min_floor = 0.1            # Minimum reliability value (prevents total suppression)
        reliability_good_cooldown_steps = 4      # After a bad update, wait this many steps before rewarding that scale

        # --- Debug ---
        debug_print_interval = 0           # Set >0 only for active exploit debugging

        #===================================================================

        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()

        self.compute_pcn_activations()
        self._maybe_log_grid_diagnostics()
        self.update_hmaps(update_loc=True, update_hdn=True, update_pcn=False, update_scale_priority=False)
        self.check_goal_reached()
        if self.done:
            return

        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            self._update_distance_tracking()

        # Save old PCN activations only for the legacy TD-learning path.
        old_pcn_activations = None
        if (not self.use_unified_multiscale) and self.td_learning:
            old_pcn_activations = [
                pcn.place_cell_activations.clone() for pcn in self.pcns
            ]

        # Exploit can only begin with at least tau_w steps
        if self.step_count <= self.tau_w:
            return

        #-------------------------------------------------------------------
        # 1b) Initialize scale reliability tracking on first call
        #-------------------------------------------------------------------
        if use_scale_reliability and self.scale_reliability is None:
            num_scales = len(self.scales)
            self.scale_reliability = torch.ones(num_scales, dtype=self.dtype, device=self.device)
            self.loop_scale_contributions = torch.zeros(num_scales, dtype=self.dtype, device=self.device)
            self.reliability_good_cooldown = torch.zeros(num_scales, dtype=torch.long, device=self.device)
            print(f"[EXPLOIT_V12] Initialized scale reliability tracking for {num_scales} scales")

        # Decay cooldown timers each step so recently penalized scales cannot be rewarded immediately
        if use_scale_reliability and hasattr(self, 'reliability_good_cooldown'):
            self.reliability_good_cooldown = torch.maximum(
                self.reliability_good_cooldown - 1,
                torch.zeros_like(self.reliability_good_cooldown)
            )

        #-------------------------------------------------------------------
        # 1c) Check if all scale reliabilities have hit near-minimum threshold
        #-------------------------------------------------------------------
        reliability_exploration_threshold = 0.15  # Trigger exploration if all scales below this
        if use_scale_reliability and self.scale_reliability is not None:
            all_near_minimum = torch.all(self.scale_reliability < reliability_exploration_threshold).item()
            if all_near_minimum and not hasattr(self, 'triggered_min_reliability_exploration'):
                # All scales have been pushed near minimum reliability - trigger forced exploration
                print(f"[EXPLOIT] All scale reliabilities below threshold ({reliability_exploration_threshold}). Triggering forced exploration.")
                print(f"  Reliability values: {self.scale_reliability.cpu().numpy()}")
                self.force_explore_count = forced_exploration_steps
                self.triggered_min_reliability_exploration = True  # Flag to prevent repeated triggers
            elif not all_near_minimum:
                # Reset flag when at least one scale recovers above threshold
                if hasattr(self, 'triggered_min_reliability_exploration'):
                    delattr(self, 'triggered_min_reliability_exploration')

        #-------------------------------------------------------------------
        # 2) Forced Exploration Check
        #-------------------------------------------------------------------
        if self.force_explore_count > 0:
            self.force_explore_count -= 1
            if self.force_explore_count == 0:
                print("[EXPLOIT] Forced exploration complete. Resuming hierarchical navigation...")

            # During forced exploration, gradually regenerate reliability for all scales
            if use_scale_reliability and self.scale_reliability is not None:
                # Apply good factor uniformly to all scales to help them recover
                self.scale_reliability = torch.clamp(
                    self.scale_reliability * reliability_good_factor,
                    min=reliability_min_floor,
                    max=1.0
                )

            self.explore()
            return

        #-------------------------------------------------------------------
        # 3) Debug flag (used throughout exploit_v12)
        #-------------------------------------------------------------------
        debug_enabled = (debug_print_interval > 0 and self.step_count % debug_print_interval == 0)

        #-------------------------------------------------------------------
        # 4) Loop Detection - Detect excessive rotation and force exploration
        #-------------------------------------------------------------------
        if not hasattr(self, 'rotation_accumulator'):
            self.rotation_accumulator = 0.0
            self.rotation_loop_count = 0
            self.steps_since_last_loop = 0

        if not hasattr(self, 'last_heading_deg') or self.last_heading_deg is None:
            self.last_heading_deg = self.current_heading_deg

        heading_diff = self.current_heading_deg - self.last_heading_deg
        heading_diff = ((heading_diff + 180) % 360) - 180  # Normalize to [-180, 180]
        self.rotation_accumulator += abs(heading_diff)
        self.last_heading_deg = self.current_heading_deg

        # Track scale contributions to rotation (for reliability penalties)
        if use_scale_reliability and self.last_scale_weights is not None:
            self.loop_scale_contributions += self.last_scale_weights * (abs(heading_diff) / 360.0)

        if self.rotation_accumulator >= 360.0:
            self.rotation_loop_count += 1
            self.rotation_accumulator -= 360.0
            if self.steps_since_last_loop > max_steps_between_loops:
                self.rotation_loop_count = 1
            self.steps_since_last_loop = 0

            # Check for reliability penalty threshold (earlier trigger)
            if use_scale_reliability and self.rotation_loop_count >= loop_threshold_reliability:
                if torch.sum(self.loop_scale_contributions) > 0:
                    for scale_idx in range(len(self.scales)):
                        contribution = self.loop_scale_contributions[scale_idx]
                        # Apply weighted multiplicative decay based on contribution
                        effective_factor = 1.0 - contribution * (1.0 - reliability_bad_factor)
                        self.scale_reliability[scale_idx] = torch.clamp(
                            self.scale_reliability[scale_idx] * effective_factor,
                            min=reliability_min_floor,
                            max=1.0
                        )
                        # Start cooldown for any scale that was penalized this step
                        if contribution > 0 and reliability_good_cooldown_steps > 0:
                            self.reliability_good_cooldown[scale_idx] = reliability_good_cooldown_steps
                    if debug_enabled:
                        print(f"[RELIABILITY] Loop penalty applied after {self.rotation_loop_count} loops")
                        print(f"[RELIABILITY] Loop contributions: {self.loop_scale_contributions.cpu().numpy()}")
                        print(f"[RELIABILITY] Updated reliability: {self.scale_reliability.cpu().numpy()}")
                    # Reset loop contributions after penalty
                    self.loop_scale_contributions.fill_(0.0)

            # Check for forced exploration threshold (later trigger)
            if self.rotation_loop_count >= loop_threshold_forced_exploration:
                self.rotation_loop_count = 0
                self.rotation_accumulator = 0.0
                self.steps_since_last_loop = 0
                print(f"[EXPLOIT_V12] Detected {loop_threshold_forced_exploration} loops within {max_steps_between_loops} steps.")

                # Start forced exploration (if enabled)
                if use_loop_forced_exploration:
                    print(f"[EXPLOIT_V12] Forcing exploration for {forced_exploration_steps} steps.")
                    self.force_explore_count = forced_exploration_steps
                    self.explore()
                    return
                else:
                    print(f"[EXPLOIT_V12] Forced exploration disabled. Resetting loop counter and continuing with reliability penalties.")
        else:
            self.steps_since_last_loop += 1
            if self.steps_since_last_loop > max_steps_between_loops:
                self.rotation_loop_count = 0

        #-------------------------------------------------------------------
        # 5) Prepare distance-per-direction array for safety filtering
        #-------------------------------------------------------------------
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        distances_per_hd = self._compute_directional_clearance(boundaries_rolled)
        if self.use_unified_multiscale:
            # --- Unified reliability: single scalar in [reliability_min_floor, 1.0] ---
            if not hasattr(self, 'unified_reliability'):
                self.unified_reliability = 1.0
            if not hasattr(self, 'prev_unified_scale_entropies'):
                self.prev_unified_scale_entropies = None
            if not getattr(self, "_exploit_visibility_synced", False):
                self._sync_unified_recurrent_visibility_mask(prune_weights=False)
                self._exploit_visibility_synced = True
            original_w_in_effective = getattr(self.unified_rcn, "w_in_effective", None)
            original_reward_denom_override = getattr(
                self.unified_rcn,
                "reward_denominator_override",
                None,
            )
            room_mask_log = "roommask=inactive"
            safety_mask_log = "door_safety=disabled(hard_mask_off)"
            if original_w_in_effective is not None:
                reward_mode = str(
                    getattr(self.unified_rcn, "reward_normalization_mode", "input_l1")
                ).strip().lower()
                goal_map_mode = str(
                    getattr(self.unified_rcn, "goal_map_generation_mode", "")
                ).strip().lower()
                if (
                    goal_map_mode != "paper_room_local_replay"
                    and reward_mode in {"weight_mass", "weight_l1", "legacy"}
                ):
                    self.unified_rcn.reward_denominator_override = (
                        torch.sum(torch.abs(original_w_in_effective))
                        .detach()
                        .to(
                            device=self.unified_rcn.w_in.device,
                            dtype=original_w_in_effective.dtype,
                        )
                    )
                else:
                    self.unified_rcn.reward_denominator_override = None
                room_reward_mask, room_mask_log = self._get_unified_exploit_room_reward_mask(
                    self.unified_rcn.w_in.device
                )
                # Log masking state changes and periodically.
                prev_mask_log = getattr(self, "_prev_room_mask_log", None)
                if room_mask_log != prev_mask_log or self.step_count % 20 == 0:
                    curr_x, curr_z = self._current_planar_xy()
                    print(
                        f"[ROOMMASK] step={self.step_count} agent=({curr_x:.2f},{curr_z:.2f}) "
                        f"{room_mask_log}"
                    )
                    self._prev_room_mask_log = room_mask_log
                if room_reward_mask is not None:
                    room_mask_cache = room_reward_mask.detach().to(dtype=torch.bool).cpu()
                    prev_room_mask_cache = getattr(
                        self,
                        "_last_unified_exploit_room_reward_mask",
                        None,
                    )
                    if (
                        prev_room_mask_cache is None
                        or prev_room_mask_cache.shape != room_mask_cache.shape
                        or not torch.equal(prev_room_mask_cache, room_mask_cache)
                    ):
                        self._unified_reward_hmap_cache = None
                    self._last_unified_exploit_room_reward_mask = room_mask_cache
                    self.unified_rcn.w_in_effective = (
                        original_w_in_effective
                        * room_reward_mask.to(
                            device=self.unified_rcn.w_in.device,
                            dtype=original_w_in_effective.dtype,
                        ).unsqueeze(0)
                    )
                else:
                    if getattr(self, "_last_unified_exploit_room_reward_mask", None) is not None:
                        self._unified_reward_hmap_cache = None
                    self._last_unified_exploit_room_reward_mask = None

            try:
                (
                    final_direction_deg,
                    expected_value,
                    combined_vector,
                    macro_returns,
                    macro_vectors,
                    sampling_variances,
                    direction_probs,
                    scale_weights,
                    scale_entropies,
                ) = self.unified_pcn.unified_preplay_sampling(
                    unified_rcn=self.unified_rcn,
                    n_hd=self.n_hd,
                    num_steps=num_preplay_steps,
                    discount_factor=discount_factor,
                    within_direction_beta=within_scale_beta,
                    scale_selection_beta=scale_selection_beta,
                    ema_lambda=ema_lambda,
                    prev_scale_entropies=self.prev_unified_scale_entropies,
                    scale_reliability=self.scale_reliability if use_scale_reliability else None,
                    entropy_exponent=entropy_exponent,
                    reliability_exponent=reliability_exponent,
                    variance_lambda=variance_lambda,
                    use_entropy_ema=use_entropy_ema,
                    enable_scale_arbitration=False,
                    scale_prior_mix=0.0,
                    num_samples=num_samples_per_direction,
                    sampling_strategy=sampling_strategy,
                    sampling_temperature=sampling_temperature,
                    sample_aggregation=sample_aggregation,
                    debug=debug_enabled,
                )
                self.prev_unified_scale_entropies = scale_entropies
                self.last_unified_transition_planner_log = "clean_preplay=hierarchical_scale_blocks"
                macro_returns = torch.nan_to_num(
                    macro_returns,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                macro_vectors = torch.nan_to_num(
                    macro_vectors,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                direction_probs = torch.nan_to_num(
                    direction_probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                direction_probs = torch.clamp(direction_probs, min=0.0)
                direction_prob_sum = torch.sum(direction_probs)
                direction_prob_sum_valid = bool(torch.isfinite(direction_prob_sum).item())
                if (not direction_prob_sum_valid) or float(direction_prob_sum.item()) <= 1e-9:
                    direction_probs = torch.full(
                        (self.n_hd,),
                        1.0 / float(max(1, int(self.n_hd))),
                        dtype=self.dtype,
                        device=self.device,
                    )
                else:
                    direction_probs = direction_probs / torch.clamp(
                        direction_prob_sum,
                        min=1e-9,
                    )

                safe_thresholds = torch.full(
                    (self.n_hd,),
                    float(max(0.0, min_safe_distance)),
                    dtype=self.dtype,
                    device=self.device,
                )
                doorway_safe_bins = torch.zeros(
                    self.n_hd,
                    dtype=torch.bool,
                    device=self.device,
                )
                (
                    safe_thresholds,
                    doorway_safe_bins,
                    safety_mask_log,
                ) = self._get_unified_checkpoint_door_safety_profile(
                    base_safe_distance=min_safe_distance,
                    target_device=self.device,
                    target_dtype=self.dtype,
                )
                hard_collision_distance = float(
                    max(
                        0.05,
                        min(
                            float(max(0.0, min_safe_distance)) * 0.15,
                            getattr(self, "unified_hard_collision_distance", 0.12),
                        ),
                    )
                )
                distances_per_hd = torch.nan_to_num(
                    distances_per_hd,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                safe_thresholds_clamped = torch.clamp(safe_thresholds, min=1e-6)
                safety_weights = torch.clamp(
                    distances_per_hd / safe_thresholds_clamped,
                    min=0.0,
                    max=1.0,
                )
                safety_weights = torch.nan_to_num(
                    safety_weights,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                doorway_soft_mask = doorway_safe_bins & (
                    distances_per_hd > hard_collision_distance
                )
                if torch.any(doorway_soft_mask):
                    doorway_floor = float(
                        max(
                            0.20,
                            min(
                                0.60,
                                getattr(self, "unified_checkpoint_door_probability_floor", 0.30),
                            ),
                        )
                    )
                    safety_weights = torch.where(
                        doorway_soft_mask,
                        torch.maximum(
                            safety_weights,
                            torch.full_like(safety_weights, doorway_floor),
                        ),
                        safety_weights,
                    )
                hard_block_mask = distances_per_hd <= hard_collision_distance
                safety_weights = torch.where(
                    hard_block_mask & (~doorway_safe_bins),
                    torch.zeros_like(safety_weights),
                    safety_weights,
                )
                safe_mask = safety_weights > 1e-6
                fallback_log = ""
                if not torch.any(safe_mask):
                    max_clearance = torch.max(distances_per_hd)
                    if float(max_clearance.item()) > 1e-6:
                        safe_mask = distances_per_hd >= (max_clearance - 1e-6)
                        safety_weights = torch.where(
                            safe_mask,
                            torch.ones_like(safety_weights),
                            torch.zeros_like(safety_weights),
                        )
                        fallback_log = f",fallback=max_clearance({float(max_clearance.item()):.2f})"
                    elif torch.any(doorway_safe_bins):
                        safe_mask = doorway_safe_bins.clone()
                        safety_weights = torch.where(
                            safe_mask,
                            torch.ones_like(safety_weights),
                            torch.zeros_like(safety_weights),
                        )
                        fallback_log = ",fallback=doorway_bins"
                    else:
                        safe_mask = torch.ones(
                            self.n_hd,
                            dtype=torch.bool,
                            device=self.device,
                        )
                        safety_weights = torch.ones_like(safety_weights)
                        fallback_log = ",fallback=all_bins"
                safety_mask_log = (
                    f"{safety_mask_log},safe_bins={int(torch.sum(safe_mask).item())},"
                    f"hard_collision={hard_collision_distance:.2f}{fallback_log}"
                )
                safe_probs = torch.nan_to_num(
                    direction_probs * safety_weights,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                safe_returns = torch.nan_to_num(
                    macro_returns.clone(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                self.last_unified_room_mask_log = room_mask_log
                self.last_unified_safety_mask_log = safety_mask_log

                if debug_enabled:
                    clearance_txt = ", ".join(
                        f"{float(v):.2f}" for v in distances_per_hd.detach().cpu().tolist()
                    )
                    threshold_txt = ", ".join(
                        f"{float(v):.2f}" for v in safe_thresholds.detach().cpu().tolist()
                    )
                    safe_bin_txt = [int(idx) for idx in torch.nonzero(safe_mask, as_tuple=False).view(-1).tolist()]
                    doorway_bin_txt = [int(idx) for idx in torch.nonzero(doorway_safe_bins, as_tuple=False).view(-1).tolist()]
                    print(
                        "[UNIFIED-EXPLOIT] "
                        f"{room_mask_log} | {safety_mask_log} | "
                        f"{self.last_unified_transition_planner_log} | "
                        f"clearance=[{clearance_txt}] thresholds=[{threshold_txt}] "
                        f"safe_bins={safe_bin_txt} doorway_bins={doorway_bin_txt}"
                    )

                prob_sum = torch.sum(safe_probs)
                prob_sum_is_finite = bool(torch.isfinite(prob_sum).item())
                prob_sum_value = float(prob_sum.item()) if prob_sum_is_finite else 0.0
                if prob_sum_value <= 1e-9 and torch.any(safe_mask):
                    masked_returns = torch.where(
                        safe_mask,
                        macro_returns,
                        torch.full_like(macro_returns, -1e9),
                    )
                    masked_returns = masked_returns - torch.max(masked_returns)
                    safe_probs = torch.softmax(masked_returns, dim=0)
                    prob_sum = torch.sum(safe_probs)
                    prob_sum_is_finite = bool(torch.isfinite(prob_sum).item())
                    prob_sum_value = float(prob_sum.item()) if prob_sum_is_finite else 0.0
                if prob_sum_value <= 1e-9:
                    if debug_enabled:
                        print("[EXPLOIT-ACTION] safe_probs sum ≈ 0 → explore()")
                    self._reset_unified_heading_commit_state(
                        reason="fallback_explore_zero_safe_probs"
                    )
                    self.explore()
                    self.last_scale_weights = None
                    return

                safe_probs = torch.nan_to_num(
                    safe_probs / torch.clamp(prob_sum, min=1e-9),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                joint_direction_scores = torch.nan_to_num(
                    safe_probs * safe_returns,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                best_safe_idx = torch.argmax(joint_direction_scores)
                expected_value = torch.sum(safe_probs * safe_returns)
                combined_vector = torch.nan_to_num(
                    torch.sum(
                        safe_probs.unsqueeze(1) * macro_vectors,
                        dim=0,
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                preplay_commit_dir = getattr(
                    self.unified_pcn,
                    "last_preplay_commit_dir",
                    None,
                )
                goal_map_debug = getattr(
                    self,
                    "_loaded_goal_map_debug",
                    "unknown",
                )
                safe_angles = self._get_hd_bin_angles(
                    target_device=self.device,
                    target_dtype=self.dtype,
                )
                use_committed_heading = False
                committed_idx = int(best_safe_idx.item())
                doorway_commit_log = ""
                candidate_branch = "best_safe"
                candidate_actionable_mask = safe_mask.clone()

                if torch.any(doorway_safe_bins):
                    doorway_scores = torch.where(
                        doorway_safe_bins,
                        joint_direction_scores,
                        torch.full_like(joint_direction_scores, -1e9),
                    )
                    doorway_best_idx = int(torch.argmax(doorway_scores).item())
                    doorway_best_score = float(doorway_scores[doorway_best_idx].item())
                    global_best_score = float(joint_direction_scores[best_safe_idx].item())
                    commit_threshold = float(max(0.0, self.unified_preplay_commit_threshold))
                    doorway_competitive = doorway_best_score > -1e8 and (
                        global_best_score <= 1e-9
                        or doorway_best_score >= (1.0 - commit_threshold) * global_best_score
                    )
                    if doorway_competitive:
                        use_committed_heading = True
                        committed_idx = doorway_best_idx
                        candidate_branch = "doorway_commit"
                        candidate_actionable_mask = safe_mask & doorway_safe_bins
                        doorway_commit_log = (
                            f",doorway_commit=idx{doorway_best_idx}"
                            f"/score={doorway_best_score:.4f}"
                        )

                if not use_committed_heading:
                    topk = torch.topk(
                        joint_direction_scores,
                        k=min(2, int(joint_direction_scores.numel())),
                    ).values
                    best_score = float(topk[0].item()) if topk.numel() >= 1 else 0.0
                    second_score = float(topk[1].item()) if topk.numel() >= 2 else 0.0
                    commit_threshold = float(max(0.0, self.unified_preplay_commit_threshold))
                    if best_score > 1e-9 and (best_score - second_score) >= commit_threshold * best_score:
                        use_committed_heading = True
                        committed_idx = int(best_safe_idx.item())
                        candidate_branch = "score_commit"
                        doorway_commit_log = (
                            f",score_commit=idx{committed_idx}"
                            f"/margin={best_score - second_score:.4f}"
                        )
                if doorway_commit_log:
                    self.last_unified_safety_mask_log = (
                        f"{self.last_unified_safety_mask_log}{doorway_commit_log}"
                    )

                if use_committed_heading:
                    candidate_idx = int(committed_idx)
                    action_angle = safe_angles[candidate_idx]
                    combined_vector = torch.stack(
                        [torch.cos(action_angle), torch.sin(action_angle)]
                    )
                elif float(torch.norm(combined_vector).item()) < 1e-6:
                    candidate_idx = int(best_safe_idx.item())
                    candidate_branch = "fallback_best_safe"
                    action_angle = safe_angles[best_safe_idx]
                else:
                    candidate_branch = "trajectory_vector"
                    action_angle = torch.atan2(combined_vector[1], combined_vector[0])
                    if action_angle < 0:
                        action_angle += 2 * np.pi
                    candidate_idx = int(
                        self._angle_to_hd_bin(float(action_angle.item()))
                    )

                persistence_reset_reason = self._sync_unified_heading_commit_target()
                prev_committed_idx = getattr(
                    self,
                    "last_unified_committed_hd_bin",
                    None,
                )
                committed_score = None
                final_persisted_bin = int(candidate_idx)
                persistence_outcome = "init"

                if prev_committed_idx is not None:
                    if not isinstance(prev_committed_idx, int):
                        persistence_reset_reason = persistence_reset_reason or "committed_bin_invalid"
                        self._reset_unified_heading_commit_state(
                            reason="committed_bin_invalid",
                            clear_target_signature=False,
                        )
                        prev_committed_idx = None
                    elif prev_committed_idx < 0 or prev_committed_idx >= int(self.n_hd):
                        persistence_reset_reason = persistence_reset_reason or "committed_bin_invalid"
                        self._reset_unified_heading_commit_state(
                            reason="committed_bin_invalid",
                            clear_target_signature=False,
                        )
                        prev_committed_idx = None
                    elif not bool(candidate_actionable_mask[prev_committed_idx].item()):
                        persistence_reset_reason = (
                            persistence_reset_reason or "committed_bin_not_actionable"
                        )
                        self._reset_unified_heading_commit_state(
                            reason="committed_bin_not_actionable",
                            clear_target_signature=False,
                        )
                        prev_committed_idx = None
                    else:
                        committed_score_tensor = torch.nan_to_num(
                            joint_direction_scores[prev_committed_idx],
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        committed_score = float(
                            max(0.0, float(committed_score_tensor.item()))
                        )

                candidate_score_tensor = torch.nan_to_num(
                    joint_direction_scores[candidate_idx],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                candidate_score = float(
                    max(0.0, float(candidate_score_tensor.item()))
                )
                if prev_committed_idx is not None:
                    commit_threshold = float(
                        max(0.0, self.unified_preplay_commit_threshold)
                    )
                    if candidate_score > committed_score * (1.0 + commit_threshold):
                        persistence_outcome = "switch"
                        final_persisted_bin = int(candidate_idx)
                    else:
                        persistence_outcome = "hold"
                        final_persisted_bin = int(prev_committed_idx)
                        action_angle = safe_angles[final_persisted_bin]
                        combined_vector = torch.stack(
                            [torch.cos(action_angle), torch.sin(action_angle)]
                        )
                else:
                    persistence_outcome = "init"

                self.last_unified_committed_hd_bin = int(final_persisted_bin)
                prev_committed_txt = (
                    "none" if prev_committed_idx is None else f"idx{int(prev_committed_idx)}"
                )
                committed_score_txt = (
                    "none"
                    if committed_score is None
                    else f"{float(committed_score):.4f}"
                )
                reset_reason_txt = (
                    "none"
                    if persistence_reset_reason is None
                    else str(persistence_reset_reason)
                )
                print(
                    "[UNIFIED-PERSIST] "
                    f"candidate=idx{int(candidate_idx)} "
                    f"branch={candidate_branch} "
                    f"final=idx{int(final_persisted_bin)} "
                    f"candidate_score={candidate_score:.4f} "
                    f"committed_prev={prev_committed_txt} "
                    f"committed_score={committed_score_txt} "
                    f"outcome={persistence_outcome} "
                    f"reset={reset_reason_txt}"
                )

                self.action_heading_deg = float(torch.rad2deg(action_angle).item())
                self.scale_idx = 0
                pre_execute_heading_deg = float(self.current_heading_deg)
                trace_signature = (
                    str(goal_map_debug),
                    int(preplay_commit_dir) if preplay_commit_dir is not None else None,
                    int(final_persisted_bin),
                    round(float(self.action_heading_deg), 1),
                    str(room_mask_log),
                )
                prev_trace_signature = getattr(
                    self,
                    "_prev_unified_exploit_trace_signature",
                    None,
                )
                trace_enabled = (
                    debug_enabled
                    or self.step_count % 20 == 0
                    or trace_signature != prev_trace_signature
                )
                self._prev_unified_exploit_trace_signature = trace_signature

                if trace_enabled:
                    curr_x, curr_z = self._current_planar_xy()
                    print(
                        f"[EXPLOIT-ACTION] heading={self.action_heading_deg:.1f}° "
                        f"agent=({curr_x:.2f},{curr_z:.2f}) "
                        f"current_heading={self.current_heading_deg:.1f}° "
                        f"goal_map={goal_map_debug} "
                        f"expected_value={float(expected_value):.4f}"
                    )

                move_success = self._execute_movement(self.action_heading_deg)
                if not move_success:
                    self._reset_unified_heading_commit_state(reason="movement_failed")
                if trace_enabled:
                    executed_heading_deg = float(self.current_heading_deg)
                    executed_turn_deg = (
                        (executed_heading_deg - pre_execute_heading_deg + 180.0) % 360.0
                    ) - 180.0
                    actual_move_heading_deg = getattr(
                        self,
                        "last_executed_move_heading_deg",
                        None,
                    )
                    actual_move_bin = getattr(
                        self,
                        "last_executed_move_hd_bin",
                        None,
                    )
                    actual_move_distance = float(
                        getattr(self, "last_executed_move_distance", 0.0)
                    )
                    move_heading_txt = (
                        "none"
                        if actual_move_heading_deg is None
                        else f"{float(actual_move_heading_deg):.1f}°"
                    )
                    move_bin_txt = (
                        "none" if actual_move_bin is None else str(int(actual_move_bin))
                    )
                    curr_x, curr_z = self._current_planar_xy()
                    print(
                        f"[EXPLOIT-TRACE] step={self.step_count} goal_map={goal_map_debug} "
                        f"agent=({curr_x:.2f},{curr_z:.2f}) room={room_mask_log} "
                        f"planned_heading={self.action_heading_deg:.1f}° "
                        f"executed_heading={executed_heading_deg:.1f}° "
                        f"executed_turn={executed_turn_deg:+.1f}° "
                        f"move_heading={move_heading_txt} "
                        f"move_bin={move_bin_txt} "
                        f"move_dist={actual_move_distance:.3f} "
                        f"move_ok={int(bool(move_success))}"
                    )

                self.last_scale_weights = scale_weights
                return
            finally:
                if original_w_in_effective is not None:
                    self.unified_rcn.w_in_effective = original_w_in_effective
                    self.unified_rcn.reward_denominator_override = original_reward_denom_override

        #-------------------------------------------------------------------
        # 6) Build scales_data for hierarchical preplay
        #-------------------------------------------------------------------
        # Keep hierarchical preplay aligned with the old model's vector-based
        # action selection. The circular-mean selector is unified-only.
        heading_selection_mode = "trajectory_vector"
        scales_data = [
            (scale_def['name'], pcn, rcn)
            for scale_def, pcn, rcn in zip(self.scales, self.pcns, self.rcns)
        ]

        # Initialize EMA state for scale weights if needed
        if not hasattr(self, 'prev_scale_entropies_v12'):
            self.prev_scale_entropies_v12 = None

        #-------------------------------------------------------------------
        # 6b) Set random seed for reproducibility (if specified)
        #-------------------------------------------------------------------
        if preplay_random_seed is not None:
            np.random.seed(preplay_random_seed)
            torch.manual_seed(preplay_random_seed)

        #-------------------------------------------------------------------
        # 7) Call hierarchical preplay (sampling mode)
        #-------------------------------------------------------------------
        (final_direction_deg, expected_value, combined_vector,
         discounted_returns, direction_vectors, joint_probs, trajectory_metadata, scale_entropies, sampling_variances) = self.pcns[0].hierarchical_multiscale_preplay_sampling(
            scales_data=scales_data,
            num_steps=num_preplay_steps,
            discount_factor=discount_factor,
            within_scale_beta=within_scale_beta,
            scale_selection_beta=scale_selection_beta,
            ema_lambda=ema_lambda,
            prev_scale_entropies=self.prev_scale_entropies_v12,
            scale_reliability=self.scale_reliability if use_scale_reliability else None,
            entropy_exponent=entropy_exponent,
            reliability_exponent=reliability_exponent,
            variance_lambda=variance_lambda,
            use_entropy_ema=use_entropy_ema,
            num_samples=num_samples_per_direction,
            sampling_strategy=sampling_strategy,
            sampling_temperature=sampling_temperature,
            sample_aggregation=sample_aggregation,
            debug=debug_enabled
        )

        # Store sampling variances for diagnostics (optional)
        if not hasattr(self, 'preplay_sampling_variances'):
            self.preplay_sampling_variances = []
        self.preplay_sampling_variances.append(sampling_variances.cpu().numpy())

        # Store scale entropies for next timestep's EMA
        self.prev_scale_entropies_v12 = scale_entropies

        #-------------------------------------------------------------------
        # 8) Trajectory Selection
        #-------------------------------------------------------------------
        # Clearance is still measured for diagnostics, but exploit no longer
        # hard-masks trajectories by doorway/wall clearance.
        safe_traj_mask = torch.ones(
            len(trajectory_metadata),
            dtype=torch.bool,
            device=self.device,
        )

        safe_probs = joint_probs[safe_traj_mask]
        safe_probs = safe_probs / torch.clamp(torch.sum(safe_probs), min=1e-9)

        safe_vectors = direction_vectors[safe_traj_mask]
        safe_returns = discounted_returns[safe_traj_mask]
        safe_metadata = [trajectory_metadata[i] for i, safe in enumerate(safe_traj_mask) if safe]

        expected_value = torch.sum(safe_probs * safe_returns)
        combined_vector = torch.sum(safe_probs.unsqueeze(1) * safe_vectors, dim=0)

        if debug_enabled:
            num_safe = len(safe_metadata)
            unsafe_count = len(trajectory_metadata) - num_safe
            print(f"[EXPLOIT_V12] Safety filter: {num_safe} safe, {unsafe_count} unsafe")

        #-------------------------------------------------------------------
        # Opposing Vectors Check (Cross-Scale)
        #-------------------------------------------------------------------
        # Check if top-2 trajectories by return have opposing direction vectors
        # This operates across ALL scales on the safe trajectories

        if heading_selection_mode != "circular_reward_mean":
            opposition_threshold = -0.75  # Cosine similarity threshold for opposition

            if len(safe_returns) >= 2:
                # Find top 2 trajectories by discounted return (cross-scale)
                idx1 = torch.argmax(safe_returns).item()

                # Find second best (mask out idx1)
                safe_returns_masked = safe_returns.clone()
                safe_returns_masked[idx1] = -float('inf')
                idx2 = torch.argmax(safe_returns_masked).item()

                # Get their direction vectors
                v1 = safe_vectors[idx1]
                v2 = safe_vectors[idx2]

                # Compute cosine similarity
                norm1 = torch.norm(v1)
                norm2 = torch.norm(v2)

                if norm1 > 1e-9 and norm2 > 1e-9:
                    cos_sim = torch.dot(v1, v2) / (norm1 * norm2)

                    # Check if vectors are strongly opposing
                    if cos_sim < opposition_threshold:
                        # Opposition detected - always use best trajectory
                        combined_vector = v1
                        expected_value = safe_returns[idx1]

                        if debug_enabled:
                            meta1 = safe_metadata[idx1]
                            meta2 = safe_metadata[idx2]
                            r1 = safe_returns[idx1].item()
                            r2 = safe_returns[idx2].item()
                            print(f"[EXPLOIT_V12] Opposition detected: cos_sim={cos_sim.item():.3f}")
                            print(f"  Top1: {meta1['scale_name'][0]}-{meta1['direction']*45:3d}° R={r1:.3f}")
                            print(f"  Top2: {meta2['scale_name'][0]}-{meta2['direction']*45:3d}° R={r2:.3f}")
                            print(f"  Resolution: Using best trajectory")

            # Fallback: if combined vector near zero, use max-return safe trajectory
            combined_magnitude = torch.norm(combined_vector)
            epsilon = 1e-6
            if combined_magnitude < epsilon:
                max_idx = torch.argmax(safe_returns)
                combined_vector = safe_vectors[max_idx]
                expected_value = safe_returns[max_idx]
                if debug_enabled:
                    print(f"[EXPLOIT_V12] Fallback: combined vector near zero, using max-return trajectory")

        #-------------------------------------------------------------------
        # 10) Compute Final Angle
        #-------------------------------------------------------------------
        if heading_selection_mode == "circular_reward_mean":
            final_direction_rad = action_angle
        else:
            final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])
        final_direction_deg = float((final_direction_rad * (180.0 / np.pi)).item())

        if final_direction_deg < 0:
            final_direction_deg += 360.0

        self.action_heading_deg = final_direction_deg

        #-------------------------------------------------------------------
        # 11) Debug Output
        #-------------------------------------------------------------------
        if debug_enabled:
            # Compute confidence (max probability among safe trajectories)
            confidence = torch.max(safe_probs).item()

            # Compute scale contributions (sum of probabilities per scale)
            scale_contributions = []
            for scale_idx in range(len(scales_data)):
                scale_mass = sum(safe_probs[i].item() for i, meta in enumerate(safe_metadata)
                               if meta['scale_idx'] == scale_idx)
                scale_contributions.append(scale_mass)

            scale_str = ' '.join([f"{scales_data[i][0][0]}:{c:.2f}"
                                 for i, c in enumerate(scale_contributions)])

            # Add reliability info to debug output
            reliability_str = ""
            if use_scale_reliability and self.scale_reliability is not None:
                reliability_vals = ' '.join([f"{scales_data[i][0][0]}:{self.scale_reliability[i].item():.2f}"
                                            for i in range(len(scales_data))])
                reliability_str = f" Reliability:[{reliability_vals}]"

            print(f"[EXPLOIT_V12 #{self.step_count}] θ={final_direction_deg:.1f}° "
                  f"V={expected_value.item():.3f} Conf={confidence:.2f} Scales:[{scale_str}]{reliability_str}")

        #-------------------------------------------------------------------
        # 11b) Store decision for credit assignment (before movement)
        #-------------------------------------------------------------------
        if use_scale_reliability:
            # Compute scale weights from executed trajectory probabilities
            executed_scale_weights = torch.zeros(len(scales_data), dtype=self.dtype, device=self.device)
            for prob, meta in zip(safe_probs, safe_metadata):
                executed_scale_weights[meta['scale_idx']] += prob
            self.last_scale_weights = executed_scale_weights

        #-------------------------------------------------------------------
        # 12) Execute Movement
        #-------------------------------------------------------------------
        self._execute_movement(self.action_heading_deg)

        #-------------------------------------------------------------------
        # 13) TD Learning Update for all scales
        #-------------------------------------------------------------------
        if self.td_learning:
            for i in range(len(self.pcns)):
                pcn, rcn = self.pcns[i], self.rcns[i]
                new_pcn_activations = pcn.place_cell_activations
                rcn.update_reward_cell_activations(new_pcn_activations, visit=False)
                observed_reward = float(rcn.reward_cell_activations.item())
                rcn.td_update(old_pcn_activations[i], observed_reward)

        #-------------------------------------------------------------------
        # 14) Collision-based Suppression and Reliability Updates
        #-------------------------------------------------------------------
        # Note: Collision will be detected in the next timestep after sense()
        if torch.any(self.collided):
            # Penalize scale reliability (ONLY if not in forced exploration mode)
            if use_scale_reliability and self.last_scale_weights is not None:
                # Check if we're in forced exploration mode - if so, skip bad updates
                in_forced_exploration = hasattr(self, 'force_explore_count') and self.force_explore_count > 0
                if not in_forced_exploration:
                    self._update_scale_reliability_bad(
                        scale_weights=self.last_scale_weights,
                        bad_factor=reliability_bad_factor,
                        attribution=reliability_attribution,
                        min_floor=reliability_min_floor,
                        cooldown=self.reliability_good_cooldown if hasattr(self, 'reliability_good_cooldown') else None,
                        cooldown_steps=reliability_good_cooldown_steps
                    )
                    if debug_enabled:
                        print(f"[RELIABILITY] Collision penalty applied. Reliability: {self.scale_reliability.cpu().numpy()}")
                elif debug_enabled:
                    print(f"[RELIABILITY] Collision during exploration - skipping penalty. Reliability: {self.scale_reliability.cpu().numpy()}")
        else:
            # No collision: naturally replenish reliability (good behavior)
            # This happens both in normal exploit and during forced exploration
            if use_scale_reliability and self.last_scale_weights is not None:
                self._update_scale_reliability_good(
                    scale_weights=self.last_scale_weights,
                    good_factor=reliability_good_factor,
                    attribution=reliability_attribution,
                    cooldown=self.reliability_good_cooldown if hasattr(self, 'reliability_good_cooldown') else None
                )

        return

    def _update_scale_reliability_bad(self, scale_weights, bad_factor, attribution='weighted', min_floor=0.1,
                                      cooldown=None, cooldown_steps=0):
        """Decrease reliability for scales that contributed to bad outcomes using multiplicative decay.

        Args:
            scale_weights: Current scale weights [num_scales] from last decision
            bad_factor: Multiplicative decay factor (e.g., 0.8 means multiply by 0.8)
            attribution: 'weighted' (proportional) or 'dominant' (argmax only)
            min_floor: Minimum reliability value to prevent total suppression
        """
        if self.scale_reliability is None:
            return

        if attribution == 'weighted':
            # Apply weighted multiplicative decay: blend between no decay (1.0) and full decay (bad_factor)
            # based on contribution
            for scale_idx in range(len(self.scale_reliability)):
                contribution = scale_weights[scale_idx].item()
                # Effective factor: interpolate between 1.0 (no penalty) and bad_factor (full penalty)
                effective_factor = 1.0 - contribution * (1.0 - bad_factor)
                self.scale_reliability[scale_idx] = torch.clamp(
                    self.scale_reliability[scale_idx] * effective_factor,
                    min=min_floor,
                    max=1.0
                )
                if cooldown is not None and contribution > 0 and cooldown_steps > 0:
                    cooldown[scale_idx] = cooldown_steps
        elif attribution == 'dominant':
            # Penalize only the dominant scale with full multiplicative decay
            dominant_idx = torch.argmax(scale_weights).item()
            self.scale_reliability[dominant_idx] = torch.clamp(
                self.scale_reliability[dominant_idx] * bad_factor,
                min=min_floor,
                max=1.0
            )
            if cooldown is not None and cooldown_steps > 0:
                cooldown[dominant_idx] = cooldown_steps

    def _update_scale_reliability_good(self, scale_weights, good_factor, attribution='weighted', cooldown=None):
        """Increase reliability for scales that contributed to good outcomes using multiplicative growth.

        Args:
            scale_weights: Current scale weights [num_scales] from last decision
            good_factor: Multiplicative growth factor (e.g., 1.1 means multiply by 1.1)
            attribution: 'weighted' (proportional) or 'dominant' (argmax only)
        """
        if self.scale_reliability is None:
            return

        if attribution == 'weighted':
            # Apply weighted multiplicative growth: blend between no growth (1.0) and full growth (good_factor)
            # based on contribution
            for scale_idx in range(len(self.scale_reliability)):
                # Skip if this scale is still in cooldown from a recent penalty
                if cooldown is not None and cooldown[scale_idx] > 0:
                    continue
                contribution = scale_weights[scale_idx].item()
                # Effective factor: interpolate between 1.0 (no reward) and good_factor (full reward)
                effective_factor = 1.0 + contribution * (good_factor - 1.0)
                # Apply multiplicative growth with cap at 1.0
                self.scale_reliability[scale_idx] = min(
                    self.scale_reliability[scale_idx] * effective_factor,
                    1.0
                )
        elif attribution == 'dominant':
            # Reward only the dominant scale with full multiplicative growth
            dominant_idx = torch.argmax(scale_weights).item()
            if cooldown is not None and cooldown[dominant_idx] > 0:
                return
            # Apply multiplicative growth with cap at 1.0
            self.scale_reliability[dominant_idx] = min(
                self.scale_reliability[dominant_idx] * good_factor,
                1.0
            )


    ########################################### SENSE ###########################################
    def sense(self):
        """
        Uses sensors to update range-image, heading, boundary data, collision flags, etc.
        """
        # Recover if physics left the robot tipped over.
        self._ensure_upright()

        # Advance simulation one timestep
        self.step(self.timestep)

        # Get the latest boundary data from range finder
        boundaries = self.range_finder.getRangeImage()

        # Update global heading (0–360)
        self.current_heading_deg = round(
            self.get_bearing_in_degrees(self.compass.getValues()), 1
        )

        # Shift boundary data based on global heading
        lidar_shift = 2.0 * float(self.current_heading_deg)
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            shifts=0,
        )
        self.boundaries = self._circular_roll_float_1d(
            self.boundaries,
            shift_bins=lidar_shift,
        )

        # Convert clockwise controller heading into standard math angle for HD cells.
        current_heading_rad = self._clockwise_heading_deg_to_standard_rad(
            self.current_heading_deg
        )
        v_in = torch.tensor(
            [np.cos(current_heading_rad), np.sin(current_heading_rad)],
            dtype=self.dtype,
            device=self.device,
        )

        # Update head direction layer activations
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)

        # Check for collisions via bumpers
        self.collided[0] = int(self.left_bumper.getValue())
        self.collided[1] = int(self.right_bumper.getValue())

    def _circular_roll_float_1d(
        self,
        values: torch.Tensor,
        shift_bins: float,
    ) -> torch.Tensor:
        """
        Circularly shift a 1D tensor by a fractional number of bins.

        `torch.roll()` only accepts integer shifts. For LiDAR alignment we
        preserve the sub-bin component by linearly interpolating between the two
        neighboring integer circular shifts.
        """
        if values.dim() != 1:
            values = values.reshape(-1)
        num_bins = int(values.shape[0])
        if num_bins <= 1:
            return values

        shift = float(shift_bins) % float(num_bins)
        base_shift = int(np.floor(shift))
        frac = float(shift - float(base_shift))
        if frac <= 1e-9:
            return torch.roll(values, shifts=base_shift)

        rolled_lo = torch.roll(values, shifts=base_shift)
        rolled_hi = torch.roll(values, shifts=base_shift + 1)
        return ((1.0 - frac) * rolled_lo) + (frac * rolled_hi)


    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """
        Converts a 'north' vector (from compass) to a global heading in degrees [0, 360).
        The simulator's 'north' often aligns with the negative Y-axis, so we do a shift.
        """
        # Angle from the x-axis
        rad = np.arctan2(north[1], north[0])

        # Convert from radians to degrees, shift by -90 deg to align with "north"
        bearing = (rad - 1.5708) / np.pi * 180.0

        # Wrap negative angles into [0, 360)
        if bearing < 0:
            bearing += 360.0

        return bearing

    def _compute_trimmed_lidar_mean(self, sigma_threshold: float = 2.5) -> float:
        """
        Compute robust proximity from LiDAR by trimming outliers.

        Procedure:
        1) Mean/std over all readings
        2) Keep inliers within mean ± sigma_threshold * std
        3) Return mean of inliers (fallback to raw mean if needed)
        """
        readings = self.boundaries.detach().float().view(-1)
        readings = readings[torch.isfinite(readings)]

        if readings.numel() == 0:
            return 0.0

        raw_mean = torch.mean(readings)
        raw_std = torch.std(readings, unbiased=False)

        if (not torch.isfinite(raw_std)) or raw_std.item() <= 1e-9:
            return float(raw_mean.item())

        lower = raw_mean - sigma_threshold * raw_std
        upper = raw_mean + sigma_threshold * raw_std
        inliers = readings[(readings >= lower) & (readings <= upper)]

        if inliers.numel() == 0:
            return float(raw_mean.item())

        return float(torch.mean(inliers).item())

    def _compute_opposite_pair_halfwidth_percentile(
        self, percentile: float = 25.0
    ) -> float:
        """
        Compute corridor-aware proximity from opposite LiDAR pairs.

        Steps:
        1) Pair each ray with its opposite (180 degrees apart)
        2) Compute pair half-width as average of opposite rays:
           0.5 * (d_i + d_opposite)
        3) Return the minimum pair-width value (most constricted opposite pair)
        """
        readings = self.boundaries.detach().float().view(-1)
        readings = readings[torch.isfinite(readings)]

        if readings.numel() == 0:
            return 0.0

        n = int(readings.numel())
        half = n // 2
        if half == 0:
            return float(torch.mean(readings).item())

        # If odd, drop the last ray so opposite pairing remains aligned.
        usable = readings[: 2 * half]
        first = usable[:half]
        second = usable[half : 2 * half]
        pair_width = 0.5 * (first + second)

        if pair_width.numel() == 0:
            return float(torch.mean(readings).item())

        value = torch.min(pair_width)
        if not torch.isfinite(value):
            return float(torch.mean(pair_width).item())
        return float(value.item())

    def _compute_proximity_local_minima(self) -> float:
        """
        Compute proximity via the bilateral-constraint arithmetic mean.

        Algorithm:
        1. Apply a 5-ray circular median filter to suppress single-ray noise.
        2. d1 = global minimum of the smoothed scan (direction of nearest wall).
           Using argmin instead of local-minima detection avoids a systematic
           bug: the median filter creates flat plateaus near wall perpendiculars
           (adjacent smoothed values become equal within float precision), so
           the strict local-minima test ``smoothed < prev & smoothed < next``
           fails and the function would fall back to returning d1 alone (no
           bilateral constraint), causing small fields along all open walls.
        3. d2 = minimum of the smoothed scan in the opposite 180° semicircle
           from d1 (indices within ±90° of the direction directly opposite d1).
           This captures "closest wall on the other side of the robot" without
           any arbitrary angular threshold — n//2 and n//4 are pure geometry.
        4. Return arithmetic mean (d1 + d2) / 2.

        The bilateral constraint fires small-scale fields only where BOTH sides
        of the robot are close (corners, doorways, corridors). A wall seen
        obliquely through a doorway at <90° from the near wall falls outside
        the opposite window and does not inflate the proximity estimate.

        Key cases:
          0 minima          -> global minimum of finite readings (open fallback)
          Corner/corridor   -> d1 ≈ d2 small, mean small   -> small fields ✓
          Open wall         -> d1 small, d2 from far wall   -> mean large, no fields ✓
          Doorway peek <90° -> peek wall outside window     -> d2 = far wall, suppressed ✓
        """
        raw = self.boundaries.detach().float().view(-1).cpu().numpy()
        n = len(raw)

        if n == 0:
            return 0.0

        finite_mask = np.isfinite(raw)
        if not np.any(finite_mask):
            return 0.0

        # Replace non-finite values with the maximum finite reading so they
        # appear as local maxima (never local minima) after smoothing.
        filled = raw.copy()
        filled[~finite_mask] = float(np.max(raw[finite_mask]))

        # --- Step 1: 5-ray circular median filter ---
        smoothed = np.median(
            np.stack([
                np.roll(filled, -2),
                np.roll(filled, -1),
                filled,
                np.roll(filled,  1),
                np.roll(filled,  2),
            ], axis=0),
            axis=0,
        )

        # --- Step 2: d1 = global minimum of smoothed scan (nearest wall direction) ---
        # We bypass local-minima detection entirely. The 5-ray median filter can create
        # flat plateaus that eliminate strict local minima (smoothed[i] == neighbours
        # within floating-point precision), causing the "no local minima" fallback to
        # trigger even when a wall is clearly present.  argmin is always well-defined
        # and correctly locates the nearest-wall direction without any such artefact.
        idx1 = int(np.argmin(smoothed))
        d1 = float(smoothed[idx1])

        # --- Step 3: d2 = nearest wall in the opposite 180° semicircle ---
        # Centre the search window directly opposite d1; accept ±90° (quarter
        # circle) around that centre.  n//2 = "half the scanner" (180°),
        # n//4 = "quarter of the scanner" (90°) — no tunable parameter.
        opposite_centre = (idx1 + n // 2) % n
        half_window = n // 4
        offsets = np.arange(-half_window, half_window + 1)
        opposite_indices = (opposite_centre + offsets) % n
        d2 = float(np.min(smoothed[opposite_indices]))

        return float((d1 + d2) / 2.0)

    def _compute_proximity_raw_local_minima(self) -> float:
        """
        Bilateral-constraint proximity using raw local minima (no median filter).

        Identical bilateral logic to _compute_proximity_local_minima but skips
        the 5-ray median filter. In a noiseless simulation environment (Webots)
        wall perpendiculars produce genuine strict local minima in the raw scan,
        so the plateau artifact that forced us to use argmin does not occur here.

        Algorithm:
        1. Find all strict local minima in the raw scan (val < both neighbours,
           circular wrap). These correspond to wall-perpendicular directions.
        2. d1 = smallest local minimum (nearest wall). If none found, falls back
           to the global minimum (same as 'local_minima' mode fallback).
        3. d2 = smallest local minimum in the ±90° window opposite d1. If none
           found in the window, falls back to the global minimum of that window.
        4. Return (d1 + d2) / 2.
        """
        raw = self.boundaries.detach().float().view(-1).cpu().numpy()
        n = len(raw)

        if n == 0:
            return 0.0

        finite_mask = np.isfinite(raw)
        if not np.any(finite_mask):
            return 0.0

        filled = raw.copy()
        filled[~finite_mask] = float(np.max(raw[finite_mask]))

        # Strict local minima: each ray must be less than both circular neighbours.
        prev_vals = np.roll(filled, 1)
        next_vals = np.roll(filled, -1)
        is_local_min = (filled < prev_vals) & (filled < next_vals)
        local_min_indices = np.where(is_local_min)[0]

        if local_min_indices.size > 0:
            idx1 = int(local_min_indices[np.argmin(filled[local_min_indices])])
        else:
            idx1 = int(np.argmin(filled))
        d1 = float(filled[idx1])

        opposite_centre = (idx1 + n // 2) % n
        half_window = n // 4
        offsets = np.arange(-half_window, half_window + 1)
        opposite_indices = (opposite_centre + offsets) % n

        opp_local_mins = opposite_indices[is_local_min[opposite_indices]]
        if opp_local_mins.size > 0:
            d2 = float(np.min(filled[opp_local_mins]))
        else:
            d2 = float(np.min(filled[opposite_indices]))

        return float((d1 + d2) / 2.0)

    def _compute_proximity_distance(self) -> float:
        """Compute proximity scalar used by scale modulation."""
        if self.proximity_mode == "trimmed_mean":
            return self._compute_trimmed_lidar_mean(
                sigma_threshold=self.proximity_trimmed_sigma
            )
        if self.proximity_mode == "opposite_pair_percentile":
            return self._compute_opposite_pair_halfwidth_percentile(
                percentile=self.proximity_pair_percentile
            )
        if self.proximity_mode == "local_minima":
            return self._compute_proximity_local_minima()
        if self.proximity_mode == "raw_local_minima":
            return self._compute_proximity_raw_local_minima()
        return float(torch.min(self.boundaries).item())

    def _record_scale_diagnostics(self) -> None:
        """Record per-step diagnostics for scale behavior."""
        if self.step_count % self.scale_diag_sample_stride != 0:
            return

        num_scales = len(self.diag_scale_indices)
        self.diag_sample_steps.append(int(self.step_count))
        try:
            self.diag_prox_values.append(float(self.prox))
        except Exception:
            self.diag_prox_values.append(float("nan"))

        pref_np = np.full(num_scales, np.nan, dtype=np.float32)
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            pref = getattr(self.unified_pcn, "last_scale_preference", None)
            if pref is not None:
                pref_np = pref.detach().float().cpu().numpy().astype(np.float32)
        self.diag_scale_pref_values.append(pref_np)

        # Activation diagnostics per scale from current activation buffers.
        active_counts = []
        active_fractions = []
        activation_sums = []
        activation_means = []
        activation_peaks = []
        pcn_activations_list = list(getattr(self, "pcn_activations_list", []))
        for scale_list_idx, scale_def in enumerate(self.scales):
            act = pcn_activations_list[scale_list_idx] if scale_list_idx < len(pcn_activations_list) else None
            num_pc = int(max(1, scale_def.get("num_pc", 0)))
            if act is None:
                active_counts.append(0)
                active_fractions.append(0.0)
                activation_sums.append(0.0)
                activation_means.append(0.0)
                activation_peaks.append(0.0)
            else:
                active_count = int(torch.sum(act > 0).item())
                active_counts.append(active_count)
                active_fractions.append(float(active_count / float(num_pc)))
                activation_sums.append(float(torch.sum(act).item()))
                activation_means.append(float(torch.mean(act).item()))
                activation_peaks.append(
                    float(torch.max(act).item()) if act.numel() > 0 else 0.0
                )
        self.diag_active_counts.append(active_counts)
        self.diag_active_fractions.append(active_fractions)
        self.diag_activation_sums.append(activation_sums)
        self.diag_activation_means.append(activation_means)
        self.diag_activation_peaks.append(activation_peaks)

        cross_scale_inhibition_means = np.full(num_scales, np.nan, dtype=np.float32)
        cross_scale_inhibition_peaks = np.full(num_scales, np.nan, dtype=np.float32)
        other_scale_activity_sums = np.full(num_scales, np.nan, dtype=np.float32)
        effective_cross_scale_factors = np.full(num_scales, np.nan, dtype=np.float32)
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            unified_pcn = self.unified_pcn
            for attr_name, target in (
                ("last_cross_scale_inhibition_mean_per_scale", cross_scale_inhibition_means),
                ("last_cross_scale_inhibition_peak_per_scale", cross_scale_inhibition_peaks),
                ("last_cross_scale_other_scale_activity_sum_per_scale", other_scale_activity_sums),
                ("last_cross_scale_effective_factor_per_scale", effective_cross_scale_factors),
            ):
                attr_value = getattr(unified_pcn, attr_name, None)
                if attr_value is None:
                    continue
                attr_np = attr_value.detach().float().cpu().numpy().astype(np.float32)
                width = min(num_scales, attr_np.shape[0])
                target[:width] = attr_np[:width]
        self.diag_cross_scale_inhibition_means.append(cross_scale_inhibition_means)
        self.diag_cross_scale_inhibition_peaks.append(cross_scale_inhibition_peaks)
        self.diag_other_scale_activity_sums.append(other_scale_activity_sums)
        self.diag_effective_cross_scale_factors.append(effective_cross_scale_factors)

        afferent_metric_rows = {
            metric_name: np.full(num_scales, np.nan, dtype=np.float32)
            for metric_name in self.diag_afferent_metric_rows
        }
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            unified_pcn = self.unified_pcn
            for metric_name, attr_name in self.afferent_diag_attr_by_metric.items():
                attr_value = getattr(unified_pcn, attr_name, None)
                if attr_value is None:
                    continue
                attr_np = attr_value.detach().float().cpu().numpy().astype(np.float32)
                width = min(num_scales, attr_np.shape[0])
                afferent_metric_rows[metric_name][:width] = attr_np[:width]
        for metric_name, row in afferent_metric_rows.items():
            self.diag_afferent_metric_rows[metric_name].append(row)

    def _build_scale_diagnostics_payload(self) -> Dict[str, Any]:
        """Summarize collected diagnostics into a compact serializable payload."""
        lambda_base = 0.20
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            lambda_base = float(
                getattr(self.unified_pcn, "cross_scale_inhibition_lambda_base", 0.20)
            )
        payload: Dict[str, Any] = {
            "samples_recorded": int(len(self.diag_sample_steps)),
            "steps_recorded": int(len(self.diag_prox_values)),
            "scale_indices": list(self.diag_scale_indices),
            "proximity_mode": str(getattr(self, "proximity_mode", "")),
            "proximity_pair_percentile": float(getattr(self, "proximity_pair_percentile", 0.0)),
            "pcn_cross_scale_inhibition_base_enabled": bool(
                getattr(self, "pcn_cross_scale_inhibition_base_enabled", False)
            ),
            "cross_scale_inhibition_lambda_base": lambda_base,
            "proximity_stats": {},
            "proximity_histogram": {},
            "scale_preference_stats": {},
            "active_cell_stats": {},
            "active_fraction_stats": {},
            "activation_sum_stats": {},
            "activation_mean_stats": {},
            "activation_peak_stats": {},
            "cross_scale_inhibition_mean_stats": {},
            "cross_scale_inhibition_peak_stats": {},
            "other_scale_activity_sum_stats": {},
            "effective_cross_scale_factor_stats": {},
            "sampled_history": {
                "steps": [int(step) for step in self.diag_sample_steps],
                "proximity": [float(v) for v in self.diag_prox_values],
                "per_scale": {
                    str(scale_idx): {} for scale_idx in self.diag_scale_indices
                },
            },
        }
        for metric_name in self.diag_afferent_metric_rows:
            payload[f"{metric_name}_stats"] = {}

        def _summarize_per_scale_matrix(
            rows,
            target_key: str,
            include_nonzero_step_ratio: bool = False,
            include_median: bool = False,
        ) -> None:
            if len(rows) == 0:
                return
            arr = np.asarray(rows, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] == 0:
                return
            n_scales = min(arr.shape[1], len(self.diag_scale_indices))
            for i in range(n_scales):
                vals = arr[:, i]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                stats = {
                    "count": int(vals.size),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                }
                if include_median:
                    stats["median"] = float(np.median(vals))
                if include_nonzero_step_ratio:
                    stats["nonzero_step_ratio"] = float(np.mean(vals > 0))
                payload[target_key][str(self.diag_scale_indices[i])] = stats

        def _store_sampled_history(metric_name: str, rows) -> None:
            if len(rows) == 0:
                return
            arr = np.asarray(rows, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] == 0:
                return
            n_scales = min(arr.shape[1], len(self.diag_scale_indices))
            for i in range(n_scales):
                payload["sampled_history"]["per_scale"][str(self.diag_scale_indices[i])][
                    metric_name
                ] = arr[:, i].tolist()

        prox = np.asarray(self.diag_prox_values, dtype=np.float64)
        prox = prox[np.isfinite(prox)]
        if prox.size > 0:
            payload["proximity_stats"] = {
                "count": int(prox.size),
                "min": float(np.min(prox)),
                "max": float(np.max(prox)),
                "mean": float(np.mean(prox)),
                "std": float(np.std(prox)),
                "p05": float(np.percentile(prox, 5)),
                "p25": float(np.percentile(prox, 25)),
                "p50": float(np.percentile(prox, 50)),
                "p75": float(np.percentile(prox, 75)),
                "p95": float(np.percentile(prox, 95)),
            }
            hist_counts, hist_edges = np.histogram(prox, bins=20)
            payload["proximity_histogram"] = {
                "bins": hist_edges.tolist(),
                "counts": hist_counts.tolist(),
            }

        _summarize_per_scale_matrix(self.diag_scale_pref_values, "scale_preference_stats")
        _summarize_per_scale_matrix(
            self.diag_active_counts,
            "active_cell_stats",
            include_nonzero_step_ratio=True,
            include_median=True,
        )
        _summarize_per_scale_matrix(
            self.diag_active_fractions,
            "active_fraction_stats",
            include_nonzero_step_ratio=True,
        )
        _summarize_per_scale_matrix(self.diag_activation_sums, "activation_sum_stats")
        _summarize_per_scale_matrix(self.diag_activation_means, "activation_mean_stats")
        _summarize_per_scale_matrix(self.diag_activation_peaks, "activation_peak_stats")
        _summarize_per_scale_matrix(
            self.diag_cross_scale_inhibition_means,
            "cross_scale_inhibition_mean_stats",
        )
        _summarize_per_scale_matrix(
            self.diag_cross_scale_inhibition_peaks,
            "cross_scale_inhibition_peak_stats",
        )
        _summarize_per_scale_matrix(
            self.diag_other_scale_activity_sums,
            "other_scale_activity_sum_stats",
        )
        _summarize_per_scale_matrix(
            self.diag_effective_cross_scale_factors,
            "effective_cross_scale_factor_stats",
        )
        for metric_name, rows in self.diag_afferent_metric_rows.items():
            _summarize_per_scale_matrix(rows, f"{metric_name}_stats")

        _store_sampled_history("scale_preference", self.diag_scale_pref_values)
        _store_sampled_history("active_count", self.diag_active_counts)
        _store_sampled_history("active_fraction", self.diag_active_fractions)
        _store_sampled_history("activation_sum", self.diag_activation_sums)
        _store_sampled_history("activation_mean", self.diag_activation_means)
        _store_sampled_history("activation_peak", self.diag_activation_peaks)
        _store_sampled_history(
            "cross_scale_inhibition_mean",
            self.diag_cross_scale_inhibition_means,
        )
        _store_sampled_history(
            "cross_scale_inhibition_peak",
            self.diag_cross_scale_inhibition_peaks,
        )
        _store_sampled_history(
            "other_scale_activity_sum",
            self.diag_other_scale_activity_sums,
        )
        _store_sampled_history(
            "effective_cross_scale_factor",
            self.diag_effective_cross_scale_factors,
        )
        for metric_name, rows in self.diag_afferent_metric_rows.items():
            _store_sampled_history(metric_name, rows)

        return payload

    ########################################### COMPUTE ###########################################
    def compute_pcn_activations(self):
        """
        Uses current boundary- and HD-activations to update place-cell activations
        and store relevant data for analysis/debugging.
        """
        if self.use_unified_multiscale:
            # Compute and pass concatenated grid activations across scales.
            self.grid_activations_list = []
            self.pcn_activations_list = []
            position = self._current_planar_xy()
            prev_unified_acts = self.unified_pcn.place_cell_activations.detach()

            if self.gcn_scale_invariant:
                shared_gcn = next((g for g in self.gcns if g is not None), None)
                shared_acts = (
                    shared_gcn.get_grid_cell_activations(position, use_mask=True)
                    if shared_gcn is not None
                    else None
                )
                for gcn in self.gcns:
                    if gcn is not None and shared_acts is not None:
                        self.grid_activations_list.append(shared_acts)
                    else:
                        self.grid_activations_list.append(None)
            else:
                for gcn in self.gcns:
                    if gcn is not None:
                        self.grid_activations_list.append(
                            gcn.get_grid_cell_activations(position, use_mask=True)
                        )
                    else:
                        self.grid_activations_list.append(None)

            valid_grid = [g for g in self.grid_activations_list if g is not None]
            concatenated_grid = torch.cat(valid_grid, dim=0) if valid_grid else None
            if concatenated_grid is not None:
                expected_num_gc = int(getattr(self.unified_pcn, "num_grid_total", concatenated_grid.numel()))
                if int(concatenated_grid.numel()) != expected_num_gc:
                    if not hasattr(self, "_warned_grid_size_mismatch"):
                        self._warned_grid_size_mismatch = True
                        print(
                            f"[DRIVER] WARNING: Unified grid activation size mismatch "
                            f"(got={int(concatenated_grid.numel())}, expected={expected_num_gc}). "
                            f"Falling back to BVC-only input for this step."
                        )
                    concatenated_grid = None

            robust_distance = self._compute_proximity_distance()
            self.prox = float(robust_distance)
            self.unified_pcn.get_place_cell_activations(
                distances=self.boundaries,
                grid_activations=concatenated_grid,
                hd_activations=self.hd_activations,
                collided=torch.any(self.collided),
                proximity=robust_distance,
            )
            if (
                self.record_experience_transitions
                and hasattr(self, "unified_rcn")
                and hasattr(self.unified_rcn, "observe_transition")
            ):
                self.unified_rcn.observe_transition(
                    prev_unified_acts,
                    self.unified_pcn.place_cell_activations,
                )

            self.pcn_activations_list = self.unified_pcn.get_activations_per_scale()
            self._update_phase1_stability_metrics()
            self._record_scale_diagnostics()
            self.step(self.timestep)
            return

        # Get robot position for grid cell computation
        position = self._current_planar_xy()

        # Store grid cell activations
        self.grid_activations_list = []

        # Compute grid cell activations for each scale
        if self.gcn_scale_invariant:
            shared_gcn = next((g for g in self.gcns if g is not None), None)
            shared_acts = (
                shared_gcn.get_grid_cell_activations(position, use_mask=True)
                if shared_gcn is not None
                else None
            )
            for gcn in self.gcns:
                if gcn is not None and shared_acts is not None:
                    self.grid_activations_list.append(shared_acts)
                else:
                    self.grid_activations_list.append(None)
        else:
            for gcn in self.gcns:
                if gcn is not None:
                    # Get grid cell activations for current position
                    grid_activations = gcn.get_grid_cell_activations(position, use_mask=True)
                    self.grid_activations_list.append(grid_activations)
                else:
                    # If no grid cells for this scale, add None as placeholder
                    self.grid_activations_list.append(None)

        # Proximity diagnostic for non-unified mode as well.
        self.prox = self._compute_proximity_distance()

        # For convenience, store them in a list
        self.pcn_activations_list = []

        # For each scale's PCN, call get_place_cell_activations
        for i, pcn in enumerate(self.pcns):
            grid_input = self.grid_activations_list[i]

            pcn.get_place_cell_activations(
                distances=self.boundaries,
                grid_activations=grid_input,
                hd_activations=self.hd_activations,
                collided=torch.any(self.collided),
            )
            if self.plot_bvc:
                pcn.bvc_layer.plot_activation(
                    self.pcn.bvc_layer.plot_activation(self.boundaries.cpu())
                )
            # Append activations to pcn_activations_list
            act = pcn.place_cell_activations.clone().detach()
            self.pcn_activations_list.append(pcn.place_cell_activations)

        self._update_phase1_stability_metrics()
        self._record_scale_diagnostics()
        # Advance simulation one timestep
        self.step(self.timestep)

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        if getattr(self, "done", False) or getattr(self, "_trial_terminal_handled", False):
            return
        curr_pos = self.robot.getField("translation").getSFVec3f()
        time_limit = 120 # minutes

        hebb_multi_goal_coverage = (
            self.robot_mode == RobotMode.LEARN_HEBB
            and self.multi_goal_mode
            and hasattr(self, "goal_visit_counts")
            and hasattr(self, "coverage_grid")
        )

        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING, RobotMode.PLOTTING_AUTO) and not hebb_multi_goal_coverage:
            # Use trial elapsed time to avoid cumulative time issues in AUTO modes
            trial_elapsed_time = self.getTime() - self.trial_start_time
            if trial_elapsed_time >= 60 * self.run_time_minutes:
                self.stop()
                is_plotting_mode = self.robot_mode in (RobotMode.PLOTTING, RobotMode.PLOTTING_AUTO)
                self.save(include_pcn=not is_plotting_mode,
                        include_rcn=not is_plotting_mode,
                        include_gcn=not is_plotting_mode,
                        include_hmaps=True)
                self.done = True
                # Only pause if not in AUTO mode, or if this is the last trial
                if (self.robot_mode != RobotMode.PLOTTING_AUTO or
                    self.current_auto_trial == self.num_auto_trials):
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                return

        elif self.robot_mode == RobotMode.DMTP and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            self._current_planar_tensor(),
            atol=self.goal_r["explore"]
        ):
            self.stop()
            if self.use_unified_multiscale:
                self.unified_rcn.update_reward_cell_activations(
                    self.unified_pcn.place_cell_activations, visit=True
                )
                self.unified_rcn.replay(self.unified_pcn, use_scale_gate=True)
                self.save(include_pcn=True, include_rcn=True, include_gcn=False)
            else:
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                    rcn.replay(pcn=pcn)
                self.save(include_pcn=True, include_rcn=True, include_gcn=True)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return
        
        elif self.robot_mode == RobotMode.EXPLOIT:
            # Check if either goal reached or time expired
            goal_reached = torch.allclose(
                torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
                self._current_planar_tensor(),
                atol=self.goal_r["exploit"]
            )
            time_expired = self.getTime() >= 30 * time_limit
            
            if goal_reached or time_expired:
                if self.stats_collector:
                    # Update and save stats once
                    self.stats_collector.update_stat("trial_id", self.trial_id)
                    self.stats_collector.update_stat("start_location", self.start_loc)
                    self.stats_collector.update_stat("goal_location", self.goal_location)
                    self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
                    self.stats_collector.update_stat("total_time_secs", round(self.getTime(), 2))
                    self.stats_collector.update_stat("success", self.getTime() <= time_limit * 60)
                    self.stats_collector.save_stats(self.trial_id)
                    
                    # Print stats
                    print(f"Trial {self.trial_id} completed.")
                    print(f"Start location: {self.start_loc}")
                    print(f"Goal location: {self.goal_location}")
                    print(f"Total distance traveled: {round(self.compute_path_length(), 2)} meters.")
                    print(f"Total time taken: {round(self.getTime(), 2)} seconds.")
                    print(f"Success: {self.getTime() <= time_limit * 60}")
                    
                    self.stop()
                    self._reset_unified_heading_commit_state(reason="exploit_terminal")
                    self.save(save_trajectory=True)
                    self.done = True
                    return
                else:
                    self.stop()
                    self._reset_unified_heading_commit_state(reason="exploit_terminal")
                    if self.use_unified_multiscale:
                        self.unified_rcn.update_reward_cell_activations(
                            self.unified_pcn.place_cell_activations, visit=True
                        )
                        self.unified_rcn.replay(self.unified_pcn, use_scale_gate=True)
                        self.save(
                            include_pcn=True if self.td_learning else False,
                            include_rcn=True if self.td_learning else False,
                            include_gcn=False,
                        )
                    else:
                        for pcn, rcn in zip(self.pcns, self.rcns):
                            rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                            rcn.replay(pcn=pcn)
                        self.save(
                            include_pcn=True if self.td_learning else False,
                            include_rcn=True if self.td_learning else False,
                            include_gcn=True if self.td_learning else False,
                        )
                    self.done = True
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                    return

        elif hebb_multi_goal_coverage:
            # LEARN_HEBB multi-goal completion gate: require BOTH coverage and per-goal learning criteria.
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = self._current_planar_tensor()
            counting_allowed = self._two_phase_allows_counting()
            if not counting_allowed:
                if self.step_count % self.training_log_interval_steps == 0 and self.step_count > 0:
                    print(
                        f"[TRAIN-2P][PHASE1] step={self.step_count} "
                        f"LEARN_HEBB counters/termination disabled until phase2."
                    )
                return

            for goal in self.goals:
                goal_name = goal["name"]
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)

                if distance <= goal["radius"]:
                    if not goal["visited"]:
                        print(f"[LEARN_HEBB] First visit to {goal['name']} goal at {goal['location']}")
                        goal["visited"] = True
                    if not self.goal_currently_in[goal_name]:
                        steps_since_last = self.step_count - self.goal_last_count_step[goal_name]
                        if steps_since_last >= self.goal_visit_cooldown_steps:
                            self.goal_visit_counts[goal_name] += 1
                            self.goal_last_count_step[goal_name] = int(self.step_count)
                            self.goal_currently_in[goal_name] = True
                            print(f"[LEARN_HEBB] {goal_name} visit #{self.goal_visit_counts[goal_name]}")
                        else:
                            # Entered again too soon: treat as same visit episode.
                            self.goal_currently_in[goal_name] = True
                    if self._should_defer_unified_goal_associations():
                        self._record_goal_contact_step(goal_name)
                    # Potentiate edge: last crossed checkpoint -> this goal
                    _last_crossed = getattr(self, "checkpoint_last_crossed", None)
                    if _last_crossed is not None:
                        _lc_cp, _lc_dir = _last_crossed
                        _goal_edges = getattr(self, "checkpoint_to_goal_edges", None)
                        if _goal_edges is not None:
                            _gk = (int(_lc_cp), str(_lc_dir), str(goal_name))
                            _goal_edges[_gk] = _goal_edges.get(_gk, 0) + 1
                    if not self._should_defer_unified_goal_associations():
                        self._handle_goal_learning(goal)
                else:
                    if (
                        self.goal_currently_in[goal_name]
                        and distance > (goal["radius"] + self.goal_exit_hysteresis)
                    ):
                        self.goal_currently_in[goal_name] = False

            self._update_checkpoint_visit_counts(current_position)
            self._update_checkpoint_pc_groups(current_position)

            coverage_reached = self._check_coverage_complete()
            learning_complete = self._check_multi_goal_learning_complete()

            if coverage_reached and learning_complete:
                if self._should_defer_unified_goal_associations():
                    learning_complete = self._finalize_deferred_goal_associations()
                    if not learning_complete:
                        print(
                            "[LEARN_HEBB] Deferred goal associations are still incomplete; continuing training."
                        )
                        return
                print(f"[LEARN_HEBB] Coverage + learning complete. "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")
                self.stop()
                self._create_multi_goal_reward_maps()
                self._save_multi_goal_data()
                self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                self.done = True
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            elif trial_elapsed_time >= 60 * self.run_time_minutes:
                print(f"[LEARN_HEBB] Time limit reached but criteria unmet; continuing. "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Learning complete: {learning_complete}")

        elif self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_ADAPTIVE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = self._current_planar_tensor()

            # Two-phase coverage mode: phase 1 waits for low drift before enabling
            # goal/checkpoint counting and reward seeding. Strict LEARN_LOCATIONS_TWO_PHASE
            # additionally performs the OJAS -> STDP handoff at this point.
            if self.two_phase_learning and self.phase2_start_step is None:
                if self._phase1_is_stable():
                    self._start_phase2_stdp()
                elif self.step_count % self.training_log_interval_steps == 0 and self.step_count > 0:
                    status = self._phase1_last_stability_status or self._get_phase1_stability_status()
                    print(
                        f"[TRAIN-2P][PHASE1] step={self.step_count} "
                        f"revisit_bins={int(status['revisit_bins'])}/{self.phase1_min_revisit_bins}, "
                        f"revisit_frac={float(status['revisit_fraction']):.2f}/"
                        f"{self.phase1_revisit_fraction_threshold:.2f}, "
                        f"bin_cos={float(status['bin_cos_mean']):.4f}/"
                        f"{self.phase1_revisit_cosine_threshold:.3f}, "
                        f"p{int(self.phase1_revisit_low_percentile)}="
                        f"{float(status['bin_cos_low']):.4f}/"
                        f"{self.phase1_revisit_low_percentile_threshold:.3f}, "
                        f"streak={self._phase1_stability_streak}/"
                        f"{self.phase1_stability_required_streak} "
                        f"| counting=DISABLED"
                    )
                return

            self._maybe_stop_ojas_on_stabilization()
            counting_allowed = self._two_phase_allows_counting()

            # Check all goals for learning
            for goal in self.goals:
                goal_name = goal["name"]
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)

                if distance <= goal["radius"]:
                    if counting_allowed:
                        # Track first visit (for backward compatibility)
                        if not goal["visited"]:
                            print(f"[LEARN_LOCATIONS_COVERAGE] First visit to {goal['name']} goal at {goal['location']}")
                            goal["visited"] = True

                        # Increment visit count when entering goal (not already in it)
                        if not self.goal_currently_in[goal_name]:
                            steps_since_last = self.step_count - self.goal_last_count_step[goal_name]
                            if steps_since_last >= self.goal_visit_cooldown_steps:
                                self.goal_visit_counts[goal_name] += 1
                                self.goal_last_count_step[goal_name] = int(self.step_count)
                                self.goal_currently_in[goal_name] = True
                                print(f"[LEARN_LOCATIONS_COVERAGE] {goal_name} visit #{self.goal_visit_counts[goal_name]}")
                            else:
                                # Entered again too soon: treat as same visit episode.
                                self.goal_currently_in[goal_name] = True
                        else:
                            self.goal_currently_in[goal_name] = True

                        if self._should_defer_unified_goal_associations():
                            self._record_goal_contact_step(goal_name)
                        # Potentiate edge: last crossed checkpoint -> this goal
                        _last_crossed = getattr(self, "checkpoint_last_crossed", None)
                        if _last_crossed is not None:
                            _lc_cp, _lc_dir = _last_crossed
                            _goal_edges = getattr(self, "checkpoint_to_goal_edges", None)
                            if _goal_edges is not None:
                                _gk = (int(_lc_cp), str(_lc_dir), str(goal_name))
                                _goal_edges[_gk] = _goal_edges.get(_gk, 0) + 1
                        if not self._should_defer_unified_goal_associations():
                            self._handle_goal_learning(goal)
                else:
                    # Mark that robot has left this goal zone
                    if (
                        self.goal_currently_in[goal_name]
                        and distance > (goal["radius"] + self.goal_exit_hysteresis)
                    ):
                        self.goal_currently_in[goal_name] = False

            self._update_checkpoint_visit_counts(current_position)
            self._update_checkpoint_pc_groups(current_position)

            # Check termination conditions: (time limit OR coverage reached) AND
            # learning complete AND configured minimum duration elapsed.
            minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
            coverage_reached = self._check_coverage_complete()
            learning_complete = self._check_multi_goal_learning_complete()
            minimum_duration_reached = (
                trial_elapsed_time >= self.minimum_learning_duration_seconds
            )
            if not counting_allowed:
                minimum_time_reached = False
                coverage_reached = False
                learning_complete = False
                minimum_duration_reached = False

            if (minimum_time_reached or coverage_reached) and learning_complete and minimum_duration_reached:
                if self._should_defer_unified_goal_associations():
                    learning_complete = self._finalize_deferred_goal_associations()
                    if not learning_complete:
                        print(
                            "[LEARN_LOCATIONS_COVERAGE] Deferred goal associations are still incomplete; continuing training."
                        )
                        return
                reason = "Coverage target reached" if coverage_reached else "Time limit reached"
                print(f"[LEARN_LOCATIONS_COVERAGE] {reason} and learning complete! "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")
                self.stop()
                finalize_start = time.perf_counter()

                # In two-phase mode, build path-transition graph after STDP phase from
                # stored PC trajectories to avoid laggy per-step transition updates.
                if (
                    self.two_phase_learning
                    and self.defer_experience_build_until_phase2_end
                    and self.use_unified_multiscale
                ):
                    stage_start = time.perf_counter()
                    self._build_experience_transitions_from_hmaps()
                    print(
                        f"[LEARN_LOCATIONS_COVERAGE] Deferred transition build finished in "
                        f"{time.perf_counter() - stage_start:.2f}s"
                    )

                # Filter checkpoint visit log to remove dead-end traversals
                stage_start = time.perf_counter()
                self._compute_valid_checkpoints()
                print(
                    f"[LEARN_LOCATIONS_COVERAGE] Checkpoint filtering finished in "
                    f"{time.perf_counter() - stage_start:.2f}s"
                )

                # Create reward maps for each goal
                stage_start = time.perf_counter()
                self._create_multi_goal_reward_maps()
                print(
                    f"[LEARN_LOCATIONS_COVERAGE] Reward-map build finished in "
                    f"{time.perf_counter() - stage_start:.2f}s"
                )
                stage_start = time.perf_counter()
                self._save_multi_goal_data()
                print(
                    f"[LEARN_LOCATIONS_COVERAGE] Goal-association save finished in "
                    f"{time.perf_counter() - stage_start:.2f}s"
                )

                # Save trial completion time for AUTO mode
                if self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO:
                    self._save_trial_completion_time(trial_elapsed_time)

                stage_start = time.perf_counter()
                self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                print(
                    f"[LEARN_LOCATIONS_COVERAGE] Final network/hmap save finished in "
                    f"{time.perf_counter() - stage_start:.2f}s"
                )
                print(
                    f"[LEARN_LOCATIONS_COVERAGE] Total finalize time: "
                    f"{time.perf_counter() - finalize_start:.2f}s"
                )
                self.done = True

                # Only pause if not in AUTO mode, or if this is the last trial
                if (self.robot_mode != RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO or
                    self.current_auto_trial == self.num_auto_trials):
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            elif coverage_reached and not learning_complete:
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                          f"but learning incomplete, continuing...")
                    self.last_coverage_incomplete_log_step = self.step_count
            elif minimum_time_reached and not learning_complete:
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Time limit reached but learning incomplete, continuing...")
                    self.last_coverage_incomplete_log_step = self.step_count
            elif (minimum_time_reached or coverage_reached) and learning_complete and not minimum_duration_reached:
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    remaining = max(0.0, self.minimum_learning_duration_seconds - trial_elapsed_time)
                    print(f"[LEARN_LOCATIONS_COVERAGE] Ready to finish but waiting for minimum duration "
                          f"({remaining/60:.1f} min remaining)...")
                    self.last_coverage_incomplete_log_step = self.step_count

        elif self.robot_mode == RobotMode.PLOTTING_COVERAGE_AUTO:
            # Coverage-based stopping for plotting mode (no learning, just exploration until coverage target)
            trial_elapsed_time = self.getTime() - self.trial_start_time

            # Check termination conditions: coverage reached OR time limit as fallback
            coverage_reached = self._check_coverage_complete()
            minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes

            if coverage_reached or minimum_time_reached:
                reason = "Coverage target reached" if coverage_reached else "Fallback time limit reached"
                print(f"[PLOTTING_COVERAGE_AUTO] {reason}! "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")
                self.stop()

                # Save only hmaps (no network saving in plotting mode)
                self.save(include_pcn=False, include_rcn=False, include_gcn=False, include_hmaps=True)
                self.done = True

                # Only pause if this is the last trial
                if self.current_auto_trial == self.num_auto_trials:
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                return

        elif self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = self._current_planar_tensor()

            # Check only active goal for exploitation
            active_goals = [g for g in self.goals if g.get("active", False)]
            for goal in active_goals:
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)

                if distance <= goal["radius"]:
                    self._handle_random_goal_exploitation(goal)
                    return

            # Check distance-based termination condition
            if (hasattr(self, 'optimal_path_distance') and self.optimal_path_distance and
                hasattr(self, 'path_failure_ratio') and self.path_failure_ratio):
                path_ratio = self.total_distance_traveled / self.optimal_path_distance

                if path_ratio >= self.path_failure_ratio:
                    print(f"[EXPLOIT_RANDOM] Path failure ratio reached: {path_ratio:.2f} "
                          f"(traveled: {self.total_distance_traveled:.1f}m, optimal: {self.optimal_path_distance:.1f}m)")
                    self._handle_random_exploitation_timeout()
                    return

            # Check time limit as fallback
            if self.step_count > 10 and trial_elapsed_time >= 60 * self.run_time_minutes:
                print(f"[EXPLOIT_RANDOM] Fallback time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
                self._handle_random_exploitation_timeout()
                return

    ##########################################################################
    #                           GOAL HANDLING METHODS                        #
    ##########################################################################

    def _handle_goal_learning(self, goal):
        """Handle goal visits during learning mode - keep best place cell by connection strength"""
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            for scale_idx, scale_acts in enumerate(self.pcn_activations_list):
                most_active_idx = self._select_goal_association_pc(
                    scale_acts=scale_acts,
                    goal_name=goal["name"],
                    scale_idx=scale_idx,
                )
                if most_active_idx is None:
                    continue
                activation_value = scale_acts[most_active_idx].item()

                if activation_value <= 0.01:
                    continue

                stored_idx = self.goal_place_cell_associations[goal["name"]][scale_idx]

                if stored_idx is None:
                    self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                    self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                    self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
                    continue

                if stored_idx == most_active_idx:
                    continue

                stored_activation = self.goal_place_cell_activations[goal["name"]][scale_idx]
                activation_ratio = activation_value / (stored_activation + 1e-6)

                if activation_ratio > (1.0 + self.ACTIVATION_SIMILARITY_THRESHOLD):
                    self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                    self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                    self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
                elif activation_ratio < (1.0 - self.ACTIVATION_SIMILARITY_THRESHOLD):
                    pass
                else:
                    old_strength = self._compute_place_cell_connection_strength(self.unified_pcn, stored_idx, scale_idx)
                    new_strength = self._compute_place_cell_connection_strength(self.unified_pcn, most_active_idx, scale_idx)
                    if new_strength >= old_strength:
                        self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                        self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                        self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
            return

        for scale_idx, pcn in enumerate(self.pcns):
            # Find most active place cell for this scale
            most_active_idx = self._select_goal_association_pc(
                scale_acts=pcn.place_cell_activations,
                goal_name=goal["name"],
                scale_idx=scale_idx,
            )
            if most_active_idx is None:
                continue
            activation_value = pcn.place_cell_activations[most_active_idx].item()

            # Only consider if there's meaningful activation
            if activation_value > 0.01:
                stored_idx = self.goal_place_cell_associations[goal["name"]][scale_idx]

                # Case 1: First association (no stored place cell yet)
                if stored_idx is None:
                    self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                    self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                    self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
                    print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                          f"First PC association -> PC {most_active_idx} (activation: {activation_value:.3f})")

                # Case 2: Same place cell as before - keep it
                elif stored_idx == most_active_idx:
                    # No need to update or compare
                    pass

                # Case 3: Different place cell - hybrid comparison (activation-primary, connection-secondary)
                else:
                    stored_activation = self.goal_place_cell_activations[goal["name"]][scale_idx]

                    # Debug: Print that we're entering comparison
                    if not hasattr(self, '_debug_comparison_started'):
                        self._debug_comparison_started = True
                        print(f"\n[DEBUG] Starting hybrid place cell comparison for {goal['name']} scale {scale_idx}")
                        print(f"[DEBUG] Stored PC {stored_idx} (act: {stored_activation:.3f}) vs Active PC {most_active_idx} (act: {activation_value:.3f})\n")

                    # Step 1: Compare activations first
                    activation_ratio = activation_value / (stored_activation + 1e-6)

                    # If new activation is significantly better (>20% higher)
                    if activation_ratio > (1.0 + self.ACTIVATION_SIMILARITY_THRESHOLD):
                        self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                        self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                        self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
                        print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                              f"PC {stored_idx} -> PC {most_active_idx} "
                              f"(activation: {stored_activation:.3f} -> {activation_value:.3f}, +{(activation_ratio-1)*100:.1f}%)")

                    # If old activation is significantly better (>20% higher)
                    elif activation_ratio < (1.0 - self.ACTIVATION_SIMILARITY_THRESHOLD):
                        print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                              f"Keeping PC {stored_idx} (activation: {stored_activation:.3f} > {activation_value:.3f}, {(1-activation_ratio)*100:.1f}% better)")

                    # Step 2: Activations are similar - use connection strength as tiebreaker
                    else:
                        old_strength = self._compute_place_cell_connection_strength(pcn, stored_idx, scale_idx)
                        new_strength = self._compute_place_cell_connection_strength(pcn, most_active_idx, scale_idx)

                        # Use recency bias: if new is >= old, take new
                        if new_strength >= old_strength:
                            self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                            self.goal_association_step[goal["name"]][scale_idx] = self.step_count
                            self.goal_place_cell_activations[goal["name"]][scale_idx] = activation_value
                            print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                                  f"PC {stored_idx} -> PC {most_active_idx} "
                                  f"(similar activation, strength: {old_strength:.1f} -> {new_strength:.1f})")
                        else:
                            print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                                  f"Keeping PC {stored_idx} (similar activation, strength: {old_strength:.1f} > {new_strength:.1f})")

    def _compute_place_cell_connection_strength(self, pcn, pc_idx, scale_idx=0):
        """
        Compute place cell quality based on recurrent connection strength.

        Args:
            pcn: Place cell network
            pc_idx: Index of place cell to evaluate

        Returns:
            float: Total recurrent connection strength
        """
        if self.use_unified_multiscale and hasattr(pcn, "w_rec_unified"):
            start = pcn.scale_boundaries[scale_idx]
            end = pcn.scale_boundaries[scale_idx + 1]
            global_idx = start + int(pc_idx)
            w_rec = pcn.w_rec_unified
            outgoing = torch.sum(torch.abs(w_rec[:, global_idx, start:end])).item()
            incoming = torch.sum(torch.abs(w_rec[:, start:end, global_idx])).item()
            return outgoing + incoming

        # w_rec_tripartite shape: (n_hd, num_pc, num_pc)
        # where w_rec[hd, i, j] is connection from PC i to PC j for head direction hd

        # Try multiple metrics to see which gives non-zero values:

        # 1. Outgoing connections (from pc_idx to all others) - use absolute value
        outgoing = torch.sum(torch.abs(pcn.w_rec_tripartite[:, pc_idx, :])).item()

        # 2. Incoming connections (from all others to pc_idx) - use absolute value
        incoming = torch.sum(torch.abs(pcn.w_rec_tripartite[:, :, pc_idx])).item()

        # 3. Total connectivity (bidirectional)
        total_connectivity = outgoing + incoming

        # 4. Max connection strength
        max_connection = torch.max(torch.abs(pcn.w_rec_tripartite[:, pc_idx, :])).item()

        # 5. Number of strong connections
        strong_connection_count = torch.sum(torch.abs(pcn.w_rec_tripartite[:, pc_idx, :]) > 0.01).item()

        # Debug: print first time we see this to verify calculations
        if not hasattr(self, '_debug_connection_printed'):
            self._debug_connection_printed = True
            print(f"\n[DEBUG] Connection metrics for PC {pc_idx}:")
            print(f"  Outgoing: {outgoing:.6f}")
            print(f"  Incoming: {incoming:.6f}")
            print(f"  Total: {total_connectivity:.6f}")
            print(f"  Max: {max_connection:.6f}")
            print(f"  Strong count: {strong_connection_count}")
            print(f"  w_rec shape: {pcn.w_rec_tripartite.shape}")
            print(f"  w_rec min/max: {pcn.w_rec_tripartite.min().item():.6f} / {pcn.w_rec_tripartite.max().item():.6f}\n")

        # Use total bidirectional connectivity
        return total_connectivity

    def _update_checkpoint_visit_counts(self, current_position: torch.Tensor):
        """Update directed checkpoint crossing counters from doorway traversals."""
        if (
            not self.detected_doorways
            or not hasattr(self, "checkpoint_visit_counts")
            or not hasattr(self, "checkpoint_crossing_counts")
        ):
            return
        if self.min_checkpoint_visits <= 0:
            return

        curr_x = float(current_position[0].item())
        curr_y = float(current_position[1].item())
        for checkpoint_idx, (cx, cy) in enumerate(self.detected_doorways):
            checkpoint_support_region, _ = self._infer_checkpoint_support_region(
                float(cx),
                float(cy),
            )
            distance = math.sqrt(
                self._support_region_distance_sq(
                    curr_x,
                    curr_y,
                    checkpoint_support_region,
                    default_point=(float(cx), float(cy)),
                )
            )
            side = 0
            frame = self._support_region_frame(checkpoint_support_region)
            if frame is not None:
                cx0, cy0 = frame["center"]
                nx, ny = frame["normal"]
                signed = ((curr_x - cx0) * nx) + ((curr_y - cy0) * ny)
                side_eps = max(0.05, float(frame.get("half_width", 0.0)) + 0.03)
                if signed > side_eps:
                    side = 1
                elif signed < -side_eps:
                    side = -1

            if distance <= self.checkpoint_visit_radius:
                if not self.checkpoint_currently_in.get(checkpoint_idx, False):
                    self.checkpoint_currently_in[checkpoint_idx] = True
                    entry_side = (
                        self.checkpoint_last_clear_side.get(checkpoint_idx, 0)
                        if self.checkpoint_last_clear_side.get(checkpoint_idx, 0) != 0
                        else side
                    )
                    self.checkpoint_entry_side[checkpoint_idx] = int(entry_side)
                    self.checkpoint_crossed_centerline[checkpoint_idx] = False
                    if hasattr(self, "checkpoint_crossing_step_candidate"):
                        self.checkpoint_crossing_step_candidate[checkpoint_idx] = -1
                entry_side = int(self.checkpoint_entry_side.get(checkpoint_idx, 0))
                if entry_side != 0 and side != 0 and side != entry_side:
                    if not bool(
                        self.checkpoint_crossed_centerline.get(checkpoint_idx, False)
                    ):
                        if hasattr(self, "checkpoint_crossing_step_candidate"):
                            self.checkpoint_crossing_step_candidate[checkpoint_idx] = int(
                                self.step_count
                            )
                    self.checkpoint_crossed_centerline[checkpoint_idx] = True
            else:
                if (
                    self.checkpoint_currently_in.get(checkpoint_idx, False)
                    and distance > (self.checkpoint_visit_radius + self.goal_exit_hysteresis)
                ):
                    exit_side = side
                    entry_side = int(self.checkpoint_entry_side.get(checkpoint_idx, 0))
                    crossed_centerline = bool(
                        self.checkpoint_crossed_centerline.get(checkpoint_idx, False)
                    )
                    if (
                        entry_side != 0
                        and exit_side != 0
                        and exit_side != entry_side
                        and crossed_centerline
                    ):
                        steps_since_last = self.step_count - self.checkpoint_last_count_step.get(
                            checkpoint_idx, -10**9
                        )
                        if steps_since_last >= self.goal_visit_cooldown_steps:
                            dir_key = "neg_to_pos" if entry_side < exit_side else "pos_to_neg"
                            dir_counts = self.checkpoint_crossing_counts.setdefault(
                                checkpoint_idx,
                                {"neg_to_pos": 0, "pos_to_neg": 0},
                            )
                            dir_counts[dir_key] = int(dir_counts.get(dir_key, 0)) + 1
                            self.checkpoint_visit_counts[checkpoint_idx] = int(
                                dir_counts.get("neg_to_pos", 0) + dir_counts.get("pos_to_neg", 0)
                            )
                            self.checkpoint_last_count_step[checkpoint_idx] = int(self.step_count)
                            candidate_step = int(
                                getattr(
                                    self,
                                    "checkpoint_crossing_step_candidate",
                                    {},
                                ).get(checkpoint_idx, -1)
                            )
                            crossing_step = (
                                candidate_step if candidate_step >= 0 else int(self.step_count)
                            )
                            self._record_checkpoint_crossing_step(
                                checkpoint_idx=checkpoint_idx,
                                direction_key=dir_key,
                                step=crossing_step,
                            )
                            # Potentiate edge: last distinct checkpoint -> this checkpoint
                            _last_crossed = getattr(self, "checkpoint_last_crossed", None)
                            if _last_crossed is not None:
                                _lc_cp, _lc_dir = _last_crossed
                                if int(_lc_cp) != int(checkpoint_idx):
                                    _edge_key = (int(_lc_cp), str(_lc_dir), int(checkpoint_idx), str(dir_key))
                                    _cp_edges = getattr(self, "checkpoint_to_checkpoint_edges", None)
                                    if _cp_edges is not None:
                                        _cp_edges[_edge_key] = _cp_edges.get(_edge_key, 0) + 1
                            self.checkpoint_last_crossed = (int(checkpoint_idx), str(dir_key))
                            print(
                                f"[CHECKPOINT] checkpoint_{checkpoint_idx} crossing "
                                f"{'-+' if dir_key == 'neg_to_pos' else '+-'} "
                                f"#{dir_counts[dir_key]} "
                                f"(total={self.checkpoint_visit_counts[checkpoint_idx]})"
                            )
                    self.checkpoint_currently_in[checkpoint_idx] = False
                    self.checkpoint_entry_side[checkpoint_idx] = 0
                    self.checkpoint_crossed_centerline[checkpoint_idx] = False
                    if hasattr(self, "checkpoint_crossing_step_candidate"):
                        self.checkpoint_crossing_step_candidate[checkpoint_idx] = -1
                    if exit_side != 0:
                        self.checkpoint_last_clear_side[checkpoint_idx] = int(exit_side)
                elif not self.checkpoint_currently_in.get(checkpoint_idx, False) and side != 0:
                    self.checkpoint_last_clear_side[checkpoint_idx] = int(side)

    def _update_checkpoint_pc_groups(self, current_position: torch.Tensor):
        """Track which PCs fire near each checkpoint during learning.

        Instead of geometric side assignment, we accumulate a proximity signal:
        which PCs are active when the agent is near each checkpoint.  These
        become the "checkpoint PC cluster" used later to partition the
        recurrent-weight graph into rooms.
        """
        if not self.detected_doorways or not hasattr(self, "unified_pcn"):
            return

        # Lazy initialisation.
        if not hasattr(self, "checkpoint_proximity_pcs") or not self.checkpoint_proximity_pcs:
            num_pc = int(self.unified_pcn.num_pc_total)
            self.checkpoint_proximity_pcs = {}
            for idx in range(len(self.detected_doorways)):
                self.checkpoint_proximity_pcs[idx] = torch.zeros(
                    num_pc, dtype=torch.float32, device=self.device
                )

        activations = self.unified_pcn.place_cell_activations.detach()
        active_mask = activations > 0.01
        if not active_mask.any():
            return

        curr_x = float(current_position[0].item())
        curr_y = float(current_position[1].item())

        proximity_radius = float(getattr(self, "checkpoint_visit_radius", 0.8)) * 2.0
        for ck_idx, (cx, cy) in enumerate(self.detected_doorways):
            dist = math.hypot(curr_x - float(cx), curr_y - float(cy))
            if dist < proximity_radius:
                self.checkpoint_proximity_pcs[ck_idx] = torch.max(
                    self.checkpoint_proximity_pcs[ck_idx], activations
                )

    def _build_checkpoint_boundary_masks(
        self,
        percentile: float = 80.0,
        device=None,
    ) -> Dict[int, torch.Tensor]:
        """Build per-checkpoint boundary masks using scale-relative thresholds.

        For each checkpoint and each scale, a cell is part of the boundary if
        its max observed activation exceeds the ``percentile``-th percentile of
        all nonzero max activations in that scale's population at that checkpoint.
        This prevents large-scale cells (which fire more strongly on average)
        from dominating the boundary while small-scale cells are excluded.

        Returns:
            Dict mapping checkpoint index → bool tensor (num_pc_total,).
        """
        proximity_pcs = getattr(self, "checkpoint_proximity_pcs", {})
        if not proximity_pcs or not hasattr(self, "unified_pcn"):
            return {}
        scale_boundaries = list(self.unified_pcn.scale_boundaries)
        target_device = device or self.device
        boundary_masks: Dict[int, torch.Tensor] = {}

        for cp_idx, max_act in proximity_pcs.items():
            max_act = max_act.to(target_device)
            mask = torch.zeros_like(max_act, dtype=torch.bool)

            for s in range(len(scale_boundaries) - 1):
                start = int(scale_boundaries[s])
                end = int(scale_boundaries[s + 1])
                scale_slice = max_act[start:end]
                if scale_slice.numel() == 0:
                    continue
                # Only consider cells that had *some* activation (>0)
                active = scale_slice[scale_slice > 0]
                if active.numel() == 0:
                    continue
                threshold = float(
                    torch.quantile(active, percentile / 100.0).item()
                )
                mask[start:end] = scale_slice >= threshold

            boundary_masks[int(cp_idx)] = mask
            count = int(torch.count_nonzero(mask).item())
            print(
                f"[BOUNDARY] cp{cp_idx}: {count} boundary cells "
                f"(p{percentile:.0f} scale-relative)"
            )

        return boundary_masks

    def _checkpoint_support_side_from_point(
        self,
        checkpoint_idx: int,
        point_x: float,
        point_y: float,
        fallback_side: int = 0,
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[dict]]:
        """Return the signed side of a point relative to one checkpoint doorway."""
        if (
            checkpoint_idx < 0
            or checkpoint_idx >= len(getattr(self, "detected_doorways", []))
        ):
            return int(fallback_side), None, None

        cx, cy = self.detected_doorways[checkpoint_idx]
        support_region, _ = self._infer_checkpoint_support_region(float(cx), float(cy))
        frame = self._support_region_frame(support_region)
        if frame is None:
            return int(fallback_side), support_region, None

        frame_cx, frame_cy = frame["center"]
        nx, ny = frame["normal"]
        signed = ((float(point_x) - frame_cx) * nx) + ((float(point_y) - frame_cy) * ny)
        side_eps = max(0.05, float(frame.get("half_width", 0.0)) + 0.03)
        if signed > side_eps:
            return 1, support_region, frame
        if signed < -side_eps:
            return -1, support_region, frame
        return int(fallback_side), support_region, frame

    @staticmethod
    def _checkpoint_group_partition_is_valid(
        group_a: torch.Tensor,
        group_b: torch.Tensor,
    ) -> bool:
        """Connectivity room groups must be substantive, non-empty, and disjoint."""
        mask_a = torch.as_tensor(group_a).view(-1) > 0.0
        mask_b = torch.as_tensor(group_b).view(-1) > 0.0
        if not bool(torch.any(mask_a).item()) or not bool(torch.any(mask_b).item()):
            return False
        count_a = int(torch.count_nonzero(mask_a).item())
        count_b = int(torch.count_nonzero(mask_b).item())
        overlap = mask_a & mask_b
        if bool(torch.any(overlap).item()):
            return False
        union_count = int(torch.count_nonzero(mask_a | mask_b).item())
        smaller = int(min(count_a, count_b))
        if smaller < 8:
            return False
        if union_count > 0 and (float(smaller) / float(union_count)) < 0.005:
            return False
        return True

    def _build_checkpoint_room_partition_seed_set(
        self,
        checkpoint_idx: int,
        proximity_vec: Optional[torch.Tensor],
        centers: Optional[torch.Tensor],
        visited: Optional[torch.Tensor],
    ) -> set:
        """
        Build a doorway bottleneck seed set for room partitioning.

        A strict proximity-only cutoff can leave too few checkpoint PCs to cut
        the recurrent graph at the doorway. Combining a looser proximity mask
        with a small geometry-defined doorway band makes the bottleneck much
        more reliable.
        """
        num_pc = int(self.unified_pcn.num_pc_total)
        seed_ids: set = set()

        prox = None
        prox_max = 0.0
        if proximity_vec is not None:
            prox = torch.as_tensor(
                proximity_vec, dtype=torch.float32, device=self.device
            ).view(-1)
            if prox.numel() == num_pc:
                prox_max = float(torch.max(prox).item())
                if prox_max > 0.0:
                    prox_mask = prox >= (prox_max * 0.05)
                    seed_ids.update(
                        int(idx)
                        for idx in torch.nonzero(
                            prox_mask, as_tuple=False
                        ).view(-1).tolist()
                    )

        if (
            centers is not None
            and visited is not None
            and 0 <= checkpoint_idx < len(self.detected_doorways)
        ):
            cx, cy = self.detected_doorways[checkpoint_idx]
            support_region, _ = self._infer_checkpoint_support_region(
                float(cx), float(cy)
            )
            # The doorway cut must be wide enough to sever all paths
            # between rooms.  Small-scale cells (sigma_r ≈ 0.5 m) have
            # dense overlapping fields, so a narrow radius leaves
            # alternative paths around the removed PCs.  Using 1.5×
            # the visit radius covers the full doorway width.
            doorway_radius = max(
                0.8,
                1.5 * float(getattr(self, "checkpoint_visit_radius", 0.8)),
            )
            dist_sq = self._support_region_distance_sq_torch(
                centers[:, 0],
                centers[:, 1],
                support_region,
                default_point=(float(cx), float(cy)),
            )
            doorway_mask = visited & (dist_sq <= (doorway_radius * doorway_radius))
            seed_ids.update(
                int(idx)
                for idx in torch.nonzero(
                    doorway_mask, as_tuple=False
                ).view(-1).tolist()
            )

        min_seed_nodes = int(
            max(8, getattr(self, "checkpoint_group_min_seed_nodes", 16))
        )
        if prox is not None and prox_max > 0.0 and len(seed_ids) < min_seed_nodes:
            positive_idx = torch.nonzero(prox > 0.0, as_tuple=False).view(-1)
            if positive_idx.numel() > 0:
                k = min(int(positive_idx.numel()), min_seed_nodes)
                top_local = torch.topk(prox[positive_idx], k=k).indices
                seed_ids.update(
                    int(positive_idx[idx].item()) for idx in top_local.tolist()
                )

        return {int(idx) for idx in seed_ids if 0 <= int(idx) < num_pc}

    def _build_room_groups_from_geometry(self):
        """Partition PCs into rooms using wall geometry and checkpoint positions.

        Checkpoints are **doorways** — they sit at gaps in walls.  We block
        those gaps, then flood-fill the remaining free space to discover the
        enclosed rooms on each side.

        Algorithm:
        1. Build a 2D occupancy grid from wall rectangles.
        2. Block each checkpoint doorway (small patch around the checkpoint).
        3. Connected-component flood-fill on remaining free cells → each
           component is a room.
        4. Assign every visited PC to the room containing its center.
        5. Build ``checkpoint_pc_groups`` and ``_room_partition_data``.

        Works for all environments (1-6).
        """
        if not hasattr(self, "unified_pcn"):
            return

        world_name = getattr(self, "world_name", None)
        if world_name is None:
            print("[ROOM-GEOM] No world_name attribute — skipping geometry partition")
            return

        # --- 1. Gather checkpoint and goal positions ---
        checkpoint_positions = get_env_checkpoint_positions(world_name)
        goal_pos = get_env_goal_pos(world_name)

        if not checkpoint_positions:
            checkpoint_positions = list(getattr(self, "detected_doorways", []))

        if goal_pos is None:
            for g in getattr(self, "goals", []):
                loc = g.get("location")
                if loc is not None:
                    goal_pos = (float(loc[0]), float(loc[1]))
                    break

        # --- 2. Build occupancy grid from wall geometry ---
        try:
            config = get_world_config(world_name)
        except ValueError:
            print(f"[ROOM-GEOM] No world config for {world_name} — skipping")
            return

        env_size = config.get("size", [20.0, 20.0])
        half_w, half_h = env_size[0] / 2.0, env_size[1] / 2.0
        resolution = 200
        cell_size = env_size[0] / resolution

        def world_to_grid(wx, wy):
            col = max(0, min(resolution - 1, int((wx + half_w) / cell_size)))
            row = max(0, min(resolution - 1, int((wy + half_h) / cell_size)))
            return row, col

        # occupancy: True = free, False = blocked
        occupancy = np.ones((resolution, resolution), dtype=bool)

        obstacles = config.get("obstacles", [])
        for obs in obstacles:
            if obs.get("type") != "rectangle":
                continue
            bounds = obs["bounds"]
            x_min, y_min = bounds[0]
            x_max, y_max = bounds[1]
            col_min = max(0, int((x_min + half_w) / cell_size) - 1)
            col_max = min(resolution - 1, int((x_max + half_w) / cell_size) + 1)
            row_min = max(0, int((y_min + half_h) / cell_size) - 1)
            row_max = min(resolution - 1, int((y_max + half_h) / cell_size) + 1)
            occupancy[row_min:row_max + 1, col_min:col_max + 1] = False

        # --- 3. Block doorways at checkpoint positions ---
        # The blocked patch must span the entire doorway gap to seal it.
        # Checkpoints aren't always centered in the gap (e.g. cp2 at x=-0.6
        # in a gap spanning x=-2.0 to x=-0.1), so we need a radius large
        # enough to bridge from the checkpoint to the nearest wall on each side.
        # The widest gaps in environments 1-6 are ~2m; 1.5m radius covers them.
        doorway_block_radius = max(3, int(1.5 / cell_size))  # ~1.5 m
        checkpoint_grid_cells: Dict[int, List[Tuple[int, int]]] = {}
        for cp_idx, (cx, cy) in enumerate(checkpoint_positions):
            cr, cc = world_to_grid(float(cx), float(cy))
            blocked_cells = []
            # Only block cells that are currently FREE.  Cells that are
            # already walls stay as walls — this prevents the blocking
            # circle from tunneling through a wall into an adjacent room
            # (e.g. cp3's circle extending past wall4 into the top-left).
            # We also flood-fill from the checkpoint center through free
            # cells only, so the block stays within the doorway gap.
            block_visited = set()
            block_queue = deque([(cr, cc)])
            while block_queue:
                br, bc = block_queue.popleft()
                if (br, bc) in block_visited:
                    continue
                dr, dc = br - cr, bc - cc
                if dr * dr + dc * dc > doorway_block_radius * doorway_block_radius:
                    continue
                if not (0 <= br < resolution and 0 <= bc < resolution):
                    continue
                block_visited.add((br, bc))
                if not occupancy[br, bc]:
                    continue  # Hit a wall — don't cross it
                occupancy[br, bc] = False
                blocked_cells.append((br, bc))
                for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nbr, nbc = br + ddr, bc + ddc
                    if (nbr, nbc) not in block_visited:
                        block_queue.append((nbr, nbc))
            checkpoint_grid_cells[cp_idx] = blocked_cells

        # --- 4. Flood-fill to find connected components (rooms) ---
        room_grid = np.full((resolution, resolution), -1, dtype=np.int32)
        num_rooms = 0

        for r in range(resolution):
            for c in range(resolution):
                if not occupancy[r, c] or room_grid[r, c] >= 0:
                    continue
                # New room — BFS from this cell
                room_id = num_rooms
                num_rooms += 1
                room_grid[r, c] = room_id
                bfs_queue = deque([(r, c)])
                while bfs_queue:
                    cr, cc = bfs_queue.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < resolution and 0 <= nc < resolution:
                            if occupancy[nr, nc] and room_grid[nr, nc] == -1:
                                room_grid[nr, nc] = room_id
                                bfs_queue.append((nr, nc))

        print(f"[ROOM-GEOM] Flood-fill found {num_rooms} rooms in {world_name}")

        # --- 5. Determine which rooms each checkpoint connects ---
        # Temporarily unblock doorway cells and check which room IDs border them.
        checkpoint_adjacent_rooms: Dict[int, List[int]] = {}
        for cp_idx, blocked_cells in checkpoint_grid_cells.items():
            adjacent_rooms: set = set()
            for br, bc in blocked_cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = br + dr, bc + dc
                    if 0 <= nr < resolution and 0 <= nc < resolution:
                        rid = int(room_grid[nr, nc])
                        if rid >= 0:
                            adjacent_rooms.add(rid)
            checkpoint_adjacent_rooms[cp_idx] = sorted(adjacent_rooms)

        # --- 6. Assign PCs to rooms ---
        device = self.device
        num_pc = int(self.unified_pcn.num_pc_total)
        centers, visited, spatial_log = self._get_unified_pc_spatial_support(device)
        if centers is None or visited is None:
            print("[ROOM-GEOM] No spatial support available — skipping")
            return

        component_ids = torch.full((num_pc,), -1, dtype=torch.long, device=device)
        centers_np = centers.detach().cpu().numpy()

        for pc_idx in range(num_pc):
            if not bool(visited[pc_idx].item()):
                continue
            px, py = float(centers_np[pc_idx, 0]), float(centers_np[pc_idx, 1])
            row, col = world_to_grid(px, py)
            rid = int(room_grid[row, col])
            if rid >= 0:
                component_ids[pc_idx] = rid
            else:
                # PC center lands on a wall/doorway — find nearest free cell
                best_rid = -1
                best_d = float("inf")
                search_r = 5
                for dr in range(-search_r, search_r + 1):
                    for dc in range(-search_r, search_r + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < resolution and 0 <= nc < resolution:
                            nrid = int(room_grid[nr, nc])
                            if nrid >= 0:
                                d = dr * dr + dc * dc
                                if d < best_d:
                                    best_d = d
                                    best_rid = nrid
                if best_rid >= 0:
                    component_ids[pc_idx] = best_rid

        # --- 7. Build checkpoint_pc_groups ---
        self.checkpoint_pc_groups = {}

        # Room membership sets
        room_members: Dict[int, set] = {i: set() for i in range(num_rooms)}
        for pc_idx in range(num_pc):
            cid = int(component_ids[pc_idx].item())
            if cid >= 0:
                room_members[cid].add(pc_idx)

        for cp_idx, adj_rooms in checkpoint_adjacent_rooms.items():
            group_a = torch.zeros(num_pc, dtype=torch.float32, device=device)
            group_b = torch.zeros(num_pc, dtype=torch.float32, device=device)

            if len(adj_rooms) >= 2:
                cp_x, cp_y = float(checkpoint_positions[cp_idx][0]), float(checkpoint_positions[cp_idx][1])
                side_a_rooms = []
                side_b_rooms = []

                for adj_rid in adj_rooms:
                    pcs = sorted(room_members.get(adj_rid, set()))
                    if not pcs:
                        continue
                    adj_centers = centers_np[pcs]
                    mean_x = float(np.mean(adj_centers[:, 0]))
                    mean_y = float(np.mean(adj_centers[:, 1]))
                    dx = mean_x - cp_x
                    dy = mean_y - cp_y
                    if abs(dx) >= abs(dy):
                        side = 1 if dx >= 0 else -1
                    else:
                        side = 1 if dy >= 0 else -1
                    if side >= 0:
                        side_a_rooms.append(adj_rid)
                    else:
                        side_b_rooms.append(adj_rid)

                if not side_a_rooms or not side_b_rooms:
                    side_a_rooms = adj_rooms[:len(adj_rooms) // 2]
                    side_b_rooms = adj_rooms[len(adj_rooms) // 2:]

                for rid in side_a_rooms:
                    for pc in room_members.get(rid, set()):
                        group_a[pc] = 1.0
                for rid in side_b_rooms:
                    for pc in room_members.get(rid, set()):
                        group_b[pc] = 1.0
            elif len(adj_rooms) == 1:
                for pc in room_members.get(adj_rooms[0], set()):
                    group_a[pc] = 1.0

            self.checkpoint_pc_groups[cp_idx] = {
                "group_a": group_a,
                "group_b": group_b,
            }
            n_a = int(group_a.sum().item())
            n_b = int(group_b.sum().item())
            print(
                f"[ROOM-GEOM] cp{cp_idx}: group_a={n_a} group_b={n_b} "
                f"adjacent_rooms={adj_rooms}"
            )

        # --- 8. Build _room_partition_data ---
        # Room adjacency graph: two rooms are connected if a checkpoint
        # borders both of them.
        room_graph: Dict[int, set] = {}
        checkpoint_components: Dict[int, List[int]] = {}
        for cp_idx, adj_rooms in checkpoint_adjacent_rooms.items():
            checkpoint_components[cp_idx] = adj_rooms
            for i in range(len(adj_rooms)):
                for j in range(i + 1, len(adj_rooms)):
                    room_graph.setdefault(adj_rooms[i], set()).add(adj_rooms[j])
                    room_graph.setdefault(adj_rooms[j], set()).add(adj_rooms[i])

        self._room_partition_data = {
            "component_ids": component_ids,
            "visited": visited.to(device),
            "checkpoint_components": checkpoint_components,
            "graph": {k: sorted(v) for k, v in room_graph.items()},
            "num_components": num_rooms,
        }

        room_sizes = [len(room_members.get(i, set())) for i in range(num_rooms)]
        print(
            f"[ROOM-GEOM] Geometry partition: {num_rooms} rooms "
            f"(sizes: {room_sizes}) | checkpoints: {len(checkpoint_positions)} "
            f"| {spatial_log}"
        )

    def _build_room_groups_from_connectivity(self):
        """Partition PCs into rooms using learned recurrent connectivity.

        Uses the recurrent weight matrix W_rec as a directed graph.  Checkpoint
        PCs (identified by proximity accumulation during learning) form a
        bottleneck.  Removing them and flood-filling the remaining graph yields
        connected components that correspond to rooms.

        Each checkpoint then stores two groups (``group_a``, ``group_b``) of
        PC indices.  At exploit time, we check which group contains the goal's
        associated PCs to decide which group to mask.
        """
        if not self.detected_doorways or not hasattr(self, "unified_pcn"):
            return
        proximity_pcs = getattr(self, "checkpoint_proximity_pcs", {})
        if not proximity_pcs:
            return

        device = self.device
        num_pc = int(self.unified_pcn.num_pc_total)
        centers, visited, spatial_log = self._get_unified_pc_spatial_support(device)

        # 1) Build direction-agnostic adjacency from current recurrent weights.
        # Oja's normalization weakens connections that are no longer reinforced,
        # so the final w_rec reflects where cells are now, not where they were
        # during early training.
        w_rec = self.unified_pcn.w_rec_unified  # (n_hd, num_pc, num_pc)
        adj_max, _ = w_rec.max(dim=0)  # (num_pc, num_pc)
        # Symmetrise so connectivity is bidirectional for flood-fill.
        adj_sym = torch.max(adj_max, adj_max.T)
        adj_global_max = float(adj_sym.max().item())
        if adj_global_max <= 0.0:
            return
        # Per-scale thresholds: each scale's connections are evaluated
        # relative to its own weight range.  Small-scale cells have
        # inherently weaker weights than large-scale cells, so a single
        # global threshold crushes small-scale connectivity.
        thresh_frac = 0.01
        adj_binary = torch.zeros(num_pc, num_pc, dtype=torch.bool, device=device)
        scale_thresholds = []
        for s in range(self.unified_pcn.num_scales):
            s_start = int(self.unified_pcn.scale_boundaries[s])
            s_end = int(self.unified_pcn.scale_boundaries[s + 1])
            block = adj_sym[s_start:s_end, s_start:s_end]
            block_max = float(block.max().item())
            thresh = block_max * thresh_frac if block_max > 0 else 0.0
            scale_thresholds.append(thresh)
            adj_binary[s_start:s_end, s_start:s_end] = (block > thresh)
        # Cross-scale blocks: use the smaller scale's threshold so that
        # weaker cross-scale edges from small-field cells are preserved.
        for s1 in range(self.unified_pcn.num_scales):
            for s2 in range(s1 + 1, self.unified_pcn.num_scales):
                s1s = int(self.unified_pcn.scale_boundaries[s1])
                s1e = int(self.unified_pcn.scale_boundaries[s1 + 1])
                s2s = int(self.unified_pcn.scale_boundaries[s2])
                s2e = int(self.unified_pcn.scale_boundaries[s2 + 1])
                cross_thresh = min(scale_thresholds[s1], scale_thresholds[s2])
                block_12 = adj_sym[s1s:s1e, s2s:s2e]
                block_21 = adj_sym[s2s:s2e, s1s:s1e]
                adj_binary[s1s:s1e, s2s:s2e] = (block_12 > cross_thresh)
                adj_binary[s2s:s2e, s1s:s1e] = (block_21 > cross_thresh)
        # Zero diagonal (no self-loops).
        adj_binary.fill_diagonal_(False)
        threshold_log = ", ".join(
            f"s{i}={t:.4f}" for i, t in enumerate(scale_thresholds)
        )
        print(f"[ROOM-GROUPS] Adjacency source: w_rec_unified (per-scale thresholds: {threshold_log})")

        # 2) For each checkpoint, identify its PC cluster from proximity mass.
        self.checkpoint_pc_groups = {}
        all_checkpoint_pcs = set()
        checkpoint_pc_sets = {}
        for ck_idx in range(len(self.detected_doorways)):
            cp_indices = self._build_checkpoint_room_partition_seed_set(
                checkpoint_idx=ck_idx,
                proximity_vec=proximity_pcs.get(ck_idx),
                centers=centers,
                visited=visited,
            )
            if not cp_indices:
                continue
            checkpoint_pc_sets[ck_idx] = cp_indices
            all_checkpoint_pcs.update(cp_indices)

        if not checkpoint_pc_sets:
            return

        # 3) Remove checkpoint PCs from the graph and BFS to find components.
        #    Work on CPU for set/BFS operations.
        adj_cpu = adj_binary.cpu()
        remaining = set(range(num_pc)) - all_checkpoint_pcs
        if visited is not None:
            visited_ids = set(
                int(idx)
                for idx in torch.nonzero(
                    visited, as_tuple=False
                ).view(-1).tolist()
            )
            remaining &= visited_ids
        # Only keep PCs that have at least one connection.
        connected_remaining = set()
        for pc in remaining:
            if adj_cpu[pc].any():
                connected_remaining.add(pc)
        remaining = connected_remaining

        # BFS — find connected components among remaining PCs.
        visited_set: set = set()
        components: List[set] = []
        for seed in remaining:
            if seed in visited_set:
                continue
            component: set = set()
            queue = [seed]
            while queue:
                node = queue.pop()
                if node in visited_set:
                    continue
                visited_set.add(node)
                component.add(node)
                # Neighbours in remaining only (skip checkpoint PCs).
                neighbours = torch.nonzero(adj_cpu[node], as_tuple=False).view(-1).tolist()
                for nb in neighbours:
                    if nb not in visited_set and nb in remaining:
                        queue.append(nb)
            if component:
                components.append(component)

        if not components:
            return

        print(
            f"[ROOM-GROUPS] Found {len(components)} connected components "
            f"(sizes: {sorted([len(c) for c in components], reverse=True)[:10]}) "
            f"| {spatial_log}"
        )

        # 4) For each checkpoint, find the two largest components that connect
        #    to it (i.e. that have edges to checkpoint PCs).
        for ck_idx, cp_pcs in checkpoint_pc_sets.items():
            # Score each component by how many edges it has to this checkpoint's PCs.
            comp_scores = []
            for comp_idx, comp in enumerate(components):
                edge_count = 0
                for cp_pc in cp_pcs:
                    neighbours = set(torch.nonzero(adj_cpu[cp_pc], as_tuple=False).view(-1).tolist())
                    edge_count += len(neighbours & comp)
                if edge_count > 0:
                    comp_scores.append((comp_idx, edge_count, len(comp)))

            if not comp_scores:
                continue

            # Sort by edge count descending, then prefer one attached component
            # from each side of the checkpoint separator when spatial support
            # is available.
            comp_scores.sort(key=lambda t: (-t[1], -t[2]))

            group_a = torch.zeros(num_pc, dtype=torch.float32, device=device)
            group_b = torch.zeros(num_pc, dtype=torch.float32, device=device)
            assigned = False

            _, _, frame = self._checkpoint_support_side_from_point(
                ck_idx,
                self.detected_doorways[ck_idx][0],
                self.detected_doorways[ck_idx][1],
                fallback_side=0,
            )
            side_score_map = {}
            if frame is not None and centers is not None:
                frame_cx, frame_cy = frame["center"]
                nx, ny = frame["normal"]
                signed = (
                    ((centers[:, 0] - frame_cx) * nx)
                    + ((centers[:, 1] - frame_cy) * ny)
                ).detach().cpu()
                doorway_eps = max(
                    0.05,
                    float(frame.get("half_width", 0.0))
                    + float(getattr(self, "goal_map_checkpoint_directional_band", 0.12)),
                )
                for comp_idx, edge_count, comp_size in comp_scores:
                    comp_nodes = sorted(int(pc) for pc in components[comp_idx])
                    if not comp_nodes:
                        continue
                    comp_signed = signed[comp_nodes]
                    off_axis = torch.abs(comp_signed) > doorway_eps
                    if bool(torch.any(off_axis).item()):
                        comp_signed = comp_signed[off_axis]
                    if comp_signed.numel() == 0:
                        continue
                    side = 1 if float(torch.mean(comp_signed).item()) >= 0.0 else -1
                    side_score = (
                        int(edge_count),
                        int(comp_size),
                        float(torch.mean(torch.abs(comp_signed)).item()),
                    )
                    best = side_score_map.get(side)
                    if best is None or side_score > best[0]:
                        side_score_map[side] = (side_score, comp_idx)

            if side_score_map:
                pos_entry = side_score_map.get(1)
                neg_entry = side_score_map.get(-1)
                if pos_entry is not None:
                    for pc in components[pos_entry[1]]:
                        group_a[pc] = 1.0
                if neg_entry is not None:
                    for pc in components[neg_entry[1]]:
                        group_b[pc] = 1.0
                assigned = bool(pos_entry is not None and neg_entry is not None)

            if not assigned and len(comp_scores) >= 2:
                for pc in components[comp_scores[0][0]]:
                    group_a[pc] = 1.0
                for pc in components[comp_scores[1][0]]:
                    group_b[pc] = 1.0
            elif not assigned and len(comp_scores) == 1:
                # Only one connected room found — put it in group_a.
                for pc in components[comp_scores[0][0]]:
                    group_a[pc] = 1.0

            self.checkpoint_pc_groups[ck_idx] = {
                "group_a": group_a,
                "group_b": group_b,
            }
            n_a = int(group_a.sum().item())
            n_b = int(group_b.sum().item())
            n_cp = len(cp_pcs)
            print(
                f"[ROOM-GROUPS] cp{ck_idx}: group_a={n_a} group_b={n_b} "
                f"checkpoint_pcs={n_cp}"
            )

        # 5) Build and store the full room partition for exploit-time use.
        component_ids = torch.full((num_pc,), -1, dtype=torch.long, device=device)
        for comp_idx, comp in enumerate(components):
            for pc in comp:
                component_ids[int(pc)] = int(comp_idx)

        # Build room adjacency graph: two components are adjacent if a
        # checkpoint's PCs connect to both.
        checkpoint_components: Dict[int, List[int]] = {}
        room_graph: Dict[int, set] = {}
        for ck_idx, cp_pcs in checkpoint_pc_sets.items():
            adjacent_comps: set = set()
            for cp_pc in cp_pcs:
                neighbours = set(torch.nonzero(adj_cpu[cp_pc], as_tuple=False).view(-1).tolist())
                for nb in neighbours:
                    c = int(component_ids[nb].item())
                    if c >= 0:
                        adjacent_comps.add(c)
            checkpoint_components[int(ck_idx)] = sorted(adjacent_comps)
            # Every pair of components adjacent to the same checkpoint is connected
            adj_list = sorted(adjacent_comps)
            for i in range(len(adj_list)):
                for j in range(i + 1, len(adj_list)):
                    room_graph.setdefault(adj_list[i], set()).add(adj_list[j])
                    room_graph.setdefault(adj_list[j], set()).add(adj_list[i])

        self._room_partition_data = {
            "component_ids": component_ids,
            "visited": visited.to(device) if visited is not None else torch.ones(num_pc, dtype=torch.bool, device=device),
            "checkpoint_components": checkpoint_components,
            "graph": {k: sorted(v) for k, v in room_graph.items()},
            "num_components": len(components),
        }
        print(
            f"[ROOM-GROUPS] Stored room partition: {len(components)} components, "
            f"graph_edges={sum(len(v) for v in room_graph.values()) // 2}"
        )

    def _ensure_checkpoint_pc_groups_loaded(self):
        """No-op: room partition is now derived from replay boundaries.

        Kept for backward compatibility with callers that expect this method
        to exist.  The old geometry-based partition is no longer needed.
        """
        pass

    def _get_unified_checkpoint_room_partition(
        self,
        target_device,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Room partition is now derived from replay boundaries.

        Returns None — the old geometry/connectivity-based partition is no
        longer used.  Replay-driven room discovery happens inside
        ``_build_paper_room_local_goal_map`` using checkpoint boundary masks.
        """
        self._ensure_checkpoint_pc_groups_loaded()
        return None, "roomown=replay_boundaries(no_precomputed_partition)"

    def _resolve_goal_component_from_partition(
        self,
        room_partition: Dict[str, Any],
        goal_x: float,
        goal_z: float,
        target_device,
    ) -> int:
        """Find the room component that contains the goal location."""
        component_ids = room_partition.get("component_ids")
        if component_ids is None:
            return -1
        centers, visited, _ = self._get_unified_pc_spatial_support(target_device)
        if centers is None or visited is None:
            return -1
        goal_t = torch.tensor(
            [float(goal_x), float(goal_z)],
            dtype=torch.float32,
            device=target_device,
        )
        dist_sq = torch.sum((centers - goal_t.unsqueeze(0)) ** 2, dim=1)
        comp_ids_t = component_ids.to(target_device)
        # Only consider visited PCs with a valid component
        valid = visited & (comp_ids_t >= 0)
        if not bool(torch.any(valid).item()):
            return -1
        dist_sq = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))
        nearest_pc = int(torch.argmin(dist_sq).item())
        return int(comp_ids_t[nearest_pc].item())

    def _room_graph_distances_from_goal(
        self,
        graph: Dict[int, List[int]],
        goal_component: int,
    ) -> Dict[int, float]:
        """BFS over the room adjacency graph to get distances from the goal room."""
        distances: Dict[int, float] = {int(goal_component): 0.0}
        queue = [int(goal_component)]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            current_dist = distances[current]
            for neighbor in graph.get(current, []):
                neighbor = int(neighbor)
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1.0
                    queue.append(neighbor)
        return distances

    def _load_saved_multi_goal_learning_state(
        self,
        hydrate_dense_hmaps: bool = False,
    ) -> bool:
        """Restore saved goal/checkpoint learning state needed for reward-map rebuilds."""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        assoc_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        if not os.path.exists(assoc_path):
            print(f"[REBUILD] Missing saved goal associations: {assoc_path}")
            return False

        try:
            with open(assoc_path, "rb") as f:
                assoc_data = pickle.load(f)
        except Exception as exc:
            print(f"[REBUILD] Failed to load goal associations from {assoc_path}: {exc}")
            return False

        saved_goals = assoc_data.get("goals", None)
        if isinstance(saved_goals, list) and saved_goals:
            active_names = {
                str(goal.get("name"))
                for goal in getattr(self, "goals", [])
                if bool(goal.get("active", False))
            }
            restored_goals = []
            for saved_goal in saved_goals:
                restored_goal = dict(saved_goal)
                restored_goal["visited"] = bool(
                    assoc_data.get("goal_visit_counts", {}).get(
                        str(saved_goal.get("name")),
                        restored_goal.get("visited", False),
                    )
                )
                restored_goal["active"] = str(saved_goal.get("name")) in active_names
                restored_goals.append(restored_goal)
            self.goals = restored_goals
            self.multi_goal_mode = len(self.goals) > 1

        goal_names = [str(goal["name"]) for goal in getattr(self, "goals", [])]
        default_per_goal = {goal_name: [None] * len(self.scales) for goal_name in goal_names}
        default_steps = {goal_name: [] for goal_name in goal_names}
        default_counts = {goal_name: 0 for goal_name in goal_names}

        self.goal_place_cell_associations = assoc_data.get(
            "goal_place_cell_associations",
            default_per_goal,
        )
        self.goal_association_step = assoc_data.get(
            "goal_association_step",
            default_per_goal,
        )
        self.goal_place_cell_activations = assoc_data.get(
            "goal_place_cell_activations",
            default_per_goal,
        )
        self.goal_contact_steps = assoc_data.get(
            "goal_contact_steps",
            default_steps,
        )
        self.goal_visit_counts = assoc_data.get(
            "goal_visit_counts",
            default_counts,
        )
        self.goal_currently_in = {goal_name: False for goal_name in goal_names}
        self.goal_last_count_step = {goal_name: -10**9 for goal_name in goal_names}

        self.checkpoint_visit_counts = assoc_data.get("checkpoint_visit_counts", {})
        self.checkpoint_crossing_counts = assoc_data.get("checkpoint_crossing_counts", {})
        self.checkpoint_crossing_steps = assoc_data.get("checkpoint_crossing_steps", {})
        self.checkpoint_to_checkpoint_edges = assoc_data.get("checkpoint_to_checkpoint_edges", {})
        self.checkpoint_to_goal_edges = assoc_data.get("checkpoint_to_goal_edges", {})
        self.phase2_start_step = assoc_data.get("phase2_start_step", self.phase2_start_step)
        self.adaptive_stdp_start_step = assoc_data.get(
            "adaptive_stdp_start_step",
            self.adaptive_stdp_start_step,
        )
        self.goal_associations_finalized = bool(
            assoc_data.get("goal_associations_finalized", True)
        )
        self.defer_goal_association_until_training_end = bool(
            assoc_data.get(
                "defer_goal_association_until_training_end",
                getattr(self, "defer_goal_association_until_training_end", False),
            )
        )

        saved_groups = assoc_data.get("checkpoint_pc_groups", {})
        if saved_groups:
            self.checkpoint_pc_groups = {
                int(k): {
                    side: (
                        t.to(self.device)
                        if isinstance(t, torch.Tensor)
                        else torch.as_tensor(t, dtype=self.dtype, device=self.device)
                    )
                    for side, t in v.items()
                }
                for k, v in saved_groups.items()
            }
        saved_proximity = assoc_data.get("checkpoint_proximity_pcs", {})
        if saved_proximity:
            self.checkpoint_proximity_pcs = {
                int(k): (
                    t.to(self.device)
                    if isinstance(t, torch.Tensor)
                    else torch.as_tensor(t, dtype=self.dtype, device=self.device)
                )
                for k, t in saved_proximity.items()
            }

        saved_total_steps = int(max(0, assoc_data.get("total_steps", 0)))
        if saved_total_steps > 0:
            self.step_count = max(int(self.step_count), saved_total_steps)

        if hydrate_dense_hmaps:
            source = self._get_unified_dense_hmap_source()
            if source is not None:
                self.hmap_pcn_activities = [
                    acts.clone().to(self.device)
                    if isinstance(acts, torch.Tensor)
                    else torch.as_tensor(acts, dtype=self.dtype, device=self.device)
                    for acts in source.get("scale_acts", [])
                ]
                n_valid = int(source.get("n_valid", 0))
                if n_valid > 0:
                    self.step_count = max(int(self.step_count), n_valid)
                print(f"[REBUILD] Hydrated dense hmaps from {source.get('log', 'unknown')}")

        print(
            f"[REBUILD] Restored saved multi-goal state from {assoc_path} "
            f"(goals={len(goal_names)}, total_steps={self.step_count})"
        )
        return True

    def rebuild_reward_maps_from_saved_state(self) -> bool:
        """Rebuild goal/checkpoint reward maps from the current saved training state."""
        print("[REBUILD] Starting reward-map rebuild from saved state")
        restore_ok = self._load_saved_multi_goal_learning_state(
            hydrate_dense_hmaps=bool(
                self._should_defer_unified_goal_associations()
                or (
                    self.two_phase_learning
                    and self.defer_experience_build_until_phase2_end
                    and self.use_unified_multiscale
                )
            )
            and not bool(getattr(self, "lightweight_hmaps", False)),
        )
        if not restore_ok:
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return False

        rebuild_start = time.perf_counter()

        stage_start = time.perf_counter()
        self._compute_valid_checkpoints()
        print(
            f"[REBUILD] Checkpoint filtering finished in "
            f"{time.perf_counter() - stage_start:.2f}s"
        )

        stage_start = time.perf_counter()
        self._create_multi_goal_reward_maps()
        print(
            f"[REBUILD] Reward-map build finished in "
            f"{time.perf_counter() - stage_start:.2f}s"
        )

        stage_start = time.perf_counter()
        self._save_multi_goal_data()
        print(
            f"[REBUILD] Goal-association save finished in "
            f"{time.perf_counter() - stage_start:.2f}s"
        )

        print(
            f"[REBUILD] Total rebuild time: "
            f"{time.perf_counter() - rebuild_start:.2f}s"
        )
        self.done = True
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
        return True

    def _check_multi_goal_learning_complete(self):
        """Check if multi-goal learning is complete"""
        # All goals must be visited
        if not all(goal["visited"] for goal in self.goals):
            return False

        if not self._goal_associations_ready_for_completion():
            return False

        # All goals must have minimum number of visits
        for goal_name, visit_count in self.goal_visit_counts.items():
            if visit_count < self.min_goal_visits:
                return False
        # All checkpoints must have a few visits before training can end.
        if self.min_checkpoint_visits > 0 and self.detected_doorways:
            for checkpoint_idx in range(len(self.detected_doorways)):
                dir_counts = self.checkpoint_crossing_counts.get(
                    checkpoint_idx,
                    {"neg_to_pos": 0, "pos_to_neg": 0},
                )
                if (
                    int(dir_counts.get("neg_to_pos", 0)) < self.min_checkpoint_visits
                    or int(dir_counts.get("pos_to_neg", 0)) < self.min_checkpoint_visits
                ):
                    return False

        return True

    # ── Hierarchical sub-goal navigation ─────────────────────────────────────────────────

    def _load_checkpoints_from_scene(self):
        """
        Scan the Webots scene tree for Goal-proto objects whose name starts with
        'checkpoint_'. Extract their (x, y) positions (hmap_loc[:, 0/1] convention)
        and store in self.detected_doorways, sorted by the numeric suffix.
        Safe to call before any world node exists — silently skips on failure.
        """
        try:
            root = self.getRoot()
            children = root.getField("children")
            n = children.getCount()
        except Exception:
            print("[CHECKPOINT] Could not access scene tree; no checkpoints loaded.")
            return

        raw = []
        for i in range(n):
            try:
                node = children.getMFNode(i)
            except Exception:
                continue
            if node is None:
                continue
            name_field = node.getField("name")
            if name_field is None:
                continue
            name = name_field.getSFString()
            if not name.startswith("checkpoint_"):
                continue
            try:
                num = int(name.split("_")[1])
            except (IndexError, ValueError):
                num = 9999
            trans = node.getField("translation").getSFVec3f()
            raw.append((num, float(trans[0]), float(trans[1])))
            print(f"[CHECKPOINT] Found {name} at ({trans[0]:.2f}, {trans[1]:.2f})")

        raw.sort(key=lambda t: t[0])
        self.detected_doorways = [(x, y) for (_, x, y) in raw]
        print(f"[CHECKPOINT] Loaded {len(self.detected_doorways)} checkpoints from scene")

    def _compute_valid_checkpoints(self):
        """
        Mark all scene checkpoints as valid candidates for the replay boost.

        Dead-end filtering is handled organically by the experience transition
        matrix: checkpoints the replay wave never reaches (because the agent
        never walked through them toward the goal) are never boosted, regardless
        of being listed here. A stack-based bounce filter was previously used but
        over-cancelled junctions that were visited during dead-end exploration
        before later being traversed on the correct goal path.
        """
        self._valid_checkpoints = set(range(len(self.detected_doorways)))
        print(f"[CHECKPOINT] All {len(self._valid_checkpoints)} checkpoints marked valid; "
              f"dead-end filtering delegated to experience transition matrix")

    def _build_unified_seed_from_position(
        self,
        goal_x,
        goal_z,
        rcn_device,
        seed_sigma: Optional[float] = None,
        center_radius_factor: Optional[float] = None,
        center_min_radius: Optional[float] = None,
        support_region=None,
    ):
        """
        Build one unified replay seed from the nearest learned place-cell centers.

        This is intentionally narrow. It does not average over all visit-history
        activations near the target. Instead, for each scale, it picks the nearest
        visited place cell(s) to the target support region and seeds only that
        compact set. This mirrors targeted reactivation of the specific place
        representation associated with the subgoal.

        Falls back to the visit-history method when place-field centers are not
        yet available (e.g. very early in training).
        """
        seed_activations = torch.zeros(self.unified_pcn.num_pc_total, device=rcn_device)

        # Try to get learned place-field centers.
        centers, visited_cells, spatial_log = self._get_unified_pc_spatial_support(rcn_device)
        if centers is None or visited_cells is None or not torch.any(visited_cells):
            return self._build_unified_seed_from_position_visit_history(
                goal_x=goal_x,
                goal_z=goal_z,
                rcn_device=rcn_device,
                seed_sigma=seed_sigma,
                support_region=support_region,
            )

        centers_x = centers[:, 0].to(device=rcn_device, dtype=torch.float32)
        centers_y = centers[:, 1].to(device=rcn_device, dtype=torch.float32)
        dist_sq_all = self._support_region_distance_sq_torch(
            centers_x,
            centers_y,
            support_region=support_region,
            default_point=(float(goal_x), float(goal_z)),
        )
        dist_sq_all = torch.where(
            visited_cells,
            dist_sq_all,
            torch.full_like(dist_sq_all, 1e12),
        )
        best_idx = int(torch.argmin(dist_sq_all).item())
        best_dist = float(torch.sqrt(torch.clamp(dist_sq_all[best_idx], min=0.0)).item())
        if not torch.isfinite(dist_sq_all[best_idx]) or best_dist >= 1e6:
            return self._build_unified_seed_from_position_visit_history(
                goal_x=goal_x,
                goal_z=goal_z,
                rcn_device=rcn_device,
                seed_sigma=seed_sigma,
                support_region=support_region,
            )

        seed_activations[best_idx] = 1.0
        chosen_scale = 0
        for s_idx in range(len(self.scales)):
            s_start = int(self.unified_pcn.scale_boundaries[s_idx])
            s_end = int(self.unified_pcn.scale_boundaries[s_idx + 1])
            if s_start <= best_idx < s_end:
                chosen_scale = int(s_idx)
                break

        seed_source = (
            "seed=nearest_place_cell_center "
            f"idx={best_idx},scale={chosen_scale},dist={best_dist:.2f}m"
        )

        # If no cells were close enough at any scale, fall back to visit history.
        if float(torch.max(torch.abs(seed_activations)).item()) <= 1e-9:
            return self._build_unified_seed_from_position_visit_history(
                goal_x=goal_x,
                goal_z=goal_z,
                rcn_device=rcn_device,
                seed_sigma=seed_sigma,
                support_region=support_region,
            )

        return seed_activations, seed_source

    def _build_unified_seed_from_position_visit_history(
        self,
        goal_x,
        goal_z,
        rcn_device,
        seed_sigma: Optional[float] = None,
        support_region=None,
    ):
        """
        Fallback seed builder using Gaussian-weighted visit history.

        Used when place-field centers are not yet available (early training).
        """
        seed_activations = torch.zeros(self.unified_pcn.num_pc_total, device=rcn_device)
        hmap_ok = (
            bool(self.hmap_pcn_activities)
            and self.step_count > 0
            and not getattr(self, "lightweight_hmaps", False)
        )
        if not hmap_ok:
            return seed_activations, "seed=no_dense_hmaps"

        n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
        hmap_x = torch.from_numpy(self.hmap_loc[:n_valid, 0]).float().to(rcn_device)
        hmap_z = torch.from_numpy(self.hmap_loc[:n_valid, 1]).float().to(rcn_device)
        dist_sq_all = self._support_region_distance_sq_torch(
            hmap_x,
            hmap_z,
            support_region=support_region,
            default_point=(float(goal_x), float(goal_z)),
        )

        sigma = float(max(0.25, seed_sigma if seed_sigma is not None else 0.8))
        weights = torch.exp(-dist_sq_all / (2.0 * sigma ** 2))
        weight_sum = float(weights.sum().item())

        if weight_sum <= 1e-12:
            nearest_idx = int(torch.argmin(dist_sq_all).item())
            weights = torch.zeros_like(dist_sq_all)
            weights[nearest_idx] = 1.0
            weight_sum = 1.0
            seed_source = f"seed=visit_fallback_nearest({nearest_idx})"
        else:
            seed_source = f"seed=visit_fallback_gauss(sigma={sigma:.2f}m)"
        if support_region is not None:
            if support_region.get("kind") == "segment":
                p0 = support_region["p0"]
                p1 = support_region["p1"]
                seed_source += (
                    f" | doorway={support_region.get('orientation', 'segment')}"
                    f"(p0=({p0[0]:.2f},{p0[1]:.2f}),"
                    f"p1=({p1[0]:.2f},{p1[1]:.2f}),"
                    f"half_w={support_region.get('half_width', 0.0):.2f})"
                )

        weights = weights / max(weight_sum, 1e-12)
        scale_logs = []
        for s_idx in range(len(self.scales)):
            s_start = self.unified_pcn.scale_boundaries[s_idx]
            s_end = self.unified_pcn.scale_boundaries[s_idx + 1]
            if s_idx >= len(self.hmap_pcn_activities):
                continue
            acts = self.hmap_pcn_activities[s_idx][:n_valid].float().to(rcn_device)
            seed_slice = torch.mv(acts.T, weights)
            seed_activations[s_start:s_end] = seed_slice
            n_active = int((seed_slice > 1e-6).sum().item())
            scale_logs.append(f"scale{s_idx}:{n_active}seeded")

        if scale_logs:
            seed_source = seed_source + " | " + ", ".join(scale_logs)
        return seed_activations, seed_source

    def _relay_support_indices(
        self,
        seed_activations: torch.Tensor,
        topk: int = 12,
        threshold_fraction: float = 0.35,
    ) -> List[int]:
        """Return a compact support set for one goal/checkpoint seed vector."""
        vec = torch.as_tensor(seed_activations).detach().float().cpu().view(-1).abs()
        if vec.numel() == 0:
            return []
        peak = float(torch.max(vec).item())
        if peak <= 1e-9:
            return []
        idx = torch.nonzero(vec >= (threshold_fraction * peak), as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            k = min(int(max(1, topk)), int(vec.numel()))
            idx = torch.topk(vec, k=k).indices
        elif idx.numel() > topk:
            keep = torch.topk(vec[idx], k=int(max(1, topk))).indices
            idx = idx[keep]
        return [int(i) for i in idx.tolist()]

    def _loop_erase_compact_segment(
        self,
        compact_positions: List[int],
        compact_dom_ids: List[int],
    ) -> List[int]:
        """
        Loop-erase one compact dominant-state segment.

        This keeps a direct first-passage trunk between segment boundaries rather
        than preserving all random-walk wandering that happened inside the
        segment. It is intentionally stronger than the global short-window
        denoiser because segmented checkpoint replay should follow the main
        arrival trunk into the downstream subgoal, not every exploratory branch
        that happened earlier in the episode.
        """
        if len(compact_positions) <= 1:
            return list(compact_positions)

        stack_positions: List[int] = []
        stack_ids: List[int] = []
        id_to_stack_idx: Dict[int, int] = {}

        for compact_pos, dom_id in zip(compact_positions, compact_dom_ids):
            if dom_id in id_to_stack_idx:
                loop_start = id_to_stack_idx[dom_id]
                for removed_dom in stack_ids[loop_start + 1 :]:
                    id_to_stack_idx.pop(removed_dom, None)
                stack_positions = stack_positions[: loop_start + 1]
                stack_ids = stack_ids[: loop_start + 1]
                stack_positions[loop_start] = compact_pos
                id_to_stack_idx[dom_id] = loop_start
            else:
                id_to_stack_idx[dom_id] = len(stack_ids)
                stack_positions.append(compact_pos)
                stack_ids.append(dom_id)

        return stack_positions

    def _build_segmented_checkpoint_transition_graphs(
        self,
        goal_x: float,
        goal_z: float,
        goal_radius: float,
        checkpoint_seeds: List[torch.Tensor],
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        """
        Build separate denoised transition graphs for the goal segment and each
        checkpoint's upstream segment.

        Segments are assigned from successful goal-reaching episodes as direct
        trunks between consecutive subgoal crossings:
        - the goal segment owns the compact path from goal back to the nearest
          checkpoint crossing
        - each checkpoint then owns the loop-erased trunk from its crossing back
          to the next upstream checkpoint crossing or episode start

        This avoids giving a checkpoint source the entire random-walk prefix
        before it was crossed, which previously created room-wide peaks in odd
        places.
        """
        if not checkpoint_seeds:
            return None, "segmented=no_checkpoints"

        history, history_log = self._extract_compact_unified_history(
            start_step=start_step,
            end_step=end_step,
        )
        if history is None:
            return None, f"segmented=disabled({history_log})"

        time_indices = history["time_indices"]
        if len(time_indices) < 2:
            return None, f"segmented=too_short({history_log})"

        compact_pos = self.hmap_loc[time_indices, :2]
        compact_goal_dist = np.linalg.norm(
            compact_pos - np.asarray([float(goal_x), float(goal_z)], dtype=np.float32),
            axis=1,
        )
        inside_goal = compact_goal_dist <= float(max(0.0, goal_radius))
        goal_entries = [
            idx
            for idx in range(len(time_indices))
            if inside_goal[idx] and (idx == 0 or not inside_goal[idx - 1])
        ]
        if not goal_entries:
            return None, f"segmented=no_goal_entries({history_log})"

        valid_checkpoints = getattr(self, "_valid_checkpoints", None)
        checkpoint_radius = float(
            max(
                0.0,
                self.checkpoint_visit_radius
                if self.checkpoint_visit_radius > 0.0
                else goal_radius,
            )
        )
        checkpoint_positions = []
        checkpoint_regions = {}
        checkpoint_indices = []
        for checkpoint_idx, (cx, cy) in enumerate(self.detected_doorways):
            if valid_checkpoints is not None and checkpoint_idx not in valid_checkpoints:
                continue
            checkpoint_indices.append(int(checkpoint_idx))
            checkpoint_positions.append((float(cx), float(cy)))
            checkpoint_region, _ = self._infer_checkpoint_support_region(
                float(cx),
                float(cy),
            )
            checkpoint_regions[int(checkpoint_idx)] = checkpoint_region

        if not checkpoint_positions:
            return None, f"segmented=no_valid_checkpoints({history_log})"

        compact_dom_ids: List[int] = []
        for t in time_indices:
            vec_t = torch.cat([arr[t] for arr in history["scale_arrays"]], dim=0)
            compact_dom_ids.append(int(torch.argmax(vec_t).item()))

        goal_edge_times: List[tuple[int, int]] = []
        checkpoint_edge_times: Dict[int, List[tuple[int, int]]] = {
            checkpoint_idx: [] for checkpoint_idx in checkpoint_indices
        }
        checkpoint_depths: Dict[int, List[int]] = {
            checkpoint_idx: [] for checkpoint_idx in checkpoint_indices
        }
        goal_segment_count = 0
        goal_raw_edge_count = 0
        checkpoint_segment_counts: Dict[int, int] = {
            checkpoint_idx: 0 for checkpoint_idx in checkpoint_indices
        }
        checkpoint_raw_edge_counts: Dict[int, int] = {
            checkpoint_idx: 0 for checkpoint_idx in checkpoint_indices
        }
        goal_support_times: List[int] = []
        goal_support_ids: List[int] = []
        goal_trunk_time_sequences: List[List[int]] = []
        checkpoint_support_times: Dict[int, List[int]] = {
            checkpoint_idx: [] for checkpoint_idx in checkpoint_indices
        }
        checkpoint_support_ids: Dict[int, List[int]] = {
            checkpoint_idx: [] for checkpoint_idx in checkpoint_indices
        }
        checkpoint_trunk_time_sequences: Dict[int, List[List[int]]] = {
            checkpoint_idx: [] for checkpoint_idx in checkpoint_indices
        }
        checkpoint_prefix_steps = int(
            max(
                0,
                getattr(self, "goal_map_checkpoint_segment_prefix_steps", 120),
            )
        )
        checkpoint_trunk_cap_steps = int(
            max(
                0,
                getattr(self, "goal_map_checkpoint_segment_trunk_cap_steps", 72),
            )
        )

        def _append_segment_edges(
            owner: tuple[str, Optional[int]],
            start_pos: int,
            end_pos: int,
        ) -> int:
            nonlocal goal_segment_count, goal_raw_edge_count
            if end_pos - start_pos < 1:
                return 0

            if owner[0] == "checkpoint" and checkpoint_prefix_steps > 0:
                start_pos = max(int(start_pos), int(end_pos) - int(checkpoint_prefix_steps))
                if end_pos - start_pos < 1:
                    return 0

            raw_edges = int(end_pos - start_pos)
            segment_positions = list(range(start_pos, end_pos + 1))
            segment_dom_ids = compact_dom_ids[start_pos : end_pos + 1]
            trunk_positions = self._loop_erase_compact_segment(
                segment_positions,
                segment_dom_ids,
            )
            if owner[0] == "checkpoint" and checkpoint_trunk_cap_steps > 0:
                keep_nodes = int(checkpoint_trunk_cap_steps) + 1
                if len(trunk_positions) > keep_nodes:
                    trunk_positions = trunk_positions[-keep_nodes:]
            if len(trunk_positions) < 2:
                return 0

            edge_pairs = [
                (time_indices[trunk_positions[i]], time_indices[trunk_positions[i + 1]])
                for i in range(len(trunk_positions) - 1)
            ]
            trunk_time_indices = [time_indices[pos] for pos in trunk_positions]
            trunk_support_ids = [compact_dom_ids[pos] for pos in trunk_positions]
            if owner[0] == "goal":
                goal_edge_times.extend(edge_pairs)
                goal_support_times.extend(trunk_time_indices)
                goal_support_ids.extend(trunk_support_ids)
                goal_trunk_time_sequences.append(list(trunk_time_indices))
                goal_segment_count += 1
                goal_raw_edge_count += raw_edges
            else:
                checkpoint_idx = int(owner[1])
                checkpoint_edge_times[checkpoint_idx].extend(edge_pairs)
                checkpoint_support_times[checkpoint_idx].extend(trunk_time_indices)
                checkpoint_support_ids[checkpoint_idx].extend(trunk_support_ids)
                checkpoint_trunk_time_sequences[checkpoint_idx].append(
                    list(trunk_time_indices)
                )
                checkpoint_segment_counts[checkpoint_idx] += 1
                checkpoint_raw_edge_counts[checkpoint_idx] += raw_edges
            return len(edge_pairs)

        for goal_entry_idx_pos, goal_entry in enumerate(goal_entries):
            lower_bound = 0 if goal_entry_idx_pos == 0 else (goal_entries[goal_entry_idx_pos - 1] + 1)
            current_owner = ("goal", None)
            encountered_checkpoints = set()
            depth = 0
            segment_end = goal_entry

            for pos_idx in range(goal_entry - 1, lower_bound - 1, -1):
                curr_pos = compact_pos[pos_idx]
                nearest_checkpoint = None
                nearest_dist = float("inf")
                for checkpoint_idx, (cx, cy) in zip(checkpoint_indices, checkpoint_positions):
                    if checkpoint_idx in encountered_checkpoints:
                        continue
                    dist = math.sqrt(
                        self._support_region_distance_sq(
                            float(curr_pos[0]),
                            float(curr_pos[1]),
                            checkpoint_regions.get(int(checkpoint_idx)),
                            default_point=(float(cx), float(cy)),
                        )
                    )
                    if dist <= checkpoint_radius and dist < nearest_dist:
                        nearest_checkpoint = checkpoint_idx
                        nearest_dist = dist

                if nearest_checkpoint is not None:
                    _append_segment_edges(
                        owner=current_owner,
                        start_pos=pos_idx,
                        end_pos=segment_end,
                    )
                    encountered_checkpoints.add(nearest_checkpoint)
                    depth += 1
                    checkpoint_depths[nearest_checkpoint].append(depth)
                    current_owner = ("checkpoint", nearest_checkpoint)
                    segment_end = pos_idx

            _append_segment_edges(
                owner=current_owner,
                start_pos=lower_bound,
                end_pos=segment_end,
            )

        if not goal_edge_times:
            return None, f"segmented=no_goal_edges({history_log})"

        goal_counts, goal_edges = self._build_unified_transition_counts_from_edge_pairs(
            history["scale_arrays"],
            goal_edge_times,
        )

        checkpoint_counts = {}
        checkpoint_order = []
        for checkpoint_idx, edge_times in checkpoint_edge_times.items():
            if not edge_times:
                continue
            counts, _ = self._build_unified_transition_counts_from_edge_pairs(
                history["scale_arrays"],
                edge_times,
            )
            checkpoint_counts[int(checkpoint_idx)] = counts
            depth_samples = checkpoint_depths.get(checkpoint_idx, [])
            depth_score = float(np.mean(depth_samples)) if depth_samples else float("inf")
            checkpoint_order.append((depth_score, int(checkpoint_idx), len(edge_times)))

        checkpoint_order.sort(key=lambda item: (item[0], item[1]))
        ordered_checkpoint_indices = [item[1] for item in checkpoint_order]
        checkpoint_edge_summary = ", ".join(
            f"cp{cp_idx}:{len(checkpoint_edge_times[cp_idx])}"
            for cp_idx in ordered_checkpoint_indices
        )
        checkpoint_segment_summary = ", ".join(
            f"cp{cp_idx}:{checkpoint_segment_counts[cp_idx]}"
            for cp_idx in ordered_checkpoint_indices
        )
        checkpoint_raw_summary = ", ".join(
            f"cp{cp_idx}:{checkpoint_raw_edge_counts[cp_idx]}"
            for cp_idx in ordered_checkpoint_indices
        )
        checkpoint_order_summary = ",".join(
            f"cp{cp_idx}" for cp_idx in ordered_checkpoint_indices
        )
        log = (
            f"segmented({history_log},goal_entries={len(goal_entries)},"
            f"goal_edges={len(goal_edge_times)},"
            f"goal_segments={goal_segment_count},"
            f"goal_raw_edges={goal_raw_edge_count},"
            f"checkpoint_edges={{ {checkpoint_edge_summary} }},"
            f"checkpoint_segments={{ {checkpoint_segment_summary} }},"
            f"checkpoint_raw_edges={{ {checkpoint_raw_summary} }},"
            f"order=[{checkpoint_order_summary}])"
        )
        return {
            "goal_counts": goal_counts,
            "checkpoint_counts": checkpoint_counts,
            "checkpoint_order": ordered_checkpoint_indices,
            "scale_arrays": history["scale_arrays"],
            "time_indices": list(time_indices),
            "goal_support_times": sorted(set(goal_support_times)),
            "goal_support_ids": sorted(set(goal_support_ids)),
            "goal_trunk_time_sequences": list(goal_trunk_time_sequences),
            "checkpoint_support_times": {
                checkpoint_idx: sorted(set(times))
                for checkpoint_idx, times in checkpoint_support_times.items()
                if times
            },
            "checkpoint_support_ids": {
                checkpoint_idx: sorted(set(ids))
                for checkpoint_idx, ids in checkpoint_support_ids.items()
                if ids
            },
            "checkpoint_trunk_time_sequences": {
                checkpoint_idx: [list(seq) for seq in trunk_sequences]
                for checkpoint_idx, trunk_sequences in checkpoint_trunk_time_sequences.items()
                if trunk_sequences
            },
        }, log

    def _build_trajectory_state_segment_fields(
        self,
        scale_arrays,
        compact_time_indices: List[int],
        trunk_time_sequences: Optional[List[List[int]]],
        support_mask: torch.Tensor,
        target_device,
        source_kind: str,
        room_fill_gain: float,
        room_decay_tau: float,
        core_gain: float,
        support_time_indices: Optional[List[int]] = None,
    ):
        """
        Build a room-local reward field from compact trajectory states.

        The field is constructed in two phases:
        1. Backbone: start from the actual contact states for the source room and
           replay backward along recorded trunk sequences with per-scale decay.
        2. Fill: after the backbone is complete, spread reward to nearby states in
           the same owned room as a function of distance to the backbone.

        The final room field is projected back into unified PC weights with a
        simple Hebbian-style accumulation over the saved compact history.
        """
        num_pc_total = int(getattr(self.unified_rcn, "num_place_cells_total", 0))
        zeros = torch.zeros(num_pc_total, dtype=torch.float32, device=target_device)
        support_mask_t = torch.as_tensor(
            support_mask, dtype=torch.bool, device=target_device
        ).view(-1)
        if num_pc_total <= 0 or not bool(torch.any(support_mask_t).item()):
            return zeros, zeros, {
                "path_mode": "trajectory_state_backbone(empty_support)",
                "scale_log": "lambda=unavailable",
                "candidate_states": 0,
                "backbone_states": 0,
                "contact_states": 0,
                "max_hops": 0,
                "trunk_count": 0,
                "backbone_log": "trajectory_state=empty_support",
            }

        max_steps = min(arr.shape[0] for arr in scale_arrays) if scale_arrays else 0
        compact_times = [
            int(t)
            for t in (compact_time_indices or [])
            if 0 <= int(t) < int(max_steps)
        ]
        if not compact_times:
            compact_times = sorted(
                {
                    int(t)
                    for seq in (trunk_time_sequences or [])
                    for t in seq
                    if 0 <= int(t) < int(max_steps)
                }
                | {
                    int(t)
                    for t in (support_time_indices or [])
                    if 0 <= int(t) < int(max_steps)
                }
            )
        if not compact_times:
            return zeros, zeros, {
                "path_mode": "trajectory_state_backbone(no_times)",
                "scale_log": "lambda=unavailable",
                "candidate_states": 0,
                "backbone_states": 0,
                "contact_states": 0,
                "max_hops": 0,
                "trunk_count": 0,
                "backbone_log": "trajectory_state=no_compact_times",
            }

        compact_index_t = torch.as_tensor(compact_times, dtype=torch.long)
        compact_blocks = [
            arr.index_select(0, compact_index_t).to(target_device, dtype=torch.float32)
            for arr in scale_arrays
        ]
        compact_history = torch.clamp(torch.cat(compact_blocks, dim=1), min=0.0)
        time_to_local_idx = {int(t): idx for idx, t in enumerate(compact_times)}

        lambda_per_pc = getattr(self.unified_rcn, "lambda_per_pc", None)
        if (
            not isinstance(lambda_per_pc, torch.Tensor)
            or lambda_per_pc.numel() != num_pc_total
        ):
            scale_boundaries = list(
                getattr(self.unified_rcn, "scale_boundaries", [0, num_pc_total])
            )
            lambda_per_scale = list(
                getattr(
                    self.unified_rcn,
                    "lambda_per_scale",
                    [float(getattr(self.unified_rcn, "lambda_global", 40.0))],
                )
            )
            if len(scale_boundaries) < 2:
                scale_boundaries = [0, num_pc_total]
            if len(lambda_per_scale) != len(scale_boundaries) - 1:
                lambda_global = float(getattr(self.unified_rcn, "lambda_global", 40.0))
                lambda_per_scale = [lambda_global] * (len(scale_boundaries) - 1)
            lambda_blocks = []
            for scale_idx, (start, end) in enumerate(
                zip(scale_boundaries[:-1], scale_boundaries[1:])
            ):
                width = int(max(0, int(end) - int(start)))
                lam = float(max(1e-6, lambda_per_scale[scale_idx]))
                lambda_blocks.append(
                    torch.full((width,), lam, dtype=torch.float32, device=target_device)
                )
            lambda_per_pc_t = (
                torch.cat(lambda_blocks, dim=0)
                if lambda_blocks
                else torch.full(
                    (num_pc_total,),
                    float(max(1e-6, getattr(self.unified_rcn, "lambda_global", 40.0))),
                    dtype=torch.float32,
                    device=target_device,
                )
            )
        else:
            lambda_per_pc_t = lambda_per_pc.to(target_device, dtype=torch.float32).view(-1)
        lambda_per_pc_t = torch.clamp(lambda_per_pc_t, min=1e-6)

        scale_boundaries = list(getattr(self.unified_rcn, "scale_boundaries", [0, num_pc_total]))
        lambda_per_scale = list(getattr(self.unified_rcn, "lambda_per_scale", []))
        if len(scale_boundaries) < 2:
            scale_boundaries = [0, num_pc_total]
        if len(lambda_per_scale) != len(scale_boundaries) - 1:
            lambda_global = float(getattr(self.unified_rcn, "lambda_global", 40.0))
            lambda_per_scale = [lambda_global] * (len(scale_boundaries) - 1)
        slowest_lambda = float(max(lambda_per_scale)) if lambda_per_scale else float(
            getattr(self.unified_rcn, "lambda_global", 40.0)
        )
        scale_log = "lambda=[" + ",".join(f"{float(lam):.2f}" for lam in lambda_per_scale) + "]"

        source_sequences: List[List[int]] = []
        for seq in (trunk_time_sequences or []):
            seq_valid = [
                int(t)
                for t in seq
                if 0 <= int(t) < int(max_steps) and int(t) in time_to_local_idx
            ]
            if seq_valid:
                source_sequences.append(seq_valid)
        if not source_sequences and support_time_indices:
            source_sequences = [
                [int(t)]
                for t in support_time_indices
                if 0 <= int(t) < int(max_steps) and int(t) in time_to_local_idx
            ]

        backbone_hops: Dict[int, int] = {}
        for seq in source_sequences:
            for hop, t in enumerate(reversed(seq)):
                prev_hop = backbone_hops.get(int(t))
                if prev_hop is None or int(hop) < int(prev_hop):
                    backbone_hops[int(t)] = int(hop)
        if not backbone_hops:
            return zeros, zeros, {
                "path_mode": "trajectory_state_backbone(no_backbone)",
                "scale_log": scale_log,
                "candidate_states": 0,
                "backbone_states": 0,
                "contact_states": 0,
                "max_hops": 0,
                "trunk_count": len(source_sequences),
                "backbone_log": "trajectory_state=no_backbone_states",
            }

        backbone_times = sorted(backbone_hops.keys())
        backbone_local_idx = torch.as_tensor(
            [int(time_to_local_idx[t]) for t in backbone_times],
            dtype=torch.long,
            device=target_device,
        )
        backbone_states = compact_history.index_select(0, backbone_local_idx)
        backbone_state_peak = torch.amax(backbone_states, dim=1, keepdim=True).clamp_min(1e-8)
        backbone_states = backbone_states / backbone_state_peak
        backbone_hop_t = torch.as_tensor(
            [float(backbone_hops[t]) for t in backbone_times],
            dtype=torch.float32,
            device=target_device,
        ).unsqueeze(1)
        backbone_decay = torch.exp(-backbone_hop_t / lambda_per_pc_t.unsqueeze(0))
        if float(core_gain) != 1.0:
            core_mult = torch.where(
                backbone_hop_t <= 0.0,
                torch.full_like(backbone_hop_t, float(core_gain)),
                torch.ones_like(backbone_hop_t),
            )
            backbone_decay = backbone_decay * core_mult
        path_raw = torch.sum(backbone_states * backbone_decay, dim=0)
        path_raw = torch.where(support_mask_t, path_raw, torch.zeros_like(path_raw))

        support_overlap = torch.amax(
            compact_history[:, support_mask_t], dim=1
        ).view(-1)
        candidate_mask = support_overlap > 1e-8
        backbone_compact_mask = torch.zeros(
            (len(compact_times),), dtype=torch.bool, device=target_device
        )
        backbone_compact_mask[backbone_local_idx] = True
        fill_candidate_mask = candidate_mask & (~backbone_compact_mask)
        fill_raw = torch.zeros_like(path_raw)
        if bool(torch.any(fill_candidate_mask).item()) and float(room_fill_gain) > 1e-8:
            fill_local_idx = torch.nonzero(fill_candidate_mask, as_tuple=False).squeeze(1)
            fill_states = compact_history.index_select(0, fill_local_idx)
            fill_state_peak = torch.amax(fill_states, dim=1, keepdim=True).clamp_min(1e-8)
            fill_states = fill_states / fill_state_peak
            fill_times = [compact_times[int(idx)] for idx in fill_local_idx.detach().cpu().tolist()]
            fill_positions = torch.as_tensor(
                self.hmap_loc[fill_times, :2], dtype=torch.float32, device=target_device
            )
            backbone_positions = torch.as_tensor(
                self.hmap_loc[backbone_times, :2],
                dtype=torch.float32,
                device=target_device,
            )
            backbone_scalar = torch.exp(
                -backbone_hop_t.view(-1) / max(1e-6, slowest_lambda)
            )
            if float(core_gain) != 1.0:
                backbone_scalar = torch.where(
                    backbone_hop_t.view(-1) <= 0.0,
                    backbone_scalar * float(core_gain),
                    backbone_scalar,
                )
            dist_matrix = torch.cdist(fill_positions, backbone_positions, p=2)
            nearest_dist, nearest_idx = torch.min(dist_matrix, dim=1)
            nearest_score = backbone_scalar.index_select(0, nearest_idx)
            fill_score = (
                float(room_fill_gain)
                * nearest_score
                * torch.exp(-nearest_dist / max(1e-6, float(room_decay_tau)))
            )
            fill_raw = torch.sum(fill_states * fill_score.unsqueeze(1), dim=0)
            fill_raw = torch.where(support_mask_t, fill_raw, torch.zeros_like(fill_raw))

        max_hops = max(int(hop) for hop in backbone_hops.values()) if backbone_hops else 0
        metadata = {
            "path_mode": f"trajectory_state_backbone({source_kind})",
            "scale_log": scale_log,
            "candidate_states": int(torch.count_nonzero(candidate_mask).item()),
            "backbone_states": int(len(backbone_times)),
            "contact_states": int(sum(1 for hop in backbone_hops.values() if int(hop) == 0)),
            "max_hops": int(max_hops),
            "trunk_count": int(len(source_sequences)),
            "backbone_log": (
                "trajectory_state("
                f"trunks={len(source_sequences)},"
                f"backbone={len(backbone_times)},"
                f"contacts={sum(1 for hop in backbone_hops.values() if int(hop) == 0)},"
                f"max_hops={max_hops},"
                f"fill_candidates={int(torch.count_nonzero(fill_candidate_mask).item())},"
                f"fill_tau={float(room_decay_tau):.2f},"
                f"fill_gain={float(room_fill_gain):.2f})"
            ),
        }
        return path_raw, fill_raw, metadata

    def _build_segment_support_mask(
        self,
        source_counts: torch.Tensor,
        source_seed: torch.Tensor,
        local_kernel: Optional[torch.Tensor],
        support_node_ids: Optional[List[int]] = None,
        doorway_blocker_node_ids: Optional[List[int]] = None,
        own_doorway_node_ids: Optional[List[int]] = None,
        bridge_node_ids: Optional[List[int]] = None,
        doorway_support_regions: Optional[List[dict]] = None,
        support_time_indices: Optional[List[int]] = None,
        scale_arrays: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Build a coarse ownership mask for a segmented reward source.

        Segmented checkpoint histories separate the experience graph per source,
        but the local neighbor spread can still leak outside that source's
        traversed segment. This mask expands the source's graph support by a few
        local-kernel hops and is used to keep each segment's reward map confined
        to the region it should own before segment composition.
        """
        counts = torch.as_tensor(
            source_counts, dtype=torch.float32, device=self.device
        )
        if counts.dim() != 2:
            return torch.ones(
                self.unified_rcn.num_place_cells_total,
                dtype=torch.bool,
                device=self.device,
            )

        support = torch.zeros(
            self.unified_rcn.num_place_cells_total,
            dtype=torch.float32,
            device=self.device,
        )
        support_anchor_ids: List[int] = []
        if support_node_ids:
            valid_idx = [
                int(idx)
                for idx in support_node_ids
                if 0 <= int(idx) < int(support.numel())
            ]
            if valid_idx:
                support[torch.as_tensor(valid_idx, device=self.device)] = 1.0
                support_anchor_ids.extend(valid_idx)
        elif support_time_indices and scale_arrays:
            for t in support_time_indices:
                if t < 0:
                    continue
                try:
                    vec_t = torch.cat([arr[t] for arr in scale_arrays], dim=0)
                except (IndexError, RuntimeError, TypeError):
                    continue
                active_idx = self._relay_support_indices(
                    vec_t,
                    # Segment ownership should track the backbone cells that
                    # anchor a trunk, not a broad swath of all moderately
                    # active cells at each timepoint. A tighter support set
                    # lets the downstream goal segment stop pre-claiming whole
                    # upstream rooms before checkpoint relay gets a chance to
                    # fill them.
                    topk=8,
                    threshold_fraction=0.50,
                )
                if active_idx:
                    support[torch.as_tensor(active_idx, device=self.device)] = 1.0
                    support_anchor_ids.extend(int(idx) for idx in active_idx)

        if float(torch.max(support).item()) <= 0.0:
            support = (
                (torch.sum(counts, dim=0) + torch.sum(counts, dim=1)) > 1e-9
            ).to(dtype=torch.float32)
            support_anchor_ids.extend(
                int(idx)
                for idx in torch.nonzero(support > 0.0, as_tuple=False).squeeze(1).tolist()
            )
        seed = torch.as_tensor(
            source_seed, dtype=torch.float32, device=self.device
        ).view(-1)
        if seed.numel() != support.numel():
            fixed_seed = torch.zeros_like(support)
            n_copy = min(seed.numel(), support.numel())
            fixed_seed[:n_copy] = seed[:n_copy]
            seed = fixed_seed
        seed_support_idx = self._relay_support_indices(
            seed,
            topk=12,
            threshold_fraction=0.35,
        )
        if seed_support_idx:
            support[torch.as_tensor(seed_support_idx, device=self.device)] = 1.0
            support_anchor_ids.extend(int(idx) for idx in seed_support_idx)

        support_anchor_ids = sorted(set(int(idx) for idx in support_anchor_ids))
        doorway_blockers = sorted(
            set(
                int(idx)
                for idx in (doorway_blocker_node_ids or [])
                if 0 <= int(idx) < int(support.numel())
            )
        )
        own_doorway_ids = sorted(
            set(
                int(idx)
                for idx in (own_doorway_node_ids or [])
                if 0 <= int(idx) < int(support.numel())
            )
        )
        bridge_ids = sorted(
            set(
                int(idx)
                for idx in (bridge_node_ids or [])
                if 0 <= int(idx) < int(support.numel())
            )
        )
        if bridge_ids:
            support[torch.as_tensor(bridge_ids, device=self.device)] = 1.0

        if local_kernel is None:
            return support > 0

        kernel = (
            torch.as_tensor(local_kernel, dtype=torch.float32, device=self.device) > 1e-9
        ).to(dtype=torch.float32)
        if kernel.dim() != 2 or kernel.shape[0] != support.numel():
            return support > 0

        if support_anchor_ids and doorway_blockers and doorway_support_regions:
            centers, visited_cells, _ = self._get_unified_pc_spatial_support(self.device)
            if centers is not None and visited_cells is not None:
                doorway_bounds = []
                for region in doorway_support_regions:
                    bounds = self._support_region_bounds(region)
                    if bounds is not None:
                        doorway_bounds.append(bounds)
                anchor_room_ids = [
                    int(idx) for idx in support_anchor_ids if int(idx) not in doorway_blockers
                ]
                if anchor_room_ids and doorway_bounds:
                    kernel_cpu = kernel.detach().cpu().numpy()
                    centers_cpu = centers.detach().cpu().numpy()
                    visited_cpu = visited_cells.detach().cpu().numpy().astype(bool)
                    blocked = set(int(idx) for idx in doorway_blockers)
                    anchor_set = set(anchor_room_ids)
                    candidate_nodes = {
                        int(idx) for idx in np.nonzero(visited_cpu)[0].tolist()
                    }
                    candidate_nodes.update(anchor_set)
                    candidate_nodes.update(int(idx) for idx in own_doorway_ids)

                    components: List[set[int]] = []
                    seen: set[int] = set()
                    for node in sorted(candidate_nodes):
                        if node in blocked or node in seen:
                            continue
                        comp = set()
                        queue = deque([node])
                        seen.add(node)
                        while queue:
                            cur = queue.popleft()
                            comp.add(cur)
                            nbrs = np.flatnonzero(kernel_cpu[cur] > 1e-9)
                            for nxt in nbrs.tolist():
                                nxt = int(nxt)
                                if nxt in blocked or nxt in seen:
                                    continue
                                p0 = centers_cpu[cur]
                                p1 = centers_cpu[nxt]
                                crosses_doorway = False
                                for bounds in doorway_bounds:
                                    if self._segment_intersects_rectangle(
                                        p0, p1, bounds, margin=0.0
                                    ):
                                        crosses_doorway = True
                                        break
                                if crosses_doorway:
                                    continue
                                seen.add(nxt)
                                queue.append(nxt)
                        if comp:
                            components.append(comp)

                    if components:
                        scored = []
                        for comp in components:
                            overlap = len(comp & anchor_set)
                            if overlap > 0:
                                scored.append((overlap, len(comp), comp))
                        if scored:
                            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                            chosen_comp = scored[0][2]
                            room_support_ids = set(chosen_comp)
                            room_support_ids.update(anchor_set)
                            room_support_ids.update(int(idx) for idx in own_doorway_ids)
                            room_support_ids.update(int(idx) for idx in bridge_ids)
                            support = torch.zeros_like(support)
                            support[
                                torch.as_tensor(
                                    sorted(room_support_ids),
                                    device=self.device,
                                )
                            ] = 1.0
                            return support > 0

        # Keep segment ownership close to the actual traversed trunk. Using the
        # full neighbor-spread radius here lets a short source segment claim
        # almost the whole environment before composition, which collapses the
        # checkpoint-specific gradient into a mostly goal-owned map.
        expand_steps = int(max(0, getattr(self, "goal_map_segment_support_steps", 1)))
        frontier = support.clone()
        expanded = support.clone()
        for _ in range(expand_steps):
            wave = torch.mv(kernel, frontier)
            frontier = torch.where(
                (wave > 1e-9) & (expanded <= 0.0),
                torch.ones_like(wave),
                torch.zeros_like(wave),
            )
            if float(torch.max(frontier).item()) <= 0.0:
                break
            expanded = torch.maximum(expanded, frontier)

        return expanded > 0

    def _expand_node_ids_with_kernel(
        self,
        base_node_ids: Optional[List[int]],
        local_kernel: Optional[torch.Tensor],
        steps: int = 1,
    ) -> List[int]:
        """
        Expand a small support set by a few local-kernel hops.

        This is used to create a shared doorway bridge zone around checkpoints so
        downstream segments can still reach the doorway and upstream relays can
        start from a small buffered target instead of an exact point-set match.
        """
        base_ids = sorted(
            set(
                int(idx)
                for idx in (base_node_ids or [])
                if idx is not None
            )
        )
        if not base_ids or local_kernel is None or int(steps) <= 0:
            return base_ids

        kernel = torch.as_tensor(local_kernel, dtype=torch.float32, device=self.device)
        if kernel.dim() != 2:
            return base_ids

        n_nodes = int(kernel.shape[0])
        valid_ids = [idx for idx in base_ids if 0 <= idx < n_nodes]
        if not valid_ids:
            return []

        kernel_bin = (kernel > 1e-9).to(dtype=torch.float32)
        expanded = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)
        expanded[torch.as_tensor(valid_ids, device=self.device)] = 1.0
        frontier = expanded.clone()

        for _ in range(int(steps)):
            frontier = torch.clamp(
                torch.matmul(kernel_bin, frontier.unsqueeze(1)).squeeze(1),
                min=0.0,
                max=1.0,
            )
            frontier = torch.where(expanded > 0.0, torch.zeros_like(frontier), frontier)
            if float(torch.max(frontier).item()) <= 0.0:
                break
            expanded = torch.maximum(expanded, frontier)

        return sorted(
            int(idx)
            for idx in torch.nonzero(expanded > 0.0, as_tuple=False).squeeze(1).tolist()
        )

    def _pc_graph_weighted_distances_from_sources(
        self,
        neighbors: List[List[tuple[int, float]]],
        sources: List[int],
    ) -> List[float]:
        """Weighted multi-source Dijkstra over the constrained place-cell graph."""
        num_nodes = len(neighbors)
        dist = [math.inf] * num_nodes
        heap = []
        for src in sources:
            src = int(src)
            if 0 <= src < num_nodes and not math.isfinite(dist[src]):
                dist[src] = 0.0
                heapq.heappush(heap, (0.0, src))
        while heap:
            cost_u, node = heapq.heappop(heap)
            if cost_u > dist[node]:
                continue
            for nxt, edge_cost in neighbors[node]:
                next_cost = cost_u + float(edge_cost)
                if next_cost < dist[nxt]:
                    dist[nxt] = next_cost
                    heapq.heappush(heap, (next_cost, nxt))
        return dist

    def _compute_checkpoint_topology_schedule(
        self,
        reward_rcn,
        goal_seed: torch.Tensor,
        checkpoint_seeds: List[torch.Tensor],
        path_constraint: Optional[torch.Tensor],
    ):
        """
        Compute topology-aware checkpoint relay weights and first-hop goal gain.

        Distances are measured on the same constrained place-cell path graph used
        by unified goal-map replay. Checkpoint weights combine:
        1. a topological distance-to-goal prior, and
        2. an adaptive floor that ensures the relay can reach the next upstream
           checkpoint on the inferred relay tree.
        """
        if not checkpoint_seeds:
            return 1.0, [], "relay_topology=none"

        base_gamma = float(max(0.0, getattr(self, "checkpoint_boost_gamma", 0.5)))
        min_relay_weight = float(
            max(0.0, getattr(self, "checkpoint_min_relay_weight", 1.0))
        )

        counts = getattr(reward_rcn, "experience_transition_counts", None)
        if counts is None:
            fallback = [max(base_gamma, min_relay_weight)] * len(checkpoint_seeds)
            return 1.0, fallback, "relay_topology=no_counts"
        counts = torch.as_tensor(counts).detach().float().cpu()
        if float(torch.sum(counts).item()) <= 1e-9:
            fallback = [max(base_gamma, min_relay_weight)] * len(checkpoint_seeds)
            return 1.0, fallback, "relay_topology=empty_counts"

        topology_transition = counts.clone()
        topk = int(max(1, getattr(self, "goal_map_path_topk", 16)))
        if 0 < topk < topology_transition.shape[1]:
            vals, idx = torch.topk(topology_transition, k=topk, dim=1)
            sparse = torch.zeros_like(topology_transition)
            sparse.scatter_(1, idx, vals)
            topology_transition = sparse

        if path_constraint is not None:
            constraint = torch.as_tensor(path_constraint).detach().float().cpu()
            if constraint.shape == topology_transition.shape:
                row_max = torch.max(constraint, dim=1, keepdim=True).values
                constraint = torch.where(
                    row_max > 1e-12,
                    constraint / row_max,
                    torch.zeros_like(constraint),
                )
                topology_transition = topology_transition * torch.clamp(
                    constraint, min=0.0
                )

        row_max = torch.max(topology_transition, dim=1, keepdim=True).values
        topology_transition = torch.where(
            row_max > 1e-12,
            topology_transition / row_max,
            torch.zeros_like(topology_transition),
        )

        topo_eps = 1e-5
        symmetric = torch.maximum(topology_transition, topology_transition.T)
        symmetric.fill_diagonal_(0.0)
        edge_idx = torch.nonzero(symmetric > topo_eps, as_tuple=False)
        num_nodes = int(symmetric.shape[0])
        neighbors: List[List[tuple[int, float]]] = [[] for _ in range(num_nodes)]
        path_decay = float(max(1e-6, getattr(self, "goal_map_path_decay", 0.97)))
        decay_denom = max(1e-6, -math.log(path_decay))
        for src, dst in edge_idx.tolist():
            weight = float(symmetric[src, dst].item())
            edge_signal = max(1e-8, min(1.0, path_decay * weight))
            edge_cost = max(1.0, -math.log(edge_signal) / decay_denom)
            neighbors[int(src)].append((int(dst), float(edge_cost)))
        if all(len(row) == 0 for row in neighbors):
            fallback = [max(base_gamma, min_relay_weight)] * len(checkpoint_seeds)
            return 1.0, fallback, "relay_topology=disconnected"

        support_sets = [self._relay_support_indices(goal_seed)]
        support_sets.extend(self._relay_support_indices(seed) for seed in checkpoint_seeds)
        if not support_sets[0]:
            fallback = [max(base_gamma, min_relay_weight)] * len(checkpoint_seeds)
            return 1.0, fallback, "relay_topology=no_goal_support"

        per_node_dist = []
        for support in support_sets:
            per_node_dist.append(
                self._pc_graph_weighted_distances_from_sources(neighbors, support)
                if support
                else [math.inf] * num_nodes
            )

        relay_count = len(checkpoint_seeds)
        pairwise = [[math.inf] * (relay_count + 1) for _ in range(relay_count + 1)]
        goal_dist = [0.0] * (relay_count + 1)
        for i, dist_i in enumerate(per_node_dist):
            for j, support_j in enumerate(support_sets):
                if not support_j:
                    continue
                pairwise[i][j] = float(min((dist_i[idx] for idx in support_j), default=math.inf))
            if i > 0:
                goal_dist[i] = pairwise[0][i]

        topo_beta = float(max(0.0, getattr(self, "checkpoint_beta", 0.03)))
        trigger_threshold = float(max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01)))
        arrival_target = float(
            max(trigger_threshold, getattr(self, "checkpoint_target_arrival", 0.15))
        )
        relay_margin = float(max(1.0, getattr(self, "checkpoint_relay_margin", 1.15)))
        max_relay_weight = float(max(1.0, getattr(self, "checkpoint_max_relay_weight", 4.0)))
        max_goal_gain = float(max(1.0, getattr(self, "goal_map_max_initial_backbone_gain", 2.0)))

        parent = [None] * (relay_count + 1)
        parent_gap = [0.0] * (relay_count + 1)
        tolerance = 2.0
        for node_idx in range(1, relay_count + 1):
            if not math.isfinite(goal_dist[node_idx]):
                continue
            best_parent = 0
            best_gap = pairwise[0][node_idx]
            best_score = (1, math.inf, 0.0, best_gap)
            for cand_idx in range(0, relay_count + 1):
                if cand_idx == node_idx:
                    continue
                if not math.isfinite(goal_dist[cand_idx]) or goal_dist[cand_idx] >= goal_dist[node_idx]:
                    continue
                gap = pairwise[cand_idx][node_idx]
                if not math.isfinite(gap):
                    continue
                residual = abs(goal_dist[node_idx] - (goal_dist[cand_idx] + gap))
                on_path = 0 if residual <= tolerance else 1
                score = (
                    on_path,
                    0.0 if on_path == 0 else residual,
                    -goal_dist[cand_idx],
                    gap,
                )
                if score < best_score:
                    best_score = score
                    best_parent = cand_idx
                    best_gap = gap
            parent[node_idx] = best_parent
            parent_gap[node_idx] = float(best_gap) if math.isfinite(best_gap) else 0.0

        children: Dict[int, List[float]] = {i: [] for i in range(relay_count + 1)}
        for node_idx in range(1, relay_count + 1):
            if parent[node_idx] is not None:
                children[int(parent[node_idx])].append(parent_gap[node_idx])

        checkpoint_weights: List[float] = []
        for relay_idx in range(1, relay_count + 1):
            d_topo = goal_dist[relay_idx]
            base_weight = (
                base_gamma * math.exp(-topo_beta * d_topo)
                if math.isfinite(d_topo)
                else 0.0
            )
            max_child_gap = max(children.get(relay_idx, []) or [0.0])
            adaptive_floor = (
                (arrival_target * relay_margin) / (path_decay ** max_child_gap)
                if max_child_gap > 0.0
                else 0.0
            )
            leaf_floor = min_relay_weight if max_child_gap <= 0.0 else 0.0
            checkpoint_weights.append(
                float(
                    min(
                        max_relay_weight,
                        max(base_weight, adaptive_floor, leaf_floor),
                    )
                )
            )

        first_gap = max(children.get(0, []) or [0.0])
        if first_gap > 0.0:
            goal_seed_gain = min(
                max_goal_gain,
                max(1.0, (arrival_target * relay_margin) / (path_decay ** first_gap)),
            )
        else:
            goal_seed_gain = 1.0

        reachable = sum(1 for d in goal_dist[1:] if math.isfinite(d))
        mean_weight = float(np.mean(checkpoint_weights)) if checkpoint_weights else 0.0
        log = (
            f"relay_topology=pc_graph_weighted(reachable={reachable}/{relay_count},"
            f"first_gap={first_gap:.1f},goal_gain={goal_seed_gain:.2f},"
            f"mean_cp_w={mean_weight:.2f})"
        )
        return float(goal_seed_gain), checkpoint_weights, log

    def _build_weighted_goal_map_topology_neighbors(
        self,
        reward_rcn,
        path_constraint: Optional[torch.Tensor],
    ):
        """Build one weighted symmetric graph over place cells for route selection."""
        counts = getattr(reward_rcn, "experience_transition_counts", None)
        if counts is None:
            return None, "route_topology=no_counts"

        topology_transition = torch.as_tensor(counts).detach().float().cpu()
        if float(torch.sum(topology_transition).item()) <= 1e-9:
            return None, "route_topology=empty_counts"

        topk = int(max(1, getattr(self, "goal_map_path_topk", 16)))
        if 0 < topk < topology_transition.shape[1]:
            vals, idx = torch.topk(topology_transition, k=topk, dim=1)
            sparse = torch.zeros_like(topology_transition)
            sparse.scatter_(1, idx, vals)
            topology_transition = sparse

        if path_constraint is not None:
            constraint = torch.as_tensor(path_constraint).detach().float().cpu()
            if constraint.shape == topology_transition.shape:
                row_max = torch.max(constraint, dim=1, keepdim=True).values
                constraint = torch.where(
                    row_max > 1e-12,
                    constraint / row_max,
                    torch.zeros_like(constraint),
                )
                topology_transition = topology_transition * torch.clamp(
                    constraint, min=0.0
                )

        row_max = torch.max(topology_transition, dim=1, keepdim=True).values
        topology_transition = torch.where(
            row_max > 1e-12,
            topology_transition / row_max,
            torch.zeros_like(topology_transition),
        )

        symmetric = torch.maximum(topology_transition, topology_transition.T)
        symmetric.fill_diagonal_(0.0)
        edge_idx = torch.nonzero(symmetric > 1e-5, as_tuple=False)
        num_nodes = int(symmetric.shape[0])
        neighbors = [[] for _ in range(num_nodes)]
        path_decay = float(max(1e-6, getattr(self, "goal_map_path_decay", 0.97)))
        decay_denom = max(1e-6, -math.log(path_decay))

        for src, dst in edge_idx.tolist():
            weight = float(symmetric[src, dst].item())
            edge_signal = max(1e-8, min(1.0, path_decay * weight))
            edge_cost = max(1.0, -math.log(edge_signal) / decay_denom)
            neighbors[int(src)].append((int(dst), float(edge_cost)))

        if all(len(row) == 0 for row in neighbors):
            return None, "route_topology=disconnected"

        return neighbors, (
            f"route_topology=pc_graph(nodes={num_nodes},edges={int(edge_idx.shape[0])})"
        )

    def _build_goal_map_selection_cloud(
        self,
        reward_rcn,
        path_transition: torch.Tensor,
        neighbor_transition: torch.Tensor,
        seed_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build one permissive global route-selection cloud from a seed.

        This deliberately disables checkpoint relays and room ownership masks so
        route selection can ask whether a checkpoint connects into the
        goal-directed graph at all, independently of the sharper final reward map.
        """
        path_values, neighbor_values, _ = reward_rcn._build_goal_map_path_and_neighbor_values(
            path_transition=path_transition,
            neighbor_transition=neighbor_transition,
            seed_activations=seed_activations,
        )
        return torch.maximum(path_values, neighbor_values)

    def _cloud_support_indices(
        self,
        cloud_values: torch.Tensor,
        threshold_fraction: float,
        extra_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Return support indices for one replay cloud without top-k truncation."""
        vec = torch.as_tensor(cloud_values).detach().float().cpu().view(-1).abs()
        if vec.numel() == 0:
            base_ids = []
        else:
            peak = float(torch.max(vec).item())
            if peak <= 1e-9:
                base_ids = []
            else:
                idx = torch.nonzero(
                    vec >= (float(max(1e-6, threshold_fraction)) * peak),
                    as_tuple=False,
                ).squeeze(1)
                base_ids = [int(i) for i in idx.tolist()]

        if extra_ids:
            base_set = set(base_ids)
            for idx in extra_ids:
                idx = int(idx)
                if 0 <= idx < int(vec.numel()):
                    base_set.add(idx)
            base_ids = sorted(base_set)

        return base_ids

    def _format_route_selected_checkpoint_log(
        self,
        route_data: Optional[Dict[str, Any]],
        bonus_by_checkpoint: Optional[Dict[int, float]] = None,
    ) -> str:
        """Serialize one compact route-selection summary into the goal-map debug log."""
        if not route_data:
            return "route_select=disabled"

        selected = list(route_data.get("selected_order", []))
        anchors = list(route_data.get("direct_anchors", []))
        fallback_anchor = route_data.get("fallback_anchor", None)
        parent_map = route_data.get("parent_map", {})
        direct_scores = route_data.get("direct_scores", {})
        edge_scores = route_data.get("edge_scores", {})
        distances = route_data.get("goal_distances", {})
        base_weights = route_data.get("base_weights", {})
        used_fallback = bool(route_data.get("used_fallback_anchor", False))
        mode = str(route_data.get("mode", "route_selected_tree"))
        route_count = int(route_data.get("route_count", 0))
        episode_count = int(route_data.get("episode_count", 0))
        route_examples = list(route_data.get("route_examples", []))
        bonus_by_checkpoint = bonus_by_checkpoint or {}

        selected_txt = "[" + ",".join(f"cp{int(idx)}" for idx in selected) + "]"
        anchors_txt = "[" + ",".join(f"cp{int(idx)}" for idx in anchors) + "]"
        parents_txt = "{ " + ",".join(
            f"cp{int(child)}:{'goal' if parent is None else f'cp{int(parent)}'}"
            for child, parent in sorted(parent_map.items(), key=lambda item: int(item[0]))
        ) + " }"
        direct_txt = "{ " + ",".join(
            f"cp{int(idx)}:{float(score):.2f}"
            for idx, score in sorted(direct_scores.items(), key=lambda item: int(item[0]))
        ) + " }"
        edge_txt = "{ " + ",".join(
            f"cp{int(child)}->cp{int(parent)}:{float(score):.2f}"
            for (child, parent), score in sorted(
                edge_scores.items(),
                key=lambda item: (int(item[0][0]), int(item[0][1])),
            )
        ) + " }"
        dist_txt = "{ " + ",".join(
            f"cp{int(idx)}:{float(dist):.1f}"
            for idx, dist in sorted(distances.items(), key=lambda item: int(item[0]))
            if math.isfinite(float(dist))
        ) + " }"
        base_txt = "{ " + ",".join(
            f"cp{int(idx)}:{float(weight):.2f}"
            for idx, weight in sorted(base_weights.items(), key=lambda item: int(item[0]))
        ) + " }"
        bonus_txt = "{ " + ",".join(
            f"cp{int(idx)}:{float(weight):.2f}"
            for idx, weight in sorted(
                bonus_by_checkpoint.items(),
                key=lambda item: int(item[0]),
            )
        ) + " }"
        fallback_txt = (
            f"cp{int(fallback_anchor)}"
            if fallback_anchor is not None
            else "none"
        )
        routes_txt = "[" + ";".join(route_examples[:4]) + "]" if route_examples else "[]"

        return (
            "route_select("
            f"mode={mode},"
            f"selected={selected_txt},"
            f"anchors={anchors_txt},"
            f"fallback={int(used_fallback)}:{fallback_txt},"
            f"parents={parents_txt},"
            f"score_goal={direct_txt},"
            f"score_edge={edge_txt},"
            f"dist={dist_txt},"
            f"base={base_txt},"
            f"bonus={bonus_txt},"
            f"episodes={episode_count},routes={route_count},"
            f"examples={routes_txt})"
        )

    def _extract_selected_checkpoint_order_from_goal_map_log(
        self,
        debug_log: str,
    ) -> List[int]:
        """Parse selected checkpoints from the serialized route_select(...) debug log."""
        text = str(debug_log or "")
        marker = "selected=["
        start = text.find(marker)
        if start < 0:
            return []
        end = text.find("]", start + len(marker))
        if end < 0:
            return []
        raw = text[start + len(marker) : end].strip()
        if not raw:
            return []

        parsed: List[int] = []
        for token in raw.split(","):
            token = token.strip()
            if token.startswith("cp"):
                token = token[2:]
            try:
                parsed.append(int(token))
            except ValueError:
                continue
        return parsed

    def _checkpoint_direction_key_toward_target(
        self,
        checkpoint_idx: int,
        target_xy: Tuple[float, float],
    ) -> Optional[str]:
        """Infer which directed crossing would move from a checkpoint toward a target."""
        if not (
            0 <= int(checkpoint_idx) < len(getattr(self, "detected_doorways", []))
        ):
            return None

        checkpoint_xy = self.detected_doorways[int(checkpoint_idx)]
        support_region, _ = self._infer_checkpoint_support_region(
            float(checkpoint_xy[0]),
            float(checkpoint_xy[1]),
        )
        frame = self._support_region_frame(support_region)
        if frame is None:
            return None

        frame_cx, frame_cy = frame["center"]
        nx, ny = frame["normal"]
        signed = (
            (float(target_xy[0]) - float(frame_cx)) * float(nx)
            + (float(target_xy[1]) - float(frame_cy)) * float(ny)
        )
        side_eps = max(
            0.05,
            0.5 * max(0.0, float(frame.get("half_width", 0.0))) + 0.02,
        )
        if abs(float(signed)) <= side_eps:
            return None
        return "neg_to_pos" if float(signed) > 0.0 else "pos_to_neg"

    def _compute_directed_route_memory_tree(
        self,
        goal_x: float,
        goal_z: float,
        goal_radius: float,
        checkpoint_seed_bank: List[tuple[int, torch.Tensor]],
        checkpoint_seed_support_id_bank: Dict[int, List[int]],
        candidate_checkpoint_ids: Optional[List[int]],
        start_step: int,
        end_step: Optional[int],
        target_device,
        goal_name: Optional[str] = None,
    ):
        """
        Recover one checkpoint chain directly from recorded successful subgoal order.

        For each goal-reaching episode, walk backward from the goal contact to the
        latest checkpoint crossing whose direction points toward the current
        downstream target. Repeating this yields the observed upstream chain:
        checkpoint -> downstream checkpoint -> ... -> goal.
        """
        if not checkpoint_seed_bank:
            return None, "route_select=no_checkpoints"

        candidate_ids = (
            {int(idx) for idx in candidate_checkpoint_ids}
            if candidate_checkpoint_ids is not None
            else None
        )
        checkpoint_ids = sorted(
            int(checkpoint_idx)
            for checkpoint_idx, _checkpoint_seed in checkpoint_seed_bank
            if candidate_ids is None or int(checkpoint_idx) in candidate_ids
        )
        if not checkpoint_ids:
            return None, "route_select=no_candidate_checkpoints"

        history_start = int(max(0, start_step))
        history_end = int(self.step_count if end_step is None else end_step)
        if history_end < history_start:
            return None, "route_select=disabled(empty_route_window)"

        goal_name_norm = str(goal_name or "").strip()
        if not goal_name_norm:
            for goal in getattr(self, "goals", []):
                goal_loc = goal.get("location", [])
                if (
                    len(goal_loc) >= 2
                    and abs(float(goal_loc[0]) - float(goal_x)) <= 1e-6
                    and abs(float(goal_loc[1]) - float(goal_z)) <= 1e-6
                ):
                    goal_name_norm = str(goal.get("name", "")).strip()
                    break

        raw_goal_steps = []
        goal_contact_steps = getattr(self, "goal_contact_steps", {})
        if goal_name_norm:
            raw_goal_steps = list(goal_contact_steps.get(goal_name_norm, []))
        if not raw_goal_steps and hasattr(self, "hmap_loc"):
            n_valid = min(int(history_end) + 1, int(self.hmap_loc.shape[0]))
            goal_radius_sq = float(max(0.0, goal_radius)) ** 2
            for step in range(int(history_start), int(n_valid)):
                dx = float(self.hmap_loc[int(step), 0]) - float(goal_x)
                dy = float(self.hmap_loc[int(step), 1]) - float(goal_z)
                if (dx * dx) + (dy * dy) <= goal_radius_sq:
                    raw_goal_steps.append(int(step))

        goal_steps = sorted(
            {
                int(step)
                for step in raw_goal_steps
                if history_start <= int(step) <= history_end
            }
        )
        goal_event_groups = self._group_replay_history_steps_into_events(goal_steps)
        if not goal_event_groups:
            return None, "route_select=disabled(no_goal_events)"

        checkpoint_events = []
        crossing_steps = getattr(self, "checkpoint_crossing_steps", {})
        for checkpoint_idx in checkpoint_ids:
            dir_steps = crossing_steps.get(int(checkpoint_idx), {})
            for direction_key in ("neg_to_pos", "pos_to_neg"):
                raw_steps = dir_steps.get(direction_key, [])
                filtered_steps = [
                    int(step)
                    for step in raw_steps
                    if history_start <= int(step) <= history_end
                ]
                grouped = self._group_replay_history_steps_into_events(filtered_steps)
                for event_group in grouped:
                    if not event_group:
                        continue
                    checkpoint_events.append(
                        {
                            "checkpoint_idx": int(checkpoint_idx),
                            "direction_key": str(direction_key),
                            "step": int(event_group[-1]),
                        }
                    )

        if not checkpoint_events:
            return None, "route_select=disabled(no_checkpoint_events)"
        checkpoint_events.sort(
            key=lambda item: (
                int(item["step"]),
                int(item["checkpoint_idx"]),
                str(item["direction_key"]),
            )
        )

        goal_link_counts: Dict[int, int] = {}
        edge_link_counts: Dict[Tuple[int, int], int] = {}
        child_occurrences: Dict[int, int] = {}
        depth_samples: Dict[int, List[float]] = {}
        route_examples: List[str] = []
        route_count = 0
        previous_goal_step = history_start - 1

        checkpoint_positions = {
            int(idx): (
                float(self.detected_doorways[int(idx)][0]),
                float(self.detected_doorways[int(idx)][1]),
            )
            for idx in checkpoint_ids
            if 0 <= int(idx) < len(getattr(self, "detected_doorways", []))
        }
        if not checkpoint_positions:
            return None, "route_select=disabled(no_checkpoint_positions)"

        for goal_group in goal_event_groups:
            if not goal_group:
                continue
            goal_step = int(goal_group[-1])
            episode_start = int(max(history_start, previous_goal_step + 1))
            previous_goal_step = int(goal_step)

            current_target_xy = (float(goal_x), float(goal_z))
            current_target_cp = None
            current_step = int(goal_step)
            used_checkpoints = set()
            route_chain_near_to_far: List[int] = []

            while True:
                best_event = None
                for event in reversed(checkpoint_events):
                    event_step = int(event["step"])
                    if event_step >= current_step:
                        continue
                    if event_step < episode_start:
                        break

                    checkpoint_idx = int(event["checkpoint_idx"])
                    if checkpoint_idx in used_checkpoints:
                        continue

                    required_direction = self._checkpoint_direction_key_toward_target(
                        checkpoint_idx,
                        current_target_xy,
                    )
                    if (
                        required_direction is not None
                        and str(event["direction_key"]) != required_direction
                    ):
                        continue

                    best_event = event
                    break

                if best_event is None:
                    break

                checkpoint_idx = int(best_event["checkpoint_idx"])
                used_checkpoints.add(checkpoint_idx)
                child_occurrences[checkpoint_idx] = int(
                    child_occurrences.get(checkpoint_idx, 0)
                ) + 1
                depth_samples.setdefault(checkpoint_idx, []).append(
                    float(len(route_chain_near_to_far) + 1)
                )

                if current_target_cp is None:
                    goal_link_counts[checkpoint_idx] = int(
                        goal_link_counts.get(checkpoint_idx, 0)
                    ) + 1
                else:
                    edge_key = (int(checkpoint_idx), int(current_target_cp))
                    edge_link_counts[edge_key] = int(
                        edge_link_counts.get(edge_key, 0)
                    ) + 1

                route_chain_near_to_far.append(int(checkpoint_idx))
                current_target_cp = int(checkpoint_idx)
                current_target_xy = checkpoint_positions.get(
                    int(checkpoint_idx),
                    current_target_xy,
                )
                current_step = int(best_event["step"])

            if route_chain_near_to_far:
                route_count += 1
                route_examples.append(
                    "->".join(
                        [f"cp{int(idx)}" for idx in reversed(route_chain_near_to_far)]
                        + ["goal"]
                    )
                )

        if not depth_samples:
            return None, "route_select=disabled(no_directed_routes)"

        mean_depth = {
            int(checkpoint_idx): float(np.mean(samples))
            for checkpoint_idx, samples in depth_samples.items()
            if samples
        }
        selected_order = sorted(
            mean_depth.keys(),
            key=lambda checkpoint_idx: (
                float(mean_depth[int(checkpoint_idx)]),
                int(checkpoint_idx),
            ),
        )
        if not selected_order:
            return None, "route_select=disabled(no_selected_checkpoints)"

        direct_scores: Dict[int, float] = {}
        edge_scores: Dict[Tuple[int, int], float] = {}
        for checkpoint_idx in selected_order:
            child_total = max(1, int(child_occurrences.get(int(checkpoint_idx), 0)))
            direct_scores[int(checkpoint_idx)] = float(
                goal_link_counts.get(int(checkpoint_idx), 0)
            ) / float(child_total)
        for (child_idx, parent_idx), count in edge_link_counts.items():
            child_total = max(1, int(child_occurrences.get(int(child_idx), 0)))
            edge_scores[(int(child_idx), int(parent_idx))] = float(count) / float(
                child_total
            )

        parent_map: Dict[int, Optional[int]] = {}
        direct_anchors: List[int] = []
        for checkpoint_idx in selected_order:
            checkpoint_idx = int(checkpoint_idx)
            if int(goal_link_counts.get(checkpoint_idx, 0)) > 0:
                parent_map[checkpoint_idx] = None
                direct_anchors.append(checkpoint_idx)
                continue

            parent_candidates = [
                (int(parent_idx), int(count))
                for (child_idx, parent_idx), count in edge_link_counts.items()
                if int(child_idx) == checkpoint_idx
                and int(parent_idx) in mean_depth
                and float(mean_depth[int(parent_idx)]) + 1e-6
                < float(mean_depth[checkpoint_idx])
            ]
            if parent_candidates:
                parent_candidates.sort(
                    key=lambda item: (
                        -int(item[1]),
                        -float(mean_depth[int(item[0])]),
                        int(item[0]),
                    )
                )
                parent_map[checkpoint_idx] = int(parent_candidates[0][0])
                continue

            shallower = [
                int(other_idx)
                for other_idx in selected_order
                if float(mean_depth[int(other_idx)]) + 1e-6
                < float(mean_depth[checkpoint_idx])
            ]
            if shallower:
                shallower.sort(
                    key=lambda other_idx: (
                        -float(mean_depth[int(other_idx)]),
                        int(other_idx),
                    )
                )
                parent_map[checkpoint_idx] = int(shallower[0])
            else:
                parent_map[checkpoint_idx] = None
                direct_anchors.append(checkpoint_idx)

        goal_distances = {
            int(checkpoint_idx): float(mean_depth[int(checkpoint_idx)])
            for checkpoint_idx in selected_order
        }

        base_gamma = float(max(0.0, getattr(self, "checkpoint_boost_gamma", 0.5)))
        min_relay_weight = float(
            max(0.0, getattr(self, "checkpoint_min_relay_weight", 1.0))
        )
        topo_beta = float(max(0.0, getattr(self, "checkpoint_beta", 0.03)))
        trigger_threshold = float(
            max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
        )
        arrival_target = float(
            max(
                trigger_threshold,
                getattr(self, "checkpoint_target_arrival", 0.15),
            )
        )
        relay_margin = float(max(1.0, getattr(self, "checkpoint_relay_margin", 1.15)))
        max_relay_weight = float(
            max(1.0, getattr(self, "checkpoint_max_relay_weight", 4.0))
        )
        path_decay = float(max(1e-6, getattr(self, "goal_map_path_decay", 0.97)))

        children = {int(checkpoint_idx): [] for checkpoint_idx in selected_order}
        for checkpoint_idx, parent_idx in parent_map.items():
            if parent_idx is None:
                continue
            children.setdefault(int(parent_idx), []).append(int(checkpoint_idx))

        base_weights: Dict[int, float] = {}
        for checkpoint_idx in selected_order:
            checkpoint_idx = int(checkpoint_idx)
            d_topo = float(goal_distances.get(checkpoint_idx, math.inf))
            base_weight = (
                base_gamma * math.exp(-topo_beta * d_topo)
                if math.isfinite(d_topo)
                else 0.0
            )
            child_depth_gaps = [
                max(
                    1.0,
                    float(goal_distances.get(int(child_idx), d_topo)) - float(d_topo),
                )
                for child_idx in children.get(checkpoint_idx, [])
            ]
            max_child_gap = max(child_depth_gaps) if child_depth_gaps else 0.0
            adaptive_floor = (
                (arrival_target * relay_margin) / (path_decay ** max_child_gap)
                if max_child_gap > 0.0
                else 0.0
            )
            leaf_floor = min_relay_weight if max_child_gap <= 0.0 else 0.0
            base_weights[checkpoint_idx] = float(
                min(
                    max_relay_weight,
                    max(base_weight, adaptive_floor, leaf_floor),
                )
            )

        route_data = {
            "mode": "directed_route_memory_tree",
            "selected_order": selected_order,
            "selected_set": set(selected_order),
            "direct_anchors": sorted(set(int(idx) for idx in direct_anchors)),
            "fallback_anchor": None,
            "used_fallback_anchor": False,
            "parent_map": parent_map,
            "direct_scores": direct_scores,
            "edge_scores": edge_scores,
            "goal_distances": goal_distances,
            "base_weights": base_weights,
            "route_count": int(route_count),
            "episode_count": int(len(goal_event_groups)),
            "route_examples": list(route_examples[:8]),
        }
        route_log = self._format_route_selected_checkpoint_log(route_data)
        memory_log = (
            "route_memory="
            f"events(goal={len(goal_event_groups)},checkpoint={len(checkpoint_events)},"
            f"routes={route_count},window={history_start}:{history_end})"
        )
        return route_data, f"{route_log},{memory_log}"

    # ------------------------------------------------------------------
    # Multi-route memory graph: online edge-matrix approach
    # ------------------------------------------------------------------

    def _build_multi_route_memory_graph(
        self,
        goal_x: float,
        goal_z: float,
        goal_radius: float,
        checkpoint_seed_bank: List[tuple[int, torch.Tensor]],
        checkpoint_seed_support_id_bank: Dict[int, List[int]],
        candidate_checkpoint_ids: Optional[List[int]],
        start_step: int,
        end_step: Optional[int],
        target_device,
        goal_name: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Build a multi-route memory graph from the online checkpoint edge matrix.

        Instead of replaying episodes and denoising sequences, this reads the
        checkpoint_to_checkpoint_edges and checkpoint_to_goal_edges dicts that
        were potentiated during training, then extracts all routes to the
        specified goal via backward BFS.

        Returns a route_data dict that contains both the rich graph fields
        (downstream_targets, checkpoint_depth, edge_counts, etc.) AND the
        flattened tree fields (selected_order, parent_map, etc.) for Stage 1
        compatibility with existing room-assignment code.
        """
        # --- resolve goal name ---
        goal_name_norm = str(goal_name or "").strip()
        if not goal_name_norm:
            for g in getattr(self, "goals", []):
                gx, gz = float(g.get("x", 0.0)), float(g.get("z", 0.0))
                if abs(gx - goal_x) < 0.05 and abs(gz - goal_z) < 0.05:
                    goal_name_norm = str(g.get("name", ""))
                    break
        if not goal_name_norm:
            return None, "route_select=disabled(no_goal_name_for_multi_route)"

        # --- 1. find goal-linked checkpoints ---
        cp_goal_edges = getattr(self, "checkpoint_to_goal_edges", {})
        goal_link_counts: Dict[int, int] = {}
        goal_link_direction_detail: Dict[int, Dict[str, int]] = {}
        for (src_cp, src_dir, g_name), count in cp_goal_edges.items():
            if str(g_name) != goal_name_norm:
                continue
            src_cp = int(src_cp)
            goal_link_counts[src_cp] = goal_link_counts.get(src_cp, 0) + int(count)
            goal_link_direction_detail.setdefault(src_cp, {})[str(src_dir)] = (
                goal_link_direction_detail.get(src_cp, {}).get(str(src_dir), 0) + int(count)
            )

        if not goal_link_counts:
            return None, "route_select=disabled(no_goal_edges_in_multi_route)"

        # --- 2. build directed adjacency from checkpoint-to-checkpoint edges ---
        cp_cp_edges = getattr(self, "checkpoint_to_checkpoint_edges", {})
        downstream_targets: Dict[int, set] = {}
        edge_counts: Dict[Tuple[int, int], int] = {}
        edge_direction_detail: Dict[Tuple[int, int], Dict[Tuple[str, str], int]] = {}

        for (src_cp, src_dir, dst_cp, dst_dir), count in cp_cp_edges.items():
            src_cp, dst_cp = int(src_cp), int(dst_cp)
            downstream_targets.setdefault(src_cp, set()).add(dst_cp)
            pair_key = (src_cp, dst_cp)
            edge_counts[pair_key] = edge_counts.get(pair_key, 0) + int(count)
            dir_key = (str(src_dir), str(dst_dir))
            edge_direction_detail.setdefault(pair_key, {})[dir_key] = (
                edge_direction_detail.get(pair_key, {}).get(dir_key, 0) + int(count)
            )

        # --- 3. add goal links to downstream_targets ---
        GOAL_SENTINEL = -1
        for cp_idx in goal_link_counts:
            downstream_targets.setdefault(cp_idx, set()).add(GOAL_SENTINEL)

        # --- 4. BFS backward from goal to discover reachable checkpoints ---
        # Build reverse adjacency: for each node, who can reach it?
        reverse_adj: Dict[int, set] = {}
        for src, targets in downstream_targets.items():
            for dst in targets:
                reverse_adj.setdefault(dst, set()).add(src)

        checkpoint_depth: Dict[int, int] = {}
        bfs_queue: list = []
        # Start from goal sentinel
        for cp_idx in reverse_adj.get(GOAL_SENTINEL, set()):
            if cp_idx not in checkpoint_depth:
                checkpoint_depth[cp_idx] = 1
                bfs_queue.append(cp_idx)

        head = 0
        while head < len(bfs_queue):
            current = bfs_queue[head]
            head += 1
            current_depth = checkpoint_depth[current]
            for upstream in reverse_adj.get(current, set()):
                if upstream == GOAL_SENTINEL:
                    continue
                if upstream not in checkpoint_depth or checkpoint_depth[upstream] > current_depth + 1:
                    checkpoint_depth[upstream] = current_depth + 1
                    bfs_queue.append(upstream)

        if not checkpoint_depth:
            return None, "route_select=disabled(no_reachable_checkpoints_in_multi_route)"

        # --- 5. prune to only reachable checkpoints ---
        reachable = set(checkpoint_depth.keys())
        downstream_targets_pruned: Dict[int, set] = {}
        for src in reachable:
            pruned_set = set()
            for dst in downstream_targets.get(src, set()):
                if dst == GOAL_SENTINEL or dst in reachable:
                    pruned_set.add(dst)
            if pruned_set:
                downstream_targets_pruned[src] = pruned_set

        # --- 6. build route examples (trace backward through strongest edges) ---
        route_examples: List[str] = []
        for cp_idx in sorted(goal_link_counts.keys(), key=lambda x: -goal_link_counts[x]):
            chain = [f"cp{cp_idx}", "goal"]
            visited_trace: set = {cp_idx}
            current = cp_idx
            for _ in range(20):
                best_upstream = None
                best_count = 0
                for upstream in reverse_adj.get(current, set()):
                    if upstream == GOAL_SENTINEL or upstream in visited_trace:
                        continue
                    if upstream not in reachable:
                        continue
                    c = edge_counts.get((upstream, current), 0)
                    if c > best_count:
                        best_count = c
                        best_upstream = upstream
                if best_upstream is None:
                    break
                chain.insert(0, f"cp{best_upstream}")
                visited_trace.add(best_upstream)
                current = best_upstream
            route_examples.append("->".join(chain))
            if len(route_examples) >= 16:
                break

        # --- 7. flatten to tree (Stage 1 compatibility) ---
        selected_order = sorted(
            checkpoint_depth.keys(),
            key=lambda cp: (checkpoint_depth[cp], cp),
        )

        # Count total outgoing transitions per checkpoint (for score computation)
        child_occurrences: Dict[int, int] = {}
        for cp_idx in selected_order:
            total = goal_link_counts.get(cp_idx, 0)
            for dst in downstream_targets_pruned.get(cp_idx, set()):
                if dst != GOAL_SENTINEL:
                    total += edge_counts.get((cp_idx, dst), 0)
            child_occurrences[cp_idx] = max(1, total)

        # parent_map: pick downstream target with highest edge count
        parent_map: Dict[int, Optional[int]] = {}
        direct_anchors: List[int] = []
        for cp_idx in selected_order:
            goal_count = goal_link_counts.get(cp_idx, 0)
            best_downstream = None
            best_downstream_count = 0
            for dst in downstream_targets_pruned.get(cp_idx, set()):
                if dst == GOAL_SENTINEL:
                    continue
                c = edge_counts.get((cp_idx, dst), 0)
                if c > best_downstream_count:
                    best_downstream_count = c
                    best_downstream = dst
            if goal_count >= best_downstream_count or best_downstream is None:
                parent_map[cp_idx] = None
                direct_anchors.append(cp_idx)
            else:
                parent_map[cp_idx] = best_downstream

        # Scores
        direct_scores: Dict[int, float] = {}
        edge_scores: Dict[Tuple[int, int], float] = {}
        for cp_idx in selected_order:
            total = max(1, child_occurrences[cp_idx])
            direct_scores[cp_idx] = float(goal_link_counts.get(cp_idx, 0)) / float(total)
            for dst in downstream_targets_pruned.get(cp_idx, set()):
                if dst != GOAL_SENTINEL:
                    edge_scores[(cp_idx, dst)] = float(
                        edge_counts.get((cp_idx, dst), 0)
                    ) / float(total)

        goal_distances: Dict[int, float] = {
            cp: float(d) for cp, d in checkpoint_depth.items()
        }

        # Base weights (same computation as directed_route_memory_tree)
        base_gamma = float(max(0.0, getattr(self, "checkpoint_boost_gamma", 0.5)))
        min_relay_weight = float(
            max(0.0, getattr(self, "checkpoint_min_relay_weight", 1.0))
        )
        topo_beta = float(max(0.0, getattr(self, "checkpoint_beta", 0.03)))
        trigger_threshold = float(
            max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
        )
        arrival_target = float(
            max(
                trigger_threshold,
                getattr(self, "checkpoint_target_arrival", 0.15),
            )
        )
        relay_margin = float(max(1.0, getattr(self, "checkpoint_relay_margin", 1.15)))
        max_relay_weight = float(
            max(1.0, getattr(self, "checkpoint_max_relay_weight", 4.0))
        )
        path_decay = float(max(1e-6, getattr(self, "goal_map_path_decay", 0.97)))

        children_map: Dict[int, List[int]] = {cp: [] for cp in selected_order}
        for cp_idx, parent_idx in parent_map.items():
            if parent_idx is not None:
                children_map.setdefault(int(parent_idx), []).append(int(cp_idx))

        base_weights: Dict[int, float] = {}
        for cp_idx in selected_order:
            d_topo = float(goal_distances.get(cp_idx, math.inf))
            base_weight = (
                base_gamma * math.exp(-topo_beta * d_topo)
                if math.isfinite(d_topo)
                else 0.0
            )
            child_depth_gaps = [
                max(
                    1.0,
                    float(goal_distances.get(int(child_idx), d_topo)) - float(d_topo),
                )
                for child_idx in children_map.get(cp_idx, [])
            ]
            max_child_gap = max(child_depth_gaps) if child_depth_gaps else 0.0
            adaptive_floor = (
                (arrival_target * relay_margin) / (path_decay ** max_child_gap)
                if max_child_gap > 0.0
                else 0.0
            )
            leaf_floor = min_relay_weight if max_child_gap <= 0.0 else 0.0
            base_weights[cp_idx] = float(
                min(
                    max_relay_weight,
                    max(base_weight, adaptive_floor, leaf_floor),
                )
            )

        # Count total observed goal contacts for episode_count
        goal_contact_steps = getattr(self, "goal_contact_steps", {})
        raw_goal_steps = goal_contact_steps.get(goal_name_norm, [])
        episode_count = len(raw_goal_steps)

        # --- 8. assemble return ---
        route_data: Dict[str, Any] = {
            "mode": "multi_route_memory_graph",
            "selected_order": selected_order,
            "selected_set": set(selected_order),
            "direct_anchors": sorted(set(int(idx) for idx in direct_anchors)),
            "fallback_anchor": None,
            "used_fallback_anchor": False,
            "parent_map": parent_map,
            "direct_scores": direct_scores,
            "edge_scores": edge_scores,
            "goal_distances": goal_distances,
            "base_weights": base_weights,
            "route_count": sum(goal_link_counts.values()),
            "episode_count": episode_count,
            "route_examples": route_examples,
            # Rich graph fields (for later stages)
            "downstream_targets": {
                k: sorted(v) for k, v in downstream_targets_pruned.items()
            },
            "checkpoint_depth": dict(checkpoint_depth),
            "edge_counts": dict(edge_counts),
            "edge_direction_detail": dict(edge_direction_detail),
            "goal_link_counts": dict(goal_link_counts),
        }

        route_log = self._format_route_selected_checkpoint_log(route_data)
        branch_count = sum(
            1 for targets in downstream_targets_pruned.values()
            if len(targets) > 1
        )
        graph_log = (
            f"multi_route_graph="
            f"cp_cp_edges={len(cp_cp_edges)},cp_goal_edges={sum(1 for k in cp_goal_edges if str(k[2]) == goal_name_norm)},"
            f"reachable={len(reachable)},branches={branch_count},"
            f"max_depth={max(checkpoint_depth.values()) if checkpoint_depth else 0}"
        )
        return route_data, f"{route_log},{graph_log}"

    def _compute_route_selected_checkpoint_tree(
        self,
        reward_rcn,
        goal_seed: torch.Tensor,
        checkpoint_seed_bank: List[tuple[int, torch.Tensor]],
        checkpoint_seed_support_id_bank: Dict[int, List[int]],
        candidate_checkpoint_ids: Optional[List[int]],
        path_constraint: Optional[torch.Tensor],
        local_kernel: Optional[torch.Tensor],
        recurrent_weights_max: torch.Tensor,
        arrival_buffer_steps: int,
        goal_name: Optional[str] = None,
        goal_x: Optional[float] = None,
        goal_z: Optional[float] = None,
        goal_radius: Optional[float] = None,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        """
        Select goal-relevant checkpoints on the full denoised graph, then assign
        one downstream parent per selected checkpoint.
        """
        if not checkpoint_seed_bank:
            return None, "route_select=no_checkpoints"

        candidate_ids = (
            {int(idx) for idx in candidate_checkpoint_ids}
            if candidate_checkpoint_ids is not None
            else None
        )
        checkpoint_seed_map = {
            int(checkpoint_idx): checkpoint_seed
            for checkpoint_idx, checkpoint_seed in checkpoint_seed_bank
            if candidate_ids is None or int(checkpoint_idx) in candidate_ids
        }
        if not checkpoint_seed_map:
            return None, "route_select=no_candidate_checkpoints"

        def _fallback_to_directed(reason: str):
            if goal_x is None or goal_z is None:
                return None, None
            fallback_route_data, fallback_route_log = (
                self._compute_directed_route_memory_tree(
                    goal_x=float(goal_x),
                    goal_z=float(goal_z),
                    goal_radius=float(max(0.0, goal_radius or 0.0)),
                    checkpoint_seed_bank=checkpoint_seed_bank,
                    checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                    candidate_checkpoint_ids=candidate_checkpoint_ids,
                    start_step=int(start_step),
                    end_step=end_step,
                    target_device=reward_rcn.w_in.device,
                    goal_name=goal_name,
                )
            )
            if fallback_route_data is None:
                return None, None
            return (
                fallback_route_data,
                f"{fallback_route_log},fallback_from={reason}",
            )

        topology_neighbors, topology_log = self._build_weighted_goal_map_topology_neighbors(
            reward_rcn=reward_rcn,
            path_constraint=path_constraint,
        )
        if topology_neighbors is None:
            fallback_route_data, fallback_route_log = _fallback_to_directed(
                f"graph_{topology_log}"
            )
            if fallback_route_data is not None:
                return fallback_route_data, fallback_route_log
            return None, f"route_select=disabled({topology_log})"

        selection_rcn = copy.deepcopy(reward_rcn)
        selection_rcn.goal_map_checkpoint_seeds = None
        selection_rcn.goal_map_checkpoint_presence = None
        selection_rcn.goal_map_checkpoint_weights = None
        selection_rcn.goal_map_checkpoint_threshold = 0.0
        selection_rcn.goal_map_seed_gain = 1.0
        selection_rcn.goal_map_checkpoint_scale = 0.0
        selection_rcn.goal_map_path_constraint = path_constraint
        selection_rcn.goal_map_neighbor_constraint = local_kernel
        selection_rcn.goal_map_spatial_constraint = (
            local_kernel if local_kernel is not None else path_constraint
        )

        route_device = selection_rcn.w_in.device
        recurrent_weights_max = recurrent_weights_max.to(route_device)
        local_transition = selection_rcn._prepare_replay_transition(
            recurrent_weights_max
        )
        backward_transition = selection_rcn._prepare_goal_map_path_transition(
            local_transition
        )
        forward_transition = selection_rcn._row_max_normalize(
            torch.transpose(backward_transition, 0, 1)
        )
        neighbor_transition = selection_rcn._prepare_goal_map_neighbor_transition(
            recurrent_weights_max
        )

        support_fraction = float(
            max(1e-4, getattr(self, "goal_map_route_selection_support_fraction", 0.15))
        )
        overlap_threshold = float(
            max(0.0, getattr(self, "goal_map_route_selection_overlap_threshold", 0.08))
        )
        goal_cloud = self._build_goal_map_selection_cloud(
            selection_rcn,
            backward_transition,
            neighbor_transition,
            goal_seed.to(route_device, dtype=torch.float32),
        )
        goal_support = set(
            self._cloud_support_indices(goal_cloud, threshold_fraction=support_fraction)
        )
        goal_seed_support = self._relay_support_indices(
            goal_seed,
            topk=16,
            threshold_fraction=0.35,
        )
        if not goal_seed_support:
            goal_seed_support = list(goal_support)
        if not goal_support:
            goal_support = set(goal_seed_support)
        if not goal_support:
            fallback_route_data, fallback_route_log = _fallback_to_directed(
                f"graph_no_goal_support_{topology_log}"
            )
            if fallback_route_data is not None:
                return fallback_route_data, fallback_route_log
            return None, f"route_select=disabled(no_goal_support,{topology_log})"

        route_buffers = {}
        forward_support = {}
        backward_support = {}
        direct_scores = {}
        goal_distances = {}
        per_node_dist = {}
        seed_supports = {}

        for checkpoint_idx, checkpoint_seed in checkpoint_seed_map.items():
            doorway_ids = checkpoint_seed_support_id_bank.get(int(checkpoint_idx), [])
            route_buffers[int(checkpoint_idx)] = self._expand_node_ids_with_kernel(
                doorway_ids,
                local_kernel,
                steps=arrival_buffer_steps,
            )
            extra_ids = route_buffers[int(checkpoint_idx)] or doorway_ids

            checkpoint_seed_support = doorway_ids or self._relay_support_indices(
                checkpoint_seed,
                topk=16,
                threshold_fraction=0.35,
            )
            if extra_ids:
                checkpoint_seed_support = sorted(
                    {int(idx) for idx in checkpoint_seed_support}.union(
                        int(idx) for idx in extra_ids
                    )
                )
            seed_supports[int(checkpoint_idx)] = checkpoint_seed_support

            dist_map = self._pc_graph_weighted_distances_from_sources(
                topology_neighbors,
                checkpoint_seed_support,
            )
            per_node_dist[int(checkpoint_idx)] = dist_map
            goal_distances[int(checkpoint_idx)] = float(
                min((dist_map[idx] for idx in goal_seed_support), default=math.inf)
            )

            checkpoint_seed_t = checkpoint_seed.to(route_device, dtype=torch.float32)
            forward_cloud = self._build_goal_map_selection_cloud(
                selection_rcn,
                forward_transition,
                neighbor_transition,
                checkpoint_seed_t,
            )
            backward_cloud = self._build_goal_map_selection_cloud(
                selection_rcn,
                backward_transition,
                neighbor_transition,
                checkpoint_seed_t,
            )
            forward_support[int(checkpoint_idx)] = set(
                self._cloud_support_indices(
                    forward_cloud,
                    threshold_fraction=support_fraction,
                    extra_ids=extra_ids,
                )
            )
            backward_support[int(checkpoint_idx)] = set(
                self._cloud_support_indices(
                    backward_cloud,
                    threshold_fraction=support_fraction,
                    extra_ids=extra_ids,
                )
            )
            direct_scores[int(checkpoint_idx)] = float(
                len(forward_support[int(checkpoint_idx)] & goal_support)
                / max(1, len(forward_support[int(checkpoint_idx)]))
            )

        selected = set()
        parent_map = {}
        direct_anchors = []
        for checkpoint_idx, score in sorted(
            direct_scores.items(),
            key=lambda item: (goal_distances.get(int(item[0]), math.inf), -float(item[1]), int(item[0])),
        ):
            if not math.isfinite(goal_distances.get(int(checkpoint_idx), math.inf)):
                continue
            if score >= overlap_threshold:
                selected.add(int(checkpoint_idx))
                parent_map[int(checkpoint_idx)] = None
                direct_anchors.append(int(checkpoint_idx))

        used_fallback_anchor = False
        fallback_anchor = None
        if not selected:
            reachable = [
                (float(goal_distances.get(int(checkpoint_idx), math.inf)), int(checkpoint_idx))
                for checkpoint_idx in checkpoint_seed_map
                if math.isfinite(goal_distances.get(int(checkpoint_idx), math.inf))
            ]
            if reachable:
                reachable.sort(key=lambda item: (item[0], item[1]))
                fallback_anchor = int(reachable[0][1])
                used_fallback_anchor = True
                selected.add(fallback_anchor)
                parent_map[fallback_anchor] = None
            else:
                fallback_route_data, fallback_route_log = _fallback_to_directed(
                    f"graph_no_reachable_anchor_{topology_log}"
                )
                if fallback_route_data is not None:
                    return fallback_route_data, fallback_route_log
                return None, f"route_select=disabled(no_reachable_anchor,{topology_log})"

        edge_scores = {}
        while True:
            additions = []
            for checkpoint_idx in checkpoint_seed_map:
                checkpoint_idx = int(checkpoint_idx)
                if checkpoint_idx in selected:
                    continue
                d_i = goal_distances.get(checkpoint_idx, math.inf)
                if not math.isfinite(d_i):
                    continue
                best_parent = None
                best_score = -1.0
                for parent_idx in selected:
                    d_j = goal_distances.get(int(parent_idx), math.inf)
                    if not math.isfinite(d_j) or d_j >= d_i:
                        continue
                    score = float(
                        len(
                            forward_support[checkpoint_idx]
                            & backward_support[int(parent_idx)]
                        )
                        / max(1, len(forward_support[checkpoint_idx]))
                    )
                    if score > best_score:
                        best_score = score
                        best_parent = int(parent_idx)
                if best_parent is not None and best_score >= overlap_threshold:
                    additions.append(
                        (
                            float(d_i),
                            -float(best_score),
                            int(checkpoint_idx),
                            int(best_parent),
                            float(best_score),
                        )
                    )
            if not additions:
                break
            additions.sort()
            for _, _, checkpoint_idx, best_parent, best_score in additions:
                if checkpoint_idx in selected:
                    continue
                selected.add(int(checkpoint_idx))
                parent_map[int(checkpoint_idx)] = int(best_parent)
                edge_scores[(int(checkpoint_idx), int(best_parent))] = float(best_score)

        selected_order = sorted(
            selected,
            key=lambda checkpoint_idx: (
                float(goal_distances.get(int(checkpoint_idx), math.inf)),
                int(checkpoint_idx),
            ),
        )

        dist_maps = {"goal": self._pc_graph_weighted_distances_from_sources(
            topology_neighbors,
            goal_seed_support,
        )}
        for checkpoint_idx in selected_order:
            dist_maps[int(checkpoint_idx)] = per_node_dist[int(checkpoint_idx)]

        base_gamma = float(max(0.0, getattr(self, "checkpoint_boost_gamma", 0.5)))
        min_relay_weight = float(
            max(0.0, getattr(self, "checkpoint_min_relay_weight", 1.0))
        )
        topo_beta = float(max(0.0, getattr(self, "checkpoint_beta", 0.03)))
        trigger_threshold = float(
            max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
        )
        arrival_target = float(
            max(
                trigger_threshold,
                getattr(self, "checkpoint_target_arrival", 0.15),
            )
        )
        relay_margin = float(max(1.0, getattr(self, "checkpoint_relay_margin", 1.15)))
        max_relay_weight = float(
            max(1.0, getattr(self, "checkpoint_max_relay_weight", 4.0))
        )
        path_decay = float(max(1e-6, getattr(self, "goal_map_path_decay", 0.97)))
        children = {int(checkpoint_idx): [] for checkpoint_idx in selected_order}
        for checkpoint_idx, parent_idx in parent_map.items():
            if parent_idx is None:
                continue
            children[int(parent_idx)].append(int(checkpoint_idx))

        base_weights = {}
        for checkpoint_idx in selected_order:
            d_topo = float(goal_distances.get(int(checkpoint_idx), math.inf))
            base_weight = (
                base_gamma * math.exp(-topo_beta * d_topo)
                if math.isfinite(d_topo)
                else 0.0
            )
            child_gaps = []
            for child_idx in children.get(int(checkpoint_idx), []):
                child_support = seed_supports.get(int(child_idx), [])
                if not child_support:
                    continue
                gap = min(
                    (
                        dist_maps[int(checkpoint_idx)][node_idx]
                        for node_idx in child_support
                    ),
                    default=math.inf,
                )
                if math.isfinite(gap):
                    child_gaps.append(float(gap))
            max_child_gap = max(child_gaps) if child_gaps else 0.0
            adaptive_floor = (
                (arrival_target * relay_margin) / (path_decay ** max_child_gap)
                if max_child_gap > 0.0
                else 0.0
            )
            leaf_floor = min_relay_weight if max_child_gap <= 0.0 else 0.0
            base_weights[int(checkpoint_idx)] = float(
                min(
                    max_relay_weight,
                    max(base_weight, adaptive_floor, leaf_floor),
                )
            )

        route_data = {
            "selected_order": selected_order,
            "selected_set": set(selected_order),
            "direct_anchors": direct_anchors,
            "fallback_anchor": fallback_anchor,
            "used_fallback_anchor": used_fallback_anchor,
            "parent_map": parent_map,
            "direct_scores": direct_scores,
            "edge_scores": edge_scores,
            "goal_distances": goal_distances,
            "base_weights": base_weights,
            "topology_log": topology_log,
        }
        route_log = self._format_route_selected_checkpoint_log(route_data)
        return route_data, f"{route_log},{topology_log}"

    def _get_goal_map_world_obstacles(self):
        """Return world obstacles used to block local reward spread across walls."""
        if not bool(getattr(self, "goal_map_spatial_obstacle_block", True)):
            return []
        cache = getattr(self, "_goal_map_world_obstacles_cache", None)
        if cache is not None:
            return cache
        try:
            cache = list(get_world_obstacles(self.world_name))
        except Exception:
            cache = []
        self._goal_map_world_obstacles_cache = cache
        return cache

    @staticmethod
    def _obstacle_rect_bounds(obstacle):
        """Return normalized rectangle bounds as xmin, xmax, ymin, ymax."""
        bounds = obstacle.get("bounds")
        if not bounds or obstacle.get("type") != "rectangle":
            return None
        xmin = min(bounds[0][0], bounds[1][0])
        xmax = max(bounds[0][0], bounds[1][0])
        ymin = min(bounds[0][1], bounds[1][1])
        ymax = max(bounds[0][1], bounds[1][1])
        return float(xmin), float(xmax), float(ymin), float(ymax)

    def _infer_checkpoint_support_region(
        self,
        checkpoint_x: float,
        checkpoint_y: float,
    ):
        """
        Infer a doorway-span support region for one checkpoint wedged between walls.

        The region is modeled as a short line segment with a small half-width,
        so any point inside the doorway slit counts as arriving at the checkpoint.
        """
        obstacles = self._get_goal_map_world_obstacles()
        if not obstacles:
            return None, "doorway=no_obstacles"

        cache = getattr(self, "_checkpoint_support_region_cache", None)
        if cache is None:
            cache = {}
            self._checkpoint_support_region_cache = cache
        cache_key = (
            str(getattr(self, "world_name", "")),
            round(float(checkpoint_x), 3),
            round(float(checkpoint_y), 3),
        )
        if cache_key in cache:
            region = cache[cache_key]
            if region is None:
                return None, "doorway=no_pair"
            if region.get("kind") == "segment":
                p0 = region["p0"]
                p1 = region["p1"]
                return (
                    region,
                    "doorway="
                    f"{region.get('orientation', 'segment')}("
                    f"p0=({p0[0]:.2f},{p0[1]:.2f}),"
                    f"p1=({p1[0]:.2f},{p1[1]:.2f}),"
                    f"half_w={region.get('half_width', 0.0):.2f})",
                )
            return region, "doorway=point"

        tol = 0.25
        candidates = []
        rects = []
        for obstacle in obstacles:
            rect = self._obstacle_rect_bounds(obstacle)
            if rect is None:
                continue
            xmin, xmax, ymin, ymax = rect
            width = max(1e-6, xmax - xmin)
            height = max(1e-6, ymax - ymin)
            orientation = "vertical" if height >= width else "horizontal"
            rects.append((orientation, xmin, xmax, ymin, ymax))

        # Add arena boundary walls as virtual obstacles so checkpoints at
        # wall endpoints near the arena edge can form doorway pairs.
        arena_half = 10.0
        try:
            from core.robot.webots_worlds import get_world_config
            wname = str(getattr(self, "world_name", ""))
            if wname:
                wsize = get_world_config(wname).get("size", [20.0, 20.0])
                arena_half = float(max(wsize[0], wsize[1])) / 2.0
        except Exception:
            pass
        bw = 0.06  # arena wall thickness
        rects.append(("vertical",  -arena_half - bw, -arena_half, -arena_half, arena_half))
        rects.append(("vertical",   arena_half, arena_half + bw, -arena_half, arena_half))
        rects.append(("horizontal", -arena_half, arena_half, -arena_half - bw, -arena_half))
        rects.append(("horizontal", -arena_half, arena_half,  arena_half, arena_half + bw))

        for idx_a in range(len(rects)):
            orient_a, ax0, ax1, ay0, ay1 = rects[idx_a]
            for idx_b in range(idx_a + 1, len(rects)):
                orient_b, bx0, bx1, by0, by1 = rects[idx_b]
                if orient_a != orient_b:
                    continue
                if orient_a == "vertical":
                    overlap_min = max(ax0, bx0)
                    overlap_max = min(ax1, bx1)
                    if overlap_max < overlap_min - 1e-6:
                        continue
                    if ay1 <= by0:
                        gap_min, gap_max = ay1, by0
                    elif by1 <= ay0:
                        gap_min, gap_max = by1, ay0
                    else:
                        continue
                    if not (gap_min - tol <= checkpoint_y <= gap_max + tol):
                        continue
                    if not (overlap_min - tol <= checkpoint_x <= overlap_max + tol):
                        continue
                    axis_center = 0.5 * (overlap_min + overlap_max)
                    gap_mid = 0.5 * (gap_min + gap_max)
                    half_width = max(0.15, 0.5 * max(0.0, overlap_max - overlap_min) + 0.05)
                    score = abs(checkpoint_x - axis_center) + 0.25 * abs(checkpoint_y - gap_mid)
                    candidates.append(
                        (
                            score,
                            {
                                "kind": "segment",
                                "orientation": "vertical",
                                "p0": (axis_center, gap_min),
                                "p1": (axis_center, gap_max),
                                "half_width": half_width,
                            },
                        )
                    )
                else:
                    overlap_min = max(ay0, by0)
                    overlap_max = min(ay1, by1)
                    if overlap_max < overlap_min - 1e-6:
                        continue
                    if ax1 <= bx0:
                        gap_min, gap_max = ax1, bx0
                    elif bx1 <= ax0:
                        gap_min, gap_max = bx1, ax0
                    else:
                        continue
                    if not (gap_min - tol <= checkpoint_x <= gap_max + tol):
                        continue
                    if not (overlap_min - tol <= checkpoint_y <= overlap_max + tol):
                        continue
                    axis_center = 0.5 * (overlap_min + overlap_max)
                    gap_mid = 0.5 * (gap_min + gap_max)
                    half_width = max(0.15, 0.5 * max(0.0, overlap_max - overlap_min) + 0.05)
                    score = abs(checkpoint_y - axis_center) + 0.25 * abs(checkpoint_x - gap_mid)
                    candidates.append(
                        (
                            score,
                            {
                                "kind": "segment",
                                "orientation": "horizontal",
                                "p0": (gap_min, axis_center),
                                "p1": (gap_max, axis_center),
                                "half_width": half_width,
                            },
                        )
                    )

        region = min(candidates, key=lambda item: item[0])[1] if candidates else None

        # Fallback: if no wall pair found, find the nearest wall endpoint and
        # create a perpendicular support region through the checkpoint.
        if region is None and rects:
            best_dist = float("inf")
            best_region = None
            for orient, xmin, xmax, ymin, ymax in rects:
                cx_wall = 0.5 * (xmin + xmax)
                cy_wall = 0.5 * (ymin + ymax)
                dist = math.hypot(checkpoint_x - cx_wall, checkpoint_y - cy_wall)
                if dist >= best_dist:
                    continue
                # Build a segment perpendicular to this wall, through the checkpoint.
                wall_w = max(1e-6, xmax - xmin)
                wall_h = max(1e-6, ymax - ymin)
                half_width = max(0.15, 0.5 * min(wall_w, wall_h) + 0.05)
                if orient == "horizontal":
                    # Wall runs along x → doorway gap runs along y
                    # Perpendicular segment at checkpoint x, spanning wall_h.
                    seg_len = max(0.5, 2.0 * half_width)
                    best_region = {
                        "kind": "segment",
                        "orientation": "vertical",
                        "p0": (checkpoint_x, checkpoint_y - seg_len),
                        "p1": (checkpoint_x, checkpoint_y + seg_len),
                        "half_width": half_width,
                    }
                else:
                    # Wall runs along y → doorway gap runs along x
                    seg_len = max(0.5, 2.0 * half_width)
                    best_region = {
                        "kind": "segment",
                        "orientation": "horizontal",
                        "p0": (checkpoint_x - seg_len, checkpoint_y),
                        "p1": (checkpoint_x + seg_len, checkpoint_y),
                        "half_width": half_width,
                    }
                best_dist = dist
            region = best_region

        cache[cache_key] = region
        if region is None:
            return None, "doorway=no_pair"
        p0 = region["p0"]
        p1 = region["p1"]
        return (
            region,
            "doorway="
            f"{region.get('orientation', 'segment')}("
            f"p0=({p0[0]:.2f},{p0[1]:.2f}),"
            f"p1=({p1[0]:.2f},{p1[1]:.2f}),"
            f"half_w={region.get('half_width', 0.0):.2f})",
        )

    @staticmethod
    def _support_region_distance_sq(
        x,
        y,
        support_region,
        default_point=None,
    ) -> float:
        """Return squared distance from a point to a point/doorway support region."""
        if support_region is None:
            if default_point is None:
                return 0.0
            dx = float(x) - float(default_point[0])
            dy = float(y) - float(default_point[1])
            return dx * dx + dy * dy
        if support_region.get("kind") != "segment":
            if default_point is None:
                return 0.0
            dx = float(x) - float(default_point[0])
            dy = float(y) - float(default_point[1])
            return dx * dx + dy * dy

        x0, y0 = support_region["p0"]
        x1, y1 = support_region["p1"]
        vx = float(x1) - float(x0)
        vy = float(y1) - float(y0)
        seg_len_sq = max(1e-12, vx * vx + vy * vy)
        t = ((float(x) - float(x0)) * vx + (float(y) - float(y0)) * vy) / seg_len_sq
        t = min(1.0, max(0.0, t))
        proj_x = float(x0) + t * vx
        proj_y = float(y0) + t * vy
        dx = float(x) - proj_x
        dy = float(y) - proj_y
        dist = math.hypot(dx, dy)
        half_width = float(max(0.0, support_region.get("half_width", 0.0)))
        dist = max(0.0, dist - half_width)
        return dist * dist

    @staticmethod
    def _support_region_distance_sq_torch(
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        support_region,
        default_point=None,
    ) -> torch.Tensor:
        """Vectorized squared distance to a point/doorway support region."""
        if support_region is None or support_region.get("kind") != "segment":
            if default_point is None:
                return torch.zeros_like(x_t)
            dx = x_t - float(default_point[0])
            dy = y_t - float(default_point[1])
            return dx * dx + dy * dy

        x0, y0 = support_region["p0"]
        x1, y1 = support_region["p1"]
        vx = float(x1) - float(x0)
        vy = float(y1) - float(y0)
        seg_len_sq = max(1e-12, vx * vx + vy * vy)
        t = ((x_t - float(x0)) * vx + (y_t - float(y0)) * vy) / seg_len_sq
        t = torch.clamp(t, min=0.0, max=1.0)
        proj_x = float(x0) + t * vx
        proj_y = float(y0) + t * vy
        dx = x_t - proj_x
        dy = y_t - proj_y
        dist = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0))
        half_width = float(max(0.0, support_region.get("half_width", 0.0)))
        dist = torch.clamp(dist - half_width, min=0.0)
        return dist * dist

    @staticmethod
    def _support_region_bounds(support_region):
        """Return rectangle bounds covering a doorway support region."""
        if support_region is None or support_region.get("kind") != "segment":
            return None
        x0, y0 = support_region["p0"]
        x1, y1 = support_region["p1"]
        half_width = float(max(0.0, support_region.get("half_width", 0.0)))
        xmin = min(float(x0), float(x1)) - half_width
        xmax = max(float(x0), float(x1)) + half_width
        ymin = min(float(y0), float(y1)) - half_width
        ymax = max(float(y0), float(y1)) + half_width
        return [[xmin, ymin], [xmax, ymax]]

    @staticmethod
    def _support_region_frame(support_region):
        """Return the centerline frame for a doorway-span support region."""
        if support_region is None or support_region.get("kind") != "segment":
            return None
        x0, y0 = support_region["p0"]
        x1, y1 = support_region["p1"]
        vx = float(x1) - float(x0)
        vy = float(y1) - float(y0)
        norm = math.hypot(vx, vy)
        if norm <= 1e-9:
            return None
        tx = vx / norm
        ty = vy / norm
        nx = -ty
        ny = tx
        return {
            "center": (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1))),
            "tangent": (tx, ty),
            "normal": (nx, ny),
            "half_width": float(max(0.0, support_region.get("half_width", 0.0))),
        }

    def _build_checkpoint_directional_replay_mask(
        self,
        checkpoint_idx: int,
        support_mask: torch.Tensor,
        support_region,
        downstream_field: Optional[torch.Tensor] = None,
        arrival_mask: Optional[torch.Tensor] = None,
        downstream_support_ids: Optional[List[int]] = None,
        downstream_target_xy: Optional[Tuple[float, float]] = None,
        target_device=None,
        strict_upstream_only: bool = False,
    ):
        """
        Restrict checkpoint relay to the upstream side of the checkpoint using
        doorway geometry only.

        The downstream room is determined from the next route target
        (next checkpoint or the goal). The returned mask keeps only PCs on the
        upstream side plus the doorway band itself.
        """
        if target_device is None:
            target_device = self.device
        base_mask = torch.as_tensor(
            support_mask, dtype=torch.bool, device=target_device
        ).view(-1)
        if not torch.any(base_mask):
            return base_mask, "dir=empty"

        frame = self._support_region_frame(support_region)
        if frame is None:
            return base_mask, "dir=no_segment"

        centers, visited, spatial_log = self._get_unified_pc_spatial_support(target_device)
        if centers is None or visited is None:
            return base_mask, spatial_log.replace("spatial=", "dir=")

        if downstream_target_xy is None:
            return base_mask, "dir=no_target"

        cx, cy = frame["center"]
        nx, ny = frame["normal"]
        doorway_keep = max(
            0.15,
            float(frame.get("half_width", 0.0))
            + float(getattr(self, "goal_map_checkpoint_directional_band", 0.12)),
        )
        signed = ((centers[:, 0] - cx) * nx) + ((centers[:, 1] - cy) * ny)
        target_x = float(downstream_target_xy[0])
        target_y = float(downstream_target_xy[1])
        target_signed = ((target_x - cx) * nx) + ((target_y - cy) * ny)
        if abs(float(target_signed)) <= max(0.05, 0.5 * doorway_keep):
            return base_mask, "dir=target:ambiguous"

        downstream_sign = 1.0 if float(target_signed) > 0.0 else -1.0
        directional_mask = base_mask.clone()
        valid_base = base_mask & visited
        if torch.any(valid_base):
            directional_mask[valid_base] = (
                (signed[valid_base] * downstream_sign) <= doorway_keep
            )
        pruned = int(torch.count_nonzero(base_mask & ~directional_mask).item())
        side_txt = "+" if downstream_sign > 0.0 else "-"
        strict_txt = ",strict=1" if strict_upstream_only else ""
        return (
            directional_mask,
            f"dir=geom_target:{side_txt},pruned={pruned},band={doorway_keep:.2f}{strict_txt}",
        )

    @staticmethod
    def _segment_intersects_rectangle(
        p0,
        p1,
        bounds,
        margin: float = 0.0,
    ) -> bool:
        """Liang-Barsky segment-vs-rectangle test for axis-aligned obstacle bounds."""
        xmin = min(bounds[0][0], bounds[1][0]) - margin
        xmax = max(bounds[0][0], bounds[1][0]) + margin
        ymin = min(bounds[0][1], bounds[1][1]) - margin
        ymax = max(bounds[0][1], bounds[1][1]) + margin

        x0 = float(p0[0])
        y0 = float(p0[1])
        x1 = float(p1[0])
        y1 = float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        t0 = 0.0
        t1 = 1.0

        for p, q in (
            (-dx, x0 - xmin),
            (dx, xmax - x0),
            (-dy, y0 - ymin),
            (dy, ymax - y0),
        ):
            if abs(p) <= 1e-12:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
        return True

    def _candidate_edge_crosses_obstacle(
        self,
        p0,
        p1,
        obstacles,
        margin: float,
    ) -> bool:
        """Return True if a local kernel edge would pass through any world obstacle."""
        for obstacle in obstacles:
            if obstacle.get("type") != "rectangle":
                continue
            bounds = obstacle.get("bounds")
            if not bounds:
                continue
            if self._segment_intersects_rectangle(p0, p1, bounds, margin=margin):
                return True
        return False

    def _compute_directional_clearance(self, boundaries_rolled: torch.Tensor) -> torch.Tensor:
        """
        Compute one obstacle-clearance score per head-direction sector.

        Using the minimum LiDAR ray is overly conservative around narrow openings:
        one ray grazing a door frame can zero the whole sector. A low percentile
        keeps sectors unsafe when most rays are blocked, but still allows valid
        doorway directions when enough of the sector is open.
        """
        num_points_per_hd = len(boundaries_rolled) // self.n_hd
        if num_points_per_hd <= 0:
            return torch.zeros(self.n_hd, device=self.device, dtype=self.dtype)
        percentile = float(
            min(100.0, max(0.0, getattr(self, "exploit_direction_clearance_percentile", 20.0)))
        )
        q = percentile / 100.0
        usable = boundaries_rolled[: self.n_hd * num_points_per_hd]
        usable = torch.nan_to_num(
            usable.to(device=self.device, dtype=self.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        sectors = usable.view(self.n_hd, num_points_per_hd)
        if q <= 0.0:
            clearances = torch.min(sectors, dim=1).values
        else:
            clearances = torch.quantile(sectors, q, dim=1)
        clearances = torch.nan_to_num(clearances, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(clearances, min=0.0).to(device=self.device, dtype=self.dtype)

    def _get_unified_checkpoint_door_safety_profile(
        self,
        base_safe_distance: float,
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Relax the hard exploit safety threshold for the active checkpoint doorway.

        The normal clearance threshold can wrongly zero the doorway-facing HD bin
        because that sector also sees the door frame. For unified exploit, use the
        current route-selected checkpoint sequence to identify the next doorway and
        apply a smaller threshold to the aligned heading bin (plus immediate neighbors).
        """
        device = self.device if target_device is None else target_device
        dtype = self.dtype if target_dtype is None else target_dtype
        thresholds = torch.full(
            (self.n_hd,),
            float(max(0.0, base_safe_distance)),
            dtype=dtype,
            device=device,
        )
        doorway_bins = torch.zeros(self.n_hd, dtype=torch.bool, device=device)
        self._last_unified_active_checkpoint_target = None

        if not getattr(self, "use_unified_multiscale", False):
            return thresholds, doorway_bins, "door_safety=no_unified_mode"
        if not getattr(self, "detected_doorways", None):
            return thresholds, doorway_bins, "door_safety=no_checkpoints"

        relaxed_distance = float(
            max(
                0.05,
                min(
                    float(base_safe_distance),
                    getattr(self, "unified_checkpoint_door_safe_distance", 0.35),
                ),
            )
        )
        if relaxed_distance >= float(base_safe_distance):
            return thresholds, doorway_bins, "door_safety=no_relaxation"

        route_cache = getattr(self, "_unified_exploit_route_stage_cache", None)
        checkpoint_ids: List[int] = []
        if isinstance(route_cache, dict):
            route_order = [int(idx) for idx in route_cache.get("route_order_far_to_near", [])]
            progress = int(route_cache.get("progress", 0))
            checkpoint_ids = route_order[progress:]

        if not checkpoint_ids:
            debug_log = getattr(getattr(self, "unified_rcn", None), "goal_map_debug_log", "")
            checkpoint_ids = self._extract_selected_checkpoint_order_from_goal_map_log(debug_log)

        valid_candidates = []
        curr_x, curr_y = self._current_planar_xy()
        curr_xy = torch.tensor([curr_x, curr_y], dtype=torch.float32, device=device)
        for checkpoint_idx in checkpoint_ids:
            if 0 <= int(checkpoint_idx) < len(self.detected_doorways):
                cx, cy = self.detected_doorways[int(checkpoint_idx)]
                delta = torch.tensor(
                    [float(cx) - float(curr_x), float(cy) - float(curr_y)],
                    dtype=torch.float32,
                    device=device,
                )
                dist = float(torch.norm(delta).item())
                valid_candidates.append((dist, int(checkpoint_idx), delta))

        if not valid_candidates:
            return thresholds, doorway_bins, "door_safety=no_active_checkpoint"

        valid_candidates.sort(key=lambda item: (item[0], item[1]))
        nearest_dist, checkpoint_idx, delta = valid_candidates[0]
        self._last_unified_active_checkpoint_target = int(checkpoint_idx)
        if nearest_dist <= 1e-6:
            cx, cy = self.detected_doorways[int(checkpoint_idx)]
            region, _ = self._infer_checkpoint_support_region(float(cx), float(cy))
            frame = self._support_region_frame(region)
            if frame is None:
                return thresholds, doorway_bins, f"door_safety=cp{checkpoint_idx}:degenerate"
            direction_vec = torch.tensor(
                [float(frame["normal"][0]), float(frame["normal"][1])],
                dtype=torch.float32,
                device=device,
            )
            delta = direction_vec

        angle = float(torch.atan2(delta[1], delta[0]).item())
        if angle < 0.0:
            angle += 2.0 * np.pi
        center_bin = self._angle_to_hd_bin(angle)
        bin_radius = int(max(0, getattr(self, "unified_checkpoint_door_bin_radius", 1)))
        marked_bins = []
        for offset in range(-bin_radius, bin_radius + 1):
            bin_idx = (center_bin + offset) % int(self.n_hd)
            doorway_bins[bin_idx] = True
            thresholds[bin_idx] = relaxed_distance
            marked_bins.append(int(bin_idx))

        return (
            thresholds,
            doorway_bins,
            "door_safety="
            f"cp{checkpoint_idx},dist={nearest_dist:.2f},center_bin={center_bin},"
            f"bins={sorted(set(marked_bins))},base={float(base_safe_distance):.2f},"
            f"relaxed={relaxed_distance:.2f},curr_xy=({float(curr_xy[0].item()):.2f},{float(curr_xy[1].item()):.2f})",
        )

    def _get_hd_bin_angles(
        self,
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return one clockwise world-space angle per HD bin in radians."""
        device = self.device if target_device is None else target_device
        dtype = self.dtype if target_dtype is None else target_dtype
        return -torch.arange(
            self.n_hd,
            device=device,
            dtype=dtype,
        ) * (2.0 * np.pi / float(self.n_hd))

    def _angle_to_hd_bin(self, angle_rad: float) -> int:
        """Map a world-space angle back onto the clockwise HD bin index."""
        angle = float(angle_rad) % (2.0 * np.pi)
        bin_width = 2.0 * np.pi / float(self.n_hd)
        clockwise_angle = (-angle) % (2.0 * np.pi)
        return int(round(clockwise_angle / bin_width)) % int(self.n_hd)

    def _compute_circular_mean_action_angle(
        self,
        direction_rewards: torch.Tensor,
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
        fallback_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Convert one reward value per heading bin into a circular-mean action angle.

        The reward magnitudes are treated as weights on a circle. If the reward
        mass is degenerate or perfectly cancels out, the selector falls back to
        the strongest reward bin so exploit still commits to a concrete heading.
        """
        device = self.device if target_device is None else target_device
        dtype = self.dtype if target_dtype is None else target_dtype
        reward_weights = torch.as_tensor(direction_rewards, device=device, dtype=dtype)
        reward_weights = torch.nan_to_num(reward_weights, nan=0.0, posinf=0.0, neginf=0.0)
        reward_weights = torch.clamp(reward_weights, min=0.0)
        angles = self._get_hd_bin_angles(target_device=device, target_dtype=dtype)

        if reward_weights.numel() != self.n_hd:
            raise ValueError(
                f"Expected {self.n_hd} direction rewards, got {reward_weights.numel()}"
            )

        max_idx = int(torch.argmax(reward_weights).item())
        if fallback_idx is None:
            fallback_idx = max_idx
        fallback_idx = int(fallback_idx) % int(self.n_hd)

        total_reward = float(torch.sum(reward_weights).item())
        if total_reward <= 1e-9:
            action_angle = angles[fallback_idx]
            combined_vector = torch.stack(
                [torch.cos(action_angle), torch.sin(action_angle)]
            )
            selection_log = (
                "selection=circular_reward_mean "
                f"fallback=zero_reward idx={fallback_idx}"
            )
            return action_angle, combined_vector, selection_log

        normalized_rewards = reward_weights / total_reward
        combined_vector = torch.stack(
            [
                torch.sum(normalized_rewards * torch.cos(angles)),
                torch.sum(normalized_rewards * torch.sin(angles)),
            ]
        )

        if float(torch.norm(combined_vector).item()) < 1e-6:
            action_angle = angles[max_idx]
            combined_vector = torch.stack(
                [torch.cos(action_angle), torch.sin(action_angle)]
            )
            selection_log = (
                "selection=circular_reward_mean "
                f"fallback=cancelled idx={max_idx} reward={reward_weights[max_idx].item():.3f}"
            )
            return action_angle, combined_vector, selection_log

        action_angle = torch.atan2(combined_vector[1], combined_vector[0])
        if action_angle < 0:
            action_angle += 2.0 * np.pi
        selection_log = (
            "selection=circular_reward_mean "
            f"reward_sum={total_reward:.3f} peak_idx={max_idx} "
            f"peak_reward={reward_weights[max_idx].item():.3f}"
        )
        return action_angle, combined_vector, selection_log

    def _aggregate_direction_reward_means(
        self,
        trajectory_returns: torch.Tensor,
        trajectory_metadata: List[Dict[str, Any]],
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Aggregate sampled trajectory returns into one mean reward per heading bin."""
        device = self.device if target_device is None else target_device
        dtype = self.dtype if target_dtype is None else target_dtype
        returns = torch.as_tensor(trajectory_returns, device=device, dtype=dtype)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        returns = torch.clamp(returns, min=0.0)

        reward_sums = torch.zeros(self.n_hd, device=device, dtype=dtype)
        reward_counts = torch.zeros(self.n_hd, device=device, dtype=dtype)
        for reward_value, meta in zip(returns, trajectory_metadata):
            direction_idx = int(meta.get("direction", 0)) % int(self.n_hd)
            reward_sums[direction_idx] += reward_value
            reward_counts[direction_idx] += 1.0

        nonzero_mask = reward_counts > 0.0
        direction_rewards = torch.zeros_like(reward_sums)
        direction_rewards[nonzero_mask] = (
            reward_sums[nonzero_mask] / reward_counts[nonzero_mask]
        )
        return direction_rewards

    def _get_unified_dense_hmap_source(self):
        """
        Return dense unified hmap data for exploit-time spatial reasoning.

        Prefer in-memory hmaps from the current run. If exploit is started from a
        saved network without an active learning session, fall back to dense hmaps
        saved on disk for the current environment/trial.
        """
        exploit_style_mode = getattr(self, "robot_mode", None) in {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }
        frozen_cache = getattr(self, "_frozen_unified_dense_hmaps_cache", None)
        if exploit_style_mode and isinstance(frozen_cache, dict):
            return frozen_cache

        hmap_ok = (
            bool(self.hmap_pcn_activities)
            and self.step_count > 0
            and not getattr(self, "lightweight_hmaps", False)
        )
        if hmap_ok:
            n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
            if exploit_style_mode:
                cache = {
                    "source": "memory_frozen",
                    "step_count": int(self.step_count),
                    "n_valid": n_valid,
                    "hmap_loc": self.hmap_loc[:n_valid].copy(),
                    "scale_acts": [
                        acts[:n_valid].detach().cpu() for acts in self.hmap_pcn_activities
                    ],
                    "log": f"dense_hmaps=memory_frozen(step={int(self.step_count)})",
                }
                self._frozen_unified_dense_hmaps_cache = cache
                return cache
            cache = getattr(self, "_live_unified_dense_hmaps_cache", None)
            if (
                isinstance(cache, dict)
                and cache.get("n_valid") == n_valid
                and cache.get("step_count") == int(self.step_count)
            ):
                return cache
            cache = {
                "source": "memory",
                "step_count": int(self.step_count),
                "n_valid": n_valid,
                "hmap_loc": self.hmap_loc[:n_valid].copy(),
                "scale_acts": [acts[:n_valid].detach().cpu() for acts in self.hmap_pcn_activities],
                "log": "dense_hmaps=memory",
            }
            self._live_unified_dense_hmaps_cache = cache
            return cache

        cache = getattr(self, "_saved_unified_dense_hmaps_cache", None)
        if cache is not None:
            return cache

        candidate_dirs = []
        primary_hmap_dir = os.path.normpath(os.path.abspath(self.hmap_dir))
        candidate_dirs.append(primary_hmap_dir)

        canonical_rel_dir = os.path.normpath(
            os.path.abspath(os.path.join("pkl", self.world_name, "hmaps"))
        )
        if canonical_rel_dir not in candidate_dirs:
            candidate_dirs.append(canonical_rel_dir)

        canonical_abs_dir = os.path.normpath(
            os.path.abspath(
                os.path.join(
                    PROJECT_ROOT,
                    "webots",
                    "controllers",
                    "msg_controller",
                    "pkl",
                    self.world_name,
                    "hmaps",
                )
            )
        )
        if canonical_abs_dir not in candidate_dirs:
            candidate_dirs.append(canonical_abs_dir)

        hmap_loc = None
        scale_acts = []
        selected_dir = None
        selected_note = "none"
        for candidate_dir in candidate_dirs:
            loc_path = os.path.join(candidate_dir, "hmap_loc.pkl")
            if not os.path.exists(loc_path):
                continue

            local_scale_acts = []
            try:
                with open(loc_path, "rb") as f:
                    local_hmap_loc = np.array(pickle.load(f))
                if local_hmap_loc.ndim != 2 or local_hmap_loc.shape[0] <= 1:
                    continue

                missing_scale = False
                for scale_def in self.scales:
                    scale_idx = scale_def["scale_index"]
                    scale_path = os.path.join(
                        candidate_dir, f"hmap_pcn_scale_{scale_idx}.pkl"
                    )
                    if not os.path.exists(scale_path):
                        missing_scale = True
                        break
                    with open(scale_path, "rb") as f:
                        acts = pickle.load(f)
                    if isinstance(acts, torch.Tensor):
                        acts_t = acts.detach().cpu().float()
                    else:
                        acts_t = torch.as_tensor(np.array(acts), dtype=torch.float32)
                    local_scale_acts.append(acts_t)

                if missing_scale or not local_scale_acts:
                    continue

                hmap_loc = local_hmap_loc
                scale_acts = local_scale_acts
                selected_dir = candidate_dir
                selected_note = (
                    "primary"
                    if candidate_dir == primary_hmap_dir
                    else "canonical_fallback"
                )
                break
            except Exception:
                continue

        if selected_dir is None or hmap_loc is None or not scale_acts:
            return None

        if not scale_acts:
            return None

        # Some saved hmaps include an initial zero row. Drop it when it carries no activity.
        if all(float(acts[0].abs().sum().item()) <= 1e-9 for acts in scale_acts):
            hmap_loc = hmap_loc[1:]
            scale_acts = [acts[1:] for acts in scale_acts]

        n_valid = min(int(hmap_loc.shape[0]), *(int(acts.shape[0]) for acts in scale_acts))
        if n_valid <= 1:
            return None

        cache = {
            "source": "disk",
            "n_valid": n_valid,
            "hmap_loc": hmap_loc[:n_valid].copy(),
            "scale_acts": [acts[:n_valid].clone() for acts in scale_acts],
            "log": (
                (
                    "dense_hmaps=disk(lightweight)"
                    if getattr(self, "lightweight_hmaps", False)
                    else "dense_hmaps=disk"
                )
                + f"({selected_note},dir={selected_dir})"
            ),
        }
        self._saved_unified_dense_hmaps_cache = cache
        return cache

    def _get_unified_dense_hdn_source(self):
        """
        Return dense HD hmaps for rebuilding the transition->HD decoder.
        """
        exploit_style_mode = getattr(self, "robot_mode", None) in {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }
        frozen_cache = getattr(self, "_frozen_unified_dense_hdn_cache", None)
        if exploit_style_mode and isinstance(frozen_cache, dict):
            return frozen_cache

        hmap_ok = (
            hasattr(self, "hmap_hdn")
            and self.step_count > 0
            and not getattr(self, "lightweight_hmaps", False)
        )
        if hmap_ok:
            n_valid = min(int(self.step_count) + 1, int(self.hmap_hdn.shape[0]))
            if exploit_style_mode:
                cache = {
                    "source": "memory_frozen",
                    "step_count": int(self.step_count),
                    "n_valid": n_valid,
                    "hmap_hdn": self.hmap_hdn[:n_valid].detach().cpu().float(),
                    "log": f"dense_hdn=memory_frozen(step={int(self.step_count)})",
                }
                self._frozen_unified_dense_hdn_cache = cache
                return cache
            cache = getattr(self, "_live_unified_dense_hdn_cache", None)
            if (
                isinstance(cache, dict)
                and cache.get("n_valid") == n_valid
                and cache.get("step_count") == int(self.step_count)
            ):
                return cache
            cache = {
                "source": "memory",
                "step_count": int(self.step_count),
                "n_valid": n_valid,
                "hmap_hdn": self.hmap_hdn[:n_valid].detach().cpu().float(),
                "log": "dense_hdn=memory",
            }
            self._live_unified_dense_hdn_cache = cache
            return cache

        cache = getattr(self, "_saved_unified_dense_hdn_cache", None)
        if cache is not None:
            return cache

        hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
        if not os.path.exists(hdn_path):
            return None

        try:
            with open(hdn_path, "rb") as f:
                hmap_hdn = pickle.load(f)
            if isinstance(hmap_hdn, torch.Tensor):
                hdn_t = hmap_hdn.detach().cpu().float()
            else:
                hdn_t = torch.as_tensor(np.array(hmap_hdn), dtype=torch.float32)
        except Exception:
            return None

        if hdn_t.ndim != 2 or hdn_t.shape[0] <= 1:
            return None

        if float(hdn_t[0].abs().sum().item()) <= 1e-9:
            hdn_t = hdn_t[1:]

        cache = {
            "source": "disk",
            "n_valid": int(hdn_t.shape[0]),
            "hmap_hdn": hdn_t,
            "log": (
                "dense_hdn=disk(lightweight)"
                if getattr(self, "lightweight_hmaps", False)
                else "dense_hdn=disk"
            ),
        }
        self._saved_unified_dense_hdn_cache = cache
        return cache

    def _derive_dense_hdn_from_locations(self, hmap_source):
        """
        Legacy fallback: derive HD populations from successive trajectory displacements.

        This is only used when saved HD hmaps are unavailable or degenerate. The
        runtime exploit readout remains coordinate-free; this just repairs older
        artifacts that never recorded HD activity properly.
        """
        if hmap_source is None:
            return None
        coords = np.array(hmap_source["hmap_loc"][: int(hmap_source["n_valid"]), :2], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[0] <= 1:
            return None

        hdn = torch.zeros((coords.shape[0], self.n_hd), dtype=torch.float32)
        angles = torch.linspace(
            0.0,
            2.0 * np.pi,
            steps=self.n_hd + 1,
            dtype=torch.float32,
        )[:-1]
        kernels = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        prev_vec = None
        for step in range(1, coords.shape[0]):
            delta = coords[step] - coords[step - 1]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                vec = torch.tensor(delta / norm, dtype=torch.float32)
                prev_vec = vec
            elif prev_vec is not None:
                vec = prev_vec
            else:
                continue
            hdn[step] = torch.matmul(kernels, vec)

        return {
            "source": "disk_loc_bootstrap",
            "n_valid": int(hdn.shape[0]),
            "hmap_hdn": hdn,
            "log": "dense_hdn=loc_bootstrap",
        }

    def _get_unified_pc_spatial_support(self, target_device):
        """
        Estimate unified place-field centers from dense hmaps.

        Returns:
            centers: (num_pc_total, 2) tensor of [x, z] centers of mass
            visited: (num_pc_total,) bool tensor marking cells with enough support
            log: debug string
        """
        device_key = str(target_device)
        source = self._get_unified_dense_hmap_source()
        if source is None:
            return None, None, "spatial=no_dense_hmaps"

        n_valid = int(source["n_valid"])
        cache = getattr(self, "_unified_pc_spatial_cache", None)
        if (
            cache is not None
            and cache.get("n_valid") == n_valid
            and cache.get("source") == source.get("source")
        ):
            device_cache = cache.setdefault("device_cache", {})
            cached_tensors = device_cache.get(device_key)
            if cached_tensors is None:
                cached_tensors = (
                    cache["centers"].to(target_device),
                    cache["visited"].to(target_device),
                )
                device_cache[device_key] = cached_tensors
            return cached_tensors[0], cached_tensors[1], cache["log"]

        num_pc_total = int(self.unified_pcn.num_pc_total)
        centers = torch.zeros((num_pc_total, 2), dtype=torch.float32, device=target_device)
        visited = torch.zeros(num_pc_total, dtype=torch.bool, device=target_device)

        hmap_loc = source["hmap_loc"]
        scale_acts = source["scale_acts"]
        hmap_x = torch.from_numpy(hmap_loc[:n_valid, 0]).float().to(target_device)
        hmap_z = torch.from_numpy(hmap_loc[:n_valid, 1]).float().to(target_device)

        for s_idx in range(len(self.scales)):
            if s_idx >= len(scale_acts):
                continue
            s_start = int(self.unified_pcn.scale_boundaries[s_idx])
            s_end = int(self.unified_pcn.scale_boundaries[s_idx + 1])
            acts = scale_acts[s_idx][:n_valid].float().to(target_device)
            acts_sum = acts.sum(dim=0)
            valid = acts_sum > 0.05
            if not valid.any():
                continue
            denom = acts_sum.clamp(min=1e-12)
            centers[s_start:s_end, 0] = (acts.T @ hmap_x) / denom
            centers[s_start:s_end, 1] = (acts.T @ hmap_z) / denom
            visited[s_start:s_end] = valid

        active_count = int(visited.sum().item())
        log = f"spatial=centers(active={active_count},source={source.get('source','?')})"
        self._unified_pc_spatial_cache = {
            "n_valid": n_valid,
            "source": source.get("source"),
            "centers": centers.detach().cpu(),
            "visited": visited.detach().cpu(),
            "device_cache": {
                device_key: (
                    centers.detach(),
                    visited.detach(),
                )
            },
            "log": log,
        }
        return centers, visited, log

    def _get_unified_spatial_visibility_mask(self, target_device):
        """
        Build a generic place-cell visibility mask from world obstacles.

        Any edge whose straight line between place-field centers crosses an
        obstacle rectangle is disallowed. This is environment-agnostic as long
        as the world has obstacle definitions in `webots_worlds.py`.
        """
        centers, visited, centers_log = self._get_unified_pc_spatial_support(target_device)
        if centers is None or visited is None:
            return None, None, centers_log

        source = self._get_unified_dense_hmap_source()
        n_valid = int(source["n_valid"]) if source is not None else min(int(self.step_count) + 1, self.hmap_loc.shape[0])
        obstacle_block = bool(getattr(self, "goal_map_spatial_obstacle_block", True))
        obstacle_margin = float(max(0.0, getattr(self, "goal_map_obstacle_margin", 0.05)))
        max_pair_distance = getattr(self, "unified_recurrent_max_pair_distance", 3.0)
        if max_pair_distance is not None:
            max_pair_distance = float(max_pair_distance)
        cache = getattr(self, "_unified_visibility_mask_cache", None)
        if (
            cache is not None
            and cache.get("n_valid") == n_valid
            and bool(cache.get("obstacle_block", False)) == obstacle_block
            and abs(float(cache.get("obstacle_margin", -1.0)) - obstacle_margin) <= 1e-9
            and float(cache.get("max_pair_distance", -999.0)) == float(max_pair_distance if max_pair_distance is not None else -1.0)
        ):
            return (
                cache["mask"].to(target_device),
                cache["visited"].to(target_device),
                cache["log"],
            )

        num_pc_total = int(self.unified_pcn.num_pc_total)
        mask = torch.eye(num_pc_total, dtype=torch.float32, device=target_device)
        active_idx = torch.nonzero(visited, as_tuple=False).squeeze(1)
        if active_idx.numel() <= 1:
            return mask, visited, "visibility=insufficient_active_cells"

        mask[active_idx[:, None], active_idx[None, :]] = 1.0
        distance_blocked_pairs = 0
        active_centers = centers[active_idx]
        if max_pair_distance is not None and max_pair_distance > 0.0:
            pair_distances = torch.cdist(active_centers, active_centers)
            far_mask = pair_distances > max_pair_distance
            far_mask.fill_diagonal_(False)
            if torch.any(far_mask):
                mask_active = mask[active_idx[:, None], active_idx[None, :]]
                mask_active[far_mask] = 0.0
                mask[active_idx[:, None], active_idx[None, :]] = mask_active
                distance_blocked_pairs = int(torch.triu(far_mask, diagonal=1).sum().item())

        obstacles = self._get_goal_map_world_obstacles() if obstacle_block else []
        blocked_pairs = 0
        if obstacles:
            active_centers = active_centers.detach().cpu().numpy()
            mask_active = mask[active_idx[:, None], active_idx[None, :]].detach().cpu().numpy()
            active_count = int(active_idx.numel())
            for row_idx in range(active_count):
                p0 = active_centers[row_idx]
                for col_idx in range(row_idx + 1, active_count):
                    p1 = active_centers[col_idx]
                    if self._candidate_edge_crosses_obstacle(
                        p0,
                        p1,
                        obstacles=obstacles,
                        margin=obstacle_margin,
                    ):
                        mask_active[row_idx, col_idx] = 0.0
                        mask_active[col_idx, row_idx] = 0.0
                        blocked_pairs += 1
            mask_active = torch.from_numpy(mask_active).to(target_device, dtype=torch.float32)
            mask[active_idx[:, None], active_idx[None, :]] = mask_active

        log = (
            f"visibility=geom(active={int(active_idx.numel())},"
            f"blocked_pairs={blocked_pairs},distance_blocked={distance_blocked_pairs},"
            f"wall_block={int(bool(obstacles))},max_pair_distance={max_pair_distance})"
        )
        self._unified_visibility_mask_cache = {
            "n_valid": n_valid,
            "obstacle_block": obstacle_block,
            "obstacle_margin": obstacle_margin,
            "max_pair_distance": float(max_pair_distance if max_pair_distance is not None else -1.0),
            "mask": mask.detach().cpu(),
            "visited": visited.detach().cpu(),
            "log": log,
        }
        return mask, visited, f"{centers_log},{log}"

    def _get_unified_exploit_room_reward_mask(self, device):
        """
        Build an exploit-time per-PC mask using replay-derived room masks.

        For each checkpoint the agent has crossed, suppress that checkpoint's
        upstream room cells (the cells that received reward during that
        checkpoint's replay).  Room masks are derived from the replay process
        itself, not from geometry.

        Falls back to a simple checkpoint-proximity suppression if no
        replay-derived masks are available.
        """
        masks = getattr(self, "_replay_derived_room_masks", None)
        if not masks:
            return None, "roommask=no_replay_data"

        if not self.detected_doorways:
            return None, "roommask=no_doorways"

        curr_x, curr_z = self._current_planar_xy()
        goal_name = self._get_active_goal_name_for_exploit()
        goal_loc = self._get_goal_location_by_name(goal_name) if goal_name else None
        if goal_loc is None:
            return None, "roommask=no_goal_loc"
        goal_x, goal_z = float(goal_loc[0]), float(goal_loc[1])

        num_pc = 0
        for v in masks.values():
            num_pc = max(num_pc, int(v.numel()))
            break
        if num_pc == 0:
            return None, "roommask=empty_masks"

        result_mask = torch.ones(num_pc, dtype=torch.float32, device=device)
        masked_logs: List[str] = []

        for ck_idx, (cx, cz) in enumerate(self.detected_doorways):
            source_key = f"cp{ck_idx}"
            room_cells = masks.get(source_key)
            if room_cells is None:
                continue

            # Determine if agent has crossed this checkpoint toward goal
            support_region, _ = self._infer_checkpoint_support_region(
                float(cx), float(cz)
            )
            frame = self._support_region_frame(support_region)
            if frame is None:
                continue

            frame_cx, frame_cz = frame["center"]
            nx, nz = frame["normal"]
            doorway_keep = max(
                0.15,
                float(frame.get("half_width", 0.0))
                + float(getattr(self, "goal_map_checkpoint_directional_band", 0.12)),
            )

            # Direction toward goal
            goal_signed = (
                (goal_x - frame_cx) * nx + (goal_z - frame_cz) * nz
            )
            if abs(goal_signed) <= 0.05:
                continue
            downstream_side = 1.0 if goal_signed > 0.0 else -1.0

            # Agent position relative to checkpoint
            agent_signed = (
                (curr_x - frame_cx) * nx + (curr_z - frame_cz) * nz
            )
            # Agent has crossed if it's on the downstream (goal) side
            if float(agent_signed * downstream_side) < -float(doorway_keep):
                continue

            # Suppress this checkpoint's upstream room cells
            room_cells_t = room_cells.to(device=device, dtype=torch.bool)
            suppress_count = int(torch.count_nonzero(room_cells_t).item())
            if suppress_count > 0:
                result_mask[room_cells_t] = 0.0
                masked_logs.append(
                    f"cp{ck_idx}:replay(n={suppress_count},"
                    f"agent_signed={agent_signed:.2f})"
                )

        if not masked_logs:
            return None, "roommask=all_clear"

        return result_mask, f"roommask=[{','.join(masked_logs)}]"

    def _sync_unified_recurrent_visibility_mask(self, prune_weights: bool = True):
        """
        Push the generic geometry-based visibility mask into the unified PCN.

        This blocks across-wall recurrent/preplay edges for any world whose
        obstacles are defined in `webots_worlds.py`.
        """
        if not getattr(self, "use_unified_multiscale", False):
            return "visibility=no_unified_mode"
        if not hasattr(self, "unified_pcn") or self.unified_pcn is None:
            return "visibility=no_unified_pcn"

        mask, _, log = self._get_unified_spatial_visibility_mask(self.unified_pcn.device)
        if mask is None:
            self.unified_pcn.recurrent_visibility_mask = None
            return log

        mask = mask.to(device=self.unified_pcn.device, dtype=self.unified_pcn.dtype)
        self.unified_pcn.recurrent_visibility_mask = mask
        if prune_weights and hasattr(self.unified_pcn, "w_rec_unified"):
            self.unified_pcn.w_rec_unified.data *= mask.unsqueeze(0)
        return log

    def _get_unified_reward_spatial_kernel(
        self,
        target_device,
        sigma: Optional[float] = None,
        topk: Optional[int] = None,
    ):
        """
        Build one cross-scale spatial smoothing kernel over all visited place cells.

        This treats scale identity as irrelevant: neighboring fields in physical
        space smooth one another even when they belong to different scales.
        """
        centers, visited, centers_log = self._get_unified_pc_spatial_support(target_device)
        if centers is None or visited is None:
            return None, None, centers_log.replace("spatial=", "smooth=")

        n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
        num_pc_total = int(self.unified_pcn.num_pc_total)
        sigma = float(
            max(
                0.1,
                getattr(self, "goal_map_smoothing_sigma", 1.0)
                if sigma is None
                else sigma,
            )
        )
        topk = int(
            max(
                1,
                getattr(self, "goal_map_smoothing_topk", 24)
                if topk is None
                else topk,
            )
        )
        obstacle_block = bool(getattr(self, "goal_map_spatial_obstacle_block", True))
        obstacle_margin = float(max(0.0, getattr(self, "goal_map_obstacle_margin", 0.05)))
        cache = getattr(self, "_unified_reward_spatial_cache", None)

        if (
            cache is not None
            and cache.get("n_valid") == n_valid
            and abs(float(cache.get("sigma", -1.0)) - sigma) <= 1e-9
            and int(cache.get("topk", -1)) == topk
            and bool(cache.get("obstacle_block", False)) == obstacle_block
            and abs(float(cache.get("obstacle_margin", -1.0)) - obstacle_margin) <= 1e-9
        ):
            return (
                cache["kernel"].to(target_device),
                cache["visited"].to(target_device),
                cache["log"],
            )

        active_idx = torch.nonzero(visited, as_tuple=False).squeeze(1)
        if active_idx.numel() <= 1:
            return None, visited, "smooth=insufficient_active_cells"

        active_centers = centers[active_idx]
        dx = active_centers[:, 0].unsqueeze(1) - active_centers[:, 0].unsqueeze(0)
        dz = active_centers[:, 1].unsqueeze(1) - active_centers[:, 1].unsqueeze(0)
        dist_sq = (dx * dx) + (dz * dz)
        kernel_active = torch.exp(-dist_sq / (2.0 * sigma * sigma))

        active_topk = min(topk, int(active_idx.numel()))
        vals, idx = torch.topk(kernel_active, k=active_topk, dim=1)

        visibility_mask, _, visibility_log = self._get_unified_spatial_visibility_mask(
            target_device
        )
        obstacles = self._get_goal_map_world_obstacles() if obstacle_block else []
        if visibility_mask is not None and obstacles:
            active_visibility = visibility_mask[active_idx[:, None], active_idx[None, :]]
            vals = vals * torch.gather(active_visibility, 1, idx)

        sparse_active = torch.zeros_like(kernel_active)
        sparse_active.scatter_(1, idx, vals)
        kernel_active = sparse_active

        row_sum = kernel_active.sum(dim=1, keepdim=True).clamp(min=1e-12)
        kernel_active = kernel_active / row_sum

        kernel = torch.zeros(
            (num_pc_total, num_pc_total), dtype=torch.float32, device=target_device
        )
        kernel[active_idx[:, None], active_idx[None, :]] = kernel_active
        kernel_log = (
            f"smooth=unified_gauss(sigma={sigma:.2f}m,topk={active_topk},"
            f"active={int(active_idx.numel())},wall_block={int(bool(obstacles))})"
        )
        self._unified_reward_spatial_cache = {
            "n_valid": n_valid,
            "sigma": sigma,
            "topk": topk,
            "obstacle_block": obstacle_block,
            "obstacle_margin": obstacle_margin,
            "kernel": kernel.detach().cpu(),
            "visited": visited.detach().cpu(),
            "log": f"{kernel_log},{visibility_log}",
        }
        return kernel, visited, f"{kernel_log},{visibility_log}"

    def _apply_unified_reward_smoothing(self, goal_rcn):
        """
        Fill spatial dead bands after replay using one shared cross-scale kernel.
        """
        raw_weights = torch.clamp(goal_rcn.w_in.clone(), min=0.0)
        goal_rcn.w_in = raw_weights
        goal_rcn.w_in_effective = raw_weights.clone()

        passes = int(max(0, getattr(self, "goal_map_smoothing_passes", 0)))
        mix = float(min(1.0, max(0.0, getattr(self, "goal_map_smoothing_mix", 0.0))))
        if passes <= 0 or mix <= 0.0:
            return "smooth=disabled"

        kernel, visited, kernel_log = self._get_unified_reward_spatial_kernel(goal_rcn.w_in.device)
        if kernel is None or visited is None:
            return kernel_log

        raw_1d = raw_weights[0]
        effective = raw_1d.clone()
        for _ in range(passes):
            neighbor_fill = torch.mv(kernel, effective)
            effective = torch.maximum(
                raw_1d,
                ((1.0 - mix) * effective) + (mix * neighbor_fill),
            )

        effective = torch.where(visited, effective, torch.zeros_like(effective))
        goal_rcn.w_in_effective = effective.unsqueeze(0)
        return f"{kernel_log},passes={passes},mix={mix:.2f}"

    def _build_paper_room_local_goal_map(
        self,
        goal_rcn,
        goal_name: Optional[str],
        goal_x: float,
        goal_z: float,
        seed_activations: torch.Tensor,
        checkpoint_seed_bank: List[Tuple[int, torch.Tensor]],
        checkpoint_seed_support_id_bank: Dict[int, List[int]],
        checkpoint_support_region_bank: Dict[int, Optional[Dict[str, Any]]],
        history_start: int,
        history_log: str,
        visibility_log: str,
        path_constraint: Optional[torch.Tensor],
        local_kernel: Optional[torch.Tensor],
        path_kernel_log: str,
        local_kernel_log: str,
    ) -> Tuple[Any, str]:
        """
        Build one piecewise room-local reward map using simple paper-style replay.

        Route selection still chooses the checkpoint chain. Each room then gets
        exactly one sink:
        - goal room -> goal
        - upstream room -> checkpoint that exits that room toward the goal

        Reward within each room is shaped only by additive backward replay on a
        cleaned room-restricted recurrent substrate.
        """
        device = goal_rcn.w_in.device
        eps = 1e-8
        num_pc_total = int(getattr(self.unified_pcn, "num_pc_total", 0))
        goal_name = str(goal_name or "")
        goal_radius = 0.8
        for goal in getattr(self, "goals", []):
            goal_loc = goal.get("location", [])
            if goal_name and str(goal.get("name", "")) == goal_name:
                goal_radius = float(goal.get("radius", goal_radius))
                break
            if (
                len(goal_loc) >= 2
                and abs(float(goal_loc[0]) - float(goal_x)) <= 1e-6
                and abs(float(goal_loc[1]) - float(goal_z)) <= 1e-6
            ):
                goal_radius = float(goal.get("radius", goal_radius))
                break

        goal_rcn.goal_map_local_anchor_weights = None
        goal_rcn.goal_map_local_anchor_gate_d2 = None
        goal_rcn.goal_map_local_anchor_query_sigma = None
        goal_rcn.goal_map_local_anchor_names = None
        goal_rcn.reward_denominator_override = None

        centers, visited, centers_log = self._get_unified_pc_spatial_support(device)
        if centers is None or visited is None:
            centers = torch.zeros((num_pc_total, 2), dtype=torch.float32, device=device)
            visited = torch.ones(num_pc_total, dtype=torch.bool, device=device)
            centers_log = "spatial=fallback_all_cells"

        recurrent_weights_max = torch.clamp(
            torch.max(self.unified_pcn.w_rec_unified.to(device), dim=0)[0],
            min=0.0,
        )
        visibility_mask, _, _ = self._get_unified_spatial_visibility_mask(device)
        if (
            visibility_mask is not None
            and visibility_mask.shape == recurrent_weights_max.shape
        ):
            recurrent_weights_max = recurrent_weights_max * visibility_mask.to(
                device=device, dtype=recurrent_weights_max.dtype
            )

        replay_transition = recurrent_weights_max.clone()
        substrate_log = "substrate=recurrent_max"
        counts = getattr(goal_rcn, "experience_transition_counts", None)
        if isinstance(counts, torch.Tensor):
            counts_t = torch.as_tensor(
                counts, dtype=torch.float32, device=device
            )
            if counts_t.shape == replay_transition.shape and bool(
                torch.any(counts_t > 0.0).item()
            ):
                counts_sym = torch.maximum(counts_t, counts_t.T)
                replay_transition = replay_transition * (counts_sym > 0.0).to(
                    replay_transition.dtype
                )
                substrate_log = (
                    "substrate=cleaned_recurrent("
                    f"edges={int(torch.count_nonzero(counts_sym > 0.0).item())})"
                )

        exp_topk = getattr(self, "goal_map_experience_transition_topk", None)
        if exp_topk is not None and 0 < int(exp_topk) < int(replay_transition.shape[1]):
            vals, idx = torch.topk(replay_transition, k=int(exp_topk), dim=1)
            sparse = torch.zeros_like(replay_transition)
            sparse.scatter_(1, idx, vals)
            replay_transition = sparse
            substrate_log += f",topk={int(exp_topk)}"

        route_selected_data = None
        route_log = "route_select=disabled"
        ordered_checkpoint_ids = [int(checkpoint_idx) for checkpoint_idx, _ in checkpoint_seed_bank]
        if checkpoint_seed_bank:
            arrival_buffer_steps = int(
                max(
                    0,
                    getattr(
                        self,
                        "goal_map_checkpoint_arrival_buffer_steps",
                        1,
                    ),
                )
            )
            try:
                route_selected_data, route_log = self._compute_route_selected_checkpoint_tree(
                    reward_rcn=goal_rcn,
                    goal_seed=seed_activations,
                    checkpoint_seed_bank=checkpoint_seed_bank,
                    checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                    candidate_checkpoint_ids=ordered_checkpoint_ids,
                    path_constraint=path_constraint,
                    local_kernel=local_kernel,
                    recurrent_weights_max=recurrent_weights_max,
                    arrival_buffer_steps=arrival_buffer_steps,
                )
            except Exception as exc:
                route_selected_data = None
                route_log = f"route_select=error({exc})"

        # --- Build checkpoint boundary masks for replay-driven room discovery ---
        # Default percentile=0 includes ALL cells that fired near the
        # checkpoint.  The flood fill handles actual room separation;
        # the barrier just needs enough cells to disconnect the graph.
        boundary_masks = self._build_checkpoint_boundary_masks(
            percentile=float(getattr(self, "checkpoint_boundary_percentile", 0.0)),
            device=device,
        )
        room_partition_log = f"roomown=replay_boundaries(n_cp={len(boundary_masks)})"

        selected_order = (
            list(route_selected_data.get("selected_order", []))
            if isinstance(route_selected_data, dict)
            else []
        )
        if not selected_order and checkpoint_seed_bank:
            fallback = []
            for checkpoint_idx, _seed in checkpoint_seed_bank:
                if 0 <= int(checkpoint_idx) < len(self.detected_doorways):
                    cx, cy = self.detected_doorways[int(checkpoint_idx)]
                    dist = math.hypot(float(cx) - float(goal_x), float(cy) - float(goal_z))
                    fallback.append((dist, int(checkpoint_idx)))
            fallback.sort(key=lambda item: (item[0], item[1]))
            selected_order = [checkpoint_idx for _, checkpoint_idx in fallback]

        checkpoint_positions = {
            int(idx): (float(cx), float(cy))
            for idx, (cx, cy) in enumerate(getattr(self, "detected_doorways", []))
        }

        # --- Build checkpoint geometry for seed direction selection ---
        checkpoint_geometry: Dict[int, Dict[str, Any]] = {}
        for order_idx, checkpoint_idx in enumerate(selected_order):
            checkpoint_idx = int(checkpoint_idx)
            if checkpoint_idx not in checkpoint_positions:
                continue
            support_region = checkpoint_support_region_bank.get(checkpoint_idx)
            if support_region is None:
                support_region, _ = self._infer_checkpoint_support_region(
                    *checkpoint_positions[checkpoint_idx]
                )
            frame = self._support_region_frame(support_region)
            if frame is None:
                continue
            frame_cx, frame_cy = frame["center"]
            nx, ny = frame["normal"]
            checkpoint_geometry[checkpoint_idx] = {
                "support_region": support_region,
                "frame_center": (frame_cx, frame_cy),
                "frame_normal": (nx, ny),
            }

        # --- Replay-driven room discovery ---
        # Instead of precomputing rooms, we build a single "goal" assignment
        # with ALL visited cells as its initial mask (bounded by checkpoint
        # barriers).  Then for each checkpoint in route order, we build an
        # assignment whose mask = ~(other boundaries) & ~(already_rewarded).
        # The replay loop fills each room to completion; the already_rewarded
        # mask tracks which cells have been claimed.
        #
        # Room masks are derived AFTER replay, from which cells received
        # reward in each iteration.

        room_assignments: List[Dict[str, Any]] = []

        # Union of ALL checkpoint boundaries — used to block goal replay
        all_checkpoint_barrier = torch.zeros(
            num_pc_total, dtype=torch.bool, device=device
        )
        for cp_idx, bmask in boundary_masks.items():
            all_checkpoint_barrier |= bmask.to(device)

        def _flood_reachable_from_seed(
            transition: torch.Tensor,
            seed_mask: torch.Tensor,
            barrier_mask: torch.Tensor,
            max_steps: int = 50,
            threshold: float = 1e-8,
        ) -> torch.Tensor:
            """Flood fill through transition, blocked by barrier cells.

            Zeroes out all connections to/from barrier cells in the
            transition matrix, then iteratively walks from seed_mask
            to find every reachable cell.  Returns a bool mask.
            """
            pass_float = (1.0 - barrier_mask.float())
            cut_transition = transition * (
                pass_float.unsqueeze(0) * pass_float.unsqueeze(1)
            )
            reachable = seed_mask.clone()
            frontier = seed_mask.float()
            for _ in range(max_steps):
                spread = torch.mv(cut_transition, frontier)
                new_cells = (spread > threshold) & (~reachable) & (~barrier_mask)
                if not bool(torch.any(new_cells).item()):
                    break
                reachable |= new_cells
                frontier = new_cells.float()
            return reachable

        # Goal room: flood fill from goal seed through transition,
        # with checkpoint boundaries acting as barriers in the graph.
        goal_seed_mask = (seed_activations.to(device) > eps).view(-1)
        if not bool(torch.any(goal_seed_mask).item()):
            # Fallback: use a sparse seed near the goal location
            _goal_dists = (
                (centers[:, 0] - float(goal_x)) ** 2
                + (centers[:, 1] - float(goal_z)) ** 2
            )
            _nearest = torch.argmin(_goal_dists)
            goal_seed_mask = torch.zeros(
                num_pc_total, dtype=torch.bool, device=device
            )
            goal_seed_mask[_nearest] = True
        goal_reachable = _flood_reachable_from_seed(
            replay_transition, goal_seed_mask, all_checkpoint_barrier
        )
        goal_spread_mask = visited.clone().to(device) & goal_reachable
        _goal_reachable_count = int(torch.count_nonzero(goal_reachable).item())
        _goal_spread_count = int(torch.count_nonzero(goal_spread_mask).item())
        print(
            f"[ROOM-FLOOD] goal: reachable={_goal_reachable_count}, "
            f"spread_mask={_goal_spread_count} "
            f"(barrier={int(torch.count_nonzero(all_checkpoint_barrier).item())} cells)"
        )
        if bool(torch.any(goal_spread_mask).item()):
            room_assignments.append({
                "source_name": "goal",
                "sink_type": "goal",
                "sink_idx": None,
                "route_pos": -1,
                "direction_key": None,
                "room_mask": goal_spread_mask,
                "support_region": None,
                "fallback_seed": torch.where(
                    goal_spread_mask,
                    seed_activations.to(device=device, dtype=torch.float32),
                    torch.zeros_like(seed_activations, device=device, dtype=torch.float32),
                ),
            })

        # Checkpoint rooms: mask = ~(other boundaries) & ~(already_rewarded)
        # already_rewarded is updated after each replay in the main loop.
        # We defer the actual mask computation to replay time (see below).
        # For now, store a placeholder that the replay loop will replace.
        for order_idx, checkpoint_idx in enumerate(selected_order):
            checkpoint_idx = int(checkpoint_idx)
            if checkpoint_idx not in checkpoint_positions:
                continue
            # Build barrier: union of all OTHER checkpoint boundaries
            other_boundaries = torch.zeros(
                num_pc_total, dtype=torch.bool, device=device
            )
            for cp_j, bmask in boundary_masks.items():
                if int(cp_j) != checkpoint_idx:
                    other_boundaries |= bmask.to(device)

            # Flood fill from this checkpoint's boundary cells through
            # the transition, blocked by other checkpoints' boundaries.
            cp_seed_mask = boundary_masks.get(checkpoint_idx)
            if cp_seed_mask is not None:
                cp_seed_mask = cp_seed_mask.to(device)
            else:
                # Fallback: seed from nearest cell to checkpoint position
                cp_pos = checkpoint_positions[checkpoint_idx]
                _cp_dists = (
                    (centers[:, 0] - float(cp_pos[0])) ** 2
                    + (centers[:, 1] - float(cp_pos[1])) ** 2
                )
                cp_seed_mask = torch.zeros(
                    num_pc_total, dtype=torch.bool, device=device
                )
                cp_seed_mask[torch.argmin(_cp_dists)] = True
            cp_reachable = _flood_reachable_from_seed(
                replay_transition, cp_seed_mask, other_boundaries
            )
            initial_spread_mask = visited.clone().to(device) & cp_reachable
            _cp_reachable_count = int(
                torch.count_nonzero(cp_reachable).item()
            )
            _cp_spread_count = int(
                torch.count_nonzero(initial_spread_mask).item()
            )
            print(
                f"[ROOM-FLOOD] cp{checkpoint_idx}: "
                f"reachable={_cp_reachable_count}, "
                f"spread_mask={_cp_spread_count} "
                f"(barrier={int(torch.count_nonzero(other_boundaries).item())} cells)"
            )

            checkpoint_seed = next(
                (
                    seed.to(device=device, dtype=torch.float32)
                    for idx, seed in checkpoint_seed_bank
                    if int(idx) == checkpoint_idx
                ),
                None,
            )
            if checkpoint_seed is None:
                checkpoint_seed = torch.zeros(
                    num_pc_total, dtype=torch.float32, device=device
                )

            # Direction key: determine which crossing direction leads upstream
            # (into unrewarded territory). This will be refined at replay time
            # using already_rewarded, but we need an initial guess for seed
            # selection. Use "both" to indicate we'll pick at replay time.
            geom = checkpoint_geometry.get(checkpoint_idx)
            support_region = geom["support_region"] if geom is not None else None

            fallback_seed = torch.where(
                initial_spread_mask, checkpoint_seed,
                torch.zeros_like(checkpoint_seed),
            )
            room_assignments.append({
                "source_name": f"cp{checkpoint_idx}",
                "sink_type": "checkpoint",
                "sink_idx": checkpoint_idx,
                "route_pos": int(order_idx),
                "direction_key": "deferred",
                "room_mask": initial_spread_mask,
                "other_boundaries": other_boundaries,
                "support_region": support_region,
                "fallback_seed": fallback_seed,
            })

        # --- Diagnostic: summarize room assignments ---
        print(
            f"[ROOM-ASSIGN] {len(room_assignments)} room assignments built "
            f"(selected_order={[int(i) for i in selected_order]}, "
            f"mode=replay_boundaries)"
        )
        for _ra in room_assignments:
            _ra_count = int(torch.count_nonzero(
                torch.as_tensor(_ra["room_mask"], dtype=torch.bool)
            ).item())
            print(
                f"  {_ra['source_name']}: sink={_ra['sink_type']}, "
                f"dir={_ra.get('direction_key','n/a')}, "
                f"initial_mask_size={_ra_count}"
            )

        history, compact_log = self._extract_compact_unified_history(
            start_step=history_start,
            end_step=int(self.step_count),
        )
        compact_history_t = None
        compact_time_indices: List[int] = []
        compact_time_to_local: Dict[int, int] = {}
        compact_time_indices_np = np.empty(0, dtype=np.int64)
        if history is not None:
            compact_time_indices = [
                int(t)
                for t in history.get("time_indices", [])
                if 0 <= int(t) < int(self.hmap_loc.shape[0])
            ]
            if compact_time_indices:
                compact_blocks = [
                    arr.index_select(
                        0,
                        torch.as_tensor(
                            compact_time_indices,
                            dtype=torch.long,
                            device=arr.device,
                        ),
                    ).to(device=device, dtype=torch.float32)
                    for arr in history.get("scale_arrays", [])
                ]
                if compact_blocks:
                    compact_history_t = torch.clamp(
                        torch.cat(compact_blocks, dim=1),
                        min=0.0,
                    )
                    compact_time_to_local = {
                        int(t): idx for idx, t in enumerate(compact_time_indices)
                    }
                    compact_time_indices_np = np.asarray(compact_time_indices, dtype=np.int64)
        else:
            compact_log = "history=no_compact_history"

        replay_timesteps = int(max(1, getattr(self, "goal_map_replay_timesteps", 12)))
        replay_tau = float(max(1e-6, getattr(self, "goal_map_paper_replay_tau", 8.0)))
        goal_replay_timesteps = getattr(self, "goal_map_goal_replay_timesteps", None)
        if goal_replay_timesteps is None:
            goal_replay_timesteps = replay_timesteps
        else:
            goal_replay_timesteps = int(max(1, goal_replay_timesteps))
        goal_replay_tau = getattr(self, "goal_map_goal_replay_tau", None)
        if goal_replay_tau is None:
            goal_replay_tau = replay_tau
        else:
            goal_replay_tau = float(max(1e-6, goal_replay_tau))
        room_norm_mode = str(
            getattr(self, "goal_map_room_normalization_mode", "per_room_peak")
        ).strip().lower()
        checkpoint_support_min_fraction = float(
            min(
                1.0,
                max(
                    0.50,
                    getattr(
                        self,
                        "goal_map_checkpoint_room_support_min_fraction",
                        0.98,
                    ),
                ),
            )
        )

        dense_hmap_source = self._get_unified_dense_hmap_source()
        dense_hmap_loc = None
        dense_scale_acts: List[torch.Tensor] = []
        dense_unified_acts: Optional[torch.Tensor] = None
        dense_valid = 0
        if isinstance(dense_hmap_source, dict):
            dense_valid = int(dense_hmap_source.get("n_valid", 0))
            dense_hmap_loc = np.array(
                dense_hmap_source.get("hmap_loc", np.empty((0, 2), dtype=np.float32))
            )[:dense_valid]
            dense_scale_acts = [
                acts[:dense_valid].to(device=device, dtype=torch.float32)
                for acts in dense_hmap_source.get("scale_acts", [])
            ]
            if dense_scale_acts:
                try:
                    dense_unified_acts = torch.clamp(
                        torch.cat(dense_scale_acts, dim=1),
                        min=0.0,
                    )
                except Exception:
                    dense_unified_acts = None

        def _build_assignment_position_mask(assignment: Dict[str, Any]) -> Optional[torch.Tensor]:
            if dense_hmap_loc is None or dense_valid <= 0:
                return None

            # Use the PC room mask to determine which dense positions belong
            # to this room: a position is "inside" if the majority of its
            # activation mass falls on PCs within the room_mask.
            if dense_unified_acts is not None:
                room_mask_t = torch.as_tensor(
                    assignment["room_mask"], dtype=torch.bool, device=device,
                ).view(-1)
                # For each dense position, compute fraction of activation in-room
                total_mass = torch.sum(dense_unified_acts, dim=1)  # (dense_valid,)
                inside_mass = torch.sum(
                    dense_unified_acts[:, room_mask_t], dim=1,
                )  # (dense_valid,)
                inside_frac = inside_mass / torch.clamp(total_mass, min=1e-8)
                return inside_frac >= 0.5

            # Fallback: geometry-based position mask
            pos_x = torch.as_tensor(
                dense_hmap_loc[:dense_valid, 0],
                dtype=torch.float32,
                device=device,
            )
            pos_y = torch.as_tensor(
                dense_hmap_loc[:dense_valid, 1],
                dtype=torch.float32,
                device=device,
            )

            if str(assignment["sink_type"]) == "goal":
                pos_mask = torch.ones(dense_valid, dtype=torch.bool, device=device)
                if selected_order:
                    nearest_goal_checkpoint = checkpoint_geometry.get(int(selected_order[0]))
                    if nearest_goal_checkpoint is not None:
                        frame_cx, frame_cy = nearest_goal_checkpoint["frame_center"]
                        nx, ny = nearest_goal_checkpoint["frame_normal"]
                        signed_pos = ((pos_x - frame_cx) * nx) + ((pos_y - frame_cy) * ny)
                        pos_mask = pos_mask & (
                            (signed_pos * nearest_goal_checkpoint["downstream_side"])
                            > nearest_goal_checkpoint["doorway_keep"]
                        )
                return pos_mask

            checkpoint_idx = int(assignment["sink_idx"])
            geom = checkpoint_geometry.get(checkpoint_idx)
            if geom is None:
                return None

            frame_cx, frame_cy = geom["frame_center"]
            nx, ny = geom["frame_normal"]
            signed_pos = ((pos_x - frame_cx) * nx) + ((pos_y - frame_cy) * ny)
            pos_mask = (signed_pos * geom["downstream_side"]) <= geom["doorway_keep"]

            route_pos = int(assignment["route_pos"])
            if route_pos + 1 < len(selected_order):
                upstream_checkpoint_idx = int(selected_order[route_pos + 1])
                upstream_geom = checkpoint_geometry.get(upstream_checkpoint_idx)
                if upstream_geom is not None:
                    up_cx, up_cy = upstream_geom["frame_center"]
                    up_nx, up_ny = upstream_geom["frame_normal"]
                    upstream_signed_pos = ((pos_x - up_cx) * up_nx) + ((pos_y - up_cy) * up_ny)
                    pos_mask = pos_mask & (
                        (upstream_signed_pos * upstream_geom["downstream_side"])
                        > upstream_geom["doorway_keep"]
                    )
            return pos_mask

        def _refine_checkpoint_support_mask(
            assignment: Dict[str, Any],
            raw_room_mask_t: torch.Tensor,
        ) -> Tuple[torch.Tensor, str]:
            if str(assignment["sink_type"]) != "checkpoint":
                return raw_room_mask_t, "support_refine=goal_passthrough"
            if dense_hmap_loc is None or dense_valid <= 0 or not dense_scale_acts:
                return raw_room_mask_t, "support_refine=no_dense_hmaps"

            position_mask = _build_assignment_position_mask(assignment)
            if position_mask is None or not bool(torch.any(position_mask).item()):
                return raw_room_mask_t, "support_refine=no_position_room_mask"

            refined_mask = raw_room_mask_t.clone()
            original_count = int(torch.count_nonzero(raw_room_mask_t).item())
            kept_count = 0
            doorway_keep_count = 0
            doorway_keep_mask = None
            if centers is not None:
                support_region = assignment.get("support_region")
                fallback_xy = checkpoint_positions.get(int(assignment["sink_idx"]), (0.0, 0.0))
                doorway_band_radius = float(
                    max(0.10, 0.35 * float(max(0.15, self.checkpoint_visit_radius)))
                )
                doorway_d2 = self._support_region_distance_sq_torch(
                    centers[:, 0],
                    centers[:, 1],
                    support_region=support_region,
                    default_point=fallback_xy,
                )
                doorway_keep_mask = raw_room_mask_t & (
                    doorway_d2 <= (doorway_band_radius * doorway_band_radius)
                )
                doorway_keep_count = int(torch.count_nonzero(doorway_keep_mask).item())
            for scale_idx in range(len(self.scales)):
                if scale_idx >= len(dense_scale_acts):
                    continue
                s_start = int(self.unified_pcn.scale_boundaries[scale_idx])
                s_end = int(self.unified_pcn.scale_boundaries[scale_idx + 1])
                slice_mask = raw_room_mask_t[s_start:s_end]
                if not bool(torch.any(slice_mask).item()):
                    refined_mask[s_start:s_end] = False
                    continue

                acts = dense_scale_acts[scale_idx]
                total_mass = torch.sum(acts, dim=0)
                inside_mass = torch.sum(acts[position_mask], dim=0)
                inside_frac = inside_mass / torch.clamp(total_mass, min=eps)
                doorway_slice = (
                    doorway_keep_mask[s_start:s_end]
                    if doorway_keep_mask is not None
                    else torch.zeros_like(slice_mask)
                )
                keep_slice = (
                    slice_mask
                    & (
                        doorway_slice
                        | (
                            (total_mass > eps)
                            & (inside_frac >= checkpoint_support_min_fraction)
                        )
                    )
                )
                refined_mask[s_start:s_end] = keep_slice
                kept_count += int(torch.count_nonzero(keep_slice).item())

            min_keep = max(8, int(math.ceil(0.05 * max(1, original_count))))
            if kept_count < min_keep:
                return (
                    raw_room_mask_t,
                    "support_refine=fallback_raw("
                    f"kept={kept_count},min={min_keep},"
                    f"frac>={checkpoint_support_min_fraction:.2f})",
                )

            return (
                refined_mask,
                "support_refine=occupancy("
                f"kept={kept_count}/{original_count},"
                f"frac>={checkpoint_support_min_fraction:.2f},"
                f"doorway_keep={doorway_keep_count})",
            )

        def _compute_goal_validation_summary(
            assignment: Dict[str, Any],
            room_values: torch.Tensor,
        ) -> Dict[str, Any]:
            if str(assignment.get("sink_type")) != "goal":
                return {}
            if dense_hmap_loc is None or dense_valid <= 0 or dense_unified_acts is None:
                return {
                    "goal_validation_available": False,
                    "goal_validation_reason": "no_dense_hmaps",
                }

            position_mask = _build_assignment_position_mask(assignment)
            if position_mask is None or not bool(torch.any(position_mask).item()):
                return {
                    "goal_validation_available": False,
                    "goal_validation_reason": "no_goal_room_positions",
                }

            room_positions_t = torch.as_tensor(
                dense_hmap_loc[:dense_valid, :2],
                dtype=torch.float32,
                device=device,
            )
            room_goal_acts = dense_unified_acts[position_mask]
            if room_goal_acts.numel() <= 0:
                return {
                    "goal_validation_available": False,
                    "goal_validation_reason": "empty_goal_room_acts",
                }

            reward_numer = torch.mv(room_goal_acts, room_values)
            reward_denom = torch.clamp(
                torch.sum(torch.abs(room_goal_acts), dim=1),
                min=1e-4,
            )
            goal_room_reward = reward_numer / reward_denom
            reward_peak = float(torch.max(goal_room_reward).item())
            if reward_peak <= eps:
                return {
                    "goal_validation_available": False,
                    "goal_validation_reason": "zero_goal_room_reward",
                }

            room_positions = room_positions_t[position_mask]
            pos_x = room_positions[:, 0]
            pos_y = room_positions[:, 1]
            min_x = float(torch.min(pos_x).item())
            max_x = float(torch.max(pos_x).item())
            min_y = float(torch.min(pos_y).item())
            max_y = float(torch.max(pos_y).item())
            boundary_width = float(max(0.0, goal_radius))
            high_reward_mask = goal_room_reward >= (0.90 * reward_peak)
            goal_region_mask = (
                ((pos_x - float(goal_x)) ** 2) + ((pos_y - float(goal_z)) ** 2)
            ) <= float(goal_radius * goal_radius)
            boundary_masks = {
                "left": pos_x <= (min_x + boundary_width),
                "right": pos_x >= (max_x - boundary_width),
                "bottom": pos_y <= (min_y + boundary_width),
                "top": pos_y >= (max_y - boundary_width),
            }
            high_reward_count = int(torch.count_nonzero(high_reward_mask).item())
            goal_region_overlap_count = int(
                torch.count_nonzero(high_reward_mask & goal_region_mask).item()
            )
            boundary_strip_overlap_counts = {
                name: int(torch.count_nonzero(high_reward_mask & mask).item())
                for name, mask in boundary_masks.items()
            }
            max_boundary_overlap = max(boundary_strip_overlap_counts.values(), default=0)
            return {
                "goal_validation_available": True,
                "goal_room_bounds": {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                },
                "goal_boundary_strip_width": boundary_width,
                "high_reward_threshold_fraction": 0.90,
                "high_reward_count": high_reward_count,
                "goal_region_count": int(torch.count_nonzero(goal_region_mask).item()),
                "boundary_strip_counts": {
                    name: int(torch.count_nonzero(mask).item())
                    for name, mask in boundary_masks.items()
                },
                "high_reward_goal_region_overlap_count": goal_region_overlap_count,
                "high_reward_goal_region_overlap_fraction": (
                    float(goal_region_overlap_count) / float(max(1, high_reward_count))
                ),
                "high_reward_boundary_strip_overlap_counts": boundary_strip_overlap_counts,
                "high_reward_boundary_strip_overlap_fractions": {
                    name: float(count) / float(max(1, high_reward_count))
                    for name, count in boundary_strip_overlap_counts.items()
                },
                "goal_peak_true_region_gt_any_boundary_strip": bool(
                    goal_region_overlap_count > max_boundary_overlap
                ),
            }

        def _get_event_step_from_group(event_group: List[int]) -> int:
            valid_steps = [
                int(step)
                for step in (event_group or [])
                if 0 <= int(step) < int(self.step_count)
            ]
            return int(valid_steps[-1]) if valid_steps else -1

        def _get_goal_episode_representative_step(
            event_group: List[int],
        ) -> Tuple[int, float, str]:
            valid_steps = sorted(
                int(step)
                for step in (event_group or [])
                if 0 <= int(step) < int(self.step_count)
            )
            if not valid_steps:
                return -1, 0.0, "goal_rep=empty"
            if len(valid_steps) == 1:
                return int(valid_steps[0]), 0.0, "goal_rep=singleton"
            first_step = int(valid_steps[0])
            last_step = int(valid_steps[-1])
            mid_step = 0.5 * float(first_step + last_step)
            ranked_steps = sorted(
                valid_steps,
                key=lambda step: (
                    -float(min(step - first_step, last_step - step)),
                    abs(float(step) - mid_step),
                    int(step),
                ),
            )
            best_step = int(ranked_steps[0])
            best_depth = float(min(best_step - first_step, last_step - best_step))
            return (
                best_step,
                best_depth,
                "goal_rep=temporal_interior("
                f"n={len(valid_steps)},depth={best_depth:.1f})",
            )

        def _get_exact_unified_activation_for_step(
            event_step: int,
        ) -> Tuple[Optional[torch.Tensor], int, str]:
            event_step = int(event_step)
            if event_step < 0:
                return None, -1, "source=invalid_step"
            if event_step in compact_time_to_local and compact_history_t is not None:
                local_idx = int(compact_time_to_local[event_step])
                return (
                    compact_history_t[local_idx].to(device=device, dtype=torch.float32),
                    local_idx,
                    "source=compact_exact",
                )
            if not getattr(self, "hmap_pcn_activities", None):
                return None, -1, "source=no_dense_hmaps"
            seed_blocks = []
            for scale_idx in range(len(self.scales)):
                if scale_idx >= len(self.hmap_pcn_activities):
                    return None, -1, f"source=missing_scale_{scale_idx}"
                scale_hist = self.hmap_pcn_activities[scale_idx]
                if scale_hist is None or event_step >= int(scale_hist.shape[0]):
                    return None, -1, f"source=step_oob_scale_{scale_idx}"
                seed_blocks.append(
                    scale_hist[event_step].to(device=device, dtype=torch.float32)
                )
            if not seed_blocks:
                return None, -1, "source=no_seed_blocks"
            return (
                torch.clamp(torch.cat(seed_blocks, dim=0), min=0.0),
                -1,
                "source=dense_exact",
            )

        def _goal_step_position_xy(event_step: int) -> Optional[Tuple[float, float]]:
            if not hasattr(self, "hmap_loc"):
                return None
            try:
                if 0 <= int(event_step) < int(self.hmap_loc.shape[0]):
                    pos = self.hmap_loc[int(event_step)]
                    return float(pos[0]), float(pos[1])
            except Exception:
                return None
            return None

        def _step_inside_goal_radius(event_step: int) -> bool:
            pos_xy = _goal_step_position_xy(event_step)
            if pos_xy is None:
                return False
            return bool(
                math.hypot(float(pos_xy[0]) - float(goal_x), float(pos_xy[1]) - float(goal_z))
                <= float(goal_radius) + 1e-6
            )

        def _normalize_goal_seed_vector(
            seed_vec: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            seed_vec = torch.clamp(
                seed_vec.to(device=device, dtype=torch.float32),
                min=0.0,
            )
            peak = float(torch.max(torch.abs(seed_vec)).item())
            if peak <= eps:
                return None
            return seed_vec / peak

        def _aggregate_goal_seed_vectors(
            seed_vecs: List[torch.Tensor],
            aggregation: str,
        ) -> Optional[torch.Tensor]:
            if not seed_vecs:
                return None
            aggregation = str(aggregation).strip().lower()
            stacked = torch.stack(
                [
                    torch.clamp(vec.to(device=device, dtype=torch.float32), min=0.0)
                    for vec in seed_vecs
                ],
                dim=0,
            )
            if aggregation == "mean":
                aggregated = torch.mean(stacked, dim=0)
            else:
                aggregated = torch.max(stacked, dim=0)[0]
            return _normalize_goal_seed_vector(aggregated)

        def _collect_goal_contact_candidate_rows(
            event_group: List[int],
            room_mask_t: torch.Tensor,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            valid_group_steps = sorted(
                int(step)
                for step in set(event_group or [])
                if 0 <= int(step) < int(self.step_count)
            )
            if not valid_group_steps:
                return [], {
                    "first_contact_step": -1,
                    "candidate_steps": [],
                    "contributor_steps": [],
                    "contributor_local_indices": [],
                    "contact_count": 0,
                }

            first_contact_step = int(valid_group_steps[0])
            candidate_steps = set(valid_group_steps)
            post_contact_steps = int(
                max(0, getattr(self, "goal_map_goal_seed_post_contact_steps", 2))
            )
            for step in range(
                int(first_contact_step + 1),
                int(first_contact_step + post_contact_steps + 1),
            ):
                if 0 <= int(step) < int(self.step_count):
                    candidate_steps.add(int(step))

            candidate_rows: List[Dict[str, Any]] = []
            for step in sorted(candidate_steps):
                if not _step_inside_goal_radius(step):
                    continue
                seed_vec, local_idx, seed_log = _get_exact_unified_activation_for_step(
                    int(step)
                )
                if seed_vec is None:
                    continue
                seed_vec = torch.where(
                    room_mask_t,
                    torch.clamp(seed_vec.to(device=device, dtype=torch.float32), min=0.0),
                    torch.zeros_like(seed_vec, device=device, dtype=torch.float32),
                )
                normalized_seed = _normalize_goal_seed_vector(seed_vec)
                if normalized_seed is None:
                    continue
                candidate_rows.append(
                    {
                        "seed_vec": normalized_seed,
                        "local_idx": int(local_idx),
                        "event_step": int(step),
                        "seed_log": (
                            f"{seed_log},clip=room,"
                            f"goal_contact={'event' if step in valid_group_steps else 'dwell'}"
                        ),
                    }
                )

            return candidate_rows, {
                "first_contact_step": int(first_contact_step),
                "candidate_steps": [int(step) for step in sorted(candidate_steps)],
                "contributor_steps": [
                    int(row["event_step"]) for row in candidate_rows
                ],
                "contributor_local_indices": [
                    int(row["local_idx"])
                    for row in candidate_rows
                    if int(row["local_idx"]) >= 0
                ],
                "contact_count": int(len(candidate_rows)),
            }

        def _build_sparse_seed_from_location(
            seed_x: float,
            seed_y: float,
            support_mask_t: torch.Tensor,
        ) -> torch.Tensor:
            sparse_seed = torch.zeros(num_pc_total, dtype=torch.float32, device=device)
            if centers is None or visited is None:
                return sparse_seed
            target = torch.tensor([float(seed_x), float(seed_y)], dtype=torch.float32, device=device)
            for scale_idx in range(len(self.scales)):
                s_start = int(self.unified_pcn.scale_boundaries[scale_idx])
                s_end = int(self.unified_pcn.scale_boundaries[scale_idx + 1])
                scale_support = support_mask_t[s_start:s_end] & visited[s_start:s_end]
                if not bool(torch.any(scale_support).item()):
                    continue
                scale_centers = centers[s_start:s_end]
                scale_dist = torch.sum((scale_centers - target.unsqueeze(0)) ** 2, dim=1)
                scale_dist = torch.where(
                    scale_support,
                    scale_dist,
                    torch.full_like(scale_dist, float("inf")),
                )
                best_local = int(torch.argmin(scale_dist).item())
                if torch.isfinite(scale_dist[best_local]):
                    sparse_seed[s_start + best_local] = 1.0
            return sparse_seed

        def _checkpoint_doorway_band_mask(
            assignment: Dict[str, Any],
            room_mask_t: torch.Tensor,
        ) -> torch.Tensor:
            checkpoint_idx = int(assignment["sink_idx"])
            support_region = assignment.get("support_region")
            fallback_xy = checkpoint_positions.get(checkpoint_idx, (0.0, 0.0))
            if support_region is None or centers is None:
                return room_mask_t
            band_radius = float(
                max(0.10, 0.35 * float(max(0.15, self.checkpoint_visit_radius)))
            )
            doorway_d2 = self._support_region_distance_sq_torch(
                centers[:, 0],
                centers[:, 1],
                support_region=support_region,
                default_point=fallback_xy,
            )
            doorway_band = room_mask_t & (doorway_d2 <= (band_radius * band_radius))
            if bool(torch.any(doorway_band).item()):
                return doorway_band
            return room_mask_t

        def _weight_checkpoint_seed_toward_doorway(
            seed_vec: torch.Tensor,
            assignment: Dict[str, Any],
            room_mask_t: torch.Tensor,
        ) -> Tuple[torch.Tensor, str]:
            if centers is None:
                return seed_vec, "doorway_weight=skipped(no_centers)"
            support_region = assignment.get("support_region")
            checkpoint_idx = int(assignment["sink_idx"])
            fallback_xy = checkpoint_positions.get(checkpoint_idx, (0.0, 0.0))
            doorway_d2 = self._support_region_distance_sq_torch(
                centers[:, 0],
                centers[:, 1],
                support_region=support_region,
                default_point=fallback_xy,
            )
            sigma = float(max(0.12, 0.45 * float(max(0.15, self.checkpoint_visit_radius))))
            sigma_sq = sigma * sigma
            doorway_gain = torch.exp(-doorway_d2 / max(1e-9, 2.0 * sigma_sq))
            doorway_gain = 0.10 + (0.90 * doorway_gain)
            weighted_seed = torch.where(
                room_mask_t,
                torch.clamp(seed_vec * doorway_gain.to(device=device, dtype=torch.float32), min=0.0),
                torch.zeros_like(seed_vec, device=device, dtype=torch.float32),
            )
            return weighted_seed, f"doorway_weight=sigma_{sigma:.3f}"

        def _select_seed_for_event_group(
            assignment: Dict[str, Any],
            event_group: List[int],
            room_mask_t: torch.Tensor,
            event_step_override: Optional[int] = None,
            seed_clip_mode: str = "auto",
            sparse_fallback_clip_mode: Optional[str] = None,
        ) -> Tuple[Optional[torch.Tensor], int, int, str]:
            event_step = (
                int(event_step_override)
                if event_step_override is not None
                else _get_event_step_from_group(event_group)
            )
            if event_step < 0:
                return None, -1, -1, "source=no_event_step"

            def _resolve_clip_mask(clip_mode: Optional[str]) -> Tuple[torch.Tensor, str]:
                clip_mode_norm = str(clip_mode or "auto").strip().lower()
                if clip_mode_norm == "auto":
                    clip_mode_norm = (
                        "doorway_band"
                        if str(assignment["sink_type"]) == "checkpoint"
                        else "room"
                    )
                if clip_mode_norm == "room":
                    return room_mask_t, "clip=room"
                if clip_mode_norm == "doorway_band" and str(assignment["sink_type"]) == "checkpoint":
                    doorway_mask_t = _checkpoint_doorway_band_mask(assignment, room_mask_t)
                    if bool(torch.any(doorway_mask_t).item()):
                        return doorway_mask_t, "clip=doorway_band"
                    return room_mask_t, "clip=room_fallback"
                return room_mask_t, "clip=room"

            clip_mode_norm = str(seed_clip_mode or "auto").strip().lower()
            clip_mask_t, clip_log = _resolve_clip_mask(seed_clip_mode)

            event_seed, local_idx, seed_source_log = _get_exact_unified_activation_for_step(
                event_step
            )
            if event_seed is not None:
                event_seed = torch.where(
                    clip_mask_t,
                    event_seed.to(device=device, dtype=torch.float32),
                    torch.zeros_like(event_seed, device=device, dtype=torch.float32),
                )
                if (
                    str(assignment["sink_type"]) == "checkpoint"
                    and clip_mode_norm == "room"
                ):
                    event_seed, weight_log = _weight_checkpoint_seed_toward_doorway(
                        event_seed,
                        assignment,
                        room_mask_t,
                    )
                    seed_source_log = f"{seed_source_log},{weight_log}"
                peak = float(torch.max(torch.abs(event_seed)).item())
                if peak > eps:
                    return (
                        event_seed,
                        int(local_idx),
                        int(event_step),
                        f"{seed_source_log},{clip_log}",
                    )

            fallback_clip_mask_t, fallback_clip_log = _resolve_clip_mask(
                sparse_fallback_clip_mode if sparse_fallback_clip_mode is not None else seed_clip_mode
            )
            fallback_xy = (
                float(self.hmap_loc[event_step, 0]),
                float(self.hmap_loc[event_step, 1]),
            ) if 0 <= event_step < int(self.hmap_loc.shape[0]) else (
                float(goal_x),
                float(goal_z),
            )
            if str(assignment["sink_type"]) == "checkpoint":
                fallback_xy = checkpoint_positions.get(
                    int(assignment["sink_idx"]),
                    fallback_xy,
                )
            elif str(assignment["sink_type"]) == "goal":
                fallback_xy = (float(goal_x), float(goal_z))
            event_seed = _build_sparse_seed_from_location(
                seed_x=float(fallback_xy[0]),
                seed_y=float(fallback_xy[1]),
                support_mask_t=fallback_clip_mask_t,
            )
            local_idx = -1
            seed_source_log = "source=sparse_nearest_pc"
            event_seed = torch.where(
                fallback_clip_mask_t,
                event_seed.to(device=device, dtype=torch.float32),
                torch.zeros_like(event_seed, device=device, dtype=torch.float32),
            )
            peak = float(torch.max(torch.abs(event_seed)).item())
            if peak <= eps:
                return None, -1, event_step, f"{seed_source_log},{fallback_clip_log},peak=0"
            return (
                event_seed,
                int(local_idx),
                int(event_step),
                f"{seed_source_log},{fallback_clip_log}",
            )

        def _select_goal_consensus_seed(
            assignment: Dict[str, Any],
            event_groups: List[List[int]],
            room_mask_t: torch.Tensor,
        ) -> Tuple[List[Tuple[torch.Tensor, int, int]], List[str], str, Dict[str, Any]]:
            goal_seed_mode = str(
                getattr(self, "goal_map_goal_seed_mode", "contact_ensemble")
            ).strip().lower()
            within_visit_aggregation = str(
                getattr(self, "goal_map_goal_seed_within_visit_aggregation", "max")
            ).strip().lower()
            across_visit_aggregation = str(
                getattr(self, "goal_map_goal_seed_across_visit_aggregation", "mean")
            ).strip().lower()
            post_contact_steps = int(
                max(0, getattr(self, "goal_map_goal_seed_post_contact_steps", 2))
            )
            state_normalization = str(
                getattr(self, "goal_map_goal_seed_state_normalization", "peak")
            ).strip().lower()
            replay_retention_mode = str(
                getattr(self, "goal_map_goal_replay_retention_mode", "baseline")
            ).strip().lower()
            replay_retention_alpha = float(
                getattr(self, "goal_map_goal_seed_self_retention_alpha", 0.15)
            )

            if goal_seed_mode == "exact_contact":
                for event_group in reversed(event_groups):
                    candidate_rows, visit_debug = _collect_goal_contact_candidate_rows(
                        event_group,
                        room_mask_t,
                    )
                    if not candidate_rows:
                        continue
                    chosen = candidate_rows[0]
                    chosen_log = (
                        f"{chosen['seed_log']},"
                        f"goal_seed_mode=exact_contact,"
                        f"goal_visit_contacts={visit_debug['contact_count']},"
                        f"goal_post_contact_steps={post_contact_steps}"
                    )
                    return (
                        [
                            (
                                chosen["seed_vec"],
                                int(chosen["local_idx"]),
                                int(chosen["event_step"]),
                            )
                        ],
                        [chosen_log],
                        "goal_contact_exact_seed",
                        {
                            "goal_seed_mode": "exact_contact",
                            "goal_replay_retention_mode": replay_retention_mode,
                            "goal_seed_within_visit_aggregation": within_visit_aggregation,
                            "goal_seed_across_visit_aggregation": across_visit_aggregation,
                            "goal_seed_post_contact_steps": int(post_contact_steps),
                            "goal_seed_state_normalization": state_normalization,
                            "goal_seed_contact_state_count": 1,
                            "goal_seed_visit_count": 1,
                            "goal_seed_visit_contact_counts": [1],
                            "goal_seed_contributor_steps": [int(chosen["event_step"])],
                            "goal_seed_contributor_local_indices": (
                                [int(chosen["local_idx"])]
                                if int(chosen["local_idx"]) >= 0
                                else []
                            ),
                            "goal_replay_self_retention_alpha": (
                                float(replay_retention_alpha)
                                if replay_retention_mode == "boosted"
                                else 0.0
                            ),
                            "goal_seed_used_association_step": False,
                        },
                    )
                return [], [], "event_seed", {}

            visit_rows: List[Dict[str, Any]] = []
            total_contributor_steps: List[int] = []
            total_contributor_local_indices: List[int] = []
            total_contact_state_count = 0
            for event_group in event_groups:
                candidate_rows, visit_debug = _collect_goal_contact_candidate_rows(
                    event_group,
                    room_mask_t,
                )
                if not candidate_rows:
                    continue
                visit_seed = _aggregate_goal_seed_vectors(
                    [row["seed_vec"] for row in candidate_rows],
                    aggregation=within_visit_aggregation,
                )
                if visit_seed is None:
                    continue
                visit_rows.append(
                    {
                        "seed_vec": visit_seed,
                        "event_step": int(max(visit_debug["contributor_steps"])),
                        "contact_count": int(visit_debug["contact_count"]),
                        "contributor_steps": [
                            int(step) for step in visit_debug["contributor_steps"]
                        ],
                        "contributor_local_indices": [
                            int(idx) for idx in visit_debug["contributor_local_indices"]
                        ],
                    }
                )
                total_contact_state_count += int(visit_debug["contact_count"])
                total_contributor_steps.extend(
                    int(step) for step in visit_debug["contributor_steps"]
                )
                total_contributor_local_indices.extend(
                    int(idx) for idx in visit_debug["contributor_local_indices"]
                )

            if not visit_rows:
                return [], [], "event_seed", {}

            final_goal_seed = _aggregate_goal_seed_vectors(
                [row["seed_vec"] for row in visit_rows],
                aggregation=across_visit_aggregation,
            )
            if final_goal_seed is None:
                return [], [], "event_seed", {}

            final_event_step = int(
                max((int(step) for step in total_contributor_steps), default=-1)
            )
            final_log = (
                "source=goal_contact_ensemble,clip=room,"
                f"goal_seed_mode=contact_ensemble,"
                f"goal_seed_contacts={int(total_contact_state_count)},"
                f"goal_seed_visits={len(visit_rows)},"
                f"goal_seed_within_visit_aggregation={within_visit_aggregation},"
                f"goal_seed_across_visit_aggregation={across_visit_aggregation},"
                f"goal_post_contact_steps={post_contact_steps}"
            )
            return (
                [
                    (
                        final_goal_seed,
                        -1,
                        int(final_event_step),
                    )
                ],
                [final_log],
                "goal_contact_ensemble_seed",
                {
                    "goal_seed_mode": "contact_ensemble",
                    "goal_replay_retention_mode": replay_retention_mode,
                    "goal_seed_within_visit_aggregation": within_visit_aggregation,
                    "goal_seed_across_visit_aggregation": across_visit_aggregation,
                    "goal_seed_post_contact_steps": int(post_contact_steps),
                    "goal_seed_state_normalization": state_normalization,
                    "goal_seed_contact_state_count": int(total_contact_state_count),
                    "goal_seed_visit_count": int(len(visit_rows)),
                    "goal_seed_visit_contact_counts": [
                        int(row["contact_count"]) for row in visit_rows
                    ],
                    "goal_seed_contributor_steps": sorted(
                        {int(step) for step in total_contributor_steps}
                    ),
                    "goal_seed_contributor_local_indices": sorted(
                        {int(idx) for idx in total_contributor_local_indices if int(idx) >= 0}
                    ),
                    "goal_replay_self_retention_alpha": (
                        float(replay_retention_alpha)
                        if replay_retention_mode == "boosted"
                        else 0.0
                    ),
                    "goal_seed_used_association_step": False,
                },
            )

        def _select_checkpoint_consensus_seed(
            assignment: Dict[str, Any],
            event_groups: List[List[int]],
            room_mask_t: torch.Tensor,
        ) -> Tuple[List[Tuple[torch.Tensor, int, int]], List[str], str, Dict[str, Any]]:
            checkpoint_candidate_steps = [
                int(_get_event_step_from_group(group))
                for group in event_groups
                if int(_get_event_step_from_group(group)) >= 0
            ]

            # Prefer a single exact directional crossing, mirroring the old
            # targeted goal seeding style: one experienced state, not a mixture.
            # But do not assume the most recent logged crossing is the best one:
            # crossing detection can lag into the next room and shift the seed
            # away from the doorway. Rank candidate crossings first by distance
            # to the actual checkpoint center, then by distance to the doorway
            # support region, then by recency.
            checkpoint_idx = int(assignment["sink_idx"])
            checkpoint_center_xy = checkpoint_positions.get(checkpoint_idx, (0.0, 0.0))
            checkpoint_candidate_center_distances: List[float] = []
            checkpoint_candidate_support_distances: List[float] = []
            for chosen_step in checkpoint_candidate_steps:
                if 0 <= int(chosen_step) < int(self.hmap_loc.shape[0]):
                    candidate_xy = self.hmap_loc[int(chosen_step)]
                    checkpoint_center_dist = float(
                        math.hypot(
                            float(candidate_xy[0]) - float(checkpoint_center_xy[0]),
                            float(candidate_xy[1]) - float(checkpoint_center_xy[1]),
                        )
                    )
                    candidate_dist_sq = self._support_region_distance_sq(
                        float(candidate_xy[0]),
                        float(candidate_xy[1]),
                        assignment.get("support_region"),
                        default_point=checkpoint_center_xy,
                    )
                    checkpoint_candidate_center_distances.append(
                        checkpoint_center_dist
                    )
                    checkpoint_candidate_support_distances.append(
                        float(math.sqrt(max(0.0, candidate_dist_sq)))
                    )
                else:
                    checkpoint_candidate_center_distances.append(float("inf"))
                    checkpoint_candidate_support_distances.append(float("inf"))

            ranked_checkpoint_candidates = sorted(
                zip(
                    checkpoint_candidate_steps,
                    checkpoint_candidate_center_distances,
                    checkpoint_candidate_support_distances,
                ),
                key=lambda item: (
                    float(item[1]),
                    float(item[2]),
                    -int(item[0]),
                ),
            )

            # Overlap threshold: reject seeds whose activation mass falls
            # mostly on already-rewarded (downstream) cells.
            _overlap_reject_frac = float(
                getattr(self, "checkpoint_seed_overlap_reject_fraction", 0.5)
            )

            for (
                chosen_step,
                chosen_center_dist,
                chosen_support_dist,
            ) in ranked_checkpoint_candidates:
                replay_seed_vec, replay_local_idx, replay_event_step, replay_seed_log = _select_seed_for_event_group(
                    assignment,
                    [],
                    room_mask_t,
                    event_step_override=int(chosen_step),
                    seed_clip_mode="room",
                    sparse_fallback_clip_mode="doorway_band",
                )
                if replay_seed_vec is None:
                    continue
                # Check overlap with already_rewarded: if the majority of
                # this seed's activation mass sits on cells that are already
                # claimed by a prior replay, this trajectory was recorded on
                # the downstream side — skip it.
                _seed_abs = torch.abs(replay_seed_vec)
                _seed_total_mass = float(torch.sum(_seed_abs).item())
                if _seed_total_mass > eps:
                    _seed_overlap_mass = float(
                        torch.sum(_seed_abs[already_rewarded]).item()
                    )
                    _seed_overlap_frac = _seed_overlap_mass / _seed_total_mass
                    if _seed_overlap_frac > _overlap_reject_frac:
                        print(
                            f"[SEED-OVERLAP] {assignment['source_name']}: "
                            f"rejecting step {int(chosen_step)} — "
                            f"{_seed_overlap_frac:.1%} of activation mass "
                            f"overlaps already_rewarded"
                        )
                        continue
                chosen_log = (
                    f"{replay_seed_log},"
                    f"checkpoint_exact=step{int(replay_event_step)},"
                    f"checkpoint_center_dist={float(chosen_center_dist):.3f},"
                    f"checkpoint_support_dist={float(chosen_support_dist):.3f},"
                    f"checkpoint_candidates={len(checkpoint_candidate_steps)}"
                )
                return (
                    [
                        (
                            replay_seed_vec,
                            int(replay_local_idx),
                            int(replay_event_step),
                        )
                    ],
                    [chosen_log],
                    "checkpoint_exact_crossing_seed",
                    {
                        "checkpoint_candidate_event_steps": [
                            int(step) for step in checkpoint_candidate_steps
                        ],
                        "checkpoint_candidate_center_distances": [
                            float(dist) for dist in checkpoint_candidate_center_distances
                        ],
                        "checkpoint_candidate_support_distances": [
                            float(dist) for dist in checkpoint_candidate_support_distances
                        ],
                        "checkpoint_selected_event_step": int(replay_event_step),
                        "checkpoint_selected_center_distance": float(
                            chosen_center_dist
                        ),
                        "checkpoint_selected_support_distance": float(
                            chosen_support_dist
                        ),
                    },
                )

            candidate_rows: List[Dict[str, Any]] = []
            for event_group in event_groups:
                seed_vec, local_idx, event_step, seed_log = _select_seed_for_event_group(
                    assignment,
                    event_group,
                    room_mask_t,
                    seed_clip_mode="doorway_band",
                    sparse_fallback_clip_mode="doorway_band",
                )
                if seed_vec is None:
                    continue
                # Overlap filter: skip trajectories whose activation mass
                # falls mostly on already-rewarded cells.
                _cand_abs = torch.abs(seed_vec)
                _cand_total = float(torch.sum(_cand_abs).item())
                if _cand_total > eps:
                    _cand_overlap = float(
                        torch.sum(_cand_abs[already_rewarded]).item()
                    )
                    if _cand_overlap / _cand_total > _overlap_reject_frac:
                        print(
                            f"[SEED-OVERLAP] {assignment['source_name']}: "
                            f"rejecting fallback step {int(event_step)} — "
                            f"{_cand_overlap / _cand_total:.1%} overlap"
                        )
                        continue
                candidate_rows.append(
                    {
                        "seed_vec": seed_vec,
                        "local_idx": int(local_idx),
                        "event_step": int(event_step),
                        "seed_log": str(seed_log),
                    }
                )

            if not candidate_rows:
                return [], [], "event_seed", {}

            if len(candidate_rows) == 1:
                chosen = candidate_rows[0]
                replay_seed_vec, replay_local_idx, replay_event_step, replay_seed_log = _select_seed_for_event_group(
                    assignment,
                    [],
                    room_mask_t,
                    event_step_override=int(chosen["event_step"]),
                    seed_clip_mode="room",
                    sparse_fallback_clip_mode="doorway_band",
                )
                if replay_seed_vec is None:
                    replay_seed_vec = chosen["seed_vec"]
                    replay_local_idx = int(chosen["local_idx"])
                    replay_event_step = int(chosen["event_step"])
                    replay_seed_log = f"{chosen['seed_log']},replay_clip=fallback_candidate"
                chosen_log = (
                    f"{replay_seed_log},"
                    "checkpoint_consensus=1.0000,"
                    "checkpoint_candidates=1"
                )
                return (
                    [
                        (
                            replay_seed_vec,
                            int(replay_local_idx),
                            int(replay_event_step),
                        )
                    ],
                    [chosen_log],
                    "checkpoint_crossing_consensus_event_seed",
                    {
                        "checkpoint_candidate_event_steps": [
                            int(chosen["event_step"])
                        ],
                        "checkpoint_candidate_consensus_scores": [1.0],
                    },
                )

            candidate_mat = torch.stack(
                [
                    torch.clamp(row["seed_vec"], min=0.0).to(
                        device=device, dtype=torch.float32
                    )
                    for row in candidate_rows
                ],
                dim=0,
            )
            candidate_norms = torch.clamp(
                torch.norm(candidate_mat, p=2, dim=1, keepdim=True),
                min=eps,
            )
            candidate_unit = candidate_mat / candidate_norms
            sim = torch.matmul(candidate_unit, candidate_unit.transpose(0, 1))
            k = int(sim.shape[0])
            consensus_scores = (
                ((sim.sum(dim=1) - 1.0) / float(max(1, k - 1)))
                .detach()
                .cpu()
                .tolist()
            )
            ranked = sorted(
                range(len(candidate_rows)),
                key=lambda idx: (
                    float(consensus_scores[idx]),
                    int(candidate_rows[idx]["event_step"]),
                ),
                reverse=True,
            )
            best_idx = int(ranked[0])
            chosen = candidate_rows[best_idx]
            replay_seed_vec, replay_local_idx, replay_event_step, replay_seed_log = _select_seed_for_event_group(
                assignment,
                [],
                room_mask_t,
                event_step_override=int(chosen["event_step"]),
                seed_clip_mode="room",
                sparse_fallback_clip_mode="doorway_band",
            )
            if replay_seed_vec is None:
                replay_seed_vec = chosen["seed_vec"]
                replay_local_idx = int(chosen["local_idx"])
                replay_event_step = int(chosen["event_step"])
                replay_seed_log = f"{chosen['seed_log']},replay_clip=fallback_candidate"
            chosen_log = (
                f"{replay_seed_log},"
                f"checkpoint_consensus={float(consensus_scores[best_idx]):.4f},"
                f"checkpoint_candidates={len(candidate_rows)}"
            )
            return (
                [
                    (
                        replay_seed_vec,
                        int(replay_local_idx),
                        int(replay_event_step),
                    )
                ],
                [chosen_log],
                "checkpoint_crossing_consensus_event_seed",
                {
                    "checkpoint_candidate_event_steps": [
                        int(row["event_step"]) for row in candidate_rows
                    ],
                    "checkpoint_candidate_consensus_scores": [
                        float(score) for score in consensus_scores
                    ],
                },
            )

        source_names: List[str] = []
        path_maps_pre_smooth: Dict[str, torch.Tensor] = {}
        fill_maps_pre_smooth: Dict[str, torch.Tensor] = {}
        total_maps_pre_smooth: Dict[str, torch.Tensor] = {}
        seed_maps_pre_replay: Dict[str, torch.Tensor] = {}
        replay_cumulative_maps_pre_smooth: Dict[str, List[torch.Tensor]] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}
        source_support_masks: Dict[str, torch.Tensor] = {}
        # Store each room's replay seeds so we can re-replay downstream
        # rooms with expanded masks to fill boundary dead zones.
        room_replay_seeds_store: Dict[str, List[Tuple[torch.Tensor, int, int]]] = {}
        room_masks_store: Dict[str, torch.Tensor] = {}
        room_sink_types: Dict[str, str] = {}
        final_weights = torch.zeros(num_pc_total, dtype=torch.float32, device=device)
        # Track which cells have received reward from earlier replays.
        # Checkpoint masks are refined at replay time by subtracting this.
        already_rewarded = torch.zeros(
            num_pc_total, dtype=torch.bool, device=device
        )
        reward_threshold_fraction = float(
            getattr(self, "replay_boundary_reward_threshold", 0.01)
        )
        # Store per-source rewarded cell masks for exploit-time room masks
        replay_derived_room_cells: Dict[str, torch.Tensor] = {}

        for assignment in room_assignments:
            source_name = str(assignment["source_name"])
            sink_type = str(assignment["sink_type"])
            sink_idx = assignment["sink_idx"]
            support_region = assignment["support_region"]

            # --- Refine checkpoint masks at replay time ---
            # Determine upstream direction first, then flood fill from
            # only upstream-side boundary cells with already_rewarded
            # as additional barrier.
            if sink_type == "checkpoint":
                other_boundaries = assignment.get("other_boundaries")
                _replay_barrier = already_rewarded.clone()
                if other_boundaries is not None:
                    _replay_barrier |= other_boundaries.to(device)

                # Determine upstream direction from already_rewarded
                geom = checkpoint_geometry.get(int(sink_idx))
                cp_boundary = boundary_masks.get(int(sink_idx))
                cp_boundary_t = (
                    cp_boundary.to(device) if cp_boundary is not None else None
                )
                upstream_sign = None  # None = unknown, use all boundary cells
                signed_boundary = None
                if geom is not None and cp_boundary_t is not None:
                    frame_cx, frame_cy = geom["frame_center"]
                    nx, ny = geom["frame_normal"]
                    signed_boundary = (
                        ((centers[:, 0] - frame_cx) * nx)
                        + ((centers[:, 1] - frame_cy) * ny)
                    )
                    pos_side_rewarded = bool(
                        torch.any(
                            already_rewarded & cp_boundary_t
                            & (signed_boundary > 0)
                        ).item()
                    )
                    neg_side_rewarded = bool(
                        torch.any(
                            already_rewarded & cp_boundary_t
                            & (signed_boundary < 0)
                        ).item()
                    )
                    if pos_side_rewarded and not neg_side_rewarded:
                        assignment["direction_key"] = "pos_to_neg"
                        upstream_sign = -1.0  # upstream is negative side
                    elif neg_side_rewarded and not pos_side_rewarded:
                        assignment["direction_key"] = "neg_to_pos"
                        upstream_sign = 1.0  # upstream is positive side
                    else:
                        goal_signed = (
                            (float(goal_x) - frame_cx) * nx
                            + (float(goal_z) - frame_cy) * ny
                        )
                        # Upstream is AWAY from goal
                        assignment["direction_key"] = (
                            "pos_to_neg" if goal_signed > 0 else "neg_to_pos"
                        )
                        upstream_sign = -1.0 if goal_signed > 0 else 1.0
                    print(
                        f"[REPLAY-SIDE] {source_name}: "
                        f"dir={assignment['direction_key']} "
                        f"(pos_rewarded={pos_side_rewarded}, "
                        f"neg_rewarded={neg_side_rewarded})"
                    )

                # Flood fill from upstream-side boundary cells only
                cp_seed = cp_boundary_t.clone() if cp_boundary_t is not None else None
                if cp_seed is not None:
                    cp_seed = cp_seed & (~_replay_barrier)
                    # Restrict seed to upstream side
                    if upstream_sign is not None:
                        upstream_mask = (signed_boundary * upstream_sign) >= 0
                        cp_seed = cp_seed & upstream_mask
                    if bool(torch.any(cp_seed).item()):
                        cp_reachable = _flood_reachable_from_seed(
                            replay_transition, cp_seed, _replay_barrier
                        )
                        raw_room_mask_t = visited.clone().to(device) & cp_reachable
                        # Exclude downstream-side boundary cells — they
                        # belong to the goal room even though the flood
                        # reaches them (they're in the seed).
                        if upstream_sign is not None:
                            _downstream_barrier = cp_boundary_t & (~upstream_mask)
                            raw_room_mask_t = raw_room_mask_t & (~_downstream_barrier)
                        _cp_flood_count = int(
                            torch.count_nonzero(raw_room_mask_t).item()
                        )
                        print(
                            f"[REPLAY-FLOOD] {source_name}: "
                            f"upstream_seed={int(cp_seed.sum())}, "
                            f"reachable={_cp_flood_count}"
                        )
                    else:
                        raw_room_mask_t = torch.as_tensor(
                            assignment["room_mask"], dtype=torch.bool, device=device
                        ).view(-1) & (~already_rewarded)
                else:
                    raw_room_mask_t = torch.as_tensor(
                        assignment["room_mask"], dtype=torch.bool, device=device
                    ).view(-1) & (~already_rewarded)
            else:
                raw_room_mask_t = torch.as_tensor(
                    assignment["room_mask"], dtype=torch.bool, device=device
                ).view(-1)

            if not bool(torch.any(raw_room_mask_t).item()):
                print(
                    f"[REPLAY-SKIP] {source_name}: empty mask after "
                    f"already_rewarded subtraction"
                )
                continue

            room_mask_t, support_refine_log = _refine_checkpoint_support_mask(
                assignment,
                raw_room_mask_t,
            )
            if not bool(torch.any(room_mask_t).item()):
                raise RuntimeError(
                    f"{source_name}: refined room support is empty; "
                    f"raw_support={int(torch.count_nonzero(raw_room_mask_t).item())}"
                )
            # Store mask and sink type for boundary re-replay pass
            room_masks_store[source_name] = room_mask_t.clone()
            room_sink_types[source_name] = sink_type
            fallback_seed = torch.as_tensor(
                assignment["fallback_seed"], dtype=torch.float32, device=device
            ).view(-1)
            fallback_seed = torch.where(
                room_mask_t,
                fallback_seed,
                torch.zeros_like(fallback_seed),
            )

            if sink_type == "goal":
                event_groups = self._get_recent_goal_contact_event_groups(
                    goal_name=goal_name,
                    max_events=int(
                        max(1, getattr(self, "goal_map_goal_replay_event_count", 4))
                    ),
                )
            else:
                _primary_dir = str(assignment["direction_key"])
                event_groups = self._get_recent_checkpoint_crossing_event_groups_for_direction(
                    checkpoint_idx=int(sink_idx),
                    direction_key=_primary_dir,
                    max_events=int(
                        max(
                            1,
                            getattr(self, "goal_map_checkpoint_replay_event_count", 1),
                        )
                    ),
                )
                # If no events in primary direction, try the opposite direction.
                # The geometry-derived downstream_side can be wrong when the
                # checkpoint normal doesn't align with the actual maze layout.
                if not event_groups:
                    _opp_dir = "pos_to_neg" if _primary_dir == "neg_to_pos" else "neg_to_pos"
                    event_groups = self._get_recent_checkpoint_crossing_event_groups_for_direction(
                        checkpoint_idx=int(sink_idx),
                        direction_key=_opp_dir,
                        max_events=int(
                            max(
                                1,
                                getattr(self, "goal_map_checkpoint_replay_event_count", 1),
                            )
                        ),
                    )
                    if event_groups:
                        print(
                            f"[ROOM-REPLAY] {source_name}: no events for dir={_primary_dir}, "
                            f"using opposite dir={_opp_dir} ({len(event_groups)} groups)"
                        )

            # --- Diagnostic: event groups for this assignment ---
            _eg_flat = [s for g in event_groups for s in g] if event_groups else []
            print(
                f"[ROOM-REPLAY] {source_name}: event_groups={len(event_groups)}, "
                f"total_steps={len(_eg_flat)}, "
                f"dir_key={assignment.get('direction_key','n/a')}"
            )
            if not event_groups:
                # Extra detail: dump raw crossing steps for this checkpoint/direction
                if sink_type == "checkpoint":
                    _cs = getattr(self, "checkpoint_crossing_steps", {})
                    _cp_dirs = _cs.get(int(sink_idx), {})
                    _hist_start = int(max(0, self._get_reward_history_start_step()))
                    for _dk, _steps in _cp_dirs.items():
                        _post = [s for s in _steps if int(s) >= _hist_start]
                        print(
                            f"  [CROSSING-DEBUG] cp{sink_idx} dir={_dk}: "
                            f"total_steps={len(_steps)}, post_phase2={len(_post)}, "
                            f"history_start={_hist_start}"
                        )

            replay_event_seeds: List[Tuple[torch.Tensor, int, int]] = []
            replay_event_seed_logs: List[str] = []
            seed_debug: Dict[str, Any] = {}
            seed_source = "event_seed"
            if sink_type == "goal":
                (
                    replay_event_seeds,
                    replay_event_seed_logs,
                    seed_source,
                    seed_debug,
                ) = _select_goal_consensus_seed(
                    assignment,
                    event_groups,
                    room_mask_t,
                )
            else:
                (
                    replay_event_seeds,
                    replay_event_seed_logs,
                    seed_source,
                    seed_debug,
                ) = _select_checkpoint_consensus_seed(
                    assignment,
                    event_groups,
                    room_mask_t,
                )
            if not replay_event_seeds:
                if float(torch.max(torch.abs(fallback_seed)).item()) <= eps and sink_type == "checkpoint":
                    checkpoint_center = checkpoint_positions.get(int(sink_idx))
                    if checkpoint_center is not None:
                        fallback_seed, _ = self._build_unified_seed_from_position(
                            goal_x=float(checkpoint_center[0]),
                            goal_z=float(checkpoint_center[1]),
                            rcn_device=device,
                            seed_sigma=float(self.checkpoint_visit_radius),
                            center_radius_factor=getattr(
                                self, "goal_map_checkpoint_seed_center_radius_factor", 0.0
                            ),
                            center_min_radius=getattr(
                                self, "goal_map_checkpoint_seed_center_min_radius", 0.0
                            ),
                            support_region=support_region,
                        )
                        fallback_seed = torch.where(
                            room_mask_t,
                            fallback_seed.to(device=device, dtype=torch.float32),
                            torch.zeros_like(fallback_seed, device=device, dtype=torch.float32),
                        )
                if float(torch.max(torch.abs(fallback_seed)).item()) <= eps and sink_type == "goal":
                    fallback_seed = _build_sparse_seed_from_location(
                        seed_x=float(goal_x),
                        seed_y=float(goal_z),
                        support_mask_t=room_mask_t,
                    )
                    fallback_seed = torch.where(
                        room_mask_t,
                        fallback_seed.to(device=device, dtype=torch.float32),
                        torch.zeros_like(fallback_seed, device=device, dtype=torch.float32),
                    )
                if float(torch.max(torch.abs(fallback_seed)).item()) > eps:
                    replay_event_seeds = [(fallback_seed, -1, -1)]
                    seed_source = (
                        "doorway_seed_fallback" if sink_type == "checkpoint" else "goal_seed_fallback"
                    )

            # Store seeds for potential boundary re-replay later
            room_replay_seeds_store[source_name] = list(replay_event_seeds)

            room_mask_float = room_mask_t.to(dtype=torch.float32)
            room_transition = replay_transition * (
                room_mask_float.unsqueeze(0) * room_mask_float.unsqueeze(1)
            )
            room_delta = torch.zeros(num_pc_total, dtype=torch.float32, device=device)
            seed_event_steps = [int(event_step) for _, _local_idx, event_step in replay_event_seeds]
            seed_local_indices = [int(local_idx) for _, local_idx, _event_step in replay_event_seeds]
            if sink_type == "goal" and seed_debug:
                contributor_steps = seed_debug.get("goal_seed_contributor_steps")
                if contributor_steps:
                    seed_event_steps = [int(step) for step in contributor_steps]
                contributor_local_indices = seed_debug.get(
                    "goal_seed_contributor_local_indices"
                )
                if contributor_local_indices is not None:
                    seed_local_indices = [int(idx) for idx in contributor_local_indices]
            goal_retention_mode = str(
                getattr(self, "goal_map_goal_replay_retention_mode", "baseline")
            ).strip().lower()
            goal_retention_alpha = (
                float(getattr(self, "goal_map_goal_seed_self_retention_alpha", 0.15))
                if sink_type == "goal" and goal_retention_mode == "boosted"
                else 0.0
            )
            goal_replay_carry_scale = (
                float(getattr(self, "goal_map_goal_replay_carry_scale", 1.0))
                if sink_type == "goal"
                else 1.0
            )
            source_replay_timesteps = (
                int(goal_replay_timesteps) if sink_type == "goal" else int(replay_timesteps)
            )
            source_replay_tau = (
                float(goal_replay_tau) if sink_type == "goal" else float(replay_tau)
            )
            record_goal_replay_trajectory = bool(
                sink_type == "goal"
                and getattr(self, "goal_map_debug_replay_record_goal", False)
            )
            replay_cumulative_steps: List[torch.Tensor] = []
            if replay_event_seeds:
                pre_replay_seed_stack = torch.stack(
                    [
                        torch.where(
                            room_mask_t,
                            torch.clamp(
                                seed_vec.to(device=device, dtype=torch.float32),
                                min=0.0,
                            ),
                            torch.zeros_like(
                                seed_vec, device=device, dtype=torch.float32
                            ),
                        )
                        for seed_vec, _local_idx, _event_step in replay_event_seeds
                    ],
                    dim=0,
                )
                seed_maps_pre_replay[source_name] = (
                    torch.max(pre_replay_seed_stack, dim=0)[0]
                    .detach()
                    .cpu()
                    .clone()
                    .view(-1)
                )
            for seed_vec, _local_idx, _event_step in replay_event_seeds:
                v_t = torch.where(
                    room_mask_t,
                    torch.clamp(seed_vec.to(device=device, dtype=torch.float32), min=0.0),
                    torch.zeros_like(seed_vec, device=device, dtype=torch.float32),
                )
                for time_step in range(source_replay_timesteps):
                    if float(torch.max(torch.abs(v_t)).item()) <= eps:
                        break
                    u_t = torch.relu(v_t)
                    u_t_peak = torch.clamp(torch.max(torch.abs(u_t)), min=eps)
                    u_hat_t = u_t / u_t_peak
                    room_delta = room_delta + (
                        float(math.exp(-float(time_step) / source_replay_tau)) * u_hat_t
                    )
                    if record_goal_replay_trajectory:
                        replay_cumulative_steps.append(
                            torch.where(
                                room_mask_t,
                                room_delta,
                                torch.zeros_like(room_delta),
                            )
                            .detach()
                            .cpu()
                            .clone()
                            .view(-1)
                        )
                    state_carry_scale = 1.0
                    if sink_type == "goal":
                        state_carry_scale = float(goal_replay_carry_scale)
                        if goal_retention_alpha > 0.0:
                            state_carry_scale = (
                                float(state_carry_scale) + float(goal_retention_alpha)
                            )
                    state_carry = float(state_carry_scale) * v_t
                    v_t = room_mask_float * torch.tanh(
                        torch.relu(state_carry + torch.mv(room_transition, v_t))
                    )

            room_delta = torch.where(room_mask_t, room_delta, torch.zeros_like(room_delta))
            if room_norm_mode == "per_room_peak":
                room_peak = float(torch.max(torch.abs(room_delta)).item())
                room_values = room_delta / max(room_peak, eps) if room_peak > eps else room_delta
            else:
                room_values = room_delta
            room_values = torch.where(
                room_mask_t,
                room_values,
                torch.zeros_like(room_values),
            )

            outside_support_mass = float(
                torch.max(
                    torch.abs(
                        room_values[~room_mask_t]
                    )
                ).item()
            ) if bool(torch.any(~room_mask_t).item()) else 0.0
            if outside_support_mass > 1e-8:
                raise RuntimeError(
                    f"{source_name}: nonzero mass outside owned support "
                    f"before composition (max={outside_support_mass:.3e})"
                )
            # Overlap with already-rewarded cells is prevented by the
            # already_rewarded mask applied when building checkpoint masks.
            # No explicit goal-room overlap guard needed.
            goal_validation = (
                _compute_goal_validation_summary(assignment, room_values)
                if sink_type == "goal"
                else {}
            )

            source_names.append(source_name)
            path_maps_pre_smooth[source_name] = room_values.detach().cpu().clone().view(-1)
            fill_maps_pre_smooth[source_name] = torch.zeros_like(
                room_values.detach().cpu()
            ).view(-1)
            total_maps_pre_smooth[source_name] = room_values.detach().cpu().clone().view(-1)
            source_support_masks[source_name] = room_mask_t.detach().cpu().clone().view(-1)
            if replay_cumulative_steps:
                replay_cumulative_maps_pre_smooth[source_name] = replay_cumulative_steps
            metadata_map[source_name] = {
                "source_role": "room_sink",
                "sink_type": sink_type,
                "checkpoint_idx": None if sink_idx is None else int(sink_idx),
                "owned_component_ids": [],
                "selected_route_position": int(assignment["route_pos"]),
                "support_count": int(torch.count_nonzero(room_mask_t).item()),
                "raw_support_count": int(torch.count_nonzero(raw_room_mask_t).item()),
                "replay_count": int(len(replay_event_seeds)),
                "replay_event_count": int(len(replay_event_seeds)),
                "replay_steps": int(source_replay_timesteps),
                "direction_key": assignment["direction_key"],
                "seed_source": seed_source,
                "seed_event_steps": list(seed_event_steps),
                "seed_local_indices": list(seed_local_indices),
                "seed_event_logs": list(replay_event_seed_logs),
                "room_normalization_mode": room_norm_mode,
                "mask_source": "geometry_route",
                "support_refine_log": support_refine_log,
                "outside_support_mass_precompose": outside_support_mass,
            }
            if seed_debug:
                metadata_map[source_name].update(seed_debug)
            if goal_validation:
                metadata_map[source_name]["goal_validation"] = goal_validation
            if sink_type == "goal":
                metadata_map[source_name]["goal_replay_carry_scale"] = float(
                    goal_replay_carry_scale
                )
                metadata_map[source_name]["goal_replay_timesteps"] = int(
                    source_replay_timesteps
                )
                metadata_map[source_name]["goal_replay_tau"] = float(source_replay_tau)
            final_weights = torch.where(room_mask_t, room_values, final_weights)

            # --- Track which cells received reward from this replay ---
            room_peak = float(torch.max(torch.abs(room_values)).item())
            _reward_thr = max(eps, reward_threshold_fraction * max(room_peak, eps))
            newly_rewarded = room_mask_t & (torch.abs(room_values) > _reward_thr)
            already_rewarded |= newly_rewarded
            replay_derived_room_cells[source_name] = newly_rewarded.detach().cpu().clone()

            # Claim the ENTIRE spread mask as already_rewarded — not just
            # cells that received above-threshold reward.  Replay may not
            # deposit meaningful reward in every cell of the room (corners,
            # edges far from the seed), but those cells still belong to this
            # room's territory.  Claiming the full mask prevents subsequent
            # replays from including unreached cells in their masks.
            _spread_claimed = int(
                torch.count_nonzero(room_mask_t & (~already_rewarded)).item()
            )
            if _spread_claimed > 0:
                already_rewarded |= room_mask_t
                print(
                    f"[REPLAY-TRACK] {source_name}: claimed full spread mask "
                    f"(+{_spread_claimed} cells beyond threshold-rewarded)"
                )

            _newly_count = int(torch.count_nonzero(newly_rewarded).item())
            _total_rewarded = int(torch.count_nonzero(already_rewarded).item())
            print(
                f"[REPLAY-TRACK] {source_name}: "
                f"newly_rewarded={_newly_count}, "
                f"total_rewarded={_total_rewarded}"
            )

        # --- Boundary re-replay pass ---
        # Each checkpoint's downstream boundary cells were excluded from
        # the goal (or previous checkpoint) replay to prevent leakage.
        # Now re-replay the downstream room from its original seeds with
        # an expanded mask that includes those boundary cells, producing
        # a natural reward gradient instead of a warm-started false peak.
        for _ra_idx, _ra in enumerate(room_assignments):
            if _ra["sink_type"] != "checkpoint":
                continue
            _cp_idx = int(_ra["sink_idx"])
            _cp_boundary = boundary_masks.get(_cp_idx)
            if _cp_boundary is None:
                continue
            _cp_boundary_dev = _cp_boundary.to(device)

            # Determine which boundary cells are downstream (goal-side)
            _geom = checkpoint_geometry.get(_cp_idx)
            if _geom is None:
                continue
            _fcx, _fcy = _geom["frame_center"]
            _fnx, _fny = _geom["frame_normal"]
            _signed = (
                ((centers[:, 0] - _fcx) * _fnx)
                + ((centers[:, 1] - _fcy) * _fny)
            )
            _dir_key = _ra.get("direction_key", "")
            if _dir_key == "neg_to_pos":
                _upstream_sign = 1.0
            elif _dir_key == "pos_to_neg":
                _upstream_sign = -1.0
            else:
                continue
            _downstream_mask = (_signed * _upstream_sign) < 0
            _downstream_boundary = _cp_boundary_dev & _downstream_mask
            _unclaimed = _downstream_boundary & (~already_rewarded)
            _unclaimed_count = int(torch.count_nonzero(_unclaimed).item())
            if _unclaimed_count == 0:
                continue

            # Find the downstream room (replayed before this checkpoint)
            _downstream_src = None
            for _j in range(_ra_idx - 1, -1, -1):
                _downstream_src = str(room_assignments[_j]["source_name"])
                break
            if _downstream_src is None or _downstream_src not in room_masks_store:
                continue
            _ds_seeds = room_replay_seeds_store.get(_downstream_src, [])
            if not _ds_seeds:
                continue

            _ds_original_mask = room_masks_store[_downstream_src]
            _expanded_mask = _ds_original_mask | _unclaimed
            _expanded_float = _expanded_mask.to(dtype=torch.float32)
            _expanded_transition = replay_transition * (
                _expanded_float.unsqueeze(0) * _expanded_float.unsqueeze(1)
            )

            # Replay parameters matching the downstream room's type
            _ds_sink_type = room_sink_types.get(_downstream_src, "")
            _ds_carry_scale = (
                float(getattr(self, "goal_map_goal_replay_carry_scale", 1.0))
                if _ds_sink_type == "goal" else 1.0
            )
            _ds_retention_alpha = 0.0
            if _ds_sink_type == "goal":
                _ds_ret_mode = str(
                    getattr(self, "goal_map_goal_replay_retention_mode", "baseline")
                ).strip().lower()
                if _ds_ret_mode == "boosted":
                    _ds_retention_alpha = float(
                        getattr(self, "goal_map_goal_seed_self_retention_alpha", 0.15)
                    )

            # Re-replay from original seeds with expanded mask
            _fill_delta = torch.zeros(
                num_pc_total, dtype=torch.float32, device=device
            )
            for _sv, _li, _es in _ds_seeds:
                _v_t = torch.where(
                    _expanded_mask,
                    torch.clamp(
                        _sv.to(device=device, dtype=torch.float32), min=0.0
                    ),
                    torch.zeros(
                        num_pc_total, dtype=torch.float32, device=device
                    ),
                )
                for _ts in range(replay_timesteps):
                    if float(torch.max(torch.abs(_v_t)).item()) <= eps:
                        break
                    _u_t = torch.relu(_v_t)
                    _u_peak = torch.clamp(
                        torch.max(torch.abs(_u_t)), min=eps
                    )
                    _u_hat = _u_t / _u_peak
                    _fill_delta = _fill_delta + (
                        float(math.exp(-float(_ts) / replay_tau)) * _u_hat
                    )
                    _sc = float(_ds_carry_scale)
                    if _ds_retention_alpha > 0.0:
                        _sc = _sc + float(_ds_retention_alpha)
                    _state_carry = _sc * _v_t
                    _v_t = _expanded_float * torch.tanh(
                        torch.relu(
                            _state_carry
                            + torch.mv(_expanded_transition, _v_t)
                        )
                    )

            # Normalize per room peak (same as main replay)
            _fill_delta = torch.where(
                _expanded_mask, _fill_delta, torch.zeros_like(_fill_delta)
            )
            if room_norm_mode == "per_room_peak":
                _fp = float(torch.max(torch.abs(_fill_delta)).item())
                _fill_values = (
                    _fill_delta / max(_fp, eps) if _fp > eps else _fill_delta
                )
            else:
                _fill_values = _fill_delta

            # Only keep reward on the previously-unclaimed boundary cells
            _boundary_values = torch.where(
                _unclaimed, _fill_values, torch.zeros_like(_fill_values)
            )
            _boundary_peak = float(
                torch.max(torch.abs(_boundary_values)).item()
            )
            if _boundary_peak <= eps:
                print(
                    f"[BOUNDARY-REREPLAY] {_downstream_src}: no reward "
                    f"reached {_unclaimed_count} boundary cells from "
                    f"cp{_cp_idx}"
                )
                continue

            # Merge into downstream room's maps
            _fill_cpu = _boundary_values.detach().cpu().clone().view(-1)
            _src_total = torch.as_tensor(
                total_maps_pre_smooth[_downstream_src],
                dtype=torch.float32,
            ).view(-1)
            total_maps_pre_smooth[_downstream_src] = torch.where(
                _unclaimed.cpu(), _fill_cpu, _src_total
            )
            path_maps_pre_smooth[_downstream_src] = (
                total_maps_pre_smooth[_downstream_src]
            )
            final_weights = torch.where(
                _unclaimed, _boundary_values, final_weights
            )
            source_support_masks[_downstream_src] = (
                source_support_masks[_downstream_src] | _unclaimed.cpu()
            )
            if _downstream_src in replay_derived_room_cells:
                replay_derived_room_cells[_downstream_src] = (
                    replay_derived_room_cells[_downstream_src]
                    | _unclaimed.cpu()
                )
            already_rewarded |= _unclaimed
            _filled_count = int(
                torch.count_nonzero(
                    _boundary_values.abs() > eps
                ).item()
            )
            print(
                f"[BOUNDARY-REREPLAY] {_downstream_src}: filled "
                f"{_filled_count}/{_unclaimed_count} downstream boundary "
                f"cells from cp{_cp_idx} "
                f"(peak={_boundary_peak:.4f})"
            )

        # --- Store replay-derived room masks for exploit-time use ---
        self._replay_derived_room_masks = {}
        for src_name, cell_mask in replay_derived_room_cells.items():
            self._replay_derived_room_masks[src_name] = cell_mask
        # Also store on goal_rcn so they persist to pkl
        goal_rcn._replay_derived_room_masks = dict(self._replay_derived_room_masks)

        for source_name, support_mask_cpu in source_support_masks.items():
            source_map_cpu = torch.as_tensor(
                total_maps_pre_smooth[source_name],
                dtype=torch.float32,
            ).view(-1)
            postcompose_outside_mass = float(
                torch.max(torch.abs(source_map_cpu[~support_mask_cpu])).item()
            ) if bool(torch.any(~support_mask_cpu).item()) else 0.0
            metadata_map[source_name]["outside_support_mass_postcompose"] = (
                postcompose_outside_mass
            )
            if postcompose_outside_mass > 1e-8:
                raise RuntimeError(
                    f"{source_name}: nonzero mass outside owned support "
                    f"after composition bookkeeping (max={postcompose_outside_mass:.3e})"
                )

        goal_rcn.w_in = torch.clamp(final_weights.unsqueeze(0), min=0.0)
        goal_rcn.w_in_effective = goal_rcn.w_in.clone()
        goal_rcn.goal_map_segment_debug = {
            "source_names": list(source_names),
            "path_maps_pre_smooth": path_maps_pre_smooth,
            "fill_maps_pre_smooth": fill_maps_pre_smooth,
            "total_maps_pre_smooth": total_maps_pre_smooth,
            "seed_maps_pre_replay": seed_maps_pre_replay,
            "replay_cumulative_maps_pre_smooth": replay_cumulative_maps_pre_smooth,
            "source_support_masks": source_support_masks,
            "metadata": metadata_map,
        }

        selected_txt = "[" + ",".join(f"cp{int(idx)}" for idx in selected_order) + "]"
        goal_replay_override_log = ""
        if int(goal_replay_timesteps) != int(replay_timesteps):
            goal_replay_override_log += f",goal_steps={int(goal_replay_timesteps)}"
        if abs(float(goal_replay_tau) - float(replay_tau)) > 1e-9:
            goal_replay_override_log += f",goal_tau={float(goal_replay_tau):.2f}"
        replay_log = (
            "goal_map=paper_room_local_replay("
            f"selected={selected_txt},"
            f"sources={len(source_names)},"
            f"room_norm={room_norm_mode},"
            f"replay_steps={replay_timesteps},"
            f"tau={replay_tau:.2f}{goal_replay_override_log})"
        )
        goal_rcn.goal_map_debug_log = (
            f"{replay_log},{history_log},{compact_log},{route_log},"
            f"{centers_log},{visibility_log},{path_kernel_log},{local_kernel_log},"
            f"{substrate_log}"
        )
        return goal_rcn, goal_rcn.goal_map_debug_log

    def _compute_reward_weights(
        self,
        goal_x,
        goal_z,
        _w_exp_norm,
        base_rcn,
        seed_sigma: Optional[float] = None,
        goal_name: Optional[str] = None,
    ):
        """
        Build reward weights for any (goal_x, goal_z) using one unified seed and one
        unified replay pass. Used for both real goals and doorway sub-goals.

        Args:
            goal_x (float): Goal x-coordinate (hmap_loc[:, 0] convention)
            goal_z (float): Goal z-coordinate (hmap_loc[:, 1] convention)
            _w_exp_norm (torch.Tensor): Legacy fallback transition matrix
            base_rcn: RCN to deepcopy; provides device and C_REWARD

        Returns:
            tuple: (goal_rcn, seed_log)
        """
        goal_rcn = copy.deepcopy(base_rcn)
        goal_rcn.w_in = torch.zeros_like(goal_rcn.w_in)
        goal_rcn.w_in_effective = goal_rcn.w_in.clone()
        goal_rcn.reward_cell_activations = torch.zeros_like(goal_rcn.reward_cell_activations)
        goal_rcn.goal_map_segment_debug = None

        rcn_device = goal_rcn.w_in.device
        seed_activations, seed_log = self._build_unified_seed_from_position(
            goal_x=goal_x,
            goal_z=goal_z,
            rcn_device=rcn_device,
            seed_sigma=seed_sigma,
            center_radius_factor=getattr(
                self, "goal_map_goal_seed_center_radius_factor", 0.0
            ),
            center_min_radius=getattr(
                self, "goal_map_goal_seed_center_min_radius", 0.0
            ),
        )
        goal_map_mode = str(
            getattr(self, "unified_goal_map_mode", "paper_room_local_replay")
        ).strip().lower()
        relay_logs = []
        checkpoint_seed_bank = []
        checkpoint_seed_support_id_bank = {}
        checkpoint_support_region_bank = {}
        need_checkpoint_seed_bank = bool(
            getattr(self, "goal_map_use_checkpoint_relays", False)
        ) or goal_map_mode == "paper_room_local_replay"
        if need_checkpoint_seed_bank:
            valid_checkpoints = getattr(self, "_valid_checkpoints", None)
            if self.detected_doorways:
                for checkpoint_idx, (cx, cy) in enumerate(self.detected_doorways):
                    if valid_checkpoints is not None and checkpoint_idx not in valid_checkpoints:
                        continue
                    checkpoint_center_radius_factor = getattr(
                        self, "goal_map_checkpoint_seed_center_radius_factor", None
                    )
                    if (
                        checkpoint_center_radius_factor is None
                        or float(checkpoint_center_radius_factor) <= 0.0
                    ):
                        checkpoint_center_radius_factor = getattr(
                            self, "goal_map_goal_seed_center_radius_factor", 0.0
                        )
                    checkpoint_center_min_radius = getattr(
                        self, "goal_map_checkpoint_seed_center_min_radius", None
                    )
                    if (
                        checkpoint_center_min_radius is None
                        or float(checkpoint_center_min_radius) <= 0.0
                    ):
                        checkpoint_center_min_radius = getattr(
                            self, "goal_map_goal_seed_center_min_radius", 0.0
                        )
                    checkpoint_support_region, checkpoint_region_log = (
                        self._infer_checkpoint_support_region(
                            float(cx),
                            float(cy),
                        )
                    )
                    checkpoint_seed, checkpoint_log = self._build_unified_seed_from_position(
                        goal_x=float(cx),
                        goal_z=float(cy),
                        rcn_device=rcn_device,
                        seed_sigma=seed_sigma,
                        center_radius_factor=checkpoint_center_radius_factor,
                        center_min_radius=checkpoint_center_min_radius,
                        support_region=checkpoint_support_region,
                    )
                    if float(torch.max(torch.abs(checkpoint_seed)).item()) <= 1e-9:
                        continue
                    cp_norm = torch.amax(torch.abs(checkpoint_seed)).clamp(min=1e-8)
                    checkpoint_seed_bank.append(
                        (int(checkpoint_idx), (checkpoint_seed / cp_norm).clone())
                    )
                    checkpoint_seed_support_id_bank[int(checkpoint_idx)] = (
                        self._relay_support_indices(
                            checkpoint_seed,
                            topk=16,
                            threshold_fraction=0.35,
                        )
                    )
                    checkpoint_support_region_bank[int(checkpoint_idx)] = checkpoint_support_region
                    relay_logs.append(
                        f"cp{checkpoint_idx}({checkpoint_log} | {checkpoint_region_log})"
                    )
        if relay_logs and bool(getattr(self, "goal_map_use_checkpoint_relays", False)):
            seed_log = f"{seed_log} | relays[{'; '.join(relay_logs)}]"

        if hasattr(goal_rcn, "_ensure_experience_buffers"):
            goal_rcn._ensure_experience_buffers()
        if hasattr(self, "_configure_unified_replay_settings"):
            self._configure_unified_replay_settings(goal_rcn, goal_map_mode=True)
        history_start = self._get_reward_history_start_step()
        history_log = f"history_start={history_start},experience_graph=raw_online"
        if bool(getattr(self, "goal_map_use_denoised_experience_graph", True)):
            denoised_counts, denoised_log = self._get_cached_unified_denoised_transition_counts(
                start_step=history_start,
                end_step=int(self.step_count),
            )
            if denoised_counts is not None:
                goal_rcn.experience_transition_counts = denoised_counts
                history_log = (
                    f"history_start={history_start},"
                    f"experience_graph=denoised({denoised_log})"
                )
            else:
                history_log = (
                    f"history_start={history_start},"
                    f"experience_graph=raw_online({denoised_log})"
                )
        goal_rcn.use_experience_replay = True
        goal_rcn.replay_timesteps = int(
            max(1, getattr(self, "goal_map_replay_timesteps", getattr(goal_rcn, "replay_timesteps", 1)))
        )

        replay_log = "goal_map=no_builder"
        if hasattr(goal_rcn, "build_goal_map_from_custom_activations"):
            goal_rcn.goal_map_local_anchor_weights = None
            goal_rcn.goal_map_local_anchor_gate_d2 = None
            goal_rcn.goal_map_local_anchor_query_sigma = None
            goal_rcn.goal_map_local_anchor_names = None
            checkpoint_mode = str(
                getattr(self, "goal_map_checkpoint_mode", "implicit_cascade")
            ).strip().lower()
            visibility_log = self._sync_unified_recurrent_visibility_mask(prune_weights=True)
            visibility_mask, _, _ = self._get_unified_spatial_visibility_mask(
                goal_rcn.w_in.device
            )
            path_kernel, _, path_kernel_log = self._get_unified_reward_spatial_kernel(
                goal_rcn.w_in.device,
                sigma=float(getattr(self, "goal_map_path_constraint_sigma", 4.0)),
                topk=int(getattr(self, "goal_map_path_constraint_topk", 128)),
            )
            local_kernel, _, local_kernel_log = self._get_unified_reward_spatial_kernel(
                goal_rcn.w_in.device,
                sigma=float(getattr(self, "goal_map_local_constraint_sigma", 2.2)),
                topk=int(getattr(self, "goal_map_local_constraint_topk", 24)),
            )
            # The path backbone should follow experienced sequential structure,
            # not line-of-sight between place-field centers. Using the geometry
            # visibility mask here can suppress valid doorway transitions whose
            # field centers happen to lie on opposite sides of a wall. Keep the
            # path graph local via `path_kernel`, and reserve wall blocking for
            # the neighbor spread / smoothing graph.
            path_constraint = path_kernel if path_kernel is not None else visibility_mask
            segmented_log = "segmented=disabled"
            segmented_data = None
            goal_map_mode = str(
                getattr(self, "unified_goal_map_mode", "paper_room_local_replay")
            ).strip().lower()
            if goal_map_mode == "paper_room_local_replay":
                goal_rcn, replay_log = self._build_paper_room_local_goal_map(
                    goal_rcn=goal_rcn,
                    goal_name=goal_name,
                    goal_x=float(goal_x),
                    goal_z=float(goal_z),
                    seed_activations=seed_activations,
                    checkpoint_seed_bank=checkpoint_seed_bank,
                    checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                    checkpoint_support_region_bank=checkpoint_support_region_bank,
                    history_start=history_start,
                    history_log=history_log,
                    visibility_log=visibility_log,
                    path_constraint=path_constraint,
                    local_kernel=local_kernel,
                    path_kernel_log=path_kernel_log,
                    local_kernel_log=local_kernel_log,
                )
                replay_log = f"{replay_log},smooth=disabled"
                return goal_rcn, f"{seed_log} | {replay_log}"
            trajectory_state_goal_map = goal_map_mode == "trajectory_state_backbone"
            room_masked_experience_goal_map = (
                goal_map_mode == "room_masked_experience_replay"
            )

            recurrent_weights_max = None
            local_transition = None

            def _prepare_segment_source(
                source_seed: torch.Tensor,
                source_counts: torch.Tensor,
                source_kind: str,
                owned_room_mask: Optional[torch.Tensor] = None,
                support_node_ids: Optional[List[int]] = None,
                doorway_blocker_node_ids: Optional[List[int]] = None,
                own_doorway_node_ids: Optional[List[int]] = None,
                bridge_node_ids: Optional[List[int]] = None,
                doorway_support_regions: Optional[List[dict]] = None,
                support_time_indices: Optional[List[int]] = None,
            ):
                nonlocal recurrent_weights_max, local_transition
                source_rcn = copy.deepcopy(goal_rcn)
                source_device = (
                    source_rcn.device if hasattr(source_rcn, "device") else goal_rcn.w_in.device
                )
                source_rcn.experience_transition_counts = source_counts.clone().to(source_device)
                source_rcn.goal_map_checkpoint_seeds = None
                source_rcn.goal_map_checkpoint_presence = None
                source_rcn.goal_map_checkpoint_weights = None
                source_rcn.goal_map_seed_gain = 1.0
                source_rcn.goal_map_path_constraint = path_constraint
                source_rcn.goal_map_neighbor_constraint = local_kernel
                source_rcn.goal_map_spatial_constraint = (
                    visibility_mask if visibility_mask is not None else local_kernel
                )
                base_neighbor_steps = int(
                    max(0, getattr(source_rcn, "goal_map_neighbor_steps", 0))
                )
                base_neighbor_decay = float(
                    getattr(source_rcn, "goal_map_neighbor_decay", 0.84)
                )
                base_neighbor_seed_scale = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(source_rcn, "goal_map_neighbor_seed_scale", 0.60),
                        ),
                    )
                )
                if source_kind == "goal":
                    # Keep goal-room spread broad enough to form a usable
                    # gradient toward the goal/checkpoint handoff without
                    # making it dramatically weaker than checkpoint fill.
                    source_rcn.goal_map_neighbor_steps = (
                        max(base_neighbor_steps, int(math.ceil(base_neighbor_steps * 1.10)))
                        if base_neighbor_steps > 0
                        else 0
                    )
                    source_rcn.goal_map_neighbor_decay = min(
                        0.90, max(base_neighbor_decay, base_neighbor_decay + 0.02)
                    )
                    source_rcn.goal_map_neighbor_seed_scale = min(
                        1.0, max(base_neighbor_seed_scale, base_neighbor_seed_scale * 1.05)
                    )
                elif source_kind == "checkpoint":
                    # Checkpoints still need room-wide coverage, but keep the
                    # lateral fill close to the goal-room tuning so one side
                    # does not collapse while the other becomes a plateau.
                    source_rcn.goal_map_neighbor_steps = (
                        max(base_neighbor_steps, int(math.ceil(base_neighbor_steps * 1.15)))
                        if base_neighbor_steps > 0
                        else 0
                    )
                    source_rcn.goal_map_neighbor_decay = min(
                        0.90, max(base_neighbor_decay, base_neighbor_decay + 0.02)
                    )
                    source_rcn.goal_map_neighbor_seed_scale = min(
                        1.0, max(base_neighbor_seed_scale, base_neighbor_seed_scale * 1.10)
                    )

                if recurrent_weights_max is None or recurrent_weights_max.device != source_device:
                    recurrent_weights_max = torch.max(
                        self.unified_pcn.w_rec_unified.to(source_device), dim=0
                    )[0]
                    local_transition = None
                if local_transition is None:
                    local_transition = source_rcn._prepare_replay_transition(
                        recurrent_weights_max
                    )

                path_transition = source_rcn._prepare_goal_map_path_transition(
                    local_transition
                )
                neighbor_transition = source_rcn._prepare_goal_map_neighbor_transition(
                    recurrent_weights_max
                )
                support_mask = self._build_segment_support_mask(
                    source_counts=source_counts,
                    source_seed=source_seed,
                    local_kernel=local_kernel,
                    support_node_ids=support_node_ids,
                    doorway_blocker_node_ids=doorway_blocker_node_ids,
                    own_doorway_node_ids=own_doorway_node_ids,
                    bridge_node_ids=bridge_node_ids,
                    doorway_support_regions=doorway_support_regions,
                    support_time_indices=support_time_indices,
                    scale_arrays=(
                        segmented_data["scale_arrays"] if segmented_data is not None else None
                    ),
                )
                if owned_room_mask is not None:
                    owned_room_mask = torch.as_tensor(
                        owned_room_mask,
                        dtype=torch.bool,
                        device=source_device,
                    ).view(-1)
                    if (
                        owned_room_mask.numel() == support_mask.numel()
                        and bool(torch.any(owned_room_mask).item())
                    ):
                        support_mask = owned_room_mask
                support_count = int(torch.count_nonzero(support_mask).item())
                if support_count > 0:
                    support_float = support_mask.to(
                        device=source_device, dtype=torch.float32
                    ).view(-1)
                    transition_gate = support_float.unsqueeze(0) * support_float.unsqueeze(1)
                    path_transition = path_transition * transition_gate
                    neighbor_transition = neighbor_transition * transition_gate
                return (
                    source_rcn,
                    path_transition,
                    neighbor_transition,
                    support_mask,
                    support_count,
                    int(max(0, getattr(source_rcn, "goal_map_neighbor_steps", 0))),
                    float(getattr(source_rcn, "goal_map_neighbor_decay", 0.84)),
                    float(
                        min(
                            1.0,
                            max(
                                0.0,
                                getattr(source_rcn, "goal_map_neighbor_seed_scale", 0.60),
                            ),
                        )
                    ),
                )

            def _relay_amplitude_from_arrival(arrival_support: float) -> float:
                checkpoint_scale = float(
                    max(0.0, getattr(self, "goal_map_checkpoint_scale", 0.25))
                )
                relay_threshold = float(
                    max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
                )
                support_mode = str(
                    getattr(self, "goal_map_checkpoint_support_mode", "threshold_normalized")
                ).strip().lower()
                if support_mode in {"threshold_normalized", "threshold", "normalized"}:
                    support_scale = float(
                        min(1.0, max(0.0, arrival_support / max(relay_threshold, 1e-6)))
                    )
                    return checkpoint_scale * support_scale
                return checkpoint_scale * max(0.0, arrival_support)

            if (
                bool(getattr(self, "goal_map_use_segmented_checkpoint_histories", True))
                and checkpoint_seed_bank
            ):
                goal_radius = 0.8
                for goal in getattr(self, "goals", []):
                    goal_loc = goal.get("location", [])
                    if (
                        len(goal_loc) >= 2
                        and abs(float(goal_loc[0]) - float(goal_x)) <= 1e-6
                        and abs(float(goal_loc[1]) - float(goal_z)) <= 1e-6
                    ):
                        goal_radius = float(goal.get("radius", goal_radius))
                        break
                segmented_data, segmented_log = self._build_segmented_checkpoint_transition_graphs(
                    goal_x=float(goal_x),
                    goal_z=float(goal_z),
                    goal_radius=float(goal_radius),
                    checkpoint_seeds=[seed for _, seed in checkpoint_seed_bank],
                    start_step=history_start,
                    end_step=int(self.step_count),
                )

            if segmented_data is not None:
                route_selected_data = None
                route_selected_log = "route_select=disabled"
                checkpoint_bridge_steps = int(
                    max(0, getattr(self, "goal_map_checkpoint_bridge_steps", 1))
                )
                goal_bridge_steps = int(
                    max(0, getattr(self, "goal_map_goal_bridge_steps", 0))
                )
                arrival_buffer_steps = int(
                    max(
                        0,
                        getattr(
                            self,
                            "goal_map_checkpoint_arrival_buffer_steps",
                            checkpoint_bridge_steps,
                        ),
                    )
                )
                if checkpoint_mode in {
                    "route_selected_tree",
                    "directed_route_memory_tree",
                    "multi_route_memory_graph",
                } and checkpoint_seed_bank:
                    if recurrent_weights_max is None or recurrent_weights_max.device != goal_rcn.w_in.device:
                        recurrent_weights_max = torch.max(
                            self.unified_pcn.w_rec_unified.to(goal_rcn.w_in.device),
                            dim=0,
                        )[0]
                    candidate_checkpoint_ids = sorted(
                        int(idx)
                        for idx in segmented_data.get("checkpoint_counts", {}).keys()
                    )
                    if checkpoint_mode == "multi_route_memory_graph":
                        try:
                            route_selected_data, route_selected_log = (
                                self._build_multi_route_memory_graph(
                                    goal_name=goal_name,
                                    goal_x=float(goal_x),
                                    goal_z=float(goal_z),
                                    goal_radius=float(goal_radius),
                                    checkpoint_seed_bank=checkpoint_seed_bank,
                                    checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                                    candidate_checkpoint_ids=candidate_checkpoint_ids,
                                    start_step=history_start,
                                    end_step=int(self.step_count),
                                    target_device=goal_rcn.w_in.device,
                                )
                            )
                        except Exception as exc:
                            route_selected_data, route_selected_log = (
                                self._compute_directed_route_memory_tree(
                                    goal_name=goal_name,
                                    goal_x=float(goal_x),
                                    goal_z=float(goal_z),
                                    goal_radius=float(goal_radius),
                                    checkpoint_seed_bank=checkpoint_seed_bank,
                                    checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                                    candidate_checkpoint_ids=candidate_checkpoint_ids,
                                    start_step=history_start,
                                    end_step=int(self.step_count),
                                    target_device=goal_rcn.w_in.device,
                                )
                            )
                            route_selected_log = f"{route_selected_log},fallback_from=multi_route_error({exc})"
                    elif checkpoint_mode == "directed_route_memory_tree":
                        route_selected_data, route_selected_log = (
                            self._compute_directed_route_memory_tree(
                                goal_name=goal_name,
                                goal_x=float(goal_x),
                                goal_z=float(goal_z),
                                goal_radius=float(goal_radius),
                                checkpoint_seed_bank=checkpoint_seed_bank,
                                checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                                candidate_checkpoint_ids=candidate_checkpoint_ids,
                                start_step=history_start,
                                end_step=int(self.step_count),
                                target_device=goal_rcn.w_in.device,
                            )
                        )
                    else:
                        route_selected_data, route_selected_log = (
                            self._compute_route_selected_checkpoint_tree(
                                reward_rcn=goal_rcn,
                                goal_seed=seed_activations,
                                checkpoint_seed_bank=checkpoint_seed_bank,
                                checkpoint_seed_support_id_bank=checkpoint_seed_support_id_bank,
                                candidate_checkpoint_ids=candidate_checkpoint_ids,
                                path_constraint=path_constraint,
                                local_kernel=local_kernel,
                                recurrent_weights_max=recurrent_weights_max,
                                arrival_buffer_steps=arrival_buffer_steps,
                                goal_name=goal_name,
                                goal_x=float(goal_x),
                                goal_z=float(goal_z),
                                goal_radius=float(goal_radius),
                                start_step=history_start,
                                end_step=int(self.step_count),
                            )
                        )
                checkpoint_doorway_blockers = sorted(
                    {
                        int(idx)
                        for ids in checkpoint_seed_support_id_bank.values()
                        for idx in ids
                    }
                )
                checkpoint_doorway_regions = [
                    region
                    for region in checkpoint_support_region_bank.values()
                    if region is not None
                ]
                room_partition = None
                room_partition_log = "roomown=disabled"
                room_component_ids = None
                room_visited_mask = None
                goal_component_idx = -1
                checkpoint_component_bank: Dict[int, List[int]] = {}
                room_goal_distances: Dict[int, float] = {}
                try:
                    room_partition, room_partition_log = self._get_unified_checkpoint_room_partition(
                        goal_rcn.w_in.device
                    )
                except Exception as exc:
                    room_partition = None
                    room_partition_log = "roomown=partition_error"
                    print(
                        f"[ROOM-PARTITION] ERROR in exploit-time replay: "
                        f"_get_unified_checkpoint_room_partition failed: {exc}"
                    )
                if room_partition is not None:
                    room_component_ids = room_partition.get("component_ids")
                    room_visited_mask = room_partition.get("visited")
                    checkpoint_component_bank = {
                        int(k): [int(v) for v in vals]
                        for k, vals in room_partition.get("checkpoint_components", {}).items()
                    }
                    goal_component_idx = int(
                        self._resolve_goal_component_from_partition(
                            room_partition,
                            goal_x=float(goal_x),
                            goal_z=float(goal_z),
                            target_device=goal_rcn.w_in.device,
                        )
                    )
                    if goal_component_idx >= 0:
                        room_goal_distances = self._room_graph_distances_from_goal(
                            room_partition.get("graph", {}),
                            goal_component_idx,
                        )

                def _build_owned_room_mask(
                    owned_components: List[int],
                    doorway_ids: Optional[List[int]] = None,
                    extra_ids: Optional[List[int]] = None,
                ) -> Optional[torch.Tensor]:
                    if room_component_ids is None or room_visited_mask is None:
                        return None
                    owned_set = {
                        int(comp_idx)
                        for comp_idx in owned_components
                        if int(comp_idx) >= 0
                    }
                    if not owned_set:
                        return None
                    comp_ids_t = room_component_ids.to(goal_rcn.w_in.device)
                    visited_t = room_visited_mask.to(goal_rcn.w_in.device)
                    mask = torch.zeros_like(visited_t, dtype=torch.bool)
                    for comp_idx in sorted(owned_set):
                        mask |= visited_t & (comp_ids_t == int(comp_idx))
                    for idx in doorway_ids or []:
                        idx = int(idx)
                        if 0 <= idx < int(mask.numel()):
                            mask[idx] = True
                    for idx in extra_ids or []:
                        idx = int(idx)
                        if not (0 <= idx < int(mask.numel())):
                            continue
                        comp_idx = int(comp_ids_t[idx].item())
                        if comp_idx < 0 or comp_idx in owned_set:
                            mask[idx] = True
                    return mask if bool(torch.any(mask).item()) else None

                def _get_checkpoint_owned_components(checkpoint_idx: int) -> List[int]:
                    adjacent = checkpoint_component_bank.get(int(checkpoint_idx), [])
                    if not adjacent:
                        return []
                    # Never let a checkpoint own the goal's room
                    adjacent = [c for c in adjacent if int(c) != int(goal_component_idx)]
                    if not adjacent:
                        return []
                    if not room_goal_distances:
                        return list(adjacent)
                    finite_adjacent = [
                        (int(comp_idx), float(room_goal_distances.get(int(comp_idx), float("inf"))))
                        for comp_idx in adjacent
                        if math.isfinite(float(room_goal_distances.get(int(comp_idx), float("inf"))))
                    ]
                    if not finite_adjacent:
                        return list(adjacent)
                    downstream_dist = min(dist for _, dist in finite_adjacent)
                    owned = [
                        comp_idx
                        for comp_idx, dist in finite_adjacent
                        if dist > downstream_dist + 1e-6
                    ]
                    if owned:
                        return owned
                    # If the partition only exposes one finite adjacent room, fall back
                    # to the farthest room available instead of letting the checkpoint
                    # own both sides.
                    finite_adjacent.sort(key=lambda item: (item[1], item[0]))
                    return [int(finite_adjacent[-1][0])]

                goal_doorway_ids = []
                active_checkpoint_order = list(segmented_data["checkpoint_order"])
                if route_selected_data is not None:
                    active_checkpoint_order = list(
                        route_selected_data.get("selected_order", active_checkpoint_order)
                    )
                if active_checkpoint_order:
                    goal_doorway_ids = checkpoint_seed_support_id_bank.get(
                        int(active_checkpoint_order[0]),
                        [],
                    )
                # --- Stage 3: Multi-route room ownership dispute resolution ---
                _resolved_checkpoint_rooms: Dict[int, List[int]] = {}
                _dispute_log_parts: List[str] = []
                if (
                    route_selected_data is not None
                    and route_selected_data.get("mode") == "multi_route_memory_graph"
                    and active_checkpoint_order
                ):
                    # Step 1: Compute raw claims for all active checkpoints
                    _raw_claims: Dict[int, List[int]] = {}
                    for _cp_idx in active_checkpoint_order:
                        _raw_claims[int(_cp_idx)] = _get_checkpoint_owned_components(int(_cp_idx))

                    # Step 2: Find rooms claimed by multiple checkpoints
                    _room_claimants: Dict[int, List[int]] = {}
                    for _cp_idx, _rooms in _raw_claims.items():
                        for _room in _rooms:
                            _room_claimants.setdefault(int(_room), []).append(int(_cp_idx))

                    # Step 3: Resolve disputes using edge strength to downstream target
                    _graph_edge_counts = route_selected_data.get("edge_counts", {})
                    _graph_goal_link_counts = route_selected_data.get("goal_link_counts", {})
                    _graph_parent_map = route_selected_data.get("parent_map", {})
                    _graph_depth = route_selected_data.get("checkpoint_depth", {})

                    _disputed_rooms: Dict[int, int] = {}  # room_comp -> winning cp_idx
                    for _room_comp, _claimants in _room_claimants.items():
                        if len(_claimants) <= 1:
                            continue
                        # Pick winner: strongest edge to downstream, then smaller depth, then lower index
                        _best_cp = _claimants[0]
                        _best_strength = -1.0
                        _best_depth = float("inf")
                        for _cp in _claimants:
                            _downstream = _graph_parent_map.get(int(_cp))
                            if _downstream is None:
                                _strength = float(_graph_goal_link_counts.get(int(_cp), 0))
                            else:
                                _strength = float(
                                    _graph_edge_counts.get((int(_cp), int(_downstream)), 0)
                                )
                            _depth = float(_graph_depth.get(int(_cp), float("inf")))
                            if (
                                _strength > _best_strength
                                or (_strength == _best_strength and _depth < _best_depth)
                                or (
                                    _strength == _best_strength
                                    and _depth == _best_depth
                                    and int(_cp) < int(_best_cp)
                                )
                            ):
                                _best_strength = _strength
                                _best_depth = _depth
                                _best_cp = _cp
                        _disputed_rooms[int(_room_comp)] = int(_best_cp)
                        _losers = [c for c in _claimants if c != _best_cp]
                        _dispute_log_parts.append(
                            f"room{_room_comp}:winner=cp{_best_cp}(str={_best_strength:.0f})"
                            f"_losers=[{','.join(f'cp{l}' for l in _losers)}]"
                        )

                    # Step 4: Build resolved ownership — losers keep their other rooms
                    for _cp_idx, _rooms in _raw_claims.items():
                        _resolved = []
                        for _room in _rooms:
                            if int(_room) in _disputed_rooms:
                                if _disputed_rooms[int(_room)] == int(_cp_idx):
                                    _resolved.append(int(_room))
                                # else: lost this room
                            else:
                                _resolved.append(int(_room))
                        _resolved_checkpoint_rooms[int(_cp_idx)] = _resolved

                def _get_resolved_checkpoint_owned_components(checkpoint_idx: int) -> List[int]:
                    """Return dispute-resolved room ownership if available, else raw."""
                    if _resolved_checkpoint_rooms and int(checkpoint_idx) in _resolved_checkpoint_rooms:
                        return _resolved_checkpoint_rooms[int(checkpoint_idx)]
                    return _get_checkpoint_owned_components(int(checkpoint_idx))

                full_source_counts = getattr(goal_rcn, "experience_transition_counts", None)
                if full_source_counts is None:
                    full_source_counts = segmented_data["goal_counts"]
                goal_bridge_ids = self._expand_node_ids_with_kernel(
                    goal_doorway_ids,
                    local_kernel,
                    steps=goal_bridge_steps,
                )
                goal_owned_room_mask = _build_owned_room_mask(
                    [int(goal_component_idx)] if int(goal_component_idx) >= 0 else [],
                    doorway_ids=goal_doorway_ids,
                    extra_ids=goal_bridge_ids,
                )
                (
                    goal_segment_rcn,
                    goal_path_transition,
                    goal_neighbor_transition,
                    goal_support_mask,
                    goal_support_count,
                    goal_neighbor_steps,
                    goal_neighbor_decay,
                    goal_neighbor_seed_scale,
                ) = _prepare_segment_source(
                    seed_activations,
                    full_source_counts,
                    "goal",
                    owned_room_mask=goal_owned_room_mask,
                    doorway_blocker_node_ids=checkpoint_doorway_blockers,
                    own_doorway_node_ids=goal_doorway_ids,
                    bridge_node_ids=goal_bridge_ids,
                    doorway_support_regions=checkpoint_doorway_regions,
                )
                segment_room_fill_gain = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(self, "goal_map_segment_room_fill_gain", 0.55),
                        ),
                    )
                )
                goal_core_gain = float(
                    max(0.0, getattr(self, "goal_map_goal_core_gain", 1.15))
                )
                checkpoint_core_gain = float(
                    max(0.0, getattr(self, "goal_map_checkpoint_core_gain", 1.00))
                )
                paper_replay_tau = float(
                    max(1e-6, getattr(self, "goal_map_paper_replay_tau", 8.0))
                )
                goal_replay_event_limit = int(
                    max(1, getattr(self, "goal_map_goal_replay_event_count", 4))
                )
                checkpoint_replay_event_limit = int(
                    max(
                        1,
                        getattr(self, "goal_map_checkpoint_replay_event_count", 1),
                    )
                )
                goal_room_decay_tau = float(
                    max(0.1, getattr(self, "goal_map_goal_room_decay_tau", 5.5))
                )
                checkpoint_room_decay_tau = float(
                    max(0.1, getattr(self, "goal_map_checkpoint_room_decay_tau", 4.0))
                )
                local_anchor_sigma = float(
                    max(0.05, getattr(self, "goal_map_local_anchor_sigma", 0.25))
                )
                local_anchor_query_sigma = float(
                    max(
                        0.05,
                        getattr(self, "goal_map_local_anchor_query_sigma", 0.50),
                    )
                )
                room_decay_floor = float(
                    min(
                        1.0,
                        max(0.0, getattr(self, "goal_map_room_decay_floor", 0.15)),
                    )
                )
                centers_t = None
                visited_t = None
                field_peak_xy_t = None
                compact_pc_mask_t = None
                compact_pc_radius_t = None
                reward_pc_compactness_log = "field_peaks=unavailable"
                spatial_centers, spatial_visited, _ = self._get_unified_pc_spatial_support(
                    goal_rcn.w_in.device
                )
                if spatial_centers is not None and spatial_visited is not None:
                    centers_t = spatial_centers.to(goal_rcn.w_in.device)
                    visited_t = spatial_visited.to(goal_rcn.w_in.device)
                (
                    field_peak_xy_raw,
                    _field_peak_val_raw,
                    compact_pc_mask_raw,
                    compact_pc_radius_raw,
                    reward_pc_compactness_log,
                ) = self._get_unified_pc_field_peak_data(goal_rcn.w_in.device)
                if field_peak_xy_raw is not None:
                    field_peak_xy_t = field_peak_xy_raw.to(goal_rcn.w_in.device)
                if compact_pc_mask_raw is not None:
                    compact_pc_mask_t = compact_pc_mask_raw.to(goal_rcn.w_in.device)
                if compact_pc_radius_raw is not None:
                    compact_pc_radius_t = compact_pc_radius_raw.to(goal_rcn.w_in.device)
                compact_history_t = None
                compact_positions_t = None
                compact_time_to_local = {}
                compact_time_indices = []
                compact_time_indices_np = np.empty(0, dtype=np.int64)
                local_anchor_scale_weights_t = None
                if (trajectory_state_goal_map or room_masked_experience_goal_map) and segmented_data is not None:
                    compact_time_indices = [
                        int(t)
                        for t in segmented_data.get("time_indices", [])
                        if 0 <= int(t) < int(self.hmap_loc.shape[0])
                    ]
                    if compact_time_indices:
                        compact_blocks = [
                            arr.index_select(
                                0,
                                torch.as_tensor(
                                    compact_time_indices,
                                    dtype=torch.long,
                                    device=arr.device,
                                ),
                            ).to(
                                goal_rcn.w_in.device, dtype=torch.float32
                            )
                            for arr in segmented_data.get("scale_arrays", [])
                        ]
                        if compact_blocks:
                            compact_history_t = torch.clamp(
                                torch.cat(compact_blocks, dim=1),
                                min=0.0,
                            )
                            compact_positions_t = torch.as_tensor(
                                self.hmap_loc[compact_time_indices, :2],
                                dtype=torch.float32,
                                device=goal_rcn.w_in.device,
                            )
                            compact_time_to_local = {
                                int(t): idx
                                for idx, t in enumerate(compact_time_indices)
                            }
                            compact_time_indices_np = np.asarray(
                                compact_time_indices,
                                dtype=np.int64,
                            )
                    scale_boundaries = list(
                        getattr(self.unified_pcn, "scale_boundaries", [])
                    )
                    if compact_history_t is not None and len(scale_boundaries) >= 2:
                        anchor_scale_weights = list(
                            getattr(
                                self,
                                "goal_map_local_anchor_scale_weights",
                                [1.0, 0.55, 0.25],
                            )
                        )
                        if not anchor_scale_weights:
                            anchor_scale_weights = [1.0, 0.55, 0.25]
                        local_anchor_scale_weights_t = torch.ones(
                            compact_history_t.shape[1],
                            dtype=torch.float32,
                            device=goal_rcn.w_in.device,
                        )
                        last_gain = float(anchor_scale_weights[-1])
                        for scale_idx, (start, end) in enumerate(
                            zip(scale_boundaries[:-1], scale_boundaries[1:])
                        ):
                            gain = float(
                                anchor_scale_weights[scale_idx]
                                if scale_idx < len(anchor_scale_weights)
                                else last_gain
                            )
                            if int(end) > int(start):
                                local_anchor_scale_weights_t[
                                    int(start) : int(end)
                                ] = max(0.0, gain)

                def _build_distance_decay_envelope(
                    support_mask: torch.Tensor,
                    support_region,
                    default_point,
                    tau: float,
                ) -> torch.Tensor:
                    if centers_t is None or visited_t is None:
                        return torch.as_tensor(
                            support_mask, dtype=torch.float32, device=goal_rcn.w_in.device
                        ).view(-1)
                    mask_t = torch.as_tensor(
                        support_mask, dtype=torch.bool, device=goal_rcn.w_in.device
                    ).view(-1)
                    d_sq = self._support_region_distance_sq_torch(
                        centers_t[:, 0],
                        centers_t[:, 1],
                        support_region,
                        default_point=default_point,
                    )
                    d = torch.sqrt(torch.clamp(d_sq, min=0.0))
                    envelope = room_decay_floor + (1.0 - room_decay_floor) * torch.exp(
                        -d / max(1e-6, float(tau))
                    )
                    envelope = torch.where(
                        visited_t & mask_t,
                        envelope,
                        torch.zeros_like(envelope),
                    )
                    return envelope

                def _apply_segment_budget(
                    values: torch.Tensor,
                    support_mask: torch.Tensor,
                    target_peak: float,
                ) -> Tuple[torch.Tensor, float]:
                    """
                    Normalize one segment contribution inside its own support and
                    then assign it a fixed peak budget.

                    This keeps rooms from competing based on how much raw support
                    they happened to claim during replay/history construction.
                    """
                    masked_values = torch.where(
                        torch.as_tensor(
                            support_mask,
                            dtype=torch.bool,
                            device=goal_rcn.w_in.device,
                        ).view(-1),
                        values,
                        torch.zeros_like(values),
                    )
                    peak = float(torch.max(masked_values).item())
                    if peak <= 1e-8 or target_peak <= 1e-8:
                        return torch.zeros_like(masked_values), peak
                    return masked_values * (float(target_peak) / peak), peak

                def _apply_compact_pc_gate(
                    values: torch.Tensor,
                    branch_label: str,
                ) -> Tuple[torch.Tensor, str]:
                    if (
                        not bool(getattr(self, "goal_map_use_compact_pc_gate", True))
                        or compact_pc_mask_t is None
                        or compact_pc_mask_t.numel() != values.numel()
                    ):
                        return values, f"{branch_label}:compact=off"
                    before = int(torch.count_nonzero(values > 1e-8).item())
                    gated = values * compact_pc_mask_t.to(values.dtype)
                    after = int(torch.count_nonzero(gated > 1e-8).item())
                    if compact_pc_radius_t is not None and before > 0:
                        kept_idx = torch.nonzero(gated > 1e-8, as_tuple=False).squeeze(1)
                        if kept_idx.numel() > 0:
                            max_kept_r = float(
                                torch.max(compact_pc_radius_t[kept_idx]).item()
                            )
                            return (
                                gated,
                                f"{branch_label}:compact={before}->{after}(max_r={max_kept_r:.2f})",
                            )
                    return gated, f"{branch_label}:compact={before}->{after}"

                def _apply_fill_source_envelope(
                    values: torch.Tensor,
                    source_xy: Tuple[float, float],
                    branch_label: str,
                ) -> Tuple[torch.Tensor, str]:
                    if field_peak_xy_t is None or field_peak_xy_t.numel() == 0:
                        return values, f"{branch_label}:env=off"
                    tau = float(
                        max(
                            1e-6,
                            getattr(self, "goal_map_fill_source_envelope_tau", 6.0),
                        )
                    )
                    floor = float(
                        min(
                            0.95,
                            max(
                                0.0,
                                getattr(self, "goal_map_fill_source_envelope_floor", 0.10),
                            ),
                        )
                    )
                    src_x, src_z = float(source_xy[0]), float(source_xy[1])
                    d = torch.sqrt(
                        torch.clamp(
                            (field_peak_xy_t[:, 0] - src_x) ** 2
                            + (field_peak_xy_t[:, 1] - src_z) ** 2,
                            min=0.0,
                        )
                    )
                    gain = floor + (1.0 - floor) * torch.exp(-d / tau)
                    return values * gain.to(values.dtype), (
                        f"{branch_label}:env(tau={tau:.2f},floor={floor:.2f})"
                    )

                def _cpu_clone_tensor(values: torch.Tensor) -> torch.Tensor:
                    return values.detach().cpu().clone().view(-1)

                def _build_local_source_anchor(
                    support_times: List[int],
                    support_mask: torch.Tensor,
                    branch_label: str,
                    support_region=None,
                    default_point: Optional[Tuple[float, float]] = None,
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
                    if (
                        compact_history_t is None
                        or compact_positions_t is None
                        or not compact_time_to_local
                        or field_peak_xy_t is None
                        or field_peak_xy_t.numel() == 0
                    ):
                        return None, None, f"{branch_label}:local_anchor=off"
                    local_ids = [
                        int(compact_time_to_local[int(t)])
                        for t in (support_times or [])
                        if int(t) in compact_time_to_local
                    ]
                    if not local_ids:
                        return None, None, f"{branch_label}:local_anchor=no_support"

                    local_idx_t = torch.as_tensor(
                        local_ids,
                        dtype=torch.long,
                        device=goal_rcn.w_in.device,
                    )
                    support_positions = compact_positions_t.index_select(0, local_idx_t)
                    source_d_sq = self._support_region_distance_sq_torch(
                        support_positions[:, 0],
                        support_positions[:, 1],
                        support_region,
                        default_point=default_point,
                    )
                    best_source_idx = int(torch.argmin(source_d_sq).item())
                    best_local_idx = int(local_ids[best_source_idx])
                    anchor_mix_count = 1
                    support_frame = self._support_region_frame(support_region)
                    if support_frame is not None:
                        center_x, center_y = support_frame["center"]
                        center_dx = support_positions[:, 0] - float(center_x)
                        center_dy = support_positions[:, 1] - float(center_y)
                        center_d_sq = center_dx * center_dx + center_dy * center_dy
                        center_rank = source_d_sq + center_d_sq
                        best_source_idx = int(torch.argmin(center_rank).item())
                        best_local_idx = int(local_ids[best_source_idx])
                    anchor_state = compact_history_t[best_local_idx]
                    anchor_peak = torch.amax(anchor_state).clamp_min(1e-8)
                    anchor_state = anchor_state / anchor_peak
                    if local_anchor_scale_weights_t is not None:
                        anchor_state = anchor_state * local_anchor_scale_weights_t

                    gate_d_sq = self._support_region_distance_sq_torch(
                        field_peak_xy_t[:, 0],
                        field_peak_xy_t[:, 1],
                        support_region,
                        default_point=default_point,
                    )
                    peak_gate = torch.exp(
                        -gate_d_sq / (2.0 * float(local_anchor_sigma) ** 2)
                    )
                    mask_t = torch.as_tensor(
                        support_mask,
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    anchor_values = torch.where(
                        mask_t,
                        anchor_state * peak_gate.to(anchor_state.dtype),
                        torch.zeros_like(anchor_state),
                    )
                    anchor_max = float(torch.max(anchor_values).item())
                    if anchor_max <= 1e-8:
                        return None, None, (
                            f"{branch_label}:local_anchor=empty("
                            f"sigma={float(local_anchor_sigma):.2f})"
                        )
                    anchor_values = anchor_values / anchor_max
                    return (
                        anchor_values,
                        gate_d_sq.to(dtype=torch.float32),
                        (
                            f"{branch_label}:local_anchor("
                            f"sigma={float(local_anchor_sigma):.2f},"
                            f"query_sigma={float(local_anchor_query_sigma):.2f},"
                            f"time={int(compact_time_indices[best_local_idx]) if compact_time_to_local else -1},"
                            f"mix={int(anchor_mix_count)})"
                        ),
                    )

                def _build_contact_seed_from_time_candidates(
                    candidate_times: List[int],
                    support_mask: torch.Tensor,
                    branch_label: str,
                    support_region=None,
                    default_point: Optional[Tuple[float, float]] = None,
                ) -> Tuple[Optional[torch.Tensor], int, int, str]:
                    if (
                        compact_history_t is None
                        or compact_positions_t is None
                        or not compact_time_to_local
                    ):
                        return None, -1, -1, f"{branch_label}:contact_seed=off"
                    local_ids = [
                        int(compact_time_to_local[int(t)])
                        for t in (candidate_times or [])
                        if int(t) in compact_time_to_local
                    ]
                    if not local_ids:
                        return None, -1, -1, f"{branch_label}:contact_seed=no_support"

                    local_idx_t = torch.as_tensor(
                        local_ids,
                        dtype=torch.long,
                        device=goal_rcn.w_in.device,
                    )
                    support_positions = compact_positions_t.index_select(0, local_idx_t)
                    mask_t = torch.as_tensor(
                        support_mask,
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    if mask_t.numel() == compact_history_t.shape[1]:
                        candidate_states = torch.clamp(
                            compact_history_t.index_select(0, local_idx_t),
                            min=0.0,
                        )
                        masked_states = torch.where(
                            mask_t.unsqueeze(0),
                            candidate_states,
                            torch.zeros_like(candidate_states),
                        )
                        masked_peaks = torch.amax(masked_states, dim=1)
                        valid_candidate_mask = masked_peaks > 1e-8
                        if torch.any(valid_candidate_mask):
                            local_idx_t = local_idx_t[valid_candidate_mask]
                            support_positions = support_positions[valid_candidate_mask]
                            local_ids = [
                                int(local_ids[idx])
                                for idx in torch.nonzero(
                                    valid_candidate_mask,
                                    as_tuple=False,
                                )
                                .squeeze(1)
                                .tolist()
                            ]
                    source_d_sq = self._support_region_distance_sq_torch(
                        support_positions[:, 0],
                        support_positions[:, 1],
                        support_region,
                        default_point=default_point,
                    )
                    best_source_idx = int(torch.argmin(source_d_sq).item())
                    best_local_idx = int(local_ids[best_source_idx])
                    support_frame = self._support_region_frame(support_region)
                    if support_frame is not None:
                        center_x, center_y = support_frame["center"]
                        center_dx = support_positions[:, 0] - float(center_x)
                        center_dy = support_positions[:, 1] - float(center_y)
                        center_d_sq = center_dx * center_dx + center_dy * center_dy
                        center_rank = source_d_sq + center_d_sq
                        best_source_idx = int(torch.argmin(center_rank).item())
                        best_local_idx = int(local_ids[best_source_idx])

                    contact_state = torch.clamp(
                        compact_history_t[best_local_idx],
                        min=0.0,
                    )
                    if mask_t.numel() == contact_state.numel():
                        contact_state = torch.where(
                            mask_t,
                            contact_state,
                            torch.zeros_like(contact_state),
                        )
                    peak = float(torch.max(contact_state).item())
                    if peak <= 1e-8:
                        contact_time = (
                            int(compact_time_indices[best_local_idx])
                            if compact_time_indices
                            else -1
                        )
                        return None, -1, int(contact_time), (
                            f"{branch_label}:contact_seed=empty("
                            f"time={contact_time})"
                        )
                    contact_state = contact_state / peak
                    contact_time = (
                        int(compact_time_indices[best_local_idx])
                        if compact_time_indices
                        else -1
                    )
                    return (
                        contact_state,
                        int(best_local_idx),
                        int(contact_time),
                        (
                            f"{branch_label}:contact_seed("
                            f"time={contact_time},"
                            f"peak={peak:.3f})"
                        ),
                    )

                def _group_support_times_into_replay_events(
                    support_times: List[int],
                ) -> List[List[int]]:
                    cleaned_times = sorted(
                        {
                            int(t)
                            for t in (support_times or [])
                            if int(t) in compact_time_to_local
                        }
                    )
                    if not cleaned_times:
                        return []
                    merge_gap_steps = int(
                        max(
                            0,
                            getattr(self, "goal_map_replay_event_merge_gap_steps", 3),
                        )
                    )
                    grouped = [[cleaned_times[0]]]
                    for time_idx in cleaned_times[1:]:
                        if int(time_idx) <= int(grouped[-1][-1]) + int(merge_gap_steps):
                            grouped[-1].append(int(time_idx))
                        else:
                            grouped.append([int(time_idx)])
                    return grouped

                def _resolve_replay_event_candidate_times(
                    candidate_times: List[int],
                ) -> List[int]:
                    if not compact_time_to_local or compact_time_indices_np.size <= 0:
                        return []
                    snap_radius = int(
                        max(
                            1,
                            getattr(self, "hmap_sample_stride", 1),
                            getattr(self, "goal_map_replay_event_merge_gap_steps", 3)
                            + 1,
                        )
                    )
                    resolved_times = []
                    for raw_time in candidate_times or []:
                        raw_time = int(raw_time)
                        if raw_time in compact_time_to_local:
                            resolved_times.append(raw_time)
                            continue
                        nearest_idx = int(
                            np.argmin(np.abs(compact_time_indices_np - raw_time))
                        )
                        snapped_time = int(compact_time_indices_np[nearest_idx])
                        if abs(snapped_time - raw_time) <= snap_radius:
                            resolved_times.append(snapped_time)
                    return sorted({int(t) for t in resolved_times})

                def _build_recent_contact_replay_event_seeds(
                    support_times: List[int],
                    trunk_time_sequences: Optional[List[List[int]]],
                    support_mask: torch.Tensor,
                    branch_label: str,
                    support_region=None,
                    default_point: Optional[Tuple[float, float]] = None,
                    max_events: int = 1,
                    event_groups: Optional[List[List[int]]] = None,
                    event_source: str = "history",
                ) -> Tuple[List[Tuple[torch.Tensor, int, int]], str]:
                    resolved_groups: List[List[int]] = []
                    for seq in event_groups or []:
                        cleaned_seq = _resolve_replay_event_candidate_times(seq)
                        if cleaned_seq:
                            resolved_groups.append(cleaned_seq)

                    source_label = str(event_source)
                    if not resolved_groups:
                        source_label = "segmented"
                        for seq in trunk_time_sequences or []:
                            cleaned_seq = [
                                int(t)
                                for t in seq
                                if int(t) in compact_time_to_local
                            ]
                            if cleaned_seq:
                                resolved_groups.append(cleaned_seq)
                    if not resolved_groups:
                        resolved_groups = _group_support_times_into_replay_events(
                            support_times
                        )
                    if not resolved_groups:
                        return [], f"{branch_label}:contact_events=no_support"

                    recent_groups = resolved_groups[-int(max(1, max_events)) :]
                    event_seeds: List[Tuple[torch.Tensor, int, int]] = []
                    event_times: List[int] = []
                    for group_idx, candidate_times in enumerate(recent_groups, start=1):
                        event_seed, local_idx, contact_time, _event_log = (
                            _build_contact_seed_from_time_candidates(
                                candidate_times=candidate_times,
                                support_mask=support_mask,
                                branch_label=f"{branch_label}_evt{group_idx}",
                                support_region=support_region,
                                default_point=default_point,
                            )
                        )
                        if event_seed is None or local_idx < 0:
                            continue
                        event_seeds.append(
                            (
                                event_seed,
                                int(local_idx),
                                int(contact_time),
                            )
                        )
                        event_times.append(int(contact_time))

                    if not event_seeds:
                        return [], (
                            f"{branch_label}:contact_events=empty("
                            f"source={source_label})"
                        )

                    time_summary = ",".join(str(int(t)) for t in event_times)
                    return (
                        event_seeds,
                        (
                            f"{branch_label}:contact_events("
                            f"source={source_label},"
                            f"n={len(event_seeds)},times=[{time_summary}])"
                        ),
                    )

                def _build_exact_contact_source_seed(
                    support_times: List[int],
                    support_mask: torch.Tensor,
                    branch_label: str,
                    support_region=None,
                    default_point: Optional[Tuple[float, float]] = None,
                ) -> Tuple[Optional[torch.Tensor], int, str]:
                    contact_state, best_local_idx, _contact_time, contact_log = (
                        _build_contact_seed_from_time_candidates(
                            candidate_times=support_times,
                            support_mask=support_mask,
                            branch_label=branch_label,
                            support_region=support_region,
                            default_point=default_point,
                        )
                    )
                    return contact_state, best_local_idx, contact_log

                def _attenuate_fill_by_scale(
                    fill_values: torch.Tensor,
                ) -> Tuple[torch.Tensor, str]:
                    scale_boundaries = getattr(self.unified_pcn, "scale_boundaries", None)
                    scale_configs = getattr(self.unified_pcn, "scale_configs", None)
                    if (
                        scale_boundaries is None
                        or scale_configs is None
                        or len(scale_boundaries) < 2
                        or len(scale_configs) == 0
                    ):
                        return fill_values, "fillscale=none"

                    extents = []
                    for cfg in scale_configs:
                        extent = float(
                            max(
                                1e-6,
                                cfg.get("sigma_pc_s", cfg.get("sigma_r", 1.0)),
                            )
                        )
                        extents.append(extent)
                    ref_extent = float(min(extents)) if extents else 1.0
                    attenuated = fill_values.clone()
                    gain_logs = []
                    for scale_idx, extent in enumerate(extents):
                        start = int(scale_boundaries[scale_idx])
                        end = int(scale_boundaries[scale_idx + 1])
                        if end <= start:
                            continue
                        gain = float(min(1.0, max(0.25, ref_extent / max(1e-6, extent))))
                        attenuated[start:end] *= gain
                        gain_logs.append(f"s{scale_idx}:{gain:.2f}")
                    if not gain_logs:
                        return attenuated, "fillscale=none"
                    return attenuated, "fillscale=" + ",".join(gain_logs)

                def _match_goal_name_by_location(
                    goal_x_match: float,
                    goal_z_match: float,
                ) -> Optional[str]:
                    for goal in getattr(self, "goals", []):
                        loc = goal.get("location", [])
                        if (
                            len(loc) >= 2
                            and abs(float(loc[0]) - float(goal_x_match)) <= 1e-6
                            and abs(float(loc[1]) - float(goal_z_match)) <= 1e-6
                        ):
                            return str(goal.get("name"))
                    return None

                def _build_goal_association_seed(
                    goal_x_match: float,
                    goal_z_match: float,
                    target_device,
                ) -> Tuple[Optional[torch.Tensor], str]:
                    goal_name = _match_goal_name_by_location(goal_x_match, goal_z_match)
                    if not goal_name:
                        return None, "assoc_goal=none"
                    if not hasattr(self, "goal_place_cell_associations"):
                        return None, f"assoc_goal={goal_name}:missing"
                    associations = self.goal_place_cell_associations.get(goal_name)
                    if not associations:
                        return None, f"assoc_goal={goal_name}:empty"
                    scale_boundaries = getattr(self.unified_pcn, "scale_boundaries", None)
                    if scale_boundaries is None or len(scale_boundaries) < len(self.scales) + 1:
                        return None, f"assoc_goal={goal_name}:no_boundaries"
                    compact_mask_all = None
                    compact_radius_all = None
                    if bool(getattr(self, "goal_map_use_compact_pc_gate", True)):
                        (
                            _peak_xy_all,
                            _peak_val_all,
                            compact_mask_all,
                            compact_radius_all,
                            _compactness_log,
                        ) = self._get_unified_pc_field_peak_data(target_device)
                    assoc_seed = torch.zeros(
                        self.unified_pcn.num_pc_total,
                        dtype=torch.float32,
                        device=target_device,
                    )
                    assoc_logs = []
                    activations = getattr(self, "goal_place_cell_activations", {}).get(
                        goal_name, []
                    )
                    steps = getattr(self, "goal_association_step", {}).get(goal_name, [])
                    for scale_idx, pc_idx in enumerate(associations):
                        if pc_idx is None:
                            continue
                        step_value = (
                            steps[scale_idx]
                            if scale_idx < len(steps)
                            else None
                        )
                        if (
                            history_start > 0
                            and step_value is not None
                            and int(step_value) < int(history_start)
                        ):
                            continue
                        start = int(scale_boundaries[scale_idx])
                        end = int(scale_boundaries[scale_idx + 1])
                        local_idx = int(pc_idx)
                        if not (0 <= local_idx < (end - start)):
                            continue
                        locality_radius = float(
                            getattr(self, "goal_assoc_locality_radius", 0.0)
                        )
                        if locality_radius > 0.0:
                            peak_stats = self._get_unified_scale_pc_peak_stats(
                                scale_idx=scale_idx,
                                candidate_indices=[local_idx],
                                target_device=target_device,
                            )
                            peak_info = peak_stats.get(local_idx)
                            if peak_info is not None:
                                peak_x, peak_z, _ = peak_info
                                peak_dist = math.hypot(
                                    float(peak_x) - float(goal_x_match),
                                    float(peak_z) - float(goal_z_match),
                                )
                                if peak_dist > locality_radius:
                                    step_txt = ""
                                    if step_value is not None:
                                        step_txt = f"@{int(step_value)}"
                                    assoc_logs.append(
                                        f"s{scale_idx}:pc{local_idx}{step_txt}:skip_nonlocal({peak_dist:.2f}m)"
                                    )
                                    continue
                        global_idx = start + local_idx
                        if compact_mask_all is not None and not bool(
                            compact_mask_all[global_idx].item()
                        ):
                            step_txt = ""
                            if step_value is not None:
                                step_txt = f"@{int(step_value)}"
                            radius_txt = ""
                            if compact_radius_all is not None:
                                try:
                                    radius_txt = (
                                        f":skip_noncompact({float(compact_radius_all[global_idx].item()):.2f}m)"
                                    )
                                except (TypeError, ValueError):
                                    radius_txt = ":skip_noncompact"
                            else:
                                radius_txt = ":skip_noncompact"
                            assoc_logs.append(
                                f"s{scale_idx}:pc{local_idx}{step_txt}{radius_txt}"
                            )
                            continue
                        weight = 1.0
                        if scale_idx < len(activations) and activations[scale_idx] is not None:
                            try:
                                weight = float(max(1e-6, float(activations[scale_idx])))
                            except (TypeError, ValueError):
                                weight = 1.0
                        assoc_seed[global_idx] = max(float(assoc_seed[global_idx].item()), weight)
                        step_txt = ""
                        if step_value is not None:
                            step_txt = f"@{int(step_value)}"
                        assoc_logs.append(f"s{scale_idx}:pc{local_idx}{step_txt}")
                    if float(torch.max(assoc_seed).item()) <= 1e-8:
                        if assoc_logs:
                            return None, f"assoc_goal={goal_name}({','.join(assoc_logs)})"
                        return None, f"assoc_goal={goal_name}:inactive"
                    return assoc_seed, f"assoc_goal={goal_name}({','.join(assoc_logs)})"

                def _build_sparse_source_seed(
                    source_seed: torch.Tensor,
                    support_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, int, str]:
                    seed_vec = torch.as_tensor(
                        source_seed,
                        dtype=torch.float32,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    seed_vec = torch.clamp(seed_vec, min=0.0)
                    if support_mask is not None:
                        support_mask_t = torch.as_tensor(
                            support_mask,
                            dtype=torch.bool,
                            device=goal_rcn.w_in.device,
                        ).view(-1)
                        if support_mask_t.numel() == seed_vec.numel():
                            seed_vec = torch.where(
                                support_mask_t,
                                seed_vec,
                                torch.zeros_like(seed_vec),
                            )
                    peak = float(torch.max(seed_vec).item())
                    sparse_seed = torch.zeros_like(seed_vec)
                    if peak <= 1e-8:
                        return sparse_seed, -1, "sparse_seed=empty"

                    scale_boundaries = getattr(self.unified_pcn, "scale_boundaries", None)
                    selected_idx = -1
                    selected_scale = -1
                    if scale_boundaries is not None and len(scale_boundaries) >= 2:
                        for scale_idx, (start, end) in enumerate(
                            zip(scale_boundaries[:-1], scale_boundaries[1:])
                        ):
                            scale_slice = seed_vec[int(start):int(end)]
                            scale_peak = float(torch.max(scale_slice).item())
                            if scale_peak <= 1e-8:
                                continue
                            selected_scale = int(scale_idx)
                            selected_idx = int(start) + int(torch.argmax(scale_slice).item())
                            break
                    if selected_idx < 0:
                        selected_idx = int(torch.argmax(seed_vec).item())
                    sparse_seed[selected_idx] = 1.0
                    if selected_scale >= 0:
                        return (
                            sparse_seed,
                            int(selected_idx),
                            f"sparse_seed=scale{selected_scale}:idx{selected_idx}",
                        )
                    return sparse_seed, int(selected_idx), f"sparse_seed=idx{selected_idx}"

                def _build_masked_path_replay_field(
                    source_rcn,
                    source_seed: torch.Tensor,
                    replay_transition: torch.Tensor,
                    support_mask: torch.Tensor,
                    source_gain: float = 1.0,
                ) -> Tuple[torch.Tensor, int, int]:
                    support_mask_t = torch.as_tensor(
                        support_mask,
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    if not bool(torch.any(support_mask_t).item()):
                        return torch.zeros_like(source_seed.view(-1)), 0, 0

                    replay_gate = support_mask_t.to(dtype=torch.float32)
                    source_wave = torch.as_tensor(
                        source_seed,
                        dtype=torch.float32,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    source_wave = torch.where(
                        support_mask_t,
                        source_wave,
                        torch.zeros_like(source_wave),
                    )
                    if float(torch.max(torch.abs(source_wave)).item()) <= 1e-8:
                        return torch.zeros_like(source_wave), 0, 0

                    masked_transition = torch.as_tensor(
                        replay_transition,
                        dtype=torch.float32,
                        device=goal_rcn.w_in.device,
                    )
                    gate_2d = replay_gate.unsqueeze(0) * replay_gate.unsqueeze(1)
                    masked_transition = masked_transition * gate_2d
                    if bool(getattr(source_rcn, "replay_row_normalize", True)):
                        row_sum = torch.sum(masked_transition, dim=1, keepdim=True)
                        masked_transition = torch.where(
                            row_sum > 1e-12,
                            masked_transition / row_sum,
                            torch.zeros_like(masked_transition),
                        )

                    replay_amplitude = float(source_gain) * source_rcn._compute_unified_amplitude(
                        replay_gate
                    )
                    weight_update = torch.zeros_like(source_wave)
                    active_steps = 0

                    for time_step in range(int(max(1, getattr(source_rcn, "replay_timesteps", 1)))):
                        current_peak = float(torch.max(torch.abs(source_wave)).item())
                        if current_peak <= 1e-8:
                            break
                        active_steps += 1
                        decay = source_rcn._decay_factor_at_step(time_step).view(-1)
                        norm_val = torch.sqrt(
                            torch.max(
                                torch.sum(source_wave**2),
                                torch.tensor(
                                    1e-12,
                                    dtype=source_wave.dtype,
                                    device=source_wave.device,
                                ),
                            )
                        )
                        normalized_wave = source_wave / norm_val
                        normalized_wave = torch.where(
                            torch.isnan(normalized_wave),
                            torch.zeros_like(normalized_wave),
                            normalized_wave,
                        )
                        normalized_wave = normalized_wave * replay_gate
                        weight_update = weight_update + (
                            decay * normalized_wave * replay_amplitude
                        )
                        source_wave = source_rcn._advance_replay_wave(
                            transition=masked_transition,
                            place_cell_activations=source_wave,
                            replay_gate=replay_gate,
                        )

                    weight_update = torch.where(
                        support_mask_t,
                        torch.clamp(weight_update, min=0.0),
                        torch.zeros_like(weight_update),
                    )
                    claimed_count = int(torch.count_nonzero(weight_update > 1e-8).item())
                    return weight_update, int(active_steps), claimed_count

                def _build_masked_experience_replay_field(
                    source_rcn,
                    replay_event_seeds: List[Tuple[torch.Tensor, int, int]],
                    support_mask: torch.Tensor,
                    source_gain: float = 1.0,
                ) -> Tuple[torch.Tensor, int, int]:
                    if not replay_event_seeds:
                        return torch.zeros_like(seed_activations.view(-1)), 0, 0
                    if recurrent_weights_max is None:
                        return torch.zeros_like(seed_activations.view(-1)), 0, 0
                    support_mask_t = torch.as_tensor(
                        support_mask,
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    if not bool(torch.any(support_mask_t).item()):
                        return torch.zeros_like(seed_activations.view(-1)), 0, 0

                    replay_gate = support_mask_t.to(dtype=torch.float32)

                    # Paper-style replay (Xalabi NN23, Alg. 1):
                    # 1) accumulate decayed, max-normalized replay states
                    # 2) advance imagined activity with v <- tanh(relu(v + W_max @ v))
                    recurrent_masked = torch.clamp(
                        torch.as_tensor(
                            recurrent_weights_max,
                            dtype=torch.float32,
                            device=goal_rcn.w_in.device,
                        ),
                        min=0.0,
                    )
                    gate_2d = replay_gate.unsqueeze(0) * replay_gate.unsqueeze(1)
                    recurrent_masked = recurrent_masked * gate_2d

                    paper_replay_tau = float(
                        max(
                            1e-6,
                            getattr(self, "goal_map_paper_replay_tau", 8.0),
                        )
                    )
                    weight_update = torch.zeros_like(replay_gate)
                    active_steps = 0

                    for source_seed, _local_idx, _contact_time in replay_event_seeds:
                        source_wave = torch.as_tensor(
                            source_seed,
                            dtype=torch.float32,
                            device=goal_rcn.w_in.device,
                        ).view(-1)
                        source_wave = torch.where(
                            support_mask_t,
                            source_wave,
                            torch.zeros_like(source_wave),
                        )
                        if float(torch.max(torch.abs(source_wave)).item()) <= 1e-8:
                            continue

                        event_update = torch.zeros_like(source_wave)
                        event_steps = 0

                        for time_step in range(
                            int(max(1, getattr(source_rcn, "replay_timesteps", 1)))
                        ):
                            current_peak = torch.max(torch.abs(source_wave))
                            if float(current_peak.item()) <= 1e-8:
                                break
                            event_steps += 1

                            decay = math.exp(-float(time_step) / paper_replay_tau)
                            normalized_wave = source_wave / torch.clamp(
                                current_peak, min=1e-12
                            )
                            normalized_wave = torch.where(
                                torch.isnan(normalized_wave),
                                torch.zeros_like(normalized_wave),
                                normalized_wave,
                            )
                            normalized_wave = normalized_wave * replay_gate
                            event_update = event_update + (
                                float(decay) * normalized_wave
                            )

                            dot_val = torch.matmul(recurrent_masked, source_wave)
                            updated_source_wave = torch.relu(dot_val + source_wave)
                            source_wave = torch.tanh(updated_source_wave) * replay_gate

                        event_update = torch.where(
                            support_mask_t,
                            torch.clamp(event_update, min=0.0),
                            torch.zeros_like(event_update),
                        )
                        event_peak = float(torch.max(event_update).item())
                        if event_peak <= 1e-8:
                            continue

                        active_steps = max(active_steps, int(event_steps))
                        weight_update = weight_update + (
                            float(source_gain) * (event_update / event_peak)
                        )

                    weight_update = torch.where(
                        support_mask_t,
                        torch.clamp(weight_update, min=0.0),
                        torch.zeros_like(weight_update),
                    )
                    claimed_count = int(torch.count_nonzero(weight_update > 1e-8).item())
                    return weight_update, int(active_steps), claimed_count

                backbone_topk = int(
                    max(1, getattr(self, "goal_map_backbone_support_topk", 8))
                )
                backbone_threshold_fraction = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_backbone_support_threshold_fraction",
                                0.45,
                            ),
                        ),
                    )
                )
                backbone_step_decay = float(
                    min(
                        1.0,
                        max(
                            1e-3,
                            getattr(self, "goal_map_backbone_step_decay", 0.93),
                        ),
                    )
                )

                trunk_smooth_steps = int(
                    max(0, getattr(self, "goal_map_backbone_smooth_steps", 1))
                )
                trunk_smooth_decay = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(self, "goal_map_backbone_smooth_decay", 0.88),
                        ),
                    )
                )
                backbone_suffix_steps = int(
                    max(1, getattr(self, "goal_map_backbone_suffix_steps", 1))
                )
                backbone_spread_seed_gain = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(self, "goal_map_backbone_spread_seed_gain", 0.35),
                        ),
                    )
                )
                goal_backbone_spread_seed_gain = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_goal_backbone_spread_seed_gain",
                                backbone_spread_seed_gain * 0.7,
                            ),
                        ),
                    )
                )
                checkpoint_backbone_spread_seed_gain = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_checkpoint_backbone_spread_seed_gain",
                                max(backbone_spread_seed_gain, 0.55),
                            ),
                        ),
                    )
                )
                checkpoint_seed_completion_gain = float(
                    min(
                        2.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_checkpoint_seed_completion_gain",
                                1.15,
                            ),
                        ),
                    )
                )
                checkpoint_path_completion_gain = float(
                    min(
                        2.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_checkpoint_path_completion_gain",
                                0.35,
                            ),
                        ),
                    )
                )
                checkpoint_room_fill_gain = float(
                    min(
                        2.0,
                        max(
                            0.0,
                            getattr(
                                self,
                                "goal_map_checkpoint_room_fill_gain",
                                max(segment_room_fill_gain, 1.0),
                            ),
                        ),
                    )
                )

                def _build_consensus_trunk_time_sequence(
                    trunk_time_sequences: Optional[List[List[int]]],
                    support_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[Optional[List[int]], str]:
                    valid_sequences = [
                        [int(t) for t in seq if int(t) >= 0]
                        for seq in (trunk_time_sequences or [])
                        if len(seq) >= 2
                    ]
                    if len(valid_sequences) < 2:
                        return None, "consensus=insufficient"

                    source_first_sequences = [
                        list(reversed(seq)) for seq in valid_sequences
                    ]
                    lengths = [len(seq) for seq in source_first_sequences]
                    if not lengths:
                        return None, "consensus=empty"

                    max_bins = int(
                        max(
                            3,
                            getattr(self, "goal_map_consensus_backbone_max_bins", 11),
                        )
                    )
                    target_bins = int(round(float(np.median(lengths))))
                    target_bins = max(3, min(max(lengths), target_bins))
                    target_bins = min(target_bins, max_bins)

                    support_mask_t = None
                    if support_mask is not None:
                        support_mask_t = torch.as_tensor(
                            support_mask,
                            dtype=torch.bool,
                            device=goal_rcn.w_in.device,
                        ).view(-1)
                    search_radius = int(
                        max(
                            0,
                            getattr(
                                self,
                                "goal_map_consensus_backbone_search_radius_steps",
                                4,
                            ),
                        )
                    )
                    support_cache: Dict[int, Tuple[int, List[int]]] = {}

                    def _support_overlap_for_time(t: int) -> Tuple[int, List[int]]:
                        cached = support_cache.get(int(t))
                        if cached is not None:
                            return cached
                        try:
                            vec_t = torch.cat(
                                [arr[int(t)] for arr in segmented_data["scale_arrays"]],
                                dim=0,
                            ).to(goal_rcn.w_in.device, dtype=torch.float32)
                        except (IndexError, RuntimeError, TypeError, ValueError):
                            cached = (0, [])
                            support_cache[int(t)] = cached
                            return cached

                        active_idx = self._relay_support_indices(
                            vec_t,
                            topk=backbone_topk,
                            threshold_fraction=backbone_threshold_fraction,
                        )
                        if not active_idx:
                            active_idx = [int(torch.argmax(vec_t).item())]
                        active_idx = [
                            int(idx)
                            for idx in active_idx
                            if 0 <= int(idx) < int(vec_t.numel())
                        ]
                        if support_mask_t is None:
                            cached = (len(active_idx), active_idx)
                        else:
                            kept_idx = [
                                int(idx)
                                for idx in active_idx
                                if bool(support_mask_t[int(idx)].item())
                            ]
                            cached = (len(kept_idx), kept_idx)
                        support_cache[int(t)] = cached
                        return cached

                    consensus_source_first: List[int] = []
                    consensus_positions: List[np.ndarray] = []
                    masked_valid_bins = 0
                    snapped_candidates = 0
                    skipped_bins = 0
                    progress_grid = np.linspace(0.0, 1.0, target_bins)
                    for progress in progress_grid:
                        candidate_times: List[int] = []
                        candidate_positions: List[np.ndarray] = []
                        candidate_overlap: List[int] = []
                        candidate_snap_steps: List[int] = []
                        for seq in source_first_sequences:
                            base_pos_idx = int(round(progress * float(len(seq) - 1)))
                            base_pos_idx = max(0, min(len(seq) - 1, base_pos_idx))
                            search_offsets = [0]
                            for delta in range(1, search_radius + 1):
                                search_offsets.extend([delta, -delta])

                            chosen_t = None
                            chosen_pos_idx = None
                            chosen_overlap = 0
                            for delta in search_offsets:
                                pos_idx = base_pos_idx + delta
                                if not (0 <= pos_idx < len(seq)):
                                    continue
                                t = int(seq[pos_idx])
                                if not (0 <= t < self.hmap_loc.shape[0]):
                                    continue
                                overlap_count, _ = _support_overlap_for_time(t)
                                if support_mask_t is not None and overlap_count <= 0:
                                    continue
                                chosen_t = t
                                chosen_pos_idx = pos_idx
                                chosen_overlap = overlap_count
                                break

                            if chosen_t is None:
                                if support_mask_t is None:
                                    chosen_t = int(seq[base_pos_idx])
                                    chosen_pos_idx = base_pos_idx
                                else:
                                    continue

                            if int(chosen_pos_idx) != int(base_pos_idx):
                                snapped_candidates += 1
                            candidate_times.append(int(chosen_t))
                            candidate_overlap.append(int(chosen_overlap))
                            candidate_snap_steps.append(
                                abs(int(chosen_pos_idx) - int(base_pos_idx))
                            )
                            candidate_positions.append(
                                np.asarray(self.hmap_loc[int(chosen_t), :2], dtype=np.float32)
                            )
                        if not candidate_times:
                            skipped_bins += 1
                            continue

                        masked_valid_bins += 1
                        candidate_xy = np.stack(candidate_positions, axis=0)
                        center_xy = np.median(candidate_xy, axis=0)
                        score = np.sum((candidate_xy - center_xy) ** 2, axis=1)
                        if consensus_positions:
                            prev_xy = consensus_positions[-1]
                            continuity = np.sum((candidate_xy - prev_xy) ** 2, axis=1)
                            score = score + 0.15 * continuity
                        if candidate_snap_steps:
                            score = score + 0.05 * np.square(
                                np.asarray(candidate_snap_steps, dtype=np.float32)
                            )
                        if candidate_overlap:
                            score = score - 0.02 * np.asarray(
                                candidate_overlap, dtype=np.float32
                            )
                        best_idx = int(np.argmin(score))
                        best_time = int(candidate_times[best_idx])
                        if consensus_source_first and consensus_source_first[-1] == best_time:
                            continue
                        consensus_source_first.append(best_time)
                        consensus_positions.append(candidate_xy[best_idx])

                    if len(consensus_source_first) < 2:
                        return None, "consensus=degenerate"

                    return (
                        list(reversed(consensus_source_first)),
                        (
                            "consensus=median_progress("
                            f"trunks={len(valid_sequences)},"
                            f"bins={target_bins},"
                            f"kept={len(consensus_source_first)},"
                            f"valid_bins={masked_valid_bins},"
                            f"snapped={snapped_candidates},"
                            f"skipped={skipped_bins})"
                        ),
                    )

                def _build_backbone_field_from_trunks(
                    trunk_time_sequences: Optional[List[List[int]]],
                    support_mask: torch.Tensor,
                    seed_core: torch.Tensor,
                    source_rcn=None,
                    smoothing_transition: Optional[torch.Tensor] = None,
                    consensus_mode: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, str]:
                    support_mask_t = torch.as_tensor(
                        support_mask,
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    ).view(-1)
                    backbone = torch.zeros_like(seed_core)
                    spread_seed = torch.zeros_like(seed_core)
                    trunk_count = 0
                    support_samples = 0
                    backbone_steps = 0
                    consensus_log = "consensus=off"
                    if not trunk_time_sequences:
                        return (
                            torch.where(support_mask_t, seed_core, torch.zeros_like(seed_core)),
                            torch.zeros_like(seed_core),
                            0,
                            0,
                            0,
                            "consensus=none",
                        )

                    sequences_to_use = trunk_time_sequences
                    if consensus_mode:
                        consensus_sequence, consensus_log = _build_consensus_trunk_time_sequence(
                            trunk_time_sequences,
                            support_mask=support_mask_t,
                        )
                        if consensus_sequence is not None:
                            sequences_to_use = [consensus_sequence]
                        else:
                            consensus_log = f"{consensus_log}|fallback=raw"

                    for trunk_time_indices in sequences_to_use:
                        if not trunk_time_indices:
                            continue
                        num_steps = len(trunk_time_indices)
                        trunk_entries = []
                        scaffold_peak = 0.0
                        suffix_peak = 0.0
                        for pos_idx, t in enumerate(trunk_time_indices):
                            try:
                                vec_t = torch.cat(
                                    [arr[int(t)] for arr in segmented_data["scale_arrays"]],
                                    dim=0,
                                ).to(goal_rcn.w_in.device, dtype=torch.float32)
                            except (IndexError, RuntimeError, TypeError, ValueError):
                                continue
                            active_idx = self._relay_support_indices(
                                vec_t,
                                topk=backbone_topk,
                                threshold_fraction=backbone_threshold_fraction,
                            )
                            if not active_idx:
                                active_idx = [int(torch.argmax(vec_t).item())]
                            active_idx = [
                                int(idx)
                                for idx in active_idx
                                if 0 <= int(idx) < int(vec_t.numel())
                            ]
                            if not active_idx:
                                continue
                            support_samples += len(active_idx)
                            step_from_source = max(0, num_steps - 1 - int(pos_idx))
                            step_weight = float(backbone_step_decay ** step_from_source)
                            active_tensor = torch.as_tensor(
                                active_idx,
                                dtype=torch.long,
                                device=goal_rcn.w_in.device,
                            )
                            selected = torch.clamp(vec_t[active_tensor], min=0.0)
                            if selected.numel() <= 0:
                                continue
                            selected_peak = float(torch.max(selected).item())
                            scaffold_peak = max(scaffold_peak, selected_peak)
                            if step_from_source < backbone_suffix_steps:
                                suffix_peak = max(suffix_peak, selected_peak)
                            trunk_entries.append(
                                (
                                    active_tensor,
                                    selected,
                                    step_weight,
                                    step_from_source,
                                )
                            )

                        if not trunk_entries:
                            continue

                        trunk_count += 1
                        backbone_steps = max(backbone_steps, int(num_steps))
                        trunk_field = torch.zeros_like(backbone)
                        trunk_scaffold = torch.zeros_like(backbone)
                        norm_peak = max(1e-8, scaffold_peak)
                        suffix_norm_peak = max(
                            1e-8,
                            suffix_peak if suffix_peak > 0.0 else scaffold_peak,
                        )
                        for active_tensor, selected, step_weight, step_from_source in trunk_entries:
                            contrib = torch.zeros_like(backbone)
                            contrib[active_tensor] = step_weight * (selected / norm_peak)
                            trunk_scaffold = torch.maximum(trunk_scaffold, contrib)
                            if step_from_source < backbone_suffix_steps:
                                sharp_contrib = torch.zeros_like(backbone)
                                sharp_contrib[active_tensor] = (
                                    step_weight * (selected / suffix_norm_peak)
                                )
                                trunk_field = torch.maximum(trunk_field, sharp_contrib)

                        trunk_field = torch.where(
                            support_mask_t,
                            trunk_field,
                            torch.zeros_like(trunk_field),
                        )
                        if (
                            source_rcn is not None
                            and smoothing_transition is not None
                            and trunk_smooth_steps > 0
                            and float(torch.max(trunk_scaffold).item()) > 1e-8
                        ):
                            smoothed_field = source_rcn._competitive_neighbor_from_source(
                                transition=smoothing_transition,
                                source_wave=trunk_scaffold,
                                num_steps=trunk_smooth_steps,
                                step_decay=trunk_smooth_decay,
                                base_values=trunk_scaffold,
                                frontier_only=False,
                            )
                            trunk_scaffold = torch.where(
                                support_mask_t,
                                torch.maximum(trunk_scaffold, smoothed_field),
                                torch.zeros_like(trunk_scaffold),
                            )

                        trunk_field_peak = float(torch.max(trunk_field).item())
                        if trunk_field_peak > 1e-8:
                            trunk_field = trunk_field / trunk_field_peak
                        trunk_scaffold_peak = float(torch.max(trunk_scaffold).item())
                        if trunk_scaffold_peak > 1e-8:
                            trunk_scaffold = trunk_scaffold / trunk_scaffold_peak
                        backbone = torch.maximum(backbone, trunk_field)
                        spread_seed = torch.maximum(spread_seed, trunk_scaffold)

                    backbone = torch.where(
                        support_mask_t,
                        torch.maximum(backbone, seed_core),
                        torch.zeros_like(backbone),
                    )
                    spread_seed = torch.where(
                        support_mask_t,
                        spread_seed,
                        torch.zeros_like(spread_seed),
                    )
                    return (
                        backbone,
                        spread_seed,
                        int(trunk_count),
                        int(support_samples),
                        int(backbone_steps),
                        consensus_log,
                    )

                goal_path_budget = float(
                    max(0.0, getattr(self, "goal_map_goal_path_budget", 1.00))
                )
                goal_fill_budget = float(
                    max(0.0, getattr(self, "goal_map_goal_fill_budget", 0.54))
                )
                checkpoint_path_budget = float(
                    max(0.0, getattr(self, "goal_map_checkpoint_path_budget", 0.98))
                )
                checkpoint_fill_budget = float(
                    max(0.0, getattr(self, "goal_map_checkpoint_fill_budget", 0.68))
                )
                goal_total_budget = max(goal_path_budget, goal_fill_budget)
                checkpoint_total_budget = max(
                    checkpoint_path_budget, checkpoint_fill_budget
                )

                goal_sparse_idx = -1
                goal_path_values = None
                goal_replay_event_count = 0
                local_anchor_name_bank = []
                local_anchor_weight_bank = []
                local_anchor_gate_d2_bank = []
                if trajectory_state_goal_map:
                    goal_path_raw, goal_neighbor_raw, goal_ts_meta = (
                        self._build_trajectory_state_segment_fields(
                            scale_arrays=segmented_data["scale_arrays"],
                            compact_time_indices=segmented_data.get("time_indices", []),
                            trunk_time_sequences=segmented_data.get(
                                "goal_trunk_time_sequences", []
                            ),
                            support_mask=goal_support_mask,
                            target_device=goal_rcn.w_in.device,
                            source_kind="goal",
                            room_fill_gain=segment_room_fill_gain,
                            room_decay_tau=goal_room_decay_tau,
                            core_gain=goal_core_gain,
                            support_time_indices=segmented_data.get(
                                "goal_support_times", []
                            ),
                        )
                    )
                    goal_replay_steps = int(goal_ts_meta.get("max_hops", 0)) + 1
                    goal_claimed_raw = int(
                        torch.count_nonzero(goal_path_raw > 1e-8).item()
                    )
                    goal_path_compact_log = "goal_path:trajectory_state"
                    goal_fill_compact_log = "goal_fill:trajectory_state"
                    goal_fill_env_log = (
                        f"goal_fill:distance_decay(tau={goal_room_decay_tau:.2f})"
                    )
                    goal_fill_scale_log = str(goal_ts_meta.get("scale_log", "lambda=na"))
                    goal_seed_log = "goal_seed=contact_core"
                    goal_path_mode = str(
                        goal_ts_meta.get("path_mode", "trajectory_state_backbone")
                    )
                    (
                        goal_local_anchor,
                        goal_local_anchor_gate_d2,
                        goal_local_anchor_log,
                    ) = _build_local_source_anchor(
                        support_times=segmented_data.get("goal_support_times", []),
                        support_mask=goal_support_mask,
                        branch_label="goal",
                        support_region=None,
                        default_point=(float(goal_x), float(goal_z)),
                    )
                    if (
                        goal_local_anchor is not None
                        and goal_local_anchor_gate_d2 is not None
                    ):
                        local_anchor_name_bank.append("goal")
                        local_anchor_weight_bank.append(goal_local_anchor)
                        local_anchor_gate_d2_bank.append(goal_local_anchor_gate_d2)
                elif room_masked_experience_goal_map:
                    goal_replay_events, goal_exact_seed_log = (
                        _build_recent_contact_replay_event_seeds(
                            support_times=segmented_data.get("goal_support_times", []),
                            trunk_time_sequences=segmented_data.get(
                                "goal_trunk_time_sequences", []
                            ),
                            support_mask=goal_support_mask,
                            branch_label="goal",
                            support_region=None,
                            default_point=(float(goal_x), float(goal_z)),
                            max_events=goal_replay_event_limit,
                            event_groups=self._get_recent_goal_contact_event_groups(
                                goal_name=goal["name"],
                                max_events=goal_replay_event_limit,
                            ),
                            event_source="goal_contact_history",
                        )
                    )
                    if not goal_replay_events:
                        goal_exact_seed = seed_activations.to(
                            goal_rcn.w_in.device, dtype=torch.float32
                        )
                        goal_exact_seed = torch.where(
                            goal_support_mask,
                            goal_exact_seed,
                            torch.zeros_like(goal_exact_seed),
                        )
                        goal_replay_events = [(goal_exact_seed, -1, -1)]
                        goal_exact_seed_log = (
                            f"{goal_exact_seed_log}|fallback=current_seed"
                        )
                    goal_replay_event_count = int(len(goal_replay_events))
                    goal_sparse_idx = int(torch.argmax(goal_replay_events[-1][0]).item())
                    goal_path_raw, goal_replay_steps, goal_claimed_raw = (
                        _build_masked_experience_replay_field(
                            goal_segment_rcn,
                            goal_replay_events,
                            goal_support_mask,
                            source_gain=goal_core_gain,
                        )
                    )
                    goal_path_compact_log = "goal_path:masked_experience_replay"
                    goal_fill_compact_log = "goal_fill:disabled"
                    goal_fill_env_log = "goal_fill:disabled"
                    goal_fill_scale_log = "fillscale=none"
                    goal_seed_log = (
                        f"{goal_exact_seed_log},tau={paper_replay_tau:.2f},"
                        f"steps={int(goal_segment_rcn.replay_timesteps)}"
                    )
                    goal_path_mode = "room_masked_experience_replay"
                    goal_neighbor_raw = torch.zeros_like(goal_path_raw)
                    (
                        goal_local_anchor,
                        goal_local_anchor_gate_d2,
                        goal_local_anchor_log,
                    ) = _build_local_source_anchor(
                        support_times=segmented_data.get("goal_support_times", []),
                        support_mask=goal_support_mask,
                        branch_label="goal",
                        support_region=None,
                        default_point=(float(goal_x), float(goal_z)),
                    )
                    if (
                        goal_local_anchor is not None
                        and goal_local_anchor_gate_d2 is not None
                    ):
                        local_anchor_name_bank.append("goal")
                        local_anchor_weight_bank.append(goal_local_anchor)
                        local_anchor_gate_d2_bank.append(goal_local_anchor_gate_d2)
                else:
                    goal_assoc_seed, goal_assoc_log = _build_goal_association_seed(
                        float(goal_x),
                        float(goal_z),
                        goal_rcn.w_in.device,
                    )
                    goal_source_seed = (
                        goal_assoc_seed
                        if goal_assoc_seed is not None
                        else seed_activations.to(goal_rcn.w_in.device, dtype=torch.float32)
                    )
                    goal_sparse_seed, goal_sparse_idx, goal_sparse_log = _build_sparse_source_seed(
                        goal_source_seed,
                        support_mask=goal_support_mask,
                    )
                    goal_path_raw, goal_replay_steps, goal_claimed_raw = (
                        _build_masked_path_replay_field(
                            goal_segment_rcn,
                            goal_sparse_seed,
                            goal_path_transition,
                            goal_support_mask,
                            source_gain=goal_core_gain,
                        )
                    )
                    goal_path_raw, goal_path_compact_log = _apply_compact_pc_gate(
                        goal_path_raw,
                        branch_label="goal_path",
                    )
                    goal_path_values, goal_path_raw_peak = _apply_segment_budget(
                        goal_path_raw,
                        goal_support_mask,
                        goal_path_budget,
                    )
                    goal_neighbor_raw = torch.zeros_like(goal_path_raw)
                    goal_fill_scale_log = "fillscale=none"
                    goal_fill_compact_log = "goal_fill:compact=off"
                    goal_fill_env_log = "goal_fill:env=off"
                    goal_local_anchor_log = "goal:local_anchor=off"
                    if float(torch.max(goal_path_values).item()) > 1e-8:
                        goal_neighbor_raw = goal_segment_rcn._competitive_neighbor_from_source(
                            transition=goal_neighbor_transition,
                            source_wave=goal_path_values,
                            num_steps=goal_neighbor_steps,
                            step_decay=goal_neighbor_decay,
                            base_values=goal_path_values,
                            frontier_only=bool(
                                getattr(goal_segment_rcn, "goal_map_neighbor_frontier_only", True)
                            ),
                        )
                        goal_neighbor_raw = torch.where(
                            goal_support_mask,
                            goal_neighbor_raw,
                            torch.zeros_like(goal_neighbor_raw),
                        )
                        goal_neighbor_raw, goal_fill_compact_log = _apply_compact_pc_gate(
                            goal_neighbor_raw,
                            branch_label="goal_fill",
                        )
                        goal_neighbor_raw, goal_fill_env_log = _apply_fill_source_envelope(
                            goal_neighbor_raw,
                            source_xy=(float(goal_x), float(goal_z)),
                            branch_label="goal_fill",
                        )
                        goal_neighbor_raw, goal_fill_scale_log = _attenuate_fill_by_scale(
                            goal_neighbor_raw
                        )
                    goal_seed_log = f"{goal_assoc_log} | {goal_sparse_log}"
                    goal_path_mode = "path_replay"
                if goal_path_values is None:
                    goal_path_values, goal_path_raw_peak = _apply_segment_budget(
                        goal_path_raw,
                        goal_support_mask,
                        goal_path_budget,
                    )
                goal_neighbor_values, goal_fill_raw_peak = _apply_segment_budget(
                    goal_neighbor_raw,
                    goal_support_mask,
                    goal_fill_budget,
                )
                composed_path = goal_path_values
                composed_neighbor = goal_neighbor_values
                composed_values = torch.maximum(composed_path, composed_neighbor)
                segment_debug = {
                    "source_names": ["goal"],
                    "path_maps_pre_smooth": {
                        "goal": _cpu_clone_tensor(composed_path),
                    },
                    "fill_maps_pre_smooth": {
                        "goal": _cpu_clone_tensor(composed_neighbor),
                    },
                    "total_maps_pre_smooth": {
                        "goal": _cpu_clone_tensor(composed_values),
                    },
                    "support_masks": {
                        "goal": _cpu_clone_tensor(goal_support_mask.to(dtype=torch.float32)),
                    },
                    "metadata": {
                        "goal": {
                            "support_count": int(goal_support_count),
                            "replay_event_count": int(goal_replay_event_count),
                            "replay_steps": int(goal_replay_steps),
                            "owned_component": int(goal_component_idx),
                            "seed_index": int(goal_sparse_idx),
                            "seed_log": str(goal_seed_log),
                            "path_budget": float(goal_path_budget),
                            "fill_budget": float(goal_fill_budget),
                            "total_budget": float(goal_total_budget),
                            "path_mode": str(goal_path_mode),
                            "path_raw_peak": float(goal_path_raw_peak),
                            "fill_raw_peak": float(goal_fill_raw_peak),
                            "compactness_log": str(goal_path_compact_log),
                            "fill_compactness_log": str(goal_fill_compact_log),
                            "fill_env_log": str(goal_fill_env_log),
                            "claimed_count": int(goal_claimed_raw),
                            "spread_mode": (
                                "trajectory_state_backbone_plus_room_fill"
                                if trajectory_state_goal_map
                                else (
                                    "room_masked_experience_replay"
                                    if room_masked_experience_goal_map
                                    else "path_replay_plus_neighbor"
                                )
                            ),
                            "fill_scale_log": str(goal_fill_scale_log),
                            "local_anchor_log": str(goal_local_anchor_log),
                        }
                    },
                }
                segment_total_fields = [composed_values.clone()]
                goal_claimed = int(torch.count_nonzero(composed_values > 1e-8).item())
                source_logs = [
                    (
                        "goal_segment("
                        f"spread={'trajectory_state_backbone_plus_room_fill' if trajectory_state_goal_map else ('room_masked_experience_replay' if room_masked_experience_goal_map else 'path_replay_plus_neighbor')},support={goal_support_count},"
                        f"events={goal_replay_event_count},"
                        f"steps={goal_replay_steps},"
                        f"owned={goal_component_idx},"
                        f"seed={goal_sparse_idx},"
                        f"budget={goal_path_budget:.2f}/{goal_fill_budget:.2f},"
                        f"pathmode={goal_path_mode},"
                        f"{goal_path_compact_log},"
                        f"{goal_fill_compact_log},"
                        f"{goal_fill_env_log},"
                        f"rawpeak={goal_path_raw_peak:.3f}/{goal_fill_raw_peak:.3f},"
                        f"claimed={goal_claimed},{goal_fill_scale_log},"
                        f"{goal_local_anchor_log})"
                    )
                ]
                if _dispute_log_parts:
                    source_logs.append(
                        f"room_disputes({len(_dispute_log_parts)}): "
                        + "; ".join(_dispute_log_parts[:8])
                    )
                checkpoint_seed_map = {
                    checkpoint_idx: checkpoint_seed
                    for checkpoint_idx, checkpoint_seed in checkpoint_seed_bank
                }
                route_bonus_weights = {}
                route_total_weights = {}
                route_selected_set = (
                    set(route_selected_data.get("selected_set", set()))
                    if route_selected_data is not None
                    else set()
                )
                route_base_weights = (
                    dict(route_selected_data.get("base_weights", {}))
                    if route_selected_data is not None
                    else {}
                )
                route_parent_map = (
                    dict(route_selected_data.get("parent_map", {}))
                    if route_selected_data is not None
                    else {}
                )
                for checkpoint_idx in active_checkpoint_order:
                    checkpoint_counts = segmented_data["checkpoint_counts"].get(checkpoint_idx)
                    checkpoint_seed = checkpoint_seed_map.get(checkpoint_idx)
                    if checkpoint_counts is None or checkpoint_seed is None:
                        continue
                    if (
                        checkpoint_mode in {"route_selected_tree", "directed_route_memory_tree", "multi_route_memory_graph"}
                        and route_selected_data is not None
                        and int(checkpoint_idx) not in route_selected_set
                    ):
                        source_logs.append(
                            f"cp{checkpoint_idx}_segment(offroute,support=0)"
                        )
                        continue
                    (
                        checkpoint_rcn,
                        checkpoint_path_transition,
                        checkpoint_neighbor_transition,
                        checkpoint_support_mask,
                        checkpoint_support_count,
                        checkpoint_neighbor_steps,
                        checkpoint_neighbor_decay,
                        checkpoint_neighbor_seed_scale,
                    ) = _prepare_segment_source(
                        checkpoint_seed,
                        full_source_counts,
                        "checkpoint",
                        owned_room_mask=_build_owned_room_mask(
                            _get_resolved_checkpoint_owned_components(int(checkpoint_idx)),
                            doorway_ids=checkpoint_seed_support_id_bank.get(
                                int(checkpoint_idx), []
                            ),
                            extra_ids=self._expand_node_ids_with_kernel(
                                checkpoint_seed_support_id_bank.get(int(checkpoint_idx), []),
                                local_kernel,
                                steps=checkpoint_bridge_steps,
                            ),
                        ),
                        doorway_blocker_node_ids=checkpoint_doorway_blockers,
                        own_doorway_node_ids=checkpoint_seed_support_id_bank.get(
                            int(checkpoint_idx), []
                        ),
                        bridge_node_ids=self._expand_node_ids_with_kernel(
                            checkpoint_seed_support_id_bank.get(int(checkpoint_idx), []),
                            local_kernel,
                            steps=checkpoint_bridge_steps,
                        ),
                        doorway_support_regions=checkpoint_doorway_regions,
                    )
                    backbone_support_field = composed_path
                    checkpoint_region = checkpoint_support_region_bank.get(
                        int(checkpoint_idx)
                    )
                    arrival_ids = self._expand_node_ids_with_kernel(
                        checkpoint_seed_support_id_bank.get(int(checkpoint_idx), []),
                        local_kernel,
                        steps=arrival_buffer_steps,
                    )
                    arrival_mask = torch.zeros(
                        checkpoint_seed.numel(),
                        dtype=torch.bool,
                        device=goal_rcn.w_in.device,
                    )
                    if arrival_ids:
                        arrival_mask[
                            torch.as_tensor(arrival_ids, device=goal_rcn.w_in.device)
                        ] = True
                    else:
                        arrival_mask = checkpoint_seed.to(
                            goal_rcn.w_in.device, dtype=torch.float32
                        ) > 1e-6
                    arrival_support = float(
                        torch.max(
                            backbone_support_field[arrival_mask]
                        ).item()
                        if torch.any(arrival_mask)
                        else 0.0
                    )
                    bonus_amplitude = _relay_amplitude_from_arrival(arrival_support)
                    base_amplitude = float(
                        route_base_weights.get(int(checkpoint_idx), 0.0)
                    )
                    downstream_parent = route_parent_map.get(int(checkpoint_idx), None)
                    if downstream_parent is None:
                        downstream_support_ids = segmented_data.get("goal_support_ids", [])
                        downstream_target_xy = (float(goal_x), float(goal_z))
                    else:
                        downstream_support_ids = (
                            segmented_data.get("checkpoint_support_ids", {}).get(
                                int(downstream_parent), []
                            )
                        )
                        if (
                            0 <= int(downstream_parent) < len(self.detected_doorways)
                        ):
                            downstream_target_xy = (
                                float(self.detected_doorways[int(downstream_parent)][0]),
                                float(self.detected_doorways[int(downstream_parent)][1]),
                            )
                        else:
                            downstream_target_xy = (float(goal_x), float(goal_z))
                    checkpoint_replay_mask = checkpoint_support_mask
                    owned_components = _get_resolved_checkpoint_owned_components(int(checkpoint_idx))
                    checkpoint_dir_log = (
                        f"dir=owned_room({','.join(str(int(comp)) for comp in owned_components)})"
                        if owned_components
                        else "dir=owned_room(?)"
                    )
                    checkpoint_replay_mask, checkpoint_dir_log = (
                        self._build_checkpoint_directional_replay_mask(
                            checkpoint_idx=int(checkpoint_idx),
                            support_mask=checkpoint_support_mask,
                            support_region=checkpoint_region,
                            downstream_field=backbone_support_field,
                            arrival_mask=arrival_mask,
                            downstream_support_ids=downstream_support_ids,
                            downstream_target_xy=downstream_target_xy,
                            target_device=goal_rcn.w_in.device,
                            strict_upstream_only=True,
                        )
                    )
                    checkpoint_replay_count = int(
                        torch.count_nonzero(checkpoint_replay_mask).item()
                    )
                    checkpoint_decay_envelope = _build_distance_decay_envelope(
                        checkpoint_replay_mask,
                        support_region=checkpoint_region,
                        default_point=self.detected_doorways[int(checkpoint_idx)]
                        if 0 <= int(checkpoint_idx) < len(self.detected_doorways)
                        else (float(goal_x), float(goal_z)),
                        tau=checkpoint_room_decay_tau,
                    )
                    relay_amplitude = bonus_amplitude
                    relay_threshold = float(
                        max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
                    )
                    if checkpoint_mode == "multi_route_memory_graph" and route_selected_data is not None:
                        # Equal replay strength for all checkpoints — no relay scaling
                        relay_amplitude = 1.0
                        route_bonus_weights[int(checkpoint_idx)] = 0.0
                        route_total_weights[int(checkpoint_idx)] = 1.0
                    elif checkpoint_mode in {"route_selected_tree", "directed_route_memory_tree"} and route_selected_data is not None:
                        relay_amplitude = min(
                            float(max(1.0, getattr(self, "checkpoint_max_relay_weight", 4.0))),
                            max(0.0, base_amplitude + bonus_amplitude),
                        )
                        route_bonus_weights[int(checkpoint_idx)] = float(bonus_amplitude)
                        route_total_weights[int(checkpoint_idx)] = float(relay_amplitude)
                    if (
                        checkpoint_mode not in {"route_selected_tree", "directed_route_memory_tree", "multi_route_memory_graph"}
                        and (arrival_support < relay_threshold or relay_amplitude <= 1e-9)
                    ):
                        source_logs.append(
                            "cp"
                            f"{checkpoint_idx}_segment(skip,arrive_backbone={arrival_support:.3f},"
                            f"amp={relay_amplitude:.3f},support={checkpoint_support_count})"
                        )
                        continue
                    if relay_amplitude <= 1e-9:
                        source_logs.append(
                            "cp"
                            f"{checkpoint_idx}_segment(skip,arrive_backbone={arrival_support:.3f},"
                            f"base={base_amplitude:.3f},bonus={bonus_amplitude:.3f},"
                            f"amp={relay_amplitude:.3f},support={checkpoint_support_count})"
                        )
                        continue

                    relay_seed = (
                        checkpoint_seed.to(goal_rcn.w_in.device, dtype=torch.float32)
                        * relay_amplitude
                    )
                    checkpoint_sparse_seed, checkpoint_sparse_idx, checkpoint_sparse_log = (
                        _build_sparse_source_seed(
                            relay_seed,
                            support_mask=checkpoint_replay_mask,
                        )
                    )
                    checkpoint_path_mode = "path_replay"
                    checkpoint_consensus_log = "consensus=off"
                    checkpoint_consensus_trunks = 0
                    checkpoint_consensus_samples = 0
                    checkpoint_consensus_steps = 0
                    checkpoint_spread_seed = torch.zeros_like(checkpoint_sparse_seed)
                    checkpoint_neighbor_seed = None
                    checkpoint_trunk_sequences = (
                        segmented_data.get("checkpoint_trunk_time_sequences", {}).get(
                            int(checkpoint_idx), []
                        )
                    )
                    checkpoint_path_raw = None
                    checkpoint_replay_steps = 0
                    checkpoint_replay_event_count = 0
                    checkpoint_claimed_raw = 0
                    checkpoint_path_fill = None
                    if trajectory_state_goal_map:
                        checkpoint_path_raw, checkpoint_neighbor_raw, checkpoint_ts_meta = (
                            self._build_trajectory_state_segment_fields(
                                scale_arrays=segmented_data["scale_arrays"],
                                compact_time_indices=segmented_data.get("time_indices", []),
                                trunk_time_sequences=checkpoint_trunk_sequences,
                                support_mask=checkpoint_replay_mask,
                                target_device=goal_rcn.w_in.device,
                                source_kind=f"checkpoint_{int(checkpoint_idx)}",
                                room_fill_gain=checkpoint_room_fill_gain,
                                room_decay_tau=checkpoint_room_decay_tau,
                                core_gain=checkpoint_core_gain,
                                support_time_indices=segmented_data.get(
                                    "checkpoint_support_times", {}
                                ).get(int(checkpoint_idx), []),
                            )
                        )
                        checkpoint_replay_steps = int(
                            checkpoint_ts_meta.get("max_hops", 0)
                        ) + 1
                        checkpoint_claimed_raw = int(
                            torch.count_nonzero(checkpoint_path_raw > 1e-8).item()
                        )
                        checkpoint_path_mode = str(
                            checkpoint_ts_meta.get(
                                "path_mode", "trajectory_state_backbone"
                            )
                        )
                        checkpoint_consensus_log = str(
                            checkpoint_ts_meta.get("backbone_log", "trajectory_state")
                        )
                        checkpoint_consensus_trunks = int(
                            checkpoint_ts_meta.get("trunk_count", 0)
                        )
                        checkpoint_consensus_samples = int(
                            checkpoint_ts_meta.get("candidate_states", 0)
                        )
                        checkpoint_consensus_steps = int(
                            checkpoint_ts_meta.get("max_hops", 0)
                        ) + 1
                        checkpoint_path_compact_log = (
                            f"cp{int(checkpoint_idx)}_path:trajectory_state"
                        )
                        checkpoint_fill_compact_log = (
                            f"cp{int(checkpoint_idx)}_fill:trajectory_state"
                        )
                        checkpoint_fill_env_log = (
                            f"cp{int(checkpoint_idx)}_fill:distance_decay("
                            f"tau={checkpoint_room_decay_tau:.2f})"
                        )
                        checkpoint_fill_scale_log = str(
                            checkpoint_ts_meta.get("scale_log", "lambda=na")
                        )
                        (
                            checkpoint_local_anchor,
                            checkpoint_local_anchor_gate_d2,
                            checkpoint_local_anchor_log,
                        ) = _build_local_source_anchor(
                            support_times=segmented_data.get(
                                "checkpoint_support_times", {}
                            ).get(int(checkpoint_idx), []),
                            support_mask=checkpoint_replay_mask,
                            branch_label=f"cp{int(checkpoint_idx)}",
                            support_region=checkpoint_region,
                            default_point=(
                                float(self.detected_doorways[int(checkpoint_idx)][0]),
                                float(self.detected_doorways[int(checkpoint_idx)][1]),
                            ),
                        )
                        if (
                            checkpoint_local_anchor is not None
                            and checkpoint_local_anchor_gate_d2 is not None
                        ):
                            local_anchor_name_bank.append(f"cp{int(checkpoint_idx)}")
                            local_anchor_weight_bank.append(checkpoint_local_anchor)
                            local_anchor_gate_d2_bank.append(
                                checkpoint_local_anchor_gate_d2
                        )
                        checkpoint_path_fill, checkpoint_path_raw_peak = _apply_segment_budget(
                            checkpoint_path_raw,
                            checkpoint_replay_mask,
                            checkpoint_path_budget,
                        )
                    elif room_masked_experience_goal_map:
                        (
                            checkpoint_replay_events,
                            checkpoint_exact_seed_log,
                        ) = _build_recent_contact_replay_event_seeds(
                            support_times=segmented_data.get(
                                "checkpoint_support_times", {}
                            ).get(int(checkpoint_idx), []),
                            trunk_time_sequences=segmented_data.get(
                                "checkpoint_trunk_time_sequences", {}
                            ).get(int(checkpoint_idx), []),
                            support_mask=checkpoint_replay_mask,
                            branch_label=f"cp{int(checkpoint_idx)}",
                            support_region=checkpoint_region,
                            default_point=(
                                float(self.detected_doorways[int(checkpoint_idx)][0]),
                                float(self.detected_doorways[int(checkpoint_idx)][1]),
                            ),
                            max_events=checkpoint_replay_event_limit,
                            event_groups=self._get_recent_checkpoint_crossing_event_groups(
                                checkpoint_idx=int(checkpoint_idx),
                                goal_name=goal["name"],
                                goal_location=(float(goal_x), float(goal_z)),
                                max_events=checkpoint_replay_event_limit,
                            ),
                            event_source="checkpoint_crossing_history",
                        )
                        if not checkpoint_replay_events:
                            checkpoint_exact_seed = relay_seed.to(
                                goal_rcn.w_in.device, dtype=torch.float32
                            )
                            checkpoint_exact_seed = torch.where(
                                checkpoint_replay_mask,
                                checkpoint_exact_seed,
                                torch.zeros_like(checkpoint_exact_seed),
                            )
                            checkpoint_replay_events = [
                                (checkpoint_exact_seed, -1, -1)
                            ]
                            checkpoint_exact_seed_log = (
                                f"{checkpoint_exact_seed_log}|fallback=relay_seed"
                            )
                        checkpoint_replay_event_count = int(
                            len(checkpoint_replay_events)
                        )
                        checkpoint_sparse_idx = int(
                            torch.argmax(checkpoint_replay_events[-1][0]).item()
                        )
                        checkpoint_path_raw, checkpoint_replay_steps, checkpoint_claimed_raw = (
                            _build_masked_experience_replay_field(
                                checkpoint_rcn,
                                checkpoint_replay_events,
                                checkpoint_replay_mask,
                                source_gain=checkpoint_core_gain,
                            )
                        )
                        checkpoint_path_mode = "room_masked_experience_replay"
                        checkpoint_consensus_log = (
                            f"{checkpoint_exact_seed_log},tau={paper_replay_tau:.2f},"
                            f"steps={int(checkpoint_rcn.replay_timesteps)}"
                        )
                        checkpoint_consensus_trunks = 0
                        checkpoint_consensus_samples = 0
                        checkpoint_consensus_steps = int(checkpoint_replay_steps)
                        checkpoint_path_compact_log = (
                            f"cp{int(checkpoint_idx)}_path:masked_experience_replay"
                        )
                        checkpoint_fill_compact_log = (
                            f"cp{int(checkpoint_idx)}_fill:disabled"
                        )
                        checkpoint_fill_env_log = (
                            f"cp{int(checkpoint_idx)}_fill:disabled"
                        )
                        checkpoint_fill_scale_log = "fillscale=none"
                        checkpoint_neighbor_raw = torch.zeros_like(checkpoint_path_raw)
                        (
                            checkpoint_local_anchor,
                            checkpoint_local_anchor_gate_d2,
                            checkpoint_local_anchor_log,
                        ) = _build_local_source_anchor(
                            support_times=segmented_data.get(
                                "checkpoint_support_times", {}
                            ).get(int(checkpoint_idx), []),
                            support_mask=checkpoint_replay_mask,
                            branch_label=f"cp{int(checkpoint_idx)}",
                            support_region=checkpoint_region,
                            default_point=(
                                float(self.detected_doorways[int(checkpoint_idx)][0]),
                                float(self.detected_doorways[int(checkpoint_idx)][1]),
                            ),
                        )
                        if (
                            checkpoint_local_anchor is not None
                            and checkpoint_local_anchor_gate_d2 is not None
                        ):
                            local_anchor_name_bank.append(f"cp{int(checkpoint_idx)}")
                            local_anchor_weight_bank.append(checkpoint_local_anchor)
                            local_anchor_gate_d2_bank.append(
                                checkpoint_local_anchor_gate_d2
                            )
                        checkpoint_path_fill, checkpoint_path_raw_peak = _apply_segment_budget(
                            checkpoint_path_raw,
                            checkpoint_replay_mask,
                            checkpoint_path_budget,
                        )
                    else:
                        checkpoint_consensus_min_claimed = int(
                            max(
                                1,
                                getattr(
                                    self,
                                    "goal_map_checkpoint_consensus_min_claimed",
                                    2,
                                ),
                            )
                        )
                        if bool(getattr(self, "goal_map_checkpoint_consensus_backbone", True)):
                            (
                                checkpoint_consensus_raw,
                                checkpoint_spread_seed,
                                checkpoint_consensus_trunks,
                                checkpoint_consensus_samples,
                                checkpoint_consensus_steps,
                                checkpoint_consensus_log,
                            ) = _build_backbone_field_from_trunks(
                                checkpoint_trunk_sequences,
                                checkpoint_replay_mask,
                                checkpoint_sparse_seed,
                                source_rcn=checkpoint_rcn,
                                smoothing_transition=checkpoint_neighbor_transition,
                                consensus_mode=True,
                            )
                            if float(torch.max(checkpoint_consensus_raw).item()) > 1e-8:
                                checkpoint_path_raw = checkpoint_consensus_raw
                                checkpoint_replay_steps = int(checkpoint_consensus_steps)
                                checkpoint_claimed_raw = int(
                                    torch.count_nonzero(checkpoint_path_raw > 1e-8).item()
                                )
                                checkpoint_path_mode = "consensus_backbone"
                            raw_backbone_needed = (
                                checkpoint_path_raw is None
                                or checkpoint_claimed_raw < checkpoint_consensus_min_claimed
                            )
                            if raw_backbone_needed and checkpoint_trunk_sequences:
                                (
                                    checkpoint_raw_backbone,
                                    checkpoint_raw_spread_seed,
                                    raw_trunk_count,
                                    raw_support_samples,
                                    raw_backbone_steps,
                                    raw_backbone_log,
                                ) = _build_backbone_field_from_trunks(
                                    checkpoint_trunk_sequences,
                                    checkpoint_replay_mask,
                                    checkpoint_sparse_seed,
                                    source_rcn=checkpoint_rcn,
                                    smoothing_transition=checkpoint_neighbor_transition,
                                    consensus_mode=False,
                                )
                                raw_backbone_peak = float(
                                    torch.max(checkpoint_raw_backbone).item()
                                )
                                raw_backbone_claimed = int(
                                    torch.count_nonzero(
                                        checkpoint_raw_backbone > 1e-8
                                    ).item()
                                )
                                if raw_backbone_peak > 1e-8 and (
                                    checkpoint_path_raw is None
                                    or raw_backbone_claimed > checkpoint_claimed_raw
                                ):
                                    prev_claimed = int(checkpoint_claimed_raw)
                                    checkpoint_path_raw = checkpoint_raw_backbone
                                    checkpoint_spread_seed = checkpoint_raw_spread_seed
                                    checkpoint_replay_steps = int(raw_backbone_steps)
                                    checkpoint_claimed_raw = int(raw_backbone_claimed)
                                    checkpoint_path_mode = "raw_trunk_backbone"
                                    checkpoint_consensus_trunks = int(raw_trunk_count)
                                    checkpoint_consensus_samples = int(raw_support_samples)
                                    checkpoint_consensus_steps = int(raw_backbone_steps)
                                    checkpoint_consensus_log = (
                                        f"{checkpoint_consensus_log}|"
                                        f"fallback=raw_trunks(claimed={prev_claimed}->{checkpoint_claimed_raw},"
                                        f"samples={raw_support_samples},steps={raw_backbone_steps})|"
                                        f"{raw_backbone_log}"
                                    )
                        if checkpoint_path_raw is None:
                            checkpoint_path_raw, checkpoint_replay_steps, checkpoint_claimed_raw = (
                                _build_masked_path_replay_field(
                                    checkpoint_rcn,
                                    checkpoint_sparse_seed,
                                    checkpoint_path_transition,
                                    checkpoint_replay_mask,
                                    source_gain=checkpoint_core_gain,
                                )
                            )
                            checkpoint_path_mode = "path_replay"
                        checkpoint_path_raw, checkpoint_path_compact_log = _apply_compact_pc_gate(
                            checkpoint_path_raw,
                            branch_label=f"cp{int(checkpoint_idx)}_path",
                        )
                        checkpoint_path_fill, checkpoint_path_raw_peak = _apply_segment_budget(
                            checkpoint_path_raw,
                            checkpoint_replay_mask,
                            checkpoint_path_budget,
                        )
                        if float(torch.max(checkpoint_spread_seed).item()) > 1e-8:
                            checkpoint_neighbor_seed, _ = _apply_segment_budget(
                                checkpoint_spread_seed,
                                checkpoint_replay_mask,
                                checkpoint_path_budget,
                            )
                        if checkpoint_neighbor_seed is None:
                            checkpoint_neighbor_seed = checkpoint_path_fill
                        else:
                            checkpoint_neighbor_seed = torch.maximum(
                                checkpoint_path_fill,
                                checkpoint_neighbor_seed,
                            )
                        checkpoint_neighbor_raw = torch.zeros_like(checkpoint_path_fill)
                        checkpoint_fill_scale_log = "fillscale=none"
                        checkpoint_fill_compact_log = (
                            f"cp{int(checkpoint_idx)}_fill:compact=off"
                        )
                        checkpoint_fill_env_log = f"cp{int(checkpoint_idx)}_fill:env=off"
                        checkpoint_local_anchor_log = (
                            f"cp{int(checkpoint_idx)}:local_anchor=off"
                        )
                        if float(torch.max(checkpoint_path_fill).item()) > 1e-8:
                            checkpoint_neighbor_raw = checkpoint_rcn._competitive_neighbor_from_source(
                                transition=checkpoint_neighbor_transition,
                                source_wave=checkpoint_neighbor_seed,
                                num_steps=checkpoint_neighbor_steps,
                                step_decay=checkpoint_neighbor_decay,
                                base_values=checkpoint_neighbor_seed,
                                frontier_only=bool(
                                    getattr(checkpoint_rcn, "goal_map_neighbor_frontier_only", True)
                                ),
                            )
                            checkpoint_neighbor_raw = torch.where(
                                checkpoint_replay_mask,
                                checkpoint_neighbor_raw,
                                torch.zeros_like(checkpoint_neighbor_raw),
                            )
                            checkpoint_neighbor_raw, checkpoint_fill_compact_log = (
                                _apply_compact_pc_gate(
                                    checkpoint_neighbor_raw,
                                    branch_label=f"cp{int(checkpoint_idx)}_fill",
                                )
                            )
                            checkpoint_neighbor_raw, checkpoint_fill_env_log = (
                                _apply_fill_source_envelope(
                                    checkpoint_neighbor_raw,
                                    source_xy=(
                                        float(self.detected_doorways[int(checkpoint_idx)][0]),
                                        float(self.detected_doorways[int(checkpoint_idx)][1]),
                                    ),
                                    branch_label=f"cp{int(checkpoint_idx)}_fill",
                                )
                            )
                            checkpoint_neighbor_raw, checkpoint_fill_scale_log = (
                                _attenuate_fill_by_scale(checkpoint_neighbor_raw)
                            )
                    if checkpoint_path_fill is None:
                        checkpoint_path_fill, checkpoint_path_raw_peak = _apply_segment_budget(
                            checkpoint_path_raw,
                            checkpoint_replay_mask,
                            checkpoint_path_budget,
                        )
                    checkpoint_neighbor_fill, checkpoint_fill_raw_peak = _apply_segment_budget(
                        checkpoint_neighbor_raw,
                        checkpoint_replay_mask,
                        checkpoint_fill_budget,
                    )
                    checkpoint_fill = torch.maximum(
                        checkpoint_path_fill, checkpoint_neighbor_fill
                    )
                    source_name = f"cp{int(checkpoint_idx)}"
                    segment_debug["source_names"].append(source_name)
                    segment_debug["path_maps_pre_smooth"][source_name] = _cpu_clone_tensor(
                        checkpoint_path_fill
                    )
                    segment_debug["fill_maps_pre_smooth"][source_name] = _cpu_clone_tensor(
                        checkpoint_neighbor_fill
                    )
                    segment_debug["total_maps_pre_smooth"][source_name] = _cpu_clone_tensor(
                        checkpoint_fill
                    )
                    segment_debug["support_masks"][source_name] = _cpu_clone_tensor(
                        checkpoint_replay_mask.to(dtype=torch.float32)
                    )
                    segment_debug["metadata"][source_name] = {
                        "support_count": int(checkpoint_support_count),
                        "replay_count": int(checkpoint_replay_count),
                        "replay_event_count": int(checkpoint_replay_event_count),
                        "replay_steps": int(checkpoint_replay_steps),
                        "path_budget": float(checkpoint_path_budget),
                        "fill_budget": float(checkpoint_fill_budget),
                        "total_budget": float(checkpoint_total_budget),
                        "seed_index": int(checkpoint_sparse_idx),
                        "seed_log": str(checkpoint_sparse_log),
                        "path_mode": str(checkpoint_path_mode),
                        "consensus_log": str(checkpoint_consensus_log),
                        "consensus_trunks": int(checkpoint_consensus_trunks),
                        "consensus_samples": int(checkpoint_consensus_samples),
                        "path_raw_peak": float(checkpoint_path_raw_peak),
                        "fill_raw_peak": float(checkpoint_fill_raw_peak),
                        "compactness_log": str(checkpoint_path_compact_log),
                        "fill_compactness_log": str(checkpoint_fill_compact_log),
                        "fill_env_log": str(checkpoint_fill_env_log),
                        "arrival_backbone": float(arrival_support),
                        "base_amplitude": float(base_amplitude),
                        "bonus_amplitude": float(bonus_amplitude),
                        "relay_amplitude": float(relay_amplitude),
                        "owned_components": [int(comp) for comp in owned_components],
                        "claimed_count": int(checkpoint_claimed_raw),
                        "direction_log": str(checkpoint_dir_log),
                        "spread_mode": (
                            "trajectory_state_backbone_plus_room_fill"
                            if trajectory_state_goal_map
                            else (
                                "room_masked_experience_replay"
                                if room_masked_experience_goal_map
                                else "path_replay_plus_neighbor"
                            )
                        ),
                        "fill_scale_log": str(checkpoint_fill_scale_log),
                        "local_anchor_log": str(checkpoint_local_anchor_log),
                    }
                    segment_total_fields.append(checkpoint_fill.clone())
                    claimed = int((checkpoint_fill > 1e-8).sum().item())
                    composed_path = torch.maximum(composed_path, checkpoint_path_fill)
                    composed_neighbor = torch.maximum(
                        composed_neighbor, checkpoint_neighbor_fill
                    )
                    composed_values = torch.maximum(composed_path, composed_neighbor)
                    source_logs.append(
                        "cp"
                        f"{checkpoint_idx}_segment(arrive_backbone={arrival_support:.3f},"
                        f"base={base_amplitude:.3f},"
                        f"bonus={bonus_amplitude:.3f},"
                        f"amp={relay_amplitude:.3f},support={checkpoint_support_count},"
                        f"events={checkpoint_replay_event_count},"
                        f"replay={checkpoint_replay_count},"
                        f"steps={checkpoint_replay_steps},"
                        f"owned={','.join(str(int(comp)) for comp in owned_components) if owned_components else '?'}," 
                        f"seed={checkpoint_sparse_idx},"
                        f"budget={checkpoint_path_budget:.2f}/{checkpoint_fill_budget:.2f},"
                        f"pathmode={checkpoint_path_mode},"
                        f"{checkpoint_consensus_log},"
                        f"{checkpoint_path_compact_log},"
                        f"{checkpoint_fill_compact_log},"
                        f"{checkpoint_fill_env_log},"
                        f"rawpeak={checkpoint_path_raw_peak:.3f}/{checkpoint_fill_raw_peak:.3f},"
                        f"claimed={claimed},"
                        f"{checkpoint_fill_scale_log},"
                        f"{checkpoint_local_anchor_log},"
                        f"{checkpoint_dir_log})"
                    )

                peak = float(torch.max(composed_values).item())
                if peak > 1e-12:
                    composed_values = composed_values / peak
                if segment_total_fields:
                    segment_stack = torch.stack(segment_total_fields, dim=0)
                    winner_values, winner_source_idx = torch.max(segment_stack, dim=0)
                    if peak > 1e-12:
                        normalized_total_maps = {
                            name: _cpu_clone_tensor(field / peak)
                            for name, field in zip(segment_debug["source_names"], segment_total_fields)
                        }
                        winner_values = winner_values / peak
                    else:
                        normalized_total_maps = {
                            name: _cpu_clone_tensor(field)
                            for name, field in zip(segment_debug["source_names"], segment_total_fields)
                        }
                    segment_debug["total_maps_normalized_pre_smooth"] = normalized_total_maps
                    segment_debug["winner_source_idx_pre_smooth"] = (
                        winner_source_idx.detach().cpu().clone().view(-1)
                    )
                    segment_debug["winner_values_pre_smooth"] = (
                        winner_values.detach().cpu().clone().view(-1)
                    )
                else:
                    segment_debug["total_maps_normalized_pre_smooth"] = {}
                    segment_debug["winner_source_idx_pre_smooth"] = torch.empty(
                        0, dtype=torch.long
                    )
                    segment_debug["winner_values_pre_smooth"] = torch.empty(
                        0, dtype=torch.float32
                    )
                segment_debug["global_peak_pre_smooth"] = float(peak)
                segment_debug["composed_path_pre_smooth"] = _cpu_clone_tensor(composed_path)
                segment_debug["composed_fill_pre_smooth"] = _cpu_clone_tensor(composed_neighbor)
                segment_debug["composed_total_pre_smooth"] = _cpu_clone_tensor(
                    composed_values if peak <= 1e-12 else (composed_values * peak)
                )

                goal_rcn.w_in = composed_values.unsqueeze(0)
                goal_rcn.w_in_effective = goal_rcn.w_in.clone()
                if local_anchor_weight_bank and local_anchor_gate_d2_bank:
                    goal_rcn.goal_map_local_anchor_weights = torch.stack(
                        local_anchor_weight_bank, dim=0
                    ).to(goal_rcn.w_in.device, dtype=torch.float32)
                    goal_rcn.goal_map_local_anchor_gate_d2 = torch.stack(
                        local_anchor_gate_d2_bank, dim=0
                    ).to(goal_rcn.w_in.device, dtype=torch.float32)
                    goal_rcn.goal_map_local_anchor_query_sigma = torch.full(
                        (len(local_anchor_weight_bank),),
                        float(local_anchor_query_sigma),
                        dtype=torch.float32,
                        device=goal_rcn.w_in.device,
                    )
                    goal_rcn.goal_map_local_anchor_names = list(local_anchor_name_bank)
                else:
                    goal_rcn.goal_map_local_anchor_weights = None
                    goal_rcn.goal_map_local_anchor_gate_d2 = None
                    goal_rcn.goal_map_local_anchor_query_sigma = None
                    goal_rcn.goal_map_local_anchor_names = None
                goal_rcn.reward_cell_activations = torch.zeros_like(goal_rcn.reward_cell_activations)
                goal_rcn.goal_map_checkpoint_mode = checkpoint_mode
                goal_rcn.goal_map_segment_debug = segment_debug
                replay_log = (
                    "goal_map=segmented_checkpoint_cascade("
                    f"num_sources={1 + len(active_checkpoint_order)}),"
                    f"{history_log},{segmented_log},{visibility_log},"
                    f"{path_kernel_log},{local_kernel_log},{room_partition_log},"
                    f"{self._format_route_selected_checkpoint_log(route_selected_data, route_bonus_weights)},"
                    + ";".join(source_logs)
                )
                goal_rcn.goal_map_debug_log = replay_log
            else:
                topology_log = "relay_topology=disabled"
                checkpoint_weight_bank = []
                goal_seed_gain = 1.0
                if checkpoint_seed_bank:
                    # Use the checkpoint seed itself as the relay ensemble.
                    # This stays local to the checkpoint neighborhood without the
                    # over-broad path-kernel expansion that was suppressing firing.
                    checkpoint_presence_bank = [
                        checkpoint_seed.to(goal_rcn.w_in.device, dtype=torch.float32).clone()
                        for _, checkpoint_seed in checkpoint_seed_bank
                    ]
                    checkpoint_mode = str(
                        getattr(self, "goal_map_checkpoint_mode", "implicit_cascade")
                    ).strip().lower()
                    if checkpoint_mode in {
                        "topology_schedule",
                        "route_selected_tree",
                        "directed_route_memory_tree",
                        "multi_route_memory_graph",
                    }:
                        goal_seed_gain, checkpoint_weight_bank, topology_log = (
                            self._compute_checkpoint_topology_schedule(
                                reward_rcn=goal_rcn,
                                goal_seed=seed_activations,
                                checkpoint_seeds=[seed for _, seed in checkpoint_seed_bank],
                                path_constraint=path_constraint,
                            )
                        )
                    else:
                        checkpoint_scale = float(
                            max(0.0, getattr(self, "goal_map_checkpoint_scale", 0.25))
                        )
                        checkpoint_weight_bank = [checkpoint_scale] * len(checkpoint_seed_bank)
                        goal_seed_gain = 1.0
                        topology_log = (
                            "relay_mode=implicit_cascade("
                            f"cp_scale={checkpoint_scale:.2f},"
                            f"threshold={float(max(0.0, getattr(self, 'checkpoint_boost_threshold', 0.01))):.2f})"
                        )
                    goal_rcn.goal_map_checkpoint_seeds = torch.stack(
                        [seed for _, seed in checkpoint_seed_bank], dim=0
                    ).to(goal_rcn.w_in.device)
                    goal_rcn.goal_map_checkpoint_presence = torch.stack(
                        checkpoint_presence_bank, dim=0
                    ).to(goal_rcn.w_in.device)
                    goal_rcn.goal_map_checkpoint_weights = torch.tensor(
                        checkpoint_weight_bank,
                        dtype=torch.float32,
                        device=goal_rcn.w_in.device,
                    )
                    goal_rcn.goal_map_checkpoint_threshold = float(
                        max(0.0, getattr(self, "checkpoint_boost_threshold", 0.01))
                    )
                    goal_rcn.goal_map_seed_gain = float(goal_seed_gain)
                    goal_rcn.goal_map_checkpoint_mode = checkpoint_mode
                    goal_rcn.goal_map_checkpoint_scale = float(
                        checkpoint_weight_bank[0] if checkpoint_weight_bank else 0.0
                    )
                    goal_rcn.goal_map_checkpoint_support_mode = str(
                        getattr(self, "goal_map_checkpoint_support_mode", "threshold_normalized")
                    ).strip().lower()
                else:
                    goal_rcn.goal_map_checkpoint_seeds = None
                    goal_rcn.goal_map_checkpoint_presence = None
                    goal_rcn.goal_map_checkpoint_weights = None
                    goal_rcn.goal_map_seed_gain = 1.0
                    goal_rcn.goal_map_checkpoint_mode = str(
                        getattr(self, "goal_map_checkpoint_mode", "implicit_cascade")
                    ).strip().lower()
                    goal_rcn.goal_map_checkpoint_scale = float(
                        max(0.0, getattr(self, "goal_map_checkpoint_scale", 0.25))
                    )
                    goal_rcn.goal_map_checkpoint_support_mode = str(
                        getattr(self, "goal_map_checkpoint_support_mode", "threshold_normalized")
                    ).strip().lower()
                goal_rcn.goal_map_path_constraint = path_constraint
                goal_rcn.goal_map_neighbor_constraint = local_kernel
                goal_rcn.goal_map_spatial_constraint = (
                    visibility_mask if visibility_mask is not None else local_kernel
                )
                replay_log = goal_rcn.build_goal_map_from_custom_activations(
                    unified_pcn=self.unified_pcn,
                    custom_activations=seed_activations,
                )
                replay_log = (
                    f"{replay_log},{history_log},{topology_log},{segmented_log},"
                    f"{visibility_log},{path_kernel_log},{local_kernel_log}"
                )
                goal_rcn.goal_map_debug_log = replay_log
        elif hasattr(goal_rcn, "replay_with_custom_activations"):
            visibility_log = self._sync_unified_recurrent_visibility_mask(prune_weights=True)
            goal_rcn.replay_with_custom_activations(
                unified_pcn=self.unified_pcn,
                custom_activations=seed_activations,
                use_scale_gate=True,
            )
            replay_log = f"goal_map=additive_replay,{history_log},{visibility_log}"
        else:
            # Legacy fallback: use the supplied transition matrix directly if the
            # unified replay helper is unavailable.
            C_REWARD = float(getattr(goal_rcn, "C_REWARD", 5.0))
            weight_update = torch.zeros_like(goal_rcn.w_in)
            unified_replay_steps = max(
                1, int(getattr(self, "goal_map_replay_timesteps", 12))
            )
            lambda_unified = float(max(1.0, getattr(goal_rcn, "lambda_s", 20.0)))
            A_unified = C_REWARD / max(lambda_unified, 1e-6)

            v = seed_activations
            for t in range(unified_replay_steps):
                decay = math.exp(-t / lambda_unified)
                norm_val = torch.sqrt(torch.max(
                    torch.sum(v ** 2), torch.tensor(1e-12, device=rcn_device)
                ))
                v_norm = v / norm_val
                v_norm = torch.where(torch.isnan(v_norm), torch.zeros_like(v_norm), v_norm)
                weight_update[0, :] += A_unified * decay * v_norm
                v = torch.tanh(torch.relu(torch.matmul(_w_exp_norm, v_norm) + v_norm))

            max_val = torch.max(torch.abs(weight_update))
            if torch.isfinite(max_val) and max_val > 1e3:
                weight_update = weight_update / max_val
            goal_rcn.w_in = goal_rcn.w_in + weight_update
            replay_log = f"goal_map=legacy_fallback,{history_log}"

        goal_rcn.w_in = torch.clamp(goal_rcn.w_in, min=0.0)
        goal_rcn.w_in_effective = torch.clamp(goal_rcn.w_in.clone(), min=0.0)
        return goal_rcn, f"{seed_log} | {replay_log} | smooth=disabled"

    def _create_multi_goal_reward_maps(self):
        """Create reward maps for each goal-scale combination"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        os.makedirs(multi_goal_dir, exist_ok=True)
        build_start = time.perf_counter()

        if self._should_defer_unified_goal_associations():
            self._finalize_deferred_goal_associations()

        # Whether full hmap activation histories are available for goal-visit lookup.
        # When available, we use the ACTUAL recorded activation vector at the goal-visit step —
        # exactly what v11 did with pcn.place_cell_activations, just stored in hmaps.
        T = self.step_count
        hmap_ok = (
            bool(self.hmap_pcn_activities)
            and T > 0
            and not getattr(self, "lightweight_hmaps", False)
        )

        if self.use_unified_multiscale:
            print(f"[LEARN_LOCATIONS] Creating unified goal reward maps for {len(self.goals)} goals")

            created_maps = 0
            for goal in self.goals:
                associations = self.goal_place_cell_associations[goal["name"]]
                candidate_scales = [i for i, pc_idx in enumerate(associations) if pc_idx is not None]
                if not candidate_scales:
                    print(f"[WARNING] No place-cell association for goal {goal['name']}; skipping unified reward map.")
                    continue

                goal_x = float(goal["location"][0])
                goal_y = float(goal["location"][1])
                goal_rcn, seed_log = self._compute_reward_weights(
                    goal_x,
                    goal_y,
                    None,
                    self.unified_rcn,
                    seed_sigma=float(goal.get("radius", 0.8)),
                    goal_name=str(goal["name"]),
                )

                unified_goal_path = os.path.join(multi_goal_dir, f"unified_rcn_goal_{goal['name']}.pkl")
                _dump_pickle_file(unified_goal_path, goal_rcn)
                created_maps += 1
                print(
                    f"[LEARN_LOCATIONS] Created unified reward map for goal '{goal['name']}': "
                    f"{seed_log}"
                )

            print(
                f"[LEARN_LOCATIONS] Successfully created {created_maps} unified goal reward maps "
                f"in {time.perf_counter() - build_start:.2f}s"
            )
            return

        print(f"[LEARN_LOCATIONS] Creating {len(self.goals)} goals x {len(self.scales)} scales reward maps")

        created_maps = 0
        for goal in self.goals:
            goal_steps = self.goal_association_step[goal["name"]]
            for scale_idx, (pcn, rcn) in enumerate(zip(self.pcns, self.rcns)):
                pc_idx = self.goal_place_cell_associations[goal["name"]][scale_idx]

                if pc_idx is None:
                    print(f"[WARNING] No place cell associated with {goal['name']} for scale {scale_idx}")
                    continue

                # Use actual recorded activations at goal-visit step (v11 approach).
                seed_activations = None
                step = goal_steps[scale_idx]
                if hmap_ok and step is not None and scale_idx < len(self.hmap_pcn_activities):
                    step = int(step)
                    if step < self.hmap_pcn_activities[scale_idx].shape[0]:
                        scale_acts = self.hmap_pcn_activities[scale_idx][step]
                        if hasattr(scale_acts, "detach"):
                            scale_acts = scale_acts.detach().cpu()
                        seed_activations = scale_acts.float().to(rcn.device if hasattr(rcn, "device") else torch.device("cpu"))

                if seed_activations is None:
                    # Fallback: single-cell artificial pattern.
                    seed_activations = torch.zeros_like(pcn.place_cell_activations)
                    seed_activations[int(pc_idx)] = 1.0

                scale_name = self.scales[scale_idx]["name"]
                n_active = int((seed_activations > 0.05).sum())
                src = f"hmap@step{int(step)}({n_active}active)" if (hmap_ok and step is not None) else f"fallback(pc={pc_idx})"
                print(f"[LEARN_LOCATIONS] Creating reward map for {goal['name']} scale {scale_idx}: {src}")

                goal_rcn = copy.deepcopy(rcn)
                # Start each goal map from a clean reward state to avoid all-goal contamination.
                goal_rcn.w_in = torch.zeros_like(goal_rcn.w_in)
                goal_rcn.w_in_effective = goal_rcn.w_in.clone()
                goal_rcn.reward_cell_activations = torch.zeros_like(goal_rcn.reward_cell_activations)
                # Use pure STDP-learned weights for replay (matches v11 behaviour — no trajectory blending).
                goal_rcn.use_experience_replay = False
                goal_rcn.update_reward_cell_activations(seed_activations, visit=True)
                if hasattr(goal_rcn, "replay_with_custom_activations"):
                    goal_rcn.replay_with_custom_activations(pcn=pcn, custom_activations=seed_activations)
                else:
                    goal_rcn.replay(pcn=pcn)

                goal_rcn_path = os.path.join(multi_goal_dir, f"rcn_scale_{scale_idx}_goal_{goal['name']}.pkl")
                _dump_pickle_file(goal_rcn_path, goal_rcn)
                created_maps += 1
                print(f"[LEARN_LOCATIONS] Created: {scale_name}_goal_{goal['name']}")

        print(
            f"[LEARN_LOCATIONS] Successfully created {created_maps} reward maps "
            f"in {time.perf_counter() - build_start:.2f}s"
        )
    def _save_multi_goal_data(self):
        """Save multi-goal specific data"""
        # Room partition is now derived from replay boundaries at exploit time.
        # No geometry-based room groups need to be pre-built.
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        # Save goal associations
        associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        association_data = {
            "goal_place_cell_associations": self.goal_place_cell_associations,
            "goal_association_step": self.goal_association_step,
            "goal_place_cell_activations": self.goal_place_cell_activations,
            "goal_contact_steps": getattr(self, "goal_contact_steps", {}),
            "goal_visit_counts": self.goal_visit_counts,
            "checkpoint_visit_counts": self.checkpoint_visit_counts,
            "checkpoint_crossing_counts": self.checkpoint_crossing_counts,
            "checkpoint_crossing_steps": getattr(
                self, "checkpoint_crossing_steps", {}
            ),
            "checkpoint_to_checkpoint_edges": getattr(
                self, "checkpoint_to_checkpoint_edges", {}
            ),
            "checkpoint_to_goal_edges": getattr(
                self, "checkpoint_to_goal_edges", {}
            ),
            "checkpoint_pc_groups": {
                k: {side: t.detach().cpu() for side, t in v.items()}
                for k, v in getattr(self, "checkpoint_pc_groups", {}).items()
            },
            "checkpoint_proximity_pcs": {
                k: t.detach().cpu()
                for k, t in getattr(self, "checkpoint_proximity_pcs", {}).items()
            },
            "checkpoints": self.detected_doorways,
            "goals": self.goals,
            "scales": [{"scale_index": s["scale_index"], "name": s["name"]} for s in self.scales],
            "phase2_start_step": self.phase2_start_step,
            "adaptive_stdp_start_step": self.adaptive_stdp_start_step,
            "total_steps": self.step_count,
            "final_time": self.getTime(),
            "goal_associations_finalized": getattr(
                self, "goal_associations_finalized", False
            ),
            "defer_goal_association_until_training_end": getattr(
                self, "defer_goal_association_until_training_end", False
            ),
        }
        _dump_pickle_file(associations_path, association_data)

        print(f"[LEARN_LOCATIONS] Saved goal associations to {associations_path}")

        # Print summary
        print(f"[LEARN_LOCATIONS] Final associations:")
        for goal_name, associations in self.goal_place_cell_associations.items():
            goal_info = next(g for g in self.goals if g["name"] == goal_name)
            visit_count = self.goal_visit_counts[goal_name]
            print(f"  {goal_name} at {goal_info['location']}: {associations} (visits: {visit_count})")
        if self.checkpoint_crossing_counts:
            print(f"[LEARN_LOCATIONS] Checkpoint crossings: {self.checkpoint_crossing_counts}")


    def _save_trial_completion_time(self, trial_elapsed_time):
        """Save trial completion time to JSON for AUTO mode"""
        import json

        if self.trial_base_dir is None:
            print("[WARNING] Cannot save trial completion time - not in AUTO mode")
            return

        completion_data = {
            "trial_number": self.current_auto_trial,
            "world_name": self.world_name,
            "completion_time_seconds": trial_elapsed_time,
            "coverage_percentage": self.current_coverage_percentage * 100,
            "total_steps": self.step_count,
            "trial_name": self.auto_trial_name
        }

        json_path = os.path.join(self.trial_base_dir, "trial_completion_time.json")
        with open(json_path, "w") as f:
            json.dump(completion_data, f, indent=2)

        print(f"[AUTO_TRIAL] Saved trial completion time to {json_path}")
        print(f"[AUTO_TRIAL] Trial {self.current_auto_trial}: {trial_elapsed_time:.1f}s, Coverage: {self.current_coverage_percentage*100:.1f}%")

    def _handle_random_goal_exploitation(self, goal):
        """Handle goal reached during EXPLOIT_LOCATIONS_RANDOM mode"""
        if getattr(self, "_trial_terminal_handled", False):
            return
        self._trial_terminal_handled = True

        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time

            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            self.stats_collector.update_stat("goal_location", goal["location"])
            self.stats_collector.update_stat("goal_name", goal["name"])
            self.stats_collector.update_stat("total_distance_traveled", round(self.total_distance_traveled, 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("goal_reached", True)

            # Add random spawn specific stats
            if hasattr(self, 'optimal_path_distance') and self.optimal_path_distance:
                path_ratio = self.total_distance_traveled / self.optimal_path_distance
                self.stats_collector.update_stat("optimal_path_distance", round(self.optimal_path_distance, 2))
                self.stats_collector.update_stat("path_ratio", round(path_ratio, 2))
                print(f"[RANDOM_EXPLOIT] Goal '{goal['name']}' reached! "
                      f"Distance: {self.total_distance_traveled:.1f}m, "
                      f"Time: {trial_time:.1f}s, "
                      f"Ratio: {path_ratio:.2f}")

            self.stats_collector.update_stat("spawn_method", "random")
            self.stats_collector.update_stat("path_failure_ratio", self.path_failure_ratio)
            self.stats_collector.update_stat("termination_reason", "goal_reached")
            self.stats_collector.save_stats(self.trial_id)

        self.stop()
        self._reset_unified_heading_commit_state(reason="goal_reached")
        if not getattr(self, "lightweight_hmaps", False):
            self.save(include_hmaps=True)
        else:
            print("[RANDOM_EXPLOIT] Skipping per-trial hmap save in lightweight mode")
        self.done = True

    def _handle_random_exploitation_timeout(self):
        """Handle timeout during EXPLOIT_LOCATIONS_RANDOM mode"""
        if getattr(self, "_trial_terminal_handled", False):
            return
        self._trial_terminal_handled = True

        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time

            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)

            # Get active goal info
            active_goal = next((g for g in self.goals if g.get("active", False)), None)
            if active_goal:
                self.stats_collector.update_stat("goal_location", active_goal["location"])
                self.stats_collector.update_stat("goal_name", active_goal["name"])

            self.stats_collector.update_stat("total_distance_traveled", round(self.total_distance_traveled, 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("goal_reached", False)

            # Add random spawn specific stats
            if hasattr(self, 'optimal_path_distance') and self.optimal_path_distance:
                path_ratio = self.total_distance_traveled / self.optimal_path_distance
                self.stats_collector.update_stat("optimal_path_distance", round(self.optimal_path_distance, 2))
                self.stats_collector.update_stat("path_ratio", round(path_ratio, 2))
                termination_reason = "distance_limit" if path_ratio >= self.path_failure_ratio else "time_limit"
            else:
                termination_reason = "time_limit"

            self.stats_collector.update_stat("spawn_method", "random")
            self.stats_collector.update_stat("path_failure_ratio", self.path_failure_ratio)
            self.stats_collector.update_stat("termination_reason", termination_reason)

            print(f"[RANDOM_EXPLOIT] Trial timeout - "
                  f"Distance: {self.total_distance_traveled:.1f}m, "
                  f"Time: {trial_time:.1f}s, "
                  f"Reason: {termination_reason}")

            self.stats_collector.save_stats(self.trial_id)

        self.stop()
        self._reset_unified_heading_commit_state(reason="exploit_timeout")
        if not getattr(self, "lightweight_hmaps", False):
            self.save(include_hmaps=True)
        else:
            print("[RANDOM_EXPLOIT] Skipping per-trial hmap save in lightweight mode")
        self.done = True

    ########################################### AUTO PILOT ###########################################

    def auto_pilot(self):
        """
        A fallback or finalizing method that manually drives the robot to the goal
        location when it is close or already exploiting.
        """
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_xy = self._current_planar_xy()

        # Keep moving until close enough to goal
        while not torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor(curr_xy, dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"],
        ):
            curr_xy = self._current_planar_xy()
            delta_x = curr_xy[0] - self.goal_location[0]
            delta_y = curr_xy[1] - self.goal_location[1]

            # Compute desired heading to face the goal
            if delta_x >= 0:
                theta = torch.atan2(
                    torch.abs(
                        torch.tensor(delta_y, dtype=self.dtype, device=self.device)
                    ),
                    torch.abs(
                        torch.tensor(delta_x, dtype=self.dtype, device=self.device)
                    ),
                ).item()
                if delta_y >= 0:
                    desired = 2 * np.pi - theta
                else:
                    desired = np.pi + theta
            elif delta_y >= 0:
                theta = torch.atan2(
                    torch.abs(
                        torch.tensor(delta_y, dtype=self.dtype, device=self.device)
                    ),
                    torch.abs(
                        torch.tensor(delta_x, dtype=self.dtype, device=self.device)
                    ),
                ).item()
                desired = (np.pi / 2) - theta
            else:
                theta = torch.atan2(
                    torch.abs(
                        torch.tensor(delta_x, dtype=self.dtype, device=self.device)
                    ),
                    torch.abs(
                        torch.tensor(delta_y, dtype=self.dtype, device=self.device)
                    ),
                ).item()
                desired = np.pi - theta

            # Turn to desired heading
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))

            # Move forward one step
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True, update_hdn=True, update_pcn=True, update_gcn=True)
            self.forward()
            s_start += 1

    ########################################### HELPER METHODS ###########################################

    def manual_control(self):
        """Enables manual control of the robot using keyboard inputs.

        Controls:
            w or UP_ARROW: Move forward
            a or LEFT_ARROW: Rotate counterclockwise
            s or DOWN_ARROW: Stop movement
            d or RIGHT_ARROW: Rotate clockwise

        Note:
        If control is not working try to click into the sim environment again.
        Sometimes resetting the sim makes the keyboard disconnect.
        """
        k = self.keyboard.getKey()
        if k == ord("W") or k == self.keyboard.UP:
            self.forward()
        elif k == ord("A") or k == self.keyboard.LEFT:
            self.rotate(direction=1, speed_factor=0.3)
        elif k == ord("D") or k == self.keyboard.RIGHT:
            self.rotate(direction=-1, speed_factor=0.3)
        elif k == ord("S") or k == self.keyboard.DOWN:
            self.stop()

        # Always step simulation forward and update sensors
        self.sense()
        self.step(self.timestep)

    def rotate(self, direction: int, speed_factor: float = 0.3):
        """Rotates the robot continuously in the specified direction.

        Args:
            direction (int): 1 for clockwise, -1 for counterclockwise
            speed_factor (float): Multiplier for rotation speed (0.0 to 1.0)
        """
        speed = self.max_speed * speed_factor
        self.left_speed = speed * direction
        self.right_speed = -speed * direction
        self.move()

    def forward(self):
        """Moves the robot forward at maximum speed.

        Sets both wheels to max speed, updates motor movement and sensor readings.
        """
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()

    def compass_based_turn_to_heading(self, target_heading_deg, debug=False):
        """Compass-based turning that breaks large turns into smaller increments.

        Uses compass feedback for accurate turning by breaking large angles into
        smaller steps and verifying each step.

        Args:
            target_heading_deg: Target heading in degrees [0, 360)
            debug: Whether to print debug information

        Returns:
            bool: True if turn was successful, False otherwise
        """
        MAX_SINGLE_TURN = 10.0  # Maximum degrees to turn in one step
        ACCEPTABLE_ERROR = 10.0  # Accept slightly coarser heading lock during exploit turns
        MAX_ATTEMPTS = 30       # Maximum number of turn attempts

        initial_heading = self.current_heading_deg

        # Calculate the shortest angle to turn
        def angle_difference(target, current):
            diff = target - current
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            return diff

        total_angle_needed = angle_difference(target_heading_deg, initial_heading)

        if debug:
            print(f"  Total turn needed: {total_angle_needed:+.1f}°")

        # If already close enough, don't turn
        if abs(total_angle_needed) < 3.0:
            if debug:
                print("  Already at target heading")
            return True

        current_heading = initial_heading
        attempts = 0

        while abs(angle_difference(target_heading_deg, current_heading)) > ACCEPTABLE_ERROR and attempts < MAX_ATTEMPTS:
            attempts += 1

            # Calculate remaining turn needed
            remaining_turn = angle_difference(target_heading_deg, current_heading)

            # Limit turn to MAX_SINGLE_TURN
            if abs(remaining_turn) > MAX_SINGLE_TURN:
                turn_this_step = MAX_SINGLE_TURN * (1 if remaining_turn > 0 else -1)
            else:
                turn_this_step = remaining_turn

            if debug:
                print(f"  Attempt {attempts}: turning {turn_this_step:+.1f}° (remaining: {remaining_turn:+.1f}°)")

            # Execute small turn
            pre_turn_heading = self.current_heading_deg
            self.turn(np.radians(turn_this_step))
            post_turn_heading = self.current_heading_deg

            # Update current heading for next iteration
            current_heading = post_turn_heading

            # Verify this step
            actual_turn = angle_difference(post_turn_heading, pre_turn_heading)
            step_error = abs(actual_turn - turn_this_step)

            if debug and step_error > 10:
                print(f"    Warning: large step error {step_error:.1f}°")

        # Final verification
        final_error = abs(angle_difference(target_heading_deg, current_heading))
        success = final_error <= ACCEPTABLE_ERROR

        if debug:
            total_turn_actual = angle_difference(current_heading, initial_heading)
            print(f"  Final: {initial_heading:.0f}° → {current_heading:.0f}° (error: {final_error:.1f}°)")
            print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")

        return success

    def _execute_movement(self, heading_deg: float, show_debug: bool = False) -> bool:
        """Shared helper: Turn to heading and move forward with verification.

        Args:
            heading_deg: Target heading in degrees.
            show_debug: Whether to print debug info.

        Returns:
            bool: Success of turning/movement.
        """
        self.last_executed_move_heading_deg = None
        self.last_executed_move_hd_bin = None
        self.last_executed_move_distance = 0.0

        angle_to_turn_deg = heading_deg - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180

        if show_debug:
            print(f"Turning from {self.current_heading_deg:.0f}° to {heading_deg:.0f}°")

        success = self.compass_based_turn_to_heading(heading_deg, show_debug)

        if not success:
            if show_debug:
                print("TURNING FAILED - aborting action without fallback movement")
            return False

        # Record pre-movement position
        pre_move_pos = self.robot.getField("translation").getSFVec3f()
        forward_steps = self._get_action_forward_steps()

        # Move forward
        for _ in range(forward_steps):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True, update_hdn=True, update_pcn=True, update_gcn=True)
            self.forward()
            self.check_goal_reached()
            if self.done:
                return True

            # Update rotation accumulator (kept for detection)
            if hasattr(self, 'last_heading_deg') and self.last_heading_deg is not None:
                heading_diff = self.current_heading_deg - self.last_heading_deg
                heading_diff = ((heading_diff + 180) % 360) - 180
                self.rotation_accumulator += abs(heading_diff)
                self.last_heading_deg = self.current_heading_deg
            else:
                # Initialize last_heading_deg if it's None
                self.last_heading_deg = self.current_heading_deg

        # Verify movement if debug
        post_move_pos = self.robot.getField("translation").getSFVec3f()
        pre_move_xy = self._translation_to_planar_xy(pre_move_pos)
        post_move_xy = self._translation_to_planar_xy(post_move_pos)
        actual_dx = post_move_xy[0] - pre_move_xy[0]
        actual_dy = post_move_xy[1] - pre_move_xy[1]
        actual_distance = float(np.sqrt(actual_dx**2 + actual_dy**2))
        actual_heading_deg = self._planar_delta_to_clockwise_heading_deg(
            actual_dx,
            actual_dy,
        )
        if actual_heading_deg is None:
            self.last_executed_move_heading_deg = None
            self.last_executed_move_hd_bin = None
        else:
            self.last_executed_move_heading_deg = float(actual_heading_deg)
            actual_heading_rad = np.deg2rad(-float(actual_heading_deg))
            self.last_executed_move_hd_bin = self._angle_to_hd_bin(actual_heading_rad)
        self.last_executed_move_distance = actual_distance

        if show_debug:
            if actual_heading_deg is not None:
                angle_error = abs(float(actual_heading_deg) - heading_deg)
                if angle_error > 180:
                    angle_error = 360 - angle_error

                status = "✓" if angle_error < 15 else "⚠" if angle_error < 30 else "✗"
                print(
                    f"Moved {actual_distance:.3f}m at {float(actual_heading_deg):.0f}° "
                    f"(target: {heading_deg}°) {status}"
                )

        return True

    def _get_nominal_forward_distance_per_step(self) -> float:
        """Approximate forward distance covered by one forward cycle."""
        dt_seconds = float(self.timestep) / 1000.0
        return float(max(1e-6, self.wheel_radius * self.max_speed * dt_seconds))

    def _get_action_forward_steps(self) -> int:
        """
        Choose how many forward cycles to execute for one action decision.

        Unified exploit uses a conservative fraction of the local lookahead
        distance so action execution scales with the planning horizon instead of
        always committing the fixed legacy `tau_w` chunk.
        """
        exploit_style_mode = getattr(self, "robot_mode", None) in {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }
        if not (exploit_style_mode and getattr(self, "use_unified_multiscale", False)):
            return int(max(1, self.tau_w))

        lookahead = float(getattr(self, "unified_spatial_lookahead_distance", 1.0))
        step_fraction = float(getattr(self, "unified_exploit_step_fraction", 0.25))
        target_distance = max(0.01, step_fraction * max(0.05, lookahead))
        nominal_step_distance = self._get_nominal_forward_distance_per_step()
        forward_steps = int(max(1, round(target_distance / nominal_step_distance)))
        return int(min(max(1, forward_steps), max(1, 4 * int(self.tau_w))))

    def turn(self, angle: float, circle: bool = False):
        """Rotates the robot by the specified angle.

        Args:
            angle (float): Rotation angle in radians. Positive for counterclockwise, negative for clockwise.
            circle (bool, optional): If True, only right wheel moves, causing rotation around left wheel.
                   If False, wheels move in opposite directions. Defaults to False.
        """
        self.stop()
        self.move()
        l_offset = self.left_position_sensor.getValue()
        r_offset = self.right_position_sensor.getValue()
        self.sense()
        neg = -1.0 if (angle < 0.0) else 1.0
        if circle:
            self.left_motor.setVelocity(0)
        else:
            self.left_motor.setVelocity(neg * self.max_speed / 2)
        self.right_motor.setVelocity(-neg * self.max_speed / 2)
        while True:
            l = self.left_position_sensor.getValue() - l_offset
            r = self.right_position_sensor.getValue() - r_offset
            dl = l * self.wheel_radius
            dr = r * self.wheel_radius
            orientation = neg * (dl - dr) / self.axle_length
            self.sense()
            if not orientation < neg * angle:
                break
        self.stop()
        self.sense()

    def stop(self):
        """Stops the robot by setting both wheel velocities to zero.

        Sets both left and right motor velocities to 0, bringing the robot to a complete stop.
        """
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def move(self):
        """Updates motor positions and velocities based on current speed settings.

        Sets motor positions to infinity for continuous rotation and applies
        the current left_speed and right_speed values to the motors.

        Note:
            Position is set to infinity to allow continuous rotation rather than
            targeting a specific angle.
        """
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(self.left_speed)
        self.right_motor.setVelocity(self.right_speed)

    def compute_path_length(self):
        """
        Computes the total path length based on the agent's movement in the environment.

        Returns:
            float: Total path length computed from the differences in consecutive coordinates.
        """
        path_length = 0
        for i in range(self.hmap_loc[:, 0].shape[0] - 1):
            current_position = np.array([self.hmap_loc[:, 0][i], self.hmap_loc[:, 1][i]])
            next_position = np.array([self.hmap_loc[:, 0][i + 1], self.hmap_loc[:, 1][i + 1]])
            path_length += np.linalg.norm(next_position - current_position)

        return path_length

    def update_hmaps(self,
                    update_loc=False,
                    update_hdn=False,
                    update_pcn=False,
                    update_gcn=False,
                    update_scale_priority=False,
                    update_prox=False):
        """
        Store agent position, head direction activations, place cell activations,
        grid cell activations, scale priority (previously alpha), and proximity values.

        Parameters:
        - update_loc (bool): Whether to update agent location history.
        - update_hdn (bool): Whether to update head direction activations.
        - update_pcn (bool): Whether to update place cell activations.
        - update_gcn (bool): Whether to update grid cell activations.
        - update_scale_priority (bool): Whether to update scale priority (dominant scale index).
        - update_prox (bool): Whether to update proximity values.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # 1) Update agent location if requested
            if update_loc:
                planar = self._translation_to_planar_xy(curr_pos)
                self.hmap_loc[self.step_count] = np.array(
                    [planar[0], planar[1], self._translation_height(curr_pos)],
                    dtype=np.float32,
                )

            # 2) Update head direction activations (Direct assignment)
            if update_hdn:
                self.hmap_hdn[self.step_count] = self.hd_activations

            # 3) Dynamically resize hmap_pcn_activities if needed
            if (update_pcn and not self.lightweight_hmaps and
                len(self.hmap_pcn_activities) != len(self.pcn_activations_list)):
                self.hmap_pcn_activities = [
                    torch.zeros((self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32)
                    for act in self.pcn_activations_list
                ]

        # 4) Update place cell activations for each scale (Direct assignment)
        scale_idx_map = self.scale_history_index_by_scale_idx

        if update_pcn:
            for scale_def, act in zip(self.scales, self.pcn_activations_list):
                scale_idx = scale_def["scale_index"]

                # Ensure the scale index exists in the mapping
                if scale_idx not in scale_idx_map:
                    continue

                if self.lightweight_hmaps:
                    # Sample compact PC stats only (no dense per-cell history).
                    if self.step_count % self.hmap_sample_stride == 0:
                        act_t = act.detach().float()
                        k = min(self.hmap_topk, act_t.numel())
                        if k > 0:
                            top_vals, top_idx = torch.topk(act_t, k=k)
                            sample = {
                                "step": int(self.step_count),
                                "mean": float(act_t.mean().item()),
                                "max": float(act_t.max().item()),
                                "topk_idx": top_idx.to(device="cpu", dtype=torch.int64).numpy(),
                                "topk_val": top_vals.to(device="cpu", dtype=torch.float32).numpy(),
                            }
                            self.hmap_compact_stats["pcn"][scale_idx].append(sample)
                else:
                    mapped_index = scale_idx_map[scale_idx]  # Convert scale index to valid list index

                    # Ensure correct shape
                    if self.hmap_pcn_activities[mapped_index].shape[1] != act.shape[0]:
                        self.hmap_pcn_activities[mapped_index] = torch.zeros(
                            (self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32
                        )

                    # Store activations directly
                    self.hmap_pcn_activities[mapped_index][self.step_count] = act

        # 4.5) Update grid cell activations for each scale
        if update_gcn:
            for scale_def, act in zip(self.scales, self.grid_activations_list):
                if act is None:  # Skip if no grid cells for this scale
                    continue

                scale_idx = scale_def["scale_index"]

                # Ensure the scale index exists in the mapping
                if scale_idx not in scale_idx_map:
                    continue

                if self.lightweight_hmaps:
                    # Sample compact GC stats only (no dense per-cell history).
                    if self.step_count % self.hmap_sample_stride == 0:
                        act_t = act.detach().float()
                        k = min(self.hmap_topk, act_t.numel())
                        if k > 0:
                            top_vals, top_idx = torch.topk(act_t, k=k)
                            sample = {
                                "step": int(self.step_count),
                                "mean": float(act_t.mean().item()),
                                "max": float(act_t.max().item()),
                                "topk_idx": top_idx.to(device="cpu", dtype=torch.int64).numpy(),
                                "topk_val": top_vals.to(device="cpu", dtype=torch.float32).numpy(),
                            }
                            self.hmap_compact_stats["gcn"][scale_idx].append(sample)
                else:
                    mapped_index = scale_idx_map[scale_idx]  # Convert scale index to valid list index

                    # Ensure correct shape
                    if self.hmap_gcn_activities[mapped_index].shape[1] != act.shape[0]:
                        self.hmap_gcn_activities[mapped_index] = torch.zeros(
                            (self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32
                        )

                    # Store activations directly
                    self.hmap_gcn_activities[mapped_index][self.step_count] = act

        # 5) Update scale priority
        if update_scale_priority and hasattr(self, 'scale_idx'):
            self.hmap_scale_priority[self.step_count] = self.scale_idx

        # 6) Update proximity value if available
        if update_prox:
            if hasattr(self, "prox") and self.prox is not None:
                prox_value = float(self.prox)
            elif hasattr(self, "boundaries") and self.boundaries is not None:
                prox_value = float(torch.min(self.boundaries).item())
            else:
                prox_value = 0.0
            self.hmap_prox[self.step_count] = prox_value

        self.step_count += 1

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to the goal.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        # Get current position from the robot node
        curr_xy = self._current_planar_xy()

        # Distance from current position to goal location
        distance_to_goal = torch.norm(
            torch.tensor(
                [
                    curr_xy[0] - self.goal_location[0],
                    curr_xy[1] - self.goal_location[1],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Determine the correct goal radius based on the current mode
        if self.mode == RobotMode.EXPLOIT:
            goal_radius = self.goal_r["exploit"]
        else:  # Default to "explore" goal radius for all other modes
            goal_radius = self.goal_r["explore"]

        # Return 1.0 reward if within goal radius, else 0.0
        if distance_to_goal <= goal_radius:
            return 1.0  # Goal reached
        else:
            return 0.0

    def save(
        self,
        include_pcn: bool = False,
        include_rcn: bool = False,
        include_gcn: bool = False,
        include_hmaps: bool = False,
        save_trajectory: bool = False,
    ):
        """
        Saves:
        - PCN networks (one file per scale) if include_pcn=True
        - RCN networks (one file per scale) if include_rcn=True
        - GCN networks (one file per scale) if include_gcn=True
        - The history maps if include_hmaps=True
        - The agent's path if save_trajectory=True
        """
        files_saved = []
        save_start = time.perf_counter()

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1) Save each scale's PCN (if requested)
        # ----------------------------------------------------------------------
        if include_pcn:
            if self.use_unified_multiscale:
                pcn_path = os.path.join(self.network_dir, "unified_pcn.pkl")
                _dump_pickle_file(pcn_path, self.unified_pcn)
                files_saved.append(pcn_path)
            else:
                for scale_def, pcn in zip(self.scales, self.pcns):
                    scale_idx = scale_def["scale_index"]  # Get correct scale index
                    pcn_path = os.path.join(self.network_dir, f"pcn_scale_{scale_idx}.pkl")
                    _dump_pickle_file(pcn_path, pcn)
                    files_saved.append(pcn_path)

        # ----------------------------------------------------------------------
        # 2) Save each scale's RCN (if requested)
        # ----------------------------------------------------------------------
        if include_rcn:
            if self.use_unified_multiscale:
                rcn_path = os.path.join(self.network_dir, "unified_rcn.pkl")
                _dump_pickle_file(rcn_path, self.unified_rcn)
                files_saved.append(rcn_path)
            else:
                for scale_def, rcn in zip(self.scales, self.rcns):
                    scale_idx = scale_def["scale_index"]
                    rcn_path = os.path.join(self.network_dir, f"rcn_scale_{scale_idx}.pkl")
                    _dump_pickle_file(rcn_path, rcn)
                    files_saved.append(rcn_path)

        # ----------------------------------------------------------------------
        # 2.5) Save each scale's GCN (if requested)
        # ----------------------------------------------------------------------
        if include_gcn:
            for scale_def, gcn in zip(self.scales, self.gcns):
                if gcn is not None:
                    scale_idx = scale_def["scale_index"]
                    gcn_path = os.path.join(self.network_dir, f"gcn_scale_{scale_idx}.pkl")
                    _dump_pickle_file(gcn_path, gcn)
                    files_saved.append(gcn_path)

        # ----------------------------------------------------------------------
        # 3) Save the history maps if requested
        # ----------------------------------------------------------------------
        if include_hmaps:
            # Determine if we should use trial_id prefix (for multi-trial modes like EXPLOIT_LOCATIONS_RANDOM)
            # Check if we're in a multi-trial mode by seeing if hmaps_folder was explicitly provided
            use_trial_prefix = self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}

            if use_trial_prefix:
                trial_id = getattr(self, "trial_id", "default")
                prefix = f"{trial_id}_"
            else:
                prefix = ""

            # (a) Agent location
            hmap_loc_path = os.path.join(self.hmap_dir, f"{prefix}hmap_loc.pkl")
            _dump_pickle_file(hmap_loc_path, self.hmap_loc[: self.step_count])
            files_saved.append(hmap_loc_path)

            # (b) Head direction history
            hmap_hdn_path = os.path.join(self.hmap_dir, f"{prefix}hmap_hdn.pkl")
            _dump_pickle_file(hmap_hdn_path, self.hmap_hdn[: self.step_count].cpu())
            files_saved.append(hmap_hdn_path)

            # (c, d) Place/Grid activity logs
            if self.lightweight_hmaps:
                compact_path = os.path.join(self.hmap_dir, f"{prefix}hmap_compact_stats.pkl")
                compact_payload = {
                    "sample_stride": self.hmap_sample_stride,
                    "topk": self.hmap_topk,
                    "steps_recorded": int(self.step_count),
                    "pcn": self.hmap_compact_stats.get("pcn", {}),
                    "gcn": self.hmap_compact_stats.get("gcn", {}),
                }
                _dump_pickle_file(compact_path, compact_payload)
                files_saved.append(compact_path)
            else:
                # Full dense logs (legacy/high-detail mode)
                for scale_def, pc_history in zip(self.scales, self.hmap_pcn_activities):
                    scale_idx = scale_def["scale_index"]  # Get correct scale index
                    hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_pcn_scale_{scale_idx}.pkl")

                    pc_data = pc_history[: self.step_count].cpu().numpy()
                    _dump_pickle_file(hmap_scale_path, pc_data)
                    files_saved.append(hmap_scale_path)

                for scale_def, gc_history in zip(self.scales, self.hmap_gcn_activities):
                    if gc_history.numel() > 0:  # Only save if there are grid cells
                        scale_idx = scale_def["scale_index"]
                        hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_gcn_scale_{scale_idx}.pkl")

                        gc_data = gc_history[: self.step_count].cpu().numpy()
                        _dump_pickle_file(hmap_scale_path, gc_data)
                        files_saved.append(hmap_scale_path)

            # (e) Prox values
            if hasattr(self, "hmap_prox"):
                hmap_prox_path = os.path.join(self.hmap_dir, f"{prefix}hmap_prox.pkl")
                prox_data = self.hmap_prox[: self.step_count].cpu().numpy()
                _dump_pickle_file(hmap_prox_path, prox_data)
                files_saved.append(hmap_prox_path)

            # (f) Scale priority (only for multi-trial modes)
            if hasattr(self, "hmap_scale_priority") and use_trial_prefix:
                hmap_scale_priority_path = os.path.join(self.hmap_dir, f"{prefix}hmap_scale_priority.pkl")
                scale_priority_data = self.hmap_scale_priority[: self.step_count].cpu().numpy()
                _dump_pickle_file(hmap_scale_priority_path, scale_priority_data)
                files_saved.append(hmap_scale_priority_path)

            # (g) Run diagnostics for scale gating/recruitment behavior.
            hmap_diag_path = os.path.join(self.hmap_dir, f"{prefix}hmap_scale_diagnostics.pkl")
            diag_payload = self._build_scale_diagnostics_payload()
            _dump_pickle_file(hmap_diag_path, diag_payload)
            files_saved.append(hmap_diag_path)

        # ----------------------------------------------------------------------
        # 4) Save the agent's path if requested
        # ----------------------------------------------------------------------
        if save_trajectory:
            # Get world name and parse scale names
            scale_name_list = [scale["name"] for scale in self.scales]
            scale_order = ["small", "medium", "large", "xlarge"]
            scale_name_list = sorted(scale_name_list, key=lambda x: scale_order.index(x))
            scale_combination = "_".join(scale_name_list)

            # Define the correct base directory
            base_stats_dir = os.path.join(PROJECT_ROOT, "analysis", "stats", self.world_name, scale_combination)
            hmaps_path_dir = os.path.join(base_stats_dir, "hmaps")
            os.makedirs(hmaps_path_dir, exist_ok=True)

            # Trial ID-based filename
            trial_id = getattr(self, "trial_id", "default")
            hmap_loc_file = os.path.join(hmaps_path_dir, f"{trial_id}_hmap_loc.pkl")
            hmap_scale_priority_file = os.path.join(hmaps_path_dir, f"{trial_id}_hmap_scale_priority.pkl")

            _dump_pickle_file(hmap_loc_file, self.hmap_loc[:self.step_count])
            files_saved.append(hmap_loc_file)

            _dump_pickle_file(
                hmap_scale_priority_file,
                self.hmap_scale_priority[: self.step_count].cpu().numpy(),
            )
            files_saved.append(hmap_scale_priority_file)

            print(f"Saved path data for trial {trial_id} in {scale_combination}.")

        # ----------------------------------------------------------------------
        # 5) Print saved files
        # ----------------------------------------------------------------------
        # Show messagebox only if:
        # - No stats collector AND
        # - (Not in AUTO mode OR this is the last trial)
        auto_modes = {RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                      RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
                      RobotMode.PLOTTING_AUTO,
                      RobotMode.PLOTTING_COVERAGE_AUTO}
        show_messagebox = (not self.stats_collector and
                          (self.robot_mode not in auto_modes or
                           self.current_auto_trial == self.num_auto_trials))

        if show_messagebox:
             root = tk.Tk()
             root.withdraw()
             root.attributes("-topmost", True)
             root.update()
             messagebox.showinfo("Information", "Press OK to save data")
             root.destroy()

        print(f"Files Saved: {files_saved}")
        print(f"Saving Done! ({time.perf_counter() - save_start:.2f}s)")

    def clear(self):
        """
        Removes all scale-specific PCN/RCN/GCN files and any hmap files.

        This includes:
        - pcn_scale_*.pkl
        - rcn_scale_*.pkl
        - gcn_scale_*.pkl
        - pcn.pkl, rcn.pkl, gcn.pkl (legacy)
        - hmap_* files in self.hmap_dir
        """
        # 1. Remove all per-scale PCN/RCN/GCN files
        if os.path.exists(self.network_dir):
            for fname in os.listdir(self.network_dir):
                # Delete pcn_scale_*, rcn_scale_*, or gcn_scale_* files
                if (fname.startswith("pcn_scale_") or
                    fname.startswith("rcn_scale_") or
                    fname.startswith("gcn_scale_") or
                    fname in {"unified_pcn.pkl", "unified_rcn.pkl"}):
                    full_path = os.path.join(self.network_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

            # 2. Remove the old single-scale files if present
            for legacy_file in ["pcn.pkl", "rcn.pkl", "gcn.pkl"]:
                path = os.path.join(self.network_dir, legacy_file)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed: {path}")
                    except FileNotFoundError:
                        pass

        # 3. Remove any scale-specific hmap files (and all hmap files) from self.hmap_dir
        if os.path.exists(self.hmap_dir):
            for fname in os.listdir(self.hmap_dir):
                # For example: hmap_scale_0.pkl, hmap_scale_1.pkl, or any other hmap_*
                if fname.startswith("hmap_"):
                    full_path = os.path.join(self.hmap_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

        print("[DRIVER] Finished clearing old scale PCNs, RCNs, GCNs, and hmap files.")
