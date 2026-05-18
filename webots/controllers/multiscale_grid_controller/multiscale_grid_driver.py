import numpy as np
from numpy.random import default_rng
import pickle
import os
import json
import tkinter as tk
from tkinter import N, messagebox
from typing import Optional, List, Dict, Any, Sequence
import torch
from controller import Supervisor
from astropy.stats import circmean
import random
import math
import copy
import re

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from preplay_defaults import UNIFIED_PREPLAY_DEFAULTS

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.multiscale_pcn import PlaceCellLayer
from core.layers.multiscale_pcn_with_gcn import MultiscalePlaceCellWithGrid
from core.layers.multiscale_pcn_with_gcn_v2 import UnifiedMultiScalePCN
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer_test import C_LAMBDA
from core.layers.unified_reward_cell import (
    DEFAULT_REPLAY_LONG_TAU,
    DEFAULT_REPLAY_LONG_WEIGHT,
    DEFAULT_REPLAY_TAU,
    DEFAULT_REPLAY_TIMESTEPS,
    UnifiedRewardCell,
)
from core.robot.robot_mode import RobotMode
from core.robot.webots_worlds import (
    build_goal_config_from_world,
    get_world_agent_start,
    get_world_config,
    get_world_size,
    obstacle_distance,
)
from analysis.stats.stats_collector import stats_collector

# Replay step budget is proportional to lambda_s to normalize spread per time constant
STEPS_PER_LAMBDA = 8  # Adjust to push farther (higher) or be more local (lower)

def _steps_for_scale(scale_def: Dict[str, Any]) -> int:
    """Derive custom replay timesteps proportional to lambda_s for this scale."""
    sigma_pc_s = scale_def.get("sigma_pc_s", scale_def.get("sigma_r", 1.0))
    return int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA))


class UnifiedPCNSliceView:
    """Lightweight per-scale view over a unified PCN."""

    def __init__(self, unified_pcn: UnifiedMultiScalePCN, scale_idx: int):
        self._pcn = unified_pcn
        self.scale_idx = int(scale_idx)
        self.exploit_preplay_supported = False
        self.reward_replay_supported = False

    def _bounds(self) -> tuple[int, int]:
        return self._pcn.scale_boundaries[self.scale_idx : self.scale_idx + 2]

    @property
    def num_pc(self) -> int:
        start, end = self._bounds()
        return int(end - start)

    @property
    def place_cell_activations(self) -> torch.Tensor:
        start, end = self._bounds()
        return self._pcn.place_cell_activations[start:end]

    @place_cell_activations.setter
    def place_cell_activations(self, value) -> None:
        start, end = self._bounds()
        self._pcn.place_cell_activations[start:end] = value

    @property
    def w_rec_tripartite(self) -> torch.Tensor:
        start, end = self._bounds()
        return self._pcn.w_rec_unified[:, start:end, start:end]

    @property
    def bvc_layer(self):
        return self._pcn.bvc_layers[self.scale_idx]

    def __getattr__(self, name):
        return getattr(self._pcn, name)

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
        stats_collector: Optional[stats_collector] = None,
        trial_id: Optional[str] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        goal_config: Optional[Dict[str, Any]] = None,
        trial_config: Optional[Dict[str, Any]] = None,
        max_dist: Optional[float] = None,
        plot_bvc: Optional[bool] = False,
        td_learning: Optional[bool] = False,
        use_prox_mod: Optional[bool] = False,
        environment_size: Optional[List[float]] = None,
        grid_size: Optional[float] = None,
        coverage_percentage: Optional[float] = None,
        min_goal_visits: int = 3,
        unified_recurrent_preplay_horizon: int = UNIFIED_PREPLAY_DEFAULTS["unified_recurrent_preplay_horizon"],
        unified_preplay_discount_factor: float = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
        unified_preplay_within_direction_beta: float = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
        unified_preplay_num_samples: int = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
        unified_preplay_sampling_temperature: float = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
        unified_preplay_turn_offsets: Optional[List[int]] = None,
        unified_preplay_normalize_transitions: bool = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_normalize_transitions"],
        unified_preplay_global_score_normalization: bool = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_global_score_normalization"],
        unified_preplay_executable_rollouts: bool = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_executable_rollouts"],
        unified_preplay_boundary_mode: str = UNIFIED_PREPLAY_DEFAULTS["unified_preplay_boundary_mode"],
        unified_preplay_microtrajectory_safety_margin: float = UNIFIED_PREPLAY_DEFAULTS[
            "unified_preplay_microtrajectory_safety_margin"
        ],
        unified_preplay_blocked_return_penalty: float = UNIFIED_PREPLAY_DEFAULTS[
            "unified_preplay_blocked_return_penalty"
        ],
        unified_preplay_no_reward_threshold: float = UNIFIED_PREPLAY_DEFAULTS[
            "unified_preplay_no_reward_threshold"
        ],
        unified_preplay_decision_diagnostics: bool = UNIFIED_PREPLAY_DEFAULTS[
            "unified_preplay_decision_diagnostics"
        ],
        unified_preplay_decision_diagnostics_stride: int = UNIFIED_PREPLAY_DEFAULTS[
            "unified_preplay_decision_diagnostics_stride"
        ],
        stdp_rectified_scale_centering: bool = False,
        stdp_rectified_hd_gate: bool = True,
        stdp_winner_hd_gate: bool = False,
        stdp_min_input_mass: float = 0.0,
        exploit_bvc_context_gaussian_modulation: bool = True,
        exploit_cross_scale_inhibition: bool = False,
        optimal_path_distance: Optional[float] = None,
        path_failure_ratio: Optional[float] = None,
        paths_folder: Optional[str] = None,
        hmaps_folder: Optional[str] = None,
        auto_trial_name: Optional[str] = None,
        num_auto_trials: int = 5,
        current_auto_trial: int = 1,
        exploit_debug_logging: bool = False,
        exploit_hd_score_logging: bool = False,
        exploit_loop_recovery: bool = UNIFIED_PREPLAY_DEFAULTS["exploit_loop_recovery"],
        debug_heading_convention_test: bool = False,
        debug_heading_test_forward_steps: int = 4,
        debug_heading_test_start_loc: Optional[List[float]] = None,
    ):
        """
        Initializes the Driver class, setting up the robot's sensors and neural networks.
        """
        if mode == RobotMode.DMTP:
            print("[DRIVER] DMTP selected; redirecting to LEARN_LOCATIONS_COVERAGE")
            mode = RobotMode.LEARN_LOCATIONS_COVERAGE

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
        else:
            world_path = self.getWorldPath()
        self.world_name = world_name
        self.world_file_path = world_path
        world_compass_cfg = self._load_world_compass_config(self.world_file_path)
        self.world_coordinate_system = world_compass_cfg["coordinate_system"]
        self.world_sim_release = world_compass_cfg["sim_release"]
        self.compass_device_frame = world_compass_cfg["compass_device_frame"]
        self.world_up_axis_index = int(world_compass_cfg["up_axis_index"])
        self.world_north_axis_index = int(world_compass_cfg["north_axis_index"])
        self.world_east_axis_index = int(world_compass_cfg["east_axis_index"])
        self.world_north_direction = world_compass_cfg["north_direction"]
        self.world_bearing_axis_labels = tuple(world_compass_cfg["bearing_axis_labels"])

        if environment_size is None:
            try:
                environment_size = get_world_size(self.world_name)
            except Exception:
                environment_size = None

        if start_loc is None:
            try:
                start_loc = get_world_agent_start(self.world_name)
            except Exception:
                start_loc = None

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

        self.unified_pcn_mode = True
        self.unified_pcn_path = os.path.join(self.network_dir, "pcn_unified.pkl")
        self.unified_rcn_path = os.path.join(self.network_dir, "unified_rcn_goal.pkl")
        self.unified_pcn_bundle = None
        self.global_w_rec_tripartite = None
        self.pcn_scale_slices = []

        # Stats / trial info
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        # Head direction layer size
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 8
        self.max_dist = max_dist
        self._diag_default_clearance = float(max_dist) if max_dist is not None else 0.0
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756

        # Simulation run time
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))

        # Exploration/exploitation radius
        self.goal_r = {"explore": 0.3, "exploit": 0.5}
        self.goal_location = goal_location if goal_location else [-3, 3]
        self.start_loc = start_loc

        # Default single scale if none provided
        if not scales:
            print("[DRIVER] No scales provided, using default scale")
            scales = [{
                "name": "default_scale",
                "num_pc": 500,
                "sigma_r": 0.5,
                "sigma_theta": 1.0,
                "gamma_cross": 0.5,
            }]
        self.scales = scales
        
        self.td_learning = td_learning
        self.use_prox_mod = use_prox_mod
        self.unified_recurrent_preplay_horizon = int(max(1, unified_recurrent_preplay_horizon))
        self.unified_preplay_discount_factor = float(unified_preplay_discount_factor)
        self.unified_preplay_within_direction_beta = float(unified_preplay_within_direction_beta)
        self.unified_preplay_num_samples = int(max(1, unified_preplay_num_samples))
        self.unified_preplay_sampling_temperature = float(max(1e-6, unified_preplay_sampling_temperature))
        turn_offsets = (
            list(unified_preplay_turn_offsets)
            if unified_preplay_turn_offsets is not None
            else list(UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"])
        )
        if len(turn_offsets) == 0:
            raise ValueError("unified_preplay_turn_offsets must contain at least one turn offset")
        self.unified_preplay_turn_offsets = [int(offset) for offset in turn_offsets]
        self.unified_preplay_normalize_transitions = bool(unified_preplay_normalize_transitions)
        self.unified_preplay_global_score_normalization = bool(
            unified_preplay_global_score_normalization
        )
        self.unified_preplay_executable_rollouts = bool(unified_preplay_executable_rollouts)
        boundary_mode = str(
            unified_preplay_boundary_mode
            or UNIFIED_PREPLAY_DEFAULTS["unified_preplay_boundary_mode"]
        ).strip().lower()
        if boundary_mode not in {"hard_block", "none"}:
            raise ValueError(
                "unified_preplay_boundary_mode must be one of "
                "'hard_block' or 'none'"
            )
        self.unified_preplay_boundary_mode = boundary_mode
        self.unified_preplay_microtrajectory_safety_margin = float(
            max(0.0, unified_preplay_microtrajectory_safety_margin)
        )
        self.unified_preplay_blocked_return_penalty = float(unified_preplay_blocked_return_penalty)
        self.unified_preplay_no_reward_threshold = float(
            max(0.0, unified_preplay_no_reward_threshold)
        )
        self.unified_preplay_decision_diagnostics = bool(unified_preplay_decision_diagnostics)
        self.unified_preplay_decision_diagnostics_stride = int(
            max(1, unified_preplay_decision_diagnostics_stride)
        )
        self._preplay_decision_diagnostics_path = None
        self._preplay_decision_diagnostics_counter = 0
        self.stdp_rectified_scale_centering = bool(stdp_rectified_scale_centering)
        self.stdp_rectified_hd_gate = bool(stdp_rectified_hd_gate)
        self.stdp_winner_hd_gate = bool(stdp_winner_hd_gate)
        self.stdp_min_input_mass = float(max(0.0, stdp_min_input_mass))
        self.exploit_bvc_context_gaussian_modulation = bool(
            exploit_bvc_context_gaussian_modulation
        )
        self.exploit_cross_scale_inhibition = bool(exploit_cross_scale_inhibition)
        self.stdp_transition_quality_gate_enabled = True
        self.exploit_lidar_heading_blocker_enabled = True
        self.exploit_debug_log_interval = 100
        self._last_exploit_collision_log_step = -10**9
        self._last_exploit_hd_score_log_step = -10**9
        self.exploit_debug_logging = bool(exploit_debug_logging)
        self.exploit_hd_score_logging = bool(exploit_hd_score_logging)
        self.exploit_loop_recovery = bool(exploit_loop_recovery)
        self.debug_heading_convention_test = bool(debug_heading_convention_test)
        self.debug_heading_test_forward_steps = int(max(1, debug_heading_test_forward_steps))
        self.debug_heading_test_start_loc = (
            list(debug_heading_test_start_loc)
            if debug_heading_test_start_loc is not None
            else None
        )
        self._debug_heading_convention_test_ran = False
        self._suppress_goal_checks_for_heading_test = False

        # Store coverage parameters
        self.environment_size = environment_size
        self.grid_size = grid_size
        self.coverage_percentage = coverage_percentage
        self.min_goal_visits = min_goal_visits
        self.goal_visit_cooldown_s = 6.0
        self.goal_exit_hysteresis = 0.1
        self.goal_events_by_goal = {}
        self._current_goal_event_by_goal = {}
        self.goal_visit_counts = {}
        self.goal_currently_in = {}
        self.goal_last_count_time_s_by_goal = {}

        # Store random spawn parameters
        self.optimal_path_distance = optimal_path_distance
        self.path_failure_ratio = path_failure_ratio
        self.paths_folder = paths_folder

        # Initialize distance tracking for random spawn mode
        self.total_distance_traveled = 0.0
        self.last_position = None
        self.current_compass_values = [0.0, 0.0, 0.0]
        self.current_compass_heading_deg = 0.0
        self.current_compass_heading_deg_exact = 0.0
        self.current_heading_deg = 0.0
        self.current_heading_deg_exact = 0.0
        self.current_heading_deg_legacy = 0.0

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
                    0,
                    random.uniform(-2.3, 2.3)
                ]
                # Check against all goals in unified system
                min_dist = float('inf')
                for goal in self.goals:
                    dist = np.sqrt(
                        (candidate[0] - goal["location"][0]) ** 2 +
                        (candidate[2] - goal["location"][1]) ** 2
                    )
                    min_dist = min(min_dist, dist)
                if min_dist >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(candidate)
            self.robot.resetPhysics()
        else:
            if self.start_loc is not None:
                self.robot.getField("translation").setSFVec3f([self.start_loc[0], 0, self.start_loc[1]])
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

        # Load or init Grid Cell Networks / PCNs / RCNs
        self.gcns = []
        self.init_grid_cell_networks()
        self.pcns = []
        self.load_pcns(enable_ojas, enable_stdp)
        self.load_rcns()
        pcn_files = [os.path.basename(self.unified_pcn_path)] if self.unified_pcn_mode else [
            f"pcn_scale_{scale_def['scale_index']}.pkl" for scale_def in self.scales
        ]
        rcn_files = [os.path.basename(self.unified_rcn_path)]
        print(f"[DRIVER] Using PCNs: {pcn_files}")
        print(f"[DRIVER] Using RCNs: {rcn_files}")

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Initialize alpha as a tensor of zeros with the same length as the number of scales
        self.alpha = torch.zeros(len(self.scales), dtype=self.dtype, device=self.device)

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        self.hmap_scale_priority = torch.zeros(
            (self.num_steps, len(self.scales)),  # row per step, col per scale
            device="cuda", 
            dtype=torch.float32
        )
        self.hmap_learning_diagnostics = {
            "time_s": np.zeros(self.num_steps, dtype=np.float32),
            "x": np.zeros(self.num_steps, dtype=np.float32),
            "z": np.zeros(self.num_steps, dtype=np.float32),
            "heading_deg": np.zeros(self.num_steps, dtype=np.float32),
            "heading_delta_deg": np.zeros(self.num_steps, dtype=np.float32),
            "step_displacement": np.zeros(self.num_steps, dtype=np.float32),
            # Latched over the interval since the previous hmap write.
            "min_lidar_distance": np.zeros(self.num_steps, dtype=np.float32),
            "min_lidar_distance_current": np.zeros(self.num_steps, dtype=np.float32),
            # Latched over the interval since the previous hmap write.
            "collision_any": np.zeros(self.num_steps, dtype=np.int8),
            "collision_left": np.zeros(self.num_steps, dtype=np.int8),
            "collision_right": np.zeros(self.num_steps, dtype=np.int8),
            "collision_any_current": np.zeros(self.num_steps, dtype=np.int8),
            "collision_left_current": np.zeros(self.num_steps, dtype=np.int8),
            "collision_right_current": np.zeros(self.num_steps, dtype=np.int8),
            "stdp_transition_eligibility": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_displacement": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_heading_error_deg": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_forward_progress": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_wheel_travel": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_progress_ratio": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_transition_collision_latched": np.zeros(self.num_steps, dtype=np.int8),
            "stdp_transition_collision_current": np.zeros(self.num_steps, dtype=np.int8),
            "stdp_connection_decay_scale": np.zeros(self.num_steps, dtype=np.float32),
            "stdp_updated": np.zeros(self.num_steps, dtype=np.int8),
            "pcn_learning_step_count": np.zeros(self.num_steps, dtype=np.int32),
        }
        self._last_diag_position = None
        self._last_diag_heading_deg = None
        self._diag_collision_latch = np.zeros(2, dtype=np.int8)
        self.current_min_lidar_distance = float(self._diag_default_clearance)
        self._diag_min_lidar_since_log = float(self._diag_default_clearance)
        self.last_stdp_transition_eligibility = 0.0
        self.last_stdp_transition_displacement = 0.0
        self.last_stdp_transition_heading_error_deg = 0.0
        self.last_stdp_transition_forward_progress = 0.0
        self.last_stdp_transition_wheel_travel = 0.0
        self.last_stdp_transition_progress_ratio = 0.0
        self.last_stdp_transition_collision_latched = False
        self.last_stdp_transition_collision_current = False
        self._last_diag_left_wheel_position = None
        self._last_diag_right_wheel_position = None

        # For multi-scale place cell logs
        self.hmap_pcn_activities = []
        for scale_def in self.scales:
            n_pc = scale_def["num_pc"]
            self.hmap_pcn_activities.append(
                torch.zeros((self.num_steps, n_pc), device=self.device, dtype=torch.float32)
            )

        # For multi-scale grid cell logs
        self.hmap_gcn_activities = []
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
        self.last_execute_movement_collision = False
        
        # Legacy exploit_v12 state retained only for backwards compatibility.
        self.force_explore_count = 0
        self.scale_reliability = None
        self.last_scale_weights = None
        self.loop_scale_contributions = None
        self.exploit_loop_history = []
        self.exploit_loop_history_size = max(8, 2 * int(self.n_hd))
        self.exploit_loop_forced_explore_steps = 12
        self.exploit_loop_cooldown = 0
        self._exploit_decode_hmap_cache = None
        self._exploit_decode_status_logged = False

        # Keep direct unified references available throughout the driver.
        self.pcn = getattr(self, "pcn", None)
        self.rcn = getattr(self, "rcn", None)

        # For EXPLOIT_LOCATIONS_RANDOM, load goal-specific RCNs
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO} and hasattr(self, 'active_goal_name'):
            self._load_goal_specific_rcns(self.active_goal_name)

        self.plot_bvc = plot_bvc

        # Step once
        self.step(self.timestep)

        # Track trial start time for per-trial timeouts
        self.trial_start_time = self.getTime()
        print(f"[DRIVER] Trial started at simulation time: {self.trial_start_time:.1f}s")

    ##########################################################################
    #                           PCN / RCN LOADING                            #
    ##########################################################################
    def load_pcns(self, enable_ojas: Optional[bool], enable_stdp: Optional[bool]):
        self.unified_pcn_bundle = None
        self.global_w_rec_tripartite = None
        self.pcn_scale_slices = []

        pcn = self._load_or_init_unified_pcn(enable_ojas, enable_stdp)
        self.pcn = pcn
        self.pcns = [UnifiedPCNSliceView(self.pcn, idx) for idx in range(self.pcn.num_scales)]
        self.global_w_rec_tripartite = self.pcn.w_rec_unified
        self.pcn_scale_slices = [
            tuple(self.pcn.scale_boundaries[i : i + 2]) for i in range(self.pcn.num_scales)
        ]
        self.unified_pcn_bundle = self.pcn
        print(
            f"[DRIVER] Using unified PCN with {self.pcn.num_scales} scales and "
            f"{self.pcn.num_pc_total} total place cells"
        )

    def _apply_unified_ablation_settings(self, pcn: UnifiedMultiScalePCN) -> UnifiedMultiScalePCN:
        # Keep modulation config aligned with the current scale definitions when
        # reusing a saved unified PCN pickle.
        if hasattr(pcn, "scale_configs") and isinstance(getattr(pcn, "scale_configs", None), list):
            for cfg_idx, current_cfg in enumerate(self.scales):
                if cfg_idx >= len(pcn.scale_configs):
                    break
                pcn_cfg = pcn.scale_configs[cfg_idx]
                for key in ("name", "sigma_r", "mod_d_opt", "mod_sigma", "d_opt", "sigma_tune_k"):
                    if key in current_cfg:
                        pcn_cfg[key] = current_cfg[key]

        # Use proximity-based scale gating during learning. In exploit it is
        # controlled explicitly so gaussian modulation can be tested before
        # adding cross-scale inhibition.
        exploit_mode = self._is_exploit_mode()
        context_mode = (
            "bvc_context"
            if (not exploit_mode or self.exploit_bvc_context_gaussian_modulation)
            else "none"
        )
        pcn.bvc_context_modulation_mode = pcn._normalize_bvc_context_modulation_mode(context_mode)
        pcn.bvc_context_gain_floor = 0.0
        pcn.bvc_context_gain_strength = 1.0
        pcn.bvc_excitation_modulation_floor = 0.0
        pcn.grid_inhibition_mode = pcn._normalize_grid_inhibition_mode("sum")
        pcn.learning_stdp_start_steps = 0
        pcn.stdp_rectified_scale_centering = self.stdp_rectified_scale_centering
        pcn.stdp_rectified_hd_gate = self.stdp_rectified_hd_gate
        pcn.stdp_winner_hd_gate = self.stdp_winner_hd_gate
        pcn.stdp_min_input_mass = self.stdp_min_input_mass
        pcn.enforce_grid_structural_mask = False
        pcn.clamp_afferent_weights_nonnegative = False
        pcn.disable_cross_scale_inhibition = bool(
            exploit_mode and not self.exploit_cross_scale_inhibition
        )
        pcn.preplay_normalize_transitions = bool(self.unified_preplay_normalize_transitions)
        pcn.preplay_global_score_normalization = bool(
            self.unified_preplay_global_score_normalization
        )
        pcn._invalidate_preplay_transition_cache()
        pcn.cache_preplay_transitions = bool(self._is_exploit_mode())
        pcn.enable_live_diagnostics = bool(
            getattr(self, "exploit_debug_logging", False)
            and self._is_exploit_mode()
        )
        pcn.configure_connection_decay_rates(1e-4)
        pcn.configure_eta_stdp(0.3)
        pcn.exploit_preplay_supported = False
        pcn.reward_replay_supported = False
        return pcn

    def _load_world_compass_config(self, world_path: Optional[str]) -> Dict[str, Any]:
        coordinate_system = "NUE"
        north_direction = None
        sim_release = "unknown"

        if world_path:
            try:
                world_text = Path(world_path).read_text(encoding="utf-8", errors="ignore")
                release_match = re.search(r"#VRML_SIM\s+(R\d{4}[ab])", world_text)
                if release_match:
                    sim_release = release_match.group(1)
                coordinate_match = re.search(r'coordinateSystem\s+"([A-Z]{3})"', world_text)
                if coordinate_match:
                    coordinate_system = coordinate_match.group(1).upper()
                north_match = re.search(
                    r"northDirection\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
                    world_text,
                )
                if north_match:
                    north_direction = [float(north_match.group(i)) for i in range(1, 4)]
            except OSError as exc:
                print(f"[WARN] Failed to read world file for compass config: {world_path} ({exc})")

        if coordinate_system not in {"ENU", "NUE", "EUN"}:
            print(f"[WARN] Unsupported coordinateSystem '{coordinate_system}', falling back to NUE")
            coordinate_system = "NUE"

        north_axis_index = coordinate_system.index("N")
        east_axis_index = coordinate_system.index("E")
        up_axis_index = coordinate_system.index("U")
        if north_direction is None:
            north_direction = [0.0, 0.0, 0.0]
            north_direction[north_axis_index] = 1.0

        axis_labels = {0: "x", 1: "y", 2: "z"}
        compass_device_frame = "FLU" if self._webots_release_uses_flu(sim_release) else "LEGACY_Y_UP"
        return {
            "coordinate_system": coordinate_system,
            "sim_release": sim_release,
            "compass_device_frame": compass_device_frame,
            "north_axis_index": north_axis_index,
            "east_axis_index": east_axis_index,
            "up_axis_index": up_axis_index,
            "north_direction": north_direction,
            "bearing_axis_labels": (
                axis_labels[north_axis_index],
                axis_labels[east_axis_index],
            ),
        }

    @staticmethod
    def _normalize_heading_deg(angle_deg: float) -> float:
        angle = float(angle_deg) % 360.0
        return angle + 360.0 if angle < 0.0 else angle

    @classmethod
    def _compass_heading_to_world_heading_deg(cls, compass_heading_deg: float) -> float:
        return cls._normalize_heading_deg(float(compass_heading_deg) + 90.0)

    @classmethod
    def _world_heading_to_controller_heading_deg(cls, world_heading_deg: float) -> float:
        return cls._normalize_heading_deg(float(world_heading_deg) - 90.0)

    def _world_heading_to_hd_index(self, world_heading_deg: float) -> int:
        """Map a world/math heading onto the nearest internal HD direction index."""
        step_deg = 360.0 / float(max(1, self.n_hd))
        return int(round(self._normalize_heading_deg(world_heading_deg) / step_deg)) % int(self.n_hd)

    def _current_controller_heading_deg(self) -> float:
        return float(
            getattr(
                self,
                "current_compass_heading_deg",
                self._world_heading_to_controller_heading_deg(
                    getattr(self, "current_heading_deg", 0.0)
                ),
            )
        )

    @staticmethod
    def _webots_release_uses_flu(sim_release: str) -> bool:
        match = re.fullmatch(r"R(\d{4})([ab])", str(sim_release).strip())
        if not match:
            return True
        year = int(match.group(1))
        suffix = match.group(2)
        return (year, suffix) >= (2022, "a")

    @classmethod
    def _signed_heading_error_deg(cls, target_heading_deg: float, source_heading_deg: float) -> float:
        diff = cls._normalize_heading_deg(target_heading_deg) - cls._normalize_heading_deg(source_heading_deg)
        return (diff + 180.0) % 360.0 - 180.0

    @staticmethod
    def _current_position_xz(curr_pos: Sequence[float]) -> tuple[float, float]:
        return float(curr_pos[0]), float(curr_pos[2])

    def _heading_from_xz_vector(self, dx: float, dz: float) -> float:
        return self._normalize_heading_deg(math.degrees(math.atan2(float(dz), float(dx))))

    def _vector_heading_deg(
        self,
        vector: torch.Tensor,
        fallback_hd: Optional[int] = None,
    ) -> float:
        if vector is None or int(vector.numel()) < 2:
            fallback = 0 if fallback_hd is None else int(fallback_hd)
            return self._normalize_heading_deg(
                fallback * (360.0 / float(max(1, self.n_hd)))
            )

        vx = float(vector[0].item())
        vz = float(vector[1].item())
        if math.hypot(vx, vz) < 1e-6:
            fallback = 0 if fallback_hd is None else int(fallback_hd)
            return self._normalize_heading_deg(
                fallback * (360.0 / float(max(1, self.n_hd)))
            )
        return self._normalize_heading_deg(math.degrees(math.atan2(vz, vx)))

    def _exploit_forward_primitive_distance(self) -> float:
        """Distance implied by one exploit forward primitive under current timing."""
        motor_steps = max(1, 2 * int(getattr(self, "tau_w", 1)) - 1)
        return float(
            abs(float(getattr(self, "max_speed", 0.0)))
            * float(getattr(self, "wheel_radius", 0.0))
            * (float(getattr(self, "timestep", 0.0)) / 1000.0)
            * float(motor_steps)
        )

    def _should_log_exploit_detail(self) -> bool:
        if not bool(getattr(self, "exploit_debug_logging", False)):
            return False
        interval = int(max(1, getattr(self, "exploit_debug_log_interval", 10)))
        return int(getattr(self, "step_count", 0)) % interval == 0

    def _should_log_exploit_hd_scores(self) -> bool:
        if not bool(getattr(self, "exploit_hd_score_logging", False)):
            return False
        if not self._is_exploit_mode():
            return False
        interval = int(max(1, getattr(self, "exploit_debug_log_interval", 10)))
        step_count = int(getattr(self, "step_count", 0))
        last_step = int(getattr(self, "_last_exploit_hd_score_log_step", -10**9))
        if step_count - last_step < interval:
            return False
        self._last_exploit_hd_score_log_step = step_count
        return True

    def _should_log_exploit_collision(self) -> bool:
        if not bool(getattr(self, "exploit_debug_logging", False)) or not self._is_exploit_mode():
            return False
        interval = int(max(1, getattr(self, "exploit_debug_log_interval", 10)))
        step_count = int(getattr(self, "step_count", 0))
        last_step = int(getattr(self, "_last_exploit_collision_log_step", -10**9))
        if step_count - last_step < interval:
            return False
        self._last_exploit_collision_log_step = step_count
        return True

    def _lidar_clearance_for_world_heading(self, world_heading_deg: float) -> float:
        boundaries = torch.as_tensor(
            getattr(self, "boundaries", torch.empty(0)),
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1)
        count = int(boundaries.numel())
        if count <= 0:
            return float(getattr(self, "max_dist", self._diag_default_clearance))

        # The world-aligned boundary vector is stored opposite the math heading
        # convention used by navigation: index 0 reads world 180, index 180 reads
        # world 0. Flip at this boundary so HD masks use navigation headings.
        heading = self._normalize_heading_deg(world_heading_deg + 180.0)
        idx = int(round((heading / 360.0) * float(count))) % count
        clearance = float(boundaries[idx].item())
        if not math.isfinite(clearance) or clearance <= 0.0:
            return float(getattr(self, "max_dist", self._diag_default_clearance))
        return clearance

    def _lidar_obstacle_points_world_local(self) -> torch.Tensor:
        boundaries = torch.as_tensor(
            getattr(self, "boundaries", torch.empty(0)),
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1)
        count = int(boundaries.numel())
        if count <= 0:
            return torch.empty((0, 2), dtype=self.dtype, device=self.device)

        max_dist = float(getattr(self, "max_dist", self._diag_default_clearance) or 0.0)
        finite = torch.isfinite(boundaries)
        positive = boundaries > 1e-6
        if max_dist > 0.0:
            obstacle = boundaries < (0.995 * max_dist)
        else:
            obstacle = positive
        valid = finite & positive & obstacle
        if not bool(torch.any(valid).item()):
            return torch.empty((0, 2), dtype=self.dtype, device=self.device)

        idx = torch.arange(count, dtype=self.dtype, device=self.device)[valid]
        distances = boundaries[valid]
        # Inverse of _lidar_clearance_for_world_heading(): boundary index 0 is
        # world 180 deg, so convert back into world/math local coordinates.
        headings = ((idx / float(count)) * 360.0 - 180.0) * (math.pi / 180.0)
        return torch.stack(
            [distances * torch.cos(headings), distances * torch.sin(headings)],
            dim=1,
        )

    def _build_exploit_microtrajectory_blocker(self, step_distance: float):
        obstacle_points = self._lidar_obstacle_points_world_local()
        step_distance = float(max(0.0, step_distance))
        margin = float(max(0.0, getattr(self, "unified_preplay_microtrajectory_safety_margin", 0.0)))
        footprint_radius = 0.5 * float(max(0.0, getattr(self, "axle_length", 0.0)))
        safety_radius = max(0.03, footprint_radius + margin)
        safety_radius_sq = safety_radius * safety_radius
        max_dist = float(getattr(self, "max_dist", self._diag_default_clearance) or 0.0)

        def _blocked(
            positions: torch.Tensor,
            direction_ids: torch.Tensor,
            step_idx: int,
        ) -> torch.Tensor:
            positions_t = torch.as_tensor(positions, dtype=self.dtype, device=self.device).view(-1, 2)
            dirs_t = torch.as_tensor(direction_ids, dtype=torch.long, device=self.device).view(-1)
            if int(dirs_t.numel()) != int(positions_t.shape[0]):
                raise ValueError("positions and direction_ids must have the same batch size")

            angles = dirs_t.to(dtype=self.dtype) * (2.0 * math.pi / float(max(1, self.n_hd)))
            deltas = step_distance * torch.stack(
                [torch.cos(angles), torch.sin(angles)],
                dim=1,
            )
            starts = positions_t
            ends = starts + deltas
            seg = ends - starts
            seg_len_sq = torch.sum(seg * seg, dim=1).clamp(min=1e-12)

            if int(obstacle_points.numel()) > 0:
                rel = obstacle_points.unsqueeze(0) - starts.unsqueeze(1)
                t = torch.sum(rel * seg.unsqueeze(1), dim=2) / seg_len_sq.view(-1, 1)
                t = torch.clamp(t, min=0.0, max=1.0)
                closest = starts.unsqueeze(1) + t.unsqueeze(2) * seg.unsqueeze(1)
                dist_sq = torch.sum((obstacle_points.unsqueeze(0) - closest) ** 2, dim=2)
                min_dist_sq = torch.min(dist_sq, dim=1).values
                blocked = min_dist_sq <= safety_radius_sq
            else:
                blocked = torch.zeros(int(dirs_t.numel()), dtype=torch.bool, device=self.device)

            if max_dist > 0.0:
                end_range = torch.linalg.vector_norm(ends, dim=1)
                blocked = blocked | (end_range + safety_radius > max_dist)
            return blocked

        return _blocked

    def _exploit_lidar_heading_blocker(
        self,
        num_headings: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        clearance_threshold = self._exploit_forward_primitive_distance()
        clearances = []
        for hd_idx in range(int(max(1, num_headings))):
            heading = hd_idx * (360.0 / float(max(1, num_headings)))
            clearances.append(self._lidar_clearance_for_world_heading(heading))

        clearances_t = torch.as_tensor(clearances, dtype=self.dtype, device=self.device)
        blocked = clearances_t <= float(clearance_threshold)
        all_blocked = bool(torch.all(blocked).item()) if int(blocked.numel()) else False
        applied = bool(torch.any(blocked).item()) and not all_blocked
        return blocked, clearances_t, float(clearance_threshold), applied

    def _apply_exploit_lidar_heading_blocker(
        self,
        macro_returns: torch.Tensor,
        within_direction_beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        blocked, clearances, clearance_threshold, applied = self._exploit_lidar_heading_blocker(
            int(macro_returns.numel())
        )
        policy_returns = macro_returns.clone()
        if applied:
            policy_returns = policy_returns.masked_fill(blocked, -1.0e9)

        centered = policy_returns - torch.max(policy_returns)
        direction_probs = torch.softmax(float(within_direction_beta) * centered, dim=0)
        candidate_angles = torch.arange(
            int(policy_returns.numel()),
            dtype=self.dtype,
            device=self.device,
        ) * (2.0 * math.pi / float(max(1, int(policy_returns.numel()))))
        candidate_vectors = torch.stack(
            [torch.cos(candidate_angles), torch.sin(candidate_angles)],
            dim=1,
        )
        combined_vector = torch.sum(direction_probs.unsqueeze(1) * candidate_vectors, dim=0)
        return (
            direction_probs,
            combined_vector,
            policy_returns,
            blocked,
            clearances,
            clearance_threshold,
            applied,
        )

    def _record_exploit_loop_position(self, x: float, z: float) -> None:
        history = list(getattr(self, "exploit_loop_history", []))
        history.append((float(x), float(z)))
        max_len = int(max(4, getattr(self, "exploit_loop_history_size", max(8, 2 * int(self.n_hd)))))
        if len(history) > max_len:
            history = history[-max_len:]
        self.exploit_loop_history = history

    def _exploit_loop_detected(self) -> bool:
        if int(getattr(self, "exploit_loop_cooldown", 0)) > 0:
            self.exploit_loop_cooldown = int(getattr(self, "exploit_loop_cooldown", 0)) - 1
            return False

        history = list(getattr(self, "exploit_loop_history", []))
        min_len = int(max(6, self.n_hd))
        if len(history) < min_len:
            return False

        recent = history[-min_len:]
        path_length = 0.0
        for prev, curr in zip(recent[:-1], recent[1:]):
            path_length += math.hypot(curr[0] - prev[0], curr[1] - prev[1])

        net_displacement = math.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1])
        primitive = max(1e-6, self._exploit_forward_primitive_distance())
        return path_length >= 4.0 * primitive and net_displacement <= 1.5 * primitive

    def _preplay_json_float(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def _preplay_json_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _preplay_tensor_vector(self, value) -> torch.Tensor:
        if value is None:
            return torch.empty(0, dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().view(-1)
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _preplay_vector_list(self, value) -> List[Optional[float]]:
        tensor = self._preplay_tensor_vector(value)
        return [self._preplay_json_float(v) for v in tensor.tolist()]

    def _preplay_int_vector_list(self, value) -> List[Optional[int]]:
        if value is None:
            return []
        tensor = torch.as_tensor(value, dtype=torch.long).detach().cpu().view(-1)
        return [self._preplay_json_int(v) for v in tensor.tolist()]

    def _preplay_score_metrics(self, scores, beta: float = 1.0) -> Dict[str, Optional[float]]:
        values = self._preplay_tensor_vector(scores)
        if int(values.numel()) == 0:
            return {"margin": None, "entropy": None, "range": None}
        if int(values.numel()) == 1:
            return {
                "margin": 0.0,
                "entropy": 0.0,
                "range": 0.0,
            }

        top2 = torch.topk(values, k=2).values
        margin = float((top2[0] - top2[1]).item())
        value_range = float((torch.max(values) - torch.min(values)).item())
        centered = values - torch.max(values)
        probs = torch.softmax(float(beta) * centered, dim=0)
        probs = torch.clamp(torch.nan_to_num(probs), min=1e-12)
        entropy = float((-(probs * torch.log(probs)).sum() / math.log(float(values.numel()))).item())
        return {
            "margin": self._preplay_json_float(margin),
            "entropy": self._preplay_json_float(entropy),
            "range": self._preplay_json_float(value_range),
        }

    def _preplay_probability_entropy(self, probs) -> Optional[float]:
        values = self._preplay_tensor_vector(probs)
        if int(values.numel()) <= 1:
            return 0.0 if int(values.numel()) == 1 else None
        values = torch.clamp(torch.nan_to_num(values), min=1e-12)
        values = values / torch.clamp(torch.sum(values), min=1e-12)
        entropy = float((-(values * torch.log(values)).sum() / math.log(float(values.numel()))).item())
        return self._preplay_json_float(entropy)

    def _preplay_diagnostics_file_path(self) -> Optional[str]:
        cached = getattr(self, "_preplay_decision_diagnostics_path", None)
        if cached:
            return cached

        trial_id = getattr(self, "trial_id", None) or "trial"
        output_dir = None
        stats = getattr(self, "stats_collector", None)
        if stats is not None:
            output_dir = getattr(stats, "output_dir", None)
        if not output_dir:
            output_dir = getattr(self, "trial_base_dir", None) or getattr(self, "network_dir", None)
        if not output_dir:
            return None

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{trial_id}_preplay_diagnostics.jsonl")
        self._preplay_decision_diagnostics_path = path
        return path

    def _write_preplay_decision_diagnostics(
        self,
        diagnostics: Optional[Dict[str, Any]],
        chosen_heading_deg: float,
        preplay_heading_deg: float,
        best_idx: int,
        macro_returns: torch.Tensor,
        policy_returns: torch.Tensor,
        direction_probs: torch.Tensor,
        sampling_variances: torch.Tensor,
        expected_value,
        curr_x: float,
        curr_z: float,
    ) -> None:
        if not bool(getattr(self, "unified_preplay_decision_diagnostics", False)):
            return
        if not diagnostics:
            return

        counter = int(getattr(self, "_preplay_decision_diagnostics_counter", 0))
        stride = int(max(1, getattr(self, "unified_preplay_decision_diagnostics_stride", 1)))
        if counter % stride != 0:
            self._preplay_decision_diagnostics_counter = counter + 1
            return

        path = self._preplay_diagnostics_file_path()
        if not path:
            return

        beta = float(
            getattr(
                self,
                "unified_preplay_within_direction_beta",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
            )
        )
        first_step_scores = diagnostics.get("first_planner_scores")
        if first_step_scores is None:
            first_step_scores = diagnostics.get("first_normalized_scores")
        if first_step_scores is None:
            first_step_scores = diagnostics.get("first_raw_scores")

        planner_metrics = self._preplay_score_metrics(macro_returns, beta=beta)
        policy_metrics = self._preplay_score_metrics(policy_returns, beta=beta)
        actual_policy_entropy = self._preplay_probability_entropy(direction_probs)
        if actual_policy_entropy is not None:
            policy_metrics["entropy"] = actual_policy_entropy
        first_metrics = self._preplay_score_metrics(first_step_scores, beta=beta)

        probs = self._preplay_tensor_vector(direction_probs)
        selected_prob = (
            self._preplay_json_float(probs[int(best_idx)].item())
            if 0 <= int(best_idx) < int(probs.numel())
            else None
        )
        expected = (
            float(expected_value.detach().cpu().item())
            if isinstance(expected_value, torch.Tensor)
            else float(expected_value)
        )

        goal = self._active_goal_for_navigation()
        goal_location = None
        goal_name = None
        goal_distance = None
        if goal is not None:
            goal_name = goal.get("name")
            if goal.get("location") is not None:
                goal_location = [self._preplay_json_float(v) for v in goal["location"]]
                goal_distance = math.hypot(
                    float(goal["location"][0]) - float(curr_x),
                    float(goal["location"][1]) - float(curr_z),
                )

        record = {
            "trial_id": getattr(self, "trial_id", None),
            "world_name": getattr(self, "world_name", None),
            "goal_name": goal_name,
            "goal_location": goal_location,
            "step_count": int(getattr(self, "step_count", 0)),
            "movement_action_count": int(getattr(self, "movement_action_count", 0)),
            "simulation_time_s": self._preplay_json_float(self.getTime()),
            "robot_position_xz": [
                self._preplay_json_float(curr_x),
                self._preplay_json_float(curr_z),
            ],
            "goal_distance": self._preplay_json_float(goal_distance),
            "robot_heading_deg": self._preplay_json_float(getattr(self, "current_heading_deg", float("nan"))),
            "selected_heading_deg": self._preplay_json_float(chosen_heading_deg),
            "preplay_heading_deg": self._preplay_json_float(preplay_heading_deg),
            "selected_heading_index": int(best_idx),
            "selected_heading_prob": selected_prob,
            "expected_value": self._preplay_json_float(expected),
            "current_reward": self._preplay_json_float(diagnostics.get("current_reward")),
            "first_step_rewards": self._preplay_vector_list(first_step_scores),
            "first_step_margin": first_metrics["margin"],
            "first_step_entropy": first_metrics["entropy"],
            "first_step_range": first_metrics["range"],
            "planner_returns": self._preplay_vector_list(macro_returns),
            "planner_margin": planner_metrics["margin"],
            "planner_entropy": planner_metrics["entropy"],
            "planner_range": planner_metrics["range"],
            "policy_returns": self._preplay_vector_list(policy_returns),
            "policy_margin": policy_metrics["margin"],
            "policy_entropy": policy_metrics["entropy"],
            "policy_range": policy_metrics["range"],
            "direction_probs": self._preplay_vector_list(direction_probs),
            "sampling_variance": self._preplay_vector_list(sampling_variances),
            "trajectory_blocked_counts": self._preplay_int_vector_list(
                diagnostics.get("trajectory_blocked_counts")
            ),
            "lidar_blocked_headings": [
                bool(v)
                for v in torch.as_tensor(
                    diagnostics.get("lidar_blocked_headings", []),
                    dtype=torch.bool,
                ).detach().cpu().view(-1).tolist()
            ],
            "lidar_clearances": self._preplay_vector_list(diagnostics.get("lidar_clearances")),
            "lidar_block_threshold": self._preplay_json_float(
                diagnostics.get("lidar_block_threshold")
            ),
            "lidar_blocker_applied": bool(diagnostics.get("lidar_blocker_applied", False)),
            "num_steps": int(diagnostics.get("num_steps", 0)),
            "num_samples": int(diagnostics.get("num_samples", 0)),
            "discount_factor": self._preplay_json_float(diagnostics.get("discount_factor")),
            "within_direction_beta": self._preplay_json_float(beta),
            "sampling_temperature": self._preplay_json_float(
                getattr(self, "unified_preplay_sampling_temperature", float("nan"))
            ),
            "turn_offsets": list(getattr(self, "unified_preplay_turn_offsets", [])),
            "return_mode": str(diagnostics.get("return_mode", "unknown")),
            "boundary_mode": str(diagnostics.get("boundary_mode", "unknown")),
            "planner_scoring_mode": str(diagnostics.get("planner_scoring_mode", "unknown")),
        }

        mode = "w" if counter == 0 else "a"
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._preplay_decision_diagnostics_counter = counter + 1

    def _run_exploit_forced_exploration(self) -> None:
        for _ in range(int(max(1, getattr(self, "tau_w", 1)))):
            self.sense()
            self.compute_pcn_activations(learn=False)
            if bool(torch.any(self.collided).item()):
                self.turn(float(np.random.uniform(-np.pi, np.pi)))
                break
            self.check_goal_reached()
            if self.done:
                return
            self.update_hmaps(update_loc=True, update_pcn=False, update_gcn=False)
            self.forward()

        self.turn(float(np.random.normal(0.0, np.deg2rad(30.0))))

    def _active_goal_for_navigation(self) -> Optional[Dict[str, Any]]:
        active_goals = [goal for goal in self.goals if goal.get("active", False)]
        if active_goals:
            return active_goals[0]
        if self.goals:
            return self.goals[0]
        return None

    def _log_exploit_navigation_snapshot(
        self,
        chosen_heading_deg: float,
        best_idx: int,
        best_return: float,
        expected_value: float,
        macro_returns: torch.Tensor,
        macro_vectors: torch.Tensor,
        direction_probs: torch.Tensor,
    ) -> None:
        if not bool(getattr(self, "exploit_debug_logging", False)):
            return

        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_x, curr_z = self._current_position_xz(curr_pos)
        heading_deg = float(getattr(self, "current_heading_deg", 0.0))

        goal = self._active_goal_for_navigation()
        goal_desc = "none"
        goal_distance = float("nan")
        direct_goal_heading = float("nan")
        chosen_vs_goal_error = float("nan")
        if goal is not None and goal.get("location") is not None:
            goal_x = float(goal["location"][0])
            goal_z = float(goal["location"][1])
            goal_radius = float(goal.get("radius", float("nan")))
            goal_distance = math.hypot(goal_x - curr_x, goal_z - curr_z)
            direct_goal_heading = self._heading_from_xz_vector(goal_x - curr_x, goal_z - curr_z)
            chosen_vs_goal_error = self._signed_heading_error_deg(
                chosen_heading_deg,
                direct_goal_heading,
            )
            goal_desc = (
                f"{goal.get('name', 'goal')}@({goal_x:.2f},{goal_z:.2f})"
                f" r={goal_radius:.2f}"
            )

        print(
            f"[EXPLOIT_DEBUG] pos=({curr_x:.2f},{curr_z:.2f}) goal={goal_desc} "
            f"dist={goal_distance:.2f} world_heading={heading_deg:.1f} "
            f"direct_goal={direct_goal_heading:.1f} "
            f"chosen={chosen_heading_deg:.1f} chosen_vs_goal={chosen_vs_goal_error:+.1f} "
            f"best_hd={best_idx} best_return={best_return:.6f} expected={expected_value:.6f}"
        )

        top_k = min(3, int(macro_returns.numel()))
        if top_k <= 0:
            return

        top_values, top_indices = torch.topk(macro_returns, k=top_k)
        top_parts = []
        for rank in range(top_k):
            idx = int(top_indices[rank].item())
            ret = float(top_values[rank].item())
            heading = self._vector_heading_deg(macro_vectors[idx], fallback_hd=idx)
            prob = float(direction_probs[idx].item()) if idx < int(direction_probs.numel()) else float("nan")
            top_parts.append(f"{idx}:{ret:.4f}@{heading:.1f}deg p={prob:.3f}")

        print(f"[EXPLOIT_DEBUG] top_macro -> {' | '.join(top_parts)}")

    def _log_exploit_pc_competition_diagnostics(self) -> None:
        if not bool(getattr(self, "exploit_debug_logging", False)):
            return
        if self.pcn is None:
            return

        diag = getattr(self.pcn, "last_competition_diagnostics", None)
        if not diag:
            return

        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_x, curr_z = self._current_position_xz(curr_pos)
        proximity = float(getattr(self, "prox", getattr(self, "current_min_lidar_distance", float("nan"))))
        scale_pref = torch.as_tensor(
            getattr(self.pcn, "last_scale_preference", torch.empty(0)),
            dtype=torch.float32,
        ).view(-1)
        bvc_gain = torch.as_tensor(
            getattr(self.pcn, "last_bvc_context_gain_per_scale", torch.empty(0)),
            dtype=torch.float32,
        ).view(-1)
        bvc_mod = torch.as_tensor(
            getattr(self.pcn, "last_bvc_excitation_modulation_per_scale", torch.empty(0)),
            dtype=torch.float32,
        ).view(-1)

        print(
            f"[EXPLOIT_PC_DIAG] pos=({curr_x:.2f},{curr_z:.2f}) "
            f"proximity={proximity:.3f} "
            f"context_mode={getattr(self.pcn, 'bvc_context_modulation_mode', 'unknown')} "
            f"cross_inhibition_disabled={int(bool(getattr(self.pcn, 'disable_cross_scale_inhibition', False)))} "
            f"total_prev_mass={float(diag.get('total_activity', 0.0)):.6f}"
        )

        for item in diag.get("per_scale", []):
            idx = int(item.get("scale_idx", -1))
            if idx < 0:
                continue
            cfg = self.scales[idx] if idx < len(self.scales) else {}
            name = str(cfg.get("name", f"scale_{idx}"))
            pref = float(scale_pref[idx].item()) if idx < int(scale_pref.numel()) else float("nan")
            gain = float(bvc_gain[idx].item()) if idx < int(bvc_gain.numel()) else float("nan")
            mod = float(bvc_mod[idx].item()) if idx < int(bvc_mod.numel()) else float("nan")
            afferent = item.get("afferent", {})
            afferent_inh = item.get("afferent_inhibition", {})
            recurrent = item.get("recurrent_inhibition", {})
            cross = item.get("cross_scale_inhibition", {})
            net = item.get("net_drive", {})
            update = item.get("activation_update", {})
            print(
                f"[EXPLOIT_PC_DIAG] scale={name} "
                f"pref={pref:.3f} bvc_gain={gain:.3f} bvc_mod={mod:.3f} "
                f"mass_prev={float(item.get('current_mass', 0.0)):.6f} "
                f"mass_new={float(item.get('new_mass', 0.0)):.6f} "
                f"active={float(item.get('active_fraction', 0.0)):.4f} "
                f"aff_mean={float(afferent.get('mean', 0.0)):.6f} "
                f"aff_max={float(afferent.get('max', 0.0)):.6f} "
                f"inh_aff_abs={float(afferent_inh.get('abs_mean', 0.0)):.6f} "
                f"inh_rec_abs={float(recurrent.get('abs_mean', 0.0)):.6f} "
                f"inh_cross_abs={float(cross.get('abs_mean', 0.0)):.6f} "
                f"net_mean={float(net.get('mean', 0.0)):.6f} "
                f"net_max={float(net.get('max', 0.0)):.6f} "
                f"update_mean={float(update.get('mean', 0.0)):.6f} "
                f"update_max={float(update.get('max', 0.0)):.6f}"
            )

    def _log_exploit_decode_status_once(self, message: str) -> None:
        if bool(getattr(self, "_exploit_decode_status_logged", False)):
            return
        if bool(getattr(self, "exploit_hd_score_logging", False)) or bool(
            getattr(self, "exploit_debug_logging", False)
        ):
            print(f"[EXPLOIT_HD_DECODE] {message}")
        self._exploit_decode_status_logged = True

    def _candidate_exploit_decode_hmap_dirs(self) -> List[str]:
        candidates = []
        for path in (
            getattr(self, "hmap_dir", None),
            os.path.join("pkl", str(getattr(self, "world_name", "")), "hmaps"),
        ):
            if path and path not in candidates:
                candidates.append(path)

        pkl_root = "pkl"
        world_name = str(getattr(self, "world_name", ""))
        if os.path.isdir(pkl_root) and world_name:
            for dirname in sorted(os.listdir(pkl_root)):
                if dirname.startswith(world_name) or world_name.startswith(dirname):
                    path = os.path.join(pkl_root, dirname, "hmaps")
                    if path not in candidates:
                        candidates.append(path)
        return candidates

    def _load_hmap_decode_source(self, hmap_dir: str) -> tuple[Optional[np.ndarray], Optional[torch.Tensor], str]:
        loc_path = os.path.join(hmap_dir, "hmap_loc.pkl")
        pcn_path = os.path.join(hmap_dir, "hmap_pcn.pkl")
        if not os.path.exists(loc_path):
            return None, None, f"missing {loc_path}"

        with open(loc_path, "rb") as f:
            hmap_loc = np.asarray(pickle.load(f), dtype=np.float32)

        if os.path.exists(pcn_path):
            with open(pcn_path, "rb") as f:
                hmap_pcn = pickle.load(f)
            return hmap_loc, torch.as_tensor(hmap_pcn, dtype=torch.float32, device="cpu"), hmap_dir

        blocks = []
        missing = []
        for scale_def in self.scales:
            scale_idx = int(scale_def["scale_index"])
            scale_path = os.path.join(hmap_dir, f"hmap_pcn_scale_{scale_idx}.pkl")
            if not os.path.exists(scale_path):
                missing.append(scale_path)
                continue
            with open(scale_path, "rb") as f:
                blocks.append(torch.as_tensor(pickle.load(f), dtype=torch.float32, device="cpu"))
        if missing:
            return None, None, f"missing PC hmap file(s), first missing {missing[0]}"
        if not blocks:
            return None, None, f"no PC hmap files in {hmap_dir}"
        return hmap_loc, torch.cat(blocks, dim=1), hmap_dir

    def _load_exploit_decode_hmap_cache(self) -> Optional[Dict[str, torch.Tensor]]:
        cached = getattr(self, "_exploit_decode_hmap_cache", None)
        if cached is not None:
            return cached

        load_errors = []
        try:
            hmap_loc = None
            hmap_states = None
            source_dir = None
            for hmap_dir in self._candidate_exploit_decode_hmap_dirs():
                loc_candidate, state_candidate, status = self._load_hmap_decode_source(hmap_dir)
                if loc_candidate is None or state_candidate is None:
                    load_errors.append(status)
                    continue
                hmap_loc = loc_candidate
                hmap_states = state_candidate
                source_dir = status
                break
        except Exception as exc:
            self._log_exploit_decode_status_once(f"failed to load hmaps for decode: {exc}")
            self._exploit_decode_hmap_cache = None
            return None

        if hmap_loc is None or hmap_states is None:
            reason = "; ".join(str(err) for err in load_errors[:3]) or "no candidate hmap dirs"
            self._log_exploit_decode_status_once(f"disabled: {reason}")
            return None

        if hmap_loc.ndim != 2 or hmap_states.dim() != 2:
            self._log_exploit_decode_status_once(
                f"disabled: invalid hmap shapes loc={getattr(hmap_loc, 'shape', None)} "
                f"pcn={tuple(hmap_states.shape) if hasattr(hmap_states, 'shape') else None}"
            )
            return None
        expected_width = int(getattr(self.pcn, "num_pc_total", hmap_states.shape[1]))
        if int(hmap_states.shape[1]) != expected_width:
            self._log_exploit_decode_status_once(
                f"disabled: hmap PC width {int(hmap_states.shape[1])} "
                f"does not match PCN width {expected_width}"
            )
            return None

        row_count = min(int(hmap_loc.shape[0]), int(hmap_states.shape[0]))
        if row_count <= 0:
            self._log_exploit_decode_status_once("disabled: no hmap rows")
            return None
        hmap_loc = hmap_loc[:row_count]
        hmap_states = torch.clamp(torch.nan_to_num(hmap_states[:row_count]), min=0.0)
        if hmap_loc.shape[1] >= 3:
            loc_xz = hmap_loc[:, [0, 2]]
        elif hmap_loc.shape[1] >= 2:
            loc_xz = hmap_loc[:, [0, 1]]
        else:
            return None

        state_norms = torch.linalg.vector_norm(hmap_states, ord=2, dim=1)
        finite_loc = np.isfinite(loc_xz).all(axis=1)
        valid = torch.as_tensor(finite_loc, dtype=torch.bool) & (state_norms > 1e-8)
        if int(valid.sum().item()) <= 0:
            self._log_exploit_decode_status_once("disabled: no finite nonzero hmap PC states")
            return None

        hmap_states = hmap_states[valid]
        state_norms = state_norms[valid].unsqueeze(1)
        loc_xz_t = torch.as_tensor(loc_xz[valid.cpu().numpy()], dtype=torch.float32, device="cpu")
        cache = {
            "states_norm": hmap_states / torch.clamp(state_norms, min=1e-8),
            "loc_xz": loc_xz_t,
        }
        self._exploit_decode_hmap_cache = cache
        self._log_exploit_decode_status_once(
            f"loaded {int(loc_xz_t.shape[0])} real hmap states from {source_dir}"
        )
        return cache

    def _distance_to_nearest_wall_from_xz(self, x: float, z: float) -> float:
        try:
            world_config = get_world_config(self.world_name)
        except Exception:
            return float("nan")

        size = world_config.get("size", None)
        distances = []
        if size and len(size) >= 2:
            half_x = float(size[0]) / 2.0
            half_z = float(size[1]) / 2.0
            distances.extend([half_x - abs(float(x)), half_z - abs(float(z))])

        for obstacle in world_config.get("obstacles", []):
            if obstacle.get("type") != "rectangle":
                continue
            distances.append(obstacle_distance(float(x), float(z), obstacle))

        if not distances:
            return float("nan")
        return max(0.0, float(min(distances)))

    def _decode_preplay_states_to_hmap(
        self,
        predicted_states,
        goal: Optional[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        states = torch.as_tensor(predicted_states, dtype=torch.float32, device="cpu")
        if states.dim() != 2 or int(states.shape[0]) <= 0:
            return []

        defaults = [
            {
                "cosine": float("nan"),
                "x": float("nan"),
                "z": float("nan"),
                "goal_dist": float("nan"),
                "wall_dist": float("nan"),
            }
            for _ in range(int(states.shape[0]))
        ]
        cache = self._load_exploit_decode_hmap_cache()
        if cache is None:
            return defaults

        queries = torch.clamp(torch.nan_to_num(states), min=0.0)
        query_norms = torch.linalg.vector_norm(queries, ord=2, dim=1, keepdim=True)
        valid_query = (query_norms.squeeze(1) > 1e-8)
        if not bool(torch.any(valid_query).item()):
            return defaults

        query_normed = queries / torch.clamp(query_norms, min=1e-8)
        similarities = cache["states_norm"] @ query_normed.t()
        best_cos, best_idx = torch.max(similarities, dim=0)
        loc_xz = cache["loc_xz"]

        goal_x = goal_z = None
        if goal is not None and goal.get("location") is not None:
            goal_x = float(goal["location"][0])
            goal_z = float(goal["location"][1])

        decoded = []
        for row_idx in range(int(states.shape[0])):
            if not bool(valid_query[row_idx].item()):
                decoded.append(defaults[row_idx])
                continue
            loc = loc_xz[int(best_idx[row_idx].item())]
            x = float(loc[0].item())
            z = float(loc[1].item())
            goal_dist = (
                math.hypot(x - goal_x, z - goal_z)
                if goal_x is not None and goal_z is not None
                else float("nan")
            )
            decoded.append(
                {
                    "cosine": float(best_cos[row_idx].item()),
                    "x": x,
                    "z": z,
                    "goal_dist": float(goal_dist),
                    "wall_dist": self._distance_to_nearest_wall_from_xz(x, z),
                }
            )
        return decoded

    def _log_exploit_preplay_hd_scores(
        self,
        diagnostics: Optional[Dict[str, Any]],
        chosen_heading_deg: float,
        preplay_heading_deg: float,
        best_idx: int,
        macro_returns: torch.Tensor,
        macro_vectors: torch.Tensor,
        direction_probs: torch.Tensor,
    ) -> None:
        if not (
            bool(getattr(self, "exploit_hd_score_logging", False))
            or bool(getattr(self, "exploit_debug_logging", False))
        ):
            return
        if not diagnostics:
            return

        def _as_float_tensor(value, shape_cols: Optional[int] = None) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if shape_cols is not None and tensor.dim() == 1:
                tensor = tensor.view(1, -1)
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        def _format_scale_values(scale_names: Sequence[str], values: torch.Tensor) -> str:
            values_flat = torch.as_tensor(values, dtype=torch.float32).view(-1)
            parts = []
            for scale_idx, value in enumerate(values_flat.tolist()):
                name = scale_names[scale_idx] if scale_idx < len(scale_names) else f"scale_{scale_idx}"
                parts.append(f"{name}={float(value):.6f}")
            return " ".join(parts)

        scale_names = [
            str(name)
            for name in diagnostics.get(
                "scale_names",
                [f"scale_{idx}" for idx in range(getattr(self.pcn, "num_scales", 0))],
            )
        ]
        raw_returns = _as_float_tensor(diagnostics.get("raw_returns", []))
        scale_returns = _as_float_tensor(
            diagnostics.get("scale_returns", []),
            shape_cols=len(scale_names),
        )
        current_mass = _as_float_tensor(diagnostics.get("current_mass_by_scale", []))
        predicted_mass = _as_float_tensor(
            diagnostics.get("predicted_mass_by_scale", []),
            shape_cols=len(scale_names),
        )
        first_predicted_mass = _as_float_tensor(
            diagnostics.get("first_predicted_mass_by_scale", []),
            shape_cols=len(scale_names),
        )
        predicted_states = diagnostics.get("predicted_states", [])
        policy_returns = _as_float_tensor(diagnostics.get("policy_returns", macro_returns.detach().cpu()))
        lidar_blocked = torch.as_tensor(
            diagnostics.get("lidar_blocked_headings", []),
            dtype=torch.bool,
        ).view(-1)
        lidar_clearances = _as_float_tensor(diagnostics.get("lidar_clearances", []))
        lidar_block_threshold = float(diagnostics.get("lidar_block_threshold", float("nan")))
        lidar_blocker_applied = int(bool(diagnostics.get("lidar_blocker_applied", False)))
        trajectory_blocked_counts = torch.as_tensor(
            diagnostics.get("trajectory_blocked_counts", []),
            dtype=torch.long,
        ).view(-1)
        trajectory_dirs = torch.as_tensor(
            diagnostics.get("trajectory_dirs", []),
            dtype=torch.long,
        )
        if trajectory_dirs.dim() == 1:
            trajectory_dirs = trajectory_dirs.view(1, -1)
        trajectory_step_distance = float(diagnostics.get("trajectory_step_distance", 0.0))

        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_x, curr_z = self._current_position_xz(curr_pos)
        goal = self._active_goal_for_navigation()
        direct_goal_heading = float("nan")
        goal_distance = float("nan")
        if goal is not None and goal.get("location") is not None:
            goal_x = float(goal["location"][0])
            goal_z = float(goal["location"][1])
            direct_goal_heading = self._heading_from_xz_vector(goal_x - curr_x, goal_z - curr_z)
            goal_distance = math.hypot(goal_x - curr_x, goal_z - curr_z)
        decoded_states = self._decode_preplay_states_to_hmap(predicted_states, goal)

        num_steps = int(diagnostics.get("num_steps", 0))
        num_samples = int(diagnostics.get("num_samples", 0))
        discount_factor = float(diagnostics.get("discount_factor", float("nan")))
        planner_scoring_mode = str(diagnostics.get("planner_scoring_mode", "unknown"))
        action_vector_mode = str(diagnostics.get("action_vector_mode", "unknown"))
        boundary_mode = str(diagnostics.get("boundary_mode", "unknown"))
        current_reward = float(diagnostics.get("current_reward", float("nan")))
        print(
            f"[EXPLOIT_HD_SCORE] summary chosen={chosen_heading_deg:.1f} "
            f"preplay_chosen={preplay_heading_deg:.1f} direct_goal={direct_goal_heading:.1f} "
            f"goal_dist={goal_distance:.3f} best_hd={best_idx} "
            f"steps={num_steps} samples={num_samples} discount={discount_factor:.3f} "
            f"current_reward={current_reward:.6f} scoring={planner_scoring_mode} "
            f"action_vector={action_vector_mode} "
            f"boundary_mode={boundary_mode} "
            f"lidar_blocker={lidar_blocker_applied} block_dist={lidar_block_threshold:.3f} "
            f"traj_step={trajectory_step_distance:.3f}"
        )

        hd_count = min(int(self.n_hd), int(macro_returns.numel()), int(raw_returns.numel()))
        for hd_idx in range(hd_count):
            candidate_heading = self._normalize_heading_deg(
                hd_idx * (360.0 / float(max(1, self.n_hd)))
            )
            macro_heading = self._vector_heading_deg(macro_vectors[hd_idx], fallback_hd=hd_idx)
            prob = (
                float(direction_probs[hd_idx].item())
                if hd_idx < int(direction_probs.numel())
                else float("nan")
            )
            planner_return = float(macro_returns[hd_idx].item())
            policy_return = (
                float(policy_returns[hd_idx].item())
                if hd_idx < int(policy_returns.numel())
                else planner_return
            )
            raw_return = float(raw_returns[hd_idx].item())
            blocked = int(bool(lidar_blocked[hd_idx].item())) if hd_idx < int(lidar_blocked.numel()) else 0
            clearance = (
                float(lidar_clearances[hd_idx].item())
                if hd_idx < int(lidar_clearances.numel())
                else float("nan")
            )
            scale_return_text = (
                _format_scale_values(scale_names, scale_returns[hd_idx])
                if hd_idx < int(scale_returns.shape[0])
                else ""
            )
            current_mass_text = _format_scale_values(scale_names, current_mass)
            first_pred_mass_text = (
                _format_scale_values(scale_names, first_predicted_mass[hd_idx])
                if hd_idx < int(first_predicted_mass.shape[0])
                else ""
            )
            predicted_mass_text = (
                _format_scale_values(scale_names, predicted_mass[hd_idx])
                if hd_idx < int(predicted_mass.shape[0])
                else ""
            )
            pred1_total = (
                float(torch.sum(first_predicted_mass[hd_idx]).item())
                if hd_idx < int(first_predicted_mass.shape[0])
                else float("nan")
            )
            predn_total = (
                float(torch.sum(predicted_mass[hd_idx]).item())
                if hd_idx < int(predicted_mass.shape[0])
                else float("nan")
            )
            decoded = decoded_states[hd_idx] if hd_idx < len(decoded_states) else {}
            decode_cos = float(decoded.get("cosine", float("nan")))
            decode_x = float(decoded.get("x", float("nan")))
            decode_z = float(decoded.get("z", float("nan")))
            decode_goal_dist = float(decoded.get("goal_dist", float("nan")))
            decode_wall_dist = float(decoded.get("wall_dist", float("nan")))
            traj_blocked = (
                int(trajectory_blocked_counts[hd_idx].item())
                if hd_idx < int(trajectory_blocked_counts.numel())
                else 0
            )
            if hd_idx < int(trajectory_dirs.shape[0]):
                traj_seq = [
                    int(v)
                    for v in trajectory_dirs[hd_idx].view(-1).tolist()
                    if int(v) >= 0
                ]
                traj_text = ",".join(str(v) for v in traj_seq)
            else:
                traj_text = ""
            print(
                f"[EXPLOIT_HD_SCORE] hd={hd_idx} candidate={candidate_heading:.1f} "
                f"macro_angle={macro_heading:.1f} chosen={chosen_heading_deg:.1f} "
                f"direct_goal={direct_goal_heading:.1f} p={prob:.3f} "
                f"raw_return={raw_return:.6f} planner_return={planner_return:.6f} "
                f"policy_return={policy_return:.6f} lidar_clearance={clearance:.3f} "
                f"blocked={blocked} traj_blocked={traj_blocked} traj=[{traj_text}] "
                f"pred1_total={pred1_total:.6f} predN_total={predn_total:.6f} "
                f"decode_cos={decode_cos:.4f} "
                f"decode=({decode_x:.2f},{decode_z:.2f}) "
                f"decode_goal_dist={decode_goal_dist:.3f} "
                f"decode_wall_dist={decode_wall_dist:.3f} "
                f"scale_return {scale_return_text} "
                f"current_mass {current_mass_text} "
                f"pred1_mass {first_pred_mass_text} "
                f"predN_mass {predicted_mass_text}"
            )

    def _is_exploit_mode(self) -> bool:
        return self.robot_mode in {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }

    def _debug_heading_test_start_translation(self) -> List[float]:
        translation_field = self.robot.getField("translation")
        current = [float(v) for v in translation_field.getSFVec3f()]
        configured = getattr(self, "debug_heading_test_start_loc", None)
        if configured is None:
            return current

        if len(configured) >= 3:
            return [float(configured[0]), float(configured[1]), float(configured[2])]
        if len(configured) >= 2:
            return [float(configured[0]), current[1], float(configured[1])]
        return current

    def _debug_heading_test_current_rotation(self) -> Optional[List[float]]:
        try:
            rotation_field = self.robot.getField("rotation")
            if rotation_field is None:
                return None
            return [float(v) for v in rotation_field.getSFRotation()]
        except Exception:
            return None

    def _debug_heading_test_restore_pose(
        self,
        translation: Sequence[float],
        rotation: Optional[Sequence[float]],
    ) -> None:
        self.stop()
        self.robot.getField("translation").setSFVec3f([float(v) for v in translation])
        if rotation is not None:
            try:
                rotation_field = self.robot.getField("rotation")
                if rotation_field is not None:
                    rotation_field.setSFRotation([float(v) for v in rotation])
            except Exception as exc:
                print(f"[HEADING_TEST] Could not restore robot rotation: {exc}")
        self.robot.resetPhysics()
        self.last_heading_deg = None
        self.last_execute_movement_collision = False
        for _ in range(2):
            self.sense()

    def _debug_heading_test_move_forward(self) -> bool:
        passive_exploit_modes = {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }
        learn_flag = False if self.robot_mode in passive_exploit_modes else None
        self.last_execute_movement_collision = False

        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations(learn=learn_flag)
            self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
            self.forward()
            self.check_goal_reached()
            if self.done:
                if bool(getattr(self, "exploit_debug_logging", False)) and self._is_exploit_mode():
                    curr_pos = self.robot.getField("translation").getSFVec3f()
                    curr_x, curr_z = self._current_position_xz(curr_pos)
                    goal_distance = float("nan")
                    goal_desc = "none"
                    goal_radius_done = float("nan")
                    if active_goal is not None and active_goal.get("location") is not None:
                        goal_x = float(active_goal["location"][0])
                        goal_z = float(active_goal["location"][1])
                        goal_radius_done = float(active_goal.get("radius", float("nan")))
                        goal_distance = math.hypot(goal_x - curr_x, goal_z - curr_z)
                        goal_desc = f"{active_goal.get('name', 'goal')}@({goal_x:.2f},{goal_z:.2f})"
                    print(
                        f"[EXPLOIT_MOVE] goal_check_done target_world={world_heading_deg:.1f} "
                        f"target_controller={controller_heading_deg:.1f} "
                        f"pos=({curr_x:.2f},{curr_z:.2f}) "
                        f"goal={goal_desc} r={goal_radius_done:.2f} "
                        f"dist={goal_distance:.3f}"
                    )
                return True
            if bool(torch.any(self.collided).item()):
                self.stop()
                self.last_execute_movement_collision = True
                return False

            if hasattr(self, "last_heading_deg") and self.last_heading_deg is not None:
                heading_diff = self.current_heading_deg - self.last_heading_deg
                heading_diff = ((heading_diff + 180) % 360) - 180
                self.rotation_accumulator += abs(heading_diff)
                self.last_heading_deg = self.current_heading_deg
            else:
                self.last_heading_deg = self.current_heading_deg

        return True

    def run_heading_convention_test(self) -> None:
        """Debug-only Webots compass/movement convention probe."""
        if bool(getattr(self, "_debug_heading_convention_test_ran", False)):
            return

        self._debug_heading_convention_test_ran = True
        original_tau_w = int(self.tau_w)
        original_step_count = int(self.step_count)
        original_done = bool(self.done)
        original_total_distance = float(getattr(self, "total_distance_traveled", 0.0))
        original_last_position = copy.deepcopy(getattr(self, "last_position", None))
        original_last_heading = getattr(self, "last_heading_deg", None)
        original_rotation_accumulator = float(getattr(self, "rotation_accumulator", 0.0))
        original_rotation_loop_count = int(getattr(self, "rotation_loop_count", 0))
        original_steps_since_last_loop = int(getattr(self, "steps_since_last_loop", 0))
        original_diag_position = copy.deepcopy(getattr(self, "_last_diag_position", None))
        original_diag_heading = getattr(self, "_last_diag_heading_deg", None)
        original_diag_left_wheel = getattr(self, "_last_diag_left_wheel_position", None)
        original_diag_right_wheel = getattr(self, "_last_diag_right_wheel_position", None)
        original_diag_collision_latch = np.asarray(
            getattr(self, "_diag_collision_latch", np.zeros(2, dtype=np.int8))
        ).copy()
        original_diag_min_lidar = float(
            getattr(self, "_diag_min_lidar_since_log", self._diag_default_clearance)
        )

        start_translation = self._debug_heading_test_start_translation()
        start_rotation = self._debug_heading_test_current_rotation()
        self._suppress_goal_checks_for_heading_test = True

        records = []
        target_headings = tuple(float(angle) for angle in range(0, 360, 45))
        heading_tolerance_deg = 5.0
        try:
            self.tau_w = int(max(1, getattr(self, "debug_heading_test_forward_steps", 4)))
            self._debug_heading_test_restore_pose(start_translation, start_rotation)

            initial_pos = self.robot.getField("translation").getSFVec3f()
            initial_x, initial_y = self._current_position_xz(initial_pos)
            initial_compass_heading = self._current_controller_heading_deg()
            initial_heading = float(getattr(self, "current_heading_deg", 0.0))
            print("[HEADING_TEST] Starting Webots compass/heading convention sanity check")
            print(
                "[HEADING_TEST] World planar coordinates are logged as x,y "
                "(driver uses Webots x,z for this plane)."
            )
            print(
                "[HEADING_TEST] Expected math convention: "
                "0=+x/east, 90=+y/north, 180=-x/west, 270=-y/south"
            )
            print(
                f"[HEADING_TEST] safe_start=({initial_x:.4f}, {initial_y:.4f}) "
                f"reported_compass_heading={initial_compass_heading:.2f} deg "
                f"reported_world_heading={initial_heading:.2f} deg "
                f"forward_steps={self.tau_w} "
                f"turn_tolerance={heading_tolerance_deg:.1f} deg"
            )

            for target_heading in target_headings:
                self.done = False
                self._debug_heading_test_restore_pose(start_translation, start_rotation)

                heading_before_turn = float(getattr(self, "current_heading_deg", 0.0))
                compass_before_turn = self._current_controller_heading_deg()
                controller_target_heading = self._world_heading_to_controller_heading_deg(target_heading)
                turn_success = bool(
                    self.compass_based_turn_to_heading(controller_target_heading, debug=False)
                )
                reported_compass_heading = self._current_controller_heading_deg()
                reported_heading = float(getattr(self, "current_heading_deg", 0.0))
                turn_error = self._signed_heading_error_deg(target_heading, reported_heading)
                aligned = abs(turn_error) <= heading_tolerance_deg

                if not aligned:
                    records.append(
                        {
                            "target": target_heading,
                            "controller_target": controller_target_heading,
                            "reported_compass_heading": reported_compass_heading,
                            "reported_heading": reported_heading,
                            "turn_error": turn_error,
                            "turn_success": turn_success,
                            "skipped": True,
                        }
                    )
                    print(
                        f"[HEADING_TEST] target={target_heading:6.1f} deg "
                        f"heading_before_turn={heading_before_turn:7.2f} deg "
                        f"compass_before_turn={compass_before_turn:7.2f} deg "
                        f"controller_target={controller_target_heading:7.2f} deg "
                        f"reported_compass_heading={reported_compass_heading:7.2f} deg "
                        f"reported_world_heading={reported_heading:7.2f} deg "
                        f"world_turn_error={turn_error:+7.2f} deg "
                        f"turn_success={turn_success} skipped=True"
                    )
                    continue

                before_pos = self.robot.getField("translation").getSFVec3f()
                before_x, before_y = self._current_position_xz(before_pos)

                move_success = bool(self._debug_heading_test_move_forward())

                after_pos = self.robot.getField("translation").getSFVec3f()
                after_x, after_y = self._current_position_xz(after_pos)
                reported_heading_after_move = float(getattr(self, "current_heading_deg", 0.0))
                reported_compass_heading_after_move = self._current_controller_heading_deg()
                dx = after_x - before_x
                dy = after_y - before_y
                distance = math.hypot(dx, dy)
                displacement_angle = (
                    self._normalize_heading_deg(math.degrees(math.atan2(dy, dx)))
                    if distance > 1e-6
                    else float("nan")
                )
                offset = (
                    self._normalize_heading_deg(displacement_angle - reported_compass_heading)
                    if distance > 1e-6
                    else float("nan")
                )
                world_error = (
                    self._signed_heading_error_deg(displacement_angle, reported_heading)
                    if distance > 1e-6
                    else float("nan")
                )
                collision = bool(getattr(self, "last_execute_movement_collision", False))

                records.append(
                    {
                        "target": target_heading,
                        "controller_target": controller_target_heading,
                        "reported_compass_heading": reported_compass_heading,
                        "reported_heading": reported_heading,
                        "reported_compass_heading_after_move": reported_compass_heading_after_move,
                        "reported_heading_after_move": reported_heading_after_move,
                        "turn_error": turn_error,
                        "turn_success": turn_success,
                        "dx": dx,
                        "dy": dy,
                        "distance": distance,
                        "displacement_angle": displacement_angle,
                        "offset": offset,
                        "world_error": world_error,
                        "move_success": move_success,
                        "collision": collision,
                        "skipped": False,
                    }
                )

                print(
                    f"[HEADING_TEST] target={target_heading:6.1f} deg "
                    f"controller_target={controller_target_heading:7.2f} deg "
                    f"reported_compass_heading={reported_compass_heading:7.2f} deg "
                    f"reported_world_heading={reported_heading:7.2f} deg "
                    f"world_turn_error={turn_error:+7.2f} deg "
                    f"before=({before_x: .4f}, {before_y: .4f}) "
                    f"after=({after_x: .4f}, {after_y: .4f}) "
                    f"dx={dx:+.4f} dy={dy:+.4f} "
                    f"disp_angle={displacement_angle:7.2f} deg "
                    f"offset_disp_minus_compass={offset:7.2f} deg "
                    f"world_error_disp_minus_world={world_error:+7.2f} deg "
                    f"reported_compass_after_move={reported_compass_heading_after_move:7.2f} deg "
                    f"reported_world_after_move={reported_heading_after_move:7.2f} deg "
                    f"move_success={move_success} collision={collision} skipped=False"
                )

            print("[HEADING_TEST] Summary: commanded heading -> measured world displacement")
            for record in records:
                if record.get("skipped", False):
                    print(
                        f"[HEADING_TEST] {record['target']:6.1f} deg -> skipped "
                        f"(controller_target={record['controller_target']:.2f} deg, "
                        f"compass={record['reported_compass_heading']:.2f} deg, "
                        f"world={record['reported_heading']:.2f} deg, "
                        f"world_turn_err={record['turn_error']:+.2f} deg, "
                        f"turn_success={record['turn_success']})"
                    )
                    continue
                print(
                    f"[HEADING_TEST] {record['target']:6.1f} deg -> "
                    f"{record['displacement_angle']:7.2f} deg "
                    f"(dx={record['dx']:+.4f}, dy={record['dy']:+.4f}, "
                    f"dist={record['distance']:.4f}, "
                    f"compass={record['reported_compass_heading']:.2f} deg, "
                    f"world={record['reported_heading']:.2f} deg, "
                    f"offset_disp_minus_compass={record['offset']:.2f} deg, "
                    f"world_err={record['world_error']:+.2f} deg, "
                    f"world_turn_err={record['turn_error']:+.2f} deg, "
                    f"ok={record['move_success']}, collision={record['collision']})"
                )

            valid_offsets = [
                float(record["offset"])
                for record in records
                if not record.get("skipped", False)
                and math.isfinite(float(record.get("offset", float("nan"))))
            ]
            valid_world_errors = [
                float(record["world_error"])
                for record in records
                if not record.get("skipped", False)
                and math.isfinite(float(record.get("world_error", float("nan"))))
            ]
            if valid_offsets:
                sin_sum = sum(math.sin(math.radians(offset)) for offset in valid_offsets)
                cos_sum = sum(math.cos(math.radians(offset)) for offset in valid_offsets)
                mean_offset = self._normalize_heading_deg(math.degrees(math.atan2(sin_sum, cos_sum)))
                max_abs_dev = max(
                    abs(self._signed_heading_error_deg(mean_offset, offset))
                    for offset in valid_offsets
                )
                print(
                    f"[HEADING_TEST] Valid offset mean={mean_offset:.2f} deg "
                    f"max_abs_circular_deviation={max_abs_dev:.2f} deg "
                    f"valid_rows={len(valid_offsets)} skipped_rows={len(records) - len(valid_offsets)}"
                )
            if valid_world_errors:
                mean_world_error = sum(valid_world_errors) / float(len(valid_world_errors))
                max_abs_world_error = max(abs(error) for error in valid_world_errors)
                print(
                    f"[HEADING_TEST] Valid world error mean={mean_world_error:+.2f} deg "
                    f"max_abs_world_error={max_abs_world_error:.2f} deg"
                )
            else:
                print(f"[HEADING_TEST] No valid rows; skipped_rows={len(records)}")
        finally:
            self._debug_heading_test_restore_pose(start_translation, start_rotation)
            self.tau_w = original_tau_w
            self.step_count = original_step_count
            self.done = original_done
            self.total_distance_traveled = original_total_distance
            self.last_position = original_last_position
            self.last_heading_deg = original_last_heading
            self.rotation_accumulator = original_rotation_accumulator
            self.rotation_loop_count = original_rotation_loop_count
            self.steps_since_last_loop = original_steps_since_last_loop
            self._last_diag_position = original_diag_position
            self._last_diag_heading_deg = original_diag_heading
            self._last_diag_left_wheel_position = original_diag_left_wheel
            self._last_diag_right_wheel_position = original_diag_right_wheel
            self._diag_collision_latch[:] = original_diag_collision_latch
            self._diag_min_lidar_since_log = original_diag_min_lidar
            self._suppress_goal_checks_for_heading_test = False
            self.trial_start_time = self.getTime()

    def _load_or_init_unified_pcn(
        self,
        enable_ojas: Optional[bool],
        enable_stdp: Optional[bool],
    ) -> UnifiedMultiScalePCN:
        if self.unified_pcn_mode and os.path.exists(self.unified_pcn_path):
            with open(self.unified_pcn_path, "rb") as f:
                pcn = pickle.load(f)
            if isinstance(pcn, UnifiedMultiScalePCN):
                print(f"[DRIVER] Loaded unified PCN from {self.unified_pcn_path}")
                pcn.device = self.device
                pcn.dtype = self.dtype
                pcn.world_name = self.world_name
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp
                pcn = self._apply_unified_ablation_settings(pcn)
                return pcn
            print(f"[DRIVER] Existing unified PCN pickle was not a UnifiedMultiScalePCN, rebuilding: {self.unified_pcn_path}")

        print("[DRIVER] Creating unified PCN")
        gamma_pp = float(np.mean([cfg.get("gamma_pp", 0.5) for cfg in self.scales]))
        gamma_cross = float(np.mean([cfg.get("gamma_cross", cfg.get("gamma_pp", 0.5)) for cfg in self.scales]))
        gamma_pb = float(np.mean([cfg.get("gamma_pb", 0.3) for cfg in self.scales]))
        gamma_pg = float(np.mean([cfg.get("gamma_pg", 0.3) for cfg in self.scales]))
        grid_influence = float(np.mean([cfg.get("grid_influence", 0.3) for cfg in self.scales]))
        alpha_pb = float(np.mean([cfg.get("alpha_pb", math.sqrt(0.5)) for cfg in self.scales]))
        alpha_pg = float(np.mean([cfg.get("alpha_pg", math.sqrt(0.5)) for cfg in self.scales]))
        pcn = UnifiedMultiScalePCN(
            scale_configs=self.scales,
            timestep=self.timestep,
            n_hd=self.n_hd,
            n_res=720,
            max_dist=self.max_dist,
            world_name=self.world_name,
            enable_ojas=enable_ojas if enable_ojas is not None else False,
            enable_stdp=enable_stdp if enable_stdp is not None else False,
            grid_influence=grid_influence,
            gamma_pp=gamma_pp,
            gamma_cross=gamma_cross,
            gamma_pb=gamma_pb,
            gamma_pg=gamma_pg,
            alpha_pb=alpha_pb,
            alpha_pg=alpha_pg,
            grid_inhibition_mode="sum",
            learning_stdp_start_steps=0,
            connection_decay_rate=1e-4,
            bvc_context_modulation_mode="bvc_context",
            bvc_context_gain_floor=0.0,
            bvc_context_gain_strength=1.0,
            bvc_excitation_modulation_floor=0.0,
            enforce_grid_structural_mask=False,
            stdp_rectified_scale_centering=self.stdp_rectified_scale_centering,
            stdp_rectified_hd_gate=self.stdp_rectified_hd_gate,
            stdp_winner_hd_gate=self.stdp_winner_hd_gate,
            stdp_min_input_mass=self.stdp_min_input_mass,
            enable_live_diagnostics=False,
            device=self.device,
            dtype=self.dtype,
        )
        pcn = self._apply_unified_ablation_settings(pcn)
        return pcn

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
                pcn.enable_prox_mod = self.use_prox_mod
                if hasattr(pcn, "bvc_layer"):
                    pcn.bvc_layer.scale_name = scale_def.get("name", getattr(pcn.bvc_layer, "scale_name", None))
                    pcn.bvc_layer.enable_prox_mod = self.use_prox_mod

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
                pcn.enable_prox_mod = self.use_prox_mod
                if hasattr(pcn, "bvc_layer"):
                    pcn.bvc_layer.scale_name = scale_def.get("name", getattr(pcn.bvc_layer, "scale_name", None))
                    pcn.bvc_layer.enable_prox_mod = self.use_prox_mod

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
                scale_name=scale_def.get("name"),
                enable_prox_mod=self.use_prox_mod,
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
                    enable_prox_mod=self.use_prox_mod,
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
                    enable_prox_mod=self.use_prox_mod,
                    device=self.device,
                )
                print(f"[DRIVER] Created standard PlaceCellLayer without grid cells, w_in_ratio={w_in_init_ratio}")

        return pcn

    def load_rcns(self):
        self.rcn = self._load_or_init_unified_rcn()

    def _load_or_init_unified_rcn(self):
        try:
            with open(self.unified_rcn_path, "rb") as f:
                rcn = pickle.load(f)
            compatible = (
                getattr(rcn, "num_place_cells", None) == self.pcn.num_pc_total
                and hasattr(rcn, "replay_from_seed")
                and hasattr(rcn, "build_goal_reward_from_events")
            )
            if compatible:
                print(f"[DRIVER] Loaded existing unified RCN from {self.unified_rcn_path}")
                rcn.device = self.device
                rcn.reconfigure_from_scale_configs(self.scales)
                rcn.replay_timesteps = int(DEFAULT_REPLAY_TIMESTEPS)
                rcn.replay_tau = float(DEFAULT_REPLAY_TAU)
                rcn.replay_long_tau = float(DEFAULT_REPLAY_LONG_TAU)
                rcn.replay_long_weight = float(DEFAULT_REPLAY_LONG_WEIGHT)
                return rcn
            print(f"[DRIVER] Rebuilding incompatible unified RCN pickle: {self.unified_rcn_path}")
        except Exception:
            print(f"[DRIVER] Initializing new unified RCN for {self.unified_rcn_path}")

        return UnifiedRewardCell(
            num_place_cells=self.pcn.num_pc_total,
            scale_configs=self.scales,
            device=self.device,
        )

    def _load_goal_specific_rcns(self, goal_name):
        """Load a goal-specific unified RCN for EXPLOIT_LOCATIONS_RANDOM mode."""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        if not os.path.exists(multi_goal_dir):
            print(f"[WARNING] Multi-goal rewards directory not found: {multi_goal_dir}")
            print(f"[WARNING] Make sure to run LEARN_LOCATIONS_COVERAGE first!")
            return

        print(f"[DRIVER] Loading goal-specific unified RCN for goal: {goal_name}")
        requested_goal = next((goal for goal in getattr(self, "goals", []) if goal["name"] == goal_name), None)
        requested_location = requested_goal.get("location") if requested_goal is not None else None

        candidate_paths = [
            os.path.join(multi_goal_dir, f"unified_rcn_goal_{goal_name}.pkl")
        ]
        available_goal_descriptions = []
        summary_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "rb") as f:
                    summary = pickle.load(f)
                for saved_goal in summary.get("goals", []):
                    saved_name = saved_goal.get("name", "default")
                    saved_location = saved_goal.get("location")
                    saved_path = os.path.join(multi_goal_dir, f"unified_rcn_goal_{saved_name}.pkl")
                    if os.path.exists(saved_path):
                        available_goal_descriptions.append(f"{saved_name}@{saved_location}")
                        if requested_location is not None and saved_location == requested_location:
                            candidate_paths.append(saved_path)
            except Exception as exc:
                print(f"[WARNING] Failed to read goal reward summary {summary_path}: {exc}")

        seen_paths = set()
        candidate_paths = [path for path in candidate_paths if not (path in seen_paths or seen_paths.add(path))]

        goal_rcn_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if goal_rcn_path is None:
            if len(getattr(self, "goals", [])) == 1 and self.rcn is not None:
                fallback_default = os.path.join(multi_goal_dir, "unified_rcn_goal_default.pkl")
                if os.path.exists(fallback_default):
                    goal_rcn_path = fallback_default
                else:
                    print(
                        f"[DRIVER] Goal-specific unified RCN not found for sole goal '{goal_name}'. "
                        f"Using default unified RCN at {self.unified_rcn_path}"
                    )
                    return
            else:
                available_desc = ", ".join(available_goal_descriptions) if available_goal_descriptions else "none"
                requested_desc = f"{goal_name}@{requested_location}" if requested_location is not None else goal_name
                message = (
                    f"Goal-specific unified RCN not found for {requested_desc}. "
                    f"Available learned goals: {available_desc}. "
                    f"Run LEARN_LOCATIONS_COVERAGE with matching goals before EXPLOIT_LOCATIONS_RANDOM."
                )
                print(f"[ERROR] {message}")
                raise FileNotFoundError(message)

        with open(goal_rcn_path, "rb") as f:
            rcn = pickle.load(f)
        compatible = (
            getattr(rcn, "num_place_cells", None) == self.pcn.num_pc_total
            and hasattr(rcn, "replay_from_seed")
            and hasattr(rcn, "compute_reward_activations_batched")
        )
        if not compatible:
            raise ValueError(f"Incompatible goal-specific unified RCN: {goal_rcn_path}")
        rcn.device = self.device
        rcn.reconfigure_from_scale_configs(self.scales)
        rcn.replay_timesteps = int(DEFAULT_REPLAY_TIMESTEPS)
        rcn.replay_tau = float(DEFAULT_REPLAY_TAU)
        rcn.replay_long_tau = float(DEFAULT_REPLAY_LONG_TAU)
        rcn.replay_long_weight = float(DEFAULT_REPLAY_LONG_WEIGHT)
        self.rcn = rcn
        print(f"[DRIVER] Loaded goal-specific unified RCN: {goal_rcn_path}")

    ##########################################################################
    #                        GRID CELL NETWORK INITIALIZATION                #
    ##########################################################################

    def init_grid_cell_networks(self):
        """Initialize grid cell networks for each scale based on scale parameters."""
        self.gcns = []

        for scale_def in self.scales:
            scale_idx = scale_def["scale_index"]
            fname = f"gcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)

            try:
                # Try to load existing grid cell network
                with open(path, "rb") as f:
                    gcn = pickle.load(f)
                print(f"[DRIVER] Loaded existing GCN from {path}")
            except (FileNotFoundError, pickle.UnpicklingError):
                # Initialize new grid cell network with scale-specific parameters
                print(f"[DRIVER] Initializing new GCN for scale {scale_idx}")

                # Extract grid cell parameters for module-based GCN
                num_grid_cells = scale_def.get("num_grid_cells", 0)
                num_modules = scale_def.get("num_modules", 8)
                cells_per_module = scale_def.get("cells_per_module", max(1, num_grid_cells // max(1, num_modules)))
                spread_range = scale_def.get("spread_range", (1.2, 1.2))
                scale_multiplier = scale_def.get("scale_multiplier", 1.0)
                translation_scale = scale_def.get("translation_scale", 1.0)
                mask_resolution = scale_def.get("mask_resolution", 128)
                smooth_sigma = scale_def.get("smooth_sigma", 1.5)

                # Skip grid cell creation if num_grid_cells is 0
                if num_grid_cells == 0:
                    self.gcns.append(None)
                    continue

                # Create new grid cell network (module + phase-based + world mask)
                gcn = GridCellLayer(
                    num_modules=num_modules,
                    cells_per_module=cells_per_module,
                    spread_range=spread_range,
                    scale_multiplier=scale_multiplier,
                    translation_scale=translation_scale,
                    threshold=0.7,
                    threshold_type='soft',
                    normalization='per-cell',
                    world_name=self.world_name,
                    mask_resolution=mask_resolution,
                    smooth_sigma=smooth_sigma,
                    device=self.device.type,
                    dtype=self.dtype,
                )

            # Add to list of grid cell networks
            self.gcns.append(gcn)

    ##########################################################################
    #                        GOAL / TRIAL / COVERAGE SETUP                   #
    ##########################################################################

    def _setup_goals(self, goal_config, goal_location):
        """Setup unified goal system supporting both single and multi-goal modes"""
        self.goals = []
        self.multi_goal_mode = False

        multi_goal_world_modes = {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
            RobotMode.PLOTTING_AUTO,
            RobotMode.PLOTTING_COVERAGE_AUTO,
        }

        if goal_config is None:
            inferred_goal_config = build_goal_config_from_world(self.world_name)
            if inferred_goal_config is not None:
                if (
                    inferred_goal_config.get("type") == "multi"
                    and self.robot_mode not in multi_goal_world_modes
                ):
                    first_goal = inferred_goal_config["goals"][0]
                    inferred_goal_config = {
                        "type": "single",
                        "name": first_goal["name"],
                        "location": first_goal["location"],
                        "radius": first_goal["radius"],
                    }
                goal_config = inferred_goal_config
            else:
                # Legacy single-goal fallback
                self.goals.append({
                    "name": "default",
                    "location": goal_location if goal_location else [-3, 3],
                    "radius": self.goal_r["explore"],
                    "visited": False,
                    "active": True
                })
                self.goal_location = self.goals[0]["location"]
        if goal_config is not None and goal_config["type"] == "single":
            # Single goal configuration
            self.goals.append({
                "name": goal_config.get("name", "default"),
                "location": goal_config["location"],
                "radius": goal_config.get("radius", self.goal_r["explore"]),
                "visited": False,
                "active": True
            })
            self.goal_location = self.goals[0]["location"]
        elif goal_config is not None and goal_config["type"] == "multi":
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

        learning_goal_modes = {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        }
        if self.robot_mode in learning_goal_modes:
            self.goal_events_by_goal = {
                goal["name"]: [] for goal in self.goals
            }
            self._current_goal_event_by_goal = {
                goal["name"]: [] for goal in self.goals
            }
            self.goal_visit_counts = {
                goal["name"]: 0 for goal in self.goals
            }
            self.goal_currently_in = {
                goal["name"]: False for goal in self.goals
            }
            self.goal_last_count_time_s_by_goal = {
                goal["name"]: -1e9 for goal in self.goals
            }

        if self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.PLOTTING_COVERAGE_AUTO,
        }:
            if self.environment_size and self.grid_size and self.coverage_percentage:
                self._setup_coverage_tracking(self.environment_size, self.grid_size, self.coverage_percentage)

        if self.multi_goal_mode:
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

    def _update_distance_tracking(self):
        """Update total distance traveled for random spawn mode"""
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            current_pos = self.robot.getField("translation").getSFVec3f()
            current_position_2d = [current_pos[0], current_pos[2]]  # [x, z]

            if self.last_position is not None:
                # Calculate distance moved since last update
                dx = current_position_2d[0] - self.last_position[0]
                dz = current_position_2d[1] - self.last_position[1]
                distance_moved = math.sqrt(dx*dx + dz*dz)
                self.total_distance_traveled += distance_moved

            self.last_position = current_position_2d


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

        if (
            self._is_exploit_mode()
            and bool(getattr(self, "debug_heading_convention_test", False))
        ):
            self.run_heading_convention_test()

        while not self.done:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode in (RobotMode.LEARN_OJAS,
                                      RobotMode.LEARN_HEBB,
                                      RobotMode.PLOTTING,
                                      RobotMode.PLOTTING_AUTO,
                                      RobotMode.PLOTTING_COVERAGE_AUTO,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO):
                self.explore()
            elif self.robot_mode in (RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO):
                self.exploit()
            else:
                print("Unknown state. Exiting...")
                break

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        """
        Handles exploration for multi-scale usage, calling compute_pcn_activations().
        """
        for _ in range(self.tau_w):
            # 1) Sense environment
            self.sense()

            # 4) compute pcn_activations => fill self.pcn_activations_list
            self.compute_pcn_activations()

            # 5) Continue passive exploration; unified reward learning happens from goal-contact events
                # # Turn towards heading 225°
            # 6) If collisions => turn away
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break

            # 7) Check goal, update hmaps, forward
            self.check_goal_reached()
            if self.done:
                return

            # Update coverage tracking if in LEARN_LOCATIONS_COVERAGE or PLOTTING_COVERAGE_AUTO mode
            if self.robot_mode in {RobotMode.LEARN_LOCATIONS_COVERAGE, RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                                   RobotMode.PLOTTING_COVERAGE_AUTO}:
                curr_pos = self.robot.getField("translation").getSFVec3f()
                robot_pos = [curr_pos[0], curr_pos[2]]  # [x, z] coordinates
                self._update_coverage(robot_pos)

            # Update distance tracking if in EXPLOIT_LOCATIONS_RANDOM mode
            if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
                self._update_distance_tracking()

            self.update_hmaps(update_loc=True,
                              update_pcn=True,
                              update_gcn=True,
                              update_scale_priority=True if self.robot_mode == RobotMode.EXPLOIT else False)
            self.forward()

        # A small random turn at the end
        self.turn(np.random.normal(0, np.deg2rad(30)))

    ########################################### EXPLOIT ###########################################
    def exploit(self):
        """Main exploit entrypoint using the simple unified preplay algorithm."""
        return self.exploit_unified_simple()

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
        #===================================================================
        # EXPLOIT V12 PARAMETERS
        #===================================================================

        # --- Random Seed for Reproducibility ---
        preplay_random_seed = None  # Set to integer (e.g., 42) for reproducible sampling, None for random

        # --- Hierarchical Preplay Configuration ---
        num_preplay_steps = 3              # 2 Number of steps per trajectory
        discount_factor = 0.7              # 0.9 Temporal discount (gamma) 95
        within_scale_beta = 500           # 2 Inverse temperature for P(d|s) within each scale 300/500
        scale_selection_beta = 2.0         # Inverse temperature for scale selection softmax (higher = more decisive)
        ema_lambda = 0.25                   # 0.1 EMA decay for scale entropies (0.1 = 10% new, 90% old)
        use_entropy_ema = True             # Toggle: True = apply EMA smoothing to entropies, False = use raw entropies
        entropy_exponent = 2.0             # Exponent for inverse entropy simplex (higher = more sensitive to entropy differences)
        reliability_exponent = 2.0         # Exponent for reliability scores (higher = more sensitive to reliability differences)
        variance_lambda = 1.0              # Weight for variance term in composite entropy (higher = more penalty for high variance)

        # --- Stochastic Sampling Preplay (Biologically Plausible Alternative) ---
        num_samples_per_direction = 10     # Number of trajectory samples per (scale, direction)
        sampling_strategy = "learned"      # "uniform" (random turns) or "learned" (W_rec-weighted turns)
        sampling_temperature = 1         # 1 Softmax temperature for "learned" strategy (higher = more random)

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
        debug_print_interval = 0           # Set >0 temporarily to print exploit_v12 debug info every N steps

        #===================================================================

        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()

        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_scale_priority=True)
        self.check_goal_reached()

        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            self._update_distance_tracking()

        # Save old PCN activations for each scale (for TD learning)
        old_pcn_activations = [pcn.place_cell_activations.clone() for pcn in self.pcns]

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
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        #-------------------------------------------------------------------
        # 6) Build scales_data for hierarchical preplay
        #-------------------------------------------------------------------
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

        if self.pcns and not getattr(self.pcns[0], "exploit_preplay_supported", True):
            return

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
            debug=debug_enabled
        )

        # Store sampling variances for diagnostics (optional)
        if not hasattr(self, 'preplay_sampling_variances'):
            self.preplay_sampling_variances = []
        self.preplay_sampling_variances.append(sampling_variances.cpu().numpy())

        # Store scale entropies for next timestep's EMA
        self.prev_scale_entropies_v12 = scale_entropies

        #-------------------------------------------------------------------
        # 8) Apply Safety Filtering
        #-------------------------------------------------------------------
        # Build safety mask: check if direction has sufficient clearance
        safe_traj_mask = torch.tensor(
            [distances_per_hd[traj_meta['direction']] >= min_safe_distance
             for traj_meta in trajectory_metadata],
            dtype=torch.bool,
            device=self.device,
        )

        if not torch.any(safe_traj_mask):
            # No safe directions -> explore instead of taking an unsafe action
            print("[EXPLOIT_V12] No safe trajectories available. Forcing exploration.")
            self.explore()
            return

        # Filter to safe trajectories and renormalize
        safe_probs = joint_probs[safe_traj_mask]
        safe_probs = safe_probs / torch.clamp(torch.sum(safe_probs), min=1e-9)

        safe_vectors = direction_vectors[safe_traj_mask]
        safe_returns = discounted_returns[safe_traj_mask]
        safe_metadata = [trajectory_metadata[i] for i, safe in enumerate(safe_traj_mask) if safe]

        # Recompute combined_vector and expected_value from safe trajectories only
        combined_vector = torch.sum(safe_probs.unsqueeze(1) * safe_vectors, dim=0)
        expected_value = torch.sum(safe_probs * safe_returns)

        if debug_enabled:
            num_safe = len(safe_metadata)
            unsafe_count = len(trajectory_metadata) - num_safe
            print(f"[EXPLOIT_V12] Safety filter: {num_safe} safe, {unsafe_count} unsafe")

        #-------------------------------------------------------------------
        # Opposing Vectors Check (Cross-Scale)
        #-------------------------------------------------------------------
        # Check if top-2 trajectories by return have opposing direction vectors
        # This operates across ALL scales on the safe trajectories

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
        # 13) Collision-based Suppression and Reliability Updates
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

    def exploit_unified_simple(self):
        """
        Unified exploit loop using sampled microtrajectories per head direction.

        This follows the simple stochastic delta-preplay algorithm:
        - force the first imagined step to follow the chosen macro head direction
        - branch only on later steps via tanh(relu(W_d x - x))
        - score each candidate by the unified reward-cell readout
        - sample microtrajectory turns from those local softmax probabilities
        - keep the max-return sample per macro head direction
        - softmax across macro directions and execute the weighted-sum heading
        """
        self.sense()
        self.compute_pcn_activations(learn=False)
        log_exploit_detail = self._should_log_exploit_detail()
        log_exploit_hd_scores = self._should_log_exploit_hd_scores()
        save_preplay_diagnostics = bool(
            getattr(self, "unified_preplay_decision_diagnostics", False)
        )
        collect_preplay_diagnostics = bool(
            log_exploit_detail or log_exploit_hd_scores or save_preplay_diagnostics
        )
        if log_exploit_detail:
            self._log_exploit_pc_competition_diagnostics()
        self.update_hmaps(update_loc=True, update_pcn=False, update_gcn=False, update_scale_priority=False)
        self.check_goal_reached()
        if self.done:
            return

        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_x, curr_z = self._current_position_xz(curr_pos)
        self._record_exploit_loop_position(curr_x, curr_z)

        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            self._update_distance_tracking()

        loop_recovery_enabled = bool(getattr(self, "exploit_loop_recovery", False))
        if loop_recovery_enabled and int(getattr(self, "force_explore_count", 0)) > 0:
            self.force_explore_count -= 1
            self._run_exploit_forced_exploration()
            return
        if not loop_recovery_enabled:
            self.force_explore_count = 0

        if loop_recovery_enabled and self._exploit_loop_detected():
            self.force_explore_count = int(max(1, getattr(self, "exploit_loop_forced_explore_steps", 6)))
            self.exploit_loop_history = []
            self.exploit_loop_cooldown = self.force_explore_count
            if bool(getattr(self, "exploit_debug_logging", False)):
                print(
                    f"[EXPLOIT_LOOP] Detected closed recent trajectory at "
                    f"pos=({curr_x:.2f},{curr_z:.2f}); exploring for "
                    f"{self.force_explore_count} cycles"
                )
            self._run_exploit_forced_exploration()
            return

        if self.step_count <= self.tau_w:
            return

        num_preplay_steps = int(max(1, getattr(self, "unified_recurrent_preplay_horizon", 3)))
        discount_factor = float(
            getattr(
                self,
                "unified_preplay_discount_factor",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
            )
        )
        within_direction_beta = float(
            getattr(
                self,
                "unified_preplay_within_direction_beta",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
            )
        )
        num_samples_per_direction = int(
            max(
                1,
                getattr(
                    self,
                    "unified_preplay_num_samples",
                    UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
                ),
            )
        )
        sampling_temperature = float(
            max(
                1e-6,
                getattr(
                    self,
                    "unified_preplay_sampling_temperature",
                    UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
                ),
            )
        )
        turn_offsets = list(
            getattr(
                self,
                "unified_preplay_turn_offsets",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"],
            )
        )
        boundary_mode = str(
            getattr(
                self,
                "unified_preplay_boundary_mode",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_boundary_mode"],
            )
        ).strip().lower()
        trajectory_blocked_fn = None
        trajectory_step_distance = None
        if (
            bool(getattr(self, "unified_preplay_executable_rollouts", False))
            and boundary_mode == "hard_block"
        ):
            trajectory_step_distance = self._exploit_forward_primitive_distance()
            trajectory_blocked_fn = self._build_exploit_microtrajectory_blocker(
                trajectory_step_distance
            )

        try:
            preplay_result = self.pcn.unified_preplay_sampling(
                unified_rcn=self.rcn,
                n_hd=self.n_hd,
                num_steps=num_preplay_steps,
                discount_factor=discount_factor,
                within_direction_beta=within_direction_beta,
                num_samples=num_samples_per_direction,
                sampling_temperature=sampling_temperature,
                turn_offsets=turn_offsets,
                return_diagnostics=collect_preplay_diagnostics,
                trajectory_blocked_fn=trajectory_blocked_fn,
                trajectory_step_distance=trajectory_step_distance,
                blocked_return_penalty=float(
                    getattr(
                        self,
                        "unified_preplay_blocked_return_penalty",
                        UNIFIED_PREPLAY_DEFAULTS["unified_preplay_blocked_return_penalty"],
                    )
                ),
            )
            if collect_preplay_diagnostics:
                (
                    _final_direction_deg,
                    expected_value,
                    combined_vector,
                    macro_returns,
                    macro_vectors,
                    sampling_variances,
                    direction_probs,
                    preplay_diagnostics,
                ) = preplay_result
            else:
                (
                    _final_direction_deg,
                    expected_value,
                    combined_vector,
                    macro_returns,
                    macro_vectors,
                    sampling_variances,
                    direction_probs,
                ) = preplay_result
                preplay_diagnostics = None
        except Exception as exc:
            print(f"[EXPLOIT] unified preplay failed: {exc}")
            return

        macro_returns = torch.nan_to_num(macro_returns, nan=0.0, posinf=0.0, neginf=0.0)
        macro_vectors = torch.nan_to_num(macro_vectors, nan=0.0, posinf=0.0, neginf=0.0)
        use_lidar_heading_blocker = (
            bool(getattr(self, "exploit_lidar_heading_blocker_enabled", True))
            and boundary_mode == "hard_block"
        )
        if use_lidar_heading_blocker:
            (
                direction_probs,
                combined_vector,
                policy_returns,
                lidar_blocked,
                lidar_clearances,
                lidar_block_threshold,
                lidar_blocker_applied,
            ) = self._apply_exploit_lidar_heading_blocker(
                macro_returns=macro_returns,
                within_direction_beta=within_direction_beta,
            )
        else:
            policy_returns = macro_returns
            centered = policy_returns - torch.max(policy_returns)
            direction_probs = torch.softmax(float(within_direction_beta) * centered, dim=0)
            candidate_angles = torch.arange(
                int(policy_returns.numel()),
                dtype=self.dtype,
                device=self.device,
            ) * (2.0 * math.pi / float(max(1, int(policy_returns.numel()))))
            candidate_vectors = torch.stack(
                [torch.cos(candidate_angles), torch.sin(candidate_angles)],
                dim=1,
            )
            combined_vector = torch.sum(direction_probs.unsqueeze(1) * candidate_vectors, dim=0)
            lidar_clearances = torch.full_like(macro_returns, float("nan"))
            lidar_block_threshold = float("nan")
            lidar_blocker_applied = False
            lidar_blocked = torch.zeros_like(macro_returns, dtype=torch.bool)

        expected_value = torch.sum(direction_probs * macro_returns)
        best_idx = int(torch.argmax(policy_returns).item())

        if preplay_diagnostics is not None:
            planner_scoring_mode = f"{preplay_diagnostics.get('planner_scoring_mode', 'unknown')}"
            if use_lidar_heading_blocker:
                planner_scoring_mode += "+lidar_initial_hd_blocker"
            preplay_diagnostics["planner_scoring_mode"] = planner_scoring_mode
            preplay_diagnostics["policy_returns"] = policy_returns.detach().cpu()
            preplay_diagnostics["lidar_blocked_headings"] = lidar_blocked.detach().cpu()
            preplay_diagnostics["lidar_clearances"] = lidar_clearances.detach().cpu()
            preplay_diagnostics["lidar_block_threshold"] = float(lidar_block_threshold)
            preplay_diagnostics["lidar_blocker_applied"] = bool(lidar_blocker_applied)
            preplay_diagnostics["boundary_mode"] = boundary_mode

        combined_magnitude = torch.norm(combined_vector)
        if float(combined_magnitude.item()) < 1e-6:
            best_angle = best_idx * (2.0 * math.pi / float(max(1, self.n_hd)))
            combined_vector = torch.tensor(
                [math.cos(best_angle), math.sin(best_angle)],
                dtype=self.dtype,
                device=self.device,
            )
            expected_value = macro_returns[best_idx]

        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])
        final_direction_deg = float((final_direction_rad * (180.0 / math.pi)).item())
        if final_direction_deg < 0.0:
            final_direction_deg += 360.0

        preplay_direction_deg = final_direction_deg
        self.action_heading_deg = final_direction_deg
        if log_exploit_detail:
            self._log_exploit_navigation_snapshot(
                chosen_heading_deg=self.action_heading_deg,
                best_idx=best_idx,
                best_return=float(macro_returns[best_idx].item()),
                expected_value=(
                    float(expected_value.item())
                    if isinstance(expected_value, torch.Tensor)
                    else float(expected_value)
                ),
                macro_returns=policy_returns,
                macro_vectors=macro_vectors,
                direction_probs=direction_probs,
            )
        if log_exploit_hd_scores or log_exploit_detail:
            self._log_exploit_preplay_hd_scores(
                diagnostics=preplay_diagnostics,
                chosen_heading_deg=self.action_heading_deg,
                preplay_heading_deg=preplay_direction_deg,
                best_idx=best_idx,
                macro_returns=macro_returns,
                macro_vectors=macro_vectors,
                direction_probs=direction_probs,
            )
        self._write_preplay_decision_diagnostics(
            diagnostics=preplay_diagnostics,
            chosen_heading_deg=self.action_heading_deg,
            preplay_heading_deg=preplay_direction_deg,
            best_idx=best_idx,
            macro_returns=macro_returns,
            policy_returns=policy_returns,
            direction_probs=direction_probs,
            sampling_variances=sampling_variances,
            expected_value=expected_value,
            curr_x=curr_x,
            curr_z=curr_z,
        )
        self._execute_movement(self.action_heading_deg)

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
        # Advance simulation one timestep
        self.step(self.timestep)

        # Get the latest boundary data from range finder
        boundaries = self.range_finder.getRangeImage()

        # Update global heading (0–360)
        # Boundary conversion: raw compass/controller frame -> world/math frame.
        self.current_compass_values = [float(v) for v in self.compass.getValues()]
        compass_heading = self._normalize_heading_deg(
            self.get_bearing_in_degrees(self.current_compass_values)
        )
        self.current_compass_heading_deg_exact = compass_heading
        self.current_compass_heading_deg = compass_heading
        self.current_heading_deg_exact = self._compass_heading_to_world_heading_deg(compass_heading)
        self.current_heading_deg = self.current_heading_deg_exact

        # Shift boundary data based on global heading
        # Boundary and HD inputs consume the corrected world heading.
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * int(round(self.current_heading_deg)),
        )
        finite_boundaries = self.boundaries[torch.isfinite(self.boundaries)]
        self.current_min_lidar_distance = (
            float(torch.min(finite_boundaries).item())
            if int(finite_boundaries.numel()) > 0
            else float(self._diag_default_clearance)
        )
        self._diag_min_lidar_since_log = min(
            float(self._diag_min_lidar_since_log),
            float(self.current_min_lidar_distance),
        )
        if self.use_prox_mod:
            self.prox = self.compute_proximity(boundaries)

        # Convert heading to radians for HD-layer input
        current_heading_rad = np.deg2rad(self.current_heading_deg)
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
        self._diag_collision_latch[0] = max(self._diag_collision_latch[0], int(self.collided[0].item()))
        self._diag_collision_latch[1] = max(self._diag_collision_latch[1], int(self.collided[1].item()))


    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """
        Converts a Webots compass north vector to the raw controller compass
        heading in degrees [0, 360). This is not the internal world heading.
        """
        rad = np.arctan2(north[1], north[0])
        bearing = (rad - (math.pi / 2.0)) / np.pi * 180.0
        if bearing < 0:
            bearing += 360.0
        return bearing

    def compute_proximity(self, boundaries):
        """
        Compute the literal nearest-wall distance from LiDAR readings.
        """
        lidar_tensor = torch.as_tensor(boundaries, dtype=self.dtype, device=self.device)
        finite_lidar = lidar_tensor[torch.isfinite(lidar_tensor)]
        if int(finite_lidar.numel()) == 0:
            return torch.as_tensor(self.max_dist, dtype=self.dtype, device=self.device)
        return torch.clamp(torch.min(finite_lidar), min=0.0, max=float(self.max_dist))

    def _compute_stdp_transition_eligibility(self) -> float:
        """Return transition quality for recurrent STDP/decay on the current PCN update."""
        current_collision = bool(torch.any(self.collided).item()) if hasattr(self, "collided") else False
        collision_latched = bool(
            np.any(getattr(self, "_diag_collision_latch", np.zeros(2, dtype=np.int8)))
        )
        self.last_stdp_transition_collision_current = current_collision
        self.last_stdp_transition_collision_latched = collision_latched

        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_x, curr_z = self._current_position_xz(curr_pos)
        prev_position = getattr(self, "_last_diag_position", None)
        prev_heading = getattr(self, "_last_diag_heading_deg", None)
        prev_left_wheel = getattr(self, "_last_diag_left_wheel_position", None)
        prev_right_wheel = getattr(self, "_last_diag_right_wheel_position", None)
        curr_left_wheel = float(self.left_position_sensor.getValue())
        curr_right_wheel = float(self.right_position_sensor.getValue())

        self.last_stdp_transition_forward_progress = 0.0
        self.last_stdp_transition_wheel_travel = 0.0
        self.last_stdp_transition_progress_ratio = 0.0
        if prev_position is None:
            self.last_stdp_transition_displacement = 0.0
            self.last_stdp_transition_heading_error_deg = 0.0
            self.last_stdp_transition_eligibility = 0.0
            return 0.0

        dx = float(curr_x - float(prev_position[0]))
        dz = float(curr_z - float(prev_position[1]))
        displacement = float(math.hypot(dx, dz))
        self.last_stdp_transition_displacement = displacement

        if displacement <= 1e-8:
            heading_error = 180.0
        else:
            displacement_heading = self._heading_from_xz_vector(dx, dz)
            heading_error = abs(
                self._signed_heading_error_deg(
                    displacement_heading,
                    float(prev_heading if prev_heading is not None else getattr(self, "current_heading_deg", 0.0)),
                )
            )
        self.last_stdp_transition_heading_error_deg = float(heading_error)

        if not bool(getattr(self, "stdp_transition_quality_gate_enabled", True)):
            eligibility = 0.0 if current_collision else 1.0
            self.last_stdp_transition_eligibility = float(eligibility)
            return float(eligibility)

        if current_collision or collision_latched:
            self.last_stdp_transition_eligibility = 0.0
            return 0.0

        if prev_left_wheel is None or prev_right_wheel is None:
            self.last_stdp_transition_eligibility = 0.0
            return 0.0

        heading_rad = math.radians(
            float(prev_heading if prev_heading is not None else getattr(self, "current_heading_deg", 0.0))
        )
        forward_x = math.cos(heading_rad)
        forward_z = math.sin(heading_rad)
        forward_progress = float((dx * forward_x) + (dz * forward_z))
        left_travel = self.wheel_radius * (curr_left_wheel - float(prev_left_wheel))
        right_travel = self.wheel_radius * (curr_right_wheel - float(prev_right_wheel))
        wheel_travel = float(0.5 * (left_travel + right_travel))
        expected_travel = abs(wheel_travel)
        if expected_travel <= 1e-9:
            progress_ratio = 0.0
        else:
            progress_ratio = forward_progress / expected_travel

        self.last_stdp_transition_forward_progress = forward_progress
        self.last_stdp_transition_wheel_travel = wheel_travel
        self.last_stdp_transition_progress_ratio = float(progress_ratio)

        eligibility = min(1.0, max(0.0, float(progress_ratio)))
        self.last_stdp_transition_eligibility = eligibility
        return eligibility

    ########################################### COMPUTE ###########################################
    def compute_pcn_activations(self, learn: Optional[bool] = None):
        """
        Uses current boundary- and HD-activations to update place-cell activations
        and store relevant data for analysis/debugging.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        position = [curr_pos[0], curr_pos[2]]  # [x, z]

        if self.pcn is None:
            raise RuntimeError("Unified PCN has not been initialized.")

        # Keep GCN input generation on the original driver-managed path so the
        # unified ablation only changes PCN organization, not grid dynamics.
        self.grid_activations_list = []
        for scale_idx, gcn in enumerate(self.gcns):
            if gcn is not None:
                grid_activations = gcn.get_grid_cell_activations(position, use_mask=True)
                grid_activations = grid_activations.to(dtype=self.dtype, device=self.device)
            else:
                grid_start, grid_end = self.pcn.grid_boundaries[scale_idx : scale_idx + 2]
                grid_activations = torch.zeros(
                    grid_end - grid_start,
                    dtype=self.dtype,
                    device=self.device,
                )
            self.grid_activations_list.append(grid_activations)
        unified_grid_activations = (
            torch.cat(self.grid_activations_list)
            if self.grid_activations_list
            else torch.zeros(0, dtype=self.dtype, device=self.device)
        )
        transition_eligibility = self._compute_stdp_transition_eligibility()

        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            grid_activations=unified_grid_activations,
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided),
            proximity=float(self.prox) if self.use_prox_mod else None,
            learn=learn,
            transition_eligibility=transition_eligibility,
        )

        self.grid_activations_list = [
            act.clone().detach() for act in getattr(self.pcn, "last_grid_activations", [])
        ]
        self.pcn_activations_list = [
            act.clone().detach() for act in self.pcn.get_activations_per_scale()
        ]

        if self.plot_bvc and getattr(self.pcn, "bvc_layers", None):
            self.pcn.bvc_layers[0].plot_activation(self.boundaries.cpu())

        # Advance simulation one timestep
        self.step(self.timestep)

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        if bool(getattr(self, "_suppress_goal_checks_for_heading_test", False)):
            return

        curr_pos = self.robot.getField("translation").getSFVec3f()
        time_limit = 120 # minutes

        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING, RobotMode.PLOTTING_AUTO):
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

        elif self.robot_mode == RobotMode.EXPLOIT:
            # Check if either goal reached or time expired
            goal_pos = self.goals[0] if self.goals else {
                "location": self.goal_location,
                "radius": self.goal_r["exploit"],
            }
            goal_reached = self._distance_to_goal(goal_pos) <= float(goal_pos["radius"])
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
                    self.save(save_trajectory=True)
                    self.done = True
                    return
                else:
                    self.stop()
                    self.save(include_hmaps=True)
                    self.done = True
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                    return

        elif self.robot_mode in {RobotMode.LEARN_LOCATIONS_COVERAGE, RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO}:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

            # Check all goals for learning
            for goal in self.goals:
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)
                at_goal = bool((distance <= goal["radius"]).item())

                if at_goal:
                    if not goal["visited"]:
                        print(f"[LEARN_LOCATIONS_COVERAGE] First visit to {goal['name']} goal at {goal['location']}")
                        goal["visited"] = True
                self._update_goal_contact(goal, at_goal=at_goal)

            # Check termination conditions: (time limit OR coverage reached) AND learning complete
            minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
            coverage_reached = self._check_coverage_complete()
            learning_complete = self._check_multi_goal_learning_complete()

            if (minimum_time_reached or coverage_reached) and learning_complete:
                reason = "Coverage target reached" if coverage_reached else "Time limit reached"
                print(f"[LEARN_LOCATIONS_COVERAGE] {reason} and learning complete! "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")
                self.stop()

                for goal in self.goals:
                    self._finalize_goal_event(goal["name"])

                self._create_multi_goal_reward_maps()
                self._save_multi_goal_data()

                # Save trial completion time for AUTO mode
                if self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO:
                    self._save_trial_completion_time(trial_elapsed_time)

                self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                self.done = True

                # Only pause if not in AUTO mode, or if this is the last trial
                if (self.robot_mode != RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO or
                    self.current_auto_trial == self.num_auto_trials):
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            elif coverage_reached and not learning_complete:
                print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                      f"but learning incomplete, continuing...")
            elif minimum_time_reached and not learning_complete:
                print(f"[LEARN_LOCATIONS_COVERAGE] Time limit reached but learning incomplete, continuing...")

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
            current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

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

    def _distance_to_goal(self, goal) -> float:
        curr_pos = self.robot.getField("translation").getSFVec3f()
        return float(
            np.linalg.norm(
                [
                    curr_pos[0] - float(goal["location"][0]),
                    curr_pos[2] - float(goal["location"][1]),
                ]
            )
        )

    def _update_goal_contact(self, goal, at_goal: Optional[bool] = None):
        """Collect unified PC activation episodes while the robot remains inside a goal zone."""
        goal_name = goal["name"]
        if goal_name not in self._current_goal_event_by_goal or self.pcn is None:
            return

        if at_goal is None:
            at_goal = self._distance_to_goal(goal) <= float(goal["radius"])

        if at_goal:
            self._current_goal_event_by_goal[goal_name].append(
                self.pcn.place_cell_activations.detach().cpu()
            )
            now_s = float(self.getTime())
            if (not self.goal_currently_in[goal_name]) and (
                now_s - self.goal_last_count_time_s_by_goal[goal_name] >= self.goal_visit_cooldown_s
            ):
                self.goal_visit_counts[goal_name] += 1
                self.goal_last_count_time_s_by_goal[goal_name] = now_s
                print(
                    f"[LEARN_LOCATIONS] {goal_name} visit "
                    f"{self.goal_visit_counts[goal_name]}/{self.min_goal_visits}"
                )
            self.goal_currently_in[goal_name] = True
            return

        if self.goal_currently_in.get(goal_name, False):
            dist = self._distance_to_goal(goal)
            if dist > float(goal["radius"]) + self.goal_exit_hysteresis:
                self.goal_currently_in[goal_name] = False
                self._finalize_goal_event(goal_name)

    def _finalize_goal_event(self, goal_name: str):
        if goal_name not in self._current_goal_event_by_goal:
            return
        if self._current_goal_event_by_goal[goal_name]:
            self.goal_events_by_goal[goal_name].append(self._current_goal_event_by_goal[goal_name])
            self._current_goal_event_by_goal[goal_name] = []

    def _build_goal_reward(self, goal_name: str) -> bool:
        if self.rcn is None or self.pcn is None:
            return False
        goal_events = list(self.goal_events_by_goal.get(goal_name, []))
        if self._current_goal_event_by_goal.get(goal_name):
            goal_events.append(self._current_goal_event_by_goal[goal_name])
        if not goal_events:
            return False
        built = self.rcn.build_goal_reward_from_events(self.pcn, goal_events, replace=True)
        if built:
            print(f"[RCN] Built replay reward map from goal events for {goal_name}")
        return built

    def _check_multi_goal_learning_complete(self):
        """Check if multi-goal learning is complete"""
        if not all(goal["visited"] for goal in self.goals):
            return False

        for goal_name, visit_count in self.goal_visit_counts.items():
            if visit_count < self.min_goal_visits:
                return False
            finalized_events = len(self.goal_events_by_goal.get(goal_name, []))
            live_event = 1 if self._current_goal_event_by_goal.get(goal_name) else 0
            if finalized_events + live_event <= 0:
                return False

        return True

    def _create_multi_goal_reward_maps(self):
        """Create goal-specific unified reward maps from collected goal-contact events."""
        if self.pcn is None or self.rcn is None:
            raise RuntimeError("Unified PCN/RCN must be available before creating goal reward maps.")

        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        os.makedirs(multi_goal_dir, exist_ok=True)

        print(f"[LEARN_LOCATIONS] Creating {len(self.goals)} unified goal reward maps")

        created_maps = 0
        single_goal_rcn = None
        for goal in self.goals:
            goal_rcn = copy.deepcopy(self.rcn)
            goal_events = list(self.goal_events_by_goal.get(goal["name"], []))
            if self._current_goal_event_by_goal.get(goal["name"]):
                goal_events.append(self._current_goal_event_by_goal[goal["name"]])
            if not goal_events:
                print(f"[WARNING] Skipping goal {goal['name']}: no goal-contact events collected")
                continue
            if not goal_rcn.build_goal_reward_from_events(self.pcn, goal_events, replace=True):
                print(f"[WARNING] Skipping goal {goal['name']}: unable to build reward map from events")
                continue

            goal_rcn_path = os.path.join(
                multi_goal_dir, f"unified_rcn_goal_{goal['name']}.pkl"
            )
            with open(goal_rcn_path, "wb") as f:
                pickle.dump(goal_rcn, f)

            created_maps += 1
            print(f"[LEARN_LOCATIONS] Created unified goal reward map: {goal['name']}")

            if len(self.goals) == 1:
                single_goal_rcn = goal_rcn

        if single_goal_rcn is not None:
            self.rcn = single_goal_rcn
            print(
                f"[LEARN_LOCATIONS] Promoted sole goal reward map to default unified RCN: "
                f"{self.unified_rcn_path}"
            )

        print(f"[LEARN_LOCATIONS] Successfully created {created_maps} unified goal reward maps")

    def _save_multi_goal_data(self):
        """Save multi-goal specific data"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        association_data = {
            "goal_visit_counts": self.goal_visit_counts,
            "goal_event_counts": {
                goal["name"]: (
                    len(self.goal_events_by_goal.get(goal["name"], []))
                    + (1 if self._current_goal_event_by_goal.get(goal["name"]) else 0)
                )
                for goal in self.goals
            },
            "goals": self.goals,
            "scales": [{"scale_index": s["scale_index"], "name": s["name"]} for s in self.scales],
            "total_steps": self.step_count,
            "final_time": self.getTime()
        }
        with open(associations_path, "wb") as f:
            pickle.dump(association_data, f)

        print(f"[LEARN_LOCATIONS] Saved goal-event summary to {associations_path}")

        print(f"[LEARN_LOCATIONS] Final goal-event summary:")
        for goal_name, event_count in association_data["goal_event_counts"].items():
            goal_info = next(g for g in self.goals if g["name"] == goal_name)
            visit_count = self.goal_visit_counts[goal_name]
            print(
                f"  {goal_name} at {goal_info['location']}: "
                f"{event_count} events, {visit_count} visits"
            )

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
        self.save(include_hmaps=True)
        self.done = True

    def _handle_random_exploitation_timeout(self):
        """Handle timeout during EXPLOIT_LOCATIONS_RANDOM mode"""
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
        self.save(include_hmaps=True)
        self.done = True

    ########################################### AUTO PILOT ###########################################

    def auto_pilot(self):
        """
        A fallback or finalizing method that manually drives the robot to the goal
        location when it is close or already exploiting.
        """
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()

        # Keep moving until close enough to goal
        while not torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor(
                [curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device
            ),
            atol=self.goal_r["explore"],
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            desired_world_heading = self._heading_from_xz_vector(
                self.goal_location[0] - curr_pos[0],
                self.goal_location[1] - curr_pos[2],
            )
            self._execute_movement(desired_world_heading)
            s_start += self.tau_w

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
            target_heading_deg: Raw Webots compass/controller target heading in degrees [0, 360)
            debug: Whether to print debug information

        Returns:
            bool: True if turn was successful, False otherwise
        """
        MAX_SINGLE_TURN = 10.0  # Maximum degrees to turn in one step
        ACCEPTABLE_ERROR = 5.0  # Acceptable final error in degrees
        MAX_ATTEMPTS = 30       # Maximum number of turn attempts

        target_heading_deg = self._normalize_heading_deg(target_heading_deg)
        initial_heading = self._current_controller_heading_deg()

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
            pre_turn_heading = self._current_controller_heading_deg()
            self.turn(np.radians(turn_this_step))
            post_turn_heading = self._current_controller_heading_deg()

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
            heading_deg: Internal world target heading in degrees.
            show_debug: Whether to print debug info.

        Returns:
            bool: Success of turning/movement.
        """
        passive_exploit_modes = {
            RobotMode.EXPLOIT,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        }
        self.last_execute_movement_collision = False
        learn_flag = False if self.robot_mode in passive_exploit_modes else None
        log_neural_hmaps = self.robot_mode not in passive_exploit_modes
        world_heading_deg = self._normalize_heading_deg(heading_deg)
        controller_heading_deg = self._world_heading_to_controller_heading_deg(world_heading_deg)

        if show_debug:
            print(
                f"Turning from world={self.current_heading_deg:.0f} deg "
                f"(compass={self._current_controller_heading_deg():.0f} deg) "
                f"to world={world_heading_deg:.0f} deg "
                f"(controller={controller_heading_deg:.0f} deg)"
            )

        success = self.compass_based_turn_to_heading(controller_heading_deg, show_debug)

        if not success:
            if show_debug:
                print("TURNING FAILED - falling back to exploration")
            fallback_pre_pos = self.robot.getField("translation").getSFVec3f()
            self.turn(np.random.uniform(-np.pi/4, np.pi/4))
            for _ in range(3):
                if log_neural_hmaps:
                    self.sense()
                    self.compute_pcn_activations(learn=learn_flag)
                    self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
                    self.forward()
                else:
                    self.forward()
                    self.update_hmaps(update_loc=True, update_pcn=False, update_gcn=False)
                self.check_goal_reached()
            if self._should_log_exploit_detail() and self._is_exploit_mode():
                fallback_post_pos = self.robot.getField("translation").getSFVec3f()
                fallback_dx = float(fallback_post_pos[0] - fallback_pre_pos[0])
                fallback_dz = float(fallback_post_pos[2] - fallback_pre_pos[2])
                fallback_distance = math.hypot(fallback_dx, fallback_dz)
                fallback_angle = (
                    self._heading_from_xz_vector(fallback_dx, fallback_dz)
                    if fallback_distance > 1e-6
                    else float("nan")
                )
                print(
                    f"[EXPLOIT_MOVE] turn_failed target_world={world_heading_deg:.1f} "
                    f"target_controller={controller_heading_deg:.1f} "
                    f"compass_now={self._current_controller_heading_deg():.1f} "
                    f"fallback_move={fallback_distance:.3f}m "
                    f"fallback_angle={fallback_angle:.1f}"
                )
            return False

        # Record pre-movement position
        pre_move_pos = self.robot.getField("translation").getSFVec3f()
        pre_move_x, pre_move_z = self._current_position_xz(pre_move_pos)
        active_goal = self._active_goal_for_navigation()
        goal_desc = "none"
        goal_radius = float("nan")
        pre_goal_distance = float("nan")
        post_goal_distance = float("nan")
        pre_direct_goal_heading = float("nan")
        post_direct_goal_heading = float("nan")
        if active_goal is not None and active_goal.get("location") is not None:
            goal_x = float(active_goal["location"][0])
            goal_z = float(active_goal["location"][1])
            goal_radius = float(active_goal.get("radius", float("nan")))
            pre_goal_distance = math.hypot(goal_x - pre_move_x, goal_z - pre_move_z)
            pre_direct_goal_heading = self._heading_from_xz_vector(
                goal_x - pre_move_x,
                goal_z - pre_move_z,
            )
            goal_desc = f"{active_goal.get('name', 'goal')}@({goal_x:.2f},{goal_z:.2f})"

        # Move forward
        for _ in range(self.tau_w):
            if log_neural_hmaps:
                self.sense()
                self.compute_pcn_activations(learn=learn_flag)
                self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
                self.forward()
            else:
                self.forward()
                self.update_hmaps(update_loc=True, update_pcn=False, update_gcn=False)
            self.check_goal_reached()
            if self.done:
                return True
            if bool(torch.any(self.collided).item()):
                if show_debug:
                    print("MOVEMENT BLOCKED BY WALL")
                self.stop()
                self.last_execute_movement_collision = True
                if self._should_log_exploit_collision():
                    curr_pos = self.robot.getField("translation").getSFVec3f()
                    curr_x, curr_z = self._current_position_xz(curr_pos)
                    goal = self._active_goal_for_navigation()
                    goal_distance = float("nan")
                    goal_desc = "none"
                    if goal is not None and goal.get("location") is not None:
                        goal_x = float(goal["location"][0])
                        goal_z = float(goal["location"][1])
                        goal_distance = math.hypot(goal_x - curr_x, goal_z - curr_z)
                        goal_desc = f"{goal.get('name', 'goal')}@({goal_x:.2f},{goal_z:.2f})"
                    print(
                        f"[EXPLOIT_DEBUG] forward collision world_heading={float(self.current_heading_deg):.1f} "
                        f"compass_heading={float(self._current_controller_heading_deg()):.1f} "
                        f"pos=({curr_x:.2f},{curr_z:.2f}) goal={goal_desc} dist={goal_distance:.2f} "
                        f"left={int(self.collided[0].item())} right={int(self.collided[1].item())}"
                    )
                return False

            # Update rotation accumulator (kept for detection)
            if hasattr(self, 'last_heading_deg') and self.last_heading_deg is not None:
                heading_diff = self.current_heading_deg - self.last_heading_deg
                heading_diff = ((heading_diff + 180) % 360) - 180
                self.rotation_accumulator += abs(heading_diff)
                self.last_heading_deg = self.current_heading_deg
            else:
                # Initialize last_heading_deg if it's None
                self.last_heading_deg = self.current_heading_deg

        post_move_pos = self.robot.getField("translation").getSFVec3f()
        post_move_x, post_move_z = self._current_position_xz(post_move_pos)
        actual_dx = post_move_x - pre_move_x
        actual_dy = post_move_z - pre_move_z
        actual_distance = np.sqrt(actual_dx**2 + actual_dy**2)
        actual_angle = float("nan")
        angle_error = float("nan")

        if actual_distance > 0.001:
            actual_angle = np.degrees(np.arctan2(actual_dy, actual_dx))
            actual_angle = (actual_angle + 360) % 360
            angle_error = abs(actual_angle - world_heading_deg)
            if angle_error > 180:
                angle_error = 360 - angle_error

        entered_goal = False
        if active_goal is not None and active_goal.get("location") is not None:
            goal_x = float(active_goal["location"][0])
            goal_z = float(active_goal["location"][1])
            post_goal_distance = math.hypot(goal_x - post_move_x, goal_z - post_move_z)
            post_direct_goal_heading = self._heading_from_xz_vector(
                goal_x - post_move_x,
                goal_z - post_move_z,
            )
            entered_goal = bool(post_goal_distance <= goal_radius)

        if self._should_log_exploit_detail() and self._is_exploit_mode():
            print(
                f"[EXPLOIT_MOVE] target_world={world_heading_deg:.1f} "
                f"target_controller={controller_heading_deg:.1f} "
                f"actual_disp={actual_angle:.1f} "
                f"disp_err={angle_error:.1f} "
                f"move={actual_distance:.3f}m "
                f"goal={goal_desc} r={goal_radius:.2f} "
                f"dist={pre_goal_distance:.3f}->{post_goal_distance:.3f} "
                f"direct={pre_direct_goal_heading:.1f}->{post_direct_goal_heading:.1f} "
                f"entered={int(entered_goal)}"
            )

        # Verify movement if debug
        if show_debug:
            post_move_pos = self.robot.getField("translation").getSFVec3f()
            actual_dx = post_move_pos[0] - pre_move_pos[0]
            actual_dy = post_move_pos[2] - pre_move_pos[2]
            actual_distance = np.sqrt(actual_dx**2 + actual_dy**2)

            if actual_distance > 0.001:
                actual_angle = np.degrees(np.arctan2(actual_dy, actual_dx))
                actual_angle = (actual_angle + 360) % 360
                angle_error = abs(actual_angle - world_heading_deg)
                if angle_error > 180:
                    angle_error = 360 - angle_error

                status = "✓" if angle_error < 15 else "⚠" if angle_error < 30 else "✗"
                print(
                    f"Moved {actual_distance:.3f}m at {actual_angle:.0f} deg "
                    f"(world target: {world_heading_deg:.0f} deg, "
                    f"controller target: {controller_heading_deg:.0f} deg) {status}"
                )

        return True

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
            current_position = np.array(
                [self.hmap_loc[:, 2][i], self.hmap_loc[:, 0][i]]
            )
            next_position = np.array(
                [self.hmap_loc[:, 2][i + 1], self.hmap_loc[:, 0][i + 1]]
            )
            path_length += np.linalg.norm(next_position - current_position)

        return path_length

    def update_hmaps(self,
                    update_loc=False,
                    update_hdn=False,
                    update_pcn=False,
                    update_gcn=False,
                    update_scale_priority=False):
        """
        Store agent position, head direction activations, place cell activations,
        and grid cell activations.

        Parameters:
        - update_loc (bool): Whether to update agent location history.
        - update_hdn (bool): Whether to update head direction activations.
        - update_pcn (bool): Whether to update place cell activations.
        - update_gcn (bool): Whether to update grid cell activations.
        - update_scale_priority (bool): Whether to update scale priority (dominant scale index).
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # 1) Update agent location if requested
            if update_loc:
                self.hmap_loc[self.step_count] = curr_pos

            # 1.5) Persist step-level learning diagnostics for later transition analysis.
            diag = getattr(self, "hmap_learning_diagnostics", None)
            if diag is not None:
                diag["time_s"][self.step_count] = float(self.getTime())
                diag["x"][self.step_count] = float(curr_pos[0])
                diag["z"][self.step_count] = float(curr_pos[2])
                heading_deg = float(getattr(self, "current_heading_deg", 0.0))
                diag["heading_deg"][self.step_count] = heading_deg
                prev_heading = self._last_diag_heading_deg
                if prev_heading is None:
                    heading_delta = 0.0
                else:
                    heading_delta = ((heading_deg - prev_heading + 180.0) % 360.0) - 180.0
                diag["heading_delta_deg"][self.step_count] = float(heading_delta)

                prev_position = self._last_diag_position
                if prev_position is None:
                    displacement = 0.0
                else:
                    displacement = float(
                        np.linalg.norm(
                            np.asarray([curr_pos[0], curr_pos[2]], dtype=np.float32) - prev_position
                        )
                    )
                diag["step_displacement"][self.step_count] = displacement

                min_lidar_distance_current = float(
                    getattr(self, "current_min_lidar_distance", float(self._diag_default_clearance))
                )
                min_lidar_distance = float(
                    getattr(self, "_diag_min_lidar_since_log", min_lidar_distance_current)
                )
                diag["min_lidar_distance"][self.step_count] = min_lidar_distance
                diag["min_lidar_distance_current"][self.step_count] = min_lidar_distance_current

                collision_left_current = int(bool(self.collided[0].item())) if len(self.collided) > 0 else 0
                collision_right_current = int(bool(self.collided[1].item())) if len(self.collided) > 1 else 0
                collision_left = int(bool(self._diag_collision_latch[0]))
                collision_right = int(bool(self._diag_collision_latch[1]))
                diag["collision_left"][self.step_count] = collision_left
                diag["collision_right"][self.step_count] = collision_right
                diag["collision_any"][self.step_count] = int(bool(collision_left or collision_right))
                diag["collision_left_current"][self.step_count] = collision_left_current
                diag["collision_right_current"][self.step_count] = collision_right_current
                diag["collision_any_current"][self.step_count] = int(
                    bool(collision_left_current or collision_right_current)
                )
                diag["stdp_transition_eligibility"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_eligibility", 0.0)
                )
                diag["stdp_transition_displacement"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_displacement", displacement)
                )
                diag["stdp_transition_heading_error_deg"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_heading_error_deg", 0.0)
                )
                diag["stdp_transition_forward_progress"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_forward_progress", 0.0)
                )
                diag["stdp_transition_wheel_travel"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_wheel_travel", 0.0)
                )
                diag["stdp_transition_progress_ratio"][self.step_count] = float(
                    getattr(self, "last_stdp_transition_progress_ratio", 0.0)
                )
                diag["stdp_transition_collision_latched"][self.step_count] = int(
                    bool(getattr(self, "last_stdp_transition_collision_latched", False))
                )
                diag["stdp_transition_collision_current"][self.step_count] = int(
                    bool(getattr(self, "last_stdp_transition_collision_current", False))
                )
                diag["stdp_connection_decay_scale"][self.step_count] = float(
                    getattr(self.pcn, "last_connection_decay_scale", 0.0)
                ) if self.pcn is not None else 0.0
                diag["stdp_updated"][self.step_count] = int(
                    bool(getattr(self.pcn, "last_learning_stdp_active", False))
                ) if self.pcn is not None else 0
                diag["pcn_learning_step_count"][self.step_count] = int(
                    getattr(self.pcn, "learning_step_count", 0)
                ) if self.pcn is not None else 0

                self._last_diag_position = np.asarray([curr_pos[0], curr_pos[2]], dtype=np.float32)
                self._last_diag_heading_deg = heading_deg
                self._last_diag_left_wheel_position = float(self.left_position_sensor.getValue())
                self._last_diag_right_wheel_position = float(self.right_position_sensor.getValue())
                self._diag_collision_latch[:] = 0
                self._diag_min_lidar_since_log = float(self._diag_default_clearance)

            # 2) Update head direction activations (Direct assignment)
            if update_hdn:
                self.hmap_hdn[self.step_count] = self.hd_activations

            # 3) Dynamically resize hmap_pcn_activities if needed
            if update_pcn and len(self.hmap_pcn_activities) != len(self.pcn_activations_list):
                self.hmap_pcn_activities = [
                    torch.zeros((self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32)
                    for act in self.pcn_activations_list
                ]

        # 4) Update place cell activations for each scale (Direct assignment)
        scale_idx_map = {scale_def["scale_index"]: i for i, scale_def in enumerate(self.scales)}

        if update_pcn:
            for scale_def, act in zip(self.scales, self.pcn_activations_list):
                scale_idx = scale_def["scale_index"]

                # Ensure the scale index exists in the mapping
                if scale_idx not in scale_idx_map:
                    continue

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

        self.step_count += 1

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to the goal.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        # Get current position from the robot node
        curr_pos = self.robot.getField("translation").getSFVec3f()

        # Distance from current position to goal location
        distance_to_goal = torch.norm(
            torch.tensor(
                [
                    curr_pos[0] - self.goal_location[0],
                    curr_pos[2] - self.goal_location[1],
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
        - unified PCN if include_pcn=True
        - unified RCN if include_rcn=True
        - GCN networks (one file per scale) if include_gcn=True
        - The history maps if include_hmaps=True
        - The agent's path if save_trajectory=True
        """
        files_saved = []

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1) Save the unified PCN (if requested)
        # ----------------------------------------------------------------------
        if include_pcn:
            if self.pcn is not None:
                if self._is_exploit_mode():
                    print(
                        f"[DRIVER] Exploit mode active; leaving base PCN untouched at "
                        f"{self.unified_pcn_path}"
                    )
                else:
                    with open(self.unified_pcn_path, "wb") as f:
                        pickle.dump(self.pcn, f)
                    files_saved.append(self.unified_pcn_path)
            else:
                print("[WARN] Unified PCN is not available; nothing saved for PCN.")

        # ----------------------------------------------------------------------
        # 2) Save the unified RCN (if requested)
        # ----------------------------------------------------------------------
        if include_rcn:
            if self.rcn is not None:
                with open(self.unified_rcn_path, "wb") as f:
                    pickle.dump(self.rcn, f)
                files_saved.append(self.unified_rcn_path)
            else:
                print("[WARN] Unified RCN is not available; nothing saved for RCN.")

        # ----------------------------------------------------------------------
        # 2.5) Save each scale's GCN (if requested)
        # ----------------------------------------------------------------------
        if include_gcn:
            for scale_def, gcn in zip(self.scales, self.gcns):
                if gcn is not None:
                    scale_idx = scale_def["scale_index"]
                    gcn_path = os.path.join(self.network_dir, f"gcn_scale_{scale_idx}.pkl")
                    with open(gcn_path, "wb") as f:
                        pickle.dump(gcn, f)
                    files_saved.append(gcn_path)

        # ----------------------------------------------------------------------
        # 3) Save the history maps if requested
        # ----------------------------------------------------------------------
        if include_hmaps:
            use_trial_prefix = self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}

            if use_trial_prefix:
                trial_id = getattr(self, "trial_id", "default")
                prefix = f"{trial_id}_"
            else:
                prefix = ""

            # (a) Agent location
            hmap_loc_path = os.path.join(self.hmap_dir, f"{prefix}hmap_loc.pkl")
            with open(hmap_loc_path, "wb") as f:
                pickle.dump(self.hmap_loc[: self.step_count], f)
            files_saved.append(hmap_loc_path)

            if hasattr(self, "hmap_scale_priority") and (use_trial_prefix or self._is_exploit_mode()):
                hmap_scale_priority_path = os.path.join(self.hmap_dir, f"{prefix}hmap_scale_priority.pkl")
                with open(hmap_scale_priority_path, "wb") as f:
                    scale_priority_data = self.hmap_scale_priority[: self.step_count].cpu().numpy()
                    pickle.dump(scale_priority_data, f)
                files_saved.append(hmap_scale_priority_path)

            if not self._is_exploit_mode():
                # (b) Head direction history
                hmap_hdn_path = os.path.join(self.hmap_dir, f"{prefix}hmap_hdn.pkl")
                with open(hmap_hdn_path, "wb") as f:
                    pickle.dump(self.hmap_hdn[: self.step_count].cpu(), f)
                files_saved.append(hmap_hdn_path)

                # (c) Place-cell history maps for each scale
                if self.hmap_pcn_activities:
                    unified_pcn_history = np.concatenate(
                        [pc_history[: self.step_count].cpu().numpy() for pc_history in self.hmap_pcn_activities],
                        axis=1,
                    )
                    unified_pcn_path = os.path.join(self.hmap_dir, f"{prefix}hmap_pcn.pkl")
                    with open(unified_pcn_path, "wb") as f:
                        pickle.dump(unified_pcn_history, f)
                    files_saved.append(unified_pcn_path)

                for scale_def, pc_history in zip(self.scales, self.hmap_pcn_activities):
                    scale_idx = scale_def["scale_index"]
                    hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_pcn_scale_{scale_idx}.pkl")

                    with open(hmap_scale_path, "wb") as f:
                        pc_data = pc_history[: self.step_count].cpu().numpy()
                        pickle.dump(pc_data, f)
                    files_saved.append(hmap_scale_path)

                # (d) Grid-cell history maps for each scale
                for scale_def, gc_history in zip(self.scales, self.hmap_gcn_activities):
                    if gc_history.numel() > 0:
                        scale_idx = scale_def["scale_index"]
                        hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_gcn_scale_{scale_idx}.pkl")

                        with open(hmap_scale_path, "wb") as f:
                            gc_data = gc_history[: self.step_count].cpu().numpy()
                            pickle.dump(gc_data, f)
                        files_saved.append(hmap_scale_path)

                # (e) Step-level learning diagnostics for later STDP / wall-contact analysis
                if hasattr(self, "hmap_learning_diagnostics") and self.hmap_learning_diagnostics is not None:
                    hmap_learning_diag_path = os.path.join(self.hmap_dir, f"{prefix}hmap_learning_diagnostics.pkl")
                    learning_diag_data = {
                        key: value[: self.step_count]
                        for key, value in self.hmap_learning_diagnostics.items()
                    }
                    with open(hmap_learning_diag_path, "wb") as f:
                        pickle.dump(learning_diag_data, f)
                    files_saved.append(hmap_learning_diag_path)

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

            with open(hmap_loc_file, "wb") as f:
                pickle.dump(self.hmap_loc[:self.step_count], f)
                files_saved.append(hmap_loc_file)

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
        print("Saving Done!")

    def clear(self):
        """
        Removes unified/per-scale network files and any hmap files.

        This includes:
        - pcn_scale_*.pkl
        - unified/default RCN files
        - gcn_scale_*.pkl
        - pcn.pkl, rcn.pkl, gcn.pkl (legacy)
        - hmap_* files in self.hmap_dir
        """
        # 1. Remove all network files in the main network directory
        if os.path.exists(self.network_dir):
            for fname in os.listdir(self.network_dir):
                if (fname.startswith("pcn_scale_") or
                    fname == "pcn_unified.pkl" or
                    fname == "unified_rcn_goal.pkl" or
                    fname.startswith("rcn_scale_") or
                    fname.startswith("gcn_scale_")):
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

        # 3. Remove goal-specific reward-map files if present
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        if os.path.exists(multi_goal_dir):
            for fname in os.listdir(multi_goal_dir):
                if (
                    fname.startswith("unified_rcn_goal_")
                    or fname.startswith("rcn_scale_")
                    or fname == "goal_associations.pkl"
                ):
                    full_path = os.path.join(multi_goal_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

        # 4. Remove any scale-specific hmap files (and all hmap files) from self.hmap_dir
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

        print("[DRIVER] Finished clearing old PCNs, RCNs, GCNs, goal maps, and hmap files.")
