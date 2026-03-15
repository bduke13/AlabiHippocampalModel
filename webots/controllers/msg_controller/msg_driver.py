import numpy as np
from numpy.random import default_rng
import pickle
import os
import tkinter as tk
from tkinter import N, messagebox
from typing import Optional, List, Dict, Any
import torch
from controller import Supervisor
from astropy.stats import circmean
import random
import math
import copy
from collections import deque

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
from analysis.stats.stats_collector import stats_collector

# Replay step budget is proportional to lambda_s to normalize spread per time constant
STEPS_PER_LAMBDA = 8  # Adjust to push farther (higher) or be more local (lower)

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
        use_unified_multiscale: Optional[bool] = False,
        environment_size: Optional[List[float]] = None,
        grid_size: Optional[float] = None,
        coverage_percentage: Optional[float] = None,
        min_goal_visits: int = 3,
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
        goal_assoc_unique_topk: int = 16,
        goal_assoc_max_activation_drop: float = 0.08,
        record_experience_transitions: bool = True,
        two_phase_learning: bool = False,
        phase1_min_steps: int = 2500,
        phase1_max_steps: int = 20000,
        phase1_bin_size: float = 0.5,
        phase1_min_revisit_bins: int = 15,
        phase1_revisit_cosine_threshold: float = 0.90,
        phase1_revisit_window: int = 200,
        defer_experience_build_until_phase2_end: bool = True,
        goal_map_replay_timesteps: int = 18,
        hybrid_path_replay_weight: float = 0.8,
        hybrid_diffusion_replay_weight: float = 0.2,
        prune_experience_loops: bool = True,
        loop_prune_min_top1: float = 0.08,
        loop_prune_min_top1_to_top2_ratio: float = 1.15,
        pcn_gate_mode: str = "normal",
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
        # Goal-visit debounce settings:
        # - Cooldown prevents rapid re-counting from jitter near goal boundary.
        # - Exit hysteresis requires moving farther out before "leaving" the goal.
        self.goal_visit_cooldown_seconds = max(0.0, float(goal_visit_cooldown_seconds))
        self.goal_visit_cooldown_steps = max(
            1, int(round(self.goal_visit_cooldown_seconds / (self.timestep / 1000.0)))
        )
        self.goal_exit_hysteresis = max(0.0, float(goal_exit_hysteresis))
        self.pcn_gate_mode = str(pcn_gate_mode).strip().lower()
        if self.pcn_gate_mode not in {"normal", "no_gate_no_inhibition", "no_gate_with_inhibition"}:
            print(f"[DRIVER] Unknown pcn_gate_mode='{self.pcn_gate_mode}', falling back to 'normal'")
            self.pcn_gate_mode = "normal"
        self.proximity_mode = str(proximity_mode).strip().lower()
        if self.proximity_mode not in {"min", "trimmed_mean", "opposite_pair_percentile", "local_minima", "raw_local_minima"}:
            print(f"[DRIVER] Unknown proximity_mode='{self.proximity_mode}', falling back to 'min'")
            self.proximity_mode = "min"
        self.proximity_trimmed_sigma = max(0.1, float(proximity_trimmed_sigma))
        self.proximity_pair_percentile = float(
            min(100.0, max(0.0, proximity_pair_percentile))
        )
        self.gcn_scale_invariant = bool(gcn_scale_invariant)
        self.goal_assoc_unique_topk = int(max(1, goal_assoc_unique_topk))
        self.goal_assoc_max_activation_drop = float(
            min(0.5, max(0.0, goal_assoc_max_activation_drop))
        )
        self.record_experience_transitions = bool(record_experience_transitions)
        self._prev_unified_pcn_activations = None
        self.two_phase_learning = bool(two_phase_learning)
        if mode == RobotMode.LEARN_LOCATIONS_TWO_PHASE and not self.two_phase_learning:
            print("[DRIVER] Forcing two_phase_learning=True for LEARN_LOCATIONS_TWO_PHASE mode.")
            self.two_phase_learning = True
        self.phase1_min_steps = int(max(100, phase1_min_steps))
        self.phase1_max_steps = int(max(self.phase1_min_steps + 100, phase1_max_steps))
        self.phase1_bin_size = float(max(0.1, phase1_bin_size))
        self.phase1_min_revisit_bins = int(max(1, phase1_min_revisit_bins))
        self.phase1_revisit_cosine_threshold = float(min(1.0, max(0.0, phase1_revisit_cosine_threshold)))
        self.phase1_revisit_window = int(max(10, phase1_revisit_window))
        self.defer_experience_build_until_phase2_end = bool(
            defer_experience_build_until_phase2_end
        )
        self.goal_map_replay_timesteps = int(max(1, goal_map_replay_timesteps))
        self.hybrid_path_replay_weight = float(max(0.0, hybrid_path_replay_weight))
        self.hybrid_diffusion_replay_weight = float(max(0.0, hybrid_diffusion_replay_weight))
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
        # Webots world coordinates in this project are treated as [x, z, y]
        # with index 2 as vertical height.
        self.spawn_ground_y = 0.044
        self.upright_rotation = [0, 0, 1, 0]

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
                    self.spawn_ground_y
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
            self.robot.getField("translation").setSFVec3f(candidate)
            self.robot.getField("rotation").setSFRotation(self.upright_rotation)
            self.robot.resetPhysics()
        else:
            if self.start_loc is not None:
                self.robot.getField("translation").setSFVec3f([self.start_loc[0], self.start_loc[1], self.spawn_ground_y])
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

        # Load or init Grid Cell Networks / PCNs / RCNs
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
        # Diagnostics for post-run analysis of scale gating and recruitment behavior.
        self.diag_scale_indices = [int(s["scale_index"]) for s in self.scales]
        self.diag_prox_values = []
        self.diag_scale_pref_values = []
        self.diag_active_counts = []

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
        self.checkpoint_beta = 0.1         # reward decay rate with distance to goal
        self.checkpoint_boost_gamma = 0.5        # re-injection amplitude when replay wave hits a checkpoint
        self.checkpoint_boost_threshold = 0.05  # minimum dot-product activation to trigger boost
        self._valid_checkpoints = None           # set after _compute_valid_checkpoints() — all checkpoints
        self._load_checkpoints_from_scene()

        # For EXPLOIT_LOCATIONS_RANDOM, load goal-specific RCNs
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO} and hasattr(self, 'active_goal_name'):
            self._load_goal_specific_rcns(self.active_goal_name)
        # Also enforce per-goal maps for unified exploit when multi-goal target is available.
        self._ensure_goal_specific_rcn_loaded_for_exploit()

        self.plot_bvc = plot_bvc

        # Two-phase learning state (for coverage-learning modes).
        self.two_phase_phase = "single"
        self.phase2_start_step = None
        # Revisit-consistency stability tracking:
        # _phase1_bin_activations: bin_key -> last recorded activation vector
        # _phase1_revisit_cosines: deque of cosine similarities from revisited bins
        # _phase1_bins_with_revisits: set of bin keys visited more than once
        self._phase1_bin_activations = {}
        self._phase1_revisit_cosines = deque(maxlen=self.phase1_revisit_window)
        self._phase1_bins_with_revisits = set()
        if self.two_phase_learning and self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            self.two_phase_phase = "phase1_ojas"
            # Force phase-1 flags regardless of passed defaults.
            self._set_learning_flags(enable_ojas=True, enable_stdp=False)
            if self.defer_experience_build_until_phase2_end:
                self.record_experience_transitions = False
            print(
                f"[TRAIN-2P] Phase 1 start (OJAS only). "
                f"min_steps={self.phase1_min_steps}, bin_size={self.phase1_bin_size}m, "
                f"min_revisit_bins={self.phase1_min_revisit_bins}, "
                f"revisit_cos>={self.phase1_revisit_cosine_threshold:.3f}"
            )

        if self.use_unified_multiscale:
            print("[DRIVER] *** UNIFIED MULTI-SCALE MODE ENABLED ***")
            print("[DRIVER] Using adaptive cross-scale inhibition and unified replay")

        # Coverage-learning specific startup diagnostics.
        if self.robot_mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            ojas_enabled, stdp_enabled = self._get_learning_flags()
            print(
                f"[TRAIN] Coverage learning start | unified={self.use_unified_multiscale} "
                f"| OJAS={ojas_enabled} | STDP={stdp_enabled}"
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

    def _configure_unified_replay_settings(self, unified_rcn):
        """Apply controller-level replay blending knobs to a unified RCN instance."""
        if unified_rcn is None:
            return
        path_w = float(max(0.0, getattr(self, "hybrid_path_replay_weight", 0.8)))
        diff_w = float(max(0.0, getattr(self, "hybrid_diffusion_replay_weight", 0.2)))
        if path_w + diff_w <= 1e-12:
            path_w, diff_w = 0.8, 0.2
        unified_rcn.enable_hybrid_replay = True
        unified_rcn.path_replay_weight = path_w
        unified_rcn.diffusion_replay_weight = diff_w
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
        w_in_init_ratio = float(np.mean([s["w_in_init_ratio"] for s in scale_configs]))

        try:
            with open(path, "rb") as f:
                unified_pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing unified PCN from {path}")

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
            unified_pcn.scale_configs = scale_configs
            if hasattr(unified_pcn, "_build_d_opt_per_pc"):
                unified_pcn._build_d_opt_per_pc()

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
        # Give the RCN scale boundary info so observe_transition can record per-scale top-k.
        if hasattr(self.unified_pcn, "scale_boundaries"):
            self.unified_rcn.scale_boundaries = self.unified_pcn.scale_boundaries

    def _load_goal_specific_rcns(self, goal_name):
        """Load goal-specific RCNs for EXPLOIT_LOCATIONS_RANDOM mode"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        if not os.path.exists(multi_goal_dir):
            print(f"[WARNING] Multi-goal rewards directory not found: {multi_goal_dir}")
            print(f"[WARNING] Make sure to run LEARN_HEBB or LEARN_LOCATIONS_COVERAGE first!")
            return

        print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")

        if self.use_unified_multiscale:
            unified_goal_path = os.path.join(multi_goal_dir, f"unified_rcn_goal_{goal_name}.pkl")
            try:
                with open(unified_goal_path, "rb") as f:
                    self.unified_rcn = pickle.load(f)
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
                self._configure_unified_replay_settings(self.unified_rcn)
                self.rcns = [self.unified_rcn]
                self.rcn = self.unified_rcn
                self.loaded_goal_specific_rcn_goal = goal_name
                print(f"[DRIVER] Loaded goal-specific unified RCN: {unified_goal_path}")
                return
            except FileNotFoundError:
                print(f"[ERROR] Goal-specific unified RCN not found: {unified_goal_path}")
                print(f"[ERROR] Make sure LEARN_HEBB/LEARN_LOCATIONS_COVERAGE has produced unified goal maps.")
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
                print(f"[ERROR] Make sure LEARN_HEBB/LEARN_LOCATIONS_COVERAGE has been run for this goal!")
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
                        "module_scale_ratio": float(scale_def.get("module_scale_ratio", 1.6)),
                        "translation_scale": scale_def.get("translation_scale", 1.0),
                        "mask_resolution": scale_def.get("mask_resolution", 128),
                        "smooth_sigma": scale_def.get("smooth_sigma", 1.5),
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
                module_scale_ratio = float(scale_def.get("module_scale_ratio", 1.6))
                translation_scale = scale_def.get("translation_scale", 1.0)
                mask_resolution = scale_def.get("mask_resolution", 128)
                smooth_sigma = scale_def.get("smooth_sigma", 1.5)
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
                    device=self.device.type,
                    dtype=self.dtype,
                )

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

    def _select_goal_association_pc(self, scale_acts: torch.Tensor, goal_name: str, scale_idx: int) -> int:
        """
        Select a goal association PC using uniqueness-aware top-k tie breaking.
        Prefers an unused near-top candidate over a reused global hub.
        """
        if scale_acts.numel() == 0:
            return 0
        top_k = int(min(self.goal_assoc_unique_topk, int(scale_acts.numel())))
        vals, idx = torch.topk(scale_acts, k=top_k)
        idx = idx.detach().cpu().tolist()
        vals = vals.detach().cpu().tolist()

        used = self._get_used_goal_pcs(scale_idx=scale_idx, exclude_goal=goal_name)
        top_val = float(vals[0]) if vals else 0.0
        min_allowed = top_val * (1.0 - self.goal_assoc_max_activation_drop)
        for cand_idx, cand_val in zip(idx, vals):
            if float(cand_val) < min_allowed:
                continue
            if int(cand_idx) not in used:
                return int(cand_idx)
        return int(idx[0]) if idx else 0

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
                print(
                    f"[DRIVER] Goal visit debounce enabled: cooldown={self.goal_visit_cooldown_seconds:.1f}s "
                    f"({self.goal_visit_cooldown_steps} steps), exit_hysteresis={self.goal_exit_hysteresis:.2f}m"
                )

            # Initialize coverage tracking for modes that use it
            if self.robot_mode in {
                RobotMode.LEARN_LOCATIONS_COVERAGE,
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
        if not (self.two_phase_learning and self.two_phase_phase == "phase1_ojas"):
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
        bx = int(raw_pos[0] / self.phase1_bin_size)
        by = int(raw_pos[1] / self.phase1_bin_size)
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

        # Update stored activation for this bin (tracks latest visit).
        self._phase1_bin_activations[bin_key] = vec.clone()

    def _two_phase_allows_counting(self) -> bool:
        """
        Return whether coverage/goal counters should be updated in current step.
        In two-phase mode, counting is phase-2 only.
        """
        if not self.two_phase_learning:
            return True
        return self.two_phase_phase == "phase2_stdp"

    def _phase1_is_stable(self) -> bool:
        """
        Return True when place field coding is stable by revisit consistency.

        Stability requires:
        1. Minimum step count reached (safety floor).
        2. Enough distinct bins have been revisited at least once.
        3. Enough revisit cosine samples collected.
        4. Mean revisit cosine similarity meets the threshold.

        Fallback: if step_count >= phase1_max_steps, force-transition regardless
        of stability so goal counting is never permanently blocked.
        """
        if self.step_count < self.phase1_min_steps:
            return False
        # Hard cap: force transition if phase-1 runs too long.
        if self.step_count >= self.phase1_max_steps:
            revisit_bins = len(self._phase1_bins_with_revisits)
            cos_mean = float(np.mean(self._phase1_revisit_cosines)) if self._phase1_revisit_cosines else 0.0
            print(
                f"[TRAIN-2P][PHASE1] Max steps ({self.phase1_max_steps}) reached — "
                f"force-transitioning to phase 2. "
                f"revisit_bins={revisit_bins}, revisit_cos={cos_mean:.4f}"
            )
            return True
        if len(self._phase1_bins_with_revisits) < self.phase1_min_revisit_bins:
            return False
        if len(self._phase1_revisit_cosines) < max(10, self.phase1_min_revisit_bins):
            return False
        cos_mean = float(np.mean(self._phase1_revisit_cosines))
        return cos_mean >= self.phase1_revisit_cosine_threshold

    def _reset_learning_targets_for_phase2(self):
        """Reset coverage and goal-visit/association state at phase-2 start."""
        if hasattr(self, "coverage_grid"):
            self.coverage_grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.visited_cells = 0
            self.current_coverage_percentage = 0.0
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
        if hasattr(self, "goal_place_cell_associations"):
            for goal_name in self.goal_place_cell_associations.keys():
                self.goal_place_cell_associations[goal_name] = [None] * len(self.scales)
                self.goal_association_step[goal_name] = [None] * len(self.scales)
                self.goal_place_cell_activations[goal_name] = [None] * len(self.scales)

    def _start_phase2_stdp(self):
        """Switch from phase-1 OJAS stabilization to phase-2 STDP learning."""
        if not (self.two_phase_learning and self.two_phase_phase == "phase1_ojas"):
            return
        revisit_bins = len(self._phase1_bins_with_revisits)
        cos_mean = float(np.mean(self._phase1_revisit_cosines)) if self._phase1_revisit_cosines else 0.0
        self.two_phase_phase = "phase2_stdp"
        self.phase2_start_step = int(self.step_count)
        self._set_learning_flags(enable_ojas=False, enable_stdp=True)
        self._reset_learning_targets_for_phase2()
        if hasattr(self, "unified_rcn") and hasattr(self.unified_rcn, "reset_experience_transitions"):
            self.unified_rcn.reset_experience_transitions()
        # Phase-2 transition handling:
        # - deferred mode: build once from history at the end (no per-step overhead)
        # - online mode: collect transitions each step during phase-2
        self.record_experience_transitions = not self.defer_experience_build_until_phase2_end
        print("=" * 80)
        print(
            f"[TRAIN-2P][PHASE SWITCH] PHASE 1 -> PHASE 2 at step={self.phase2_start_step} "
            f"(revisit_bins={revisit_bins}, revisit_cos={cos_mean:.4f})."
        )
        print(
            "[TRAIN-2P][PHASE SWITCH] Phase 2 uses STDP with coverage/goal counting enabled; "
            "all counters were reset at this transition."
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
        if end - start < 2:
            return
        if hasattr(self.unified_rcn, "reset_experience_transitions"):
            self.unified_rcn.reset_experience_transitions()

        scale_arrays = [h.detach().cpu() for h in self.hmap_pcn_activities]

        # Build a compact index path and optionally cancel loops:
        # if a dominant state repeats, remove the cycle between repeats.
        time_indices = list(range(start, end))
        num_dominant_valid = 0
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

            pruned_time_indices = []
            pruned_dom_ids = []
            state_to_pos = {}
            for t, dom in dominant_pairs:
                if dom in state_to_pos:
                    cut = state_to_pos[dom]
                    pruned_time_indices = pruned_time_indices[: cut + 1]
                    pruned_dom_ids = pruned_dom_ids[: cut + 1]
                    state_to_pos = {d: i for i, d in enumerate(pruned_dom_ids)}
                else:
                    state_to_pos[dom] = len(pruned_time_indices)
                    pruned_time_indices.append(t)
                    pruned_dom_ids.append(dom)
            # Fallback to full (unpruned) timeline if filtering became too sparse.
            if len(pruned_time_indices) >= 2:
                time_indices = pruned_time_indices

        num_edges = 0
        for i in range(len(time_indices) - 1):
            t_prev = time_indices[i]
            t_curr = time_indices[i + 1]
            prev = torch.cat([arr[t_prev] for arr in scale_arrays], dim=0)
            curr = torch.cat([arr[t_curr] for arr in scale_arrays], dim=0)
            self.unified_rcn.observe_transition(prev, curr)
            num_edges += 1
        print(
            f"[TRAIN-2P] Built deferred experience transitions from history: "
            f"steps={end-start}, compact_steps={len(time_indices)}, edges={num_edges}, "
            f"loop_prune={self.prune_experience_loops}, valid_dom={num_dominant_valid}"
        )

    def _maybe_log_training_progress(self):
        """Periodic verbose diagnostics for coverage-learning runs."""
        if self.robot_mode not in {
            RobotMode.LEARN_LOCATIONS_COVERAGE,
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
                f"s{item['scale_idx']}:share={item['grid_share']:.2f}"
                for item in per_scale
            ]
        )
        print(
            f"[GRID-DIAG] step={self.step_count} "
            f"| global_share={diag.get('grid_share', 0.0):.3f} "
            f"| gc_active={diag.get('gc_active_frac', 0.0):.3f} "
            f"| gc_mean={diag.get('gc_mean', 0.0):.3f} "
            f"| gc_max={diag.get('gc_max', 0.0):.3f} "
            f"| {per_scale_str}"
        )

    def _update_distance_tracking(self):
        """Update total distance traveled for random spawn mode"""
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            current_pos = self.robot.getField("translation").getSFVec3f()
            current_position_2d = [current_pos[0], current_pos[1]]  # [x, y] planar

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
            if abs(float(curr_pos[2]) - float(self.spawn_ground_y)) > 0.12:
                return True
        except Exception:
            pass

        try:
            orientation = self.robot.getOrientation()
            # Conservative tilt heuristic from rotation matrix.
            if orientation and len(orientation) == 9:
                return float(orientation[8]) < min_upright_cos
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
        self.stop()
        self.robot.getField("translation").setSFVec3f([
            float(curr_pos[0]),
            float(curr_pos[1]),
            float(self.spawn_ground_y)
        ])
        self.robot.getField("rotation").setSFRotation(self.upright_rotation)
        self.robot.resetPhysics()
        print(f"[RECOVERY] Upright reset performed ({reason}) at x={curr_pos[0]:.2f}, z={curr_pos[1]:.2f}")

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
            elif self.robot_mode in (RobotMode.LEARN_OJAS,
                                      RobotMode.LEARN_HEBB,
                                      RobotMode.DMTP,
                                      RobotMode.PLOTTING,
                                      RobotMode.PLOTTING_AUTO,
                                      RobotMode.PLOTTING_COVERAGE_AUTO,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                                      RobotMode.LEARN_LOCATIONS_TWO_PHASE):
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
                RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
                RobotMode.LEARN_LOCATIONS_TWO_PHASE,
                RobotMode.LEARN_HEBB,
                RobotMode.PLOTTING_COVERAGE_AUTO,
            }:
                if self._two_phase_allows_counting():
                    curr_pos = self.robot.getField("translation").getSFVec3f()
                    robot_pos = [curr_pos[0], curr_pos[1]]  # [x, y] planar coordinates
                    self._update_coverage(robot_pos)


            # Update distance tracking if in EXPLOIT_LOCATIONS_RANDOM mode
            if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
                self._update_distance_tracking()

            self.update_hmaps(update_loc=True,
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
        debug_print_interval = 100         # Print debug info every N steps

        #===================================================================

        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()

        self.compute_pcn_activations()
        self._maybe_log_grid_diagnostics()
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

        if self.use_unified_multiscale:
            # --- Unified reliability: single scalar in [reliability_min_floor, 1.0] ---
            if not hasattr(self, 'unified_reliability'):
                self.unified_reliability = 1.0

            (
                final_direction_deg,
                expected_value,
                combined_vector,
                macro_returns,
                sampling_variances,
                direction_probs,
            ) = self.unified_pcn.unified_preplay_sampling(
                unified_rcn=self.unified_rcn,
                n_hd=self.n_hd,
                num_steps=num_preplay_steps,
                discount_factor=discount_factor,
                within_direction_beta=within_scale_beta,
                num_samples=num_samples_per_direction,
                sampling_strategy=sampling_strategy,
                sampling_temperature=sampling_temperature,
                debug=debug_enabled,
            )

            safe_mask = distances_per_hd >= min_safe_distance
            safe_returns = macro_returns.clone()
            safe_returns[~safe_mask] = 0.0

            # Apply reliability: low trust → require stronger signal before committing
            effective_returns = safe_returns * self.unified_reliability
            if torch.max(effective_returns).item() < 0.1:
                self.explore()
                # Suppress last_scale_weights so the outer loop detection skips attribution
                self.last_scale_weights = None
                return

            safe_angles = torch.linspace(
                0,
                2 * np.pi * (1 - 1 / self.n_hd),
                self.n_hd,
                device=self.device,
                dtype=self.dtype,
            )
            safe_sin = torch.sum(torch.sin(safe_angles) * effective_returns)
            safe_cos = torch.sum(torch.cos(safe_angles) * effective_returns)
            action_angle = torch.atan2(safe_sin, safe_cos)
            if action_angle < 0:
                action_angle += 2 * np.pi

            self.action_heading_deg = float(torch.rad2deg(action_angle).item())
            self.scale_idx = 0
            self._execute_movement(self.action_heading_deg)

            # --- Update unified reliability based on collision outcome ---
            if torch.any(self.collided):
                self.unified_reliability = max(
                    self.unified_reliability * reliability_bad_factor,
                    reliability_min_floor,
                )
                if debug_enabled:
                    print(f"[UNIFIED-RELIABILITY] Collision penalty → reliability={self.unified_reliability:.3f}")
            else:
                self.unified_reliability = min(
                    self.unified_reliability * reliability_good_factor,
                    1.0,
                )

            # Set last_scale_weights to uniform so the outer loop detection can
            # accumulate rotation contributions (triggers forced exploration if needed).
            self.last_scale_weights = torch.ones(
                len(self.scales), dtype=self.dtype, device=self.device
            ) / len(self.scales)

            return

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
        self.current_heading_deg = int(
            self.get_bearing_in_degrees(self.compass.getValues())
        )

        # Shift boundary data based on global heading
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * self.current_heading_deg,
        )

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
        """Record per-step diagnostics for proximity, scale preference, and active-cell counts."""
        try:
            self.diag_prox_values.append(float(self.prox))
        except Exception:
            self.diag_prox_values.append(float("nan"))

        # Active cell counts per scale from current activation buffers.
        active_counts = []
        for act in getattr(self, "pcn_activations_list", []):
            if act is None:
                active_counts.append(0)
            else:
                active_counts.append(int(torch.sum(act > 0).item()))
        if active_counts:
            self.diag_active_counts.append(active_counts)

        # Scale preference is available in unified mode after activation update.
        pref = None
        if self.use_unified_multiscale and hasattr(self, "unified_pcn"):
            pref = getattr(self.unified_pcn, "last_scale_preference", None)
        if pref is not None:
            pref_np = pref.detach().float().cpu().numpy().astype(np.float32)
            self.diag_scale_pref_values.append(pref_np)

    def _build_scale_diagnostics_payload(self) -> Dict[str, Any]:
        """Summarize collected diagnostics into a compact serializable payload."""
        payload: Dict[str, Any] = {
            "steps_recorded": int(len(self.diag_prox_values)),
            "scale_indices": list(self.diag_scale_indices),
            "proximity_mode": str(getattr(self, "proximity_mode", "")),
            "proximity_pair_percentile": float(getattr(self, "proximity_pair_percentile", 0.0)),
            "proximity_stats": {},
            "proximity_histogram": {},
            "scale_preference_stats": {},
            "active_cell_stats": {},
        }

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

        if len(self.diag_scale_pref_values) > 0:
            pref_arr = np.asarray(self.diag_scale_pref_values, dtype=np.float64)
            n_scales = pref_arr.shape[1]
            for i in range(n_scales):
                vals = pref_arr[:, i]
                payload["scale_preference_stats"][str(self.diag_scale_indices[i])] = {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                }

        if len(self.diag_active_counts) > 0:
            act_arr = np.asarray(self.diag_active_counts, dtype=np.float64)
            n_scales = act_arr.shape[1]
            for i in range(n_scales):
                vals = act_arr[:, i]
                payload["active_cell_stats"][str(self.diag_scale_indices[i])] = {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "nonzero_step_ratio": float(np.mean(vals > 0)),
                }

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
            curr_pos = self.robot.getField("translation").getSFVec3f()
            position = [curr_pos[0], curr_pos[1]]  # [x, y] planar
            prev_unified_acts = self.unified_pcn.place_cell_activations.clone().detach()

            if self.gcn_scale_invariant:
                shared_gcn = next((g for g in self.gcns if g is not None), None)
                shared_acts = (
                    shared_gcn.get_grid_cell_activations(position, use_mask=True)
                    if shared_gcn is not None
                    else None
                )
                for gcn in self.gcns:
                    if gcn is not None and shared_acts is not None:
                        self.grid_activations_list.append(shared_acts.clone())
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
        curr_pos = self.robot.getField("translation").getSFVec3f()
        position = [curr_pos[0], curr_pos[1]]  # [x, y] planar

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
                    self.grid_activations_list.append(shared_acts.clone())
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
            torch.tensor([curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device),
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
                torch.tensor([curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device),
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
                    self.save(save_trajectory=True)
                    self.done = True
                    return
                else:
                    self.stop()
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
            current_position = torch.tensor([curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device)
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
                    self._handle_goal_learning(goal)
                else:
                    if (
                        self.goal_currently_in[goal_name]
                        and distance > (goal["radius"] + self.goal_exit_hysteresis)
                    ):
                        self.goal_currently_in[goal_name] = False

            coverage_reached = self._check_coverage_complete()
            learning_complete = self._check_multi_goal_learning_complete()

            if coverage_reached and learning_complete:
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
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        }:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = torch.tensor([curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device)

            # Two-phase mode: phase-1 runs OJAS-only until coding is stable, then
            # switches to phase-2 STDP and resets coverage/goal counters.
            if self.two_phase_learning and self.two_phase_phase == "phase1_ojas":
                if self._phase1_is_stable():
                    self._start_phase2_stdp()
                elif self.step_count % self.training_log_interval_steps == 0 and self.step_count > 0:
                    revisit_bins = len(self._phase1_bins_with_revisits)
                    cos_mean = float(np.mean(self._phase1_revisit_cosines)) if self._phase1_revisit_cosines else 0.0
                    print(
                        f"[TRAIN-2P][PHASE1] step={self.step_count} "
                        f"revisit_bins={revisit_bins}/{self.phase1_min_revisit_bins}, "
                        f"revisit_cos={cos_mean:.4f}/{self.phase1_revisit_cosine_threshold:.3f} "
                        f"| counting=DISABLED"
                    )
                return

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

                        self._handle_goal_learning(goal)
                else:
                    # Mark that robot has left this goal zone
                    if (
                        self.goal_currently_in[goal_name]
                        and distance > (goal["radius"] + self.goal_exit_hysteresis)
                    ):
                        self.goal_currently_in[goal_name] = False

            # Check termination conditions: (time limit OR coverage reached) AND learning complete AND minimum 5h elapsed
            minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
            coverage_reached = self._check_coverage_complete()
            learning_complete = self._check_multi_goal_learning_complete()
            minimum_5h_reached = trial_elapsed_time >= 5 * 3600
            if not counting_allowed:
                minimum_time_reached = False
                coverage_reached = False
                learning_complete = False
                minimum_5h_reached = False

            if (minimum_time_reached or coverage_reached) and learning_complete and minimum_5h_reached:
                reason = "Coverage target reached" if coverage_reached else "Time limit reached"
                print(f"[LEARN_LOCATIONS_COVERAGE] {reason} and learning complete! "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")
                self.stop()

                # In two-phase mode, build path-transition graph after STDP phase from
                # stored PC trajectories to avoid laggy per-step transition updates.
                if (
                    self.two_phase_learning
                    and self.defer_experience_build_until_phase2_end
                    and self.use_unified_multiscale
                ):
                    self._build_experience_transitions_from_hmaps()

                # Filter checkpoint visit log to remove dead-end traversals
                self._compute_valid_checkpoints()

                # Create reward maps for each goal
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
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                          f"but learning incomplete, continuing...")
                    self.last_coverage_incomplete_log_step = self.step_count
            elif minimum_time_reached and not learning_complete:
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Time limit reached but learning incomplete, continuing...")
                    self.last_coverage_incomplete_log_step = self.step_count
            elif (minimum_time_reached or coverage_reached) and learning_complete and not minimum_5h_reached:
                if (self.step_count - self.last_coverage_incomplete_log_step) >= self.coverage_incomplete_log_interval_steps:
                    remaining = max(0.0, 5 * 3600 - trial_elapsed_time)
                    print(f"[LEARN_LOCATIONS_COVERAGE] Ready to finish but waiting for 5h minimum "
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
            current_position = torch.tensor([curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device)

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

    def _check_multi_goal_learning_complete(self):
        """Check if multi-goal learning is complete"""
        # All goals must be visited
        if not all(goal["visited"] for goal in self.goals):
            return False

        # Each goal must have at least one place-cell association (not necessarily all scales).
        for goal_name, associations in self.goal_place_cell_associations.items():
            if all(pc_idx is None for pc_idx in associations):
                return False

        # All goals must have minimum number of visits
        for goal_name, visit_count in self.goal_visit_counts.items():
            if visit_count < self.min_goal_visits:
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

    def _compute_reward_weights(self, goal_x, goal_z, _w_exp_norm, base_rcn):
        """
        Build reward weights for any (goal_x, goal_z) position using trajectory Gaussian seed
        and reverse-replay through the experience transition matrix. Used for both real goals
        and doorway sub-goals.

        Args:
            goal_x (float): Goal x-coordinate (hmap_loc[:, 0] convention)
            goal_z (float): Goal z-coordinate (hmap_loc[:, 1] convention)
            _w_exp_norm (torch.Tensor): Row-normalised reverse transition matrix (already on device)
            base_rcn: RCN to deepcopy; provides device and C_REWARD

        Returns:
            RCN: Deepcopy of base_rcn with trained w_in/w_in_effective
        """
        goal_rcn = copy.deepcopy(base_rcn)
        goal_rcn.w_in = torch.zeros_like(goal_rcn.w_in)
        goal_rcn.w_in_effective = goal_rcn.w_in.clone()
        goal_rcn.reward_cell_activations = torch.zeros_like(goal_rcn.reward_cell_activations)

        rcn_device = goal_rcn.w_in.device
        hmap_ok = (
            bool(self.hmap_pcn_activities)
            and self.step_count > 0
            and not getattr(self, "lightweight_hmaps", False)
        )
        n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
        hmap_x = torch.from_numpy(self.hmap_loc[:n_valid, 0]).float().to(rcn_device)
        hmap_z = torch.from_numpy(self.hmap_loc[:n_valid, 1]).float().to(rcn_device)
        dist_sq_all = (hmap_x - goal_x) ** 2 + (hmap_z - goal_z) ** 2

        seed_activations = torch.zeros(self.unified_pcn.num_pc_total, device=rcn_device)
        if hmap_ok:
            for s_idx in range(len(self.scales)):
                s_start = self.unified_pcn.scale_boundaries[s_idx]
                s_end = self.unified_pcn.scale_boundaries[s_idx + 1]
                if s_idx >= len(self.hmap_pcn_activities):
                    continue
                acts = self.hmap_pcn_activities[s_idx][:n_valid].float().to(rcn_device)
                sigma_s = float(self.scales[s_idx].get("sigma_r", 1.0)) * 2.0
                w_g = torch.exp(-dist_sq_all / (2.0 * sigma_s ** 2))
                w_sum = w_g.sum()
                if w_sum < 1e-12:
                    continue
                seed_activations[s_start:s_end] = torch.mv(acts.T, w_g / w_sum)

        C_REWARD = float(getattr(goal_rcn, "C_REWARD", 5.0))
        weight_update = torch.zeros_like(goal_rcn.w_in)
        unified_replay_steps = 50
        lambda_unified = 10.0
        A_unified = C_REWARD / max(lambda_unified, 1e-6)

        v = seed_activations
        for t in range(unified_replay_steps):
            decay = math.exp(-t / lambda_unified)
            norm_val = torch.sqrt(torch.max(
                torch.sum(v ** 2), torch.tensor(1e-12, device=rcn_device)
            ))
            v_norm = v / norm_val
            v_norm = torch.where(torch.isnan(v_norm), torch.zeros_like(v_norm), v_norm)
            weight_update[0, :] += A_unified * decay * v
            v = torch.tanh(torch.relu(torch.matmul(_w_exp_norm, v_norm) + v_norm))

        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val
        goal_rcn.w_in = torch.clamp(goal_rcn.w_in + weight_update, min=0.0)
        goal_rcn.w_in_effective = torch.clamp(goal_rcn.w_in.clone(), min=0.0)
        return goal_rcn

    def _build_room_topology_from_definitions(self):
        """
        Build room topology from self.room_definitions and self.doorway_definitions.
        Computes place cell CoMs from training trajectory, assigns cells to rooms,
        identifies doorway cells by proximity, and builds the room graph.
        Populates self.room_topology.
        """
        if not self.room_definitions:
            print("[TOPOLOGY] No room_definitions set; skipping topology build.")
            return

        n_rooms = len(self.room_definitions)
        N = self.unified_pcn.num_pc_total

        # ── Compute place cell centers-of-mass from training trajectory ──────────
        n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
        hmap_x = torch.from_numpy(self.hmap_loc[:n_valid, 0]).float()
        hmap_z = torch.from_numpy(self.hmap_loc[:n_valid, 1]).float()

        cell_cx = torch.zeros(N)
        cell_cz = torch.zeros(N)
        cell_active = torch.zeros(N, dtype=torch.bool)

        for s_idx in range(len(self.scales)):
            s_start = self.unified_pcn.scale_boundaries[s_idx]
            s_end = self.unified_pcn.scale_boundaries[s_idx + 1]
            if s_idx >= len(self.hmap_pcn_activities):
                continue
            _acts = self.hmap_pcn_activities[s_idx][:n_valid].float()
            _sums = _acts.sum(dim=0)
            _valid = _sums > 0.05
            if not _valid.any():
                continue
            _dev = _acts.device
            _cx = ((_acts.T @ hmap_x.to(_dev)) / _sums.clamp(min=1e-12))
            _cz = ((_acts.T @ hmap_z.to(_dev)) / _sums.clamp(min=1e-12))
            cell_cx[s_start:s_end] = _cx.cpu()
            cell_cz[s_start:s_end] = _cz.cpu()
            cell_active[s_start:s_end] = _valid.cpu()

        # ── Assign cells to rooms via bounding box ────────────────────────────────
        room_assignments = [-1] * N
        for i in range(N):
            if not cell_active[i]:
                continue
            cx, cz = float(cell_cx[i]), float(cell_cz[i])
            for room_def in self.room_definitions:
                xmin, xmax, zmin, zmax = room_def["bounds"]
                if xmin <= cx <= xmax and zmin <= cz <= zmax:
                    room_assignments[i] = room_def["id"]
                    break

        n_assigned = sum(1 for r in room_assignments if r >= 0)
        print(f"[TOPOLOGY] Assigned {n_assigned}/{int(cell_active.sum())} active cells to {n_rooms} rooms")

        # ── Find doorway cells by proximity to doorway positions ──────────────────
        doorway_by_pair = {}
        for d_def in self.doorway_definitions:
            dx, dz = float(d_def["position"][0]), float(d_def["position"][1])
            radius = float(d_def.get("radius", 1.5))
            rA, rB = int(d_def["connects"][0]), int(d_def["connects"][1])
            pair = (min(rA, rB), max(rA, rB))
            cells = [
                i for i in range(N)
                if cell_active[i]
                and (float(cell_cx[i]) - dx) ** 2 + (float(cell_cz[i]) - dz) ** 2 <= radius ** 2
            ]
            doorway_by_pair[pair] = cells
            print(f"[TOPOLOGY] Doorway {pair}: {len(cells)} cells within r={radius}m of ({dx:.1f},{dz:.1f})")

        # ── Build room_graph ──────────────────────────────────────────────────────
        room_graph = {r: [] for r in range(n_rooms)}
        for (rA, rB), dcells in doorway_by_pair.items():
            room_graph[rA].append((rB, dcells))
            room_graph[rB].append((rA, dcells))

        # ── Determine goal room for each goal ─────────────────────────────────────
        goal_rooms = {}
        for goal in self.goals:
            gx, gz = float(goal["location"][0]), float(goal["location"][1])
            goal_rooms[goal["name"]] = 0  # default to first room
            for room_def in self.room_definitions:
                xmin, xmax, zmin, zmax = room_def["bounds"]
                if xmin <= gx <= xmax and zmin <= gz <= zmax:
                    goal_rooms[goal["name"]] = room_def["id"]
                    break
            print(f"[TOPOLOGY] Goal '{goal['name']}' -> room {goal_rooms[goal['name']]}")

        self.room_topology = {
            "n_rooms": n_rooms,
            "room_assignments": room_assignments,
            "doorway_by_pair": doorway_by_pair,
            "room_graph": room_graph,
            "goal_rooms": goal_rooms,
            "cell_coms": (cell_cx, cell_cz),
        }
        print(f"[TOPOLOGY] Built: {n_rooms} rooms, {len(doorway_by_pair)} doorway pair(s)")

    def _bfs_room_path(self, from_room, goal_room, room_graph):
        """BFS over room_graph; returns list of room ids from from_room to goal_room, or None."""
        if from_room == goal_room:
            return [from_room]
        visited = {from_room}
        queue = deque([[from_room]])
        while queue:
            path = queue.popleft()
            curr = path[-1]
            for (neighbor, _) in room_graph.get(curr, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal_room:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)
        return None

    def _create_doorway_rcns(self, _w_exp_norm):
        """
        For each goal × non-goal room, build a doorway sub-RCN seeded at the doorway
        position and save it to disk. Uses _compute_reward_weights (same logic as real goals).
        """
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        room_graph = self.room_topology["room_graph"]
        doorway_by_pair = self.room_topology["doorway_by_pair"]
        n_rooms = self.room_topology["n_rooms"]
        goal_rooms = self.room_topology["goal_rooms"]
        cell_cx, cell_cz = self.room_topology["cell_coms"]

        for goal in self.goals:
            goal_name = goal["name"]
            goal_room = goal_rooms.get(goal_name)
            if goal_room is None:
                print(f"[DOORWAY_RCN] Cannot determine room for goal '{goal_name}'; skipping.")
                continue

            for from_room_id in range(n_rooms):
                if from_room_id == goal_room:
                    continue
                path = self._bfs_room_path(from_room_id, goal_room, room_graph)
                if path is None or len(path) < 2:
                    print(f"[DOORWAY_RCN] No path: room {from_room_id} -> goal room {goal_room}; skipping.")
                    continue
                next_room = path[1]
                pair = (min(from_room_id, next_room), max(from_room_id, next_room))
                doorway_cells = doorway_by_pair.get(pair, [])
                if not doorway_cells:
                    print(f"[DOORWAY_RCN] No doorway cells for pair {pair}; skipping.")
                    continue

                # Use centroid of doorway cell CoMs as the sub-goal position
                dcx = float(cell_cx[doorway_cells].mean())
                dcz = float(cell_cz[doorway_cells].mean())
                print(
                    f"[DOORWAY_RCN] Building room{from_room_id}->goal '{goal_name}': "
                    f"doorway at ({dcx:.2f},{dcz:.2f}), {len(doorway_cells)} cells"
                )

                door_rcn = self._compute_reward_weights(dcx, dcz, _w_exp_norm, self.unified_rcn)

                path_out = os.path.join(
                    multi_goal_dir,
                    f"doorway_rcn_room{from_room_id}_goal_{goal_name}.pkl"
                )
                with open(path_out, "wb") as f:
                    pickle.dump(door_rcn, f)
                print(f"[DOORWAY_RCN] Saved: doorway_rcn_room{from_room_id}_goal_{goal_name}.pkl")

    def _get_current_room(self, position=None):
        """
        Determine current room from robot position using bounding box check.
        Caches result for _room_cache_interval steps.
        Returns 0 if no topology is available.
        """
        if not getattr(self, "room_topology", None) or self.room_topology.get("n_rooms", 1) <= 1:
            return 0

        cache = getattr(self, "_current_room_cache", None)
        interval = getattr(self, "_room_cache_interval", 5)
        if cache is not None:
            cached_room, cached_step = cache
            if (self.step_count - cached_step) < interval:
                return cached_room

        if position is None:
            pos = self.robot.getField("translation").getSFVec3f()
            x, z = float(pos[0]), float(pos[1])  # hmap_loc convention: col0=x, col1=z
        else:
            x, z = float(position[0]), float(position[1])

        current_room = 0
        for room_def in self.room_definitions:
            xmin, xmax, zmin, zmax = room_def["bounds"]
            if xmin <= x <= xmax and zmin <= z <= zmax:
                current_room = room_def["id"]
                break

        self._current_room_cache = (current_room, int(self.step_count))
        return current_room

    def _load_room_topology(self):
        """Load room topology from disk. Sets self.room_topology to None on failure."""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        topology_path = os.path.join(multi_goal_dir, "room_topology.pkl")
        if not os.path.exists(topology_path):
            print("[DRIVER] No room_topology.pkl found; hierarchical nav disabled.")
            self.room_topology = None
            return False
        try:
            with open(topology_path, "rb") as f:
                self.room_topology = pickle.load(f)
            n = self.room_topology.get("n_rooms", 0)
            d = len(self.room_topology.get("doorway_by_pair", {}))
            print(f"[DRIVER] Loaded room topology: {n} rooms, {d} doorway pair(s).")
            return True
        except Exception as e:
            print(f"[DRIVER] Failed to load room topology: {e}; hierarchical nav disabled.")
            self.room_topology = None
            return False

    def _load_doorway_rcn(self, goal_name, room_id):
        """
        Load doorway sub-RCN for (room_id, goal_name).
        Falls back to global goal RCN if file not found.
        """
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        path = os.path.join(multi_goal_dir, f"doorway_rcn_room{room_id}_goal_{goal_name}.pkl")
        if not os.path.exists(path):
            print(f"[DRIVER] Doorway RCN not found: {path}; falling back to global goal RCN.")
            self._load_goal_specific_rcns(goal_name)
            return
        try:
            with open(path, "rb") as f:
                self.unified_rcn = pickle.load(f)
            # Apply same compat patches as _load_goal_specific_rcns
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
            self._configure_unified_replay_settings(self.unified_rcn)
            self.rcns = [self.unified_rcn]
            self.rcn = self.unified_rcn
            print(f"[DRIVER] Loaded doorway RCN: room={room_id} -> goal='{goal_name}'")
        except Exception as e:
            print(f"[DRIVER] Failed to load doorway RCN: {e}; falling back to global goal RCN.")
            self._load_goal_specific_rcns(goal_name)

    # ─────────────────────────────────────────────────────────────────────────────────────

    def _create_multi_goal_reward_maps(self):
        """Create reward maps for each goal-scale combination"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        os.makedirs(multi_goal_dir, exist_ok=True)

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

            # ── Experience transition matrix (built once, reused for all goals) ──
            # T.T = reverse/backward direction: reward propagates FROM goal BACK
            # through predecessor states, routing correctly through doorways.
            # λ=10, steps=50 suppresses ring (exp(-63/10)≈0) while the direct
            # interior path (~17 hops) gives exp(-17/10)≈0.18 weight.
            _rcn_dev    = self.unified_rcn.w_in.device
            _T_counts   = self.unified_rcn.experience_transition_counts.clone()  # (N,N), CPU
            _T_counts.fill_diagonal_(0.0)
            _row_sums   = _T_counts.sum(dim=1, keepdim=True).clamp(min=1e-12)
            _w_exp_norm = (_T_counts / _row_sums).to(_rcn_dev)           # P(i→j): correct backward Bellman propagation
            _n_exp      = int((_T_counts > 0).sum().item())
            print(
                f"[LEARN_LOCATIONS] Experience transition matrix: "
                f"{_n_exp} nonzero entries"
            )

            created_maps = 0
            for goal in self.goals:
                associations = self.goal_place_cell_associations[goal["name"]]
                goal_steps = self.goal_association_step[goal["name"]]
                candidate_scales = [i for i, pc_idx in enumerate(associations) if pc_idx is not None]
                if not candidate_scales:
                    print(f"[WARNING] No place-cell association for goal {goal['name']}; skipping unified reward map.")
                    continue

                # Initialise goal RCN with clean weights.
                goal_rcn = copy.deepcopy(self.unified_rcn)
                goal_rcn.w_in = torch.zeros_like(goal_rcn.w_in)
                goal_rcn.w_in_effective = goal_rcn.w_in.clone()
                goal_rcn.reward_cell_activations = torch.zeros_like(goal_rcn.reward_cell_activations)

                rcn_device = goal_rcn.w_in.device

                # ── Trajectory Gaussian seed ───────────────────────────────────────────
                # Aggregate ALL timesteps weighted by exp(-d²/2σ²) where d = distance to
                # goal. This uses every goal-proximal experience rather than one snapshot,
                # producing a robust, noise-averaged seed that is well-localized near the
                # goal without depending on STDP connectivity quality.
                seed_log = []
                n_valid = min(int(self.step_count) + 1, self.hmap_loc.shape[0])
                hmap_x = torch.from_numpy(self.hmap_loc[:n_valid, 0]).float().to(rcn_device)
                hmap_y = torch.from_numpy(self.hmap_loc[:n_valid, 1]).float().to(rcn_device)
                goal_x = float(goal["location"][0])
                goal_y = float(goal["location"][1])
                dist_sq_all = (hmap_x - goal_x) ** 2 + (hmap_y - goal_y) ** 2  # (n_valid,)

                seed_activations = torch.zeros(self.unified_pcn.num_pc_total, device=rcn_device)
                if hmap_ok:
                    for s_idx in range(len(self.scales)):
                        s_start = self.unified_pcn.scale_boundaries[s_idx]
                        s_end   = self.unified_pcn.scale_boundaries[s_idx + 1]
                        if s_idx >= len(self.hmap_pcn_activities):
                            continue
                        acts    = self.hmap_pcn_activities[s_idx][:n_valid].float().to(rcn_device)
                        sigma_s = float(self.scales[s_idx].get("sigma_r", 1.0)) * 2.0
                        w_g     = torch.exp(-dist_sq_all / (2.0 * sigma_s ** 2))
                        w_sum   = w_g.sum()
                        if w_sum < 1e-12:
                            seed_log.append(f"scale{s_idx}:no_coverage")
                            continue
                        seed_activations[s_start:s_end] = torch.mv(acts.T, w_g / w_sum)
                        n_nz = int((seed_activations[s_start:s_end] > 1e-6).sum().item())
                        seed_log.append(f"scale{s_idx}:gauss(σ={sigma_s:.1f}m,{n_nz}cells)")
                else:
                    # Fallback: single-cell per scale when hmaps unavailable.
                    activations_conf = self.goal_place_cell_activations[goal["name"]]
                    for scale_idx in candidate_scales:
                        local_pc_idx = int(associations[scale_idx])
                        s_start = self.unified_pcn.scale_boundaries[scale_idx]
                        conf = float(activations_conf[scale_idx]) if activations_conf[scale_idx] is not None else 1.0
                        seed_activations[s_start + local_pc_idx] = max(conf, 1e-3)
                        seed_log.append(f"scale{scale_idx}:fallback(pc={local_pc_idx})")

                # ── Checkpoint boost pre-computation ─────────────────────────────
                # For each valid checkpoint (bounce-filtered), pre-compute its
                # trajectory-Gaussian seed vector and a normalised presence vector.
                # The replay loop uses these to re-amplify the propagating wave
                # when it organically reaches a checkpoint via experience transitions.
                # Checkpoints the wave never reaches (dead ends) are never boosted.
                _cp_seeds    = {}   # c_idx -> (n_pc_total,) seed tensor
                _cp_presence = {}   # c_idx -> normalised version for dot-product detection
                if self.detected_doorways and self._valid_checkpoints and hmap_ok:
                    for _ci, (_cx, _cy) in enumerate(self.detected_doorways):
                        if _ci not in self._valid_checkpoints:
                            continue
                        _cp_dist_sq  = (hmap_x - _cx) ** 2 + (hmap_y - _cy) ** 2
                        _cp_seed_vec = torch.zeros(self.unified_pcn.num_pc_total, device=rcn_device)
                        for s_idx in range(len(self.scales)):
                            s_start = self.unified_pcn.scale_boundaries[s_idx]
                            s_end   = self.unified_pcn.scale_boundaries[s_idx + 1]
                            if s_idx >= len(self.hmap_pcn_activities):
                                continue
                            acts    = self.hmap_pcn_activities[s_idx][:n_valid].float().to(rcn_device)
                            sigma_s = float(self.scales[s_idx].get("sigma_r", 1.0)) * 2.0
                            w_g     = torch.exp(-_cp_dist_sq / (2.0 * sigma_s ** 2))
                            w_sum   = w_g.sum()
                            if w_sum < 1e-12:
                                continue
                            _cp_seed_vec[s_start:s_end] = torch.mv(acts.T, w_g / w_sum)
                        _norm = _cp_seed_vec.abs().max().clamp(min=1e-12)
                        if float(_norm) > 1e-9:
                            _cp_seeds[_ci]    = _cp_seed_vec
                            _cp_presence[_ci] = _cp_seed_vec / _norm
                if _cp_seeds:
                    seed_log.append(f"checkpoints({len(_cp_seeds)}valid,boost_in_seed)")

                # ── Experience transition (reverse replay) ───────────────────────
                # λ=10, steps=50: ring path (~63 hops) is suppressed exp(-63/10)≈0,
                # direct interior path (~17 hops) dominates exp(-17/10)≈0.18.
                # Doorway path (~10-30 hops) gives non-zero gradient in far room.
                C_REWARD      = float(getattr(goal_rcn, "C_REWARD", 5.0))
                weight_update = torch.zeros_like(goal_rcn.w_in)

                w_full = _w_exp_norm  # row-normalised reverse transition matrix (CA3 replay)

                unified_replay_steps = 200
                lambda_unified       = 20.0   # Bellman γ = exp(-1/λ) = 0.951 per step
                A_unified            = C_REWARD / max(lambda_unified, 1e-6)

                # ── Multi-source backward value iteration (Bellman equation) ─────────
                # V_{k+1} = seed_combined + γ · W @ V_k
                # seed_combined = goal_seed + _boost_gamma * sum(checkpoint_seeds)
                # Checkpoints act as relay stations through sparse-transition bottlenecks
                # (doorways). Without them, V(left_room) ≈ 0 because the empirical W has
                # ~1% doorway-crossing probability vs ~99% within-room, so Bellman value
                # attenuates to near-zero before crossing the wall.
                _gamma       = math.exp(-1.0 / lambda_unified)   # 0.9512
                _boost_gamma = float(getattr(self, "checkpoint_boost_gamma", 0.5))
                seed_combined = seed_activations.clone().to(rcn_device)
                for _cp_seed in _cp_seeds.values():
                    seed_combined = seed_combined + _boost_gamma * _cp_seed.to(rcn_device)
                v = seed_combined.clone()
                for _ in range(unified_replay_steps):
                    v = seed_combined + _gamma * torch.matmul(w_full, v)

                weight_update[0, :] = A_unified * v

                print(
                    f"  [{goal['name']}] Unified replay: {unified_replay_steps} steps, "
                    f"λ={lambda_unified} (Bellman γ={_gamma:.4f}), "
                    f"multi-source ({1 + len(_cp_seeds)} seeds)"
                )

                # Stability guard, then apply.
                max_val = torch.max(torch.abs(weight_update))
                if torch.isfinite(max_val) and max_val > 1e3:
                    weight_update = weight_update / max_val
                goal_rcn.w_in = goal_rcn.w_in + weight_update
                # Clamp to ≥ 0: reward weights should be non-negative by design.
                # Defensive guard against any negative randn initialisation residue.
                goal_rcn.w_in = torch.clamp(goal_rcn.w_in, min=0.0)
                goal_rcn.w_in_effective = torch.clamp(goal_rcn.w_in.clone(), min=0.0)

                # ── Spatial Gaussian fill (dead-band removal) ─────────────────
                # The replay weights are path-dependent: cells not on the direct
                # transition path from goal get near-zero weight, creating visible
                # dead bands in the reward map.  Fix: spread each scale's weights
                # spatially using a Gaussian kernel whose width equals that scale's
                # receptive-field size.  We take element-wise max(original, smoothed)
                # so peaks are never reduced — only zero-weight cells are filled in.
                # Only w_in_effective is modified; w_in retains the raw replay result.
                if hmap_ok:
                    _smooth_factor = 1.5   # kernel σ = sigma_r × this value
                    for _s_idx in range(len(self.scales)):
                        _s_start = int(self.unified_pcn.scale_boundaries[_s_idx])
                        _s_end   = int(self.unified_pcn.scale_boundaries[_s_idx + 1])
                        if _s_idx >= len(self.hmap_pcn_activities):
                            continue
                        _acts = self.hmap_pcn_activities[_s_idx][:n_valid].float().to(rcn_device)
                        _acts_sum = _acts.sum(0)                          # (n_pc_s,)
                        _visited  = _acts_sum > 0.05                      # bool mask
                        if not _visited.any():
                            continue
                        _denom = _acts_sum.clamp(min=1e-12)
                        _x_cent = (_acts * hmap_x.unsqueeze(1)).sum(0) / _denom  # (n_pc_s,)
                        _y_cent = (_acts * hmap_y.unsqueeze(1)).sum(0) / _denom
                        _sigma_s = float(self.scales[_s_idx].get("sigma_r", 1.0)) * _smooth_factor
                        _dx  = _x_cent.unsqueeze(0) - _x_cent.unsqueeze(1)  # (n_pc_s, n_pc_s)
                        _dy  = _y_cent.unsqueeze(0) - _y_cent.unsqueeze(1)
                        _K   = torch.exp(-(_dx**2 + _dy**2) / (2.0 * _sigma_s**2))
                        _vf  = _visited.float()
                        _K   = _K * _vf.unsqueeze(0) * _vf.unsqueeze(1)  # mask unvisited
                        _K   = _K / _K.sum(1, keepdim=True).clamp(min=1e-12)
                        _w_orig   = goal_rcn.w_in_effective[0, _s_start:_s_end].clone()
                        _w_smooth = torch.mv(_K, _w_orig)
                        goal_rcn.w_in_effective[0, _s_start:_s_end] = torch.max(_w_orig, _w_smooth)

                unified_goal_path = os.path.join(multi_goal_dir, f"unified_rcn_goal_{goal['name']}.pkl")
                with open(unified_goal_path, "wb") as f:
                    pickle.dump(goal_rcn, f)
                created_maps += 1
                print(
                    f"[LEARN_LOCATIONS] Created unified reward map for goal '{goal['name']}': "
                    f"{', '.join(seed_log)}"
                )

            print(f"[LEARN_LOCATIONS] Successfully created {created_maps} unified goal reward maps")

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
                with open(goal_rcn_path, "wb") as f:
                    pickle.dump(goal_rcn, f)
                created_maps += 1
                print(f"[LEARN_LOCATIONS] Created: {scale_name}_goal_{goal['name']}")

        print(f"[LEARN_LOCATIONS] Successfully created {created_maps} reward maps")
    def _save_multi_goal_data(self):
        """Save multi-goal specific data"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        # Save goal associations
        associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        association_data = {
            "goal_place_cell_associations": self.goal_place_cell_associations,
            "goal_association_step": self.goal_association_step,
            "goal_place_cell_activations": self.goal_place_cell_activations,
            "goal_visit_counts": self.goal_visit_counts,
            "goals": self.goals,
            "scales": [{"scale_index": s["scale_index"], "name": s["name"]} for s in self.scales],
            "total_steps": self.step_count,
            "final_time": self.getTime()
        }
        with open(associations_path, "wb") as f:
            pickle.dump(association_data, f)

        print(f"[LEARN_LOCATIONS] Saved goal associations to {associations_path}")

        # Print summary
        print(f"[LEARN_LOCATIONS] Final associations:")
        for goal_name, associations in self.goal_place_cell_associations.items():
            goal_info = next(g for g in self.goals if g["name"] == goal_name)
            visit_count = self.goal_visit_counts[goal_name]
            print(f"  {goal_name} at {goal_info['location']}: {associations} (visits: {visit_count})")

        # Save room topology if built
        if getattr(self, "room_topology", None) is not None:
            topology_path = os.path.join(multi_goal_dir, "room_topology.pkl")
            with open(topology_path, "wb") as f:
                pickle.dump(self.room_topology, f)
            print(f"[LEARN_LOCATIONS] Saved room topology ({self.room_topology['n_rooms']} rooms) to {topology_path}")

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
                [curr_pos[0], curr_pos[1]], dtype=self.dtype, device=self.device
            ),
            atol=self.goal_r["explore"],
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[1] - self.goal_location[1]

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
            self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
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
        ACCEPTABLE_ERROR = 5.0  # Acceptable final error in degrees
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
        angle_to_turn_deg = heading_deg - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180

        if show_debug:
            print(f"Turning from {self.current_heading_deg:.0f}° to {heading_deg:.0f}°")

        success = self.compass_based_turn_to_heading(heading_deg, show_debug)

        if not success:
            if show_debug:
                print("TURNING FAILED - falling back to exploration")
            self.turn(np.random.uniform(-np.pi/4, np.pi/4))
            for _ in range(3):
                self.sense()
                self.compute_pcn_activations()
                self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
                self.forward()
                self.check_goal_reached()
            return False

        # Record pre-movement position
        pre_move_pos = self.robot.getField("translation").getSFVec3f()

        # Move forward
        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
            self.forward()
            self.check_goal_reached()

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
        if show_debug:
            post_move_pos = self.robot.getField("translation").getSFVec3f()
            actual_dx = post_move_pos[0] - pre_move_pos[0]
            actual_dy = post_move_pos[2] - pre_move_pos[2]
            actual_distance = np.sqrt(actual_dx**2 + actual_dy**2)

            if actual_distance > 0.001:
                actual_angle = np.degrees(np.arctan2(actual_dy, actual_dx))
                actual_angle = (actual_angle + 360) % 360
                angle_error = abs(actual_angle - heading_deg)
                if angle_error > 180:
                    angle_error = 360 - angle_error

                status = "✓" if angle_error < 15 else "⚠" if angle_error < 30 else "✗"
                print(f"Moved {actual_distance:.3f}m at {actual_angle:.0f}° (target: {heading_deg}°) {status}")

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
                self.hmap_loc[self.step_count] = curr_pos

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
        scale_idx_map = {scale_def["scale_index"]: i for i, scale_def in enumerate(self.scales)}

        if update_pcn:
            for scale_def, act in zip(self.scales, self.pcn_activations_list):
                scale_idx = scale_def["scale_index"]

                # Ensure the scale index exists in the mapping
                if scale_idx not in scale_idx_map:
                    continue

                if self.lightweight_hmaps:
                    # Sample compact PC stats only (no dense per-cell history).
                    if self.step_count % self.hmap_sample_stride == 0:
                        act_cpu = act.detach().float().cpu()
                        k = min(self.hmap_topk, act_cpu.numel())
                        if k > 0:
                            top_vals, top_idx = torch.topk(act_cpu, k=k)
                            sample = {
                                "step": int(self.step_count),
                                "mean": float(act_cpu.mean().item()),
                                "max": float(act_cpu.max().item()),
                                "topk_idx": top_idx.numpy(),
                                "topk_val": top_vals.numpy(),
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
                        act_cpu = act.detach().float().cpu()
                        k = min(self.hmap_topk, act_cpu.numel())
                        if k > 0:
                            top_vals, top_idx = torch.topk(act_cpu, k=k)
                            sample = {
                                "step": int(self.step_count),
                                "mean": float(act_cpu.mean().item()),
                                "max": float(act_cpu.max().item()),
                                "topk_idx": top_idx.numpy(),
                                "topk_val": top_vals.numpy(),
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
        curr_pos = self.robot.getField("translation").getSFVec3f()

        # Distance from current position to goal location
        distance_to_goal = torch.norm(
            torch.tensor(
                [
                    curr_pos[0] - self.goal_location[0],
                    curr_pos[1] - self.goal_location[1],
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

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1) Save each scale's PCN (if requested)
        # ----------------------------------------------------------------------
        if include_pcn:
            if self.use_unified_multiscale:
                pcn_path = os.path.join(self.network_dir, "unified_pcn.pkl")
                with open(pcn_path, "wb") as f:
                    pickle.dump(self.unified_pcn, f)
                files_saved.append(pcn_path)
            else:
                for scale_def, pcn in zip(self.scales, self.pcns):
                    scale_idx = scale_def["scale_index"]  # Get correct scale index
                    pcn_path = os.path.join(self.network_dir, f"pcn_scale_{scale_idx}.pkl")
                    with open(pcn_path, "wb") as f:
                        pickle.dump(pcn, f)
                    files_saved.append(pcn_path)

        # ----------------------------------------------------------------------
        # 2) Save each scale's RCN (if requested)
        # ----------------------------------------------------------------------
        if include_rcn:
            if self.use_unified_multiscale:
                rcn_path = os.path.join(self.network_dir, "unified_rcn.pkl")
                with open(rcn_path, "wb") as f:
                    pickle.dump(self.unified_rcn, f)
                files_saved.append(rcn_path)
            else:
                for scale_def, rcn in zip(self.scales, self.rcns):
                    scale_idx = scale_def["scale_index"]
                    rcn_path = os.path.join(self.network_dir, f"rcn_scale_{scale_idx}.pkl")
                    with open(rcn_path, "wb") as f:
                        pickle.dump(rcn, f)
                    files_saved.append(rcn_path)

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
            with open(hmap_loc_path, "wb") as f:
                pickle.dump(self.hmap_loc[: self.step_count], f)
            files_saved.append(hmap_loc_path)

            # (b) Head direction history
            hmap_hdn_path = os.path.join(self.hmap_dir, f"{prefix}hmap_hdn.pkl")
            with open(hmap_hdn_path, "wb") as f:
                pickle.dump(self.hmap_hdn[: self.step_count].cpu(), f)
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
                with open(compact_path, "wb") as f:
                    pickle.dump(compact_payload, f)
                files_saved.append(compact_path)
            else:
                # Full dense logs (legacy/high-detail mode)
                for scale_def, pc_history in zip(self.scales, self.hmap_pcn_activities):
                    scale_idx = scale_def["scale_index"]  # Get correct scale index
                    hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_pcn_scale_{scale_idx}.pkl")

                    with open(hmap_scale_path, "wb") as f:
                        pc_data = pc_history[: self.step_count].cpu().numpy()
                        pickle.dump(pc_data, f)
                    files_saved.append(hmap_scale_path)

                for scale_def, gc_history in zip(self.scales, self.hmap_gcn_activities):
                    if gc_history.numel() > 0:  # Only save if there are grid cells
                        scale_idx = scale_def["scale_index"]
                        hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_gcn_scale_{scale_idx}.pkl")

                        with open(hmap_scale_path, "wb") as f:
                            gc_data = gc_history[: self.step_count].cpu().numpy()
                            pickle.dump(gc_data, f)
                        files_saved.append(hmap_scale_path)

            # (e) Prox values
            if hasattr(self, "hmap_prox"):
                hmap_prox_path = os.path.join(self.hmap_dir, f"{prefix}hmap_prox.pkl")
                with open(hmap_prox_path, "wb") as f:
                    prox_data = self.hmap_prox[: self.step_count].cpu().numpy()
                    pickle.dump(prox_data, f)
                files_saved.append(hmap_prox_path)

            # (f) Scale priority (only for multi-trial modes)
            if hasattr(self, "hmap_scale_priority") and use_trial_prefix:
                hmap_scale_priority_path = os.path.join(self.hmap_dir, f"{prefix}hmap_scale_priority.pkl")
                with open(hmap_scale_priority_path, "wb") as f:
                    scale_priority_data = self.hmap_scale_priority[: self.step_count].cpu().numpy()
                    pickle.dump(scale_priority_data, f)
                files_saved.append(hmap_scale_priority_path)

            # (g) Run diagnostics for scale gating/recruitment behavior.
            hmap_diag_path = os.path.join(self.hmap_dir, f"{prefix}hmap_scale_diagnostics.pkl")
            with open(hmap_diag_path, "wb") as f:
                diag_payload = self._build_scale_diagnostics_payload()
                pickle.dump(diag_payload, f)
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

            with open(hmap_loc_file, "wb") as f:
                pickle.dump(self.hmap_loc[:self.step_count], f)
                files_saved.append(hmap_loc_file)

            with open(hmap_scale_priority_file, "wb") as f:
                pickle.dump(self.hmap_scale_priority[: self.step_count].cpu().numpy(), f)
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
        print("Saving Done!")

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
