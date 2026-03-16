import numpy as np
import os
from typing import Optional, List, Dict, Any
import torch
from controller import Supervisor
import random
import math
import time
import json

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.head_direction_layer import HeadDirectionLayer
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector
from driver_experiment_setup import (
    setup_coverage_tracking as setup_coverage_tracking_helper,
    setup_goals as setup_goals_helper,
    setup_trials as setup_trials_helper,
)
from driver_networks import (
    init_grid_cell_networks as init_grid_cell_networks_helper,
    load_goal_specific_rcns as load_goal_specific_rcns_helper,
    load_or_init_pcn_for_scale as load_or_init_pcn_for_scale_helper,
    load_or_init_rcn_for_scale as load_or_init_rcn_for_scale_helper,
    load_pcns as load_pcns_helper,
    load_rcns as load_rcns_helper,
)
from driver_persistence import (
    clear_driver_outputs,
    create_multi_goal_reward_maps,
    handle_random_exploitation_timeout,
    handle_random_goal_exploitation,
    save_driver_outputs,
    save_multi_goal_data,
    save_trial_completion_time,
)

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
        runtime_step_mode: str = "legacy",
        ignore_goals: bool = False,
        track_runtime_metrics: bool = False,
        environment_size: Optional[List[float]] = None,
        grid_size: Optional[float] = None,
        coverage_percentage: Optional[float] = None,
        min_goal_visits: int = 3,
        non_exploit_stop_condition: str = "time_or_coverage",
        goal_constraint_mode: str = "none",
        coverage_free_space_only: bool = True,
        optimal_path_distance: Optional[float] = None,
        path_failure_ratio: Optional[float] = None,
        paths_folder: Optional[str] = None,
        hmaps_folder: Optional[str] = None,
        auto_trial_name: Optional[str] = None,
        num_auto_trials: int = 5,
        current_auto_trial: int = 1,
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
        self.batch_execution = bool(auto_trial_name)

        # Directories for saving/loading data
        if auto_trial_name:
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

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 8
        self.max_dist = max_dist
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
                "sigma_theta": 1.0
            }]
        self.scales = scales
        self.scale_idx_map = {scale_def["scale_index"]: i for i, scale_def in enumerate(self.scales)}
        
        # Store learning rates for later use in RCN initialization
        if rcn_learning_rates is None:
            self.rcn_learning_rates = [scale_def.get("rcn_learning_rate", 0.1) for scale_def in self.scales]
        else:
            if len(rcn_learning_rates) != len(self.scales):
                raise ValueError(
                    f"Expected {len(self.scales)} rcn_learning_rates, got {len(rcn_learning_rates)}"
                )
            self.rcn_learning_rates = list(rcn_learning_rates)
        self.td_learning = td_learning
        self.use_prox_mod = use_prox_mod
        self.ignore_goals = bool(ignore_goals)
        self.track_runtime_metrics = bool(track_runtime_metrics)
        self.runtime_step_mode = str(runtime_step_mode).lower()
        valid_step_modes = {"legacy", "hybrid"}
        if self.runtime_step_mode not in valid_step_modes:
            print(f"[DRIVER] Invalid runtime_step_mode='{self.runtime_step_mode}', defaulting to 'legacy'")
            self.runtime_step_mode = "legacy"
        self.hybrid_runtime = self.runtime_step_mode == "hybrid"
        print(f"[DRIVER] Runtime step mode: {self.runtime_step_mode}")

        # Store coverage parameters
        self.environment_size = environment_size
        self.grid_size = grid_size
        self.coverage_percentage = coverage_percentage
        self.min_goal_visits = min_goal_visits
        self.non_exploit_stop_condition = str(non_exploit_stop_condition).lower()
        self.goal_constraint_mode = str(goal_constraint_mode).lower()
        self.coverage_free_space_only = bool(coverage_free_space_only)
        valid_stop_conditions = {"time", "coverage", "time_or_coverage"}
        if self.non_exploit_stop_condition not in valid_stop_conditions:
            print(f"[DRIVER] Invalid non_exploit_stop_condition='{self.non_exploit_stop_condition}', defaulting to 'time_or_coverage'")
            self.non_exploit_stop_condition = "time_or_coverage"
        valid_goal_constraint_modes = {"none", "visits", "learning"}
        if self.goal_constraint_mode not in valid_goal_constraint_modes:
            print(f"[DRIVER] Invalid goal_constraint_mode='{self.goal_constraint_mode}', defaulting to 'none'")
            self.goal_constraint_mode = "none"

        # Store random spawn parameters
        self.optimal_path_distance = optimal_path_distance
        self.path_failure_ratio = path_failure_ratio
        self.paths_folder = paths_folder

        # Initialize distance tracking for random spawn mode
        self.total_distance_traveled = 0.0
        self.last_position = None

        # Setup unified goal system
        self._setup_goals(goal_config, goal_location)

        # Setup trial configuration
        self._setup_trials(trial_config)

        # Global non-exploit coverage setup (optional, based on config toggle)
        non_exploit_modes = {
            RobotMode.LEARN_OJAS,
            RobotMode.LEARN_HEBB,
            RobotMode.DMTP,
            RobotMode.PLOTTING,
            RobotMode.LEARN_LOCATIONS_COVERAGE,
        }
        if (
            self.robot_mode in non_exploit_modes
            and self.non_exploit_stop_condition in {"coverage", "time_or_coverage"}
            and self.environment_size
            and self.grid_size
            and self.coverage_percentage
            and not hasattr(self, "target_coverage_percentage")
        ):
            self._setup_coverage_tracking(self.environment_size, self.grid_size, self.coverage_percentage)

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
        print(f"[DRIVER] Using PCNs: {[f'pcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")
        print(f"[DRIVER] Using RCNs: {[f'rcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        self.hmap_prox = torch.zeros((self.num_steps,), device=self.device, dtype=torch.float32)
        self.hmap_scale_priority = torch.zeros(self.num_steps, device=self.device, dtype=torch.float32)

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

        # Rotation tracking for excessive loop detection
        self.rotation_accumulator = 0.0
        self.rotation_loop_count = 0
        self.steps_since_last_loop = 0
        self.last_heading_deg = None
        self.done = False
        
        # Number of steps to force exploration (init to 0)
        self.force_explore_count = 0

        # Per-scale reliability tracking
        # Reset to None at start of each trial - will be re-initialized to 1.0 in exploit_v12
        self.scale_reliability = None  # Shape: [num_scales], values in [0, 1]
        self.last_scale_weights = None  # Store scale weights for credit assignment
        self.loop_scale_contributions = None  # Track scale contributions to rotation loops
        print(f"[DRIVER] Scale reliability reset for new trial (will initialize to 1.0 on first exploit_v12 call)")

        # Optionally keep a single-scale reference
        self.pcn = self.pcns[0] if self.pcns else None
        self.rcn = self.rcns[0] if self.rcns else None

        # For EXPLOIT_LOCATIONS_RANDOM, load goal-specific RCNs
        if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM and hasattr(self, 'active_goal_name'):
            self._load_goal_specific_rcns(self.active_goal_name)

        self.plot_bvc = plot_bvc

        # Runtime benchmarking timestamps (first/last simulation step wall clock)
        self.real_time_first_step = None
        self.real_time_last_step = None

        # Initial step for startup consistency.
        self._step_simulation()

        # Track trial start time for per-trial timeouts
        self.trial_start_time = self.getTime()
        print(f"[DRIVER] Trial started at simulation time: {self.trial_start_time:.1f}s")

    ##########################################################################
    #                           PCN / RCN LOADING                            #
    ##########################################################################
    def load_pcns(self, enable_ojas: Optional[bool], enable_stdp: Optional[bool]):
        return load_pcns_helper(self, enable_ojas, enable_stdp)

    def _load_or_init_pcn_for_scale(self, path, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        return load_or_init_pcn_for_scale_helper(self, path, scale_def, num_grid_cells, enable_ojas, enable_stdp)

    def load_rcns(self):
        return load_rcns_helper(self)

    def _load_or_init_rcn_for_scale(self, path, scale_def, learning_rate):
        return load_or_init_rcn_for_scale_helper(self, path, scale_def, learning_rate)

    def _load_goal_specific_rcns(self, goal_name):
        """Load goal-specific RCNs for EXPLOIT_LOCATIONS_RANDOM mode."""
        return load_goal_specific_rcns_helper(self, goal_name)

    ##########################################################################
    #                        GRID CELL NETWORK INITIALIZATION                #
    ##########################################################################

    def init_grid_cell_networks(self):
        """Initialize grid cell networks for each scale based on scale parameters."""
        return init_grid_cell_networks_helper(self)

    ##########################################################################
    #                        GOAL / TRIAL / COVERAGE SETUP                   #
    ##########################################################################

    def _setup_goals(self, goal_config, goal_location):
        return setup_goals_helper(self, goal_config, goal_location)

    def _setup_trials(self, trial_config):
        return setup_trials_helper(self, trial_config)

    def _setup_coverage_tracking(self, environment_size, grid_size, coverage_percentage):
        return setup_coverage_tracking_helper(self, environment_size, grid_size, coverage_percentage)

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

        # Ignore cells occluded by obstacles when obstacle-aware coverage is enabled
        if self.coverage_free_space_only and not self.traversable_mask[grid_z, grid_x]:
            return

        # Mark cell as visited if not already visited
        if not self.coverage_grid[grid_z, grid_x]:
            self.coverage_grid[grid_z, grid_x] = True
            self.visited_cells += 1
            self.current_coverage_percentage = self.visited_cells / self.total_coverable_cells

            # Optional: Print coverage updates at intervals
            if self.visited_cells % 100 == 0:
                print(f"[COVERAGE] Visited {self.visited_cells}/{self.total_coverable_cells} cells ({self.current_coverage_percentage*100:.1f}%)")

    def _check_coverage_complete(self):
        """Check if target coverage percentage has been reached"""
        return self.current_coverage_percentage >= self.target_coverage_percentage

    def _non_exploit_stop_condition_met(self, trial_elapsed_time: float) -> bool:
        """Evaluate configured stop condition for non-exploit modes."""
        time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
        coverage_ready = hasattr(self, "target_coverage_percentage")
        coverage_reached = self._check_coverage_complete() if coverage_ready else False

        if self.non_exploit_stop_condition == "time":
            return time_reached
        if self.non_exploit_stop_condition == "coverage":
            if not coverage_ready:
                # Fail-safe fallback if coverage is requested but not configured
                if not hasattr(self, "_coverage_stop_fallback_warned"):
                    print("[DRIVER] coverage stop requested but coverage tracking not configured; falling back to time stop.")
                    self._coverage_stop_fallback_warned = True
                return time_reached
            return coverage_reached
        # legacy-compatible default
        return time_reached or coverage_reached

    def _update_goal_visit_tracking(self, current_position, do_learning: bool = False):
        """Track per-goal visit counts, with optional learning updates."""
        if not self.multi_goal_mode:
            return
        if not hasattr(self, "goal_visit_counts") or not hasattr(self, "goal_currently_in"):
            return

        current_x, current_z = float(current_position[0]), float(current_position[1])
        for goal in self.goals:
            goal_x, goal_z = float(goal["location"][0]), float(goal["location"][1])
            dx = current_x - goal_x
            dz = current_z - goal_z
            distance = math.sqrt(dx * dx + dz * dz)

            if distance <= goal["radius"]:
                if not goal["visited"]:
                    goal["visited"] = True
                    print(f"[GOALS] First visit to {goal['name']} at {goal['location']}")

                if not self.goal_currently_in[goal["name"]]:
                    self.goal_visit_counts[goal["name"]] += 1
                    self.goal_currently_in[goal["name"]] = True
                    print(f"[GOALS] {goal['name']} visit #{self.goal_visit_counts[goal['name']]}")

                if do_learning and self.goal_constraint_mode == "learning":
                    self._handle_goal_learning(goal)
            else:
                if self.goal_currently_in[goal["name"]]:
                    self.goal_currently_in[goal["name"]] = False

    def _goal_constraints_met(self) -> bool:
        """Return whether configured goal constraints are satisfied."""
        if self.ignore_goals:
            return True
        if self.goal_constraint_mode == "none":
            return True

        if self.goal_constraint_mode == "visits":
            if not self.multi_goal_mode or not hasattr(self, "goal_visit_counts"):
                return True
            for goal_name, visit_count in self.goal_visit_counts.items():
                if visit_count < self.min_goal_visits:
                    return False
            return True

        # learning constraints: keep existing strict behavior where available
        if self.goal_constraint_mode == "learning":
            if hasattr(self, "_check_multi_goal_learning_complete"):
                return self._check_multi_goal_learning_complete()
            return False

        return True

    def _update_distance_tracking(self):
        """Update total distance traveled for random spawn mode"""
        if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
            current_pos = self.robot.getField("translation").getSFVec3f()
            current_position_2d = [current_pos[0], current_pos[2]]  # [x, z]

            if self.last_position is not None:
                # Calculate distance moved since last update
                dx = current_position_2d[0] - self.last_position[0]
                dz = current_position_2d[1] - self.last_position[1]
                distance_moved = math.sqrt(dx*dx + dz*dz)
                self.total_distance_traveled += distance_moved

            self.last_position = current_position_2d

    @staticmethod
    def _distance_to_location_2d(current_x: float, current_z: float, location) -> float:
        dx = current_x - float(location[0])
        dz = current_z - float(location[1])
        return math.sqrt(dx * dx + dz * dz)

    def _describe_non_exploit_stop_reason(self, coverage_reached: bool) -> str:
        if self.non_exploit_stop_condition == "coverage":
            return "Coverage target reached"
        if self.non_exploit_stop_condition == "time":
            return "Time limit reached"
        return "Coverage target reached" if coverage_reached else "Time limit reached"

    def _record_runtime_metrics(self, stop_reason: str, sim_elapsed_time: Optional[float] = None) -> None:
        """Record wall-clock runtime between first and last simulation step."""
        if not self.track_runtime_metrics:
            return
        if self.real_time_first_step is None or self.real_time_last_step is None:
            return

        wall_elapsed_seconds = self.real_time_last_step - self.real_time_first_step
        trial_elapsed_seconds = (
            float(sim_elapsed_time)
            if sim_elapsed_time is not None
            else float(self.getTime() - self.trial_start_time)
        )

        runtime_metrics = {
            "trial_id": self.trial_id,
            "world_name": self.world_name,
            "runtime_step_mode": self.runtime_step_mode,
            "stop_reason": stop_reason,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "trial_elapsed_seconds": trial_elapsed_seconds,
            "sim_time_ratio": (trial_elapsed_seconds / wall_elapsed_seconds) if wall_elapsed_seconds > 0 else None,
            "num_steps_recorded": int(self.step_count),
            "timestep_ms": int(self.timestep),
        }
        self.runtime_metrics = runtime_metrics

        out_name = f"{self.trial_id}_runtime_metrics.json" if self.trial_id else "runtime_metrics.json"
        out_path = os.path.join(self.hmap_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(runtime_metrics, f, indent=2)

        print(
            f"[RUNTIME] wall={wall_elapsed_seconds:.2f}s, sim={trial_elapsed_seconds:.2f}s, "
            f"ratio={runtime_metrics['sim_time_ratio']:.2f}x, mode={self.runtime_step_mode}, saved={out_path}"
        )

    def _finalize_non_exploit_completion(
        self,
        include_networks: bool,
        stop_reason: Optional[str] = None,
        sim_elapsed_time: Optional[float] = None,
    ) -> None:
        """Finalize non-exploit completion: stop, save, mark done, and pause when required."""
        if stop_reason is not None:
            self._record_runtime_metrics(stop_reason=stop_reason, sim_elapsed_time=sim_elapsed_time)
        self.stop()
        self.save(
            include_pcn=include_networks,
            include_rcn=include_networks,
            include_gcn=include_networks,
            include_hmaps=True,
        )
        self.done = True

        if not self.batch_execution:
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)


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
                print("[DRIVER] MANUAL_CONTROL is disabled in multiscale_grid_driver_v2.")
                self.done = True
                break
            elif self.robot_mode in (RobotMode.LEARN_OJAS,
                                      RobotMode.LEARN_HEBB,
                                      RobotMode.DMTP,
                                      RobotMode.PLOTTING,
                                      RobotMode.LEARN_LOCATIONS_COVERAGE):
                self.explore()
            elif self.robot_mode in (RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS_RANDOM):
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
            # 1) Advance simulation / sense state
            self.sense()

            # 4) compute pcn_activations => fill self.pcn_activations_list
            self.compute_pcn_activations()

            # 5) If DMTP or EXPOIT => reward updates
            if self.robot_mode == RobotMode.DMTP or self.robot_mode == RobotMode.EXPLOIT:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)
                    # rcn.td_update(pcn.place_cell_activations, next_reward=actual_reward)
                # # Turn towards heading 225Â°
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

            # Update coverage tracking for configured non-exploit modes
            if (
                hasattr(self, "target_coverage_percentage")
                and self.robot_mode not in {RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS_RANDOM}
            ):
                curr_pos = self.robot.getField("translation").getSFVec3f()
                robot_pos = [curr_pos[0], curr_pos[2]]  # [x, z] coordinates
                self._update_coverage(robot_pos)

            # Update distance tracking if in EXPLOIT_LOCATIONS_RANDOM mode
            if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
                self._update_distance_tracking()

            self.update_hmaps(update_loc=True,
                              update_pcn=True,
                              update_gcn=True,
                              update_scale_priority=True if self.robot_mode == RobotMode.EXPLOIT else False,
                              update_prox=True if (self.use_prox_mod and self.robot_mode == RobotMode.LEARN_OJAS) else False)
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
        - Form joint distribution: P(s,d) = P(s) Ã— P(d|s)
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
        debug_print_interval = 100         # Print debug info every N steps

        #===================================================================

        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()

        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_scale_priority=True)
        self.check_goal_reached()

        if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
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

        usable_points = num_points_per_hd * self.n_hd
        if usable_points != len(boundaries_rolled):
            boundaries_rolled = boundaries_rolled[:usable_points]
        distances_per_hd = boundaries_rolled.view(self.n_hd, num_points_per_hd).amin(dim=1)

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
                        print(f"  Top1: {meta1['scale_name'][0]}-{meta1['direction']*45:3d}Â° R={r1:.3f}")
                        print(f"  Top2: {meta2['scale_name'][0]}-{meta2['direction']*45:3d}Â° R={r2:.3f}")
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

            print(f"[EXPLOIT_V12 #{self.step_count}] Î¸={final_direction_deg:.1f}Â° "
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
                self.scale_reliability[scale_idx] = torch.clamp(
                    self.scale_reliability[scale_idx] * effective_factor,
                    max=1.0
                )
        elif attribution == 'dominant':
            # Reward only the dominant scale with full multiplicative growth
            dominant_idx = torch.argmax(scale_weights).item()
            if cooldown is not None and cooldown[dominant_idx] > 0:
                return
            # Apply multiplicative growth with cap at 1.0
            self.scale_reliability[dominant_idx] = torch.clamp(
                self.scale_reliability[dominant_idx] * good_factor,
                max=1.0
            )


    ########################################### SENSE ###########################################
    def _sense_from_devices(self):
        """Read sensors and refresh state without advancing simulation."""
        boundaries = self.range_finder.getRangeImage()
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))

        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * self.current_heading_deg,
        )

        current_heading_rad = np.deg2rad(self.current_heading_deg)
        v_in = torch.tensor(
            [np.cos(current_heading_rad), np.sin(current_heading_rad)],
            dtype=self.dtype,
            device=self.device,
        )
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)

        self.collided[0] = int(self.left_bumper.getValue())
        self.collided[1] = int(self.right_bumper.getValue())

    def _step_and_sense(self):
        """Advance one simulation step, then refresh sensor state."""
        self._step_simulation()
        self._sense_from_devices()

    def _step_only(self):
        """Advance one simulation step without reading sensors."""
        self._step_simulation()

    def _step_simulation(self):
        """Advance one simulation step and track wall-clock runtime window."""
        if self.real_time_first_step is None:
            self.real_time_first_step = time.perf_counter()
        status = self.step(self.timestep)
        self.real_time_last_step = time.perf_counter()
        return status

    def sense(self):
        """
        Update sensor-derived state and advance simulation by one step.
        """
        self._step_and_sense()

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

    ########################################### COMPUTE ###########################################
    def compute_pcn_activations(self):
        """
        Uses current boundary- and HD-activations to update place-cell activations
        and store relevant data for analysis/debugging.
        """
        # Get robot position for grid cell computation
        curr_pos = self.robot.getField("translation").getSFVec3f()
        position = [curr_pos[0], curr_pos[2]]  # [x, z]

        # Store grid cell activations
        self.grid_activations_list = []

        # Compute grid cell activations for each scale
        for gcn in self.gcns:
            if gcn is not None:
                # Get grid cell activations for current position
                grid_activations = gcn.get_grid_cell_activations(position, use_mask=False)
                self.grid_activations_list.append(grid_activations)
            else:
                # If no grid cells for this scale, add None as placeholder
                self.grid_activations_list.append(None)

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
                pcn.bvc_layer.plot_activation(self.boundaries.cpu())
            # Append activations to pcn_activations_list
            self.pcn_activations_list.append(pcn.place_cell_activations)

        # Keep historical extra step behavior for legacy/hybrid modes.
        self._step_simulation()

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, save/update trial artifacts.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        current_x, current_z = float(curr_pos[0]), float(curr_pos[2])
        time_limit_minutes = 120

        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING):
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = (current_x, current_z)
            if not self.ignore_goals:
                self._update_goal_visit_tracking(current_position, do_learning=False)

            stop_metric_met = self._non_exploit_stop_condition_met(trial_elapsed_time)
            goal_constraints_met = self._goal_constraints_met()
            if stop_metric_met and goal_constraints_met:
                is_plotting_mode = self.robot_mode == RobotMode.PLOTTING
                coverage_reached = self._check_coverage_complete() if hasattr(self, "target_coverage_percentage") else False
                stop_reason = self._describe_non_exploit_stop_reason(coverage_reached)
                self._finalize_non_exploit_completion(
                    include_networks=not is_plotting_mode,
                    stop_reason=stop_reason,
                    sim_elapsed_time=trial_elapsed_time,
                )
                return

        elif self.robot_mode == RobotMode.DMTP and self._distance_to_location_2d(current_x, current_z, self.goal_location) <= self.goal_r["explore"]:
            self.stop()
            for pcn, rcn in zip(self.pcns, self.rcns):
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                rcn.replay(pcn=pcn)
            self.save(include_pcn=True, include_rcn=True, include_gcn=True)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return
        
        elif self.robot_mode == RobotMode.EXPLOIT:
            # Check if either goal reached or time expired
            trial_elapsed_time = self.getTime() - self.trial_start_time
            goal_reached = self._distance_to_location_2d(current_x, current_z, self.goal_location) <= self.goal_r["exploit"]
            time_expired = trial_elapsed_time >= 60 * time_limit_minutes
             
            if goal_reached or time_expired:
                total_path_length = round(self.compute_path_length(), 2)
                if self.stats_collector:
                    # Update and save stats once
                    self.stats_collector.update_stat("trial_id", self.trial_id)
                    self.stats_collector.update_stat("start_location", self.start_loc)
                    self.stats_collector.update_stat("goal_location", self.goal_location)
                    self.stats_collector.update_stat("total_distance_traveled", total_path_length)
                    self.stats_collector.update_stat("total_time_secs", round(trial_elapsed_time, 2))
                    self.stats_collector.update_stat("success", goal_reached and not time_expired)
                    self.stats_collector.save_stats(self.trial_id)
                     
                    # Print stats
                    print(f"Trial {self.trial_id} completed.")
                    print(f"Start location: {self.start_loc}")
                    print(f"Goal location: {self.goal_location}")
                    print(f"Total distance traveled: {total_path_length} meters.")
                    print(f"Total time taken: {round(trial_elapsed_time, 2)} seconds.")
                    print(f"Success: {goal_reached and not time_expired}")
                    
                    self.stop()
                    self.save(save_trajectory=True)
                    self.done = True
                    return
                else:
                    self.stop()
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

        elif self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time
            current_position = (current_x, current_z)

            if not self.ignore_goals:
                self._update_goal_visit_tracking(current_position, do_learning=True)

            stop_metric_met = self._non_exploit_stop_condition_met(trial_elapsed_time)
            goal_constraints_met = self._goal_constraints_met()
            coverage_reached = self._check_coverage_complete() if hasattr(self, "target_coverage_percentage") else False

            if stop_metric_met and goal_constraints_met:
                reason = self._describe_non_exploit_stop_reason(coverage_reached)
                print(f"[LEARN_LOCATIONS_COVERAGE] {reason} and goal constraints satisfied! "
                      f"Coverage: {self.current_coverage_percentage*100:.1f}%, "
                      f"Time: {trial_elapsed_time:.1f}s")

                # Create reward maps for each goal
                if not self.ignore_goals:
                    self._create_multi_goal_reward_maps()
                    self._save_multi_goal_data()

                # Save trial completion time for batched execution
                if self.trial_base_dir is not None:
                    self._save_trial_completion_time(trial_elapsed_time)

                self._finalize_non_exploit_completion(
                    include_networks=True,
                    stop_reason=reason,
                    sim_elapsed_time=trial_elapsed_time,
                )
            elif coverage_reached and not goal_constraints_met:
                print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                      f"but goal constraints incomplete, continuing...")
            elif stop_metric_met and not goal_constraints_met:
                print(f"[LEARN_LOCATIONS_COVERAGE] Stop condition met but goal constraints incomplete, continuing...")

        elif self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
            # Calculate trial elapsed time relative to trial start
            trial_elapsed_time = self.getTime() - self.trial_start_time

            # Check only active goal for exploitation
            active_goals = [g for g in self.goals if g.get("active", False)]
            for goal in active_goals:
                distance = self._distance_to_location_2d(current_x, current_z, goal["location"])

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
        for scale_idx, pcn in enumerate(self.pcns):
            # Find most active place cell for this scale
            most_active_idx = torch.argmax(pcn.place_cell_activations).item()
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
                        old_strength = self._compute_place_cell_connection_strength(pcn, stored_idx)
                        new_strength = self._compute_place_cell_connection_strength(pcn, most_active_idx)

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

    def _compute_place_cell_connection_strength(self, pcn, pc_idx):
        """
        Compute place cell quality based on recurrent connection strength.

        Args:
            pcn: Place cell network
            pc_idx: Index of place cell to evaluate

        Returns:
            float: Total recurrent connection strength
        """
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
        if not hasattr(self, "goal_place_cell_associations"):
            return False
        if not hasattr(self, "goal_visit_counts"):
            return False

        # All goals must be visited
        if not all(goal["visited"] for goal in self.goals):
            return False

        # All goals must have place cell associations for all scales
        for goal_name, associations in self.goal_place_cell_associations.items():
            for scale_idx, pc_idx in enumerate(associations):
                if pc_idx is None:
                    return False

        # All goals must have minimum number of visits
        for goal_name, visit_count in self.goal_visit_counts.items():
            if visit_count < self.min_goal_visits:
                return False

        return True

    def _create_multi_goal_reward_maps(self):
        """Create reward maps for each goal-scale combination."""
        return create_multi_goal_reward_maps(self)

    def _save_multi_goal_data(self):
        """Save multi-goal specific data."""
        return save_multi_goal_data(self)

    def _save_trial_completion_time(self, trial_elapsed_time):
        """Save trial completion time to JSON for AUTO mode."""
        return save_trial_completion_time(self, trial_elapsed_time)

    def _handle_random_goal_exploitation(self, goal):
        """Handle goal reached during EXPLOIT_LOCATIONS_RANDOM mode."""
        return handle_random_goal_exploitation(self, goal)

    def _handle_random_exploitation_timeout(self):
        """Handle timeout during EXPLOIT_LOCATIONS_RANDOM mode."""
        return handle_random_exploitation_timeout(self)

    ########################################### HELPER METHODS ###########################################

    def forward(self):
        """Moves the robot forward at maximum speed.

        Sets both wheels to max speed, updates motor movement and sensor readings.
        """
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        if self.hybrid_runtime:
            self._step_only()
        else:
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
            print(f"  Total turn needed: {total_angle_needed:+.1f}Â°")

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
                print(f"  Attempt {attempts}: turning {turn_this_step:+.1f}Â° (remaining: {remaining_turn:+.1f}Â°)")

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
                print(f"    Warning: large step error {step_error:.1f}Â°")

        # Final verification
        final_error = abs(angle_difference(target_heading_deg, current_heading))
        success = final_error <= ACCEPTABLE_ERROR

        if debug:
            total_turn_actual = angle_difference(current_heading, initial_heading)
            print(f"  Final: {initial_heading:.0f}Â° â†’ {current_heading:.0f}Â° (error: {final_error:.1f}Â°)")
            print(f"  Result: {'âœ“ SUCCESS' if success else 'âœ— FAILED'}")

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
            print(f"Turning from {self.current_heading_deg:.0f}Â° to {heading_deg:.0f}Â°")

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

                status = "âœ“" if angle_error < 15 else "âš " if angle_error < 30 else "âœ—"
                print(f"Moved {actual_distance:.3f}m at {actual_angle:.0f}Â° (target: {heading_deg}Â°) {status}")

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
        self._step_and_sense()
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
            self._step_and_sense()
            if not orientation < neg * angle:
                break
        self.stop()
        self._step_and_sense()

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
        if self.step_count < 2:
            return 0.0

        recorded_path = self.hmap_loc[:self.step_count]
        path_length = 0
        for i in range(recorded_path.shape[0] - 1):
            current_position = np.array(
                [recorded_path[i, 2], recorded_path[i, 0]]
            )
            next_position = np.array(
                [recorded_path[i + 1, 2], recorded_path[i + 1, 0]]
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
            if update_pcn and len(self.hmap_pcn_activities) != len(self.pcn_activations_list):
                self.hmap_pcn_activities = [
                    torch.zeros((self.num_steps, act.shape[0]), device=self.device, dtype=torch.float32)
                    for act in self.pcn_activations_list
                ]

        # 4) Update place cell activations for each scale (Direct assignment)
        scale_idx_map = self.scale_idx_map

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
                        (self.num_steps, act.shape[0]), device=self.device, dtype=torch.float32
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
                        (self.num_steps, act.shape[0]), device=self.device, dtype=torch.float32
                    )

                # Store activations directly
                self.hmap_gcn_activities[mapped_index][self.step_count] = act

        # 5) Update scale priority
        if update_scale_priority and hasattr(self, 'scale_idx'):
            self.hmap_scale_priority[self.step_count] = self.scale_idx

        # 6) Update proximity value if available
        if update_prox and hasattr(self, 'prox'):
            self.hmap_prox[self.step_count] = self.prox

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
        distance_to_goal = self._distance_to_location_2d(curr_pos[0], curr_pos[2], self.goal_location)

        # Determine the correct goal radius based on the current mode
        if self.robot_mode == RobotMode.EXPLOIT:
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
        """Persist outputs (delegated to artifact helper)."""
        return save_driver_outputs(
            self,
            include_pcn=include_pcn,
            include_rcn=include_rcn,
            include_gcn=include_gcn,
            include_hmaps=include_hmaps,
            save_trajectory=save_trajectory,
        )

    def clear(self):
        """Clear saved network and hmap artifacts."""
        return clear_driver_outputs(self)

