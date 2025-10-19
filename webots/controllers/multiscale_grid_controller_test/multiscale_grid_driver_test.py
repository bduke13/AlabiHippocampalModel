import numpy as np
from numpy.random import default_rng
import pickle
import os
import tkinter as tk
from tkinter import N, messagebox
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from controller import Supervisor
from astropy.stats import circmean
import random
import math
import copy

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.multiscale_pcn import PlaceCellLayer
from core.layers.multiscale_pcn_with_gcn_test import MultiscalePlaceCellWithGrid
from core.layers.grid_cell_layer_test import GridCellLayer
from core.layers.reward_cell_layer_test import RewardCellLayerTest
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

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
        optimal_path_distance: Optional[float] = None,
        path_failure_ratio: Optional[float] = None,
        paths_folder: Optional[str] = None,
        hmaps_folder: Optional[str] = None,
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

        # Directories for saving/loading data
        if hmaps_folder:
            self.hmap_dir = hmaps_folder
        else:
            self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")

        self.network_dir = os.path.join("pkl", self.world_name, "networks")
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
        
        # Store learning rates for later use in RCN initialization
        self.rcn_learning_rates = rcn_learning_rates if rcn_learning_rates is not None else [0.1] * len(scales)
        self.td_learning = td_learning
        self.use_prox_mod = use_prox_mod

        # Store coverage parameters
        self.environment_size = environment_size
        self.grid_size = grid_size
        self.coverage_percentage = coverage_percentage

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
        self.rcns = []
        self.load_pcns(enable_ojas, enable_stdp)
        self.load_rcns()
        print(f"[DRIVER] Using PCNs: {[f'pcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")
        print(f"[DRIVER] Using RCNs: {[f'rcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Initialize alpha as a tensor of zeros with the same length as the number of scales
        self.alpha = torch.zeros(len(self.scales), dtype=self.dtype, device=self.device)

        self.hmap_scale_priority = torch.zeros(self.num_steps, device="cuda", dtype=torch.float32)

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        self.hmap_prox = torch.zeros((self.num_steps,), device="cuda", dtype=torch.float32)
        self.hmap_scale_priority = torch.zeros(
            (self.num_steps, len(self.scales)),  # row per step, col per scale
            device="cuda", 
            dtype=torch.float32
        )

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
        
        # Parameters for loop detection
        self.LOOP_THRESHOLD = 5  # Number of loops before forcing exploration
        self.MAX_STEPS_BETWEEN_LOOPS = 10  # Max steps between loops to count towards threshold
        self.force_explore_count = 0 # Number of steps to force exploration (init to 0)

        # Optionally keep a single-scale reference
        self.pcn = self.pcns[0] if self.pcns else None
        self.rcn = self.rcns[0] if self.rcns else None

        # For EXPLOIT_LOCATIONS_RANDOM, load goal-specific RCNs
        if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM and hasattr(self, 'active_goal_name'):
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
        self.pcns = []
        for i, scale_def in enumerate(self.scales):
            scale_idx = scale_def["scale_index"]
            fname = f"pcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)

            # Get corresponding grid cell network (may be None)
            gcn = self.gcns[i]
            num_grid_cells = scale_def.get("num_grid_cells", 0) if gcn else 0

            pcn = self._load_or_init_pcn_for_scale(
                path,
                scale_def,
                num_grid_cells,
                enable_ojas if enable_ojas else None,
                enable_stdp if enable_stdp else None,
            )

            self.pcns.append(pcn)

    def _load_or_init_pcn_for_scale(self, path, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        try:
            with open(path, "rb") as f:
                pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing PCN from {path}")

            # Check PCN version compatibility
            pcn_class_name = pcn.__class__.__name__

            if pcn_class_name == "MultiscalePlaceCellWithGrid":
                # Update parameters if needed
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp

                # Update grid cell parameters from scale definition
                pcn.grid_influence = scale_def.get("grid_influence", 0.5)
                pcn.gamma_pg = scale_def.get("gamma_pg", 0.3)

                # Update proximity suppression parameters from scale definition
                pcn.enable_proximity_suppression = scale_def.get("enable_proximity_suppression", True)
                pcn.proximity_threshold_factor = scale_def.get("proximity_threshold_factor", 2.0)
                pcn.proximity_suppression_steepness = scale_def.get("proximity_suppression_steepness", 10.0)
                pcn.proximity_suppression_midpoint = scale_def.get("proximity_suppression_midpoint", 0.5)

                print(f"[DRIVER] Updated MultiscalePlaceCellWithGrid PCN - grid_influence: {pcn.grid_influence}, proximity_suppression: {pcn.enable_proximity_suppression}")
            else:
                # Update legacy PCN parameters
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp

            print(f"[DRIVER] Updated PCN for {path} - enable_ojas: {pcn.enable_ojas}, enable_stdp: {pcn.enable_stdp}")

        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[DRIVER] Initializing new PCN for {path}")
            bvc = BoundaryVectorCellLayer(
                max_dist=self.max_dist,
                n_res=720,
                n_hd=self.n_hd,
                sigma_theta=scale_def.get("sigma_theta"),
                sigma_r=scale_def.get("sigma_r"),
                device=self.device,
            )

            # Check if grid cells are enabled for this scale
            if num_grid_cells > 0:
                # Get proximity suppression parameters
                enable_proximity_suppression = scale_def.get("enable_proximity_suppression", True)
                proximity_threshold_factor = scale_def.get("proximity_threshold_factor", 2.0)
                proximity_suppression_steepness = scale_def.get("proximity_suppression_steepness", 10.0)
                proximity_suppression_midpoint = scale_def.get("proximity_suppression_midpoint", 0.5)

                # Use MultiscalePlaceCellWithGrid
                pcn = MultiscalePlaceCellWithGrid(
                    bvc_layer=bvc,
                    num_pc=scale_def["num_pc"],
                    num_grid_cells=num_grid_cells,
                    timestep=self.timestep,
                    n_hd=self.n_hd,
                    enable_ojas=enable_ojas if enable_ojas is not None else False,
                    enable_stdp=enable_stdp if enable_stdp is not None else False,
                    grid_influence=scale_def.get("grid_influence", 0.5),
                    gamma_pp=scale_def.get("gamma_pp", 0.5),
                    gamma_pb=scale_def.get("gamma_pb", 0.3),
                    gamma_pg=scale_def.get("gamma_pg", 0.3),
                    enable_proximity_suppression=enable_proximity_suppression,
                    proximity_threshold_factor=proximity_threshold_factor,
                    proximity_suppression_steepness=proximity_suppression_steepness,
                    proximity_suppression_midpoint=proximity_suppression_midpoint,
                    device=self.device,
                )
                print(f"[DRIVER] Created MultiscalePlaceCellWithGrid with {num_grid_cells} grid cells, grid_influence={scale_def.get('grid_influence', 0.5)}, proximity_suppression={enable_proximity_suppression}")
            else:
                # Use standard PlaceCellLayer
                pcn = PlaceCellLayer(
                    bvc_layer=bvc,
                    num_pc=scale_def["num_pc"],
                    timestep=self.timestep,
                    n_hd=self.n_hd,
                    enable_ojas=enable_ojas,
                    enable_stdp=enable_stdp,
                    device=self.device,
                )
                print(f"[DRIVER] Created standard PlaceCellLayer without grid cells")

        return pcn

    def load_rcns(self):
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
            rcn = RewardCellLayerTest(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                device=self.device,
            )
        return rcn

    def _load_goal_specific_rcns(self, goal_name):
        """Load goal-specific RCNs for EXPLOIT_LOCATIONS_RANDOM mode"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        if not os.path.exists(multi_goal_dir):
            print(f"[WARNING] Multi-goal rewards directory not found: {multi_goal_dir}")
            print(f"[WARNING] Make sure to run LEARN_LOCATIONS_COVERAGE first!")
            return

        print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")
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
                print(f"[ERROR] Make sure LEARN_LOCATIONS_COVERAGE has been run for this goal!")
                raise

        # Update single-scale reference
        self.rcn = self.rcns[0] if self.rcns else None
        print(f"[DRIVER] Loaded {len(self.rcns)} goal-specific RCNs for '{goal_name}'")

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

                # Extract grid cell parameters or use defaults
                num_grid_cells = scale_def.get("num_grid_cells", 0)
                rotation_range = scale_def.get("rotation_range", (0, 90))
                spread_range = scale_def.get("spread_range", (1.2, 1.2))
                translation_factor = scale_def.get("translation_factor", 1.0)
                frequency_divisor = scale_def.get("frequency_divisor", 1.0)

                # Skip grid cell creation if num_grid_cells is 0
                if num_grid_cells == 0:
                    self.gcns.append(None)
                    continue

                # Create new grid cell network
                gcn = GridCellLayer(
                    num_cells=num_grid_cells,
                    size_range=(0.5, 0.5),  # Base size range (will be modified by frequency_divisor)
                    rotation_range=rotation_range,
                    spread_range=spread_range,
                    translation_factor=translation_factor,
                    frequency_divisor=frequency_divisor,
                    threshold=0.7,
                    threshold_type='soft',
                    normalization='per-cell',
                    device=self.device.type,
                    dtype=self.dtype
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
            if self.robot_mode in {RobotMode.LEARN_LOCATIONS_COVERAGE}:
                self.goal_place_cell_associations = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_association_step = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                # Initialize coverage tracking
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
                                     RobotMode.LEARN_LOCATIONS_COVERAGE):
                self.explore()
            elif self.robot_mode in (RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS_RANDOM):
                self.exploit_v3()
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

            # Update coverage tracking if in LEARN_LOCATIONS_COVERAGE mode
            if self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE:
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
        """
        Follows the reward gradient to reach the goal location, incorporating wall avoidance.
        """
        # -------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        # -------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
        self.check_goal_reached()

        # -------------------------------------------------------------------
        # 2) Detect obstacles and compute valid directions
        # -------------------------------------------------------------------
        min_safe_distance = 1.5  # Minimum distance to consider a direction safe
        num_steps_preplay = 1  # Number of future steps to "preplay"
        pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
        cancelled_angles = []  # Store blocked directions

        # Compute minimum distance in each head direction
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        # Evaluate reward potential for each valid direction
        for d in range(self.n_hd):
            if distances_per_hd[d] < min_safe_distance:
                pot_rew[d] = 0.0  # Block direction if too close to a wall
                cancelled_angles.append(d)
            else:
                # Predict place-cell activation for direction 'd'
                pcn_activations = self.pcn.preplay(d, num_steps=num_steps_preplay)

                # Update reward cell activations (without saving to memory)
                self.rcn.update_reward_cell_activations(pcn_activations, visit=False)

                # Take the maximum activation in reward cells as the "reward estimate"
                pot_rew[d] = torch.max(torch.nan_to_num(self.rcn.reward_cell_activations))

        # -------------------------------------------------------------------
        # 3) Handle case where all directions are blocked
        # -------------------------------------------------------------------
        if torch.all(pot_rew == 0.0):
            print("All directions blocked. Initiating forced exploration.")
            self.force_explore_count = 5
            self.explore()
            return

        # -------------------------------------------------------------------
        # 4) Compute circular mean of angles, weighted by the reward estimates
        # -------------------------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        
        # Exclude blocked directions from heading calculation
        valid_mask = pot_rew > 0.0
        angles_np = angles[valid_mask].cpu().numpy()
        weights_np = pot_rew[valid_mask].cpu().numpy()

        sin_component = np.sum(np.sin(angles_np) * weights_np)
        cos_component = np.sum(np.cos(angles_np) * weights_np)
        action_angle = np.arctan2(sin_component, cos_component)

        # Normalize angle to [0, 2π)
        if action_angle < 0:
            action_angle += 2 * np.pi

        # -------------------------------------------------------------------
        # 5) Convert action angle to a turn relative to the current global heading
        # -------------------------------------------------------------------
        angle_to_turn_deg = np.rad2deg(action_angle) - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180
        angle_to_turn = np.deg2rad(angle_to_turn_deg)

        # Store cancelled angles for visualization/debugging
        self.cancelled_angles_deg = [np.rad2deg(angles[d].cpu().item()) for d in cancelled_angles]

        # -------------------------------------------------------------------
        # 6) Execute the turn and optionally move forward
        # -------------------------------------------------------------------
        self.turn(angle_to_turn)
        self.forward()

        # (Optional) Re-sense and compute after movement
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)

    def exploit_v1(self):
        """
        Follows the reward gradient to reach the goal location using multiscale place field navigation.
        If a scale has insufficient reward information, it is temporarily removed from the decision-making 
        while still allowing learning.
        """
        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_scale_priority=True)
        self.check_goal_reached()

        # Save old PCN activations for each scale
        old_pcn_activations = [pcn.place_cell_activations.clone() for pcn in self.pcns]

        # Exploit can only begin with at least 10 steps
        if self.step_count <= self.tau_w:
            return

        #-------------------------------------------------------------------
        # 2) Forced Exploration Check & Cooldown Setup
        #-------------------------------------------------------------------
        if self.force_explore_count > 0:
            self.force_explore_count -= 1
            if self.force_explore_count == 0:
                print("Forced exploration complete. Checking for improved reward signal...")

                # Compute the max post-explore reward gradient
                self.max_post_explore_gradient = max(
                    torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
                    for rcn in self.rcns
                )

            self.explore()
            return

        #-------------------------------------------------------------------
        # 3) Detect excessive rotation and enforce forced exploration if needed
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

        if self.rotation_accumulator >= 360.0:
            self.rotation_loop_count += 1
            self.rotation_accumulator -= 360.0
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 1
            self.steps_since_last_loop = 0

            if self.rotation_loop_count >= self.LOOP_THRESHOLD:
                self.rotation_loop_count = 0
                self.rotation_accumulator = 0.0
                self.steps_since_last_loop = 0
                print(f"Detected {self.LOOP_THRESHOLD} consecutive loops within {self.MAX_STEPS_BETWEEN_LOOPS} steps. Initiating forced exploration.")

                # Store the preferred scale before entering forced exploration
                if hasattr(self, "last_preferred_scale_index"):
                    self.cooldown_scale_index = self.last_preferred_scale_index
                    self.cooldown_steps_remaining = 20  # Cooldown for 10 steps
                    print(f"[INFO] Cooling down scale {self.cooldown_scale_index} for 10 steps.")

                # Start forced exploration
                self.force_explore_count = 5
                self.explore()
                return
        else:
            self.steps_since_last_loop += 1
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 0

        #-------------------------------------------------------------------
        # 4) Compute potential rewards at multiple scales, skipping invalid directions
        #-------------------------------------------------------------------
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        pot_rew_scales = []
        valid_scale_indices = []
        reward_threshold = 0.1  # Threshold for considering a scale valid

        for i, scale_def in enumerate(self.scales):
            if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
                if i == getattr(self, "cooldown_scale_index", -1):
                    print(f"[INFO] Skipping scale {i} due to cooldown.")
                    continue  # Skip the scale that is on cooldown

            pcn, rcn = self.pcns[i], self.rcns[i]
            pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
            current_pcn_activations = pcn.place_cell_activations

            # Compute current reward at the agent's position
            rcn.update_reward_cell_activations(current_pcn_activations, visit=False)
            current_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))

            for d in range(self.n_hd):
                if distances_per_hd[d] < 1:
                    pot_rew[d] = 0.0
                else:
                    preplayed_pcn_activations = pcn.preplay(d)
                    rcn.update_reward_cell_activations(preplayed_pcn_activations, visit=False)
                    preplayed_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))
                    pot_rew[d] = preplayed_reward

            if pot_rew.max().item() >= reward_threshold:
                pot_rew_scales.append(pot_rew)
                valid_scale_indices.append(i)

        #-------------------------------------------------------------------
        # 3.5) If NO valid scales remain, trigger exploration
        #-------------------------------------------------------------------
        if len(pot_rew_scales) == 0:
            print(f"[WARNING] No valid scales found (all below reward threshold {reward_threshold}), forcing exploration.")
            self.explore()
            return

        #-------------------------------------------------------------------
        # 4.5) Reduce cooldown step count
        #-------------------------------------------------------------------
        if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
            self.cooldown_steps_remaining -= 1
            if self.cooldown_steps_remaining == 0:
                print(f"[INFO] Cooldown complete for scale {self.cooldown_scale_index}. Scale re-enabled.")

        #-------------------------------------------------------------------
        # 4) Normalize and blend across valid scales
        #-------------------------------------------------------------------
        pot_rew_scales = torch.stack(pot_rew_scales)
        pot_rew_scales /= (pot_rew_scales.max(dim=1, keepdim=True)[0] + 1e-6)  # Avoid div by zero

        # Compute gradients only on valid scales
        grads = torch.sum(torch.abs(torch.diff(pot_rew_scales, dim=1)), dim=1)

        # Apply Gaussian smoothing
        kernel_size = 3  # Adjust as needed (3, 5, or 7 are common choices)
        sigma = 3.0  # Standard deviation for Gaussian smoothing

        # Create Gaussian kernel
        def gaussian_kernel(size: int, sigma: float, device):
            x = torch.arange(size, dtype=self.dtype, device=device) - size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
            return kernel.view(1, 1, -1)  # Shape for 1D convolution

        gaussian = gaussian_kernel(kernel_size, sigma, self.device)

        # Apply smoothing via 1D convolution
        grads_unsq = grads.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        grads_smooth = F.conv1d(grads_unsq, gaussian, padding=kernel_size // 2).squeeze()

        # Compute initial mixing weights based on smoothed gradients
        mixing_weights = grads_smooth / (grads_smooth.sum() + 1e-6)

        mixing_weights = grads / (grads.sum() + 1e-6)
        mixing_weights = torch.clamp(mixing_weights, min=0.0, max=1.0)
        mixing_weights /= mixing_weights.sum()

        # If proximity modulation is enabled, adjust the weights
        if self.use_prox_mod:
            prox_weight = self.prox  # self.prox is in [0, 1]
            print(f"Proximity weight: {prox_weight}")
            # Bias smaller scales more heavily when close to the goal
            scale_biases = torch.tensor(
                [1.0 / (i + 1) for i in range(len(valid_scale_indices))],
                dtype=self.dtype, device=self.device
            )
            scale_biases /= scale_biases.sum()

            # Blend mixing_weights with scale_biases based on proximity weight
            mixing_weights = (1 - prox_weight) * mixing_weights + prox_weight * scale_biases
            mixing_weights /= mixing_weights.sum()

        # Determine the preferred scale
        if mixing_weights.numel() > 1:  # Ensure multiple elements in mixing_weights
            preferred_scale_index = torch.argmax(mixing_weights).item()
            print(f"Preferred scale: {preferred_scale_index}, Weight: {mixing_weights[preferred_scale_index].item()}")
        else:
            preferred_scale_index = 0  # Default to the only available scale

        # Apply hysteresis only if we have multiple scales to choose from
        if hasattr(self, "last_preferred_scale_index") and self.last_preferred_scale_index is not None:
            if mixing_weights.numel() > self.last_preferred_scale_index:  # Ensure valid index
                if abs(mixing_weights[self.last_preferred_scale_index] - mixing_weights[preferred_scale_index]) < 0.1:
                    preferred_scale_index = self.last_preferred_scale_index  # Stick with previous scale

        # Now update the last preferred scale index
        self.last_preferred_scale_index = preferred_scale_index

        # Update scale priority for logging
        self.scale_idx = valid_scale_indices[torch.argmax(mixing_weights).item()]

        if mixing_weights.numel() > 1:
            combined_pot_rew = torch.sum(mixing_weights[:, None] * pot_rew_scales, dim=0)
        else:
            # Directly use the only valid reward scale
            combined_pot_rew = pot_rew_scales.squeeze(0)  # Remove batch dim if needed


        #-------------------------------------------------------------------
        # 5) Compute action heading
        #-------------------------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * combined_pot_rew)
        cos_component = torch.sum(torch.cos(angles) * combined_pot_rew)
        action_angle = torch.atan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        #-------------------------------------------------------------------
        # 6) Execute movement
        #-------------------------------------------------------------------
        angle_to_turn_deg = self.action_heading_deg - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180
        angle_to_turn = torch.deg2rad(torch.tensor(angle_to_turn_deg, dtype=self.dtype, device=self.device))
        self.turn(angle_to_turn.item())

        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True,
                              update_pcn=True,
                              update_scale_priority=True)
            self.forward()
            self.check_goal_reached()

            # Update rotation accumulator
            heading_diff = self.current_heading_deg - getattr(self, 'last_heading_deg', self.current_heading_deg)
            heading_diff = ((heading_diff + 180) % 360) - 180
            self.rotation_accumulator += abs(heading_diff)
            self.last_heading_deg = self.current_heading_deg

            if self.done:
                return

        #--------------------------
        # 7) (Optional) TD Learning Step
        #--------------------------
        if self.td_learning:
            for i, scale_def in enumerate(self.scales):
                pcn, rcn = self.pcns[i], self.rcns[i]
                new_pcn_activations = pcn.place_cell_activations
                rcn.update_reward_cell_activations(new_pcn_activations, visit=False)
                observed_reward = float(rcn.reward_cell_activations.item())
                rcn.td_update(old_pcn_activations[i], observed_reward)

        return

    def exploit_V2(self):

        # -------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary/grid cell activations
        # -------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True)
        self.check_goal_reached()

        # -------------------------------------------------------------------
        # 2) Detect obstacles and compute valid directions
        # -------------------------------------------------------------------
        min_safe_distance = 1.5  # Minimum distance to consider a direction safe
        num_steps_preplay = 3  # Number of future steps to "preplay"
        pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
        cancelled_angles = []  # Store blocked directions

        # Compute minimum distance in each head direction
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        # -------------------------------------------------------------------
        # 3) Define step sizes for each scale and calculate valid preplay steps
        # -------------------------------------------------------------------
        # Step size represents the approximate distance covered per preplay step for each scale
        scale_step_sizes = {
            "small": 1.0,   # 1 meter per preplay step
            "medium": 6.0,  # 2 meters per preplay step
            "large": 12.0,   # 4 meters per preplay step
            "xlarge": 4.0   # 4 meters per preplay step (same as large)
        }

        # Calculate valid preplay steps for each direction and each scale
        # Structure: preplay_steps_per_direction[direction_idx][scale_idx] = num_steps
        preplay_steps_per_direction = []

        for d in range(self.n_hd):
            min_distance = distances_per_hd[d].item()  # Get minimum distance for this direction

            # Calculate valid preplay steps for each scale at this direction
            scale_preplay_steps = []
            for scale_def in self.scales:
                scale_name = scale_def["name"]
                step_size = scale_step_sizes.get(scale_name, 1.0)  # Default to 1.0 if scale not found

                # Check if direction is blocked by safety distance
                if min_distance < min_safe_distance:
                    # Direction is blocked - no preplay allowed
                    valid_steps = 0
                else:
                    # Calculate maximum steps this scale can take without hitting obstacle
                    # Formula: max_steps = floor(min_distance / step_size)
                    max_steps_for_scale = int(min_distance / step_size)

                    # Take minimum of the configured max and what's physically safe
                    # Ensure at least 1 step if not blocked
                    valid_steps = max(1, min(num_steps_preplay, max_steps_for_scale))

                scale_preplay_steps.append(valid_steps)

            preplay_steps_per_direction.append(scale_preplay_steps)

        # Optional: Print debug info for first few directions
        if self.step_count % 100 == 0:  # Print every 100 steps to avoid spam
            print(f"[EXPLOIT_V2] Preplay steps calculation (step {self.step_count}):")
            for d in range(min(3, self.n_hd)):  # Show first 3 directions
                print(f"  Direction {d} (distance: {distances_per_hd[d].item():.2f}m):")
                for i, scale_def in enumerate(self.scales):
                    print(f"    {scale_def['name']}: {preplay_steps_per_direction[d][i]} steps")

        # -------------------------------------------------------------------
        # 4) Multi-scale reward evaluation using constrained multi-step preplay
        # -------------------------------------------------------------------
        all_scale_rewards = []
        all_scale_gradients = []
        decay_factor = 0.4  # Exponential decay for step weighting 0.4
        threshold = 0.01  # Threshold to filter noise

        for scale_idx, (pcn, rcn) in enumerate(zip(self.pcns, self.rcns)):
            # Setup evaluation function for this scale's RCN
            def evaluate_with_rcn(activations):
                rcn.update_reward_cell_activations(activations, visit=False)
                return torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()

            # Temporarily override the PCN's evaluation method
            original_eval_method = getattr(pcn, '_evaluate_activations_for_reward', None)
            pcn._evaluate_activations_for_reward = evaluate_with_rcn

            # Evaluate rewards for all 8 directions using multi-step preplay
            direction_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)

            for direction in range(self.n_hd):
                # Get the valid number of preplay steps for this direction and scale
                valid_steps = preplay_steps_per_direction[direction][scale_idx]

                if valid_steps == 0:
                    # Direction is blocked
                    direction_rewards[direction] = 0.0
                else:
                    # Use multi-step preplay with scale-specific step count
                    reward = pcn.multi_step_preplay_constrained_weighted(
                        forced_first_direction=direction,
                        max_steps=valid_steps,
                        decay_factor=decay_factor,
                        debug=False
                    )
                    direction_rewards[direction] = reward

            # Restore original evaluation method
            if original_eval_method is not None:
                pcn._evaluate_activations_for_reward = original_eval_method
            elif hasattr(pcn, '_evaluate_activations_for_reward'):
                delattr(pcn, '_evaluate_activations_for_reward')

            # Apply threshold to filter noise
            thresholded_rewards = torch.where(
                direction_rewards >= threshold,
                direction_rewards,
                torch.zeros_like(direction_rewards)
            )

            # Normalize rewards for this scale
            max_reward = thresholded_rewards.max().item()
            if max_reward > 1e-6:
                normalized_rewards = thresholded_rewards / max_reward
            else:
                normalized_rewards = thresholded_rewards

            # Compute gradient (sum of absolute differences between adjacent directions)
            gradient = torch.sum(torch.abs(torch.diff(normalized_rewards))).item()

            all_scale_rewards.append(thresholded_rewards)
            all_scale_gradients.append(gradient)

        # -------------------------------------------------------------------
        # 5) Gradient-based scale weighting
        # -------------------------------------------------------------------
        total_gradient = sum(all_scale_gradients)

        if total_gradient < 1e-6:
            # All scales have zero gradients - fall back to exploration
            print("[EXPLOIT_V2] All scales have zero gradients - falling back to exploration")
            self.explore()
            return

        # Compute mixing weights based on relative gradients
        mixing_weights = torch.tensor(all_scale_gradients, dtype=self.dtype, device=self.device) / total_gradient

        if self.step_count % 100 == 0:
            print(f"[EXPLOIT_V2] Scale mixing weights (gradient-based):")
            for i, (scale_def, weight, gradient) in enumerate(zip(self.scales, mixing_weights, all_scale_gradients)):
                print(f"  Scale {i} ({scale_def['name']}): weight={weight:.3f}, gradient={gradient:.3f}")

        # -------------------------------------------------------------------
        # 6) Combine rewards across scales using mixing weights
        # -------------------------------------------------------------------
        combined_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        for direction in range(self.n_hd):
            combined_rewards[direction] = sum(
                mixing_weights[k] * all_scale_rewards[k][direction]
                for k in range(len(all_scale_rewards))
            )

        # -------------------------------------------------------------------
        # 7) Handle case where all combined rewards are zero (forced exploration)
        # -------------------------------------------------------------------
        if torch.all(combined_rewards == 0.0):
            print("All directions blocked. Initiating forced exploration.")
            self.force_explore_count = 5
            self.explore()
            return

        # -------------------------------------------------------------------
        # 8) Compute action heading using circular mean (weighted by combined rewards)
        # -------------------------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * combined_rewards)
        cos_component = torch.sum(torch.cos(angles) * combined_rewards)
        action_angle = torch.atan2(sin_component, cos_component)

        # Normalize to [0, 2π)
        if action_angle < 0:
            action_angle += 2 * np.pi

        action_heading_deg = float(torch.rad2deg(action_angle).item())

        if self.step_count % 100 == 0:
            best_direction = torch.argmax(combined_rewards).item()
            best_reward = combined_rewards[best_direction].item()
            print(f"[EXPLOIT_V2] Action: target={action_heading_deg:.0f}°, best_dir={best_direction} ({best_direction*45}°), reward={best_reward:.4f}")

        # -------------------------------------------------------------------
        # 9) Execute movement using improved turning method
        # -------------------------------------------------------------------
        self._execute_movement(action_heading_deg, False)

        # Update distance tracking if in EXPLOIT_LOCATIONS_RANDOM mode
        if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
            self._update_distance_tracking()

        return


    def exploit_v3(self):
        """
        Combines exploit_v1 structure with exploit_V2's multistep preplay logic.
        - Uses exploit_v1's rotation loop detection, cooldown mechanism, and proximity modulation
        - Uses exploit_V2's multi-step preplay with scale-specific step sizes
        - Follows the reward gradient using constrained multi-step preplay
        """
        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_scale_priority=True)
        self.check_goal_reached()

        # Save old PCN activations for each scale (for TD learning)
        old_pcn_activations = [pcn.place_cell_activations.clone() for pcn in self.pcns]

        # Exploit can only begin with at least 10 steps
        if self.step_count <= self.tau_w:
            return

        #-------------------------------------------------------------------
        # 2) Forced Exploration Check & Cooldown Setup
        #-------------------------------------------------------------------
        if self.force_explore_count > 0:
            self.force_explore_count -= 1
            if self.force_explore_count == 0:
                print("Forced exploration complete. Checking for improved reward signal...")

                # Compute the max post-explore reward gradient
                self.max_post_explore_gradient = max(
                    torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
                    for rcn in self.rcns
                )

            self.explore()
            return

        #-------------------------------------------------------------------
        # 3) Detect excessive rotation and enforce forced exploration if needed
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

        if self.rotation_accumulator >= 360.0:
            self.rotation_loop_count += 1
            self.rotation_accumulator -= 360.0
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 1
            self.steps_since_last_loop = 0

            if self.rotation_loop_count >= self.LOOP_THRESHOLD:
                self.rotation_loop_count = 0
                self.rotation_accumulator = 0.0
                self.steps_since_last_loop = 0
                print(f"Detected {self.LOOP_THRESHOLD} consecutive loops within {self.MAX_STEPS_BETWEEN_LOOPS} steps. Initiating forced exploration.")

                # Store the preferred scale before entering forced exploration
                if hasattr(self, "last_preferred_scale_index"):
                    self.cooldown_scale_index = self.last_preferred_scale_index
                    self.cooldown_steps_remaining = 20  # Cooldown for 20 steps
                    print(f"[INFO] Cooling down scale {self.cooldown_scale_index} for 20 steps.")

                # Start forced exploration
                self.force_explore_count = 5
                self.explore()
                return
        else:
            self.steps_since_last_loop += 1
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 0

        #-------------------------------------------------------------------
        # 4) Compute obstacle distances per head direction
        #-------------------------------------------------------------------
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        #-------------------------------------------------------------------
        # 5) Define scale-specific step sizes and calculate valid preplay steps
        #    (from exploit_V2)
        #-------------------------------------------------------------------
        min_safe_distance = 1.0  # Minimum distance to consider a direction safe 1.5
        num_steps_preplay = 3  # Maximum number of future steps to "preplay"

        # Step size represents the approximate distance covered per preplay step for each scale
        scale_step_sizes = {
            "small": 1.0,   # 1 meter per preplay step
            "medium": 4.0,  # 6 meters per preplay step
            "large": 12.0,  # 12 meters per preplay step
            "xlarge": 4.0   # 4 meters per preplay step
        }

        # Calculate valid preplay steps for each direction and each scale
        preplay_steps_per_direction = []

        for d in range(self.n_hd):
            min_distance = distances_per_hd[d].item()

            # Calculate valid preplay steps for each scale at this direction
            scale_preplay_steps = []
            for scale_def in self.scales:
                scale_name = scale_def["name"]
                step_size = scale_step_sizes.get(scale_name, 1.0)

                # Check if direction is blocked by safety distance
                if min_distance < min_safe_distance:
                    valid_steps = 0
                else:
                    # Calculate maximum steps this scale can take without hitting obstacle
                    max_steps_for_scale = int(min_distance / step_size)
                    valid_steps = max(1, min(num_steps_preplay, max_steps_for_scale))

                scale_preplay_steps.append(valid_steps)

            preplay_steps_per_direction.append(scale_preplay_steps)

        #-------------------------------------------------------------------
        # 6) Multi-scale reward evaluation using constrained multi-step preplay
        #    (from exploit_V2 but filtered by cooldown from v1)
        #-------------------------------------------------------------------
        all_scale_rewards = []
        all_scale_gradients = []
        valid_scale_indices = []
        decay_factor = 0.5  # Exponential decay for step weighting 0.4
        threshold = 0.015  # Threshold to filter noise  0.1
        reward_threshold = 0.15  # Threshold for considering a scale valid 0.1

        for i, (scale_def, pcn, rcn) in enumerate(zip(self.scales, self.pcns, self.rcns)):
            # Check cooldown (from exploit_v1)
            if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
                if i == getattr(self, "cooldown_scale_index", -1):
                    print(f"[INFO] Skipping scale {i} due to cooldown.")
                    continue  # Skip the scale that is on cooldown

            # Setup evaluation function for this scale's RCN
            def evaluate_with_rcn(activations):
                rcn.update_reward_cell_activations(activations, visit=False)
                return torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()

            # Temporarily override the PCN's evaluation method
            original_eval_method = getattr(pcn, '_evaluate_activations_for_reward', None)
            pcn._evaluate_activations_for_reward = evaluate_with_rcn

            # Evaluate rewards for all 8 directions using multi-step preplay
            direction_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)

            for direction in range(self.n_hd):
                # Get the valid number of preplay steps for this direction and scale
                valid_steps = preplay_steps_per_direction[direction][i]

                if valid_steps == 0:
                    # Direction is blocked
                    direction_rewards[direction] = 0.0
                else:
                    # Use multi-step preplay with scale-specific step count
                    reward = pcn.multi_step_preplay_constrained_weighted(
                        forced_first_direction=direction,
                        max_steps=valid_steps,
                        decay_factor=decay_factor,
                        debug=False
                    )
                    direction_rewards[direction] = reward

            # Restore original evaluation method
            if original_eval_method is not None:
                pcn._evaluate_activations_for_reward = original_eval_method
            elif hasattr(pcn, '_evaluate_activations_for_reward'):
                delattr(pcn, '_evaluate_activations_for_reward')

            # Apply threshold to filter noise
            thresholded_rewards = torch.where(
                direction_rewards >= threshold,
                direction_rewards,
                torch.zeros_like(direction_rewards)
            )

            # Normalize rewards for this scale
            max_reward = thresholded_rewards.max().item()
            if max_reward > 1e-6:
                normalized_rewards = thresholded_rewards / max_reward
            else:
                normalized_rewards = thresholded_rewards

            # Compute gradient (sum of absolute differences between adjacent directions)
            gradient = torch.sum(torch.abs(torch.diff(normalized_rewards))).item()

            # Only include this scale if it has sufficient reward information
            if max_reward >= reward_threshold:
                all_scale_rewards.append(normalized_rewards)
                all_scale_gradients.append(gradient)
                valid_scale_indices.append(i)

        #-------------------------------------------------------------------
        # 7) Reduce cooldown step count (from exploit_v1)
        #-------------------------------------------------------------------
        if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
            self.cooldown_steps_remaining -= 1
            if self.cooldown_steps_remaining == 0:
                print(f"[INFO] Cooldown complete for scale {self.cooldown_scale_index}. Scale re-enabled.")

        #-------------------------------------------------------------------
        # 8) If NO valid scales remain, trigger exploration
        #-------------------------------------------------------------------
        if len(all_scale_rewards) == 0:
            print(f"[WARNING] No valid scales found (all below reward threshold {reward_threshold}), forcing exploration.")
            self.explore()
            return

        #-------------------------------------------------------------------
        # 9) Gradient-based scale weighting (from exploit_v1 approach)
        #-------------------------------------------------------------------
        total_gradient = sum(all_scale_gradients)

        if total_gradient < 1e-6:
            # All scales have zero gradients - fall back to exploration
            print("[EXPLOIT_V3] All scales have zero gradients - falling back to exploration")
            self.explore()
            return

        # Compute mixing weights based on relative gradients
        mixing_weights = torch.tensor(all_scale_gradients, dtype=self.dtype, device=self.device) / total_gradient

        # Apply Gaussian smoothing (from exploit_v1)
        if len(mixing_weights) > 1:
            kernel_size = 3
            sigma = 3.0

            def gaussian_kernel(size: int, sigma: float, device):
                x = torch.arange(size, dtype=self.dtype, device=device) - size // 2
                kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                kernel /= kernel.sum()
                return kernel.view(1, 1, -1)

            gaussian = gaussian_kernel(kernel_size, sigma, self.device)
            grads_tensor = torch.tensor(all_scale_gradients, dtype=self.dtype, device=self.device)
            grads_unsq = grads_tensor.unsqueeze(0).unsqueeze(0)
            grads_smooth = F.conv1d(grads_unsq, gaussian, padding=kernel_size // 2).squeeze()
            mixing_weights = grads_smooth / (grads_smooth.sum() + 1e-6)

        mixing_weights = torch.clamp(mixing_weights, min=0.0, max=1.0)
        mixing_weights /= mixing_weights.sum()

        # If proximity modulation is enabled, adjust the weights (from exploit_v1)
        if self.use_prox_mod:
            prox_weight = self.prox  # self.prox is in [0, 1]
            print(f"Proximity weight: {prox_weight}")
            # Bias smaller scales more heavily when close to the goal
            scale_biases = torch.tensor(
                [1.0 / (i + 1) for i in range(len(valid_scale_indices))],
                dtype=self.dtype, device=self.device
            )
            scale_biases /= scale_biases.sum()

            # Blend mixing_weights with scale_biases based on proximity weight
            mixing_weights = (1 - prox_weight) * mixing_weights + prox_weight * scale_biases
            mixing_weights /= mixing_weights.sum()

        # Determine the preferred scale (from exploit_v1)
        if mixing_weights.numel() > 1:
            preferred_scale_index = torch.argmax(mixing_weights).item()
            print(f"Preferred scale: {preferred_scale_index}, Weight: {mixing_weights[preferred_scale_index].item()}")
        else:
            preferred_scale_index = 0

        # Apply hysteresis (from exploit_v1)
        if hasattr(self, "last_preferred_scale_index") and self.last_preferred_scale_index is not None:
            if mixing_weights.numel() > self.last_preferred_scale_index:
                if abs(mixing_weights[self.last_preferred_scale_index] - mixing_weights[preferred_scale_index]) < 0.1:
                    preferred_scale_index = self.last_preferred_scale_index

        # Update the last preferred scale index
        self.last_preferred_scale_index = preferred_scale_index

        # Update scale priority for logging
        self.scale_idx = valid_scale_indices[torch.argmax(mixing_weights).item()]

        #-------------------------------------------------------------------
        # 10) Combine rewards across scales using mixing weights
        #-------------------------------------------------------------------
        if mixing_weights.numel() > 1:
            combined_pot_rew = torch.sum(mixing_weights[:, None] * torch.stack(all_scale_rewards), dim=0)
        else:
            combined_pot_rew = all_scale_rewards[0]

        #-------------------------------------------------------------------
        # 11) Compute action heading
        #-------------------------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * combined_pot_rew)
        cos_component = torch.sum(torch.cos(angles) * combined_pot_rew)
        action_angle = torch.atan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        #-------------------------------------------------------------------
        # 12) Execute movement
        #-------------------------------------------------------------------
        angle_to_turn_deg = self.action_heading_deg - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180
        angle_to_turn = torch.deg2rad(torch.tensor(angle_to_turn_deg, dtype=self.dtype, device=self.device))
        self.turn(angle_to_turn.item())

        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True,
                              update_pcn=True,
                              update_scale_priority=True)
            self.forward()
            self.check_goal_reached()

            # Update rotation accumulator
            heading_diff = self.current_heading_deg - getattr(self, 'last_heading_deg', self.current_heading_deg)
            heading_diff = ((heading_diff + 180) % 360) - 180
            self.rotation_accumulator += abs(heading_diff)
            self.last_heading_deg = self.current_heading_deg

            if self.done:
                return

        #-------------------------------------------------------------------
        # 13) (Optional) TD Learning Step
        #-------------------------------------------------------------------
        if self.td_learning:
            for i, scale_def in enumerate(self.scales):
                pcn, rcn = self.pcns[i], self.rcns[i]
                new_pcn_activations = pcn.place_cell_activations
                rcn.update_reward_cell_activations(new_pcn_activations, visit=False)
                observed_reward = float(rcn.reward_cell_activations.item())
                rcn.td_update(old_pcn_activations[i], observed_reward)

        return


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
                grid_activations = gcn.get_grid_cell_activations(position)
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
                pcn.bvc_layer.plot_activation(
                    self.pcn.bvc_layer.plot_activation(self.boundaries.cpu())
                )
            # Append activations to pcn_activations_list
            act = pcn.place_cell_activations.clone().detach()
            self.pcn_activations_list.append(pcn.place_cell_activations)

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

        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING) \
                and self.getTime() >= 60 * self.run_time_minutes:
            self.stop()
            self.save(include_pcn=self.robot_mode != RobotMode.PLOTTING,
                    include_rcn=self.robot_mode != RobotMode.PLOTTING,
                    include_gcn=self.robot_mode != RobotMode.PLOTTING,
                    include_hmaps=True)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return

        elif self.robot_mode == RobotMode.DMTP and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"]
        ):
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
            goal_reached = torch.allclose(
                torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
                torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
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
            current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

            # Check all goals for learning
            for goal in self.goals:
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)

                if distance <= goal["radius"]:
                    if not goal["visited"]:
                        print(f"[LEARN_LOCATIONS_COVERAGE] First visit to {goal['name']} goal at {goal['location']}")
                        goal["visited"] = True
                    self._handle_goal_learning(goal)

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

                # Create reward maps for each goal
                self._create_multi_goal_reward_maps()
                self._save_multi_goal_data()

                self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                self.done = True
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            elif coverage_reached and not learning_complete:
                print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                      f"but learning incomplete, continuing...")
            elif minimum_time_reached and not learning_complete:
                print(f"[LEARN_LOCATIONS_COVERAGE] Time limit reached but learning incomplete, continuing...")

        elif self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
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

    def _handle_goal_learning(self, goal):
        """Handle goal visits during learning mode"""
        for scale_idx, pcn in enumerate(self.pcns):
            # Find most active place cell for this scale
            most_active_idx = torch.argmax(pcn.place_cell_activations).item()
            activation_value = pcn.place_cell_activations[most_active_idx].item()

            # Only associate if there's meaningful activation
            if activation_value > 0.01:
                old_idx = self.goal_place_cell_associations[goal["name"]][scale_idx]
                self.goal_place_cell_associations[goal["name"]][scale_idx] = most_active_idx
                self.goal_association_step[goal["name"]][scale_idx] = self.step_count

                if old_idx != most_active_idx:
                    print(f"[LEARN_LOCATIONS] {goal['name']} scale {scale_idx}: "
                        f"PC {old_idx} -> PC {most_active_idx} (activation: {activation_value:.3f})")

    def _check_multi_goal_learning_complete(self):
        """Check if multi-goal learning is complete"""
        # All goals must be visited
        if not all(goal["visited"] for goal in self.goals):
            return False

        # All goals must have place cell associations for all scales
        for goal_name, associations in self.goal_place_cell_associations.items():
            for scale_idx, pc_idx in enumerate(associations):
                if pc_idx is None:
                    return False

        return True

    def _create_multi_goal_reward_maps(self):
        """Create reward maps for each goal-scale combination"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        os.makedirs(multi_goal_dir, exist_ok=True)

        print(f"[LEARN_LOCATIONS] Creating {len(self.goals)} goals × {len(self.scales)} scales reward maps")

        created_maps = 0
        for goal in self.goals:
            for scale_idx, (pcn, rcn) in enumerate(zip(self.pcns, self.rcns)):
                pc_idx = self.goal_place_cell_associations[goal["name"]][scale_idx]

                if pc_idx is None:
                    print(f"[WARNING] No place cell associated with {goal['name']} for scale {scale_idx}")
                    continue

                # Create artificial activation pattern
                artificial_activations = torch.zeros_like(pcn.place_cell_activations)
                artificial_activations[pc_idx] = 1.0

                print(f"[LEARN_LOCATIONS] Creating reward map for {goal['name']} scale {scale_idx}: using PC {pc_idx}")

                # Create goal-specific RCN
                goal_rcn = copy.deepcopy(rcn)
                goal_rcn.update_reward_cell_activations(artificial_activations, visit=True)

                # Use replay with custom activations if available, otherwise use standard replay
                if hasattr(goal_rcn, 'replay_with_custom_activations'):
                    goal_rcn.replay_with_custom_activations(pcn=pcn, custom_activations=artificial_activations)
                else:
                    # Standard replay approach
                    goal_rcn.replay(pcn=pcn)

                # Save goal-specific RCN
                goal_rcn_path = os.path.join(
                    multi_goal_dir, f"rcn_scale_{scale_idx}_goal_{goal['name']}.pkl"
                )
                with open(goal_rcn_path, "wb") as f:
                    pickle.dump(goal_rcn, f)

                created_maps += 1
                scale_name = self.scales[scale_idx]["name"]
                print(f"[LEARN_LOCATIONS] Created: {scale_name}_goal_{goal['name']} (PC {pc_idx})")

        print(f"[LEARN_LOCATIONS] Successfully created {created_maps} reward maps")

    def _save_multi_goal_data(self):
        """Save multi-goal specific data"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")

        # Save goal associations
        associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
        association_data = {
            "goal_place_cell_associations": self.goal_place_cell_associations,
            "goal_association_step": self.goal_association_step,
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
            print(f"  {goal_name} at {goal_info['location']}: {associations}")

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
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]

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
            use_trial_prefix = self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM

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

            # (c) Place-cell history maps for each scale
            for scale_def, pc_history in zip(self.scales, self.hmap_pcn_activities):
                scale_idx = scale_def["scale_index"]  # Get correct scale index
                hmap_scale_path = os.path.join(self.hmap_dir, f"{prefix}hmap_pcn_scale_{scale_idx}.pkl")

                with open(hmap_scale_path, "wb") as f:
                    pc_data = pc_history[: self.step_count].cpu().numpy()
                    pickle.dump(pc_data, f)
                files_saved.append(hmap_scale_path)

            # (d) Grid-cell history maps for each scale
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
        if not self.stats_collector:
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