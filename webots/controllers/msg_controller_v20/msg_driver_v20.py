import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from controller import Supervisor
import random
from typing import Optional, List, Dict, Any
import tkinter as tk
from tkinter import messagebox
import copy
import sys
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import neural network layers
from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.msg_pcn_v11 import PlaceCellLayer
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer_v4 import RewardCellLayerV4
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

# Torch and NumPy settings
np.set_printoptions(precision=2)


class Driver(Supervisor):
    """
    Main driver class for multiscale spatial navigation robot.

    Orchestrates neural networks (place cells, grid cells, reward cells, etc.)
    for spatial learning and goal-directed navigation across multiple spatial scales.
    """

    def initialization(
        self,
        mode: RobotMode = RobotMode.PLOTTING,
        scales: Optional[List[Dict[str, Any]]] = None,
        robot_params: Optional[Dict[str, Any]] = None,
        simulation_params: Optional[Dict[str, Any]] = None,
        exploitation_params: Optional[Dict[str, Any]] = None,
        stats_collector: Optional[stats_collector] = None,
        trial_id: Optional[str] = None,
        world_name: Optional[str] = None,
        start_loc: Optional[List[float]] = None,
        goal_config: Optional[Dict[str, Any]] = None,
        trial_config: Optional[Dict[str, Any]] = None,
        clear_files: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize the robot driver with all necessary parameters and networks.

        Uses initialization guard to prevent double initialization in singleton pattern.

        Args:
            mode: Robot operating mode (learning, exploitation, etc.)
            scales: List of scale configuration dictionaries with all parameters
            robot_params: Physical robot parameters (speeds, dimensions, etc.)
            simulation_params: Simulation timing and environment parameters
            exploitation_params: Algorithm parameters for exploitation behavior
            stats_collector: Optional statistics collection instance
            trial_id: Unique identifier for this trial
            world_name: Name of the simulation world
            start_loc: Starting position [x, y] or None for random
            goal_config: Goal configuration (single or multi-goal)
            trial_config: Trial execution configuration
            clear_files: Whether to clear existing network files
            **kwargs: Additional parameters passed from controller
        """

        print(f"[DRIVER] Initializing Driver instance for mode: {mode}")

        # === SECTION 1: Store Parameters & Basic Setup ===
        self._store_basic_parameters(mode, scales, robot_params, simulation_params,
                                exploitation_params, stats_collector, trial_id, kwargs)
        self._setup_directories_and_device(world_name)
        
        # === SECTION 2: Initialize Hardware & Sensors ===
        self._initialize_sensors()
        
        # === SECTION 3: Setup Goals ===
        self._setup_goals_and_trials(goal_config, trial_config)
        self._setup_robot_position(start_loc, kwargs.get('randomize_start_loc', False))
        
        # === SECTION 4: Clear Files if Requested ===
        if clear_files:
            print(f"[DRIVER] Clearing files due to clear_files=True for mode {mode}")
            self.clear()
        
        # === SECTION 5: Initialize Networks (After goals are set up) ===
        self._initialize_grid_cell_networks()
        self._initialize_place_cell_networks()
        self._initialize_reward_cell_networks()
        self._initialize_head_direction_network()
        self._load_goal_specific_networks()  # Now active_goal_name is available
        
        # === SECTION 6: Initialize Logging ===
        self._initialize_logging_structures()
        
        # === SECTION 7: Final Setup ===
        self._setup_exploitation_parameters()
        self._perform_initial_step()

        print(f"[DRIVER] Driver initialization completed for mode: {mode}")

    # ================================================================================================
    # SECTION 1: PARAMETER STORAGE & BASIC SETUP
    # ================================================================================================
    
    def _store_basic_parameters(self, mode, scales, robot_params, simulation_params, exploitation_params, stats_collector, trial_id, kwargs):
        """
        Store all parameters as instance variables for easy access.
        UPDATED: Added run_time_hours storage and better parameter handling
        """
        # Core configuration
        self.robot_mode = mode
        self.scales = scales or []
        
        # Store parameter dictionaries
        self.robot_params = robot_params or {}
        self.simulation_params = simulation_params or {}  
        self.exploitation_params = exploitation_params or {}
        
        # Store kwargs for later access
        self.kwargs = kwargs
        
        # FIXED: Store run_time_hours directly
        self.run_time_hours = kwargs.get('run_time_hours', 2)
        print(f"[DEBUG] Received run_time_hours parameter: {self.run_time_hours}")

        # Calculate simulation run time (matching v15 logic)
        self.run_time_minutes = self.run_time_hours * 60

        # FIXED: Store learning parameters directly
        self.enable_ojas = kwargs.get('enable_ojas')
        self.enable_stdp = kwargs.get('enable_stdp')

        # Unpack commonly-used robot parameters
        self.max_speed = robot_params.get('max_speed', 16)
        self.wheel_radius = robot_params.get('wheel_radius', 0.031)
        self.axle_length = robot_params.get('axle_length', 0.271756)
        self.lidar_resolution = robot_params.get('lidar_resolution', 720)

        # Unpack simulation parameters
        self.timestep = simulation_params.get('timestep', 96)
        self.tau_w = simulation_params.get('tau_w', 10)

        # Calculate num_steps after timestep is available
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        
        # Unpack exploitation parameters
        self.reward_threshold = exploitation_params.get('reward_threshold', 0.1)
        self.multistep_threshold = exploitation_params.get('multistep_threshold', 0.2)
        self.obstacle_threshold = exploitation_params.get('obstacle_threshold', 0.5)
        self.multistep_steps = exploitation_params.get('multistep_steps', 3)
        
        # Store additional parameters
        self.stats_collector = stats_collector
        self.trial_id = trial_id
        print(f"[DEBUG] stats_collector initialized: {self.stats_collector is not None}, trial_id: {self.trial_id}")
        self.max_dist = kwargs.get('max_dist', 25)
        self.plot_bvc = kwargs.get('plot_bvc', False)
        self.td_learning = kwargs.get('td_learning', False)
        self.use_prox_mod = kwargs.get('use_prox_mod', False)
        self.action_mode = kwargs.get('action_mode', 'explore')
        
        # Initialize basic state
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float32
        self.step_count = 0
        self.done = False
        
        print(f"[DRIVER] Using device: {self.device}")
        print(f"[DRIVER] Learning settings - Ojas: {self.enable_ojas}, STDP: {self.enable_stdp}")

    def _setup_directories_and_device(self, world_name):
        """Setup file directories and determine world name."""
        # Determine world name
        if world_name is None:
            world_path = self.getWorldPath()
            world_name = os.path.splitext(os.path.basename(world_path))[0]
        self.world_name = world_name
        
        # Create directories
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

    def _setup_robot_position(self, start_loc, randomize_start_loc):
        """Set robot starting position (random or specified)."""
        self.robot = self.getFromDef("agent")
        self.start_loc = start_loc
        
        if randomize_start_loc:
            # Keep randomizing until we're not inside any goal radius (with margin)
            MAX_TRIES = 200
            SAFETY_MARGIN = 0.3  # meters added to each goal radius
            world_bounds = getattr(self, "world_bounds", (-8.0, 8.0, -8.0, 8.0))  # (xmin, xmax, zmin, zmax)

            for _ in range(MAX_TRIES):
                candidate = [random.uniform(world_bounds[0], world_bounds[1]), 
                            0.0, 
                            random.uniform(world_bounds[2], world_bounds[3])]

                ok = True
                if hasattr(self, "goals"):
                    for g in self.goals:
                        gx, gz = g["location"]
                        gr = g.get("radius", 0.7) + SAFETY_MARGIN
                        dx = candidate[0] - gx
                        dz = candidate[2] - gz
                        dist = (dx*dx + dz*dz) ** 0.5
                        if dist < gr:
                            ok = False
                            break

                if ok:
                    self.robot.getField("translation").setSFVec3f(candidate)
                    self.robot.resetPhysics()
                    break
        else:
            if self.start_loc is not None:
                self.robot.getField("translation").setSFVec3f([self.start_loc[0], 0, self.start_loc[1]])
                self.robot.resetPhysics()

    # ================================================================================================
    # SECTION 2: SENSOR INITIALIZATION
    # ================================================================================================
    
    def _initialize_sensors(self):
        """
        Initialize all robot sensors and basic state variables.
        No modification needed - sensor setup is hardware-specific.
        """
        # Initialize sensors
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)
        self.range_finder = self.getDevice("range-finder")
        self.range_finder.enable(self.timestep)
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)
        
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        # Bumpers
        self.collided = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.left_bumper = self.getDevice("bumper_left")
        self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice("bumper_right")
        self.right_bumper.enable(self.timestep)
        
        # Motors and position sensors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        # Initialize motor speeds
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed

    # ================================================================================================
    # SECTION 3: NETWORK INITIALIZATION (COMPLETE IMPLEMENTATIONS)
    # ================================================================================================
    
    def _initialize_grid_cell_networks(self):
        """Initialize grid cell networks for each scale."""
        self.gcns = []
        
        for scale_config in self.scales:
            scale_idx = scale_config["scale_index"]
            fname = f"gcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)
            
            try:
                # Try to load existing network
                with open(path, "rb") as f:
                    gcn = pickle.load(f)
                print(f"[DRIVER] Loaded existing GCN from {path}")
            except (FileNotFoundError, pickle.UnpicklingError):
                # Create new network
                print(f"[DRIVER] Initializing new GCN for scale {scale_idx}")
                
                num_grid_cells = scale_config.get("num_grid_cells", 0)
                if num_grid_cells == 0:
                    gcn = None
                else:
                    gcn = GridCellLayer(
                        num_cells=num_grid_cells,
                        size_range=(0.5, 0.5),
                        rotation_range=scale_config.get("rotation_range", (0, 90)),
                        spread_range=scale_config.get("spread_range", (1.2, 1.2)),
                        translation_factor=scale_config.get("translation_factor", 1.0),
                        frequency_divisor=scale_config.get("frequency_divisor", 1.0),
                        threshold=scale_config.get("grid_threshold", 0.7),
                        threshold_type='soft',
                        normalization='per-cell',
                        device=self.device.type,
                        dtype=self.dtype
                    )
            
            self.gcns.append(gcn)

    def _initialize_place_cell_networks(self):
        """Initialize place cell networks for each scale."""
        self.pcns = []
        
        for i, scale_config in enumerate(self.scales):
            scale_idx = scale_config["scale_index"]
            fname = f"pcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)
            
            # Get corresponding grid cell network
            gcn = self.gcns[i]
            num_grid_cells = scale_config.get("num_grid_cells", 0) if gcn else 0
            
            pcn = self._create_or_load_pcn(path, scale_config, num_grid_cells)
            self.pcns.append(pcn)

    def _create_or_load_pcn(self, path, scale_config, num_grid_cells):
        """Create or load a single place cell network."""
        try:
            with open(path, "rb") as f:
                pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing PCN from {path}")
            
            # Update parameters from scale_config
            self._update_pcn_parameters(pcn, scale_config)
            return pcn
            
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[DRIVER] Could not load PCN from {path}: {e}")
            return self._create_new_pcn(scale_config, num_grid_cells)

    def _create_new_pcn(self, scale_config, num_grid_cells):
        """Create a new place cell network using scale configuration."""
        print(f"[DRIVER] Initializing new PCN for scale {scale_config['scale_index']}")
        
        # Create BVC layer
        bvc = BoundaryVectorCellLayer(
            max_dist=self.max_dist,
            n_res=self.lidar_resolution,
            n_hd=scale_config.get('num_hd', 8),
            sigma_theta=scale_config.get('sigma_theta', 1.0),
            sigma_r=scale_config.get('sigma_r', 0.5),
            device=self.device,
        )
        
        # Create place cell network 
        pcn_config = scale_config.copy()
        pcn_config.update({
            'num_grid_cells': num_grid_cells,
            'timestep': self.timestep,
            'device': self.device,
            'dtype': self.dtype,
            'enable_ojas': self.enable_ojas if self.enable_ojas is not None else pcn_config.get('enable_ojas', False),
            'enable_stdp': self.enable_stdp if self.enable_stdp is not None else pcn_config.get('enable_stdp', False),
        })
        
        pcn = PlaceCellLayer(bvc_layer=bvc, **pcn_config)
        
        print(f"[DRIVER] Successfully created new PCN for scale {scale_config['name']}")
        return pcn

    def _update_pcn_parameters(self, pcn, scale_config):
        """Update existing PCN with new parameters from scale_config."""
        # Update core parameters
        pcn.grid_influence = scale_config.get('grid_influence', pcn.grid_influence)
        pcn.gamma_pp = scale_config.get('gamma_pp', pcn.gamma_pp)
        pcn.gamma_pb = scale_config.get('gamma_pb', pcn.gamma_pb)
        pcn.gamma_pg = scale_config.get('gamma_pg', pcn.gamma_pg)

        if self.enable_ojas is not None:
            pcn.enable_ojas = self.enable_ojas
        else:
            pcn.enable_ojas = scale_config.get('enable_ojas', getattr(pcn, 'enable_ojas', False))
            
        if self.enable_stdp is not None:
            pcn.enable_stdp = self.enable_stdp  
        else:
            pcn.enable_stdp = scale_config.get('enable_stdp', getattr(pcn, 'enable_stdp', False))
    
         # Update STDP parameters
        pcn.stdp_learning_rate = scale_config.get('stdp_lr', getattr(pcn, 'stdp_learning_rate', 0.05))
        pcn.tau_hd = scale_config.get('tau_hd', getattr(pcn, 'tau_hd', 0.5))
        
        # Update learning parameters
        pcn.enable_adaptive_stdp = scale_config.get('enable_adaptive_stdp', getattr(pcn, 'enable_adaptive_stdp', True))
        pcn.adaptive_initial_lr = scale_config.get('adaptive_initial_lr', getattr(pcn, 'adaptive_initial_lr', 0.10))
        pcn.adaptive_final_lr = scale_config.get('adaptive_final_lr', getattr(pcn, 'adaptive_final_lr', 0.03))
        pcn.adaptive_decay_rate = scale_config.get('adaptive_decay_rate', getattr(pcn, 'adaptive_decay_rate', 60))
        
        # Update the stored config for consistency
        pcn.config.update(scale_config)
        
        print(f"[DRIVER] Updated existing PCN parameters for scale {scale_config['scale_index']}")

    def _initialize_reward_cell_networks(self):
        """Initialize reward cell networks for each scale."""
        self.rcns = []
        
        for scale_config in self.scales:
            scale_idx = scale_config["scale_index"]
            fname = f"rcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)
            
            rcn = self._create_or_load_rcn(path, scale_config)
            self.rcns.append(rcn)

    def _create_or_load_rcn(self, path, scale_config):
        """Create or load a single reward cell network."""
        try:
            with open(path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing RCN from {path}")
            
            # Update replay parameters
            rcn.replay_timesteps = scale_config.get('replay_timesteps', 20)
            rcn.replay_decay_constant = scale_config.get('replay_decay_constant', 6)
            
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[DRIVER] Creating new RCN for {path}")
            rcn = RewardCellLayerV4(
                num_place_cells=scale_config["num_pc"],
                num_replay=3,
                learning_rate=scale_config.get('rcn_learning_rate', 0.1),
                replay_timesteps=scale_config.get('replay_timesteps', 20),
                replay_decay_constant=scale_config.get('replay_decay_constant', 6),
                device=self.device,
            )
        
        return rcn

    def _initialize_head_direction_network(self):
        """Initialize head direction network."""
        self.n_hd = self.scales[0].get('num_hd', 8) if self.scales else 8
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

    def _load_goal_specific_networks(self):
        """Load goal-specific RCNs for multi-goal exploitation modes."""
        if (self.robot_mode == RobotMode.EXPLOIT_LOCATIONS and 
            hasattr(self, 'active_goal_name')):
            self._load_goal_rcns(self.active_goal_name)

    def _load_goal_rcns(self, goal_name):
        """Load RCNs for specific goal."""
        print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")
        
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        if not os.path.exists(multi_goal_dir):
            raise FileNotFoundError(f"Multi-goal rewards directory not found: {multi_goal_dir}")
        
        for i, scale_config in enumerate(self.scales):
            scale_idx = scale_config["scale_index"]
            goal_rcn_path = os.path.join(
                multi_goal_dir, 
                f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
            )
            
            if os.path.exists(goal_rcn_path):
                with open(goal_rcn_path, "rb") as f:
                    self.rcns[i] = pickle.load(f)
                print(f"[DRIVER] Loaded goal RCN: scale {scale_idx} for goal {goal_name}")
            else:
                raise FileNotFoundError(f"Goal-specific RCN not found: {goal_rcn_path}")

    # ================================================================================================
    # SECTION 4: LOGGING INITIALIZATION
    # ================================================================================================
    
    def _initialize_logging_structures(self):
        """
        Initialize data logging structures for recording robot behavior.
        """
        # Use already calculated values from _store_basic_parameters
        # (run_time_minutes and num_steps are already calculated there)
        
        # Initialize logging arrays
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        self.hmap_prox = torch.zeros((self.num_steps,), device="cpu", dtype=torch.float32)
        self.hmap_scale_priority = torch.zeros((self.num_steps,), device="cpu", dtype=torch.float32)

        # Initialize multi-scale logging structures
        self.hmap_pcn_activities = []
        for scale_config in self.scales:
            n_pc = scale_config["num_pc"]
            self.hmap_pcn_activities.append(
                torch.zeros((self.num_steps, n_pc), device="cpu", dtype=torch.float32)
            )

        self.hmap_gcn_activities = []
        for scale_config in self.scales:
            n_gc = scale_config.get("num_grid_cells", 0)
            self.hmap_gcn_activities.append(
                torch.zeros((self.num_steps, n_gc), device="cpu", dtype=torch.float32)
            )

    # ================================================================================================
    # SECTION 5: GOAL AND TRIAL SETUP
    # ================================================================================================
    
    def _setup_goals_and_trials(self, goal_config, trial_config):
        """
        Setup goal and trial configurations using new unified goal system.
        Handles both single and multi-goal modes with proper initialization.
        """
        # Setup unified goal system
        self._setup_goals(goal_config)
        
        # Setup trial configuration
        self._setup_trials(trial_config)

    def _setup_goals(self, goal_config):
        """Convert goal_config to unified internal format"""
        if not goal_config:
            # Default single goal if none provided
            goal_config = {
                "type": "single",
                "location": [-3, 3],
                "radius": 0.7,
                "name": "default"
            }
        
        if goal_config["type"] == "single":
            # Convert single goal to unified format
            self.goals = [{
                "name": goal_config.get("name", "default"),
                "location": goal_config["location"], 
                "radius": goal_config.get("radius", 0.7),
                "visited": False,
                "active": True  # Single goals are always active
            }]
            # Set legacy attributes for backward compatibility
            self.goal_location = goal_config["location"]
            self.multi_goal_mode = False
            self.goal_r = {"explore": 1.0, "exploit": 1.0}
            
            print(f"[DRIVER] Single goal mode: {self.goals[0]['name']} at {self.goals[0]['location']}")
            
        else:  # multi
            self.goals = []
            for goal_def in goal_config["goals"]:
                goal = goal_def.copy()
                if "visited" not in goal:
                    goal["visited"] = False
                if "active" not in goal:
                    goal["active"] = False
                if "radius" not in goal:
                    goal["radius"] = 0.7  # Default radius
                self.goals.append(goal)
                    
            # Set target goal if specified (for EXPLOIT_LOCATIONS)
            if "target_goal" in goal_config:
                self._set_target_goal(goal_config["target_goal"])
                
            self.multi_goal_mode = True
            
            # Initialize goal tracking for learning modes
            if self.robot_mode == RobotMode.LEARN_LOCATIONS:
                self.goal_place_cell_associations = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_association_step = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
            
            print(f"[DRIVER] Multi-goal mode with {len(self.goals)} goals:")
            for goal in self.goals:
                status = "ACTIVE" if goal.get("active", False) else "inactive"
                print(f"  - {goal['name']}: {goal['location']} (radius: {goal['radius']}) [{status}]")

    def _setup_trials(self, trial_config):
        """Store trial configuration for reference"""
        if not trial_config:
            trial_config = {"type": "simple", "count": 1, "start_locations": [[0, 0]]}
        
        self.trial_config = trial_config
        print(f"[DRIVER] Trial config: {trial_config['type']} - {trial_config['count']} trials")

    def _set_target_goal(self, goal_name):
        """Set which goal is currently targeted for exploitation"""
        found = False
        for goal in self.goals:
            goal["active"] = (goal["name"] == goal_name)
            if goal["name"] == goal_name:
                found = True
        
        if not found:
            raise ValueError(f"Target goal '{goal_name}' not found in goals list")
        
        self.active_goal_name = goal_name
        print(f"[DRIVER] Set target goal: {goal_name}")

    # ================================================================================================
    # SECTION 6: EXPLOITATION SETUP
    # ================================================================================================
    
    def _setup_exploitation_parameters(self):
        """
        Setup exploitation algorithm parameters from exploitation_params dictionary.
        Extracts and stores commonly-used exploitation parameters as instance variables.
        """
        # Momentum parameters
        self.MOMENTUM_TYPE = self.exploitation_params.get('momentum_type', 'bonus')
        self.MOMENTUM_STRENGTH = self.exploitation_params.get('momentum_strength', 0.3)
        self.MOMENTUM_STEPS = self.exploitation_params.get('momentum_steps', 15)
        self.CHANGE_THRESHOLD = self.exploitation_params.get('change_threshold', 0.5)
        
        # Multi-step parameters
        self.MULTISTEP_THRESHOLD = self.exploitation_params.get('multistep_threshold', 0.2)
        self.MULTISTEP_MAX_SCALES = self.exploitation_params.get('multistep_max_scales', 2)
        self.ENHANCEMENT_BONUS = self.exploitation_params.get('enhancement_bonus', 1.2)
        self.MULTISTEP_STEPS = self.exploitation_params.get('multistep_steps', 3)
        self.MULTISTEP_DECAY_FACTOR = self.exploitation_params.get('multistep_decay_factor', 0.6)
        
        # Detection and recovery parameters
        self.REWARD_THRESHOLD = self.exploitation_params.get('reward_threshold', 0.1)
        self.OBSTACLE_DISTANCE_THRESHOLD = self.exploitation_params.get('obstacle_threshold', 0.5)
        self.LOOP_THRESHOLD = self.exploitation_params.get('loop_threshold', 5)
        self.MAX_STEPS_BETWEEN_LOOPS = self.exploitation_params.get('max_steps_between_loops', 10)
        self.FORCE_EXPLORE_DURATION = self.exploitation_params.get('force_explore_duration', 5)
        
        # Goal cell boosting parameters
        self.GOAL_BOOST_FACTORS = {
            0: self.exploitation_params.get('goal_boost_small', 3.0),    # small scale
            1: self.exploitation_params.get('goal_boost_medium', 2.5),   # medium scale  
            2: self.exploitation_params.get('goal_boost_large', 2.0),    # large scale
            3: self.exploitation_params.get('goal_boost_xlarge', 1.5),   # xlarge scale
        }
        
        # Scale-specific multistep parameters
        self.SCALE_MULTISTEP_STEPS = {}
        for scale_config in self.scales:
            scale_name = scale_config.get('name', 'unknown')
            self.SCALE_MULTISTEP_STEPS[scale_name] = scale_config.get('multistep_steps', self.MULTISTEP_STEPS)
        
        # Debug parameters
        self.DEBUG_EXPLOIT_V0 = self.exploitation_params.get('debug_exploit_v0', True)
        self.DEBUG_EXPLOIT_V1 = self.exploitation_params.get('debug_exploit_v1', False)
        
        # Initialize tracking variables
        self._direction_history = []
        self._last_chosen_direction = None
        self._last_direction_reward = 0.0
        
        # Initialize loop detection variables
        self.rotation_accumulator = 0.0
        self.rotation_loop_count = 0
        self.steps_since_last_loop = 0
        self.last_heading_deg = None
        self.force_explore_count = 0
        
        print(f"[DRIVER] Exploitation parameters configured:")
        print(f"  - Momentum: {self.MOMENTUM_TYPE} (strength: {self.MOMENTUM_STRENGTH})")
        print(f"  - Multi-step: {self.MULTISTEP_STEPS} steps (threshold: {self.MULTISTEP_THRESHOLD})")
        print(f"  - Reward threshold: {self.REWARD_THRESHOLD}")

    # ================================================================================================
    # SECTION 7: FINAL SETUP
    # ================================================================================================
    
    def _perform_initial_step(self):
        """
        Perform initial simulation step and setup trial timing.
        Records trial start time for per-trial timeout handling.
        """
        # Step once
        self.step(self.timestep)
        
        # Record trial start time for timeout handling
        self.trial_start_time = self.getTime()
        print(f"[DRIVER] Trial started at simulation time: {self.trial_start_time:.1f}s")
        print(f"[DEBUG] Trial time limit: {60 * self.run_time_minutes:.1f}s ({self.run_time_minutes:.1f} minutes)")
        print(f"[DEBUG] run_time_hours={self.run_time_hours}, run_time_minutes={self.run_time_minutes}")

    # ================================================================================================
    # SECTION 8: RUN LOOP & ACTION MODES
    # ================================================================================================
    
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
            if self.action_mode == 'explore':
                self.explore()
            elif self.action_mode == 'exploit_v0':
                self.exploit_v0()
            elif self.action_mode == 'exploit_v1':
                self.exploit_v1()
            elif self.action_mode == 'exploit_v15':
                self.exploit_v15()
            elif self.action_mode == 'exploit_v2':
                self.exploit_v2()
            elif self.action_mode == 'exploit_timed_fallback':  # NEW ACTION MODE
                self.exploit_with_timed_fallback()
            elif self.action_mode == 'exploit_locations':
                self.exploit_locations()
            else:
                print("Unknown action mode. Exiting...")
                break

        print(f"[DEBUG] Trial loop ended, done={self.done}")

    def reset_for_next_trial(self, new_start_loc, new_trial_id, target_goal=None, stats_collector=None):
        """
        Soft reset for next trial without reloading world (v15-style approach).
        Manually repositions robot and resets trial state variables.
        """
        print(f"[DEBUG] Soft reset: Moving robot from current position to {new_start_loc}")

        # 1. Reset robot position
        self.robot.getField("translation").setSFVec3f([new_start_loc[0], 0, new_start_loc[1]])
        self.robot.resetPhysics()

        # 2. Reset trial state variables
        self.done = False
        self.step_count = 0
        self.trial_start_time = self.getTime()  # New trial start time relative to current simulation
        self.trial_id = new_trial_id
        self.start_loc = new_start_loc

        # 3. Update stats collector for new trial
        if stats_collector:
            self.stats_collector = stats_collector
            print(f"[DEBUG] Soft reset: Updated stats_collector for trial {new_trial_id}")

        # 4. Update goal configuration for new trial
        if target_goal and self.multi_goal_mode:
            # Deactivate all goals
            for goal in self.goals:
                goal["active"] = False

            # Activate target goal
            target_goal_obj = next((g for g in self.goals if g["name"] == target_goal), None)
            if target_goal_obj:
                target_goal_obj["active"] = True
                self.active_goal_name = target_goal
                print(f"[DEBUG] Soft reset: Activated goal '{target_goal}' at {target_goal_obj['location']}")
            else:
                print(f"[ERROR] Soft reset: Target goal '{target_goal}' not found!")

        # 5. Reset path logging for new trial
        self.hmap_loc[:] = 0  # Clear location history

        print(f"[DEBUG] Soft reset completed: trial_id={self.trial_id}, start_loc={self.start_loc}, trial_start_time={self.trial_start_time:.1f}s")

    def explore(self) -> None:
        """
        Handles exploration for multi-scale usage, calling compute_pcn_activations().
        The robot moves forward unless it collides, in which case it turns away.
        """
        for _ in range(self.tau_w):
            if self.done:
                break
            
            # 1) Sense environment
            self.sense()

            # 2) Compute place cell activations
            self.compute_pcn_activations()

            # 3) If in DMTP, EXPLOIT, or EXPLOIT_LOCATIONS mode, update reward cell activations
            if self.robot_mode in {RobotMode.DMTP, RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS}:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)

            # 4) If a collision is detected, turn away and break
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break
            
            # 5) Check goal using unified system, update hmaps, move forward
            self.check_goal_reached_unified()
            self.update_hmaps(
                update_loc=True, 
                update_pcn=True,
                update_gcn=True,
                update_scale_priority=self.robot_mode == RobotMode.EXPLOIT, 
                update_prox=(self.use_prox_mod and self.robot_mode == RobotMode.LEARNING)
            )
            self.forward()

        # A small random turn at the end if enabled
        self.turn(np.random.normal(0, np.deg2rad(30)))

    def exploit_v1(self):
        """
        Enhanced multiscale exploitation with multi-step preplay on promising scales.
        Combines proven multiscale integration with sophisticated multi-step planning.
        Updated to use parameters from exploitation_params dictionary.
        """
        if self.DEBUG_EXPLOIT_V1:
            if not hasattr(self, '_exploit_v1_call_count'):
                self._exploit_v1_call_count = 0
            self._exploit_v1_call_count += 1
            
            show_detailed_debug = self._exploit_v1_call_count % 5 == 0
            if show_detailed_debug:
                print(f"\n{'='*60}")
                print(f"EXPLOIT V1 - ENHANCED MULTISCALE - STEP {self._exploit_v1_call_count}")
                print("="*60)

        # 1) Basic sensing and setup
        self._sense_and_compute()

        if self.step_count <= self.tau_w:
            return

        # 2) Compute obstacle distances
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        self.distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        # Phase 1: Single-step evaluation on ALL scales
        valid_scale_indices = []
        single_step_rewards = []

        for i, scale_def in enumerate(self.scales):
            pot_rew = self.compute_single_step_rewards(i)
            
            if pot_rew.max().item() >= self.REWARD_THRESHOLD:
                valid_scale_indices.append(i)
                single_step_rewards.append(pot_rew)
                if self.DEBUG_EXPLOIT_V1 and show_detailed_debug:
                    print(f"Scale {i} ({scale_def['name']}): max_reward={pot_rew.max().item():.3f} ✓")
            else:
                if self.DEBUG_EXPLOIT_V1:
                    print(f"Scale {i} ({scale_def['name']}): max_reward={pot_rew.max().item():.3f} (below threshold)")

        if len(valid_scale_indices) == 0:
            if self.DEBUG_EXPLOIT_V1:
                print("No valid scales found, falling back to exploration")
            self.explore()
            return

        # Phase 2: Scale ranking and selection
        mixing_weights = self.compute_scale_gradients(single_step_rewards)
        
        # Select promising scales for multi-step enhancement
        promising_scale_indices = []
        for k, scale_idx in enumerate(valid_scale_indices):
            if mixing_weights[k].item() >= self.MULTISTEP_THRESHOLD:
                promising_scale_indices.append(scale_idx)

        # Sort by weight and take top N
        promising_scale_indices = sorted(
            promising_scale_indices, 
            key=lambda scale_idx: mixing_weights[valid_scale_indices.index(scale_idx)], 
            reverse=True
        )[:self.MULTISTEP_MAX_SCALES]

        if self.DEBUG_EXPLOIT_V1:
            print(f"Valid scales: {len(valid_scale_indices)}, Enhanced scales: {promising_scale_indices}")

        # Phase 3: Multi-step enhancement on selected scales
        enhanced_rewards = {}
        
        for scale_idx in promising_scale_indices:
            if self.DEBUG_EXPLOIT_V1 and show_detailed_debug:
                print(f"  Enhancing scale {scale_idx} ({self.scales[scale_idx]['name']}) with {self.MULTISTEP_STEPS}-step preplay")
            
            pcn, rcn = self.pcns[scale_idx], self.rcns[scale_idx]
            
            # Setup evaluation function for this scale
            def evaluate_with_rcn(activations):
                rcn.update_reward_cell_activations(activations, visit=False)
                return torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
            
            # Store original evaluation method and set new one
            original_eval_method = getattr(pcn, '_evaluate_activations_for_reward', None)
            pcn._evaluate_activations_for_reward = evaluate_with_rcn
            
            try:
                # Multi-step evaluation in ALL 8 directions
                multi_step_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
                
                for direction in range(self.n_hd):
                    if self.distances_per_hd[direction] < self.OBSTACLE_DISTANCE_THRESHOLD:
                        multi_step_rewards[direction] = 0.0
                    else:
                        best_reward = pcn.multi_step_preplay(
                            forced_first_direction=direction,
                            max_steps=self.MULTISTEP_STEPS,
                            decay_factor=self.MULTISTEP_DECAY_FACTOR
                        )
                        multi_step_rewards[direction] = best_reward
                
                enhanced_rewards[scale_idx] = multi_step_rewards
                
                # Compare improvement
                if self.DEBUG_EXPLOIT_V1:
                    original_max = single_step_rewards[valid_scale_indices.index(scale_idx)].max().item()
                    enhanced_max = multi_step_rewards.max().item()
                    improvement = 100 * (enhanced_max / (original_max + 1e-6) - 1)
                    print(f"  Scale {scale_idx}: {original_max:.3f} → {enhanced_max:.3f} ({improvement:+.1f}% improvement)")
                    
            except Exception as e:
                if self.DEBUG_EXPLOIT_V1:
                    print(f"  Multi-step failed for scale {scale_idx}: {e}")
                # Fall back to single-step for this scale (don't add to enhanced_rewards)
                
            finally:
                # Restore original evaluation method
                if original_eval_method is not None:
                    pcn._evaluate_activations_for_reward = original_eval_method
                elif hasattr(pcn, '_evaluate_activations_for_reward'):
                    delattr(pcn, '_evaluate_activations_for_reward')

        # Phase 4: Reward replacement and final combination
        # Replace single-step rewards with multi-step for enhanced scales
        final_scale_rewards = single_step_rewards.copy()
        enhanced_mixing_weights = mixing_weights.clone()
        
        for scale_idx in enhanced_rewards.keys():
            scale_list_idx = valid_scale_indices.index(scale_idx)
            final_scale_rewards[scale_list_idx] = enhanced_rewards[scale_idx]
            
            # Optional: Boost mixing weight for enhanced scales
            enhanced_mixing_weights[scale_list_idx] *= self.ENHANCEMENT_BONUS

        # Renormalize mixing weights
        enhanced_mixing_weights = enhanced_mixing_weights / enhanced_mixing_weights.sum()

        # Standard weighted combination
        combined_pot_rew = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        for direction in range(self.n_hd):
            combined_pot_rew[direction] = sum(
                enhanced_mixing_weights[k] * final_scale_rewards[k][direction] 
                for k in range(len(final_scale_rewards))
            )

        # 5) Action selection and execution

        # Apply momentum to combined rewards
        momentum_adjusted_rewards = self.apply_directional_momentum(combined_pot_rew)

        # Compute action heading using circular mean
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * momentum_adjusted_rewards)
        cos_component = torch.sum(torch.cos(angles) * momentum_adjusted_rewards)
        action_angle = torch.atan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        # Determine chosen direction for momentum tracking
        chosen_direction = torch.argmax(momentum_adjusted_rewards).item()
        chosen_direction_reward = momentum_adjusted_rewards[chosen_direction].item()

        if self.DEBUG_EXPLOIT_V1:
            best_direction = torch.argmax(combined_pot_rew).item()
            best_reward = combined_pot_rew[best_direction].item()
            print(f"Final decision: target={self.action_heading_deg:.0f}°, best_dir={best_direction} ({best_direction*45}°), reward={best_reward:.4f}")
        
        # Update momentum tracking
        self.update_momentum_tracking(chosen_direction, chosen_direction_reward)

        # Execute movement with compass-based turning
        self._execute_movement(self.action_heading_deg, self.DEBUG_EXPLOIT_V1)

        if self.DEBUG_EXPLOIT_V1 and show_detailed_debug:
            print("="*60 + "\n")

        return

    def exploit_v2(self):
        """
        Enhanced multiscale exploitation with goal cell boosting for EXPLOIT_LOCATIONS mode.
        
        Key improvements over v15:
        - Detects when multi-step preplay activates goal-associated place cells
        - Applies scale-dependent boost to directions that activate goal cells
        - Smaller scales (more accurate) receive larger boosts
        - Maintains all existing gradient-based multi-scale integration
        Updated to use parameters from exploitation_params dictionary.
        """
        # Debug setup
        show_debug = self.DEBUG_EXPLOIT_V1
        show_detailed_debug = False
        
        if show_debug:
            if not hasattr(self, '_exploit_v2_call_count'):
                self._exploit_v2_call_count = 0
            self._exploit_v2_call_count += 1
            
            show_detailed_debug = self._exploit_v2_call_count % 5 == 0
            if show_detailed_debug:
                print(f"\n{'='*60}")
                print(f"EXPLOIT V2 - GOAL-BOOSTED MULTI-STEP - STEP {self._exploit_v2_call_count}")
                print("="*60)

        # 1) Basic sensing and setup
        self._sense_and_compute()

        if self.step_count <= self.tau_w:
            return

        # 2) Get active goal information for goal cell boosting
        active_goal = None
        if self.multi_goal_mode:
            active_goals = [g for g in self.goals if g.get("active", False)]
            if active_goals:
                active_goal = active_goals[0]
        
        if show_detailed_debug and active_goal:
            print(f"Active goal for boosting: {active_goal['name']} at {active_goal['location']}")

        # 3) Compute obstacle distances
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        self.distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        if show_detailed_debug:
            print("Obstacle distances per direction:")
            for d in range(self.n_hd):
                angle_deg = d * 45
                print(f"  Direction {d} ({angle_deg}°): {self.distances_per_hd[d]:.2f}m")

        # 4) Multi-step preplay on ALL scales with goal cell boosting
        all_scale_rewards = []
        all_scale_gradients = []

        for scale_idx, scale_def in enumerate(self.scales):
            try:
                if show_detailed_debug:
                    print(f"Computing goal-boosted multi-step rewards for scale {scale_idx} ({scale_def['name']})")
                
                # Get scale-specific multistep value
                scale_name = scale_def.get('name', 'unknown')
                scale_max_steps = self.SCALE_MULTISTEP_STEPS.get(scale_name, self.MULTISTEP_STEPS)
                
                if show_detailed_debug:
                    print(f"  Using {scale_max_steps} preplay steps for scale {scale_name}")
                
                # Use goal-boosted computation method with scale-specific steps
                multi_step_rewards = self.compute_multi_step_rewards_with_goal_boost(
                    scale_idx, active_goal, show_detailed_debug, max_steps=scale_max_steps
                )
                
                # Apply threshold to filter noise
                thresholded_rewards = torch.where(
                    multi_step_rewards >= self.MULTISTEP_THRESHOLD,
                    multi_step_rewards,
                    torch.zeros_like(multi_step_rewards)
                )
                
                # Normalize rewards for this scale
                max_reward = thresholded_rewards.max().item()
                if max_reward > 1e-6:
                    normalized_rewards = thresholded_rewards / max_reward
                else:
                    normalized_rewards = thresholded_rewards
                
                # Compute gradient
                gradient = torch.sum(torch.abs(torch.diff(normalized_rewards))).item()
                
                all_scale_rewards.append(thresholded_rewards)
                all_scale_gradients.append(gradient)
                
                if show_detailed_debug:
                    print(f"  Scale {scale_idx}: max_reward={max_reward:.3f}, gradient={gradient:.3f}, steps={scale_max_steps}")
                
            except Exception as e:
                if show_debug:
                    print(f"Multi-step failed for scale {scale_idx}: {e}")
                
                zero_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
                all_scale_rewards.append(zero_rewards)
                all_scale_gradients.append(0.0)

        # 5) Handle all-zero gradients case
        total_gradient = sum(all_scale_gradients)
        if total_gradient < 1e-6:
            if show_debug:
                print("All scales have zero gradients - falling back to exploration")
            self.explore()
            return
        
        # 6) Relative gradient weighting
        mixing_weights = torch.tensor(all_scale_gradients, dtype=self.dtype, device=self.device) / total_gradient
        
        if show_debug:
            print(f"Valid scales: {len(all_scale_rewards)}")
            print("Scale mixing weights (gradient-based):")
            for i, (scale_def, weight, gradient) in enumerate(zip(self.scales, mixing_weights, all_scale_gradients)):
                print(f"  Scale {i} ({scale_def['name']}): weight={weight:.3f}, gradient={gradient:.3f}")

        # 7) Combine rewards across scales
        combined_pot_rew = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        for direction in range(self.n_hd):
            combined_pot_rew[direction] = sum(
                mixing_weights[k] * all_scale_rewards[k][direction] 
                for k in range(len(all_scale_rewards))
            )

        # 8) Apply momentum to combined rewards
        momentum_adjusted_rewards = self.apply_directional_momentum(combined_pot_rew)

        # 9) Compute action heading using circular mean
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * momentum_adjusted_rewards)
        cos_component = torch.sum(torch.cos(angles) * momentum_adjusted_rewards)
        action_angle = torch.atan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        # 10) Determine chosen direction for momentum tracking
        chosen_direction = torch.argmax(momentum_adjusted_rewards).item()
        chosen_direction_reward = momentum_adjusted_rewards[chosen_direction].item()

        if show_debug:
            best_direction = torch.argmax(combined_pot_rew).item()
            best_reward = combined_pot_rew[best_direction].item()
            print(f"Final decision: target={self.action_heading_deg:.0f}°, best_dir={best_direction} ({best_direction*45}°), reward={best_reward:.4f}")
        
        # 11) Update momentum tracking
        self.update_momentum_tracking(chosen_direction, chosen_direction_reward)

        # 12) Execute movement
        self._execute_movement(self.action_heading_deg, show_debug)

        if show_detailed_debug:
            print("="*60 + "\n")

        return

    def exploit_locations(self):
        """Goal-specific exploitation using existing exploit_v1 logic"""
        if not hasattr(self, 'active_goal_name'):
            print("[ERROR] No active goal set for EXPLOIT_LOCATIONS mode")
            self.done = True
            return
        
        # Use existing exploit_v1 logic - RCNs are already loaded for the target goal
        self.exploit_v2()

    def exploit_with_timed_fallback(self):
        """
        Action mode that starts with exploit_v2 but falls back to exploration after 15 minutes.
        Useful for testing scenarios where exploitation might get stuck.
        """
        # Calculate elapsed time since trial start
        trial_elapsed_time = self.getTime() - self.trial_start_time
        fallback_time = 15 * 60  # 15 minutes in seconds
        
        # Initialize fallback flag if not exists
        if not hasattr(self, '_has_fallen_back_to_explore'):
            self._has_fallen_back_to_explore = False
            print(f"[TIMED_FALLBACK] Starting exploit_v2 with {fallback_time/60:.1f} minute fallback timer")
        
        # Check if we should switch to exploration
        if trial_elapsed_time >= fallback_time and not self._has_fallen_back_to_explore:
            self._has_fallen_back_to_explore = True
            print(f"[TIMED_FALLBACK] {fallback_time/60:.1f} minutes elapsed ({trial_elapsed_time:.1f}s), switching to exploration mode")
        
        # Execute appropriate action based on current state
        if self._has_fallen_back_to_explore:
            if not hasattr(self, '_fallback_start_time'):
                self._fallback_start_time = self.getTime()
                print(f"[TIMED_FALLBACK] Exploration fallback started at {self._fallback_start_time:.1f}s")
            self.explore()
        else:
            # Show periodic time updates
            if not hasattr(self, '_last_time_update'):
                self._last_time_update = 0
            
            # Print time remaining every 2 minutes
            if trial_elapsed_time - self._last_time_update >= 120:  # 2 minutes
                time_remaining = (fallback_time - trial_elapsed_time) / 60
                print(f"[TIMED_FALLBACK] Exploiting... {time_remaining:.1f} minutes until exploration fallback")
                self._last_time_update = trial_elapsed_time
            
            self.exploit_v1()

    # ================================================================================================
    # SECTION 9: SENSING & PERCEPTION
    # ================================================================================================
    
    def sense(self):
        """
        Uses sensors to update range-image, heading, boundary data, collision flags, etc.
        """
        # Get the latest boundary data from range finder
        boundaries = self.range_finder.getRangeImage()

        # Update global heading (0–360)
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))

        # Shift boundary data based on global heading
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * self.current_heading_deg
        )

        # Compute prox
        if self.use_prox_mod:
            self.prox = self.compute_proximity(boundaries)

        # Convert heading to radians for HD-layer input
        current_heading_rad = np.deg2rad(self.current_heading_deg)
        v_in = torch.tensor([np.cos(current_heading_rad), np.sin(current_heading_rad)],
                            dtype=self.dtype, device=self.device)

        # Update head direction layer activations
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)

        # Check for collisions via bumpers
        self.collided[0] = int(self.left_bumper.getValue())
        self.collided[1] = int(self.right_bumper.getValue())

        if torch.any(self.collided):
            if self.stats_collector:
                self.stats_collector.update_stat("collision_count", self.stats_collector.stats["collision_count"] + 1)

        # Advance simulation one timestep
        self.step(self.timestep)

    def _sense_and_compute(self):
        """Shared helper: Sense environment, compute PCN activations, update hmaps, check goal."""
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True, update_scale_priority=True)
        self.check_goal_reached_unified()

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

    def compute_proximity(self, boundaries):
        """
        Computes the visual density based on distances to walls,
        applying an exponential decay to each LiDAR reading.

        Args:
            boundaries (list): LiDAR readings indicating distances to obstacles.

        Returns:
            float: The computed visual density, emphasizing proximity to walls.
        """
        # Threshold distance for influence (e.g., max effective wall influence)
        max_influence_radius = 3  # Adjust based on environment size

        # Convert LiDAR data to a PyTorch tensor
        lidar_tensor = torch.tensor(boundaries, dtype=self.dtype, device=self.device)

        # Apply exponential decay to each LiDAR reading for wall density
        wall_densities = torch.exp(-lidar_tensor / max_influence_radius)
        wall_density = torch.mean(wall_densities)

        # Clamp between 0 and 1
        proximity = torch.clamp(wall_density, 0, 1)

        return proximity

    # ================================================================================================
    # SECTION 10: NEURAL COMPUTATION
    # ================================================================================================
    
    def compute_pcn_activations(self):
        """
        Updates place cell activations for each scale based on sensor data.
        Populates self.pcn_activations_list, which we'll use in update_hmaps().
        """

        # Get current position [x, z] from robot (Webots uses [x, y, z], y is vertical)
        curr_pos = self.robot.getField("translation").getSFVec3f()
        position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

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
                    distances=self.boundaries.cpu().numpy(),
                    angles=np.linspace(0, 2 * np.pi, 720),
                )
                activations = self.pcn.place_cell_activations.cpu().detach().numpy()
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(activations)), activations)
                plt.xlabel("Place Cell Index")
                plt.ylabel("Activation")
                plt.title("Current Place Cell Activations")
                plt.show()
            # Append activations to pcn_activations_list
            act = pcn.place_cell_activations.clone().detach()
            self.pcn_activations_list.append(act)

    def refresh_pcn_activations(self, reason=""):
        """
        Force fresh computation of place cell activations for all scales.
        This clears any potential state corruption and ensures clean activations.
        
        Args:
            reason (str): Description of why refresh is being called (for logging)
        """
        if reason:
            print(f"[DRIVER] Refreshing PCN activations: {reason}")
        else:
            print(f"[DRIVER] Refreshing PCN activations")
        
        # Re-sense the environment to get fresh sensor data
        self.sense()
        
        # Force fresh computation of all place cell activations
        self.compute_pcn_activations()
        
        # Log the results to verify we have activity
        print(f"[DRIVER] PCN activations after refresh:")
        for i, pcn in enumerate(self.pcns):
            activity_sum = torch.sum(pcn.place_cell_activations).item()
            print(f"  Scale {i}: activity_sum = {activity_sum:.6f}")

    def compute_goal_cell_boost_factor(self, scale_idx, goal_cell_activation):
        """
        Compute boost factor based on scale accuracy and goal cell activation strength.
        Updated to use goal boost factors from exploitation_params.
        """
        # Get base boost for this scale from exploitation parameters
        base_boost = self.GOAL_BOOST_FACTORS.get(scale_idx, 1.5)
        
        # Scale boost by activation strength
        activation_multiplier = 1.0 + (base_boost - 1.0) * goal_cell_activation
        
        return activation_multiplier
  
    # ================================================================================================
    # SECTION 11: GOAL CHECKING & STATE MANAGEMENT
    # ================================================================================================
    
    def check_goal_reached_unified(self):
        """Unified goal checking for all modes using new goal system."""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

        # Calculate elapsed time for this trial
        trial_elapsed_time = self.getTime() - self.trial_start_time

        # Handle mode-specific time limits and general completion
        if self.robot_mode in (RobotMode.LEARNING, RobotMode.PLOTTING) \
                and trial_elapsed_time >= 60 * self.run_time_minutes:  
            print(f"[INFO] Trial time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
            self.stop()
            self.save(include_pcn=self.robot_mode != RobotMode.PLOTTING,
                    include_rcn=self.robot_mode != RobotMode.PLOTTING,
                    include_gcn=self.robot_mode != RobotMode.PLOTTING,
                    include_hmaps=True)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return

        # Multi-goal mode handling
        if self.multi_goal_mode:
            if self.robot_mode == RobotMode.LEARN_LOCATIONS:
                # Check all goals for learning
                for goal in self.goals:
                    goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                    distance = torch.norm(current_position - goal_position)
                    
                    if distance <= goal["radius"]:
                        if not goal["visited"]:
                            print(f"[LEARN_LOCATIONS] First visit to {goal['name']} goal at {goal['location']}")
                            goal["visited"] = True
                        self._handle_goal_learning(goal)
                
                # Check if learning is complete
                minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
                if minimum_time_reached:
                    if self._check_multi_goal_learning_complete():
                        print(f"[LEARN_LOCATIONS] Learning complete! Trial time: {trial_elapsed_time:.1f}s")
                        self._create_multi_goal_reward_maps()
                        self._save_multi_goal_data()
                        self.stop()
                        self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                        self.done = True
                        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                    else:
                        print(f"[LEARN_LOCATIONS] Minimum time reached but learning incomplete, continuing...")
                        
            elif self.robot_mode == RobotMode.EXPLOIT_LOCATIONS:
                # Check only active goal for exploitation
                active_goals = [g for g in self.goals if g.get("active", False)]
                print(f"[DEBUG] EXPLOIT_LOCATIONS: {len(active_goals)} active goals, trial_elapsed: {trial_elapsed_time:.1f}s")

                for goal in active_goals:
                    goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                    distance = torch.norm(current_position - goal_position)

                    print(f"[DEBUG] Goal '{goal.get('name', 'unknown')}' distance: {distance:.2f}m, radius: {goal['radius']:.2f}m")

                    if distance <= goal["radius"]:
                        print(f"[DEBUG] GOAL REACHED! Calling _handle_goal_exploitation()")
                        self._handle_goal_exploitation(goal)
                        return

                # Check time limit for exploitation - Use trial-relative time
                time_limit_seconds = 60 * self.run_time_minutes
                print(f"[DEBUG] Time check: elapsed={trial_elapsed_time:.1f}s, limit={time_limit_seconds:.1f}s, step_count={self.step_count}")

                if self.step_count > 10 and trial_elapsed_time >= time_limit_seconds:
                    print(f"[DEBUG] TIME LIMIT REACHED! Calling _handle_exploitation_timeout()")
                    print(f"Trial time limit reached: {trial_elapsed_time:.1f}s / {time_limit_seconds:.1f}s")
                    self._handle_exploitation_timeout()
                    return
                    
        else:
            # Single goal mode handling
            goal = self.goals[0]
            goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
            distance = torch.norm(current_position - goal_position)
            
            if distance <= goal["radius"]:
                if self.robot_mode == RobotMode.DMTP:
                    self._handle_dmtp_goal_reached()
                elif self.robot_mode == RobotMode.EXPLOIT:
                    self._handle_single_goal_exploitation()
                # Other single goal modes handled by existing logic
                return
                
            # Check time limit for single goal exploitation - Use trial-relative time
            if self.robot_mode == RobotMode.EXPLOIT and self.step_count > 10:
                if trial_elapsed_time >= 60 * self.run_time_minutes:
                    print(f"Trial time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
                    self._handle_exploitation_timeout()
                    return

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to active goals.
        Updated to work with new unified goal system.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)
        
        if self.multi_goal_mode:
            # For multi-goal, check active goals
            active_goals = [g for g in self.goals if g.get("active", False)]
            for goal in active_goals:
                goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                distance = torch.norm(current_position - goal_position)
                if distance <= goal["radius"]:
                    return 1.0
            return 0.0
        else:
            # Single goal mode
            goal = self.goals[0]
            goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
            distance = torch.norm(current_position - goal_position)
            
            if distance <= goal["radius"]:
                return 1.0
            else:
                return 0.0

    # ================================================================================================
    # SECTION 12: MOVEMENT & NAVIGATION
    # ================================================================================================
    
    def forward(self):
        """Moves the robot forward at maximum speed.

        Sets both wheels to max speed, updates motor movement and sensor readings.
        """
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()

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
        if self.stats_collector:
            self.stats_collector.update_stat("turn_count", self.stats_collector.stats["turn_count"] + 1)
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

    def auto_pilot(self):
        """
        Automatic navigation to goal location using new unified goal system.
        Updated to work with both single and multi-goal modes.
        """
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()

        # Determine target goal location
        if self.multi_goal_mode:
            # Find active goal
            active_goals = [g for g in self.goals if g.get("active", False)]
            if not active_goals:
                print("[AUTO_PILOT] No active goal found in multi-goal mode")
                return
            target_location = active_goals[0]["location"]
            target_radius = active_goals[0]["radius"]
        else:
            # Single goal mode
            target_location = self.goal_location
            target_radius = self.goals[0]["radius"]

        # Keep moving until close enough to goal
        while not torch.allclose(
            torch.tensor(target_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=target_radius
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - target_location[0]
            delta_y = curr_pos[2] - target_location[1]

            # Compute desired heading to face the goal
            if delta_x >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))).item()
                if delta_y >= 0:
                    desired = 2 * np.pi - theta
                else:
                    desired = np.pi + theta
            elif delta_y >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))).item()
                desired = (np.pi / 2) - theta
            else:
                theta = torch.atan2(torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device))).item()
                desired = np.pi - theta

            # Turn to desired heading
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))

            # Move forward one step
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True,
                              update_pcn=True,
                              update_gcn=True,
                              update_scale_priority=True)
            self.forward()
            self.check_goal_reached_unified()
            s_start += 1

    def _execute_movement(self, heading_deg: float, show_debug: bool = False) -> bool:
        """Shared helper: Turn to heading and move forward with verification."""
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
                self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True, update_scale_priority=True)
                self.forward()
                self.check_goal_reached_unified()
            return False

        # Record pre-movement position
        pre_move_pos = self.robot.getField("translation").getSFVec3f()
        
        # Move forward
        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True, update_scale_priority=True)
            self.forward()
            self.check_goal_reached_unified()

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
   
    # ================================================================================================
    # SECTION 13: EXPLOITATION ALGORITHMS  
    # ================================================================================================
    
    def compass_based_turn_to_heading(self, target_heading_deg, debug=False):
        """
        NEW: Compass-based turning that breaks large turns into smaller increments
        and uses compass feedback for accuracy.
        
        Args:
            target_heading_deg: Target heading in degrees [0, 360)
            debug: Whether to print debug information
            
        Returns:
            bool: True if turn was successful, False otherwise
        """
        debug = False
        MAX_SINGLE_TURN = 10.0  # 30 Maximum degrees to turn in one step
        ACCEPTABLE_ERROR = 5.0  # 8 Acceptable final error in degrees
        MAX_ATTEMPTS = 30       # 10 Maximum number of turn attempts
        
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

    def apply_directional_momentum(self, directional_rewards):
        """
        Apply directional momentum to 8-directional reward values.
        Updated to use parameters from exploitation_params dictionary.
        
        Args:
            directional_rewards: torch.Tensor of shape (8,) with rewards per direction
            
        Returns:
            torch.Tensor: Momentum-adjusted directional rewards
        """
        if self.MOMENTUM_TYPE == 'none':
            return directional_rewards
        
        adjusted_rewards = directional_rewards.clone()
        
        if self.DEBUG_EXPLOIT_V1:
            print(f"Applying {self.MOMENTUM_TYPE} momentum (strength={self.MOMENTUM_STRENGTH})")
        
        if self.MOMENTUM_TYPE == 'bonus':
            # Give bonus for continuing in same or adjacent direction
            if self._last_chosen_direction is not None:
                for direction in range(self.n_hd):
                    direction_diff = abs(direction - self._last_chosen_direction)
                    direction_diff = min(direction_diff, self.n_hd - direction_diff)  # Wrap around
                    
                    if direction_diff == 0:  # Same direction
                        momentum_bonus = self.MOMENTUM_STRENGTH * directional_rewards[direction]
                        adjusted_rewards[direction] += momentum_bonus
                        if self.DEBUG_EXPLOIT_V1:
                            print(f"  Dir {direction}: same direction bonus +{momentum_bonus:.3f}")
                    elif direction_diff == 1:  # Adjacent direction (45° turn)
                        momentum_bonus = self.MOMENTUM_STRENGTH * 0.5 * directional_rewards[direction]
                        adjusted_rewards[direction] += momentum_bonus
                        if self.DEBUG_EXPLOIT_V1:
                            print(f"  Dir {direction}: adjacent direction bonus +{momentum_bonus:.3f}")
        
        elif self.MOMENTUM_TYPE == 'threshold':
            # Only allow direction change if significantly better
            if (self._last_chosen_direction is not None and 
                self._last_direction_reward > 0):
                
                current_best_direction = torch.argmax(directional_rewards).item()
                current_best_reward = directional_rewards[current_best_direction].item()
                
                if current_best_direction != self._last_chosen_direction:
                    # Different direction - check if improvement is sufficient
                    improvement = (current_best_reward - self._last_direction_reward) / (self._last_direction_reward + 1e-6)
                    
                    if improvement < self.CHANGE_THRESHOLD:
                        # Not enough improvement - penalize non-last directions
                        penalty_factor = 1.0 - self.MOMENTUM_STRENGTH
                        for direction in range(self.n_hd):
                            if direction != self._last_chosen_direction:
                                adjusted_rewards[direction] *= penalty_factor
                        
                        if self.DEBUG_EXPLOIT_V1:
                            print(f"  Insufficient improvement ({improvement:.2%} < {self.CHANGE_THRESHOLD:.2%}), penalizing direction changes")
        
        return adjusted_rewards

    def update_momentum_tracking(self, chosen_direction, direction_reward):
        """Update momentum tracking variables after action selection."""
        # Update direction history (simplified - no longer need extensive history for removed momentum types)
        if hasattr(self, '_direction_history'):
            self._direction_history.append(chosen_direction)
            if len(self._direction_history) > self.MOMENTUM_STEPS:
                self._direction_history.pop(0)
        else:
            self._direction_history = [chosen_direction]
        
        # Update tracking variables
        self._last_chosen_direction = chosen_direction
        self._last_direction_reward = direction_reward

    # Helper methods for different exploitation algorithms
    def compute_single_step_rewards(self, scale_idx):
        """Compute single-step rewards for a given scale."""
        pcn, rcn = self.pcns[scale_idx], self.rcns[scale_idx]
        pot_rew = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        
        for d in range(self.n_hd):
            if self.distances_per_hd[d] < self.OBSTACLE_DISTANCE_THRESHOLD:
                pot_rew[d] = 0.0
            else:
                preplayed_activations = pcn.preplay(d, num_steps=1)
                rcn.update_reward_cell_activations(preplayed_activations, visit=False)
                pot_rew[d] = torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
        
        return pot_rew

    def compute_scale_gradients(self, pot_rew_scales):
        """Compute gradient-based mixing weights for scales."""
        gradients = []
        for pot_rew in pot_rew_scales:
            grad = torch.sum(torch.abs(torch.diff(pot_rew))).item()
            gradients.append(grad)
        
        # Normalize gradients to mixing weights
        total_grad = sum(gradients) + 1e-6
        mixing_weights = torch.tensor([g / total_grad for g in gradients], dtype=self.dtype, device=self.device)
        
        return mixing_weights

    def compute_multi_step_rewards_with_goal_boost(self, scale_idx, active_goal, debug=False, decay_factor=0.6, max_steps=None):
        """
        Compute multi-step rewards with goal cell boosting for EXPLOIT_LOCATIONS mode.
        """
        if max_steps is None:
            max_steps = self.MULTISTEP_STEPS
        
        pcn, rcn = self.pcns[scale_idx], self.rcns[scale_idx]
        multi_step_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        
        # Setup evaluation function for this scale
        def evaluate_with_rcn(activations):
            rcn.update_reward_cell_activations(activations, visit=False)
            return torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
        
        # Store original evaluation method and set new one
        original_eval_method = getattr(pcn, '_evaluate_activations_for_reward', None)
        pcn._evaluate_activations_for_reward = evaluate_with_rcn
        
        try:
            # Multi-step evaluation in ALL 8 directions with goal cell boosting
            for direction in range(self.n_hd):
                if self.distances_per_hd[direction] < self.OBSTACLE_DISTANCE_THRESHOLD:
                    multi_step_rewards[direction] = 0.0
                else:
                    # UPDATED: Use new method name
                    base_reward = pcn.multi_step_preplay(
                        forced_first_direction=direction,
                        max_steps=max_steps,
                        decay_factor=decay_factor
                    )
                    
                    # Apply goal cell boost if applicable
                    boosted_reward = self.apply_goal_cell_boost(
                        base_reward, direction, scale_idx, active_goal, pcn, debug
                    )
                    
                    multi_step_rewards[direction] = boosted_reward
                    
        finally:
            # Restore original evaluation method
            if original_eval_method is not None:
                pcn._evaluate_activations_for_reward = original_eval_method
            elif hasattr(pcn, '_evaluate_activations_for_reward'):
                delattr(pcn, '_evaluate_activations_for_reward')
        
        return multi_step_rewards
    
    def apply_goal_cell_boost(self, base_reward, direction, scale_idx, active_goal, pcn, debug=False):
        """
        Apply goal cell activation boost to base reward.
        """
        if not active_goal or not hasattr(self, 'goal_place_cell_associations'):
            return base_reward
        
        goal_name = active_goal['name']
        
        # Check if we have goal associations for this goal and scale
        if (goal_name not in self.goal_place_cell_associations or 
            scale_idx >= len(self.goal_place_cell_associations[goal_name]) or
            self.goal_place_cell_associations[goal_name][scale_idx] is None):
            if debug:
                print(f"      No goal association for {goal_name} scale {scale_idx}")
            return base_reward
        
        # Get the goal-associated place cell index
        goal_pc_idx = self.goal_place_cell_associations[goal_name][scale_idx]
        
        # UPDATED: Use new method name
        final_activations = pcn.multi_step_preplay_final_state(
            direction, self.MULTISTEP_STEPS
        )
        
        # Check if goal place cell is significantly activated
        goal_cell_activation = final_activations[goal_pc_idx].item()
        activation_threshold = 0.1  # Minimum activation to consider "activated"
        
        if goal_cell_activation > activation_threshold:
            # Apply scale-dependent boost
            boost_factor = self.compute_goal_cell_boost_factor(scale_idx, goal_cell_activation)
            boosted_reward = base_reward * boost_factor
            
            if debug:
                scale_name = self.scales[scale_idx]['name']
                print(f"      GOAL CELL BOOST: Scale {scale_idx} ({scale_name}), Direction {direction}")
                print(f"        Goal PC {goal_pc_idx} activation: {goal_cell_activation:.3f}")
                print(f"        Boost factor: {boost_factor:.2f}x")
                print(f"        Reward: {base_reward:.3f} → {boosted_reward:.3f}")
            
            return boosted_reward
        
        return base_reward

    def compute_multi_step_rewards_all_directions(self, scale_idx, decay_factor=0.6):
        """
        Compute multi-step rewards for all 8 directions for a given scale using weighted steps.
        """
        pcn, rcn = self.pcns[scale_idx], self.rcns[scale_idx]
        multi_step_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        
        # Setup evaluation function for this scale
        def evaluate_with_rcn(activations):
            rcn.update_reward_cell_activations(activations, visit=False)
            return torch.max(torch.nan_to_num(rcn.reward_cell_activations)).item()
        
        # Store original evaluation method and set new one
        original_eval_method = getattr(pcn, '_evaluate_activations_for_reward', None)
        pcn._evaluate_activations_for_reward = evaluate_with_rcn
        
        try:
            # Multi-step evaluation in ALL 8 directions with weighting
            for direction in range(self.n_hd):
                if self.distances_per_hd[direction] < self.OBSTACLE_THRESHOLD:
                    multi_step_rewards[direction] = 0.0
                else:
                    # UPDATED: Use new method name
                    best_reward = pcn.multi_step_preplay(
                        forced_first_direction=direction,
                        max_steps=self.MULTISTEP_STEPS,
                        decay_factor=decay_factor
                    )
                    multi_step_rewards[direction] = best_reward
                        
        finally:
            # Restore original evaluation method
            if original_eval_method is not None:
                pcn._evaluate_activations_for_reward = original_eval_method
            elif hasattr(pcn, '_evaluate_activations_for_reward'):
                delattr(pcn, '_evaluate_activations_for_reward')
        
        return multi_step_rewards

    # ================================================================================================
    # SECTION 14: DATA LOGGING & ANALYSIS
    # ================================================================================================
    
    def update_hmaps(self,
                     update_loc=False,
                     update_hdn=False,
                     update_pcn=False,
                     update_gcn=False,
                     update_scale_priority=False,
                     update_prox=False):
        """
        Store agent position, head direction activations, place cell activations, scale priority (previously alpha), and proximity values.
        
        Parameters:
        - update_loc (bool): Whether to update agent location history.
        - update_hdn (bool): Whether to update head direction activations.
        - update_pcn (bool): Whether to update place cell activations.
        - update_scale_priority (bool): Whether to update scale priority (dominant scale index).
        - update_prox (bool): Whether to update proximity values.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # 1) Update agent location if requested
            if update_loc:
                self.hmap_loc[self.step_count] = curr_pos

            # 2) Update head direction activations
            if update_hdn:
                self.hmap_hdn[self.step_count] = self.hd_activations.clone().detach().cpu()

            # 3) Dynamically resize hmap_pcn_activities if needed
            if update_pcn and len(self.hmap_pcn_activities) != len(self.pcn_activations_list):
                self.hmap_pcn_activities = [
                    torch.zeros((self.num_steps, act.shape[0]), device="cpu", dtype=torch.float32)
                    for act in self.pcn_activations_list
                ]

        # 4) Update place cell activations for each scale
        # Create a mapping from scale_index to valid list index in hmap_pcn_activities
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
                        (self.num_steps, act.shape[0]), device="cpu", dtype=torch.float32
                    )

                # Store activations at the correct list index
                self.hmap_pcn_activities[mapped_index][self.step_count] = act.clone().detach().cpu()

            # 5) Update scale priority
            if update_scale_priority and hasattr(self, 'scale_idx'):
                # print(f"scale_idx: {self.scale_idx}")
                self.hmap_scale_priority[self.step_count] = self.scale_idx 

            # 6) Update proximity value if available
            if update_prox and hasattr(self, 'prox'):
                self.hmap_prox[self.step_count] = self.prox

        # 4) Update grid cell activations for each scale
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
                        (self.num_steps, act.shape[0]), device="cpu", dtype=torch.float32
                    )
                    
                # Store activations directly
                self.hmap_gcn_activities[mapped_index][self.step_count] = act

        # Increment step count
        self.step_count += 1

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

    # ================================================================================================
    # SECTION 15: FILE I/O & PERSISTENCE
    # ================================================================================================
    
    def save(self, include_pcn=False, include_rcn=False, include_gcn=False, 
             include_hmaps=False, save_trajectory=False):
        """
        Save networks and data to disk.
        Updated to work correctly with new scale indexing system.
        """
        files_saved = []

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Save each scale's PCN (if requested)
        if include_pcn:
            for scale_config, pcn in zip(self.scales, self.pcns):
                scale_idx = scale_config["scale_index"]
                pcn_path = os.path.join(self.network_dir, f"pcn_scale_{scale_idx}.pkl")
                with open(pcn_path, "wb") as f:
                    pickle.dump(pcn, f)
                files_saved.append(pcn_path)

        # Save each scale's RCN (if requested)
        if include_rcn:
            for scale_config, rcn in zip(self.scales, self.rcns):
                scale_idx = scale_config["scale_index"]
                rcn_path = os.path.join(self.network_dir, f"rcn_scale_{scale_idx}.pkl")
                with open(rcn_path, "wb") as f:
                    pickle.dump(rcn, f)
                files_saved.append(rcn_path)
                
        # Save each scale's GCN (if requested)
        if include_gcn:
            for scale_config, gcn in zip(self.scales, self.gcns):
                if gcn is not None:
                    scale_idx = scale_config["scale_index"]
                    gcn_path = os.path.join(self.network_dir, f"gcn_scale_{scale_idx}.pkl")
                    with open(gcn_path, "wb") as f:
                        pickle.dump(gcn, f)
                    files_saved.append(gcn_path)

        # Save the history maps if requested
        if include_hmaps:
            # Agent location
            hmap_loc_path = os.path.join(self.hmap_dir, "hmap_loc.pkl")
            with open(hmap_loc_path, "wb") as f:
                pickle.dump(self.hmap_loc[: self.step_count], f)
            files_saved.append(hmap_loc_path)

            # Head direction history
            hmap_hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
            with open(hmap_hdn_path, "wb") as f:
                pickle.dump(self.hmap_hdn[: self.step_count].cpu(), f)
            files_saved.append(hmap_hdn_path)

            # Place-cell history maps for each scale
            for scale_config, pc_history in zip(self.scales, self.hmap_pcn_activities):
                scale_idx = scale_config["scale_index"]
                hmap_scale_path = os.path.join(self.hmap_dir, f"hmap_pcn_scale_{scale_idx}.pkl")
                
                with open(hmap_scale_path, "wb") as f:
                    pc_data = pc_history[: self.step_count].cpu().numpy()
                    pickle.dump(pc_data, f)
                files_saved.append(hmap_scale_path)
                
            # Grid-cell history maps for each scale
            for scale_config, gc_history in zip(self.scales, self.hmap_gcn_activities):
                if gc_history.numel() > 0:  # Only save if there are grid cells
                    scale_idx = scale_config["scale_index"]
                    hmap_scale_path = os.path.join(self.hmap_dir, f"hmap_gcn_scale_{scale_idx}.pkl")
                    
                    with open(hmap_scale_path, "wb") as f:
                        gc_data = gc_history[: self.step_count].cpu().numpy()
                        pickle.dump(gc_data, f)
                    files_saved.append(hmap_scale_path)

            # Prox values
            if hasattr(self, "hmap_prox"):
                hmap_prox_path = os.path.join(self.hmap_dir, "hmap_prox.pkl")
                with open(hmap_prox_path, "wb") as f:
                    prox_data = self.hmap_prox[: self.step_count].cpu().numpy()
                    pickle.dump(prox_data, f)
                files_saved.append(hmap_prox_path)

        # Save the agent's path if requested
        if save_trajectory:
            # Get world name and parse scale names
            scale_name_list = [scale["name"] for scale in self.scales]
            scale_order = ["small", "medium", "large", "xlarge"]
            scale_name_list = sorted(scale_name_list, key=lambda x: scale_order.index(x) if x in scale_order else 999)
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

        # Print saved files
        print(f"[DEBUG] Save dialog check: self.stats_collector = {self.stats_collector}")
        print(f"[DEBUG] Save dialog check: self.stats_collector is not None = {self.stats_collector is not None}")
        if not self.stats_collector:
            print(f"[DEBUG] No stats_collector found (mode without save_data=True), showing save dialog...")
            print(f"[DEBUG] This is normal for single-run modes like OJAS, LEARNING, etc.")
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            messagebox.showinfo("Information", "Press OK to save data")
            root.destroy()
        else:
            print(f"[DEBUG] stats_collector found (mode with save_data=True), skipping save dialog")
            print(f"[DEBUG] Data automatically saved to analysis/stats/ folder")

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def clear(self):
            """
            Removes all scale-specific PCN/RCN/GCN files and any hmap files.
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
            # NEW: Clear multi-goal rewards directory if it exists
            multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
            if os.path.exists(multi_goal_dir):
                for fname in os.listdir(multi_goal_dir):
                    full_path = os.path.join(multi_goal_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

            print("[DRIVER] Finished clearing old scale PCNs, RCNs, GCNs, and hmap files.")

    # ================================================================================================
    # SECTION 16: GOAL-SPECIFIC METHODS (Multi-goal modes)
    # ================================================================================================
    
    def _set_target_goal(self, goal_name):
        """Set which goal is currently targeted for exploitation"""
        found = False
        for goal in self.goals:
            goal["active"] = (goal["name"] == goal_name)
            if goal["name"] == goal_name:
                found = True
        
        if not found:
            raise ValueError(f"Target goal '{goal_name}' not found in goals list")
        
        self.active_goal_name = goal_name
        print(f"[DRIVER] Set target goal: {goal_name}")

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

    def _handle_goal_exploitation(self, goal):
        """Handle goal reached during exploitation mode"""
        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time
            
            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            self.stats_collector.update_stat("goal_location", goal["location"])
            
            try:
                self.stats_collector.update_stat("goal_name", goal["name"])
            except KeyError:
                pass
                
            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("success", True)
            self.stats_collector.save_stats(self.trial_id)
            
            print(f"Trial {self.trial_id} completed successfully.")
            print(f"Reached goal: {goal['name']} at {goal['location']}")
            print(f"Total distance: {round(self.compute_path_length(), 2)}m")
            print(f"Trial time: {round(trial_time, 2)}s")

        self.stop()

        # EXISTING: Standard goal-based reward update
        if self.td_learning:
            for pcn, rcn in zip(self.pcns, self.rcns):
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                rcn.replay(pcn=pcn)

        self.save(
            include_pcn=True if self.td_learning else False,
            include_rcn=True if self.td_learning else False,
            include_gcn=True if self.td_learning else False,
            save_trajectory=True
        )
        print(f"[DEBUG] Setting self.done = True for goal reached")
        self.done = True

    def _handle_single_goal_exploitation(self):
        """Handle single goal reached during exploitation"""
        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time
            
            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            self.stats_collector.update_stat("goal_location", self.goal_location)
            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("success", True)
            self.stats_collector.save_stats(self.trial_id)
            
            print(f"Trial {self.trial_id} completed successfully.")
            print(f"Total distance: {round(self.compute_path_length(), 2)}m")
            print(f"Trial time: {round(trial_time, 2)}s")
        
        self.stop()
        
        for pcn, rcn in zip(self.pcns, self.rcns):
            rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
            rcn.replay(pcn=pcn)
          
        self.save(
            include_pcn=True if self.td_learning else False,
            include_rcn=True if self.td_learning else False,
            include_gcn=True if self.td_learning else False,
            save_trajectory=True
        )
        self.done = True

    def _handle_dmtp_goal_reached(self):
        """Handle goal reached in DMTP mode"""
        self.auto_pilot()
        self.stop()
        
        self.refresh_pcn_activations("before reward map formation in DMTP")
        
        for pcn, rcn in zip(self.pcns, self.rcns):
            rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
            rcn.replay(pcn=pcn)
        self.save(include_pcn=True, include_rcn=True, include_gcn=True)
        self.done = True
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    def _handle_exploitation_timeout(self):
        """Handle timeout during exploitation modes"""
        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time  # Use trial-relative time
            
            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            if self.multi_goal_mode:
                active_goal = next((g for g in self.goals if g.get("active", False)), None)
                if active_goal:
                    self.stats_collector.update_stat("goal_location", active_goal["location"])
                    try:
                        self.stats_collector.update_stat("goal_name", active_goal["name"])
                    except KeyError:
                        pass
            else:
                self.stats_collector.update_stat("goal_location", self.goal_location)
            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))  # Use trial time
            self.stats_collector.update_stat("success", False)
            self.stats_collector.save_stats(self.trial_id)
        
        self.stop()

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
                goal_rcn.replay_with_custom_activations(pcn=pcn, custom_activations=artificial_activations)
                
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

    # ================================================================================================
    # SECTION 16: GOAL CHECKING AND TRIAL MANAGEMENT
    # ================================================================================================

    def check_goal_reached_unified(self):
        """Unified goal checking for all modes"""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

        # Calculate elapsed time for this trial
        trial_elapsed_time = self.getTime() - self.trial_start_time
        current_sim_time = self.getTime()

        # DEBUG: Print timing info every 30 seconds
        if not hasattr(self, '_last_debug_time'):
            self._last_debug_time = 0
        if current_sim_time - self._last_debug_time >= 30:
            # print(f"[DEBUG] Trial timing - Elapsed: {trial_elapsed_time:.1f}s, Limit: {60 * self.run_time_minutes:.1f}s, Sim Time: {current_sim_time:.1f}s")
            self._last_debug_time = current_sim_time

        # Handle mode-specific time limits and general completion
        if self.robot_mode in (RobotMode.LEARNING, RobotMode.PLOTTING) \
                and trial_elapsed_time >= 60 * self.run_time_minutes:
            print(f"[INFO] Trial time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
            self.stop()
            self.save(include_pcn=self.robot_mode != RobotMode.PLOTTING,
                    include_rcn=self.robot_mode != RobotMode.PLOTTING,
                    include_gcn=self.robot_mode != RobotMode.PLOTTING,
                    include_hmaps=True)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return

        # Multi-goal mode handling
        if self.multi_goal_mode:
            if self.robot_mode == RobotMode.LEARN_LOCATIONS:
                # Check all goals for learning
                for goal in self.goals:
                    goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                    distance = torch.norm(current_position - goal_position)

                    if distance <= goal["radius"]:
                        if not goal["visited"]:
                            print(f"[LEARN_LOCATIONS] First visit to {goal['name']} goal at {goal['location']}")
                            goal["visited"] = True
                        self._handle_goal_learning(goal)

                # Check if learning is complete
                minimum_time_reached = trial_elapsed_time >= 60 * self.run_time_minutes
                if minimum_time_reached:
                    if self._check_multi_goal_learning_complete():
                        print(f"[LEARN_LOCATIONS] Learning complete! Trial time: {trial_elapsed_time:.1f}s")
                        self._create_multi_goal_reward_maps()
                        self._save_multi_goal_data()
                        self.stop()
                        self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                        self.done = True
                        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                    else:
                        print(f"[LEARN_LOCATIONS] Minimum time reached but learning incomplete, continuing...")

            elif self.robot_mode == RobotMode.EXPLOIT_LOCATIONS:
                # Check only active goal for exploitation
                active_goals = [g for g in self.goals if g.get("active", False)]
                for goal in active_goals:
                    goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
                    distance = torch.norm(current_position - goal_position)

                    if distance <= goal["radius"]:
                        self._handle_goal_exploitation(goal)
                        return

                # Check time limit for exploitation - Use trial-relative time
                if self.step_count > 10 and trial_elapsed_time >= 60 * self.run_time_minutes:
                    print(f"Trial time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
                    self._handle_exploitation_timeout()
                    return

        else:
            # Single goal mode handling
            goal = self.goals[0]
            goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
            distance = torch.norm(current_position - goal_position)

            if distance <= goal["radius"]:
                if self.robot_mode == RobotMode.DMTP:
                    self._handle_dmtp_goal_reached()
                elif self.robot_mode == RobotMode.EXPLOIT:
                    self._handle_single_goal_exploitation()
                # Other single goal modes handled by existing logic
                return

            # Check time limit for single goal exploitation - Use trial-relative time
            if self.robot_mode == RobotMode.EXPLOIT and self.step_count > 10:
                if trial_elapsed_time >= 60 * self.run_time_minutes:
                    print(f"Trial time limit reached: {trial_elapsed_time:.1f}s / {60 * self.run_time_minutes:.1f}s")
                    self._handle_exploitation_timeout()
                    return

    def _handle_goal_exploitation(self, goal):
        """Handle goal reached during exploitation mode"""
        print(f"[DEBUG] _handle_goal_exploitation() called for goal {goal.get('name', 'unknown')}")

        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time

            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            self.stats_collector.update_stat("goal_location", goal["location"])

            try:
                self.stats_collector.update_stat("goal_name", goal["name"])
            except KeyError:
                pass

            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("success", True)
            self.stats_collector.save_stats(self.trial_id)

            print(f"Trial {self.trial_id} completed successfully.")
            print(f"Reached goal: {goal['name']} at {goal['location']}")

        self.save(include_pcn=False,
            include_rcn=True if self.td_learning else False,
            include_gcn=True if self.td_learning else False,
            save_trajectory=True
        )
        self.done = True

    def _handle_exploitation_timeout(self):
        """Handle timeout during exploitation modes"""
        print(f"[DEBUG] _handle_exploitation_timeout() called")

        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time  # Use trial-relative time

            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            if self.multi_goal_mode:
                active_goal = next((g for g in self.goals if g.get("active", False)), None)
                if active_goal:
                    self.stats_collector.update_stat("goal_location", active_goal["location"])
                    try:
                        self.stats_collector.update_stat("goal_name", active_goal["name"])
                    except KeyError:
                        pass
            else:
                self.stats_collector.update_stat("goal_location", self.goal_location)
            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))  # Use trial time
            self.stats_collector.update_stat("success", False)
            self.stats_collector.save_stats(self.trial_id)

        self.stop()

        if self.td_learning:
            for pcn, rcn in zip(self.pcns, self.rcns):
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=False)
            self.save(include_pcn=True, include_rcn=True, include_gcn=True, save_trajectory=True)
        else:
            self.save(save_trajectory=True)

        print(f"[DEBUG] Setting self.done = True for timeout")
        self.done = True

    def _handle_dmtp_goal_reached(self):
        """Handle goal reached in DMTP mode"""
        self.auto_pilot()
        self.stop()

        self.refresh_pcn_activations("before reward map formation in DMTP")

        for pcn, rcn in zip(self.pcns, self.rcns):
            rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
            rcn.replay(pcn=pcn)
        self.save(include_pcn=True, include_rcn=True, include_gcn=True)
        self.done = True
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    def _handle_single_goal_exploitation(self):
        """Handle single goal reached during exploitation"""
        if self.stats_collector:
            trial_time = self.getTime() - self.trial_start_time

            self.stats_collector.update_stat("trial_id", self.trial_id)
            self.stats_collector.update_stat("start_location", self.start_loc)
            self.stats_collector.update_stat("goal_location", self.goal_location)
            self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
            self.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
            self.stats_collector.update_stat("success", True)
            self.stats_collector.save_stats(self.trial_id)

            print(f"Trial {self.trial_id} completed successfully.")
            print(f"Total distance: {round(self.compute_path_length(), 2)}m")
            print(f"Trial time: {round(trial_time, 2)}s")

        self.save(include_pcn=False,
            include_rcn=True if self.td_learning else False,
            include_gcn=True if self.td_learning else False,
            save_trajectory=True
        )
        self.done = True

    # ================================================================================================
    # SECTION 17: UTILITY METHODS
    # ================================================================================================
    
    # Note: Several helper methods from original may be unused and can be removed:
    # - Any version-specific compatibility methods
    # - Deprecated parameter handling methods  
    # - Unused debugging methods
    
    # Methods that definitely should be kept:
    # - All movement methods (forward, turn, stop, etc.)
    # - All sensing methods (sense, get_bearing_in_degrees, etc.)
    # - All goal handling methods
    # - All file I/O methods
    # - All exploitation algorithm methods