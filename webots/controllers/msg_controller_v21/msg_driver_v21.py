import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from controller import Supervisor
import random
import math
from typing import Optional, List, Dict, Any
import tkinter as tk
from tkinter import messagebox
import copy

# Add root directory to PYTHONPATH
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.msg_pcn_v21 import MultiscalePlaceCellWithGrid
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer_v4 import RewardCellLayerV4
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

import numpy as np

# Torch and NumPy settings
np.set_printoptions(precision=2)

class MultiscaleDriverWithGrid(Supervisor):
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
        replay_timesteps: Optional[List[int]] = None,
        replay_decay_constants: Optional[List[float]] = None,
        stats_collector: Optional[stats_collector] = None,
        trial_id: Optional[str] = None,
        world_name: Optional[str] = None,
        goal_config: Optional[Dict[str, Any]] = None,
        trial_config: Optional[Dict[str, Any]] = None,
        max_dist: Optional[float] = None,
        plot_bvc: Optional[bool] = False,
        td_learning: Optional[bool] = False,
        use_prox_mod: Optional[bool] = False,
        clear_files: Optional[bool] = False,
        action_mode: str = None,
        environment_size: Optional[List[float]] = None,
        grid_size: Optional[float] = None,
        coverage_percentage: Optional[float] = None,
        optimal_path_distance: Optional[float] = None,
        path_failure_ratio: Optional[float] = None,
        paths_folder: Optional[str] = None,
        hmaps_folder: Optional[str] = None,
    ):
        self.STEP_DECAY_FACTOR = 1.0 # 1.0
        self.action_mode = action_mode
        self.DEBUG_EXPLOIT_V0 = True
        self.DEBUG_EXPLOIT_V1 = True
        
        self.MOMENTUM_TYPE = 'bonus'               
        self.MOMENTUM_STRENGTH = 0.3 #0.1              
        self.MOMENTUM_STEPS = 15         #10           
        self.CHANGE_THRESHOLD = 0.5 # .5            
        
        self._direction_history = []
        self._last_chosen_direction = None
        self._last_direction_reward = 0.0

        self.REWARD_THRESHOLD = 0.1
        self.MULTISTEP_THRESHOLD = 0.2  
        self.MULTISTEP_MAX_SCALES = 2
        self.ENHANCEMENT_BONUS = 1.2
        self.MULTISTEP_STEPS = 3
        self.OBSTACLE_DISTANCE_THRESHOLD = 0.5

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
        self.max_speed = 16
        self.max_dist = max_dist
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756

        # Simulation run time
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))

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

        # Scale-specific multistep parameters
        self.SCALE_MULTISTEP_STEPS = {
            "small": 3,    # Fine scale - more steps for detailed planning
            "medium": 2,   # Medium scale - balanced planning
            "large": 1,    # Large scale - fewer steps, broader planning
            "xlarge": 1    # Extra large scale - minimal steps
        }

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
        self._setup_goals(goal_config)
        
        # Setup trial configuration
        self._setup_trials(trial_config)
        
        self.start_loc = start_loc
        
        # Store replay parameters for later use in RCN initialization
        self.replay_timesteps = replay_timesteps if replay_timesteps is not None else [40] * len(scales)
        self.replay_decay_constants = replay_decay_constants if replay_decay_constants is not None else [6.0] * len(scales)

        # Store learning rates for later use in RCN initialization
        self.rcn_learning_rates = rcn_learning_rates if rcn_learning_rates is not None else [0.1] * len(scales)
        self.td_learning = td_learning
        self.use_prox_mod = use_prox_mod

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

        if clear_files:
            print(f"[DRIVER] Clearing files due to clear_files=True for mode {mode}")
            self.clear()

        # Load or init PCNs / RCNs / GCNs
        self.gcns = []
        self.init_grid_cell_networks()
        self.pcns = []
        self.rcns = []
        self.load_pcns(enable_ojas, enable_stdp)
        self.load_rcns()
        print(f"[DRIVER] Using PCNs: {[f'pcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")
        print(f"[DRIVER] Using RCNs: {[f'rcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")

        # Load goal-specific RCNs if needed for EXPLOIT_LOCATIONS or EXPLOIT_LOCATIONS_RANDOM
        if mode in {RobotMode.EXPLOIT_LOCATIONS, RobotMode.EXPLOIT_LOCATIONS_RANDOM} and hasattr(self, 'active_goal_name'):
            self._load_goal_rcns(self.active_goal_name)

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
            (self.num_steps, len(self.scales)),  
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
        self.LOOP_THRESHOLD = 5  
        self.MAX_STEPS_BETWEEN_LOOPS = 10  
        self.force_explore_count = 0 

        # Optionally keep a single-scale reference
        self.pcn = self.pcns[0] if self.pcns else None
        self.plot_bvc = plot_bvc
        self.manual_explore = False

        # Step once
        self.step(self.timestep)

        # NEW: Track trial start time for per-trial timeouts
        self.trial_start_time = self.getTime()
        print(f"[DRIVER] Trial started at simulation time: {self.trial_start_time:.1f}s")

    #################################################################################################
    #                                    PCN / RCN / GCN LOADING                                    #
    #################################################################################################

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
                enable_ojas if enable_ojas is not None else None,
                enable_stdp if enable_stdp is not None else None,
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
                print(f"[DRIVER] PCN loaded - updating parameters")
                # Update all parameters from scale definition
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp

                # Update STDP parameters
                pcn.stdp_learning_rate = scale_def.get("stdp_learning_rate", 0.05)
                pcn.tau_hd = scale_def.get("tau_hd", 0.1)

                # Update grid and connection parameters
                pcn.grid_influence = scale_def.get("grid_influence", 0.5)
                pcn.gamma_pg = scale_def.get("gamma_pg", 0.3)
                pcn.enable_connection_decay = scale_def.get("enable_connection_decay", True)
                pcn.connection_decay_rate = scale_def.get("connection_decay_rate", 0.002)
                pcn.enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
                pcn.correlation_window = scale_def.get("correlation_window", 100)
                pcn.correlation_update_freq = scale_def.get("correlation_update_freq", 10)
                pcn.correlation_scaling = scale_def.get("correlation_scaling", 2.0)
                pcn.min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
                pcn.correlation_threshold = scale_def.get("correlation_threshold", 0.01)

                # Update adaptive learning parameters
                pcn.enable_adaptive_stdp = scale_def.get("enable_adaptive_stdp", False)
                pcn.adaptive_initial_lr = scale_def.get("adaptive_initial_lr", 0.15)
                pcn.adaptive_final_lr = scale_def.get("adaptive_final_lr", 0.02)
                pcn.adaptive_decay_rate = scale_def.get("adaptive_decay_rate", 3.0)

                print(f"[DRIVER] Updated PCN with adaptive STDP: {pcn.enable_adaptive_stdp}")
                if pcn.enable_adaptive_stdp:
                    print(f"[DRIVER] Adaptive learning: {pcn.adaptive_initial_lr} → {pcn.adaptive_final_lr} (decay: {pcn.adaptive_decay_rate})")

                return pcn

            else:
                print(f"[DRIVER] Unknown PCN version {pcn_class_name} - creating new PCN")
                return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[DRIVER] Could not load PCN from {path}: {e}")
            return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)

    def _initialize_new_pcn(self, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        """Initialize a new place cell network with adaptive learning rates."""
        print(f"[DRIVER] Initializing new PCN for scale {scale_def['scale_index']}")

        # Create BVC layer
        bvc = BoundaryVectorCellLayer(
            max_dist=self.max_dist,
            n_res=720,
            n_hd=self.n_hd,
            sigma_theta=scale_def.get("sigma_theta", 1.0),
            sigma_r=scale_def.get("sigma_r", 0.5),
            device=self.device,
        )

        # Get all parameters from scale definition
        # Adaptive learning parameters
        enable_adaptive_stdp = scale_def.get("enable_adaptive_stdp", False)
        adaptive_initial_lr = scale_def.get("adaptive_initial_lr", 0.15)
        adaptive_final_lr = scale_def.get("adaptive_final_lr", 0.02)
        adaptive_decay_rate = scale_def.get("adaptive_decay_rate", 3.0)

        # Logging and debugging parameters
        enable_debug_prints = scale_def.get("enable_debug_prints", False)
        enable_adaptive_logging = scale_def.get("enable_adaptive_logging", False)

        # STDP parameters
        stdp_learning_rate = scale_def.get("stdp_learning_rate", 0.05)
        tau_hd = scale_def.get("tau_hd", 0.1)

        # Grid and connection parameters
        grid_influence = scale_def.get("grid_influence", 0.5)
        gamma_pg = scale_def.get("gamma_pg", 0.3)
        enable_connection_decay = scale_def.get("enable_connection_decay", True)
        connection_decay_rate = scale_def.get("connection_decay_rate", 0.002)
        enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
        correlation_window = scale_def.get("correlation_window", 100)
        correlation_update_freq = scale_def.get("correlation_update_freq", 10)
        correlation_scaling = scale_def.get("correlation_scaling", 2.0)
        min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
        correlation_threshold = scale_def.get("correlation_threshold", 0.01)

        enable_proximity_suppression = scale_def.get("enable_proximity_suppression", True)
        proximity_threshold_factor = scale_def.get("proximity_threshold_factor", 2.0)
        proximity_suppression_steepness = scale_def.get("proximity_suppression_steepness", 10.0)
        proximity_suppression_midpoint = scale_def.get("proximity_suppression_midpoint", 0.5)

        # Create place cell layer with adaptive learning
        pcn = MultiscalePlaceCellWithGrid(
            bvc_layer=bvc,
            num_pc=scale_def["num_pc"],
            num_grid_cells=num_grid_cells,
            timestep=self.timestep,
            n_hd=self.n_hd,
            enable_ojas=enable_ojas if enable_ojas is not None else False,
            enable_stdp=enable_stdp if enable_stdp is not None else False,
            grid_influence=grid_influence,
            gamma_pp=scale_def.get("gamma_pp", 0.5),
            gamma_pb=scale_def.get("gamma_pb", 0.3),
            gamma_pg=gamma_pg,
            enable_connection_decay=enable_connection_decay,
            connection_decay_rate=connection_decay_rate,
            enable_correlation_weighting=enable_correlation_weighting,
            correlation_window=correlation_window,
            correlation_update_freq=correlation_update_freq,
            correlation_scaling=correlation_scaling,
            min_correlation_weight=min_correlation_weight,
            correlation_threshold=correlation_threshold,
            stdp_learning_rate=stdp_learning_rate,
            tau_hd=tau_hd,
            # Adaptive learning parameters
            enable_adaptive_stdp=enable_adaptive_stdp,
            adaptive_initial_lr=adaptive_initial_lr,
            adaptive_final_lr=adaptive_final_lr,
            adaptive_decay_rate=adaptive_decay_rate,
            # Scale name for debugging
            scale_name=scale_def.get("name", "unknown"),
            # Logging and debugging parameters
            enable_debug_prints=enable_debug_prints,
            enable_adaptive_logging=enable_adaptive_logging,
            enable_proximity_suppression=enable_proximity_suppression,
            proximity_threshold_factor=proximity_threshold_factor,
            proximity_suppression_steepness=proximity_suppression_steepness,
            proximity_suppression_midpoint=proximity_suppression_midpoint,
            device=self.device,
        )

        print(f"[DRIVER] Successfully created new PCN with:")
        print(f"  - Adaptive STDP: {enable_adaptive_stdp}")
        if enable_adaptive_stdp:
            print(f"  - Learning rate range: {adaptive_initial_lr} → {adaptive_final_lr}")
            print(f"  - Decay rate: {adaptive_decay_rate}")
        else:
            print(f"  - Fixed STDP learning rate: {stdp_learning_rate}")
        print(f"  - Correlation weighting: {enable_correlation_weighting}")
        print(f"  - Connection decay rate: {connection_decay_rate}")
        print(f"  - Grid influence: {grid_influence}")
        
        return pcn
     
    def load_rcns(self):
        self.rcns = []
        for i, scale_def in enumerate(self.scales):
            scale_idx = scale_def["scale_index"]
            fname = f"rcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)
            learning_rate = scale_def["rcn_learning_rate"]
            
            # Get replay parameters for this scale
            replay_timesteps = self.replay_timesteps[i]
            replay_decay_constant = self.replay_decay_constants[i]
            
            rcn = self._load_or_init_rcn_for_scale(
                path, scale_def, learning_rate, replay_timesteps, replay_decay_constant
            )
            self.rcns.append(rcn)
            
            # Print replay configuration for verification
            print(f"[DRIVER] Scale {scale_def['name']}: "
                f"timesteps={replay_timesteps}, decay={replay_decay_constant}, "
                f"lr={learning_rate}")

    def _load_or_init_rcn_for_scale(self, path, scale_def, learning_rate, replay_timesteps, replay_decay_constant):
        try:
            with open(path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing RCN from {path}")
            
            # Remove all the version conversion logic - just update parameters
            rcn.replay_timesteps = replay_timesteps
            rcn.replay_decay_constant = replay_decay_constant
            
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[DRIVER] Creating new RCN for {path}")
            rcn = RewardCellLayerV4(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                replay_timesteps=replay_timesteps,
                replay_decay_constant=replay_decay_constant,
                device=self.device,
            )
        
        rcn.debug_weights(f"After loading/creating RCN scale {scale_def['scale_index']}")
        return rcn

    def _load_goal_rcns(self, goal_name):
        """Load RCNs for specific goal - uses existing naming convention"""
        print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")
        
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        if not os.path.exists(multi_goal_dir):
            raise FileNotFoundError(f"Multi-goal rewards directory not found: {multi_goal_dir}")
        
        loaded_count = 0
        for i, scale_def in enumerate(self.scales):
            scale_idx = scale_def["scale_index"]
            goal_rcn_path = os.path.join(
                multi_goal_dir, 
                f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
            )
            
            if os.path.exists(goal_rcn_path):
                try:
                    with open(goal_rcn_path, "rb") as f:
                        self.rcns[i] = pickle.load(f)
                    print(f"[DRIVER] Loaded goal RCN: scale {scale_idx} for goal {goal_name}")
                    loaded_count += 1
                except Exception as e:
                    raise RuntimeError(f"Failed to load goal RCN for scale {scale_idx}, goal {goal_name}: {e}")
            else:
                raise FileNotFoundError(f"Goal-specific RCN not found: {goal_rcn_path}")
        
        print(f"[DRIVER] Successfully loaded {loaded_count} goal-specific RCNs for {goal_name}")

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
    
    ###################################### Goal & Trail Setup #######################################
    
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
            elif self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE:
                self.goal_place_cell_associations = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                self.goal_association_step = {
                    goal["name"]: [None] * len(self.scales) for goal in self.goals
                }
                # Initialize coverage tracking
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

    ########################################### RUN LOOP ############################################

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

    ############################################ EXPLORE ############################################

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
            if self.robot_mode in {RobotMode.DMTP, RobotMode.EXPLOIT, RobotMode.EXPLOIT_LOCATIONS, RobotMode.EXPLOIT_LOCATIONS_RANDOM}:
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
        if self.manual_explore:
            self.manual_control()
        else:
            self.turn(np.random.normal(0, np.deg2rad(30)))

    ############################################ EXPLOIT ############################################

    def exploit_v0(self):
        """
        Basic multi-scale exploitation logic with improved compass-based turning.
        Uses full multi-scale decision making without advanced features like multistep thresholds.
        """
        
        # Reduce output frequency for multi-scale system
        if not hasattr(self, '_exploit_v0_call_count'):
            self._exploit_v0_call_count = 0
        self._exploit_v0_call_count += 1
        
        # Show detailed debug every 10th call for complex multi-scale system
        show_detailed_debug = self.DEBUG_EXPLOIT_V0 and (self._exploit_v0_call_count % 10 == 0)
        show_debug = self.DEBUG_EXPLOIT_V0

        if show_detailed_debug:
            print(f"\n{'='*70}")
            print(f"EXPLOIT V0 - STEP {self._exploit_v0_call_count}")
            print("="*70)

        # 1) Sense and compute: update heading, place/boundary cell activations
        self._sense_and_compute()

        # Save old PCN activations for each scale
        old_pcn_activations = [pcn.place_cell_activations.clone() for pcn in self.pcns]

        # Exploit can only begin with at least 10 steps
        if self.step_count <= self.tau_w:
            return

        # 2) Forced Exploration Check & Cooldown Setup
        if getattr(self, 'force_explore_count', 0) > 0:
            self.force_explore_count -= 1
            if self.force_explore_count == 0:
                if show_debug:
                    print("Forced exploration complete. Resuming exploitation...")
            self.explore()
            return

        # 3) Detect excessive rotation and enforce forced exploration if needed
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
            if self.steps_since_last_loop > getattr(self, 'MAX_STEPS_BETWEEN_LOOPS', 10):
                self.rotation_loop_count = 1
            self.steps_since_last_loop = 0

            if self.rotation_loop_count >= getattr(self, 'LOOP_THRESHOLD', 5):
                self.rotation_loop_count = 0
                self.rotation_accumulator = 0.0
                self.steps_since_last_loop = 0
                if show_debug:
                    print("Detected excessive loops. Starting forced exploration...")

                # Store the preferred scale before entering forced exploration
                if hasattr(self, "last_preferred_scale_index"):
                    self.cooldown_scale_index = self.last_preferred_scale_index
                    self.cooldown_steps_remaining = 20
                    if show_debug:
                        print(f"Cooling down scale {self.cooldown_scale_index} for 20 steps")

                # Start forced exploration
                self.force_explore_count = 5
                self.explore()
                return
        else:
            self.steps_since_last_loop += 1
            if self.steps_since_last_loop > getattr(self, 'MAX_STEPS_BETWEEN_LOOPS', 10):
                self.rotation_loop_count = 0

        # 4) Compute potential rewards at multiple scales, skipping invalid directions
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

        if show_detailed_debug:
            print("Obstacle distances per direction:")
            for d in range(self.n_hd):
                angle_deg = d * 45
                print(f"  Direction {d} ({angle_deg}°): {distances_per_hd[d]:.2f}m")

        pot_rew_scales = []
        valid_scale_indices = []
        reward_threshold = 0.05  # Threshold for considering a scale valid

        for i, scale_def in enumerate(self.scales):
            # Check cooldown
            if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
                if i == getattr(self, "cooldown_scale_index", -1):
                    if show_detailed_debug:
                        print(f"Skipping scale {i} ({scale_def['name']}) due to cooldown")
                    continue

            pcn, rcn = self.pcns[i], self.rcns[i]
            pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
            current_pcn_activations = pcn.place_cell_activations

            # Compute current reward at the agent's position
            rcn.update_reward_cell_activations(current_pcn_activations, visit=False)
            current_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))

            for d in range(self.n_hd):
                if distances_per_hd[d] < 0.5:
                    pot_rew[d] = 0.0
                else:
                    preplayed_pcn_activations = pcn.preplay(d)
                    rcn.update_reward_cell_activations(preplayed_pcn_activations, visit=False)
                    preplayed_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))
                    pot_rew[d] = preplayed_reward

            if pot_rew.max().item() >= reward_threshold:
                pot_rew_scales.append(pot_rew)
                valid_scale_indices.append(i)
                if show_detailed_debug:
                    print(f"Scale {i} ({scale_def['name']}): max_reward={pot_rew.max().item():.4f}")

        # 4.5) If NO valid scales remain, trigger exploration
        if len(pot_rew_scales) == 0:
            if show_debug:
                print("No valid scales found, forcing exploration")
            self.explore()
            return

        # 4.6) Reduce cooldown step count
        if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
            self.cooldown_steps_remaining -= 1
            if self.cooldown_steps_remaining == 0 and show_debug:
                print(f"Cooldown complete for scale {getattr(self, 'cooldown_scale_index', 'unknown')}")

        # 5) Normalize and blend across valid scales
        pot_rew_scales = torch.stack(pot_rew_scales)
        pot_rew_scales /= (pot_rew_scales.max(dim=1, keepdim=True)[0] + 1e-6)  # Avoid div by zero

        # Compute gradients only on valid scales
        grads = torch.sum(torch.abs(torch.diff(pot_rew_scales, dim=1)), dim=1)

        # Apply Gaussian smoothing (from original)
        import torch.nn.functional as F
        kernel_size = 3
        sigma = 3.0

        def gaussian_kernel(size: int, sigma: float, device):
            x = torch.arange(size, dtype=self.dtype, device=device) - size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
            return kernel.view(1, 1, -1)

        gaussian = gaussian_kernel(kernel_size, sigma, self.device)
        grads_unsq = grads.unsqueeze(0).unsqueeze(0)
        grads_smooth = F.conv1d(grads_unsq, gaussian, padding=kernel_size // 2).squeeze()

        # Compute initial mixing weights based on smoothed gradients
        mixing_weights = grads_smooth / (grads_smooth.sum() + 1e-6)
        mixing_weights = torch.clamp(mixing_weights, min=0.0, max=1.0)
        mixing_weights /= mixing_weights.sum()
        
        # Ensure mixing_weights is at least 1-dimensional for consistent indexing
        if mixing_weights.dim() == 0:
            mixing_weights = mixing_weights.unsqueeze(0)

        # If proximity modulation is enabled, adjust the weights
        if getattr(self, 'use_prox_mod', False) and hasattr(self, 'prox'):
            prox_weight = self.prox
            if show_detailed_debug:
                print(f"Proximity weight: {prox_weight:.3f}")
            
            scale_biases = torch.tensor(
                [1.0 / (i + 1) for i in range(len(valid_scale_indices))],
                dtype=self.dtype, device=self.device
            )
            scale_biases /= scale_biases.sum()
            
            # Ensure scale_biases has same dimension as mixing_weights
            if scale_biases.dim() == 0:
                scale_biases = scale_biases.unsqueeze(0)
                
            mixing_weights = (1 - prox_weight) * mixing_weights + prox_weight * scale_biases
            mixing_weights /= mixing_weights.sum()

        # Determine the preferred scale with hysteresis
        if mixing_weights.numel() > 1:
            preferred_scale_index = torch.argmax(mixing_weights).item()
            
            # Apply hysteresis only if we have multiple scales to choose from
            if (hasattr(self, "last_preferred_scale_index") and 
                self.last_preferred_scale_index is not None and
                self.last_preferred_scale_index < len(mixing_weights)):
                
                # Check if the difference between last preferred and new preferred is small
                last_weight = mixing_weights[self.last_preferred_scale_index].item()
                new_weight = mixing_weights[preferred_scale_index].item()
                if abs(last_weight - new_weight) < 0.1:
                    preferred_scale_index = self.last_preferred_scale_index
                    
        else:
            # Only one scale available
            preferred_scale_index = 0

        self.last_preferred_scale_index = preferred_scale_index

        # Update scale priority for logging
        if mixing_weights.numel() > 1:
            self.scale_idx = valid_scale_indices[torch.argmax(mixing_weights).item()]
        else:
            self.scale_idx = valid_scale_indices[0]

        if show_detailed_debug:
            print("Scale mixing weights:")
            if mixing_weights.numel() > 1:
                for i, (scale_idx, weight) in enumerate(zip(valid_scale_indices, mixing_weights)):
                    scale_name = self.scales[scale_idx]['name']
                    marker = " <- DOMINANT" if i == preferred_scale_index else ""
                    print(f"  Scale {scale_idx} ({scale_name}): {weight:.3f}{marker}")
            else:
                # Single scale case
                scale_idx = valid_scale_indices[0]
                scale_name = self.scales[scale_idx]['name']
                print(f"  Single scale: {scale_idx} ({scale_name}): 1.000 <- ONLY")

        # Combine rewards across scales
        if mixing_weights.numel() > 1:
            combined_pot_rew = torch.sum(mixing_weights[:, None] * pot_rew_scales, dim=0)
        else:
            combined_pot_rew = pot_rew_scales.squeeze(0)

        # 6) Compute action heading
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * combined_pot_rew)
        cos_component = torch.sum(torch.cos(angles) * combined_pot_rew)
        action_angle = torch.atan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        if show_debug:
            best_direction = torch.argmax(combined_pot_rew).item()
            best_reward = combined_pot_rew[best_direction].item()
            print(f"Multi-scale decision: target={self.action_heading_deg:.0f}°, best_dir={best_direction} ({best_direction*45}°), reward={best_reward:.4f}")

        # 7) Execute movement with improved turning
        self._execute_movement(self.action_heading_deg, show_debug)

        # 8) Optional TD Learning Step
        if getattr(self, 'td_learning', False):
            for i, scale_def in enumerate(self.scales):
                pcn, rcn = self.pcns[i], self.rcns[i]
                new_pcn_activations = pcn.place_cell_activations
                rcn.update_reward_cell_activations(new_pcn_activations, visit=False)
                observed_reward = float(rcn.reward_cell_activations.item())
                rcn.td_update(old_pcn_activations[i], observed_reward)

        if show_detailed_debug:
            print("="*70)
            print("EXPLOIT V0 - STEP END")
            print("="*70 + "\n")

        return

    def exploit_locations(self):
        """Goal-specific exploitation using existing exploit_v2 logic"""
        if not hasattr(self, 'active_goal_name'):
            mode_name = "EXPLOIT_LOCATIONS_RANDOM" if self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM else "EXPLOIT_LOCATIONS"
            print(f"[ERROR] No active goal set for {mode_name} mode")
            self.done = True
            return

        # Use existing exploit_v2 logic - RCNs are already loaded for the target goal
        self.exploit_v2()

    def _sense_and_compute(self):
        """Shared helper: Sense environment, compute PCN activations, update hmaps, check goal."""
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True, update_scale_priority=True)
        self.check_goal_reached_unified()
    
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

    def apply_directional_momentum(self, directional_rewards):
        """
        Apply directional momentum to 8-directional reward values.
        
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

    def exploit_v2(self):
        """
        Enhanced multiscale exploitation with goal cell boosting for EXPLOIT_LOCATIONS mode.
        
        Key improvements over v15:
        - Detects when multi-step preplay activates goal-associated place cells
        - Applies scale-dependent boost to directions that activate goal cells
        - Smaller scales (more accurate) receive larger boosts
        - Maintains all existing gradient-based multi-scale integration
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
                scale_max_steps = self.SCALE_MULTISTEP_STEPS.get(scale_name, self.MULTISTEP_STEPS)  # Fallback to default
                
                if show_detailed_debug:
                    print(f"  Using {scale_max_steps} preplay steps for scale {scale_name}")
                
                # Use new goal-boosted computation method with scale-specific steps
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
        
        # 6) Relative gradient weighting (same as v15)
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
    
    def exploit_v3(self):
      
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
        
        # 3) Compute obstacle distances
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries) // 2)
        num_points_per_hd = len(boundaries_rolled) // self.n_hd

        self.distances_per_hd = torch.tensor([
            torch.min(boundaries_rolled[i * num_points_per_hd: (i + 1) * num_points_per_hd])
            for i in range(self.n_hd)
        ], device=self.device, dtype=self.dtype)

      

        # 4) Multi-step preplay on ALL scales with goal cell boosting
        all_scale_rewards = []
        all_scale_gradients = []

        for scale_idx, scale_def in enumerate(self.scales):
            try:
                
                # Get scale-specific multistep value
                scale_name = scale_def.get('name', 'unknown')
                scale_max_steps = self.SCALE_MULTISTEP_STEPS.get(scale_name, self.MULTISTEP_STEPS)  # Fallback to default
                
                # Use new goal-boosted computation method with scale-specific steps
                multi_step_rewards = self.compute_multi_step_rewards_with_goal_boost(
                    scale_idx, active_goal, False, max_steps=scale_max_steps
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
                
            except Exception as e:
                
                zero_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
                all_scale_rewards.append(zero_rewards)
                all_scale_gradients.append(0.0)

        # 5) Handle all-zero gradients case
        total_gradient = sum(all_scale_gradients)
        if total_gradient < 1e-6:
            self.explore()
            return
        
        # 6) Relative gradient weighting (same as v15)
        mixing_weights = torch.tensor(all_scale_gradients, dtype=self.dtype, device=self.device) / total_gradient

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
        
        # 11) Update momentum tracking
        self.update_momentum_tracking(chosen_direction, chosen_direction_reward)

        # 12) Execute movement
        self._execute_movement(self.action_heading_deg, show_debug)

        return

    def compute_multi_step_rewards_with_goal_boost(self, scale_idx, active_goal, debug=False, decay_factor=0.6, max_steps=None):
        """
        Compute multi-step rewards with goal cell boosting for EXPLOIT_LOCATIONS mode.
        
        When multi-step preplay activates place cells associated with the target goal,
        those directions receive a boost. Smaller scales (more accurate) get larger boosts.
        
        Args:
            scale_idx: Index of the scale to evaluate
            active_goal: Dictionary containing active goal information (name, location, etc.)
            debug: Whether to print debug information
            decay_factor: Exponential decay for step weighting (default: 0.6)
            max_steps: Number of preplay steps (if None, uses self.MULTISTEP_STEPS for backward compatibility)
            
        Returns:
            torch.Tensor: Multi-step weighted rewards with goal cell boosts for each direction [8]
        """
        # Use provided max_steps or fall back to default for backward compatibility
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
                    # Get base reward using weighted multi-step preplay with scale-specific steps
                    base_reward = pcn.multi_step_preplay_constrained_weighted(
                        forced_first_direction=direction,
                        max_steps=max_steps,  # Use scale-specific value
                        decay_factor=decay_factor,
                        debug=False
                    )
                    
                    # Apply goal cell boost if applicable
                    boosted_reward = self.apply_goal_cell_boost(
                        base_reward, direction, scale_idx, active_goal, pcn, debug
                    )
                    
                    multi_step_rewards[direction] = boosted_reward
                    
            if debug:
                print(f"    Scale {scale_idx} multi-step rewards (steps={max_steps}): {multi_step_rewards}")
                        
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
        
        Checks if multi-step preplay in the given direction activates place cells
        associated with the target goal. If so, applies a scale-dependent boost.
        
        Args:
            base_reward: Base reward value from multi-step preplay
            direction: Direction index (0-7) being evaluated
            scale_idx: Index of current scale
            active_goal: Active goal dictionary with name and location
            pcn: Place cell network for this scale
            debug: Whether to print debug information
            
        Returns:
            float: Boosted reward value
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
        
        # Perform multi-step preplay in this direction and check goal cell activation
        final_activations = pcn.multi_step_preplay_in_direction_final_state(
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

    def compute_goal_cell_boost_factor(self, scale_idx, goal_cell_activation):
        """
        Compute boost factor based on scale accuracy and goal cell activation strength.
        
        Smaller scales (more accurate) get larger boosts. Boost is also proportional
        to goal cell activation strength.
        
        Args:
            scale_idx: Index of current scale (smaller index = smaller scale = more accurate)
            goal_cell_activation: Activation level of goal-associated place cell [0, 1]
            
        Returns:
            float: Boost factor (>= 1.0)
        """
        # Base boost factors per scale (smaller scales get larger boosts)
        scale_base_boosts = {
            0: 3.0,   # small scale - highest accuracy, largest boost
            1: 2.5,   # medium scale  
            2: 2.0,   # large scale
            3: 1.5,   # xlarge scale - lowest accuracy, smallest boost
        }
        
        # Get base boost for this scale (default to 1.5 for unknown scales)
        base_boost = scale_base_boosts.get(scale_idx, 1.5)
        
        # Scale boost by activation strength (stronger activation = larger boost)
        # Activation range [0, 1] maps to boost multiplier range [1.0, base_boost]
        activation_multiplier = 1.0 + (base_boost - 1.0) * goal_cell_activation
        
        return activation_multiplier
    
    ############################################# SENSE ############################################
    
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
    
    ############################################ COMPUTE ###########################################
    
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

    ###################################### CHECK GOAL REACHED ######################################
    
    def check_goal_reached_unified(self):
        """Unified goal checking for all modes"""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        current_position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)

        # Update distance tracking for random spawn mode
        self._update_distance_tracking()

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

            elif self.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE:
                # Update coverage tracking
                robot_pos = [curr_pos[0], curr_pos[2]]  # [x, z] coordinates
                self._update_coverage(robot_pos)

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
                    self._create_multi_goal_reward_maps()
                    self._save_multi_goal_data()
                    self.stop()
                    self.save(include_pcn=True, include_rcn=True, include_gcn=True, include_hmaps=True)
                    self.done = True
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                elif coverage_reached and not learning_complete:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Coverage target reached ({self.current_coverage_percentage*100:.1f}%) "
                          f"but learning incomplete, continuing...")
                elif minimum_time_reached and not learning_complete:
                    print(f"[LEARN_LOCATIONS_COVERAGE] Time limit reached but learning incomplete, continuing...")

            elif self.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM:
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
        
        if self.td_learning:
            for pcn, rcn in zip(self.pcns, self.rcns):
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=False)
            self.save(include_pcn=True, include_rcn=True, include_gcn=True, save_trajectory=True)
        else:
            self.save(save_trajectory=True)
        
        self.done = True

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

    ######################################### AUTO PILOT ############################################
    
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
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"]
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]

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
            s_start += 1

    ####################################### HELPER METHODS ##########################################

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
                    torch.zeros((self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32)
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
                        (self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32
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
                        (self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32
                    )
                    
                # Store activations directly
                self.hmap_gcn_activities[mapped_index][self.step_count] = act

        # Increment step count
        self.step_count += 1

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to goal(s).
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
            # Single goal mode (original logic)
            goal = self.goals[0]
            goal_position = torch.tensor(goal["location"], dtype=self.dtype, device=self.device)
            distance = torch.norm(current_position - goal_position)
            
            if distance <= goal["radius"]:
                return 1.0
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
            if include_pcn:
                for scale_def, pcn in zip(self.scales, self.pcns):
                    # Get adaptive learning logs from PCN
                    adaptive_logs = pcn.get_adaptive_learning_logs()
                    
                    if adaptive_logs is not None:
                        scale_idx = scale_def["scale_index"]
                        adaptive_log_path = os.path.join(self.hmap_dir, f"adaptive_learning_scale_{scale_idx}.pkl")
                        
                        with open(adaptive_log_path, "wb") as f:
                            pickle.dump(adaptive_logs, f)
                        files_saved.append(adaptive_log_path)
                        
                        print(f"[DRIVER] Saved adaptive learning logs for scale {scale_idx}: {adaptive_logs['total_logged_steps']} steps")
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
            # 3) Save each scale's GCN (if requested)
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
            # 4) Save the history maps if requested
            # ----------------------------------------------------------------------
            if include_hmaps:
                # (a) Agent location
                hmap_loc_path = os.path.join(self.hmap_dir, "hmap_loc.pkl")
                with open(hmap_loc_path, "wb") as f:
                    pickle.dump(self.hmap_loc[: self.step_count], f)
                files_saved.append(hmap_loc_path)

                # (b) Head direction history
                hmap_hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
                with open(hmap_hdn_path, "wb") as f:
                    pickle.dump(self.hmap_hdn[: self.step_count].cpu(), f)
                files_saved.append(hmap_hdn_path)

                # (c) Place-cell history maps for each scale
                for scale_def, pc_history in zip(self.scales, self.hmap_pcn_activities):
                    scale_idx = scale_def["scale_index"]  # Get correct scale index
                    hmap_scale_path = os.path.join(self.hmap_dir, f"hmap_pcn_scale_{scale_idx}.pkl")
                    
                    with open(hmap_scale_path, "wb") as f:
                        pc_data = pc_history[: self.step_count].cpu().numpy()
                        pickle.dump(pc_data, f)
                    files_saved.append(hmap_scale_path)
                    
                # (d) Grid-cell history maps for each scale
                for scale_def, gc_history in zip(self.scales, self.hmap_gcn_activities):
                    if gc_history.numel() > 0:  # Only save if there are grid cells
                        scale_idx = scale_def["scale_index"]
                        hmap_scale_path = os.path.join(self.hmap_dir, f"hmap_gcn_scale_{scale_idx}.pkl")
                        
                        with open(hmap_scale_path, "wb") as f:
                            gc_data = gc_history[: self.step_count].cpu().numpy()
                            pickle.dump(gc_data, f)
                        files_saved.append(hmap_scale_path)

                # (e) Prox values
                if hasattr(self, "hmap_prox"):
                    hmap_prox_path = os.path.join(self.hmap_dir, "hmap_prox.pkl")
                    with open(hmap_prox_path, "wb") as f:
                        prox_data = self.hmap_prox[: self.step_count].cpu().numpy()
                        pickle.dump(prox_data, f)
                    files_saved.append(hmap_prox_path)

            # ----------------------------------------------------------------------
            # 5) Save the agent's path if requested
            # ----------------------------------------------------------------------
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

            # ----------------------------------------------------------------------
            # 6) Print saved files
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

    def _save_goal_rcns(self, goal_name):
        """Save current RCNs as goal-specific RCNs"""
        multi_goal_dir = os.path.join(self.network_dir, "multi_goal_rewards")
        os.makedirs(multi_goal_dir, exist_ok=True)
        
        saved_count = 0
        for i, scale_def in enumerate(self.scales):
            scale_idx = scale_def["scale_index"]
            goal_rcn_path = os.path.join(
                multi_goal_dir, 
                f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
            )
            
            with open(goal_rcn_path, "wb") as f:
                pickle.dump(self.rcns[i], f)
            saved_count += 1
        
        print(f"[DRIVER] Saved {saved_count} goal-specific RCNs for {goal_name}")
    
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

            self.stats_collector.finalize_and_save()

        # Save hmaps for this trial
        self.save(include_hmaps=True)

        self.done = True
        # Don't pause simulation - let trial loop continue

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

            self.stats_collector.finalize_and_save()

        # Save hmaps for this trial
        self.save(include_hmaps=True)

        self.done = True
        # Don't pause simulation - let trial loop continue