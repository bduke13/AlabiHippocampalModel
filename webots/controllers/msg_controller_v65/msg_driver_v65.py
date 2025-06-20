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

# Add root directory to PYTHONPATH
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.msg_pcn_v65 import MultiscalePlaceCellWithGridV65
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer_v3_normalized import RewardCellLayerV3
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
        goal_location: Optional[List[float]] = None,
        max_dist: Optional[float] = None,
        plot_bvc: Optional[bool] = False,
        td_learning: Optional[bool] = False,
        use_prox_mod: Optional[bool] = False,
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
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Visitation map parameters
        self.grid_resolution = 0.5  # Cell size in meters
        self.decay_rate = 0.99  # Decay rate for visitation counts
        
        # Initialize visitation maps
        self.visitation_map = {}  # For exploration decisions (with decay)
        self.visitation_map_metrics = {}  # For metrics only (no decay)
        
        # Stuck detection parameters
        self.stuck_steps_count = 0
        self.last_coverage_count = 0
        self.stuck_threshold_steps = 200  # Steps with no coverage increase => 'stuck'

        # Stats / trial info
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        # Head direction layer size
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 16
        self.max_dist = max_dist
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756

        # Simulation run time
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))

        # Exploration/exploitation radius
        self.goal_r = {"explore": 1.0, "exploit": 1.0}
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
                dist_to_goal = np.sqrt(
                    (candidate[0] - self.goal_location[0]) ** 2 +
                    (candidate[2] - self.goal_location[1]) ** 2
                )
                if dist_to_goal >= 1.0:
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

        # Load or init PCNs / RCNs / GCNs
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
        self.plot_bvc = plot_bvc
        self.manual_explore = False

        # Step once
        self.step(self.timestep)

    ##########################################################################
    #                        PCN / RCN / GCN LOADING                         #
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
            
            if pcn_class_name == "MultiscalePlaceCellWithGridV65":
                print(f"[DRIVER] PCN is V6.5 - updating parameters")
                # Update all parameters from scale definition for V6.5
                if enable_ojas is not None:
                    pcn.enable_ojas = enable_ojas
                if enable_stdp is not None:
                    pcn.enable_stdp = enable_stdp
                    
                # Update V6 parameters
                pcn.stdp_learning_rate = scale_def.get("stdp_learning_rate", 0.05)
                pcn.tau_hd = scale_def.get("tau_hd", 0.1)
                
                # Update V5 parameters
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
                
                # Update V6.5 adaptive learning parameters
                pcn.enable_adaptive_stdp = scale_def.get("enable_adaptive_stdp", False)
                pcn.adaptive_initial_lr = scale_def.get("adaptive_initial_lr", 0.15)
                pcn.adaptive_final_lr = scale_def.get("adaptive_final_lr", 0.02)
                pcn.adaptive_decay_rate = scale_def.get("adaptive_decay_rate", 3.0)
                
                print(f"[DRIVER] Updated V6.5 PCN with adaptive STDP: {pcn.enable_adaptive_stdp}")
                if pcn.enable_adaptive_stdp:
                    print(f"[DRIVER] Adaptive learning: {pcn.adaptive_initial_lr} → {pcn.adaptive_final_lr} (decay: {pcn.adaptive_decay_rate})")
                
                return pcn
                
            elif pcn_class_name == "MultiscalePlaceCellWithGridV6":
                print(f"[DRIVER] PCN is V6 - upgrading to V6.5")
                # For V6 PCNs, create new V6.5 and transfer weights
                new_pcn = self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)
                
                # Transfer learned weights from V6 to V6.5
                if hasattr(pcn, 'w_in'):
                    new_pcn.w_in.data = pcn.w_in.data.clone()
                if hasattr(pcn, 'w_grid') and pcn.w_grid is not None:
                    new_pcn.w_grid.data = pcn.w_grid.data.clone()
                if hasattr(pcn, 'w_rec_tripartite'):
                    new_pcn.w_rec_tripartite.data = pcn.w_rec_tripartite.data.clone()
                    
                print(f"[DRIVER] Successfully upgraded V6 PCN to V6.5 with weight transfer")
                return new_pcn
                
            elif pcn_class_name in ["MultiscalePlaceCellWithGridV42", "MultiscalePlaceCellWithGridV4"]:
                print(f"[DRIVER] PCN is {pcn_class_name} - upgrading to V6.5")
                return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)
            
            else:
                print(f"[DRIVER] Unknown PCN version {pcn_class_name} - creating new V6.5 PCN")
                return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[DRIVER] Could not load PCN from {path}: {e}")
            return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)


    def _initialize_new_pcn(self, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        """Initialize a new place cell network V6.5 with adaptive learning rates."""
        print(f"[DRIVER] Initializing new PCN V6.5 for scale {scale_def['scale_index']}")
        
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
        # V6.5 adaptive learning parameters
        enable_adaptive_stdp = scale_def.get("enable_adaptive_stdp", False)
        adaptive_initial_lr = scale_def.get("adaptive_initial_lr", 0.15)
        adaptive_final_lr = scale_def.get("adaptive_final_lr", 0.02)
        adaptive_decay_rate = scale_def.get("adaptive_decay_rate", 3.0)
        
        # V6.5 logging and debugging parameters
        enable_debug_prints = scale_def.get("enable_debug_prints", False)
        enable_adaptive_logging = scale_def.get("enable_adaptive_logging", False)
        
        # V6 parameters
        stdp_learning_rate = scale_def.get("stdp_learning_rate", 0.05)
        tau_hd = scale_def.get("tau_hd", 0.1)
        
        # V5 parameters
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
        
        # Create V6.5 place cell layer with adaptive learning
        pcn = MultiscalePlaceCellWithGridV65(
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
            # V6.5 NEW: Adaptive learning parameters
            enable_adaptive_stdp=enable_adaptive_stdp,
            adaptive_initial_lr=adaptive_initial_lr,
            adaptive_final_lr=adaptive_final_lr,
            adaptive_decay_rate=adaptive_decay_rate,
            # V6.5 NEW: Scale name for debugging
            scale_name=scale_def.get("name", "unknown"),
            # V6.5 NEW: Logging and debugging parameters
            enable_debug_prints=enable_debug_prints,
            enable_adaptive_logging=enable_adaptive_logging,
            device=self.device,
        )
        
        print(f"[DRIVER] Successfully created new V6.5 PCN with:")
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
            
            # Update replay parameters for loaded RCNs
            if hasattr(rcn, 'replay_timesteps'):
                rcn.replay_timesteps = replay_timesteps
                rcn.replay_decay_constant = replay_decay_constant
                print(f"[DRIVER] Updated replay parameters: timesteps={replay_timesteps}, decay={replay_decay_constant}")
            else:
                # Convert old RCN to new format by replacing with new instance
                print(f"[DRIVER] Converting old RCN to V2 format")
                old_weights = rcn.w_in_effective.clone()
                rcn = RewardCellLayerV3(
                    num_place_cells=scale_def["num_pc"],
                    num_replay=3,
                    learning_rate=learning_rate,
                    replay_timesteps=replay_timesteps,
                    replay_decay_constant=replay_decay_constant,
                    device=self.device,
                )
                rcn.w_in_effective = old_weights
                rcn.w_in = old_weights.clone()
                
        except:
            print(f"[DRIVER] Initializing new RCN V2 for {path} with learning rate {learning_rate}")
            print(f"[DRIVER] Replay params: timesteps={replay_timesteps}, decay={replay_decay_constant}")
            rcn = RewardCellLayerV3(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                replay_timesteps=replay_timesteps,
                replay_decay_constant=replay_decay_constant,
                device=self.device,
            )
        return rcn

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

    ########################################### RUN LOOP ###########################################

    def run(self):
        print(f"[DRIVER] Starting robot in {self.robot_mode}")
        print(f"[DRIVER] Goal at {self.goal_location}")
        
        while not self.done:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode in (RobotMode.LEARN_OJAS, 
                                     RobotMode.LEARN_HEBB, 
                                     RobotMode.DMTP, 
                                     RobotMode.PLOTTING):
                self.explore()
            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()
            else:
                print("Unknown state. Exiting...")
                break

    ########################################### EXPLORE ###########################################
    def intelligent_curiosity_explore(self):
        """
        Enhanced exploration method that follows the EXACT same structure as the original explore()
        but uses curiosity-based movement decisions instead of just moving forward.
        """
        
        # Update visitation tracking at the start of each exploration cycle
        self.update_visitation_map()
        self.decay_visitation_counts()
        
        # Check coverage progress for stuck detection
        coverage_count = len(self.visitation_map)
        if coverage_count > self.last_coverage_count:
            self.stuck_steps_count = 0
            self.last_coverage_count = coverage_count
        else:
            self.stuck_steps_count += 1
        
        # If stuck, do 360° scan before the main loop
        if self.stuck_steps_count >= self.stuck_threshold_steps:
            print("[CURIOSITY] Stuck detected. Attempting 360° scan for exit.")
            success = self.scan_for_exit(step_deg=15, threshold=0.75)
            if success:
                self.stuck_steps_count = 0
                print("[CURIOSITY] Found exit via 360° scan")
            else:
                print("[CURIOSITY] No clear exit found. Random turn fallback.")
                self.turn(np.random.uniform(-np.pi, np.pi))
                self.stuck_steps_count = 0
            return  # Skip the main exploration loop this time

        for _ in range(self.tau_w):
            if self.done:
                break
            
            # 1) Sense environment (SAME AS ORIGINAL)
            self.sense()

            # 2) Compute place cell activations (SAME AS ORIGINAL)
            self.compute_pcn_activations()

            # 3) If in DMTP or EXPLOIT mode, update reward cell activations (SAME AS ORIGINAL)
            if self.robot_mode in {RobotMode.DMTP, RobotMode.EXPLOIT}:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)

            # 4) If a collision is detected, turn away and break (SAME AS ORIGINAL)
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break
            
            # 5) Check goal, update hmaps (SAME AS ORIGINAL)
            self.check_goal_reached()
            self.update_hmaps(
                update_loc=True, 
                update_pcn=True,
                update_gcn=True,
                update_scale_priority=self.robot_mode == RobotMode.EXPLOIT, 
                update_prox=(self.use_prox_mod and self.robot_mode == RobotMode.LEARN_OJAS)
            )
            
            # 6) ONLY CHANGE: Use curiosity-based movement instead of just forward()
            self.curiosity_forward()

        # A small random turn at the end (SAME AS ORIGINAL)
        if self.manual_explore:
            self.manual_control()
        else:
            self.turn(np.random.normal(0, np.deg2rad(30)))

    def curiosity_forward(self):
        """
        Replacement for self.forward() that uses curiosity-based direction selection.
        This is the ONLY change from the original explore() method.
        """
        # Compute novelty scores for different directions
        directions, scores = self.compute_novelty_scores()
        total_score = sum(scores)
        
        if total_score <= 1e-9:
            # All directions blocked, just move forward (fallback to original behavior)
            print("[CURIOSITY] All directions blocked, moving forward anyway")
            self.forward()
            return
        
        # Find best direction
        best_idx = int(np.argmax(scores))
        chosen_dir = directions[best_idx]
        
        # Optional debug output (remove if too verbose)
        #if self.step_count % 10 == 0:  # Print every 10 steps
            #print(f"[CURIOSITY] Step {self.step_count}: {chosen_dir} chosen (scores: {[f'{d}:{s:.2f}' for d, s in zip(directions, scores)])")
        
        # Execute the chosen movement
        if chosen_dir == "forward":
            self.forward()
        elif chosen_dir == "left":
            self.turn(np.deg2rad(45))
            self.forward()
        elif chosen_dir == "right":
            self.turn(np.deg2rad(-45))
            self.forward()

    def compute_novelty_scores(self):
        """
        Computes novelty scores for forward, left, and right directions.
        FIXED: boundaries are already rotated in sense(), so don't add current_heading_deg again!
        """
        directions = ["forward", "left", "right"]
        scores = []
        
        for d in directions:
            # 1) Predict position if we move in direction d
            predicted_pos = self.predict_position(d)
            if predicted_pos is None:
                scores.append(0.0)
                continue
            
            # 2) Compute novelty from visitation map
            x_cell = int(predicted_pos[0] / self.grid_resolution)
            y_cell = int(predicted_pos[1] / self.grid_resolution)
            cell = (x_cell, y_cell)
            
            visits = self.visitation_map.get(cell, 0)
            novelty = 1.0 / (visits + 1.0)
            
            # 3) FIXED: Check LiDAR for obstacles in direction d
            # boundaries are already rotated, so index 0 = forward, no need to add current_heading_deg
            if d == "forward":
                idx = 0  # Forward direction
            elif d == "left":
                idx = int(45 * 2)  # 45 degrees left = index 90
            elif d == "right":
                idx = int(-45 * 2) % 720  # 45 degrees right = index 630 (720-90)
            
            # Get distance to obstacle (handle tensor shape properly)
            if self.boundaries.dim() > 1:
                dist_to_obstacle = self.boundaries.flatten()[idx].item()
            else:
                dist_to_obstacle = self.boundaries[idx].item()
            
            # Handle infinite distances
            if np.isinf(dist_to_obstacle) or dist_to_obstacle > 30:
                dist_to_obstacle = 30.0
            
            # 4) Apply obstacle penalty if too close
            obstacle_threshold = 0.5  # Increased threshold for better avoidance
            if dist_to_obstacle < obstacle_threshold:
                # Stronger penalty for close obstacles
                penalty_factor = max(0.05, (dist_to_obstacle / obstacle_threshold) ** 2)
            else:
                penalty_factor = 1.0
            
            final_score = novelty * penalty_factor
            scores.append(final_score)
            
            # Debug output every few steps
            if self.step_count % 20 == 0:
                print(f"[NOVELTY] {d}: dist={dist_to_obstacle:.2f}, novelty={novelty:.3f}, penalty={penalty_factor:.3f}, score={final_score:.3f}")
        
        return directions, scores

    def predict_position(self, direction):
        """
        Predict (x,z) position if we move in given direction.
        """
        step_distance = 0.5  # How far we move
        current_pos = self.robot.getField("translation").getSFVec3f()
        heading_deg = self.current_heading_deg
        
        # Adjust heading based on direction
        if direction == "left":
            heading_deg += 45
        elif direction == "right":
            heading_deg -= 45
        
        heading_rad = np.deg2rad(heading_deg)
        new_x = current_pos[0] + step_distance * np.cos(heading_rad)
        new_z = current_pos[2] + step_distance * np.sin(heading_rad)
        
        return (new_x, new_z)

    def update_visitation_map(self):
        """Update the visitation map with current cell based on robot's position."""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        x_cell = int(curr_pos[0] / self.grid_resolution)
        y_cell = int(curr_pos[2] / self.grid_resolution)
        cell = (x_cell, y_cell)
        
        # Update exploration map (with decay)
        self.visitation_map[cell] = self.visitation_map.get(cell, 0) + 1
        
        # Update metrics map (no decay)
        self.visitation_map_metrics[cell] = self.visitation_map_metrics.get(cell, 0) + 1

    def decay_visitation_counts(self):
        """Decay visitation counts to allow areas to become novel again over time."""
        for cell in self.visitation_map:
            self.visitation_map[cell] *= self.decay_rate

    def scan_for_exit(self, step_deg=15, threshold=0.75):
        """
        Rotate in increments, sense LiDAR, find best direction.
        FIXED: Just check forward direction at each orientation since boundaries are pre-rotated.
        """
        original_heading = self.current_heading_deg
        best_dist = 0.0
        best_angle = None
        
        for angle in range(0, 360, step_deg):
            target_angle_deg = (original_heading + angle) % 360
            delta = np.deg2rad(target_angle_deg - self.current_heading_deg)
            self.turn(delta)
            self.sense()  # This rotates boundaries so index 0 = forward
            
            # Check forward direction (index 0) since boundaries are already rotated
            if self.boundaries.dim() > 1:
                dist = self.boundaries.flatten()[0].item()
            else:
                dist = self.boundaries[0].item()
            
            if dist > best_dist:
                best_dist = dist
                best_angle = self.current_heading_deg
        
        if best_angle is not None and best_dist > threshold:
            angle_diff = np.deg2rad(best_angle - self.current_heading_deg)
            self.turn(angle_diff)
            print(f"[SCAN] Best angle: {best_angle:.2f}°, distance: {best_dist:.2f}")
            self.forward()
            return True
        else:
            return False
        


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

            # 3) If in DMTP or EXPLOIT mode, update reward cell activations
            if self.robot_mode in {RobotMode.DMTP, RobotMode.EXPLOIT}:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)

            # 4) If a collision is detected, turn away and break
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break
            
            # 5) Check goal, update hmaps, move forward
            self.check_goal_reached()
            self.update_hmaps(
                update_loc=True, 
                update_pcn=True,
                update_gcn=True,
                update_scale_priority=self.robot_mode == RobotMode.EXPLOIT, 
                update_prox=(self.use_prox_mod and self.robot_mode == RobotMode.LEARN_OJAS)
            )
            self.forward()

        # A small random turn at the end if enabled
        if self.manual_explore:
            self.manual_control()
        else:
            self.turn(np.random.normal(0, np.deg2rad(30)))

    ########################################### EXPLOIT ###########################################
    



    def exploit(self):
        """
        Follows the reward gradient to reach the goal location using multiscale place field navigation.
        If a scale has insufficient reward information, it is temporarily removed from the decision-making 
        while still allowing learning.
        """
        debug_exploit = True

        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps(update_loc=True, update_pcn=True, update_gcn=True, update_scale_priority=True)
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
                if debug_exploit:
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
                if debug_exploit:
                    print(f"Detected {self.LOOP_THRESHOLD} consecutive loops within {self.MAX_STEPS_BETWEEN_LOOPS} steps. Initiating forced exploration.")

                # Store the preferred scale before entering forced exploration
                if hasattr(self, "last_preferred_scale_index"):
                    self.cooldown_scale_index = self.last_preferred_scale_index
                    self.cooldown_steps_remaining = 20  # Cooldown for 10 steps
                    if debug_exploit:
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
                    if debug_exploit:
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
            if debug_exploit:
                print(f"[WARNING] No valid scales found (all below reward threshold {reward_threshold}), forcing exploration.")
            self.explore()
            return

        #-------------------------------------------------------------------
        # 4.5) Reduce cooldown step count
        #-------------------------------------------------------------------
        if hasattr(self, "cooldown_steps_remaining") and self.cooldown_steps_remaining > 0:
            self.cooldown_steps_remaining -= 1
            if self.cooldown_steps_remaining == 0:
                if debug_exploit:
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
            if debug_exploit:
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
            if debug_exploit:
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
                              update_gcn=True,
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

    def plot_current_heading(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        current_heading_rad = (3 * np.pi / 2) - np.deg2rad(self.current_heading_deg)
        action_heading_rad = (3 * np.pi / 2) - np.deg2rad(self.action_heading_deg) if hasattr(self, 'action_heading_deg') else None
        cancelled_rad_list = [(3 * np.pi / 2) - np.deg2rad(deg) for deg in getattr(self, 'cancelled_angles_deg', [])]

        angles = np.linspace(0, 2 * np.pi, len(self.boundaries), endpoint=False)
        ax.plot((angles + np.pi) % (2 * np.pi), self.boundaries.cpu().numpy().flatten(), 'b-', label="Corrected LiDAR Readings")
        ax.quiver(current_heading_rad, 0, np.cos(current_heading_rad), np.sin(current_heading_rad), scale=1, scale_units='inches', color='r', label="Current Heading")
        if action_heading_rad is not None:
            ax.quiver(action_heading_rad, 0, np.cos(action_heading_rad), np.sin(action_heading_rad), scale=1, scale_units='inches', color='g', label="Action Heading")
        if cancelled_rad_list:
            ax.scatter(cancelled_rad_list, [4.5] * len(cancelled_rad_list), c='k', marker='x', s=50, label='Cancelled Angles')

        ax.set_theta_direction(-1)
        ax.set_theta_offset(-np.pi / 2)
        ax.set_rmax(5)
        ax.set_rticks([1, 2, 3, 4, 5])
        ax.set_rlabel_position(-22.5)
        ax.grid(True)
        ax.text(current_heading_rad, 5, f'Heading: {self.current_heading_deg}°', horizontalalignment='center', verticalalignment='bottom', color='red')
        if action_heading_rad is not None:
            ax.text(action_heading_rad, 5, f'Action: {self.action_heading_deg}°', horizontalalignment='center', verticalalignment='bottom', color='green')

        plt.show()

    ########################################### SENSE ###########################################
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
    
    ########################################### COMPUTE ###########################################
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
    
    ########################################### COMPUTE ###########################################
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

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        time_limit = 180 # minutes

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
            self.auto_pilot()
            self.stop()
            for pcn, rcn in zip(self.pcns, self.rcns):
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                rcn.replay(pcn=pcn)
            self.save(include_pcn=True, include_rcn=True, include_gcn=True,)
            self.done = True
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            return
        
        elif self.robot_mode == RobotMode.EXPLOIT and self.step_count > 10:
            # Check if either goal reached or time expired
            goal_reached = torch.allclose(
                torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
                torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
                atol=self.goal_r["exploit"]
            )
            time_expired = self.getTime() >= 30 * time_limit
            if time_expired:
                #print("Time limit reached. Trial unsuccessful.")
                #self.stop()
                #self.done = True
                #return
                print("Time limit reached. Trial unsuccessful.")
    
                if self.stats_collector:
                    # Update and save stats for unsuccessful trials too
                    self.stats_collector.update_stat("trial_id", self.trial_id)
                    self.stats_collector.update_stat("start_location", self.start_loc)
                    self.stats_collector.update_stat("goal_location", self.goal_location)
                    self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
                    self.stats_collector.update_stat("total_time_secs", round(self.getTime(), 2))
                    self.stats_collector.update_stat("success", False)  # Explicitly mark as unsuccessful
                    self.stats_collector.save_stats(self.trial_id)
                    
                    print(f"Trial {self.trial_id} timed out.")
                    print(f"Start location: {self.start_loc}")
                    print(f"Goal location: {self.goal_location}")
                    print(f"Total distance traveled: {round(self.compute_path_length(), 2)} meters.")
                    print(f"Total time taken: {round(self.getTime(), 2)} seconds.")
                
                self.stop()
                
                # Still save network data if TD learning is enabled
                if self.td_learning:
                    for pcn, rcn in zip(self.pcns, self.rcns):
                        rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=False)
                    self.save(include_pcn=True, include_rcn=True, include_gcn=True, save_trajectory=True)
                else:
                    self.save(save_trajectory=True)  # At minimum, save the trajectory
                    
                self.done = True
                return
                
            if goal_reached:
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
                    print(f"Total distance traveled: {round(self.compute_path_length(), 2)} meters.")
                    print(f"Total time taken: {round(self.getTime(), 2)} seconds.")
                    print(f"Success: {self.getTime() <= time_limit * 60}")
                    self.done = True
                    self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                    return

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
                    curr_pos[2] - self.goal_location[1]
                ],
                dtype=self.dtype,
                device=self.device
            )
        )

        # Determine the correct goal radius based on the current mode
        if self.mode == RobotMode.EXPLOIT:
            goal_radius = self.goal_r["exploit"]
        else:  # Default to "explore" goal radius for all other modes
            goal_radius = self.goal_r["explore"]

        # Return 1.0 reward if within goal radius, else 0.0
        if distance_to_goal <= goal_radius:
            return 1.0 # Goal reached
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

            print("[DRIVER] Finished clearing old scale PCNs, RCNs, GCNs, and hmap files.")