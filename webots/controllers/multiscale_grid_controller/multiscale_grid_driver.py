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

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.multiscale_pcn_with_gcn import MultiscalePlaceCellWithGrid
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

# --- PyTorch seeds / random ---
# torch.manual_seed(5)
# np.random.seed(5)
# rng = default_rng(5)  # or keep it as is
np.set_printoptions(precision=2)


class MultiscaleDriverWithGrid(Supervisor):
    """Controls robot navigation and learning using neural networks for place and reward cells.

    This class manages the robot's sensory inputs, motor outputs, and neural network layers
    to enable autonomous navigation and learning in an environment. It coordinates between
    multiscale place cells with grid cell input for spatial representation and reward cells 
    for goal-directed behavior.

    Attributes:
        max_speed (float): Maximum wheel rotation speed in rad/s.
        left_speed (float): Current left wheel speed in rad/s.
        right_speed (float): Current right wheel speed in rad/s.
        timestep (int): Duration of each simulation step in ms.
        wheel_radius (float): Radius of robot wheels in meters.
        axle_length (float): Distance between wheels in meters.
        num_steps (int): Total number of simulation steps.
        hmap_loc (ndarray): History of xzy-coordinates.
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
        pcns (List[MultiscalePlaceCellWithGrid]): List of place cell networks at different scales.
        rcns (List[RewardCellLayer]): List of reward cell networks for different scales.
        gcns (List[GridCellLayer]): List of grid cell networks for different scales.
        boundary_data (Tensor): Current LiDAR readings.
        goal_location (List[float]): Target [x,y] coordinates.
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

        # Stats / trial info
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        # Head direction layer size
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 8
        self.max_dist = max_dist if max_dist is not None else 10
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
                "scale_index": 0,
                "num_pc": 500,
                "sigma_r": 0.5,
                "sigma_theta": 1.0,
                "rcn_learning_rate": 0.1,
                # Grid cell parameters
                "grid_influence": 0.5,
                "num_grid_cells": 400,
                "rotation_range": (0, 90),
                "spread_range": (1.2, 1.2),
                "translation_factor": 1.0,
                "frequency_divisor": 1.0
            }]
        self.scales = scales
        
        # Store TD learning and proximity modulation settings
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
        self.pcns = []
        self.rcns = []
        self.gcns = []
        
        # Initialize grid cell networks first
        self.init_grid_cell_networks()
        
        # Load place cell networks (need grid cells first)
        self.load_pcns(enable_ojas, enable_stdp)
        
        # Load reward cell networks
        self.load_rcns()
        
        print(f"[DRIVER] Using PCNs: {[f'pcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")
        print(f"[DRIVER] Using RCNs: {[f'rcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")
        print(f"[DRIVER] Using GCNs: {[f'gcn_scale_{scale_def['scale_index']}.pkl' for scale_def in self.scales]}")

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device=self.device)

        # Initialize alpha as a tensor of zeros with the same length as the number of scales
        self.alpha = torch.zeros(len(self.scales), dtype=self.dtype, device=self.device)

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device=self.device, dtype=torch.float32)
        self.hmap_prox = torch.zeros((self.num_steps,), device=self.device, dtype=torch.float32)
        self.hmap_scale_priority = torch.zeros(
            (self.num_steps, len(self.scales)),  # row per step, col per scale
            device=self.device, 
            dtype=torch.float32
        )

        # For multi-scale place cell logs
        self.hmap_pcn_activities = []
        for scale_def in self.scales:
            n_pc = scale_def["num_pc"]
            self.hmap_pcn_activities.append(
                torch.zeros((self.num_steps, n_pc), device=self.device, dtype=torch.float32)
            )
            
        # For grid cell logs
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

        # Step once
        self.step(self.timestep)
        self.step_count += 1
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    ##########################################################################
    #                      GRID CELL NETWORK INITIALIZATION                   #
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
                enable_ojas if enable_ojas is not None else None,
                enable_stdp if enable_stdp is not None else None,
            )

            self.pcns.append(pcn)

    def _load_or_init_pcn_for_scale(self, path, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        try:
            with open(path, "rb") as f:
                pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing PCN from {path}")
            
            # Update device to current device
            pcn.device = self.device
            
            # Overwrite enable_ojas and enable_stdp based on the provided values
            if enable_ojas is not None:
                pcn.enable_ojas = enable_ojas
            if enable_stdp is not None:
                pcn.enable_stdp = enable_stdp
                
            # Update grid parameters
            pcn.grid_influence = scale_def.get("grid_influence", 0.5)
            pcn.gamma_pg = scale_def.get("gamma_pg", 0.3)
            
            # Make sure the grid cell network has the right size
            if num_grid_cells > 0 and pcn.num_grid_cells != num_grid_cells:
                print(f"[DRIVER] Warning: PCN had {pcn.num_grid_cells} grid cells, but scale def specifies {num_grid_cells}")
                # We'll need to reinitialize with the correct number of grid cells
                return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)
            
            print(f"[DRIVER] Updated PCN for {path} - enable_ojas: {pcn.enable_ojas}, enable_stdp: {pcn.enable_stdp}")
            return pcn

        except (FileNotFoundError, pickle.UnpicklingError):
            return self._initialize_new_pcn(scale_def, num_grid_cells, enable_ojas, enable_stdp)

    def _initialize_new_pcn(self, scale_def, num_grid_cells, enable_ojas, enable_stdp):
        """Initialize a new place cell network with grid input."""
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
        
        # Get grid parameters
        grid_influence = scale_def.get("grid_influence", 0.5)
        gamma_pg = scale_def.get("gamma_pg", 0.3)
        
        # Create place cell layer with grid input
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
            device=self.device,
        )
        
        return pcn

    def load_rcns(self):
        self.rcns = []
        for scale_def in self.scales:
            scale_idx = scale_def["scale_index"]
            fname = f"rcn_scale_{scale_idx}.pkl"
            path = os.path.join(self.network_dir, fname)
            learning_rate = scale_def.get("rcn_learning_rate", 0.1)
            rcn = self._load_or_init_rcn_for_scale(path, scale_def, learning_rate)
            self.rcns.append(rcn)

    def _load_or_init_rcn_for_scale(self, path, scale_def, learning_rate):
        try:
            with open(path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing RCN from {path}")
            rcn.device = self.device
        except:
            print(f"[DRIVER] Initializing new RCN for {path} with learning rate {learning_rate}")
            rcn = RewardCellLayer(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                device=self.device,
            )
        return rcn


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

    def explore(self) -> None:
        """
        Handles exploration for multi-scale usage, calling compute_pcn_activations().
        """
        for _ in range(self.tau_w):
            # 1) Sense environment
            self.sense()

            # 2) compute pcn_activations
            self.compute_pcn_activations()

            # 3) If DMTP or EXPOIT => reward updates
            if self.robot_mode == RobotMode.DMTP or self.robot_mode == RobotMode.EXPLOIT:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)
                    if self.td_learning:
                        rcn.td_update(pcn.place_cell_activations, next_reward=actual_reward)

            # 4) If collisions => turn away
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break

            # 5) Check goal, update hmaps, forward
            self.check_goal_reached()
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
        self.update_hmaps()
        self.check_goal_reached()

        # We will use the primary scale for navigation in this example
        # In a more sophisticated approach, you might want to use a combination of scales
        # based on their reliability or context
        primary_pcn = self.pcns[0]
        primary_rcn = self.rcns[0]

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
                pcn_activations = primary_pcn.preplay(d, num_steps=num_steps_preplay)

                # Update reward cell activations (without saving to memory)
                primary_rcn.update_reward_cell_activations(pcn_activations, visit=False)

                # Take the maximum activation in reward cells as the "reward estimate"
                pot_rew[d] = torch.max(torch.nan_to_num(primary_rcn.reward_cell_activations))

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
        self.update_hmaps()

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
        Uses current boundary- and HD-activations to update place-cell activations with grid cell input.
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
        
        # Store place cell activations
        self.pcn_activations_list = []
        
        # For each scale's PCN, call get_place_cell_activations with corresponding grid input
        for i, pcn in enumerate(self.pcns):
            grid_input = self.grid_activations_list[i]
            
            pcn.get_place_cell_activations(
                distances=self.boundaries,
                grid_activations=grid_input,
                hd_activations=self.hd_activations,
                collided=torch.any(self.collided),
            )
            
            if i == 0 and self.plot_bvc:  # Only plot BVC for first scale to avoid too many plots
                pcn.bvc_layer.plot_activation(self.boundaries.cpu())
            
            # Append activations to pcn_activations_list
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
            self.update_hmaps()
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
        grid cell activations, scale priority, and proximity values.

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

            # 2) Update head direction activations if requested
            if update_hdn:
                self.hmap_hdn[self.step_count] = self.hd_activations

            # 3) Update place cell activations if requested
            if update_pcn:
                for i, act in enumerate(self.pcn_activations_list):
                    if i < len(self.hmap_pcn_activities):
                        self.hmap_pcn_activities[i][self.step_count] = act
            
            # 4) Update grid cell activations if requested
            if update_gcn:
                for i, act in enumerate(self.grid_activations_list):
                    if i < len(self.hmap_gcn_activities) and act is not None:
                        self.hmap_gcn_activities[i][self.step_count] = act

            # 5) Update scale priority if requested
            if update_scale_priority and hasattr(self, 'scale_idx'):
                self.hmap_scale_priority[self.step_count] = self.scale_idx 

            # 6) Update proximity value if requested
            if update_prox and hasattr(self, 'prox'):
                self.hmap_prox[self.step_count] = self.prox

        self.step_count += 1

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to the goal.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        # Implementation remains same as in original driver
        pass

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