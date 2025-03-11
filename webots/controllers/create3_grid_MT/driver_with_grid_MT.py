import numpy as np
from numpy.random import default_rng
import pickle
import os
import tkinter as tk
from tkinter import N, messagebox
from typing import Optional, List, Union
import torch
import torch.nn.functional as F
from controller import Supervisor
from astropy.stats import circmean
import random
import json
import shutil

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer_with_grid import PlaceCellLayerWithGrid
from core.layers.grid_cell_layer import GridCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode
from data_manager import archive_trial_data

# --- PyTorch seeds / random ---
# torch.manual_seed(5)
# np.random.seed(5)
# rng = default_rng(5)  # or keep it as is
np.set_printoptions(precision=2)

class DriverGridMT(Supervisor):
    def __init__(self, **kwargs):
        """Initializes the driver with standard and grid cell parameters.
        
        Args:
            **kwargs: Parameters for robot behavior and model configuration.
            
        Expected parameters include standard params like:
            environment_label, max_runtime_hours, randomize_start_loc, start_loc,
            goal_location, mode, movement_method, etc.
            
        And grid cell specific params like:
            num_grid_cells, grid_influence, rotation_range, spread_range,
            x_trans_range, y_trans_range, scale_multiplier, frequency_divisor
        """
        super().__init__()
        self.saved_once = False
        
        # Store standard parameters (from AlexDriver)
        self.environment_label = kwargs.get("environment_label", "default_env")
        self.max_runtime_hours = kwargs.get("max_runtime_hours", 2)
        self.randomize_start_loc = kwargs.get("randomize_start_loc", True)
        self.start_loc = kwargs.get("start_loc", [4, -4])
        self.goal_location = kwargs.get("goal_location", [-3, 3])
        self.robot_mode = kwargs.get("mode", RobotMode.PLOTTING)
        self.movement_method = kwargs.get("movement_method", "default")
        self.sigma_ang = kwargs.get("sigma_ang", 0.01)
        self.sigma_d = kwargs.get("sigma_d", 0.5)
        self.max_dist = kwargs.get("max_dist", 10)
        self.num_bvc_per_dir = kwargs.get("num_bvc_per_dir", 50)
        self.num_place_cells = kwargs.get("num_place_cells", 500)
        self.n_hd = kwargs.get("n_hd", 8)
        self.save_trial_data = kwargs.get("save_trial_data", False)
        self.trial_name = kwargs.get("trial_name", "none")
        self.run_multiple_trials = False  # Always false internally
        self.enable_ojas = kwargs.get("enable_ojas", None)
        self.enable_stdp = kwargs.get("enable_stdp", None)
        self.world_name = kwargs.get("world_name", None)
        
        # Store grid cell parameters (from DriverGrid)
        self.num_grid_cells = kwargs.get("num_grid_cells", 400)
        self.grid_influence = kwargs.get("grid_influence", 0.5)
        self.rotation_range = kwargs.get("rotation_range", (0, 90))
        self.spread_range = kwargs.get("spread_range", (1.2, 1.2))
        self.x_trans_range = kwargs.get("x_trans_range", (-1.0, 1.0))
        self.y_trans_range = kwargs.get("y_trans_range", (-1.0, 1.0))
        self.scale_multiplier = kwargs.get("scale_multiplier", 5.0)
        self.frequency_divisor = kwargs.get("frequency_divisor", 1.0)
        
        # Call initialization to set up sensors, actuators, and networks
        self.initialization()

    def initialization(self):
        """Initialize the robot, neural layers, and history maps with grid cell integration."""
        # Get the robot node
        self.robot = self.getFromDef("agent")
        # Set device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.step_count = 0

        self.show_bvc_activation = False

        # Determine world name if not provided
        if self.world_name is None:
            world_path = self.getWorldPath()
            self.world_name = os.path.splitext(os.path.basename(world_path))[0]

        # Set directory paths for saving data
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters
        self.timestep = 32 * 3
        self.tau_w = 5  # time constant for the window function

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = self.max_runtime_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        # Set starting position
        if self.randomize_start_loc:
            while True:
                INITIAL = [np.random.uniform(-2.3, 2.3), 0, np.random.uniform(-2.3, 2.3)]
                dist_to_goal = np.sqrt(
                    (INITIAL[0] - self.goal_location[0]) ** 2
                    + (INITIAL[2] - self.goal_location[1]) ** 2
                )
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        else:
            self.robot.getField("translation").setSFVec3f([self.start_loc[0], 0, self.start_loc[1]])
            self.robot.resetPhysics()

        # Initialize sensors and actuators
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)
        self.range_finder = self.getDevice("range-finder")
        self.range_finder.enable(self.timestep)
        self.left_bumper = self.getDevice("bumper_left")
        self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice("bumper_right")
        self.right_bumper.enable(self.timestep)
        self.collided = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.rotation_field = self.robot.getField("rotation")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        # Initialize boundaries
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # In LEARN_OJAS mode, clear any previous network files
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Initialize head direction layer
        self.head_direction_layer = HeadDirectionLayer(
            num_cells=self.n_hd, device=self.device
        )

        # Initialize BVC layer
        bvc = BoundaryVectorCellLayer(
            n_res=self.lidar_resolution,
            n_hd=self.n_hd,
            sigma_theta=self.sigma_ang,
            sigma_r=self.sigma_d,
            max_dist=self.max_dist,
            num_bvc_per_dir=self.num_bvc_per_dir,
            device=self.device,
        )

        # Initialize grid cell layer (new for grid integration)
        self.gcn = GridCellLayer(
            num_cells=self.num_grid_cells,
            rotation_range=self.rotation_range,
            spread_range=self.spread_range,
            x_trans_range=self.x_trans_range,
            y_trans_range=self.y_trans_range,
            scale_multiplier=self.scale_multiplier,
            frequency_divisor=self.frequency_divisor,
            device=self.device
        )

        # Load or initialize place cell network with grid inputs
        self.load_pcn(
            bvc_layer=bvc,
            num_place_cells=self.num_place_cells,
            num_grid_cells=self.num_grid_cells,
            timestep=self.timestep,
            n_hd=self.n_hd,
            enable_ojas=self.enable_ojas,
            enable_stdp=self.enable_stdp,
            grid_influence=self.grid_influence,
            device=self.device,
        )

        # Load or initialize reward cell network
        self.load_rcn(
            num_place_cells=self.num_place_cells,
            num_replay=3,
            learning_rate=0.1,
            device=self.device,
        )

        # Initialize history maps (added hmap_gcn for grid cells)
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_pcn = torch.zeros(
            (self.num_steps, self.num_place_cells), device=self.device, dtype=self.dtype
        )
        self.hmap_bvc = torch.zeros(
            (self.num_steps, bvc.num_bvc), device=self.device, dtype=self.dtype
        )
        self.hmap_hdn = torch.zeros(
            (self.num_steps, self.n_hd), device=self.device, dtype=self.dtype
        )
        self.hmap_gcn = torch.zeros(
            (self.num_steps, self.num_grid_cells), device=self.device, dtype=self.dtype
        )

        # Additional setup
        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)
        self.step(self.timestep)
        self.step_count += 1
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    def load_pcn(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_place_cells: int,
        num_grid_cells: int,
        timestep: int,
        n_hd: int,
        device: torch.device,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        grid_influence: float = 0.5,
    ):
        """Load or initialize the place cell network with grid inputs."""
        try:
            network_path = os.path.join(self.network_dir, "pcn.pkl")
            with open(network_path, "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing PCN with grid from", network_path)
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except:
            self.pcn = PlaceCellLayerWithGrid(
                bvc_layer=bvc_layer,
                num_pc=num_place_cells,
                num_grid_cells=num_grid_cells,
                timestep=timestep,
                n_hd=n_hd,
                grid_influence=grid_influence,
                device=device,
            )
            print("Initialized new PCN with grid")

        if enable_ojas is not None:
            self.pcn.enable_ojas = enable_ojas
        else:
            self.pcn.enable_ojas = self.robot_mode == RobotMode.LEARN_OJAS

        if enable_stdp is not None:
            self.pcn.enable_stdp = enable_stdp
        else:
            self.pcn.enable_stdp = self.robot_mode in (
                RobotMode.LEARN_HEBB,
                RobotMode.DMTP,
                RobotMode.EXPLOIT,
            )

    def load_rcn(
        self,
        num_place_cells: int,
        num_replay: int,
        learning_rate: float,
        device: torch.device,
    ):
        """Load or initialize the reward cell network."""
        try:
            network_path = os.path.join(self.network_dir, "rcn.pkl")
            with open(network_path, "rb") as f:
                self.rcn = pickle.load(f)
                print("Loaded existing RCN from", network_path)
                self.rcn.device = device
        except:
            self.rcn = RewardCellLayer(
                num_place_cells=num_place_cells,
                num_replay=num_replay,
                learning_rate=learning_rate,
                device=device,
            )
            print("Initialized new RCN")
        return self.rcn

    def run(self):
        """Main control loop for the robot."""
        if not hasattr(self, 'trial_start_time'):
            self.trial_start_time = self.getTime()
        self.trial_finished = False
        
        print(f"Starting robot in {self.robot_mode}")
        print(f"Goal at {self.goal_location}")
        
        while not self.trial_finished:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB,
                                     RobotMode.DMTP, RobotMode.PLOTTING):
                self.explore()
            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()
            elif self.robot_mode == RobotMode.RECORDING:
                self.recording()
            else:
                print("Unknown state. Exiting...")
                break
                
        # When the trial ends, pause the simulation
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
    
    def trial_setup(self, trial_params: dict):
        """Update instance parameters and reset simulation state for a new trial."""
        # Update standard parameters from the trial dictionary
        self.environment_label = trial_params.get("environment_label", self.environment_label)
        self.max_runtime_hours = trial_params.get("max_runtime_hours", self.max_runtime_hours)
        self.randomize_start_loc = trial_params.get("randomize_start_loc", self.randomize_start_loc)
        self.start_loc = trial_params.get("start_loc", self.start_loc)
        self.goal_location = trial_params.get("goal_location", self.goal_location)
        self.robot_mode = trial_params.get("mode", self.robot_mode)
        self.movement_method = trial_params.get("movement_method", self.movement_method)
        self.sigma_ang = trial_params.get("sigma_ang", self.sigma_ang)
        self.sigma_d = trial_params.get("sigma_d", self.sigma_d)
        self.max_dist = trial_params.get("max_dist", self.max_dist)
        self.num_bvc_per_dir = trial_params.get("num_bvc_per_dir", self.num_bvc_per_dir)
        self.num_place_cells = trial_params.get("num_place_cells", self.num_place_cells)
        self.n_hd = trial_params.get("n_hd", self.n_hd)
        self.save_trial_data = trial_params.get("save_trial_data", self.save_trial_data)
        self.trial_name = trial_params.get("trial_name", self.trial_name)
        self.enable_ojas = trial_params.get("enable_ojas", self.enable_ojas)
        self.enable_stdp = trial_params.get("enable_stdp", self.enable_stdp)
        
        # Update grid cell parameters
        self.num_grid_cells = trial_params.get("num_grid_cells", self.num_grid_cells)
        self.grid_influence = trial_params.get("grid_influence", self.grid_influence)
        self.rotation_range = trial_params.get("rotation_range", self.rotation_range)
        self.spread_range = trial_params.get("spread_range", self.spread_range)
        self.x_trans_range = trial_params.get("x_trans_range", self.x_trans_range)
        self.y_trans_range = trial_params.get("y_trans_range", self.y_trans_range)
        self.scale_multiplier = trial_params.get("scale_multiplier", self.scale_multiplier)
        self.frequency_divisor = trial_params.get("frequency_divisor", self.frequency_divisor)
        
        # Reset the robot's position
        if self.randomize_start_loc:
            while True:
                INITIAL = [np.random.uniform(-2.3, 2.3), 0, np.random.uniform(-2.3, 2.3)]
                dist_to_goal = np.sqrt(
                    (INITIAL[0] - self.goal_location[0]) ** 2
                    + (INITIAL[2] - self.goal_location[1]) ** 2
                )
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        else:
            self.robot.getField("translation").setSFVec3f([self.start_loc[0], 0, self.start_loc[1]])
            self.robot.resetPhysics()
        
        # Reset internal simulation timing and history maps
        self.run_time_minutes = self.max_runtime_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        
        # Reset history maps
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_pcn = torch.zeros(
            (self.num_steps, self.num_place_cells), device=self.device, dtype=self.dtype
        )
        self.hmap_bvc = torch.zeros(
            (self.num_steps, self.num_bvc_per_dir * self.n_hd), device=self.device, dtype=self.dtype
        )
        self.hmap_hdn = torch.zeros(
            (self.num_steps, self.n_hd), device=self.device, dtype=self.dtype
        )
        self.hmap_gcn = torch.zeros(
            (self.num_steps, self.num_grid_cells), device=self.device, dtype=self.dtype
        )
        
        # Clear saved networks
        pcn_path = os.path.join(self.network_dir, "pcn.pkl")
        rcn_path = os.path.join(self.network_dir, "rcn.pkl")
        if os.path.exists(pcn_path):
            os.remove(pcn_path)
        if os.path.exists(rcn_path):
            os.remove(rcn_path)
        
        # Clear any saved files
        self.clear()
        
        # Re-initialize BVC layer
        bvc = BoundaryVectorCellLayer(
            n_res=self.lidar_resolution,
            n_hd=self.n_hd,
            sigma_theta=self.sigma_ang,
            sigma_r=self.sigma_d,
            max_dist=self.max_dist,
            num_bvc_per_dir=self.num_bvc_per_dir,
            device=self.device,
        )
        
        # Re-initialize grid cell layer
        self.gcn = GridCellLayer(
            num_cells=self.num_grid_cells,
            rotation_range=self.rotation_range,
            spread_range=self.spread_range,
            x_trans_range=self.x_trans_range,
            y_trans_range=self.y_trans_range,
            scale_multiplier=self.scale_multiplier,
            frequency_divisor=self.frequency_divisor,
            device=self.device
        )
        
        # Reload networks with new parameters
        self.load_pcn(
            bvc_layer=bvc,
            num_place_cells=self.num_place_cells,
            num_grid_cells=self.num_grid_cells,
            timestep=self.timestep,
            n_hd=self.n_hd,
            enable_ojas=self.enable_ojas,
            enable_stdp=self.enable_stdp,
            grid_influence=self.grid_influence,
            device=self.device,
        )
        self.load_rcn(
            num_place_cells=self.num_place_cells,
            num_replay=3,
            learning_rate=0.1,
            device=self.device,
        )
        
        self.trial_finished = False
        self.saved_once = False

    def run_trial(self):
        """Run a single trial until the trial_finished flag is set."""
        print(f"Starting trial: {self.trial_name}, runtime: {self.max_runtime_hours} hours")
        
        while not self.trial_finished:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB,
                                     RobotMode.DMTP, RobotMode.PLOTTING):
                self.explore()
            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()
            elif self.robot_mode == RobotMode.RECORDING:
                self.recording()
            else:
                print("Unknown state. Exiting trial...")
                break
                
        print(f"Trial {self.trial_name} finished.")

    def run_trials(trial_list):
        """
        Run multiple trials in sequence, using the provided trial configurations.
        """
        if not trial_list:
            print("No trials to run - empty trial list")
            return
            
        # Default params with hard-coded disable_save_popup
        default_params = trial_list[0].copy()
        default_params["disable_save_popup"] = True  # Hard-coded to disable popups
        
        # Create a single driver instance that will be reused for all trials
        print(f"Creating driver with default parameters from first trial")
        bot = DriverGridMT(**default_params)
        
        # Run each trial in sequence
        for i, trial_params in enumerate(trial_list):
            print(f"\nRunning trial {i+1} of {len(trial_list)}: {trial_params.get('trial_name', f'Trial_{i+1}')}")
            
            # Hard-code disable_save_popup to always be True
            trial_params["disable_save_popup"] = True
            
            # Set up the robot and environment for this trial
            bot.trial_setup(trial_params)
            
            # Record the trial's start time for proper duration tracking
            bot.trial_start_time = bot.getTime()
            
            # Execute the trial until completion
            bot.run_trial()
            
            # Just set to FAST mode between trials but don't pause - exactly as in alex_controller
            bot.simulationSetMode(bot.SIMULATION_MODE_FAST)
            
            print(f"Trial {i+1} completed")
        
        # Only pause after ALL trials are complete
        print("\nAll trials have been completed. Pausing simulation.")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        """Handles the exploration mode logic for the robot.

        The robot moves forward for a set number of steps while:
        - Updating place and reward cell activations
        - Checking for collisions and turning if needed
        - Computing TD updates for reward learning
        - Monitoring goal proximity
        - Randomly changing direction periodically

        Returns:
            None
        """

        for s in range(self.tau_w):
            self.sense()

            if self.mode == RobotMode.DMTP:
                actual_reward = self.get_actual_reward()
                self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)
                self.rcn.td_update(
                    self.pcn.place_cell_activations, next_reward=actual_reward
                )

            if torch.any(self.collided):
                random_angle = np.random.uniform(
                    -np.pi, np.pi
                )  # Random angle between -180 and 180 degrees (in radians)
                self.turn(random_angle)
                break

            self.check_goal_reached()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()

        self.turn(np.random.normal(0, np.deg2rad(30)))  # Choose a new random direction

    ########################################### EXPLOIT ###########################################
    def exploit(self):
        """
        Follows the reward gradient to reach the goal location.
        """
        # -------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        # -------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()
        self.check_goal_reached()

        # -------------------------------------------------------------------
        # 2) Calculate potential reward for each possible head direction
        # -------------------------------------------------------------------
        num_steps_preplay = 1  # Number of future steps to "preplay"
        pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)

        # For each head direction index 'd', do a preplay and estimate reward
        for d in range(self.n_hd):
            # Predicted place-cell activation for direction 'd'
            pcn_activations = self.pcn.preplay(d, num_steps=num_steps_preplay)

            # Update reward cell activations (may not need this if we don't want to save anything in EXPLOIT)
            self.rcn.update_reward_cell_activations(pcn_activations, visit=False)

            # Example: take the maximum activation in reward cells as the "reward estimate"
            pot_rew[d] = torch.max(torch.nan_to_num(self.rcn.reward_cell_activations))

        # -------------------------------------------------------------------
        # 3) Prepare angles for computing the circular mean (no debug prints)
        # -------------------------------------------------------------------
        angles = torch.linspace(
            0,
            2 * np.pi * (1 - 1 / self.n_hd),
            self.n_hd,
            device=self.device,
            dtype=self.dtype,
        )

        # -------------------------------------------------------------------
        # 4) Compute circular mean of angles, weighted by the reward estimates
        # -------------------------------------------------------------------
        angles_np = angles.cpu().numpy()
        weights_np = pot_rew.cpu().numpy()

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
        Uses current boundary- and HD-activations to update place-cell activations
        and store relevant data for analysis/debugging.
        """

        # Get current position [x, z] from robot (Webots uses [x, y, z], y is vertical)
        curr_pos = self.robot.getField("translation").getSFVec3f()
        position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)
        
        # Compute grid cell activations
        grid_activations = self.gcn.get_grid_cell_activations(position)

        # Update place cell activations based on sensor data
        
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            grid_activations=grid_activations,  # Pass grid cell activations
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided),
        )
        if self.show_bvc_activation:
            self.pcn.bvc_layer.plot_activation(self.boundaries.cpu())
        
        # Advance simulation one timestep
        self.step(self.timestep)

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        elapsed = self.getTime() - self.trial_start_time  # Relative trial time in seconds
        trial_duration = 60 * self.run_time_minutes
        
        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING) and elapsed >= trial_duration:
            self.stop()
            # Set to real-time but DON'T pause
            self.simulationSetMode(self.SIMULATION_MODE_REAL_TIME)
            if not self.saved_once:
                # Original alex_driver behavior - directly check disable_save_popup attribute
                self.save(show_popup=(not getattr(self, "disable_save_popup", True)))
                self.saved_once = True
            self.trial_finished = True

        elif self.robot_mode == RobotMode.DMTP and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor(
                [curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device
            ),
            atol=self.goal_r["explore"],
        ):
            self.auto_pilot()
            self.rcn.update_reward_cell_activations(
                self.pcn.place_cell_activations, visit=True
            )
            self.rcn.replay(pcn=self.pcn)

            self.stop()
            self.save(include_rcn=True, include_hmaps=False)

        elif self.robot_mode == RobotMode.EXPLOIT and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor(
                [curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device
            ),
            atol=self.goal_r["exploit"],
        ):
            self.auto_pilot()
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Time taken: {self.getTime()}")

            self.stop()
            self.save(include_rcn=True)  # EXPLOIT doesn't save anything

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

    def update_hmaps(self):
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # Record position (X, Y, Z format)
            self.hmap_loc[self.step_count] = curr_pos

            # Record place cell network activations
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.detach()

            # Record Boundary Vector Cell (BVC) activations
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.detach()

            # Record Head Direction Network (HDN) activations
            self.hmap_hdn[self.step_count] = self.hd_activations.detach()

            # Record grid cell activations
            position = torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device)
            self.hmap_gcn[self.step_count] = self.gcn.get_grid_cell_activations(position).detach()

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

    def save(self, show_popup: bool = True, include_pcn: bool = True, include_rcn: bool = True, include_hmaps: bool = True):
        """
        Saves the state of the PCN, RCN, and hmap data.
        
        First saves to the standard pkl directories (hmap_dir and network_dir).
        Then, if save_trial_data is True, archives a copy to the trial_data folder.
        
        Args:
            show_popup: Whether to show a confirmation popup
            include_pcn: Whether to save the Place Cell Network
            include_rcn: Whether to save the Reward Cell Network
            include_hmaps: Whether to save the history maps
        """
        files_saved = []
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)
        
        # Save the Place Cell Network
        if include_pcn:
            pcn_path = os.path.join(self.network_dir, "pcn_with_grid.pkl")
            with open(pcn_path, "wb") as output:
                pickle.dump(self.pcn, output)
                files_saved.append(pcn_path)
            # Save Grid Cell Network
            gcn_path = os.path.join(self.network_dir, "gcn.pkl")
            with open(gcn_path, "wb") as output:
                pickle.dump(self.gcn, output)
                files_saved.append(gcn_path)

        # Save the Reward Cell Network
        if include_rcn:
            rcn_path = os.path.join(self.network_dir, "rcn.pkl")
            with open(rcn_path, "wb") as output:
                pickle.dump(self.rcn, output)
                files_saved.append(rcn_path)

        # Save the history maps
        if include_hmaps:
            hmap_loc_path = os.path.join(self.hmap_dir, "hmap_loc.pkl")
            with open(hmap_loc_path, "wb") as output:
                pickle.dump(self.hmap_loc[: self.step_count], output)
                files_saved.append(hmap_loc_path)

            hmap_pcn_path = os.path.join(self.hmap_dir, "hmap_pcn.pkl")
            with open(hmap_pcn_path, "wb") as output:
                pcn_cpu = self.hmap_pcn[: self.step_count].cpu().numpy()
                pickle.dump(pcn_cpu, output)
                files_saved.append(hmap_pcn_path)

            hmap_hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
            with open(hmap_hdn_path, "wb") as output:
                hdn_cpu = self.hmap_hdn[: self.step_count].cpu().numpy()
                pickle.dump(hdn_cpu, output)
                files_saved.append(hmap_hdn_path)

            hmap_bvc_path = os.path.join(self.hmap_dir, "hmap_bvc.pkl")
            with open(hmap_bvc_path, "wb") as output:
                bvc_cpu = self.hmap_bvc[: self.step_count].cpu().numpy()
                pickle.dump(bvc_cpu, output)
                files_saved.append(hmap_bvc_path)
                
            # Save grid cell history map
            hmap_gcn_path = os.path.join(self.hmap_dir, "hmap_gcn.pkl")
            with open(hmap_gcn_path, "wb") as output:
                gcn_cpu = self.hmap_gcn[: self.step_count].cpu().numpy()
                pickle.dump(gcn_cpu, output)
                files_saved.append(hmap_gcn_path)

        # Show a message box to confirm saving if requested
        if show_popup:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes("-topmost", True)  # Always keep the window on top
            root.update()
            messagebox.showinfo("Information", "Press OK to save data")
            root.destroy()  # Destroy the main window
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")
        
        # Archive the trial data if requested
        # This *copies* the data (doesn't move it), so original files stay in place
        if self.save_trial_data:
            trial_params = {
                "environment_label": self.environment_label,
                "max_runtime_hours": self.max_runtime_hours,
                "randomize_start_loc": self.randomize_start_loc,
                "start_loc": self.start_loc,
                "goal_location": self.goal_location,
                "mode": str(self.robot_mode),
                "movement_method": self.movement_method,
                "sigma_ang": self.sigma_ang,
                "sigma_d": self.sigma_d,
                "max_dist": self.max_dist,
                "num_bvc_per_dir": self.num_bvc_per_dir,
                "num_place_cells": self.num_place_cells,
                "n_hd": self.n_hd,
                "num_grid_cells": self.num_grid_cells,
                "grid_influence": self.grid_influence,
                "scale_multiplier": self.scale_multiplier,
                "rotation_range": str(self.rotation_range),  # Convert tuple to string for JSON
                "spread_range": str(self.spread_range),
                "x_trans_range": str(self.x_trans_range),
                "y_trans_range": str(self.y_trans_range),
                "frequency_divisor": self.frequency_divisor,
                "trial_name": self.trial_name
            }
            
            print("Archiving trial data...")
            archive_trial_data(trial_params,
                            pkl_base_dir=os.path.join("pkl", self.world_name),
                            controller_base_dir=os.path.join(os.path.dirname(__file__), "trial_data"))
            print(f"Trial data archived for: {self.trial_name}")

    def clear(self):
        """Clears saved network and history files."""
        # Network files in network_dir
        network_files = [
            os.path.join(self.network_dir, "pcn.pkl"),
            os.path.join(self.network_dir, "rcn.pkl"),
            os.path.join(self.network_dir, "gcn.pkl"),
        ]

        # History map files in hmap_dir
        hmap_files = [
            os.path.join(self.hmap_dir, "hmap_loc.pkl"),
            os.path.join(self.hmap_dir, "hmap_pcn.pkl"),
            os.path.join(self.hmap_dir, "hmap_bvc.pkl"),
            os.path.join(self.hmap_dir, "hmap_hdn.pkl"),
            os.path.join(self.hmap_dir, "hmap_gcn.pkl"),
        ]

        # Remove all files
        for file_path in network_files + hmap_files:
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except FileNotFoundError:
                print(f"File {file_path} not found")
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")