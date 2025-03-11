"""
driver_multi_trial.py

Controls robot navigation and learning using neural network layers for place and reward cells.
This updated version supports multi-trial operation via new trial_setup() and run_trial() methods.
"""

import numpy as np
from numpy.random import default_rng
import pickle
import os
import tkinter as tk
from tkinter import messagebox
from typing import Optional, List, Union
import torch
import torch.nn.functional as F
from controller import Supervisor
from astropy.stats import circmean
import random
import sys
from pathlib import Path
import json
import shutil

# Add project root to sys.path so that core modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode
from data_manager import archive_trial_data

# Set PyTorch printing options.
np.set_printoptions(precision=2)


class MultiTrialDriver(Supervisor):
    """
    MultiTrialDriver controls robot navigation and learning using neural network layers
    for place and reward cells. This version accepts a range of parameters to ease
    trial configuration and supports multi-trial operation via trial_setup() and run_trial().
    """
    # Add a class-level alias for simulation run mode.
    # In the current Webots version, SIMULATION_MODE_RUN is not defined,
    # so we alias it to SIMULATION_MODE_FAST.
    SIMULATION_MODE_RUN = Supervisor.SIMULATION_MODE_FAST

    def __init__(self, **kwargs):
        """
        Initializes the driver with parameters passed as keyword arguments.
        Expected keys include:
            environment_label, max_runtime_hours, randomize_start_loc, start_loc,
            goal_location, mode, movement_method, sigma_ang, sigma_d, max_dist,
            num_bvc_per_dir, num_place_cells, n_hd, save_trial_data, trial_name,
            run_multiple_trials, enable_ojas, enable_stdp, world_name.
        """
        super().__init__()
        self.saved_once = False
        # Store parameters (default values provided if key is missing)
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
        # The external run_multiple_trials flag is handled in the controller,
        # so we always set the internal flag to False.
        self.run_multiple_trials = False  
        self.enable_ojas = kwargs.get("enable_ojas", None)
        self.enable_stdp = kwargs.get("enable_stdp", None)
        self.world_name = kwargs.get("world_name", None)

        # Call initialization to set up sensors, actuators, and networks.
        self.initialization()

    def initialization(self):
        """Initializes hardware components, sensors, and model layers."""
        # Get the robot node.
        self.robot = self.getFromDef("agent")
        # Set device and dtype.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float32

        # Determine world name if not provided.
        if self.world_name is None:
            world_path = self.getWorldPath()
            self.world_name = os.path.splitext(os.path.basename(world_path))[0]

        # Set directory paths for saving data (standard location).
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters.
        self.timestep = 32 * 3
        self.tau_w = 5  # time constant for the window function

        # Robot parameters.
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = self.max_runtime_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        # Set starting position.
        if self.randomize_start_loc:
            while True:
                INITIAL = [random.uniform(-2.3, 2.3), 0, random.uniform(-2.3, 2.3)]
                if np.sqrt((INITIAL[0] - self.goal_location[0])**2 + (INITIAL[2] - self.goal_location[1])**2) >= 1.0:
                    break
        else:
            INITIAL = [self.start_loc[0], 0, self.start_loc[1]]
        self.robot.getField("translation").setSFVec3f(INITIAL)
        self.robot.resetPhysics()

        # Initialize sensors and actuators.
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
        self.collided = torch.zeros(2, dtype=torch.int32)
        self.rotation_field = self.robot.getField("rotation")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        # Initialize LiDAR boundaries.
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # In LEARN_OJAS mode, clear any previous network files.
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Load or initialize the Place Cell Network (PCN).
        self.load_pcn(num_place_cells=self.num_place_cells,
                      n_hd=self.n_hd,
                      timestep=self.timestep,
                      enable_ojas=self.enable_ojas,
                      enable_stdp=self.enable_stdp,
                      device=self.device)
        # Load or initialize the Reward Cell Network (RCN).
        self.load_rcn(num_place_cells=self.num_place_cells,
                      num_replay=3,
                      learning_rate=0.1,
                      device=self.device)

        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Initialize history maps.
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_pcn = torch.zeros((self.num_steps, self.num_place_cells), device=self.device, dtype=torch.float32)
        self.hmap_bvc = torch.zeros((self.num_steps, self.num_bvc_per_dir * self.n_hd), device=self.device, dtype=torch.float32)
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)
        self.step(self.timestep)
        self.step_count += 1

        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    def load_pcn(self, num_place_cells: int, n_hd: int, timestep: int,
                 enable_ojas: Optional[bool] = None, enable_stdp: Optional[bool] = None,
                 device: Optional[str] = None):
        """Loads or initializes the Place Cell Network (PCN)."""
        try:
            network_path = os.path.join(self.network_dir, "pcn.pkl")
            with open(network_path, "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing PCN from", network_path)
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except Exception as e:
            print(f"File pcn.pkl not found. Initializing new PCN. Exception: {e}")
            bvc = BoundaryVectorCellLayer(
                n_res=self.lidar_resolution,
                n_hd=n_hd,
                sigma_theta=self.sigma_ang,
                sigma_r=self.sigma_d,
                max_dist=self.max_dist,
                num_bvc_per_dir=self.num_bvc_per_dir,
                device=device,
            )
            self.pcn = PlaceCellLayer(bvc_layer=bvc,
                                      num_pc=num_place_cells,
                                      timestep=timestep,
                                      n_hd=n_hd,
                                      device=device)
            print("Initialized new PCN")

        self.pcn.enable_ojas = enable_ojas if enable_ojas is not None else (self.robot_mode == RobotMode.LEARN_OJAS)
        self.pcn.enable_stdp = enable_stdp if enable_stdp is not None else (self.robot_mode in (RobotMode.LEARN_HEBB, RobotMode.DMTP, RobotMode.EXPLOIT))
        return self.pcn

    def load_rcn(self, num_place_cells: int, num_replay: int, learning_rate: float, device: str):
        """Loads or initializes the Reward Cell Network (RCN)."""
        try:
            network_path = os.path.join(self.network_dir, "rcn.pkl")
            with open(network_path, "rb") as f:
                self.rcn = pickle.load(f)
                print("Loaded existing RCN from", network_path)
                self.rcn.device = device
        except Exception as e:
            print(f"File rcn.pkl not found. Initializing new RCN. Exception: {e}")
            self.rcn = RewardCellLayer(num_place_cells=num_place_cells,
                                       num_replay=num_replay,
                                       learning_rate=learning_rate,
                                       device=device)
            print("Initialized new RCN")
        return self.rcn

    def run(self):
        """
        The original run() method remains for single-run operation.
        In multi-trial mode, the multi-trial controller will call trial_setup() and run_trial().
        """

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
        # When the trial ends, pause the simulation.
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    def trial_setup(self, trial_params: dict):
        """
        Update instance parameters and reset simulation state for a new trial.
        This should be called between trials.
        """
        # Update parameters from the trial dictionary.
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

        # Reset the robot's position.
        if self.randomize_start_loc:
            while True:
                INITIAL = [random.uniform(-2.3, 2.3), 0, random.uniform(-2.3, 2.3)]
                if np.sqrt((INITIAL[0] - self.goal_location[0])**2 + (INITIAL[2] - self.goal_location[1])**2) >= 1.0:
                    break
        else:
            INITIAL = [self.start_loc[0], 0, self.start_loc[1]]
        self.robot.getField("translation").setSFVec3f(INITIAL)
        self.robot.resetPhysics()
        
        # Reset internal simulation timing and history maps.
        self.run_time_minutes = self.max_runtime_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_pcn = torch.zeros((self.num_steps, self.num_place_cells), device=self.device, dtype=torch.float32)
        self.hmap_bvc = torch.zeros((self.num_steps, self.num_bvc_per_dir * self.n_hd), device=self.device, dtype=torch.float32)
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        
        # **** New code: Force network reinitialization ****
        pcn_path = os.path.join(self.network_dir, "pcn.pkl")
        rcn_path = os.path.join(self.network_dir, "rcn.pkl")
        if os.path.exists(pcn_path):
            os.remove(pcn_path)
        if os.path.exists(rcn_path):
            os.remove(rcn_path)
        
        # Clear any saved files (if any exist in the current directory)
        self.clear()
        
        # Reload networks with new parameters.
        self.load_pcn(num_place_cells=self.num_place_cells,
                    n_hd=self.n_hd,
                    timestep=self.timestep,
                    enable_ojas=self.enable_ojas,
                    enable_stdp=self.enable_stdp,
                    device=self.device)
        self.load_rcn(num_place_cells=self.num_place_cells,
                    num_replay=3,
                    learning_rate=0.1,
                    device=self.device)
        self.trial_finished = False
        self.saved_once = False


    def run_trial(self):
        """
        Run a single trial until the trial_finished flag is set.
        This method uses the updated state from trial_setup().
        """
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

    def explore(self) -> None:
        """Handles exploration mode logic: updating activations, checking collisions, and moving."""
        for s in range(self.tau_w):
            self.sense()
            if self.robot_mode == RobotMode.DMTP:
                actual_reward = self.get_actual_reward()
                self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)
                self.rcn.td_update(self.pcn.place_cell_activations, next_reward=actual_reward)
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break
            self.check_goal_reached()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
        self.turn(np.random.normal(0, np.deg2rad(30)))

    def exploit(self):
        """Follows the reward gradient to reach the goal location."""
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()
        self.check_goal_reached()
        num_steps_preplay = 1
        pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
        for d in range(self.n_hd):
            pcn_activations = self.pcn.preplay(d, num_steps=num_steps_preplay)
            self.rcn.update_reward_cell_activations(pcn_activations, visit=False)
            pot_rew[d] = torch.max(torch.nan_to_num(self.rcn.reward_cell_activations))
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd,
                                device=self.device, dtype=self.dtype)
        angles_np = angles.cpu().numpy()
        weights_np = pot_rew.cpu().numpy()
        sin_component = np.sum(np.sin(angles_np) * weights_np)
        cos_component = np.sum(np.cos(angles_np) * weights_np)
        action_angle = np.arctan2(sin_component, cos_component)
        if action_angle < 0:
            action_angle += 2 * np.pi
        angle_to_turn_deg = np.rad2deg(action_angle) - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180
        angle_to_turn = np.deg2rad(angle_to_turn_deg)
        self.turn(angle_to_turn)
        self.forward()
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    def sense(self):
        """Uses sensors to update range-image, heading, boundary data, and collision flags."""
        boundaries = self.range_finder.getRangeImage()
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * self.current_heading_deg,
        )
        current_heading_rad = np.deg2rad(self.current_heading_deg)
        v_in = torch.tensor([np.cos(current_heading_rad), np.sin(current_heading_rad)],
                             dtype=self.dtype, device=self.device)
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)
        self.collided[0] = int(self.left_bumper.getValue())
        self.collided[1] = int(self.right_bumper.getValue())
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        rad = np.arctan2(north[1], north[0])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing += 360.0
        return bearing

    def compute_pcn_activations(self):
        """Updates place-cell activations using current boundary and head direction data."""
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided),
        )
        self.step(self.timestep)

    def check_goal_reached(self):
        curr_pos = self.robot.getField("translation").getSFVec3f()
        elapsed = self.getTime() - self.trial_start_time  # Relative trial time in seconds
        trial_duration = 60 * self.run_time_minutes  
        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING) and elapsed >= trial_duration:
            self.stop()
            self.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
            if not self.saved_once:
                self.save(show_popup=(not getattr(self, "disable_save_popup", True)))
                self.saved_once = True
            self.trial_finished = True
        elif self.robot_mode == RobotMode.DMTP and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"],
        ):
            self.auto_pilot()
            self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations, visit=True)
            self.rcn.replay(pcn=self.pcn)
            self.stop()
            self.save(show_popup=True, include_rcn=True, include_hmaps=False)
            self.trial_finished = True
        elif self.robot_mode == RobotMode.EXPLOIT and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["exploit"],
        ):
            self.auto_pilot()
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Time taken: {self.getTime()}")
            self.stop()
            self.save(show_popup=False, include_rcn=True)
            self.trial_finished = True

    def auto_pilot(self):
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()
        while not torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"],
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]
            if delta_x >= 0:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))
                ).item()
                desired = (2 * np.pi - theta) if delta_y >= 0 else (np.pi + theta)
            elif delta_y >= 0:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))
                ).item()
                desired = (np.pi / 2) - theta
            else:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device)),
                    torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device))
                ).item()
                desired = np.pi - theta
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
            s_start += 1

    def manual_control(self):
        """Enables manual control of the robot via keyboard."""
        k = self.keyboard.getKey()
        if k == ord("W") or k == self.keyboard.UP:
            self.forward()
        elif k == ord("A") or k == self.keyboard.LEFT:
            self.rotate(direction=1, speed_factor=0.3)
        elif k == ord("D") or k == self.keyboard.RIGHT:
            self.rotate(direction=-1, speed_factor=0.3)
        elif k == ord("S") or k == self.keyboard.DOWN:
            self.stop()
        self.sense()
        self.step(self.timestep)

    def rotate(self, direction: int, speed_factor: float = 0.3):
        speed = self.max_speed * speed_factor
        self.left_speed = speed * direction
        self.right_speed = -speed * direction
        self.move()

    def forward(self):
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()

    def turn(self, angle: float, circle: bool = False):
        self.stop()
        self.move()
        l_offset = self.left_position_sensor.getValue()
        r_offset = self.right_position_sensor.getValue()
        self.sense()
        neg = -1.0 if angle < 0.0 else 1.0
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
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def move(self):
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(self.left_speed)
        self.right_motor.setVelocity(self.right_speed)

    def compute_path_length(self):
        path_length = 0
        for i in range(self.hmap_loc[:, 0].shape[0] - 1):
            current_position = np.array([self.hmap_loc[:, 2][i], self.hmap_loc[:, 0][i]])
            next_position = np.array([self.hmap_loc[:, 2][i + 1], self.hmap_loc[:, 0][i + 1]])
            path_length += np.linalg.norm(next_position - current_position)
        return path_length

    def update_hmaps(self):
        curr_pos = self.robot.getField("translation").getSFVec3f()
        if self.step_count < self.num_steps:
            self.hmap_loc[self.step_count] = curr_pos
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.detach()
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.detach()
            self.hmap_hdn[self.step_count] = self.hd_activations.detach()
        self.step_count += 1

    def get_actual_reward(self):
        curr_pos = self.robot.getField("translation").getSFVec3f()
        distance_to_goal = torch.norm(
            torch.tensor([curr_pos[0] - self.goal_location[0], curr_pos[2] - self.goal_location[1]],
                         dtype=self.dtype, device=self.device)
        )
        goal_radius = self.goal_r["exploit"] if self.robot_mode == RobotMode.EXPLOIT else self.goal_r["explore"]
        return 1.0 if distance_to_goal <= goal_radius else 0.0

    def save(self, show_popup: bool = True, include_pcn: bool = True,
             include_rcn: bool = True, include_hmaps: bool = True):
        """
        Saves the state of the PCN, RCN, and hmap data to standard locations.
        If show_popup is True, a popup prompts the user to confirm saving.
        After saving, if save_trial_data is enabled, archives the trial data.
        """
        files_saved = []
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)
        if include_pcn:
            pcn_path = os.path.join(self.network_dir, "pcn.pkl")
            with open(pcn_path, "wb") as output:
                pickle.dump(self.pcn, output)
            files_saved.append(pcn_path)
        if include_rcn:
            rcn_path = os.path.join(self.network_dir, "rcn.pkl")
            with open(rcn_path, "wb") as output:
                pickle.dump(self.rcn, output)
            files_saved.append(rcn_path)
        if include_hmaps:
            hmap_loc_path = os.path.join(self.hmap_dir, "hmap_loc.pkl")
            with open(hmap_loc_path, "wb") as output:
                pickle.dump(self.hmap_loc[: self.step_count], output)
            files_saved.append(hmap_loc_path)
            hmap_pcn_path = os.path.join(self.hmap_dir, "hmap_pcn.pkl")
            with open(hmap_pcn_path, "wb") as output:
                pickle.dump(self.hmap_pcn[: self.step_count].cpu().numpy(), output)
            files_saved.append(hmap_pcn_path)
            hmap_hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
            with open(hmap_hdn_path, "wb") as output:
                pickle.dump(self.hmap_hdn[: self.step_count], output)
            files_saved.append(hmap_hdn_path)
            hmap_bvc_path = os.path.join(self.hmap_dir, "hmap_bvc.pkl")
            with open(hmap_bvc_path, "wb") as output:
                pickle.dump(self.hmap_bvc[: self.step_count].cpu().numpy(), output)
            files_saved.append(hmap_bvc_path)

        if show_popup:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            messagebox.showinfo("Information", "Press OK to save data")
            root.destroy()
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

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
                "trial_name": self.trial_name
            }
            archive_trial_data(trial_params,
                               pkl_base_dir=os.path.join("pkl", self.world_name),
                               controller_base_dir=os.path.join(os.path.dirname(__file__), "trial_data"))

    def clear(self):
        """Clears the saved network and hmap files."""
        files_to_remove = ["pcn.pkl", "rcn.pkl", "hmap_loc.pkl", "hmap_pcn.pkl", "hmap_bvc.pkl", "hmap_hdn.pkl"]
        for file in files_to_remove:
            try:
                os.remove(file)
                print(f"Removed {file}")
            except FileNotFoundError:
                print(f"File {file} not found.")
                pass
