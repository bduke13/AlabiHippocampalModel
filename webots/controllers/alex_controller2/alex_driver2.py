#!/usr/bin/env python
"""
File: webots/controllers/alex_controller/alex_driver.py

AlexDriver extends the original Driver functionality with a modular design.
It implements all original features (sensing, computing, exploration, exploitation,
manual control, auto-pilot, etc.) and includes a new reset_run() method for
automatic multiple-run (grid search) functionality. File-saving is now delegated to TrialManager.
"""

import numpy as np
from numpy.random import default_rng
import pickle
import os
import tkinter as tk
from tkinter import messagebox
from typing import Optional, List
import torch
import torch.nn.functional as F
from controller import Supervisor
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode
from core.robot.movement_method import MovementMethod  # For type hints and future use

# Set seeds
torch.manual_seed(5)
np.random.seed(5)
rng = default_rng(5)
np.set_printoptions(precision=2)

# Define global path for current run data (using lowercase folders)
CURRENT_RUN_DIR = os.path.join(Path(__file__).resolve().parents[3], "webots", "data", "current_run")
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)

class AlexDriver(Supervisor):
    def __init__(self):
        super().__init__()
        # Core configuration (set in initialization)
        self.robot_mode = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        # Model parameters
        self.num_place_cells = 200
        self.num_reward_cells = 1
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 5

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = 60
        self.step_count = 0
        self.num_steps = 0  # Calculated during initialization
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        # Extended multi-scale and exploration settings
        self.scale_mode = "S"  # "S", "L", or "MS"
        self.small_scale_params = {}
        self.large_scale_params = {}
        self.explore_mthd = None  # MovementMethod enum value
        self.environment_label = "default"

        # Hardware placeholders
        self.robot = None
        self.keyboard = None
        self.compass = None
        self.range_finder = None
        self.left_bumper = None
        self.right_bumper = None
        self.collided = None
        self.rotation_field = None
        self.left_motor = None
        self.right_motor = None
        self.left_position_sensor = None
        self.right_position_sensor = None

        # Sensor data
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)
        self.current_heading_deg = 0
        self.hd_activations = None

        # Neural network layers
        self.pcn = None
        self.rcn = None
        self.head_direction_layer = None

        # Multi-scale network placeholders
        self.pcn_small = None
        self.rcn_small = None
        self.pcn_large = None
        self.rcn_large = None

        # Goal information
        self.goal_location = [-1, 1]
        self.expected_reward = 0

        # History maps for logging (allocated after num_steps is known)
        self.hmap_loc = None   # shape: (num_steps, 3)
        self.hmap_g = None     # shape: (num_steps,)
        self.hmap_pcn = None   # shape: (num_steps, pcn.num_pc)
        self.hmap_bvc = None   # shape: (num_steps, pcn.bvc_layer.num_bvc)
        self.hmap_hdn = None   # shape: (num_steps, n_hd)

        # Directional reward estimates for exploitation
        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)

        # Additional logging and metrics
        self.visitation_map = {}
        self.visitation_map_metrics = {}
        self.weight_change_history = []
        self.metrics_over_time = []

    def initialization(
        self,
        mode=RobotMode.PLOTTING,
        randomize_start_loc: bool = True,
        run_time_hours: int = 1,
        start_loc: Optional[List[int]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        scale_mode: str = "S",  # "S", "L", or "MS"
        small_scale_params: Optional[dict] = None,
        large_scale_params: Optional[dict] = None,
        explore_mthd: Optional[MovementMethod] = MovementMethod.RANDOM_WALK,
        environment_label: str = "default",
    ):
        """
        Extended initialization replicating the original functionality while accepting
        new parameters for exploration method and multi-scale support.
        """
        self.robot_mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.scale_mode = scale_mode
        self.small_scale_params = small_scale_params if small_scale_params is not None else {}
        self.large_scale_params = large_scale_params if large_scale_params is not None else {}
        self.explore_mthd = explore_mthd
        self.environment_label = environment_label

        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))

        # Initialize hardware and sensors.
        self.robot = self.getFromDef("agent")
        self.keyboard = self.getKeyboard()
        if self.keyboard is not None:
            self.keyboard.enable(self.timestep)
        else:
            print("Warning: No keyboard device found; manual control may not be available.")
        self.compass = self.getDevice("compass")
        if self.compass is not None:
            self.compass.enable(self.timestep)
        else:
            print("Warning: No compass device found.")
        self.range_finder = self.getDevice("range-finder")
        if self.range_finder is not None:
            self.range_finder.enable(self.timestep)
        else:
            print("Warning: No range-finder device found.")
        self.left_bumper = self.getDevice("bumper_left")
        if self.left_bumper is not None:
            self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice("bumper_right")
        if self.right_bumper is not None:
            self.right_bumper.enable(self.timestep)
        self.collided = torch.zeros(2, dtype=torch.int32)
        self.rotation_field = self.robot.getField("rotation")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        if self.left_position_sensor is not None:
            self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        if self.right_position_sensor is not None:
            self.right_position_sensor.enable(self.timestep)

        if self.robot_mode == RobotMode.LEARN_OJAS:
            # Optionally clear previous data.
            pass

        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # Load networks.
        if self.scale_mode == "MS":
            self.load_multiscale_networks()
        elif self.scale_mode == "L":
            self.load_network(scale="L", enable_ojas=enable_ojas, enable_stdp=enable_stdp)
        else:
            self.load_network(scale="S", enable_ojas=enable_ojas, enable_stdp=enable_stdp)
            # For single-scale mode, also load the reward cell network.
            self.load_rcn(num_reward_cells=self.num_reward_cells, num_place_cells=self.num_place_cells, num_replay=6)

        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        self.goal_location = [-1, 1]
        self.expected_reward = 0

        if randomize_start_loc:
            while True:
                INITIAL = [rng.uniform(-2.3, 2.3), 0, rng.uniform(-2.3, 2.3)]
                dist_to_goal = np.sqrt((INITIAL[0] - self.goal_location[0])**2 +
                                       (INITIAL[2] - self.goal_location[1])**2)
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        else:
            if start_loc is not None:
                self.robot.getField("translation").setSFVec3f([start_loc[0], 0, start_loc[1]])
                self.robot.resetPhysics()

        # Allocate history maps.
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_g = np.zeros(self.num_steps)
        self.hmap_pcn = torch.zeros((self.num_steps, self.pcn.num_pc), device=self.device, dtype=self.dtype)
        self.hmap_bvc = torch.zeros((self.num_steps, self.pcn.bvc_layer.num_bvc), device=self.device, dtype=self.dtype)
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)
        self.step(self.timestep)
        self.step_count += 1

        self.sense()
        self.compute()

    def load_network(self, scale="S", enable_ojas: Optional[bool] = None, enable_stdp: Optional[bool] = None, device: Optional[str] = None):
        """Loads or initializes a single-scale network.
        (Future versions will use separate parameter dictionaries for small and large scales.)
        """
        if not device:
            device = self.device
        try:
            with open("pcn.pkl", "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
        except:
            bvc = BoundaryVectorCellLayer(
                n_res=self.lidar_resolution,
                n_hd=self.n_hd,
                sigma_theta=1,
                sigma_r=0.5,
                max_dist=10,
                num_bvc_per_dir=50,
                device=device,
            )
            self.pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=self.num_place_cells,
                timestep=self.timestep,
                n_hd=self.n_hd,
                device=device,
            )
            print("Initialized new Place Cell Network.")
        if enable_ojas is not None:
            self.pcn.enable_ojas = enable_ojas
        elif self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.DMTP):
            self.pcn.enable_ojas = True
        if enable_stdp is not None:
            self.pcn.enable_stdp = enable_stdp
        elif self.robot_mode in (RobotMode.LEARN_HEBB, RobotMode.DMTP):
            self.pcn.enable_stdp = True
        return self.pcn

    def load_multiscale_networks(self):
        """Loads both small-scale and large-scale networks for multi-scale operation.
        (For now, the same loading routine is used; later these will be differentiated.)
        """
        print("Loading multi-scale networks (small and large)...")
        self.pcn_small = self.load_network(enable_ojas=None, enable_stdp=None)
        self.rcn_small = self.load_rcn(num_reward_cells=self.num_reward_cells, num_place_cells=self.num_place_cells, num_replay=6)
        self.pcn_large = self.load_network(enable_ojas=None, enable_stdp=None)
        self.rcn_large = self.load_rcn(num_reward_cells=self.num_reward_cells, num_place_cells=self.num_place_cells, num_replay=6)

    def load_rcn(self, num_reward_cells: int, num_place_cells: int, num_replay: int):
        """Loads or initializes the reward cell network."""
        try:
            with open("rcn.pkl", "rb") as f:
                self.rcn = pickle.load(f)
                print("Loaded existing Reward Cell Network.")
        except:
            self.rcn = RewardCellLayer(
                num_reward_cells=num_reward_cells,
                input_dim=num_place_cells,
                num_replay=num_replay,
                device=self.device,
            )
            print("Initialized new Reward Cell Network.")
        return self.rcn

    def run(self):
        """Main control loop.
        Depending on the robot_mode, calls manual_control, explore, exploit, or recording.
        This method now checks the simulation time and, once the time limit is reached,
        it sets the simulation mode to PAUSE to stop the simulation.
        """
        print(f"Starting robot in stage {self.robot_mode}")
        print(f"Goal at {self.goal_location}")
        time_limit = 60 * self.run_time_minutes  # Convert run time in minutes to seconds.
        while True:
            current_time = self.getTime()  # Supervisor.getTime() returns simulation time in seconds.
            if current_time >= time_limit:
                print("Time limit reached. Ending run.")
                break
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()
            elif self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, 
                                      RobotMode.DMTP, RobotMode.PLOTTING, RobotMode.LEARN_ALL):
                # Future: Dispatch to different exploration methods via movement_method module.
                self.explore()
            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()
            elif self.robot_mode == RobotMode.RECORDING:
                self.recording()
            else:
                print("Unknown state. Exiting...")
                break
        
        # Once time limit is reached, pause the simulation.
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
        print("Simulation paused.")


    def explore(self) -> None:
        """Exploration mode using a basic random-walk routine (placeholder).
        Future versions will dispatch via the movement_method module.
        """
        for s in range(self.tau_w):
            self.sense()
            self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)
            actual_reward = self.get_actual_reward()
            self.rcn.td_update(self.pcn.place_cell_activations, next_reward=actual_reward)
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break
            if self.robot_mode in (RobotMode.DMTP, RobotMode.LEARN_HEBB, RobotMode.EXPLOIT):
                self.check_goal_reached()
            self.compute()
            self.forward()
            self.check_goal_reached()
        self.turn(np.random.normal(0, np.deg2rad(30)))

    def exploit(self):
        """Goal-directed navigation using learned reward maps."""
        self.stop()
        self.sense()
        self.compute()
        self.check_goal_reached()
        if self.step_count > self.tau_w:
            action_angle, max_reward, num_steps = 0, 0, 1
            pot_rew = np.empty(self.n_hd)
            pot_e = np.empty(self.n_hd)
            self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations, visit=True)
            max_reward_activation = torch.max(self.rcn.reward_cell_activations)
            if max_reward_activation <= 1e-6:
                print("Reward too low. Switching to exploration.")
                self.explore()
                return
            for d in range(self.n_hd):
                pcn_activations = self.pcn.preplay(d, num_steps=num_steps)
                self.rcn.update_reward_cell_activations(pcn_activations)
                pot_e[d] = torch.norm(pcn_activations, p=1).cpu().numpy()
                pot_rew[d] = torch.max(torch.nan_to_num(self.rcn.reward_cell_activations)).cpu().numpy()
            self.directional_reward_estimates = pot_rew
            angles = np.linspace(0, 2 * np.pi, self.n_hd, endpoint=False)
            angles_tensor = torch.tensor(angles, device=self.device, dtype=torch.float32)
            weights_tensor = torch.tensor(self.directional_reward_estimates, device=self.device, dtype=torch.float32)
            sin_sum = torch.sum(weights_tensor * torch.sin(angles_tensor))
            cos_sum = torch.sum(weights_tensor * torch.cos(angles_tensor))
            action_angle = torch.atan2(sin_sum, cos_sum).item()
            index = int(action_angle // (2 * np.pi / self.n_hd))
            max_reward = pot_rew[index]
            if max_reward <= 1e-3:
                self.explore()
                return
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                self.stop()
                self.rcn.td_update(self.pcn.place_cell_activations, max_reward)
                return
            else:
                if abs(action_angle) > np.pi:
                    action_angle -= np.sign(action_angle) * 2 * np.pi
                angle_to_turn = -np.deg2rad(np.rad2deg(action_angle) - self.current_heading_deg) % (2 * np.pi)
                self.turn(angle_to_turn)
            for s in range(self.tau_w):
                self.sense()
                self.compute()
                self.forward()
                self.check_goal_reached()
                self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)
                actual_reward = self.get_actual_reward()
                self.rcn.td_update(self.pcn.place_cell_activations, next_reward=actual_reward)

    def get_actual_reward(self):
        """Returns 1.0 if the robot is within threshold of the goal, else 0.0."""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        distance_to_goal = np.linalg.norm([curr_pos[0] - self.goal_location[0],
                                            curr_pos[2] - self.goal_location[1]])
        return 1.0 if distance_to_goal <= self.goal_r["exploit"] else 0.0

    def sense(self):
        """Updates sensor data:
        - Reads LiDAR data.
        - Obtains current heading (in degrees) from the compass.
        - Rolls LiDAR data so that the front aligns with index 0.
        - Computes head direction activations.
        - Updates collision status.
        - Advances the simulation timestep.
        """
        boundaries = self.range_finder.getRangeImage()
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))
        self.boundaries = np.roll(boundaries, 2 * self.current_heading_deg)
        current_heading_rad = np.deg2rad(self.current_heading_deg)
        v_in = np.array([np.cos(current_heading_rad), np.sin(current_heading_rad)])
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)
        self.collided[0] = int(self.left_bumper.getValue())
        self.collided[1] = int(self.right_bumper.getValue())
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """Converts compass readings to degrees."""
        rad = np.arctan2(north[1], north[0])
        bearing = (rad - 1.5708) / np.pi * 180.0
        return bearing + 360.0 if bearing < 0 else bearing

    def compute(self):
        """Computes place cell activations and updates history maps."""
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided)
        )
        self.step(self.timestep)
        if self.step_count < self.num_steps:
            self.hmap_loc[self.step_count] = self.robot.getField("translation").getSFVec3f()
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.detach()
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.detach()
            self.hmap_hdn[self.step_count] = self.hd_activations.detach()
            self.hmap_g[self.step_count] = float(self.pcn.bvc_activations.detach().cpu().sum())
        self.step_count += 1

    def check_goal_reached(self):
        """Checks if the goal is reached; if so, auto-pilots and saves data."""
        curr_pos = self.robot.getField("translation").getSFVec3f()
        if (self.robot_mode == RobotMode.EXPLOIT or self.robot_mode == RobotMode.DMTP) and \
           np.allclose(self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["exploit"]):
            self.auto_pilot()
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Started at: {np.array([self.hmap_loc[0][0], self.hmap_loc[0][2]])}")
            print(f"Current position: {np.array([curr_pos[0], curr_pos[2]])}")
            print(f"Time taken: {self.getTime()}")
            include_hmaps = False if self.robot_mode == RobotMode.DMTP else True
            # In the new design, file saving is handled externally.
            # Optionally, you could call a method here if desired.
        elif self.getTime() >= 60 * self.run_time_minutes:
            # When time limit is reached, trigger saving (handled externally).
            pass

    def auto_pilot(self):
        """Auto-pilots the robot to the goal and replays PCN activations."""
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()
        while not np.allclose(self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["explore"]):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]
            if delta_x >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y)), torch.abs(torch.tensor(delta_x))).item()
                desired = np.pi * 2 - theta if delta_y >= 0 else np.pi + theta
            elif delta_y >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y)), torch.abs(torch.tensor(delta_x))).item()
                desired = np.pi / 2 - theta
            else:
                theta = torch.atan2(torch.abs(torch.tensor(delta_x)), torch.abs(torch.tensor(delta_y))).item()
                desired = np.pi - theta
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))
            self.sense()
            self.compute()
            self.forward()
            s_start += 1
        self.rcn.replay(pcn=self.pcn)

    def manual_control(self):
        """Allows manual keyboard control."""
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
        """Rotates the robot continuously."""
        speed = self.max_speed * speed_factor
        self.left_speed = speed * direction
        self.right_speed = -speed * direction
        self.move()

    def forward(self):
        """Moves the robot forward at maximum speed."""
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()

    def turn(self, angle: float, circle: bool = False):
        """Turns the robot by a specified angle (radians)."""
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
        """Stops the robot."""
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def move(self):
        """Updates motor commands based on current speed settings."""
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(self.left_speed)
        self.right_motor.setVelocity(self.right_speed)

    def compute_path_length(self):
        """Computes the total path length from history."""
        path_length = 0
        for i in range(self.hmap_loc[:, 0].shape[0] - 1):
            current_position = np.array([self.hmap_loc[:, 2][i], self.hmap_loc[:, 0][i]])
            next_position = np.array([self.hmap_loc[:, 2][i + 1], self.hmap_loc[:, 0][i + 1]])
            path_length += np.linalg.norm(next_position - current_position)
        return path_length

    # Note: The save() and clear() methods have been removed from the driver.
    # Data saving is now entirely handled by the TrialManager.

    def reset_run(self, randomize_start_loc: bool = True, start_loc: Optional[List[int]] = None):
        """
        Resets the internal state for a new run without restarting the simulation.
        - Resets the robot's start location.
        - Clears history maps and resets the step counter.
        - Clears temporary logging data.
        """
        # Reset step counter and reallocate history maps.
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_g = np.zeros(self.num_steps)
        self.hmap_pcn = torch.zeros((self.num_steps, self.pcn.num_pc), device=self.device, dtype=self.dtype)
        self.hmap_bvc = torch.zeros((self.num_steps, self.pcn.bvc_layer.num_bvc), device=self.device, dtype=self.dtype)
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)
        
        self.visitation_map = {}
        self.visitation_map_metrics = {}
        self.weight_change_history = []
        self.metrics_over_time = []
        
        # Reset robot position
        if randomize_start_loc:
            while True:
                INITIAL = [rng.uniform(-2.3, 2.3), 0, rng.uniform(-2.3, 2.3)]
                dist_to_goal = np.sqrt((INITIAL[0] - self.goal_location[0])**2 +
                                       (INITIAL[2] - self.goal_location[1])**2)
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(INITIAL)
        else:
            if start_loc is not None:
                self.robot.getField("translation").setSFVec3f([start_loc[0], 0, start_loc[1]])
        self.robot.resetPhysics()
        print("Driver state has been reset for a new run.")
