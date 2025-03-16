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

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode

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
        mode=RobotMode.PLOTTING,
        run_time_hours: int = 2,
        randomize_start_loc: bool = True,
        start_loc: Optional[List[int]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        max_dist: float = 10,
        show_bvc_activation: bool = False,
    ):
        """Initializes the Driver class with specified parameters and sets up the robot's sensors and neural networks.

        Args:
            mode (RobotMode): The operating mode for the robot.
            randomize_start_loc (bool, optional): Whether to randomize the agent's spawn location.
                Defaults to True.
            run_time_hours (int, optional): Total run time for the simulation in hours.
                Defaults to 1.
            start_loc (Optional[List[int]], optional): Specific starting location coordinates [x,y].
                Defaults to None.
            enable_ojas (Optional[bool], optional): Flag to enable Oja's learning rule.
                If None, determined by robot mode. Defaults to None.
            enable_stdp (Optional[bool], optional): Flag to enable Spike-Timing-Dependent Plasticity.
                If None, determined by robot mode. Defaults to None.

        Returns:
            None
        """
        # Set the robot mode and device
        self.robot = self.getFromDef("agent")  # Placeholder for robot instance
        self.robot_mode = mode
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = torch.float32
        self.show_bvc_activation = show_bvc_activation

        # Set the world name and directories for saving data
        if world_name is None:
            world_path = self.getWorldPath()  # Get the full path to the world file
            world_name = os.path.splitext(os.path.basename(world_path))[
                0
            ]  # Extract just the world name
        self.world_name = world_name

        # Construct directory paths
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters
        self.num_place_cells = 3000
        self.num_bvc_per_dir = 50
        self.sigma_r = 4
        self.sigma_theta = 1
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 5  # time constant for the window function

        # Robot parameters
        self.max_speed = 16
        self.max_dist = max_dist  # Defaults to 10. This should be the longest possible distance (m) that a lidar can detect in the specific world
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        if goal_location is not None:
            self.goal_location = goal_location
        else:
            self.goal_location = [-3, 3]

        if randomize_start_loc:
            while True:
                INITIAL = [random.uniform(-2.3, 2.3), 0, random.uniform(-2.3, 2.3)]
                # Check if distance to goal is at least 1 meter
                dist_to_goal = np.sqrt(
                    (INITIAL[0] - self.goal_location[0]) ** 2
                    + (INITIAL[2] - self.goal_location[1]) ** 2
                )
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        else:
            self.robot.getField("translation").setSFVec3f(
                [start_loc[0], 0, start_loc[1]]
            )
            self.robot.resetPhysics()

        # Initialize hardware components and sensors
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

        # Initialize boundaries
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # Initialize layers
        if (
            self.robot_mode == RobotMode.LEARN_OJAS
        ):  # Delete existing pkls if in LEARN_OJAS
            self.clear()

        self.load_pcn(
            num_place_cells=self.num_place_cells,
            n_hd=self.n_hd,
            timestep=self.timestep,
            sigma_r=self.sigma_r,
            sigma_theta=self.sigma_theta,
            num_bvc_per_dir=self.num_bvc_per_dir,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
            device=self.device,
        )
        self.load_rcn(
            num_place_cells=self.num_place_cells,
            num_replay=3,
            learning_rate=0.1,
            device=self.device,
        )
        self.head_direction_layer = HeadDirectionLayer(
            num_cells=self.n_hd, device=torch.device("cpu")
        )

        # Initialize hmaps (history maps) to record activations and positions
        # NOTE: 2D cartesian coordinates are saved in X and Z
        self.hmap_loc = np.zeros((self.num_steps, 3))
        # Place cell network activation history
        self.hmap_pcn = torch.zeros(
            (self.num_steps, self.pcn.num_pc),
            device=self.device,
            dtype=torch.float32,
        )
        # Boundary Vector Cell Network activation history
        self.hmap_bvc = torch.zeros(
            (self.num_steps, self.pcn.bvc_layer.num_bvc),
            device=self.device,
            dtype=torch.float32,
        )
        # Head direction network activation history
        self.hmap_hdn = torch.zeros(
            (self.num_steps, self.n_hd),
            device="cpu",
            dtype=torch.float32,
        )

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)
        self.step(self.timestep)
        self.step_count += 1

        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    def load_pcn(
        self,
        num_place_cells: int,
        n_hd: int,
        timestep: int,
        sigma_theta: float,
        sigma_r: float,
        num_bvc_per_dir: int,
        device: torch.device,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
    ):
        """Loads an existing place cell network from disk or initializes a new one.

        Args:
            num_place_cells (int): Number of place cells in the network.
            n_hd (int): Number of head direction cells.
            timestep (int): Time step duration in milliseconds.
            enable_ojas (Optional[bool], optional): Flag to enable Oja's learning rule.
                If None, determined by robot mode. Defaults to None.
            enable_stdp (Optional[bool], optional): Flag to enable Spike-Timing-Dependent Plasticity.
                If None, determined by robot mode. Defaults to None.

        Returns:
            PlaceCellLayer: The loaded or newly initialized place cell network.
        """
        try:
            network_path = os.path.join(self.network_dir, "pcn.pkl")
            with open(network_path, "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing PCN from", network_path)
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except:
            bvc = BoundaryVectorCellLayer(
                n_res=self.lidar_resolution,
                n_hd=n_hd,
                sigma_theta=sigma_theta,
                sigma_r=sigma_r,
                max_dist=self.max_dist,
                num_bvc_per_dir=num_bvc_per_dir,
                device=device,
            )

            self.pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=num_place_cells,
                timestep=timestep,
                n_hd=n_hd,
                device=device,
            )
            print("Initialized new PCN")

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

        return self.pcn

    def load_rcn(
        self,
        num_place_cells: int,
        num_replay: int,
        learning_rate: float,
        device: torch.device,
    ):
        """Loads or initializes the reward cell network.

        Args:
            num_place_cells (int): Number of place cells providing input.
            num_replay (int): Number of replay iterations for memory consolidation.

        Returns:
            RewardCellLayer: The loaded or newly initialized reward cell network.
        """
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

    ########################################### RUN LOOP ###########################################

    def run(self):
        """Runs the main control loop of the robot.

        The method manages the robot's behavior based on its current mode:
        - MANUAL_CONTROL: Allows user keyboard control
        - LEARN_OJAS/LEARN_HEBB/DMTP/PLOTTING: Runs exploration behavior
        - EXPLOIT: Runs goal-directed navigation
        - RECORDING: Records sensor data
        """

        print(f"Starting robot in {self.robot_mode}")
        print(f"Goal at {self.goal_location}")

        while True:
            if self.robot_mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()

            elif (
                self.robot_mode == RobotMode.LEARN_OJAS
                or self.robot_mode == RobotMode.LEARN_HEBB
                or self.robot_mode == RobotMode.DMTP
                or self.robot_mode == RobotMode.PLOTTING
            ):
                self.explore()

            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()

            elif self.robot_mode == RobotMode.RECORDING:
                self.recording()

            else:
                print("Unknown state. Exiting...")
                break

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
        Follows the reward gradient to reach the goal location, incorporating wall avoidance.
        """
        # -------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        # -------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()
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
        # Update place cell activations based on sensor data
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
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

        if (
            self.robot_mode
            in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING)
            and self.getTime() >= 60 * self.run_time_minutes
        ):
            self.stop()
            self.save(
                include_pcn=self.robot_mode != RobotMode.PLOTTING,
                include_rcn=self.robot_mode != RobotMode.PLOTTING,
                include_hmaps=True,
            )

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
        include_hmaps: bool = False,
    ):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), and optionally
        the maps that store the agent's movement and activations.
        """
        files_saved = []

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Save the Place Cell Network (PCN)
        if include_pcn:
            pcn_path = os.path.join(self.network_dir, "pcn.pkl")
            with open(pcn_path, "wb") as output:
                pickle.dump(self.pcn, output)
                files_saved.append(pcn_path)

        # Save the Reward Cell Network (RCN)
        if include_rcn:
            rcn_path = os.path.join(self.network_dir, "rcn.pkl")
            with open(rcn_path, "wb") as output:
                pickle.dump(self.rcn, output)
                files_saved.append(rcn_path)

        # Save the history maps if specified
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
                pickle.dump(self.hmap_hdn[: self.step_count], output)
                files_saved.append(hmap_hdn_path)

            hmap_bvc_path = os.path.join(self.hmap_dir, "hmap_bvc.pkl")
            with open(hmap_bvc_path, "wb") as output:
                bvc_cpu = self.hmap_bvc[: self.step_count].cpu().numpy()
                pickle.dump(bvc_cpu, output)
                files_saved.append(hmap_bvc_path)

        # Show a message box to confirm saving
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes("-topmost", True)  # Always keep the window on top
        root.update()
        messagebox.showinfo("Information", "Press OK to save data")
        root.destroy()  # Destroy the main window

        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def clear(self):
        """
        Clears the saved state files for the Place Cell Network (PCN), Reward Cell Network (RCN),
        and the history maps by removing their corresponding pickle files from the appropriate directories.
        """
        # Network files in network_dir
        network_files = [
            os.path.join(self.network_dir, "pcn.pkl"),
            os.path.join(self.network_dir, "rcn.pkl"),
        ]

        # History map files in hmap_dir
        hmap_files = [
            os.path.join(self.hmap_dir, "hmap_loc.pkl"),
            os.path.join(self.hmap_dir, "hmap_pcn.pkl"),
            os.path.join(self.hmap_dir, "hmap_bvc.pkl"),
            os.path.join(self.hmap_dir, "hmap_hdn.pkl"),
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
