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

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode

# --- PyTorch seeds / random ---
torch.manual_seed(5)
np.random.seed(5)
rng = default_rng(5)  # or keep it as is
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
        hmap_x (ndarray): History of x-coordinates.
        hmap_y (ndarray): History of y-coordinates.
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
        randomize_start_loc: bool = True,
        run_time_hours: int = 1,
        start_loc: Optional[List[int]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
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
        self.robot_mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        # Model parameters
        self.num_place_cells = 200
        self.num_reward_cells = 1
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 5  # time constant for the window function

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        # Initialize hardware components and sensors
        self.robot = self.getFromDef("agent")  # Placeholder for robot instance
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

        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Initialize boundaries
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # Initialize layers
        self.load_pcn(
            num_place_cells=self.num_place_cells,
            n_hd=self.n_hd,
            timestep=self.timestep,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
        )
        self.load_rcn(
            num_reward_cells=self.num_reward_cells,
            num_place_cells=self.num_place_cells,
            num_replay=6,
        )
        self.head_direction_layer = HeadDirectionLayer(
            num_cells=self.n_hd, device="cpu"
        )

        # Initialize goal
        self.goal_location = [-1, 1]
        self.expected_reward = 0

        if randomize_start_loc:
            while True:
                INITIAL = [rng.uniform(-2.3, 2.3), 0, rng.uniform(-2.3, 2.3)]
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

        # Generate hmaps ("history maps") for different items to record
        self.hmap_loc = np.zeros(
            (self.num_steps, 3)
        )  # coordinates in the format [X, Z, Y]
        self.hmap_g = np.zeros(self.num_steps)  # goal estimates
        self.hmap_pcn = torch.zeros(
            (self.num_steps, self.pcn.num_pc),
            device=self.device,
            dtype=torch.float32,
        )  # Place cell network activation history
        self.hmap_bvc = torch.zeros(
            (self.num_steps, self.pcn.bvc_layer.num_bvc),
            device=self.device,
            dtype=torch.float32,
        )  # Boundary Vector Cell Network activation history
        self.hmap_hdn = torch.zeros(
            (self.num_steps, self.n_hd),
            device="cpu",
            dtype=torch.float32,
        )  # Head direction network activation history

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device)
        self.step(self.timestep)
        self.step_count += 1

        self.sense()
        self.compute()

    def load_pcn(
        self,
        num_place_cells: int,
        n_hd: int,
        timestep: int,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        device: Optional[str] = None,
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
                n_hd=n_hd,
                sigma_theta=1,
                sigma_r=0.5,
                max_dist=10,
                num_bvc_per_dir=50,
                device=device,
            )

            self.pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=num_place_cells,
                timestep=timestep,
                n_hd=n_hd,
                device=device,
            )
            print("Initialized new Place Cell Network.")

        if enable_ojas is not None:
            self.pcn.enable_ojas = enable_ojas
        elif (
            self.robot_mode == RobotMode.LEARN_OJAS
            or self.robot_mode == RobotMode.LEARN_HEBB
            or self.robot_mode == RobotMode.DMTP
        ):
            self.pcn.enable_ojas = True

        if enable_stdp is not None:
            self.pcn.enable_stdp = enable_stdp
        elif (
            self.robot_mode == RobotMode.LEARN_HEBB or self.robot_mode == RobotMode.DMTP
        ):
            self.pcn.enable_stdp = True

        return self.pcn

    def load_rcn(self, num_reward_cells: int, num_place_cells: int, num_replay: int):
        """Loads or initializes the reward cell network.

        Args:
            num_reward_cells (int): Number of reward cells in the network.
            num_place_cells (int): Number of place cells providing input.
            num_replay (int): Number of replay iterations for memory consolidation.

        Returns:
            RewardCellLayer: The loaded or newly initialized reward cell network.
        """
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

    ########################################### RUN LOOP ###########################################

    def run(self):
        """Runs the main control loop of the robot.

        The method manages the robot's behavior based on its current mode:
        - MANUAL_CONTROL: Allows user keyboard control
        - LEARN_OJAS/LEARN_HEBB/DMTP/PLOTTING: Runs exploration behavior
        - EXPLOIT: Runs goal-directed navigation
        - RECORDING: Records sensor data
        """

        print(f"Starting robot in stage {self.robot_mode}")
        print(f"Goal at {self.goal_location}")

        while True:
            # Handle the robot's state
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

            # TODO Do we need this reward code?
            # Update the reward cell activations
            self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)

            # Determine the actual reward (you may need to define how to calculate this)
            actual_reward = self.get_actual_reward()

            # Perform TD update
            self.rcn.td_update(
                self.pcn.place_cell_activations, next_reward=actual_reward
            )

            if torch.any(self.collided):
                random_angle = np.random.uniform(
                    -np.pi, np.pi
                )  # Random angle between -180 and 180 degrees (in radians)
                self.turn(random_angle)
                break

            if (
                self.robot_mode == RobotMode.DMTP
                or self.robot_mode == RobotMode.LEARN_HEBB
                or self.robot_mode == RobotMode.EXPLOIT
            ):
                self.check_goal_reached()

            self.compute()
            self.forward()
            self.check_goal_reached()

        self.turn(np.random.normal(0, np.deg2rad(30)))  # Choose a new random direction

    ########################################### EXPLOIT ###########################################

    def exploit(self):
        """Executes goal-directed navigation using learned reward maps.

        The robot:
        - Computes potential rewards in different directions
        - Selects movement direction with highest expected reward
        - Updates place and reward cell activations during movement
        - Monitors goal proximity and collision status
        - Switches to exploration if reward expectations are too low

        Returns:
            None
        """

        # Stop movement and update sensor readings
        self.stop()
        self.sense()
        self.compute()
        self.check_goal_reached()

        # Proceed only if enough steps have been taken
        if self.step_count > self.tau_w:
            # Initialize variables
            action_angle, max_reward, num_steps = 0, 0, 1
            pot_rew = np.empty(self.n_hd)
            pot_e = np.empty(self.n_hd)

            # Update reward cell network based on current place cell activations
            self.rcn.update_reward_cell_activations(
                self.pcn.place_cell_activations, visit=True
            )

            # Check if the reward is too low; if so, switch to exploration
            max_reward_activation = torch.max(self.rcn.reward_cell_activations)
            if max_reward_activation <= 1e-6:
                print("Reward too low. Switching to exploration.")
                self.explore()
                return

            # Calculate potential reward and energy for each direction
            for d in range(self.n_hd):
                # Simulate future place cell activations in direction 'd'
                pcn_activations = self.pcn.preplay(d, num_steps=num_steps)
                # Update reward cell activations based on the simulated activations
                self.rcn.update_reward_cell_activations(pcn_activations)

                # Store potential energy and reward
                pot_e[d] = torch.norm(pcn_activations, p=1).cpu().numpy()
                pot_rew[d] = (
                    torch.max(torch.nan_to_num(self.rcn.reward_cell_activations))
                    .cpu()
                    .numpy()
                )

            # Update directional reward estimates based on computed potential rewards
            self.directional_reward_estimates = pot_rew

            # Calculate the action angle using circular mean
            angles = np.linspace(0, 2 * np.pi, self.n_hd, endpoint=False)
            angles_tensor = torch.tensor(
                angles, device=self.device, dtype=torch.float32
            )

            # Convert the directional reward estimates (a NumPy array) into a PyTorch tensor.
            weights_tensor = torch.tensor(
                self.directional_reward_estimates,
                device=self.device,
                dtype=torch.float32,
            )

            sin_sum = torch.sum(weights_tensor * torch.sin(angles_tensor))
            cos_sum = torch.sum(weights_tensor * torch.cos(angles_tensor))
            action_angle = torch.atan2(sin_sum, cos_sum).item()

            # Determine the maximum reward for the chosen action
            index = int(action_angle // (2 * np.pi / self.n_hd))
            max_reward = pot_rew[index]

            # If the max reward is too low, switch to exploration
            if max_reward <= 1e-3:
                self.explore()
                return

            # Handle collision by turning and updating the reward cell network
            if torch.any(self.collided):
                # Generate a random angle between -180 and 180 degrees (in radians)
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                self.stop()
                self.rcn.td_update(self.pcn.place_cell_activations, max_reward)
                return
            else:
                # Adjust the action angle if necessary
                if abs(action_angle) > np.pi:
                    action_angle -= np.sign(action_angle) * 2 * np.pi
                # Calculate the angle to turn towards the desired heading
                angle_to_turn = -np.deg2rad(
                    np.rad2deg(action_angle) - self.current_heading_deg
                ) % (2 * np.pi)
                self.turn(angle_to_turn)

            # Move forward for a set duration while updating the place cell state
            for s in range(self.tau_w):
                self.sense()
                self.compute()
                self.forward()
                # Accumulate place cell activations
                self.check_goal_reached()

                # Update reward cell activations and perform TD update
                self.rcn.update_reward_cell_activations(self.pcn.place_cell_activations)
                actual_reward = self.get_actual_reward()
                self.rcn.td_update(
                    self.pcn.place_cell_activations, next_reward=actual_reward
                )

    def get_actual_reward(self):
        """Determines the actual reward for the agent at the current state.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        distance_to_goal = np.linalg.norm(
            [curr_pos[0] - self.goal_location[0], curr_pos[2] - self.goal_location[1]]
        )

        if distance_to_goal <= self.goal_r["exploit"]:
            return 1.0  # Reward for reaching the goal
        else:
            return 0.0  # No reward otherwise

    ########################################### SENSE ###########################################

    def sense(self):
        """Updates the robot's perception of its environment.

        Updates orientation, distance to obstacles (boundaries), head direction cell
        activations, and collision detection. The method performs the following steps:
        1. Captures LiDAR data for obstacle distances
        2. Gets robot heading and aligns LiDAR data
        3. Computes head direction vector and cell activations
        4. Updates collision status from bumper sensors
        5. Advances to next timestep

        Updates the following instance variables:
            boundaries: LiDAR range data (720 points)
            current_heading_deg: Current heading in degrees
            hd_activations: Head direction cell activations
            collided: Collision status from bumpers
        """

        # 1. Capture distance data from the range finder (LiDAR), which provides 720 points around the robot.
        # Shape: (720,)
        boundaries = self.range_finder.getRangeImage()

        # 2. Get the robot's current heading in degrees using the compass and convert it to an integer.
        # Shape: scalar (int)
        self.current_heading_deg = int(
            self.get_bearing_in_degrees(self.compass.getValues())
        )

        # 3. Roll the LiDAR data based on the current heading to align the 'front' with index 0.
        # Shape: (720,) - LiDAR data remains 720 points, but shifted according to the robot's current heading.
        self.boundaries = np.roll(boundaries, 2 * self.current_heading_deg)
        # self.boundaries = torch.tensor(boundaries, dtype=self.dtype, device=self.device)

        # 4. Convert the current heading from degrees to radians.
        # Shape: scalar (float) - Current heading of the robot in radians.
        current_heading_rad = np.deg2rad(self.current_heading_deg)

        # 6. Calculate the current heading vector from the heading in radians.
        # Shape: (2,) - A 2D vector representing the robot's current heading direction: [cos(theta), sin(theta)].
        v_in = np.array([np.cos(current_heading_rad), np.sin(current_heading_rad)])

        # 7. Compute the activations of the head direction cells based on the current heading vector (v_in).
        # Shape: (self.num_cells,) - A 1D array where each element represents the activation of a head direction cell.
        self.hd_activations = self.head_direction_layer.get_hd_activation(v_in=v_in)

        # 8. Update the collision status using the left bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the left bumper, 0 otherwise.
        self.collided[0] = int(self.left_bumper.getValue())

        # 9. Update the collision status using the right bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the right bumper, 0 otherwise.
        self.collided[1] = int(self.right_bumper.getValue())

        # 10. Proceed to the next timestep in the robot's control loop.
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """Converts compass readings to bearing in degrees.

        Args:
            north (List[float]): List containing the compass sensor values [x, y, z].

        Returns:
            float: Bearing angle in degrees (0-360), where 0 is North.
        """
        rad = np.arctan2(north[1], north[0])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0

        return bearing

    ########################################### COMPUTE ###########################################

    def compute(self):
        """
        Compute the activations of place cells and handle the environment interactions.
        """
        # Compute the place cell network activations
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided),
        )
        # self.pcn.bvc_layer.plot_activation(distances=self.boundaries)

        # Advance the timestep and update position
        self.step(self.timestep)

        # Update place cell and sensor maps
        if self.step_count < self.num_steps:
            self.hmap_loc[self.step_count] = self.robot.getField(
                "translation"
            ).getSFVec3f()
            # Already on GPU
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.detach()
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.detach()
            self.hmap_hdn[self.step_count] = self.hd_activations.detach()
            self.hmap_g[self.step_count] = float(
                self.pcn.bvc_activations.detach().cpu().sum()
            )

        # Increment timestep
        self.step_count += 1

    ########################################### CHECK GOAL REACHED ###########################################

    def check_goal_reached(self):
        """
        Check if the robot has reached the goal and perform necessary actions when the goal is reached.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        # DMTP Mode and exploit mode both stop when they both see the goal
        if (
            self.robot_mode == RobotMode.EXPLOIT or self.robot_mode == RobotMode.DMTP
        ) and np.allclose(
            self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["exploit"]
        ):
            self.auto_pilot()  # Navigate to the goal slowly and call rcn.replay()
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Started at: {np.array([self.hmap_loc[0][0], self.hmap_loc[0][2]])}")
            print(f"Current position: {np.array([curr_pos[0], curr_pos[2]])}")
            print(f"Time taken: {self.getTime()}")

            if self.robot_mode == RobotMode.DMTP:
                include_hmaps = False
            else:
                include_hmaps = True

            self.save(
                include_rcn=(self.robot_mode != RobotMode.EXPLOIT),
                include_pcn=(self.robot_mode != RobotMode.EXPLOIT),
                include_hmaps=include_hmaps,
            )
        elif self.getTime() >= 60 * self.run_time_minutes:
            self.save()

    ########################################### AUTO PILOT ###########################################

    def auto_pilot(self):
        print("Auto-piloting to the goal...")
        s_start = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()
        while not np.allclose(
            self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["explore"]
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]

            if delta_x >= 0:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_y)), torch.abs(torch.tensor(delta_x))
                ).item()
                desired = np.pi * 2 - theta if delta_y >= 0 else np.pi + theta
            elif delta_y >= 0:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_y)), torch.abs(torch.tensor(delta_x))
                ).item()
                desired = np.pi / 2 - theta
            else:
                theta = torch.atan2(
                    torch.abs(torch.tensor(delta_x)), torch.abs(torch.tensor(delta_y))
                ).item()
                desired = np.pi - theta
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))

            self.sense()
            self.compute()
            self.forward()
            s_start += 1

        # Replay the place cell activations
        self.rcn.replay(pcn=self.pcn)

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

    def save(
        self,
        include_pcn: bool = True,
        include_rcn: bool = True,
        include_hmaps: bool = True,
    ):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), and optionally
        the maps that store the agent's movement and activations.

        Parameters:
            include_maps (bool): If True, saves the history of the agent's path and activations.
        """

        files_saved = []
        # Save the Place Cell Network (PCN)
        if include_pcn:
            with open("pcn.pkl", "wb") as output:
                pickle.dump(self.pcn, output)
                files_saved.append("pcn.pkl")

        # Save the Reward Cell Network (RCN)
        if include_rcn:
            with open("rcn.pkl", "wb") as output:
                pickle.dump(self.rcn, output)
                files_saved.append("rcn.pkl")

        # Save the history maps if specified
        if include_hmaps:
            with open("hmap_loc.pkl", "wb") as output:
                pickle.dump(self.hmap_loc[: self.step_count], output)
                files_saved.append("hmap_loc.pkl")
            with open("hmap_pcn.pkl", "wb") as output:
                pcn_cpu = self.hmap_pcn[: self.step_count].cpu().numpy()
                pickle.dump(pcn_cpu[: self.step_count], output)
                files_saved.append("hmap_pcn.pkl")
            with open("hmap_g.pkl", "wb") as output:
                pickle.dump(self.hmap_g[: self.step_count], output)
                files_saved.append("hmap_g.pkl")
            with open("hmap_hdn.pkl", "wb") as output:
                pickle.dump(self.hmap_hdn[: self.step_count], output)
                files_saved.append("hmap_hdn.pkl")
            with open("hmap_bvc.pkl", "wb") as output:
                bvc_cpu = self.hmap_bvc[: self.step_count].cpu().numpy()
                pickle.dump(bvc_cpu, output)
                files_saved.append("hmap_bvc.pkl")

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
        and the history maps by removing their corresponding pickle files.
        """
        files_to_remove = [
            "pcn.pkl",
            "rcn.pkl",
            "hmap_loc.pkl",
            "hmap_pcn.pkl",
            "hmap_bvc.pkl",
            "hmap_g.pkl",
            "hmap_hdn.pkl",
        ]

        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass  # Ignore if the file does not exist

        print("State files cleared.")
