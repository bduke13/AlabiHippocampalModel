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
from core.layers.place_cell_layer_with_grid import PlaceCellLayerWithGrid
from core.layers.grid_cell_layer import OscillatoryInterferenceGridCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode

# --- PyTorch seeds / random ---
# torch.manual_seed(5)
# np.random.seed(5)
# rng = default_rng(5)  # or keep it as is
np.set_printoptions(precision=2)

class DriverGrid(Supervisor):
    def __init__(self):
        super(DriverGrid, self).__init__()
        # Use self.getFromDef("agent") to get robot instance, consistent with driver.py
        self.robot = self.getFromDef("agent")  # Removed robot: RobotInterface parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.step_count = 0

    def initialization(
        self,
        mode: RobotMode,
        run_time_hours: int = 2,
        randomize_start_loc: bool = True,
        start_loc: Optional[List[float]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        max_dist: float = 10,
        show_bvc_activation: bool = False,
        num_place_cells: int = 500,  # Added for compatibility and grid integration
        num_modules: int = 4,  # Grid cell parameter
        grid_spacings: List[float] = [0.3, 0.5, 0.7, 1.0],  # Grid cell parameter
        num_cells_per_module: int = 10,  # Grid cell parameter
    ):
        """Initialize the robot, neural layers, and history maps with grid cell integration.

        Args:
            mode (RobotMode): Operating mode for the robot.
            run_time_hours (int): Total simulation run time in hours.
            randomize_start_loc (bool): Whether to randomize the robot's starting location.
            start_loc (Optional[List[float]]): Specific starting location [x, y].
            enable_ojas (Optional[bool]): Enable Oja's learning rule.
            enable_stdp (Optional[bool]): Enable Spike-Timing-Dependent Plasticity.
            world_name (Optional[str]): Name of the simulation world.
            goal_location (Optional[List[float]]): Target [x, y] coordinates.
            max_dist (float): Maximum distance for boundary detection.
            show_bvc_activation (bool): Whether to visualize BVC activations.
            num_place_cells (int): Number of place cells.
            num_modules (int): Number of grid cell modules.
            grid_spacings (List[float]): Spatial scales for grid modules.
            num_cells_per_module (int): Number of grid cells per module.
        """
        self.robot_mode = mode
        self.show_bvc_activation = show_bvc_activation

        # Set world name and directories (unchanged from driver.py)
        if world_name is None:
            world_path = self.getWorldPath()
            world_name = os.path.splitext(os.path.basename(world_path))[0]
        self.world_name = world_name
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters (unchanged, except num_place_cells is now a parameter)
        self.num_place_cells = num_place_cells
        self.num_bvc_per_dir = 50
        self.sigma_r = 0.5
        self.sigma_theta = 1
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 5

        # Robot parameters (unchanged from driver.py)
        self.max_speed = 16
        self.max_dist = max_dist
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

        # Set starting location (unchanged from driver.py, but use np.random)
        if randomize_start_loc:
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
            self.robot.getField("translation").setSFVec3f([start_loc[0], 0, start_loc[1]])
            self.robot.resetPhysics()

        # Initialize sensors and motors (unchanged from driver.py)
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

        # Initialize boundaries (unchanged from driver.py)
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        # Initialize layers
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Initialize head direction layer (unchanged, but ensure device consistency)
        self.head_direction_layer = HeadDirectionLayer(
            num_cells=self.n_hd, device=self.device
        )

        # Initialize BVC layer (unchanged from driver.py)
        bvc = BoundaryVectorCellLayer(
            n_res=self.lidar_resolution,
            n_hd=self.n_hd,
            sigma_theta=self.sigma_theta,
            sigma_r=self.sigma_r,
            max_dist=self.max_dist,
            num_bvc_per_dir=self.num_bvc_per_dir,
            device=self.device,
        )

        # Initialize grid cell layer (new for grid integration)
        self.gcn = OscillatoryInterferenceGridCellLayer(
            num_modules=num_modules,
            grid_spacings=grid_spacings,
            num_cells_per_module=num_cells_per_module,
            device=self.device,
            dtype=self.dtype,
        )

        # Load or initialize place cell network with grid inputs (modified)
        self.load_pcn(
            bvc_layer=bvc,
            num_place_cells=num_place_cells,
            num_grid_cells=self.gcn.total_grid_cells,
            timestep=self.timestep,
            n_hd=self.n_hd,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
            device=self.device,
        )

        # Load or initialize reward cell network (unchanged from driver.py)
        self.load_rcn(
            num_place_cells=num_place_cells,
            num_replay=3,
            learning_rate=0.1,
            device=self.device,
        )

        # Initialize history maps (added hmap_gcn for grid cells)
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_pcn = torch.zeros(
            (self.num_steps, num_place_cells), device=self.device, dtype=self.dtype
        )
        self.hmap_bvc = torch.zeros(
            (self.num_steps, bvc.num_bvc), device=self.device, dtype=self.dtype
        )
        self.hmap_hdn = torch.zeros(
            (self.num_steps, self.n_hd), device=self.device, dtype=self.dtype
        )
        self.hmap_gcn = torch.zeros(
            (self.num_steps, self.gcn.total_grid_cells), device=self.device, dtype=self.dtype
        )

        # Additional setup (unchanged from driver.py)
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
    ):
        """Load or initialize the place cell network with grid inputs.

        Args:
            bvc_layer (BoundaryVectorCellLayer): BVC layer instance.
            num_place_cells (int): Number of place cells.
            num_grid_cells (int): Number of grid cells.
            timestep (int): Simulation timestep in ms.
            n_hd (int): Number of head direction cells.
            device (torch.device): Computation device.
            enable_ojas (Optional[bool]): Enable Oja's rule.
            enable_stdp (Optional[bool]): Enable STDP.
        """
        try:
            network_path = os.path.join(self.network_dir, "pcn_with_grid.pkl")
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

    # load_rcn method (unchanged from driver.py)
    def load_rcn(
        self,
        num_place_cells: int,
        num_replay: int,
        learning_rate: float,
        device: torch.device,
    ):
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
            hd_activations=self.hd_activations,
            grid_activations=grid_activations,
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
            # Save Grid Cell Network (GCN)
            gcn_path = os.path.join(self.network_dir, "gcn.pkl")
            with open(gcn_path, "wb") as output:
                pickle.dump(self.gcn, output)
                files_saved.append(gcn_path)

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
            # Save grid cell history map
            hmap_gcn_path = os.path.join(self.hmap_dir, "hmap_gcn.pkl")
            with open(hmap_gcn_path, "wb") as output:
                gcn_cpu = self.hmap_gcn[: self.step_count].cpu().numpy()
                pickle.dump(gcn_cpu, output)
                files_saved.append(hmap_gcn_path)

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