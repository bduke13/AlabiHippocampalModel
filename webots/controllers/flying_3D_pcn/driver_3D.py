import numpy as np
import torch
import pickle
import random
import os
from typing import Optional, List
from controller import Supervisor


# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode
from core.layers.boundary_vector_cell_layer_3D import BoundaryVectorCellLayer3D
from core.layers.head_direction_layer_3D import HeadDirectionLayer3D
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer

np.set_printoptions(precision=2)
PI = torch.tensor(np.pi)


class DriverFlying(Supervisor):

    def initialization(
        self,
        mode=RobotMode.PLOTTING,
        randomize_start_loc: bool = True,
        run_time_hours: int = 1,
        preferred_va: Optional[List[float]] = None,
        sigma_d: Optional[List[float]] = None,
        sigma_va: Optional[List[float]] = None,
        sigma_a: Optional[List[float]] = None,
        num_bvc_per_dir: Optional[int] = None,
        start_loc: Optional[List[int]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        visual_bvc: bool = None,
        file_prefix: str = "",
    ):
        self.robot_mode = mode
        self.file_prefix = file_prefix
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Model parameters
        self.num_place_cells = 5000
        self.num_reward_cells = 1
        self.n_hd_bvc = 8
        self.n_hd_hdn = 20
        self.timestep = 32 * 6
        self.tau_w = 10  # time constant for the window function

        # Parameters for 3D BVC
        # Define preferred vertical angles and corresponding sigma values
        self.preferred_vertical_angles = preferred_va
        self.visual_bvc = visual_bvc

        self.sigma_d_list = (
            sigma_d
            if sigma_d is not None
            else [0.5] * len(self.preferred_vertical_angles)
        )
        self.sigma_ang_list = (
            sigma_a
            if sigma_a is not None
            else [0.2] * len(self.preferred_vertical_angles)
        )
        self.sigma_vert_list = (
            sigma_va
            if sigma_va is not None
            else [0.02] * len(self.preferred_vertical_angles)
        )
        self.num_bvc_per_dir = num_bvc_per_dir if num_bvc_per_dir is not None else 50

        self.scaling_factors = [1.0] * len(self.preferred_vertical_angles)

        if sigma_d:
            self.sigma_d_list = sigma_d
            print(f"overriding sigma_d_list to {self.sigma_d_list}")
        if sigma_a:
            self.sigma_ang_list = sigma_a
            print(f"overriding sigma_ang_list to {self.sigma_ang_list}")

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

        self.hmap_loc = np.zeros(
            (self.num_steps, 3)
        )  # coordinates in the format [X, Z, Y]
        self.hmap_pcn = np.zeros(
            (self.num_steps, self.num_place_cells)
        )  # place cell activations
        self.hmap_bvc = np.zeros(
            (
                self.num_steps,
                len(self.preferred_vertical_angles)
                * self.n_hd_bvc
                * self.num_bvc_per_dir,
            )
        )  # BVC cell activations

        # Initialize hardware components and sensors
        self.robot = self.getFromDef("agent")  # Placeholder for robot instance
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        # Initialize touch sensor
        self.touch_sensor = self.getDevice("touch_sensor")
        self.touch_sensor.enable(self.timestep)

        # Initialize velocity
        self.velocity = [0.1, 0.1, 0.1]  # Initial velocity vector
        self.speed = 0.1  # Speed magnitude (reduced from 0.5 to 0.1)
        self.vertical_range_finder = self.getDevice("vertical-range-finder")
        self.vertical_range_finder.enable(self.timestep)
        self.collided = torch.zeros(2, dtype=torch.int32)
        self.rotation_field = self.robot.getField("rotation")

        # Initialize layers
        self.load_pcn(
            num_place_cells=self.num_place_cells,
            n_hd=self.n_hd_hdn,
            timestep=self.timestep,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
            preferred_vertical_angles=self.preferred_vertical_angles,
            sigma_d_list=self.sigma_d_list,
            sigma_ang_list=self.sigma_ang_list,
            sigma_vert_list=self.sigma_vert_list,
            scaling_factors=self.scaling_factors,
            num_bvc_per_dir=self.num_bvc_per_dir,
        )

        self.head_direction_layer = HeadDirectionLayer3D(device="cpu")

        # Initialize boundaries
        self.vertical_boundaries = torch.zeros((720, 360))

        self.step(self.timestep)
        self.step_count += 1

        # Initialize goal
        self.expected_reward = 0

        if randomize_start_loc:
            INITIAL = [
                random.uniform(-2.3, 2.3),
                random.uniform(0.5, 2),
                random.uniform(-2.3, 2.3),
            ]
            self.robot.getField("translation").setSFVec3f(INITIAL)
        else:
            self.robot.getField("translation").setSFVec3f(
                [start_loc[0], 0.1, start_loc[1]]
            )

        self.robot.resetPhysics()

    def load_pcn(
        self,
        num_place_cells: int,
        n_hd: int,
        timestep: int,
        preferred_vertical_angles: List[float],
        sigma_d_list: List[float],
        sigma_ang_list: List[float],
        sigma_vert_list: List[float],
        scaling_factors: List[float],
        num_bvc_per_dir: int,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
    ):
        try:
            with open("pcn.pkl", "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
        except:

            # Initialize BVC layer with per-layer sigma values
            bvc = BoundaryVectorCellLayer3D(
                max_dist=5,
                n_hd=self.n_hd_bvc,
                preferred_vertical_angles=preferred_vertical_angles,
                sigma_d_list=sigma_d_list,
                sigma_ang_list=sigma_ang_list,
                sigma_vert_list=sigma_vert_list,
                scaling_factors=scaling_factors,
                num_bvc_per_dir=num_bvc_per_dir,
                device=self.device,
                bottom_cutoff_percentage=1.0,
            )

            self.pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=num_place_cells,
                timestep=timestep,
                n_hd=self.n_hd_hdn,
                w_in_init_ratio=0.50,
                device=self.device,
            )

            print(
                f"Initialized new Boundary Vector Cell Network with {self.pcn.num_bvc} cells"
            )
            print(f"Initialized new Place Cell Network with {self.pcn.num_pc} cells")

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

    ########################################### RUN LOOP ###########################################

    def run(self):
        print(
            f"Starting robot in stage {self.robot_mode} for {self.num_steps} time steps"
        )
        while self.step_count <= self.num_steps:
            if (
                self.robot_mode == RobotMode.LEARN_OJAS
                or self.robot_mode == RobotMode.LEARN_HEBB
            ):

                self.explore()
            elif

        self.save()

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        self.sense()  # Preprocess sensor data
        self.compute()  # Handle activations

        # Get current position
        current_pos = self.robot.getField("translation").getSFVec3f()

        # Predict next position
        next_pos = [
            current_pos[0] + self.velocity[0],
            current_pos[1] + self.velocity[1],
            current_pos[2] + self.velocity[2],
        ]

        # Check range finder for obstacles
        range_data = self.vertical_range_finder.getRangeImage()
        if range_data is not None:
            min_distance = min(range_data)  # Closest obstacle distance
            if min_distance < 0.2:  # Collision is imminent
                # Reposition robot slightly away from the obstacle
                self.reposition_away_from_obstacle(current_pos)

                # Reflect velocity vector based on the wall's normal
                self.reflect_velocity()

                # Skip movement this step to avoid clipping
                return

        # Ensure robot stays within bounds
        bounds = 2.3
        height_bounds = 2.8
        if (
            abs(next_pos[0]) > bounds
            or abs(next_pos[2]) > bounds
            or next_pos[1] < 0.2
            or next_pos[1] > height_bounds
        ):
            self.reposition_away_from_obstacle(current_pos)
            self.reflect_velocity()
            return

        # Random walk behavior - 5% chance to adjust direction
        if random.random() < 0.15:  # 5% chance each step
            # Add Gaussian noise to each velocity component
            # Using smaller std dev for vertical (y) component
            noise = [
                np.random.normal(0, 0.05),  # x component
                np.random.normal(0, 0.02),  # y component (less variation)
                np.random.normal(0, 0.05),  # z component
            ]

            # Add noise to current velocity
            self.velocity = [v + n for v, n in zip(self.velocity, noise)]

            # Normalize to maintain constant speed
            magnitude = np.sqrt(sum(v * v for v in self.velocity))
            self.velocity = [v / magnitude * self.speed for v in self.velocity]

            # Recalculate next position with new velocity
            next_pos = [
                current_pos[0] + self.velocity[0],
                current_pos[1] + self.velocity[1],
                current_pos[2] + self.velocity[2],
            ]

        # Move the robot if no collisions detected
        self.robot.getField("translation").setSFVec3f(next_pos)

    ########################################### SENSE ###########################################

    def sense(self):
        """Preprocess sensor data for use in collision detection and navigation."""
        # Get the current heading from the compass
        self.current_heading_deg = int(
            self.get_bearing_in_degrees(self.compass.getValues())
        )

        # Get the range finder data and roll to align with heading
        vertical_data = self.vertical_range_finder.getRangeImage()
        if vertical_data is not None:
            vertical_boundaries_tf = torch.tensor(
                vertical_data, dtype=torch.float32, device=self.device
            )
            vertical_boundaries_tf = vertical_boundaries_tf.reshape(90, 180)
            self.vertical_boundaries = torch.roll(
                vertical_boundaries_tf, shifts=int(self.current_heading_deg / 2), dims=1
            )
        else:
            self.vertical_boundaries = None

        # Advance the timestep for the control loop
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """Converts compass readings to bearing in degrees.

        Args:
            north (List[float]): List containing the compass sensor values [x, y, z].

        Returns:
            float: Bearing angle in degrees (0-360), where 0 is North.
        """
        rad = np.arctan2(north[0], north[2])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0
        return bearing

    ########################################### COMPUTE ###########################################

    def compute(self):
        """
        Compute the activations of place cells and handle the environment interactions.
        """
        # Ensure points are float32
        self.vertical_boundaries = self.vertical_boundaries.to(dtype=torch.float32)

        # Convert velocity to tensor and normalize (if non-zero)
        hd_vel = torch.tensor(self.velocity, dtype=torch.float32, device="cpu")

        # Get head direction activations
        self.head_directions = self.head_direction_layer.get_hd_activation(hd_vel)

        self.pcn.get_place_cell_activations(
            self.vertical_boundaries,
            hd_activations=self.head_directions,
            collided=torch.any(self.collided).item(),
        )

        if self.visual_bvc:
            self.pcn.bvc_layer.plot_activation(self.vertical_boundaries)

        # Advance the timestep and update position
        self.step(self.timestep)

        # Update place cell and sensor maps
        if self.step_count < self.num_steps:
            self.hmap_loc[self.step_count] = self.robot.getField(
                "translation"
            ).getSFVec3f()
            self.hmap_pcn[self.step_count] = (
                self.pcn.place_cell_activations.cpu().numpy()
            )
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.cpu().numpy()

        # Increment timestep
        self.step_count += 1

    ########################################### MOVEMENT ###########################################

    def reposition_away_from_obstacle(self, current_pos):
        """Move the robot slightly away from the obstacle and towards origin."""
        # Calculate vector to origin (0, 1, 0)
        to_origin = [0 - current_pos[0], 1 - current_pos[1], 0 - current_pos[2]]

        # Normalize the vector
        magnitude = np.sqrt(sum(x * x for x in to_origin))
        if magnitude > 0:
            to_origin = [x / magnitude for x in to_origin]

        # Move slightly towards origin
        self.robot.getField("translation").setSFVec3f(
            [
                current_pos[0] + to_origin[0] * 0.1,
                current_pos[1] + to_origin[1] * 0.1,
                current_pos[2] + to_origin[2] * 0.1,
            ]
        )

    def reflect_velocity(self):
        """Invert the dominant axis of motion and add slight randomness."""
        # Find the axis with the largest absolute velocity (dominant direction)
        abs_velocities = [abs(v) for v in self.velocity]
        dominant_axis = abs_velocities.index(max(abs_velocities))

        # Only invert the dominant axis
        new_velocity = list(self.velocity)
        new_velocity[dominant_axis] = -new_velocity[dominant_axis]

        # Add small random variation to other axes to prevent getting stuck
        for i in range(3):
            if i != dominant_axis:
                new_velocity[i] += np.random.uniform(-0.05, 0.05)

        # Update velocity and normalize to maintain speed
        self.velocity = new_velocity
        magnitude = np.sqrt(sum(v * v for v in self.velocity))
        self.velocity = [v / magnitude * self.speed for v in self.velocity]

    def rotate_in_place(self):
        """Rotate the robot in place to escape deadlock."""
        self.velocity = [0.0, 0.0, 0.0]  # Stop movement
        self.rotation_field.setSFRotation([0, 1, 0, np.random.uniform(0, 2 * np.pi)])
        self.step(self.timestep * 10)  # Rotate for a short time

    ########################################### FILE IO ###########################################

    def save(
        self,
        include_pcn: bool = True,
        include_hmaps: bool = True,
    ):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), and optionally
        the maps that store the agent's movement and activations.

        Parameters:
            include_maps (bool): If True, saves the history of the agent's path and activations.
        """
        # Create directory if it doesn't exist
        if self.file_prefix:
            directory = os.path.dirname(self.file_prefix)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

        files_saved = []

        # Save parameters to text file
        params_file = f"{self.file_prefix}parameters.txt"
        with open(params_file, "w") as f:
            f.write("Model Parameters:\n")
            f.write(f"sigma_d: {self.pcn.bvc_layer.sigma_d}\n")
            f.write(f"sigma_ang: {self.pcn.bvc_layer.sigma_ang}\n")
            f.write(f"num_place_cells: {self.pcn.num_pc}\n")
            f.write(f"num_bvcs: {self.pcn.num_bvc}\n")
            f.write(f"num_bvc_per_dir: {self.num_bvc_per_dir}\n")
            f.write(f"n_hd_bvc: {self.n_hd_bvc}\n")
            f.write(f"n_hd_hdn: {self.n_hd_hdn}\n")
            f.write(f"run_time_hours: {self.run_time_minutes / 60}\n")
            f.write(f"num_simulation_steps: {self.num_steps}\n")

        files_saved.append("parameters.txt")

        # Save the Place Cell Network (PCN)
        if include_pcn:
            with open(f"{self.file_prefix}pcn.pkl", "wb") as output:
                pickle.dump(self.pcn, output)
                files_saved.append("pcn.pkl")

        # Save the history maps if specified
        if include_hmaps:
            with open(f"{self.file_prefix}hmap_loc.pkl", "wb") as output:
                pickle.dump(self.hmap_loc[: self.step_count], output)
                files_saved.append("hmap_loc.pkl")
            with open(f"{self.file_prefix}hmap_pcn.pkl", "wb") as output:
                pickle.dump(self.hmap_pcn[: self.step_count], output)
                files_saved.append("hmap_pcn.pkl")
            with open(f"{self.file_prefix}hmap_bvc.pkl", "wb") as output:
                pickle.dump(self.hmap_bvc[: self.step_count], output)
                files_saved.append("hmap_bvc.pkl")

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")
        self.done = True

    def clear(self):
        """
        Clears the saved state files for the Place Cell Network (PCN), Reward Cell Network (RCN),
        and the history maps by removing their corresponding pickle files.
        """
        files_to_remove = [
            "pcn.pkl",
            # "rcn.pkl",
            "hmap_loc.pkl",
            "hmap_pcn.pkl",
            "hmap_bvc.pkl",
            "hmap_hdn.pkl",
        ]

        for file in files_to_remove:
            try:
                os.remove(self.file_prefix + file)
            except FileNotFoundError:
                pass  # Ignore if the file does not exist

        print("State files cleared.")
