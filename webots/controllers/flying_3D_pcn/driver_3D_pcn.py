import numpy as np
import torch
import pickle
import random
import os
from typing import Optional, List
from controller import Supervisor
from tkinter import N, messagebox
import tkinter as tk


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
        phi_vert_preferred: List[float],
        sigma_rs: List[float],
        sigma_thetas: List[float],
        sigma_phis: List[float],
        scaling_factors: List[float],
        mode=RobotMode.PLOTTING,
        run_time_hours: int = 1,
        num_bvc_per_dir: int = 50,
        num_place_cells: int = 500,
        n_hd_bvc: int = 8,
        n_hd_hdn: int = 20,
        input_rows: int = 90,
        input_cols: int = 180,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        visual_bvc: bool = False,
        world_name: Optional[str] = None,
        start_location: Optional[List[int]] = None,
        randomize_start_location: bool = True,
        goal_location: Optional[List[float]] = None,
        max_dist: float = 10,
    ):

        self.robot_mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the world name and directories for saving data
        if world_name is None:
            world_path = self.getWorldPath()  # Get the full path to the world file
            world_name = os.path.splitext(os.path.basename(world_path))[
                0
            ]  # Extract just the world name
        self.world_name = world_name
        self.visual_bvc = visual_bvc

        # Construct directory paths
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters
        self.timestep = 32 * 6
        self.tau_w = 10  # time constant for the window function

        self.visual_bvc = visual_bvc

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

        # Initialize hmaps (history maps) as torch tensors on specified device
        self.hmap_loc = torch.zeros(
            (self.num_steps, 3), device=self.device
        )  # coordinates [X, Z, Y]
        self.hmap_pcn = torch.zeros(
            (self.num_steps, num_place_cells), device=self.device
        )  # place cell activations
        self.hmap_bvc = torch.zeros(
            (self.num_steps, len(phi_vert_preferred) * n_hd_bvc * num_bvc_per_dir),
            device=self.device,
        )  # BVC activations
        self.hmap_hdn = torch.zeros(
            (self.num_steps, n_hd_hdn), device=self.device
        )  # head direction activations

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

        # Clear/Initialize/Load layers
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        self.load_pcn(
            num_place_cells=num_place_cells,
            n_hd_bvc=n_hd_bvc,
            n_hd_hdn=n_hd_hdn,
            max_dist=max_dist,
            num_bvc_per_dir=num_bvc_per_dir,
            phi_vert_preferred=phi_vert_preferred,
            sigma_rs=sigma_rs,
            sigma_thetas=sigma_thetas,
            sigma_phis=sigma_phis,
            scaling_factors=scaling_factors,
            input_rows=input_rows,
            input_cols=input_cols,
            timestep=self.timestep,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
            device=self.device,
        )

        self.head_direction_layer = HeadDirectionLayer3D(device=self.device)

        # Initialize boundaries
        self.vertical_boundaries = torch.zeros((720, 360))

        self.step(self.timestep)
        self.step_count += 1

        # Initialize goal
        self.expected_reward = 0

        if randomize_start_location:
            INITIAL = [
                random.uniform(-2.3, 2.3),
                random.uniform(0.5, 2),
                random.uniform(-2.3, 2.3),
            ]
            self.robot.getField("translation").setSFVec3f(INITIAL)
        else:
            self.robot.getField("translation").setSFVec3f(
                [start_location[0], start_location[1], start_location[1]]
            )

        self.robot.resetPhysics()

    def load_pcn(
        self,
        num_place_cells: int,
        n_hd_bvc: int,
        n_hd_hdn: int,
        max_dist: float,
        num_bvc_per_dir: int,
        phi_vert_preferred: List[float],  # vertical angle preferences (ψ)
        sigma_rs: List[float],  # radial distance spread (σ_r)
        sigma_thetas: List[float],  # horizontal angle spread (σ_θ)
        sigma_phis: List[float],  # vertical angle spread (σ_φ)
        scaling_factors: List[float],
        input_rows: int,
        input_cols: int,
        timestep: int,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        device: str = "cpu",
    ):
        try:
            with open("pcn.pkl", "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except:

            # Initialize BVC layer with per-layer sigma values
            bvc = BoundaryVectorCellLayer3D(
                max_dist=max_dist,
                n_hd=n_hd_bvc,
                phi_vert_preferred=phi_vert_preferred,  # vertical angle preferences (ψ)
                sigma_rs=sigma_rs,  # radial distance spread (σ_r)
                sigma_thetas=sigma_thetas,  # horizontal angle spread (σ_θ)
                sigma_phis=sigma_phis,  # vertical angle spread (σ_φ)
                scaling_factors=scaling_factors,
                num_bvc_per_dir=num_bvc_per_dir,
                input_rows=input_rows,
                input_cols=input_cols,
                bottom_cutoff_percentage=1.0,
                device=device,
            )

            self.pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=num_place_cells,
                timestep=timestep,
                n_hd=n_hd_hdn,
                w_in_init_ratio=0.20,
                device=device,
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
            else:
                print("Robot mode not implemented yet, quitting.")
                break

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
            # Create tensor directly on target device
            self.vertical_boundaries = torch.tensor(
                vertical_data, dtype=torch.float32, device=self.device
            ).reshape(90, 180)

            # Roll operation stays on device
            self.vertical_boundaries = torch.roll(
                self.vertical_boundaries,
                shifts=int(self.current_heading_deg / 2),
                dims=1,
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
        # Ensure data is on correct device with correct dtype
        self.vertical_boundaries = self.vertical_boundaries.to(
            device=self.device, dtype=torch.float32
        )

        # Convert velocity to tensor on correct device
        hd_vel = torch.tensor(self.velocity, dtype=torch.float32, device=self.device)

        # Get head direction activations (already on correct device)
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
            # Record position as tensor
            self.hmap_loc[self.step_count] = torch.tensor(
                self.robot.getField("translation").getSFVec3f(), device=self.device
            )
            # Record network activations (keeping on device)
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.to(
                self.device
            )
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.to(self.device)
            self.hmap_hdn[self.step_count] = self.head_directions.to(self.device)

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
        include_rcn: bool = False,
        include_hmaps: bool = True,
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

        # Save history maps if specified
        if include_hmaps:
            # Convert tensors to numpy arrays only during save
            hmaps = {
                "hmap_loc.pkl": self.hmap_loc[: self.step_count].cpu().numpy(),
                "hmap_pcn.pkl": self.hmap_pcn[: self.step_count].cpu().numpy(),
                "hmap_bvc.pkl": self.hmap_bvc[: self.step_count].cpu().numpy(),
                "hmap_hdn.pkl": self.hmap_hdn[: self.step_count].cpu().numpy(),
            }

            for filename, data in hmaps.items():
                filepath = os.path.join(self.hmap_dir, filename)
                with open(filepath, "wb") as output:
                    pickle.dump(data, output)
                    files_saved.append(filepath)

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
        Clears all saved state files by removing all files in the hmap and network directories.
        """
        # Clear network directory
        if os.path.exists(self.network_dir):
            for file in os.listdir(self.network_dir):
                file_path = os.path.join(self.network_dir, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

        # Clear hmap directory
        if os.path.exists(self.hmap_dir):
            for file in os.listdir(self.hmap_dir):
                file_path = os.path.join(self.hmap_dir, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
