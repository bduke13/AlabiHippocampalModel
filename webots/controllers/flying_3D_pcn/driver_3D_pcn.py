import numpy as np
import torch
import pickle
import random
import os
from typing import Optional, List
from controller import Supervisor
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt

# Add project root to sys.path if needed
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.robot.robot_mode import RobotMode
from core.layers.boundary_vector_cell_layer_3D import BoundaryVectorCellLayer3D
from core.layers.head_direction_layer_3D import HeadDirectionLayer3D
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer

np.set_printoptions(precision=2)
PI = torch.tensor(np.pi)


# Helper function: Generate a random unit vector within a cone.
def random_cone_vector(target_vector, max_angle_deg=60):
    """
    Generates a random unit vector that is within max_angle_deg of the target_vector.

    Args:
        target_vector (array-like): The central target direction (assumed non-zero).
        max_angle_deg (float): The cone half-angle in degrees.

    Returns:
        np.ndarray: A random unit vector within the cone.
    """
    max_angle = np.deg2rad(max_angle_deg)
    # Sample cosine uniformly from [cos(max_angle), 1]
    cos_theta = np.random.uniform(np.cos(max_angle), 1)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi)

    # Create a vector in the local coordinate system (cone around z-axis)
    local_vec = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # Rotate local_vec so that the z-axis aligns with target_vector.
    v = np.array(target_vector, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return local_vec  # Fallback if target is zero.
    v = v / v_norm

    # Find rotation axis (cross product between z-axis and v)
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    if np.allclose(v, z_axis):
        return local_vec
    if np.allclose(v, -z_axis):
        # v is opposite to z; rotate local_vec by 180 degrees.
        return -local_vec

    axis = np.cross(z_axis, v)
    axis_norm = np.linalg.norm(axis)
    if axis_norm != 0:
        axis = axis / axis_norm
    else:
        axis = np.array([1, 0, 0], dtype=np.float64)

    angle = np.arccos(np.clip(np.dot(z_axis, v), -1, 1))

    # Rodrigues' rotation formula
    local_vec = local_vec.astype(np.float64)
    rotated_vec = (
        local_vec * np.cos(angle)
        + np.cross(axis, local_vec) * np.sin(angle)
        + axis * np.dot(axis, local_vec) * (1 - np.cos(angle))
    )
    # Normalize just in case
    return rotated_vec / np.linalg.norm(rotated_vec)


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
        visual_pcn: bool = False,
        world_name: Optional[str] = None,
        start_location: Optional[List[float]] = None,
        randomize_start_location: bool = True,
        goal_location: Optional[List[float]] = None,
        max_dist: float = 10,
        show_save_dialogue_and_pause=True,
        gamma_pp=0.5,
        gamma_pb=0.3,
        tau_denom=1000,
    ):
        self.robot_mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the world name and directories for saving data
        if world_name is None:
            world_path = self.getWorldPath()  # Get the full path to the world file
            world_name = os.path.splitext(os.path.basename(world_path))[0]
        self.world_name = world_name
        self.visual_bvc = visual_bvc
        self.visual_pcn = visual_pcn
        self.show_save_dialogue_and_pause = show_save_dialogue_and_pause
        self.done = False  # Setting to True stops sim

        # Construct directory paths
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Model parameters
        self.timestep = 32 * 3
        self.tau_w = 10

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60

        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": 0.3, "exploit": 0.5}

        # Initialize hardware components and sensors
        self.robot = self.getFromDef("agent")
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        self.touch_sensor = self.getDevice("touch_sensor")
        self.touch_sensor.enable(self.timestep)

        self.velocity = [0.1, 0.1, 0.1]  # Velocity in [x, y, z]
        self.speed = 0.1
        self.vertical_range_finder = self.getDevice("vertical-range-finder")
        self.vertical_range_finder.enable(self.timestep)
        self.collided = torch.zeros(2, dtype=torch.int32)
        self.rotation_field = self.robot.getField("rotation")

        # Clear/load layers
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

        self.pcn.gamma_pb = gamma_pb
        self.pcn.gamma_pp = gamma_pp
        self.pcn.tau = self.timestep / tau_denom

        self.head_direction_layer = HeadDirectionLayer3D(device=self.device)
        self.vertical_boundaries = torch.zeros((720, 360))

        self.step(self.timestep)
        self.step_count = 1
        self.expected_reward = 0

        # Define environment boundaries in Webots (x, y, z)
        self.x_min, self.x_max = -2.3, 2.3
        self.y_min, self.y_max = 0.1, 4.8
        self.z_min, self.z_max = -2.3, 2.3

        # Initialize hmaps (history maps)
        self.hmap_loc = torch.zeros(
            (self.num_steps, 3), device=self.device
        )  # [x, y, z]
        self.hmap_pcn = torch.zeros(
            (self.num_steps, num_place_cells), device=self.device
        )
        self.hmap_bvc = torch.zeros(
            (self.num_steps, self.pcn.bvc_layer.num_bvc),
            device=self.device,
        )
        self.hmap_hdn = torch.zeros((self.num_steps, n_hd_hdn), device=self.device)

        # Set the initial location
        if randomize_start_location:
            # Choose a random location within the boundaries (with a small margin)
            init_pos = [
                random.uniform(self.x_min + 0.2, self.x_max - 0.2),
                random.uniform(self.y_min + 0.2, self.y_max - 0.2),
                random.uniform(self.z_min + 0.2, self.z_max - 0.2),
            ]
        else:
            init_pos = start_location  # Expected to be in [x, y, z]
        self.robot.getField("translation").setSFVec3f(init_pos)
        self.robot.resetPhysics()

    def load_pcn(
        self,
        num_place_cells: int,
        n_hd_bvc: int,
        n_hd_hdn: int,
        max_dist: float,
        num_bvc_per_dir: int,
        phi_vert_preferred: List[float],
        sigma_rs: List[float],
        sigma_thetas: List[float],
        sigma_phis: List[float],
        scaling_factors: List[float],
        input_rows: int,
        input_cols: int,
        timestep: int,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        device: str = "cpu",
    ):
        try:
            saved_pcn_path = os.path.join(self.network_dir, "pcn.pkl")
            print(os.path.exists(saved_pcn_path))
            with open(saved_pcn_path, "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print(f"Loaded existing Place Cell Network from {saved_pcn_path}")
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except:
            print("File not found. Instantiating new PCN layer")
            bvc = BoundaryVectorCellLayer3D(
                max_dist=max_dist,
                n_hd=n_hd_bvc,
                phi_vert_preferred=phi_vert_preferred,
                sigma_rs=sigma_rs,
                sigma_thetas=sigma_thetas,
                sigma_phis=sigma_phis,
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
                w_in_init_ratio=0.50,
                device=device,
            )
            print(
                f"Initialized new Boundary Vector Cell Network with {self.pcn.num_bvc} cells"
            )
            print(f"Initialized new Place Cell Network with {self.pcn.num_pc} cells")

        if enable_ojas is not None:
            self.pcn.enable_ojas = enable_ojas
        elif self.robot_mode == RobotMode.LEARN_OJAS:
            self.pcn.enable_ojas = True

        if enable_stdp is not None:
            self.pcn.enable_stdp = enable_stdp
        elif self.robot_mode == RobotMode.LEARN_HEBB:
            self.pcn.enable_stdp = True

        print(f"enable ojas {self.pcn.enable_ojas}")
        print(f"enable stdp {self.pcn.enable_stdp}")
        print(f"num place cells {self.pcn.num_pc}")
        print(f"num bvc {self.pcn.num_bvc}")
        print(f"w_in shape {self.pcn.w_in.shape}")
        return self.pcn

    ########################################### RUN LOOP ###########################################

    def run(self):
        print(
            f"Starting robot in stage {self.robot_mode} for {self.num_steps} time steps"
        )
        while self.step_count <= self.num_steps and not self.done:
            if self.robot_mode in [
                RobotMode.LEARN_OJAS,
                RobotMode.LEARN_HEBB,
                RobotMode.PLOTTING,
            ]:
                self.explore()
            else:
                print("Robot mode not implemented yet, quitting.")
                break
        self.save()

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        self.sense()  # Preprocess sensor data
        self.compute()  # Handle activations

        # Get current position (x, y, z)
        current_pos = self.robot.getField("translation").getSFVec3f()

        # Predict next position using current velocity (in x, y, z)
        next_pos = [current_pos[i] + self.velocity[i] for i in range(3)]

        # Check vertical range finder for obstacles
        range_data = self.vertical_range_finder.getRangeImage()
        if range_data is not None:
            min_distance = min(range_data)
            if min_distance < 0.2:
                self.handle_collision(current_pos)
                return

        # Check if the next position is within the boundaries
        if not (
            self.x_min <= next_pos[0] <= self.x_max
            and self.y_min <= next_pos[1] <= self.y_max
            and self.z_min <= next_pos[2] <= self.z_max
        ):
            self.handle_collision(current_pos)
            return

        # Random walk behavior: 10% chance to adjust direction
        if random.random() < 0.50:
            noise = [np.random.normal(0, 0.025) for _ in range(3)]
            self.velocity = [self.velocity[i] + noise[i] for i in range(3)]
            mag = np.sqrt(sum(v * v for v in self.velocity))
            self.velocity = [v / mag * self.speed for v in self.velocity]
            next_pos = [current_pos[i] + self.velocity[i] for i in range(3)]

        self.robot.getField("translation").setSFVec3f(next_pos)

    ########################################### MOVEMENT ###########################################
    ###########################################
    # In your DriverFlying class:

    def handle_collision(self, current_pos):
        # Increase collision counter
        if not hasattr(self, "collision_counter"):
            self.collision_counter = 0
        self.collision_counter += 1

        self.reposition_away_from_obstacle(current_pos)
        self.reflect_velocity()

        # If stuck for 3 consecutive collisions, adjust velocity toward center
        if self.collision_counter >= 3:
            self.push_towards_center(current_pos)
            center = [0.0, 2.5, 0.0]  # Center of the room in [x, y, z]
            target_vector = [center[i] - current_pos[i] for i in range(3)]
            # Generate a random velocity vector within a 90° cone around the center direction
            new_dir = random_cone_vector(target_vector, max_angle_deg=90)
            # Set velocity to this new direction scaled by speed
            self.velocity = [float(new_dir[i]) * self.speed for i in range(3)]
            self.collision_counter = 0

    def push_towards_center(self, current_pos):
        # Define the center of the room.
        center = [0.0, 2.5, 0.0]
        # Compute vector from current position to center.
        vector_to_center = [center[i] - current_pos[i] for i in range(3)]
        norm = np.linalg.norm(vector_to_center)
        if norm == 0:
            return
        unit_vector = [v / norm for v in vector_to_center]
        push_distance = 0.1  # Adjust push distance as needed.
        new_pos = [current_pos[i] + push_distance * unit_vector[i] for i in range(3)]
        # Ensure the new position is within boundaries.
        new_pos[0] = np.clip(new_pos[0], self.x_min, self.x_max)
        new_pos[1] = np.clip(new_pos[1], self.y_min, self.y_max)
        new_pos[2] = np.clip(new_pos[2], self.z_min, self.z_max)
        self.robot.getField("translation").setSFVec3f(new_pos)

    def reposition_away_from_obstacle(self, current_pos):
        x, y, z = current_pos
        offset = 0.3  # Larger offset to clear the wall
        if x < self.x_min:
            x = self.x_min + offset
        elif x > self.x_max:
            x = self.x_max - offset
        if y < self.y_min:
            y = self.y_min + offset
        elif y > self.y_max:
            y = self.y_max - offset
        if z < self.z_min:
            z = self.z_min + offset
        elif z > self.z_max:
            z = self.z_max - offset
        self.robot.getField("translation").setSFVec3f([x, y, z])

    def reflect_velocity(self):
        # Invert the entire velocity vector to push the agent away from the wall.
        self.velocity = [-v for v in self.velocity]
        # Add small random noise to help avoid repeating the same collision.
        self.velocity = [v + np.random.uniform(-0.1, 0.1) for v in self.velocity]
        mag = np.sqrt(sum(v * v for v in self.velocity))
        self.velocity = [v / mag * self.speed for v in self.velocity]

    ########################################### SENSE ###########################################

    def sense(self):
        self.current_heading_deg = int(
            self.get_bearing_in_degrees(self.compass.getValues())
        )
        vertical_data = self.vertical_range_finder.getRangeImage()
        if vertical_data is not None:
            self.vertical_boundaries = torch.tensor(
                vertical_data, dtype=torch.float32, device=self.device
            ).reshape(90, 180)
            self.vertical_boundaries = torch.roll(
                self.vertical_boundaries,
                shifts=int(self.current_heading_deg / 2),
                dims=1,
            )
        else:
            self.vertical_boundaries = None
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        rad = np.arctan2(north[0], north[2])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing += 360.0
        return bearing

    ########################################### COMPUTE ###########################################

    def compute(self):
        self.vertical_boundaries = self.vertical_boundaries.to(
            device=self.device, dtype=torch.float32
        )
        hd_vel = torch.tensor(self.velocity, dtype=torch.float32, device=self.device)
        self.head_directions = self.head_direction_layer.get_hd_activation(hd_vel)
        self.pcn.get_place_cell_activations(
            self.vertical_boundaries,
            hd_activations=self.head_directions,
            collided=torch.any(self.collided).item(),
        )

        if self.visual_pcn:
            activations = self.pcn.place_cell_activations.cpu().detach().numpy()
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(activations)), activations)
            plt.xlabel("Place Cell Index")
            plt.ylabel("Activation")
            plt.title("Current Place Cell Activations")
            plt.show()

        if self.visual_bvc:
            self.pcn.bvc_layer.plot_activation(self.vertical_boundaries)
            self.pcn.bvc_layer.plot_activation_distribution(self.vertical_boundaries)

        self.step(self.timestep)

        if self.step_count < self.num_steps:
            self.hmap_loc[self.step_count] = torch.tensor(
                self.robot.getField("translation").getSFVec3f(), device=self.device
            )
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.to(
                self.device
            )
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.to(self.device)
            self.hmap_hdn[self.step_count] = self.head_directions.to(self.device)

        self.step_count += 1

    ########################################### FILE IO ###########################################

    def save(
        self,
        include_pcn: bool = True,
        include_rcn: bool = False,
        include_hmaps: bool = True,
    ):
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

        if self.show_save_dialogue_and_pause:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            messagebox.showinfo("Information", "Press OK to save data")
            root.destroy()

            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
        self.done = True
        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def clear(self):
        if os.path.exists(self.network_dir):
            for file in os.listdir(self.network_dir):
                file_path = os.path.join(self.network_dir, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        if os.path.exists(self.hmap_dir):
            for file in os.listdir(self.hmap_dir):
                file_path = os.path.join(self.hmap_dir, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
