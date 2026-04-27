import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from controller import Supervisor

from artifacts import (
    clear_driver_outputs,
    initialize_run_artifacts,
    save_driver_state,
    write_metrics,
)
from driver_setup import (
    initialize_devices,
    initialize_head_direction_layer,
    initialize_histories,
    initialize_robot_pose,
    initialize_runtime_state,
)
from layers.pcn import PlaceCellLayer
from layers.rcn import RewardCellLayer
from layers.bvc import BoundaryVectorCellLayer
from layers.gcn import GridCellLayer
from robot.robot_mode import RobotMode
from webots_control import (
    current_world_name,
    export_image,
    pause_simulation,
    quit_webots,
    robot_position,
)

# --- PyTorch seeds / random ---
# torch.manual_seed(5)
# np.random.seed(5)
# rng = default_rng(5)  # or keep it as is
np.set_printoptions(precision=2)
torch.set_printoptions(precision=16)

CONTROLLER_DIR = Path(__file__).resolve().parent


class Driver(Supervisor):
    """Runtime-only base controller for the isolated single-scale model."""

    def initialization(
        self,
        mode=RobotMode.PLOTTING,
        run_time_hours: int = 2,
        randomize_start_loc: bool = True,
        start_loc: Optional[List[int]] = None,
        start_rotation: Optional[List[float]] = None,
        load_networks_from_run_id: Optional[str] = None,
        load_hmaps_from_run_id: Optional[str] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        max_dist: float = 10,
        show_bvc_activation: bool = False,
        run_id: Optional[str] = None,
        quit_on_completion: bool = False,
        pause_on_completion: bool = True,
        export_image_on_completion: bool = False,
        completion_image_path: Optional[str] = None,
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
            start_rotation (Optional[List[float]], optional): Robot axis-angle rotation to restore
                at trial start. Defaults to None.
            load_networks_from_run_id (Optional[str], optional): Run ID whose saved networks should
                be loaded into this run before execution. Defaults to None.
            load_hmaps_from_run_id (Optional[str], optional): Run ID whose saved hmaps should
                seed this run's histories before execution. Defaults to None.
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
        print(f"Driver running on device {self.device}")
        self.dtype = torch.float32
        self.quit_on_completion = bool(quit_on_completion)
        self.pause_on_completion = bool(pause_on_completion)
        self.export_image_on_completion = bool(export_image_on_completion)
        self.completion_image_path = completion_image_path
        self.configured_start_loc = list(start_loc) if start_loc is not None else None
        self.configured_start_rotation = (
            list(start_rotation) if start_rotation is not None else None
        )
        self.load_networks_from_run_id = load_networks_from_run_id
        self.load_hmaps_from_run_id = load_hmaps_from_run_id
        self.configured_randomize_start_loc = bool(randomize_start_loc)
        self.trial_completed = False
        self.trial_completion_reason = None
        self.last_saved_files: list[str] = []
        self.trial_start_time_seconds = 0.0

        if world_name is None:
            world_name = current_world_name(self)
        self.world_name = world_name
        initialize_run_artifacts(
            self,
            controller_dir=CONTROLLER_DIR,
            run_id=run_id,
            mode=mode,
            run_time_hours=run_time_hours,
            randomize_start_loc=randomize_start_loc,
            start_loc=start_loc,
            start_rotation=start_rotation,
            load_networks_from_run_id=load_networks_from_run_id,
            load_hmaps_from_run_id=self.load_hmaps_from_run_id,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
            goal_location=goal_location,
            max_dist=max_dist,
            show_bvc_activation=show_bvc_activation,
        )

        initialize_runtime_state(
            self,
            run_time_hours=run_time_hours,
            goal_location=goal_location,
            max_dist=max_dist,
            show_bvc_activation=show_bvc_activation,
        )
        initialize_robot_pose(
            self,
            randomize_start_loc=randomize_start_loc,
            start_loc=start_loc,
            start_rotation=start_rotation,
        )
        initialize_devices(self)

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
        self.load_gcn(device=self.device)
        initialize_head_direction_layer(self)
        initialize_histories(self)

        # progresses the simulation physics by the timestep property within this class
        self.step(self.timestep)
        # step_count measures how many times the hmaps were updated
        self.step_count = getattr(self, "history_step_count", 1)

        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()
        self.trial_start_time_seconds = float(self.getTime())
        write_metrics(self, status="initialized")

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
            network_path = os.path.join(self.network_load_dir, "pcn.pkl")
            with open(network_path, "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing PCN from", network_path)
                self.pcn.device = device
                self.pcn.bvc_layer.device = device
        except Exception:
            if self.load_networks_from_run_id:
                print(
                    "Could not load PCN from prior run",
                    self.load_networks_from_run_id,
                    "at",
                    network_path,
                    "- initializing a new PCN instead.",
                )
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
            network_path = os.path.join(self.network_load_dir, "rcn.pkl")
            with open(network_path, "rb") as f:
                self.rcn = pickle.load(f)
                print("Loaded existing RCN from", network_path)
                self.rcn.device = device
        except Exception:
            if self.load_networks_from_run_id:
                print(
                    "Could not load RCN from prior run",
                    self.load_networks_from_run_id,
                    "at",
                    network_path,
                    "- initializing a new RCN instead.",
                )
            self.rcn = RewardCellLayer(
                num_place_cells=num_place_cells,
                num_replay=num_replay,
                learning_rate=learning_rate,
                device=device,
            )
            print("Initialized new RCN")

        return self.rcn

    def load_gcn(self, device: torch.device):
        """Initialize the grid-cell layer used for runtime history logging."""
        self.gcn = GridCellLayer(
            num_modules=self.num_grid_modules,
            cells_per_module=self.num_grid_cells_per_module,
            spread_range=self.grid_spread_range,
            scale_multiplier=self.grid_scale_multiplier,
            translation_scale=self.grid_translation_scale,
            threshold=self.grid_threshold,
            threshold_type="soft",
            normalization="per-cell",
            world_name=self.world_name,
            device=str(device),
            dtype=self.dtype,
        )
        print(
            "Initialized GCN with",
            self.gcn.total_grid_cells,
            "cells for world",
            self.world_name,
        )
        return self.gcn

    ########################################### RUN LOOP ###########################################

    def run(self):
        """Runs the main control loop of the robot.

        The method manages the robot's behavior based on its current mode:
        - LEARN_OJAS/LEARN_HEBB/DMTP/PLOTTING: Runs exploration behavior
        - EXPLOIT: Runs goal-directed navigation
        """

        print(f"Starting robot in {self.robot_mode}")
        print(f"Goal at {self.goal_location}")

        while not self.trial_completed:
            if (
                self.robot_mode == RobotMode.LEARN_OJAS
                or self.robot_mode == RobotMode.LEARN_HEBB
                or self.robot_mode == RobotMode.DMTP
                or self.robot_mode == RobotMode.PLOTTING
            ):
                self.explore()

            elif self.robot_mode == RobotMode.EXPLOIT:
                self.exploit()

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

            if self.robot_mode == RobotMode.DMTP:
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
            if self.trial_completed:
                return
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()

            if self.trial_completed:
                return

        if self.trial_completed:
            return
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
        if self.trial_completed:
            return

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
        # Update place cell activations based on sensor data
        self.pcn.get_place_cell_activations(
            distances=self.boundaries,
            hd_activations=self.hd_activations,
            collided=torch.any(self.collided),
        )
        curr_pos = robot_position(self.robot)
        self.gcn_activations = self.gcn.get_grid_cell_activations(
            [curr_pos[0], curr_pos[2]],
            use_mask=False,
        )
        if self.show_bvc_activation:
            self.pcn.bvc_layer.plot_activation(self.boundaries.cpu())

        # Advance simulation one timestep
        # self.step(self.timestep)

    ########################################### CHECK GOAL REACHED ###########################################
    def elapsed_trial_time_seconds(self) -> float:
        return max(0.0, float(self.getTime()) - float(self.trial_start_time_seconds))

    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        curr_pos = robot_position(self.robot)

        if (
            self.robot_mode
            in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING)
            and self.elapsed_trial_time_seconds() >= 60 * self.run_time_minutes
        ):
            self.stop()
            self.complete_trial(
                reason="time_limit_reached",
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

            self.complete_trial(
                reason="goal_reached_dmtp",
                include_rcn=True,
                include_hmaps=False,
            )

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

            self.complete_trial(
                reason="goal_reached_exploit",
                include_rcn=True,
            )  # EXPLOIT doesn't save anything

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
            curr_pos = robot_position(self.robot)
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
        if self.step_count < 2:
            return 0.0

        recorded_path = self.hmap_loc[: self.step_count]
        path_length = 0
        for i in range(recorded_path.shape[0] - 1):
            current_position = np.array(
                [recorded_path[i, 2], recorded_path[i, 0]]
            )
            next_position = np.array(
                [recorded_path[i + 1, 2], recorded_path[i + 1, 0]]
            )
            path_length += np.linalg.norm(next_position - current_position)

        return path_length

    def update_hmaps(self):
        curr_pos = robot_position(self.robot)

        if self.step_count < self.num_steps:
            # Record position (X, Y, Z format)
            self.hmap_loc[self.step_count] = curr_pos

            # Record place cell network activations
            self.hmap_pcn[self.step_count] = self.pcn.place_cell_activations.detach()

            # Record Boundary Vector Cell (BVC) activations
            self.hmap_bvc[self.step_count] = self.pcn.bvc_activations.detach()

            # Record Head Direction Network (HDN) activations
            self.hmap_hdn[self.step_count] = self.hd_activations.detach()

            # Record Grid Cell Network (GCN) activations
            self.hmap_gcn[self.step_count] = self.gcn_activations.detach()

        self.step_count += 1

    def get_actual_reward(self):
        """
        Computes the actual reward based on current distance to the goal.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        # Get current position from the robot node
        curr_pos = robot_position(self.robot)

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
        if self.robot_mode == RobotMode.EXPLOIT:
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
        *,
        status: str = "saved",
        completion_reason: Optional[str] = None,
        extra_metrics: Optional[dict] = None,
    ):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), and optionally
        the maps that store the agent's movement and activations.
        """
        files_saved = save_driver_state(
            self,
            include_pcn=include_pcn,
            include_rcn=include_rcn,
            include_hmaps=include_hmaps,
        )

        self.last_saved_files = files_saved
        if completion_reason is not None:
            self.trial_completion_reason = completion_reason

        metrics_extra = dict(extra_metrics or {})
        if completion_reason is not None:
            metrics_extra["completion_reason"] = completion_reason

        write_metrics(
            self,
            status=status,
            files_saved=files_saved,
            extra=metrics_extra or None,
        )

        if include_hmaps:
            try:
                from plot_run import run as generate_verification_plots
            except ImportError:
                from .plot_run import run as generate_verification_plots
            try:
                generate_verification_plots(self.run_id)
            except Exception as exc:
                print(f"[save] Warning: automatic plot generation failed for {self.run_id}: {exc}")

        if self.export_image_on_completion:
            image_path = (
                Path(self.completion_image_path)
                if self.completion_image_path
                else self.visualization_dir / "completion.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            export_image(self, image_path)

        if self.pause_on_completion:
            pause_simulation(self)
        if self.quit_on_completion:
            quit_webots(self)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def complete_trial(
        self,
        *,
        reason: str,
        include_pcn: bool = False,
        include_rcn: bool = False,
        include_hmaps: bool = False,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        if self.trial_completed:
            return

        self.trial_completed = True
        self.trial_completion_reason = reason
        self.stop()
        self.save(
            include_pcn=include_pcn,
            include_rcn=include_rcn,
            include_hmaps=include_hmaps,
            status="completed",
            completion_reason=reason,
            extra_metrics=extra_metrics,
        )

    def clear(self):
        """
        Clears the saved state files for the Place Cell Network (PCN), Reward Cell Network (RCN),
        and the history maps by removing their corresponding pickle files from the appropriate directories.
        """
        clear_driver_outputs(self)
        write_metrics(self, status="cleared")
