import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from controller import Supervisor
import random
from typing import Optional, List, Dict, Any
import tkinter as tk
from tkinter import messagebox

# Add root directory to PYTHONPATH
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.place_cell_layer import PlaceCellLayer
from core.layers.reward_cell_layer import RewardCellLayer
from core.robot.robot_mode import RobotMode

from analysis.stats.stats_collector import stats_collector
import numpy as np

# Torch and NumPy settings
np.set_printoptions(precision=2)

class Driver(Supervisor):
    def initialization(
        self,
        mode: RobotMode = RobotMode.PLOTTING,
        run_time_hours: int = 2,
        randomize_start_loc: bool = True,
        start_loc: Optional[List[float]] = None,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        scales: Optional[List[Dict[str, Any]]] = None,
        rcn_learning_rates: Optional[List[float]] = None,
        stats_collector: Optional[stats_collector] = None,
        trial_id: Optional[str] = None,
        world_name: Optional[str] = None,
        goal_location: Optional[List[float]] = None,
        max_dist: Optional[float] = None,
        plot_bvc: Optional[bool] = False,
    ):
        """
        Initializes the Driver class, setting up the robot's sensors and neural networks.
        """
        # Supervisor basics
        self.robot = self.getFromDef("agent")
        self.robot_mode = mode
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"[DRIVER] Using device: {self.device}")
        self.dtype = torch.float32

        # Determine or set world name
        if world_name is None:
            world_path = self.getWorldPath()
            world_name = os.path.splitext(os.path.basename(world_path))[0]
        self.world_name = world_name

        # Directories for saving/loading data
        self.hmap_dir = os.path.join("pkl", self.world_name, "hmaps")
        self.network_dir = os.path.join("pkl", self.world_name, "networks")
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Stats / trial info
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        # Head direction layer size
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10

        # Robot parameters
        self.max_speed = 16 if mode != RobotMode.EXPLOIT else 8
        self.max_dist = max_dist
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756

        # Simulation run time
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))

        # Exploration/exploitation radius
        self.goal_r = {"explore": 0.3, "exploit": 0.8}
        self.goal_location = goal_location if goal_location else [-3, 3]

        # Default single scale if none provided
        if not scales:
            print("[DRIVER] No scales provided, using default scale")
            scales = [{
                "name": "default_scale",
                "num_pc": 500,
                "sigma_r": 0.5,
                "sigma_theta": 1.0
            }]
        self.scales = scales
        
        # Store learning rates for later use in RCN initialization
        self.rcn_learning_rates = rcn_learning_rates if rcn_learning_rates is not None else [0.1] * len(scales)

        # Random or fixed start
        if randomize_start_loc:
            while True:
                candidate = [
                    random.uniform(-2.3, 2.3),
                    0,
                    random.uniform(-2.3, 2.3)
                ]
                dist_to_goal = np.sqrt(
                    (candidate[0] - self.goal_location[0]) ** 2 +
                    (candidate[2] - self.goal_location[1]) ** 2
                )
                if dist_to_goal >= 1.0:
                    break
            self.robot.getField("translation").setSFVec3f(candidate)
            self.robot.resetPhysics()
        else:
            if start_loc is not None:
                self.robot.getField("translation").setSFVec3f([start_loc[0], 0, start_loc[1]])
                self.robot.resetPhysics()

        # Initialize sensors
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)
        self.range_finder = self.getDevice("range-finder")
        self.range_finder.enable(self.timestep)
        self.lidar_resolution = 720
        self.boundaries = torch.zeros((self.lidar_resolution, 1), device=self.device)

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)

        # Bumpers
        self.collided = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.left_bumper = self.getDevice("bumper_left")
        self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice("bumper_right")
        self.right_bumper.enable(self.timestep)

        # Motors / position sensors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        self.step_count = 0

        # Clear if in Oja's mode
        if self.robot_mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Load or init PCNs / RCNs
        self.pcns = []
        self.rcns = []
        self.load_pcns(enable_ojas, enable_stdp)
        self.load_rcns()

        # Head direction layer
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd, device="cpu")

        # Initialize alpha as a tensor of zeros with the same length as the number of scales
        self.alpha = torch.zeros(len(self.scales), dtype=self.dtype, device=self.device)

        # Initialize hmap_alpha with the correct shape
        self.hmap_alpha = torch.zeros((self.num_steps, len(self.scales)), device="cuda", dtype=torch.float32)

        # Prep for logging
        self.hmap_loc = np.zeros((self.num_steps, 3))
        self.hmap_hdn = torch.zeros((self.num_steps, self.n_hd), device="cpu", dtype=torch.float32)

        # For multi-scale place cell logs
        self.hmap_pcn_activities = []
        for scale_def in self.scales:
            n_pc = scale_def["num_pc"]
            self.hmap_pcn_activities.append(
                torch.zeros((self.num_steps, n_pc), device=self.device, dtype=torch.float32)
            )

        self.directional_reward_estimates = torch.zeros(self.n_hd, device=self.device) 

        # Rotation tracking for excessive loop detection
        self.rotation_accumulator = 0.0
        self.rotation_loop_count = 0
        self.steps_since_last_loop = 0
        self.last_heading_deg = None
        self.done = False
        
        # Parameters for loop detection
        self.LOOP_THRESHOLD = 3  # Number of loops before forcing exploration
        self.MAX_STEPS_BETWEEN_LOOPS = 5  # Max steps between loops to count towards threshold
        self.force_explore_count = 0 # Number of steps to force exploration (init to 0)

        # Optionally keep a single-scale reference
        self.pcn = self.pcns[0] if self.pcns else None
        self.plot_bvc = plot_bvc

        # Step once
        self.step(self.timestep)

    ##########################################################################
    #                           PCN / RCN LOADING                            #
    ##########################################################################
    def load_pcns(self, enable_ojas: Optional[bool], enable_stdp: Optional[bool]):
        self.pcns = []
        for i, scale_def in enumerate(self.scales):
            fname = f"pcn_scale_{i}.pkl"
            path = os.path.join(self.network_dir, fname)
            pcn = self._load_or_init_pcn_for_scale(path, scale_def, enable_ojas, enable_stdp)
            self.pcns.append(pcn)

    def _load_or_init_pcn_for_scale(self, path, scale_def, enable_ojas, enable_stdp):
        try:
            with open(path, "rb") as f:
                pcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing PCN from {path}")
        except:
            print(f"[DRIVER] Initializing new PCN for {path}")
            bvc = BoundaryVectorCellLayer(
                max_dist=self.max_dist,
                n_res=720,
                n_hd=self.n_hd,
                sigma_theta=scale_def.get("sigma_theta"),
                sigma_r=scale_def.get("sigma_r"),
                device=self.device,
            )
            pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=scale_def["num_pc"],
                timestep=self.timestep,
                n_hd=self.n_hd,
                device=self.device,
            )
        if enable_ojas is not None:
            pcn.enable_ojas = enable_ojas
        else:
            pcn.enable_ojas = (self.robot_mode == RobotMode.LEARN_OJAS)
        if enable_stdp is not None:
            pcn.enable_stdp = enable_stdp
        else:
            pcn.enable_stdp = (self.robot_mode == RobotMode.LEARN_HEBB)

        return pcn

    def load_rcns(self):
        self.rcns = []
        for i, scale_def in enumerate(self.scales):
            fname = f"rcn_scale_{i}.pkl"
            path = os.path.join(self.network_dir, fname)
            rcn = self._load_or_init_rcn_for_scale(path, scale_def, self.rcn_learning_rates[i])
            self.rcns.append(rcn)

    def _load_or_init_rcn_for_scale(self, path, scale_def, learning_rate):
        try:
            with open(path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing RCN from {path}")
        except:
            print(f"[DRIVER] Initializing new RCN for {path} with learning rate {learning_rate}")
            rcn = RewardCellLayer(
                num_place_cells=scale_def["num_pc"],
                num_replay=3,
                learning_rate=learning_rate,
                device=self.device,
            )
        return rcn


    ########################################### RUN LOOP ###########################################

    def run(self):
        """Runs the main control loop of the robot.

        The method manages the robot's behavior based on its current mode:
        - MANUAL_CONTROL: Allows user keyboard control
        - LEARN_OJAS/LEARN_HEBB/DMTP/PLOTTING: Runs exploration behavior
        - EXPLOIT: Runs goal-directed navigation
        - RECORDING: Records sensor data
        """

        print(f"[DRIVER] Starting robot in {self.robot_mode}")
        print(f"[DRIVER] Goal at {self.goal_location}")

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
        """
        Handles exploration for multi-scale usage, calling compute_pcn_activations().
        """
        for _ in range(self.tau_w):
            # 1) Sense environment
            self.sense()

            # 4) compute pcn_activations => fill self.pcn_activations_list
            self.compute_pcn_activations()

            # 5) If DMTP => reward updates
            if self.robot_mode == RobotMode.DMTP:
                actual_reward = self.get_actual_reward()
                for pcn, rcn in zip(self.pcns, self.rcns):
                    rcn.update_reward_cell_activations(pcn.place_cell_activations)
                    # rcn.td_update(pcn.place_cell_activations, next_reward=actual_reward)
                # # Turn towards heading 225°
                # desired_heading_deg = 90
                # # Compute the minimal angular difference (normalized to [-180, 180])
                # angle_to_turn_deg = desired_heading_deg - self.current_heading_deg
                # angle_to_turn_deg = ((angle_to_turn_deg + 180) % 360) - 180
                # angle_to_turn = np.deg2rad(angle_to_turn_deg)
                # self.turn(angle_to_turn)
                # self.check_goal_reached()
                # self.update_hmaps()
                # self.forward()        
                # break 

            # 6) If collisions => turn away
            if torch.any(self.collided):
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break

            # 7) Check goal, update hmaps, forward
            self.check_goal_reached()
            self.update_hmaps()
            self.forward()

        # A small random turn at the end
        self.turn(np.random.normal(0, np.deg2rad(30)))

    ########################################### EXPLOIT ###########################################
    def exploit(self):
        """
        Follows the reward gradient to reach the goal location using multiscale place field navigation.
        """
        #-------------------------------------------------------------------
        # 1) Sense and compute: update heading, place/boundary cell activations
        #-------------------------------------------------------------------
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()
        self.check_goal_reached()

        if self.force_explore_count > 0:
            self.force_explore_count -= 1
            if self.force_explore_count == 0:
                print("Forced exploration complete. Resuming EXPLOIT.")
            self.explore()
            return

        #-------------------------------------------------------------------
        # 2) Detect excessive rotation and enforce exploration if needed
        #-------------------------------------------------------------------
        if not hasattr(self, 'rotation_accumulator'):
            self.rotation_accumulator = 0.0
            self.rotation_loop_count = 0
            self.steps_since_last_loop = 0
        
        if not hasattr(self, 'last_heading_deg') or self.last_heading_deg is None:
            self.last_heading_deg = self.current_heading_deg

        # Calculate heading difference
        heading_diff = self.current_heading_deg - getattr(self, 'last_heading_deg', self.current_heading_deg)
        heading_diff = ((heading_diff + 180) % 360) - 180  # Normalize to [-180, 180]

        # Update rotation accumulator
        self.rotation_accumulator += abs(heading_diff)

        # Update last_heading_deg AFTER calculating the difference
        self.last_heading_deg = self.current_heading_deg

        if self.rotation_accumulator >= 360.0:
            self.rotation_loop_count += 1
            self.rotation_accumulator -= 360.0
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 1
            self.steps_since_last_loop = 0

            if self.rotation_loop_count >= self.LOOP_THRESHOLD:
                self.rotation_loop_count = 0
                self.rotation_accumulator = 0.0
                self.steps_since_last_loop = 0
                print(f"Detected {self.LOOP_THRESHOLD} consecutive loops within {self.MAX_STEPS_BETWEEN_LOOPS} steps")
                self.force_explore_count = 5
                self.explore()
                return
        else:
            self.steps_since_last_loop += 1
            if self.steps_since_last_loop > self.MAX_STEPS_BETWEEN_LOOPS:
                self.rotation_loop_count = 0

        #-------------------------------------------------------------------
        # 3) Compute potential rewards at multiple scales
        #-------------------------------------------------------------------
        num_steps_preplay = 1  # Number of future steps to "preplay"
        pot_rew_scales = []

        for i, scale_def in enumerate(self.scales):
            pcn, rcn = self.pcns[i], self.rcns[i]
            pot_rew = torch.empty(self.n_hd, dtype=self.dtype, device=self.device)
            for d in range(self.n_hd):
                pcn_activations = pcn.preplay(d, num_steps=num_steps_preplay)
                rcn.update_reward_cell_activations(pcn_activations, visit=False)
                pot_rew[d] = torch.max(torch.nan_to_num(rcn.reward_cell_activations))
            pot_rew_scales.append(pot_rew)

        if len(pot_rew_scales) == 0:
            print("No valid scales selected. Defaulting to first available scale.")
            pot_rew_scales.append(torch.zeros(self.n_hd, dtype=self.dtype, device=self.device))

        # Initialize alpha if not already done
        if not hasattr(self, 'alpha'):
            self.alpha = torch.tensor(0.5, dtype=self.dtype, device=self.device)

        # Normalize and blend across scales
        pot_rew_scales = torch.stack(pot_rew_scales)
        pot_rew_scales /= (pot_rew_scales.max(dim=1, keepdim=True)[0] + 1e-6)  # Normalize

        # Compute gradients for each scale
        grads = torch.sum(torch.abs(torch.diff(pot_rew_scales, dim=1)), dim=1)  # Shape: (num_scales,)

        # Normalize the gradients to sum to 1, ensuring proper mixing
        mixing_weights = grads / (grads.sum() + 1e-6)  # Shape: (num_scales,)
        mixing_weights = torch.clamp(mixing_weights, min=0.0, max=1.0)
        mixing_weights /= mixing_weights.sum()

        # Weighted sum over all scales based on dynamic mixing_weights
        combined_pot_rew = torch.sum(mixing_weights[:, None] * pot_rew_scales, dim=0)

        # Now compute alpha as a vector that shows each scale's contribution
        self.alpha = grads / (grads.sum() + 1e-6)

        #-------------------------------------------------------------------
        # 4) Wall avoidance logic
        #-------------------------------------------------------------------
        # Roll boundaries so index 0 corresponds to "front", etc.
        boundaries_rolled = torch.roll(self.boundaries, shifts=len(self.boundaries)//2)

        num_points_per_hd = len(boundaries_rolled) // self.n_hd
        distances_per_hd = []
        cancelled_angles = []  # optional: store which angles get penalized

        for i in range(self.n_hd):
            start_idx = i * num_points_per_hd
            end_idx   = (i+1) * num_points_per_hd
            min_dist  = torch.min(boundaries_rolled[start_idx:end_idx])
            distances_per_hd.append(min_dist)

            if min_dist < 1.0:
                # This heading is "too close" to a wall
                cancelled_angles.append(i * (360 / self.n_hd))

        distances_per_hd = torch.tensor(distances_per_hd, device=self.device, dtype=self.dtype)

        # Compute penalty mask and apply an exponential penalty
        penalty_mask = distances_per_hd < 1.0
        # Example: penalty gets stronger as distance gets smaller (dist < 1)
        penalties = torch.exp(-1.0 * (1.0 - distances_per_hd))  # tweak scale if desired

        combined_pot_rew[penalty_mask] *= penalties[penalty_mask]

        # Renormalize after penalty
        combined_sum = combined_pot_rew.sum()
        if combined_sum > 1e-6:
            combined_pot_rew /= combined_sum
        else:
            # If everything got zeroed out, fallback to explore or skip
            print("All directions heavily penalized. Forcing exploration.")
            self.explore()
            return

        #-------------------------------------------------------------------
        # 5) Compute action heading
        #-------------------------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi * (1 - 1 / self.n_hd), self.n_hd, device=self.device, dtype=self.dtype)
        sin_component = torch.sum(torch.sin(angles) * combined_pot_rew)
        cos_component = torch.sum(torch.cos(angles) * combined_pot_rew)
        action_angle = torch.atan2(sin_component, cos_component)

        if action_angle < 0:
            action_angle += 2 * np.pi

        self.action_heading_deg = float(torch.rad2deg(action_angle).item())

        #-------------------------------------------------------------------
        # 6) Execute movement with `tau_w` steps
        #-------------------------------------------------------------------
        angle_to_turn_deg = self.action_heading_deg - self.current_heading_deg
        angle_to_turn_deg = (angle_to_turn_deg + 180) % 360 - 180
        angle_to_turn = torch.deg2rad(torch.tensor(angle_to_turn_deg, dtype=self.dtype, device=self.device))

        self.turn(angle_to_turn.item())

        # Move forward for `tau_w` steps
        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
            self.check_goal_reached()

            # Update rotation accumulator
            heading_diff = self.current_heading_deg - getattr(self, 'last_heading_deg', self.current_heading_deg)
            heading_diff = ((heading_diff + 180) % 360) - 180
            self.rotation_accumulator += abs(heading_diff)
            self.last_heading_deg = self.current_heading_deg

            # Collision handling
            if torch.any(self.collided):
                print("Collision detected. Turning back and retrying.")
                self.turn(np.pi)  # Reverse direction
                for _ in range(self.tau_w):
                    self.sense()
                    self.compute_pcn_activations()
                    self.update_hmaps()
                    self.forward()
                    self.check_goal_reached()
                    if torch.any(self.collided):
                        print("Still stuck after retry. Forcing future exploration.")
                        self.force_explore_count = 5
                        self.explore()
                        return
                break

            if self.done:
                return

        # Optionally re-sense
        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()


    def plot_current_heading(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        current_heading_rad = (3 * np.pi / 2) - np.deg2rad(self.current_heading_deg)
        action_heading_rad = (3 * np.pi / 2) - np.deg2rad(self.action_heading_deg) if hasattr(self, 'action_heading_deg') else None
        cancelled_rad_list = [(3 * np.pi / 2) - np.deg2rad(deg) for deg in getattr(self, 'cancelled_angles_deg', [])]

        angles = np.linspace(0, 2 * np.pi, len(self.boundaries), endpoint=False)
        ax.plot((angles + np.pi) % (2 * np.pi), self.boundaries.cpu().numpy().flatten(), 'b-', label="Corrected LiDAR Readings")
        ax.quiver(current_heading_rad, 0, np.cos(current_heading_rad), np.sin(current_heading_rad), scale=1, scale_units='inches', color='r', label="Current Heading")
        if action_heading_rad is not None:
            ax.quiver(action_heading_rad, 0, np.cos(action_heading_rad), np.sin(action_heading_rad), scale=1, scale_units='inches', color='g', label="Action Heading")
        if cancelled_rad_list:
            ax.scatter(cancelled_rad_list, [4.5] * len(cancelled_rad_list), c='k', marker='x', s=50, label='Cancelled Angles')

        ax.set_theta_direction(-1)
        ax.set_theta_offset(-np.pi / 2)
        ax.set_rmax(5)
        ax.set_rticks([1, 2, 3, 4, 5])
        ax.set_rlabel_position(-22.5)
        ax.grid(True)
        ax.text(current_heading_rad, 5, f'Heading: {self.current_heading_deg}°', horizontalalignment='center', verticalalignment='bottom', color='red')
        if action_heading_rad is not None:
            ax.text(action_heading_rad, 5, f'Action: {self.action_heading_deg}°', horizontalalignment='center', verticalalignment='bottom', color='green')

        plt.show()

    ########################################### SENSE ###########################################
    def sense(self):
        """
        Uses sensors to update range-image, heading, boundary data, collision flags, etc.
        """
        # Get the latest boundary data from range finder
        boundaries = self.range_finder.getRangeImage()

        # Update global heading (0–360)
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))

        # Shift boundary data based on global heading
        self.boundaries = torch.roll(
            torch.tensor(boundaries, dtype=self.dtype, device=self.device),
            2 * self.current_heading_deg
        )

        # Convert heading to radians for HD-layer input
        current_heading_rad = np.deg2rad(self.current_heading_deg)
        v_in = torch.tensor([np.cos(current_heading_rad), np.sin(current_heading_rad)],
                            dtype=self.dtype, device=self.device)

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
        Updates place cell activations for each scale based on sensor data.
        Populates self.pcn_activations_list, which we'll use in update_hmaps().
        """
        # For convenience, store them in a list
        self.pcn_activations_list = []

        # For each scale's PCN, call get_place_cell_activations
        for pcn in self.pcns:
            pcn.get_place_cell_activations(
                distances=self.boundaries,
                angles=torch.linspace(0, 2 * np.pi, 720, device=self.device),
                hd_activations=self.hd_activations,
                collided=torch.any(self.collided),
            )
            if self.plot_bvc:
                pcn.bvc_layer.plot_activation(self.boundaries.cpu())
            # Append activations to pcn_activations_list
            act = pcn.place_cell_activations.clone().detach()
            self.pcn_activations_list.append(act)

        # Optionally step if you want to sync
        self.step(self.timestep)

    ########################################### CHECK GOAL REACHED ###########################################
    def check_goal_reached(self):
        """
        Check if the robot has reached its goal or if time has expired.
        If reached and in the correct mode, call auto_pilot() and save logs.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        time_limit = 45 # minutes

        if self.robot_mode in (RobotMode.LEARN_OJAS, RobotMode.LEARN_HEBB, RobotMode.PLOTTING) \
                and self.getTime() >= 60 * self.run_time_minutes:
            self.stop()
            self.save(include_pcn=self.robot_mode != RobotMode.PLOTTING,
                    include_rcn=self.robot_mode != RobotMode.PLOTTING,
                    include_hmaps=True)

        elif self.robot_mode == RobotMode.DMTP and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"]
        ):
            self.auto_pilot()
            self.stop()
            for pcn, rcn in zip(self.pcns, self.rcns):
                print(f"Replay called on {rcn}")
                rcn.update_reward_cell_activations(pcn.place_cell_activations, visit=True)
                rcn.replay(pcn=pcn)
            self.save(include_pcn=True, include_rcn=True)

        elif self.robot_mode == RobotMode.EXPLOIT and torch.allclose(
            torch.tensor(self.goal_location, dtype=self.dtype, device=self.device),
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["exploit"]
        ):
            if self.stats_collector:
                # Update and save stats once
                self.stats_collector.update_stat("trial_id", self.trial_id)
                self.stats_collector.update_stat("start_location", self.start_loc)
                self.stats_collector.update_stat("goal_location", self.goal_location)
                self.stats_collector.update_stat("total_distance_traveled", round(self.compute_path_length(), 2))
                self.stats_collector.update_stat("total_time_secs", round(self.getTime(), 2))
                self.stats_collector.update_stat("success", self.getTime() <= time_limit * 60)
                self.stats_collector.save_stats(self.trial_id)
                
                # Print stats
                print(f"Trial {self.trial_id} completed.")
                print(f"Start location: {self.start_loc}")
                print(f"Goal location: {self.goal_location}")
                print(f"Total distance traveled: {round(self.compute_path_length(), 2)} meters.")
                print(f"Total time taken: {round(self.getTime(), 2)} seconds.")
                print(f"Success: {self.getTime() <= time_limit * 60}")
                
                self.stop()
                self.done = True
                self.save()
                return
            else:
                self.stop()
                self.save()
                self.done = True
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                return

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
            torch.tensor([curr_pos[0], curr_pos[2]], dtype=self.dtype, device=self.device),
            atol=self.goal_r["explore"]
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]

            # Compute desired heading to face the goal
            if delta_x >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))).item()
                if delta_y >= 0:
                    desired = 2 * np.pi - theta
                else:
                    desired = np.pi + theta
            elif delta_y >= 0:
                theta = torch.atan2(torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device))).item()
                desired = (np.pi / 2) - theta
            else:
                theta = torch.atan2(torch.abs(torch.tensor(delta_x, dtype=self.dtype, device=self.device)),
                                    torch.abs(torch.tensor(delta_y, dtype=self.dtype, device=self.device))).item()
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
        """
        Store agent position, head direction activations, place cell activations, and alpha values.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # 1) Record agent location
            self.hmap_loc[self.step_count] = curr_pos

            # 2) Record HDN on CPU
            self.hmap_hdn[self.step_count] = self.hd_activations.clone().detach().cpu()

            # 3) Dynamically resize hmap_pcn_activities if needed
            if len(self.hmap_pcn_activities) != len(self.pcn_activations_list):
                self.hmap_pcn_activities = [
                    torch.zeros((self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32)
                    for act in self.pcn_activations_list
                ]

            # 4) Record place cell activations for each scale
            for i, act in enumerate(self.pcn_activations_list):
                if self.hmap_pcn_activities[i].shape[1] != act.shape[0]:
                    self.hmap_pcn_activities[i] = torch.zeros(
                        (self.num_steps, act.shape[0]), device="cuda", dtype=torch.float32
                    )
                self.hmap_pcn_activities[i][self.step_count] = act.clone().detach().cpu()

            # 5) Ensure alpha value is recorded
            if self.step_count < self.num_steps:
                self.hmap_alpha[self.step_count, :] = self.alpha.clone().detach().cpu()

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
                    curr_pos[2] - self.goal_location[1]
                ],
                dtype=self.dtype,
                device=self.device
            )
        )

        # Determine the correct goal radius based on the current mode
        if self.mode == RobotMode.EXPLOIT:
            goal_radius = self.goal_r["exploit"]
        else:  # Default to "explore" goal radius for all other modes
            goal_radius = self.goal_r["explore"]

        # Return 1.0 reward if within goal radius, else 0.0
        if distance_to_goal <= goal_radius:
            return 1.0 # Goal reached
        else:
            return 0.0

    def save(
        self,
        include_pcn: bool = False,
        include_rcn: bool = False,
        include_hmaps: bool = False,
    ):
        """
        Saves:
        - PCN networks (one file per scale) if include_pcn=True
        - RCN networks (one file per scale) if include_rcn=True
        - The history maps if include_hmaps=True
        """
        files_saved = []

        # Ensure directories exist
        os.makedirs(self.hmap_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1) Save each scale's PCN (if requested)
        # ----------------------------------------------------------------------
        if include_pcn:
            for i, pcn in enumerate(self.pcns):
                pcn_path = os.path.join(self.network_dir, f"pcn_scale_{i}.pkl")
                with open(pcn_path, "wb") as f:
                    pickle.dump(pcn, f)
                files_saved.append(pcn_path)

        # ----------------------------------------------------------------------
        # 2) Save each scale's RCN (if requested)
        # ----------------------------------------------------------------------
        if include_rcn:
            for i, rcn in enumerate(self.rcns):
                rcn_path = os.path.join(self.network_dir, f"rcn_scale_{i}.pkl")
                with open(rcn_path, "wb") as f:
                    pickle.dump(rcn, f)
                files_saved.append(rcn_path)

        # ----------------------------------------------------------------------
        # 3) Save the history maps if requested
        # ----------------------------------------------------------------------
        if include_hmaps:
            # (a) Agent location
            hmap_loc_path = os.path.join(self.hmap_dir, "hmap_loc.pkl")
            with open(hmap_loc_path, "wb") as f:
                # store only up to self.step_count
                pickle.dump(self.hmap_loc[: self.step_count], f)
            files_saved.append(hmap_loc_path)

            # (b) Head direction history
            hmap_hdn_path = os.path.join(self.hmap_dir, "hmap_hdn.pkl")
            with open(hmap_hdn_path, "wb") as f:
                pickle.dump(self.hmap_hdn[: self.step_count].cpu(), f)
            files_saved.append(hmap_hdn_path)

            # (c) Place-cell history maps for each scale
            for i, pc_history in enumerate(self.hmap_pcn_activities):
                # pc_history shape: (num_steps, num_pc_for_scale_i)
                # We'll name file as "hmap_pcn_scale_i.pkl"
                hmap_scale_path = os.path.join(self.hmap_dir, f"hmap_pcn_scale_{i}.pkl")
                with open(hmap_scale_path, "wb") as f:
                    # Move to CPU or .numpy() if desired
                    pc_data = pc_history[: self.step_count].cpu().numpy()
                    pickle.dump(pc_data, f)
                files_saved.append(hmap_scale_path)

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        messagebox.showinfo("Information", "Press OK to save data")
        root.destroy()

        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def clear(self):
        """
        Removes all scale-specific PCN/RCN files and any hmap files.

        This includes:
        - pcn_scale_*.pkl
        - rcn_scale_*.pkl
        - pcn.pkl, rcn.pkl (legacy)
        - hmap_* files in self.hmap_dir
        """
        # 1. Remove all per-scale PCN/RCN files
        if os.path.exists(self.network_dir):
            for fname in os.listdir(self.network_dir):
                # Delete pcn_scale_* or rcn_scale_* files
                if fname.startswith("pcn_scale_") or fname.startswith("rcn_scale_"):
                    full_path = os.path.join(self.network_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

            # 2. Remove the old single-scale files if present
            for legacy_file in ["pcn.pkl", "rcn.pkl"]:
                path = os.path.join(self.network_dir, legacy_file)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed: {path}")
                    except FileNotFoundError:
                        pass

        # 3. Remove any scale-specific hmap files (and all hmap files) from self.hmap_dir
        if os.path.exists(self.hmap_dir):
            for fname in os.listdir(self.hmap_dir):
                # For example: hmap_scale_0.pkl, hmap_scale_1.pkl, or any other hmap_*
                if fname.startswith("hmap_"):
                    full_path = os.path.join(self.hmap_dir, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except FileNotFoundError:
                        pass

        print("[DRIVER] Finished clearing old scale PCNs, RCNs, and hmap files.")
