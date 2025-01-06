import time
import numpy as np
import tensorflow as tf
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from astropy.stats import circmean, circvar
import pickle
import os
import tkinter as tk
from tkinter import N, messagebox
from typing import Optional, List
from scipy.ndimage import gaussian_filter1d
from controller import Supervisor, Robot
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from layers.head_direction_layer import HeadDirectionLayer
from layers.place_cell_layer import PlaceCellLayer
from layers.reward_cell_layer import RewardCellLayer
from objects.RobotMode import RobotMode
from analysis.stats_collector import stats_collector

tf.random.set_seed(5)
np.set_printoptions(precision=2)
PI = tf.constant(np.pi)
rng = default_rng()  # random number generator
cmap = get_cmap("plasma")

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
        hmap_z (ndarray): History of place cell activations.
        hmap_h (ndarray): History of head direction activations.
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
        last_reward (float): Reward received in previous step.
    """

    def initialization(
        self,
        mode=RobotMode.PLOTTING,
        randomize_start_loc: bool = True,
        run_time_hours: int = 1,
        start_loc: Optional[List[int]] = [2,2],
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        enable_multiscale: Optional[bool] = False,
        stats_collector: Optional[stats_collector] = None,
        trial_id: Optional[str] = None,
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
        self.mode = mode
        self.done = False

        # Model parameters
        self.num_small_place_cells = 300  # Adjust as needed
        self.num_large_place_cells = 200  # Adjust as needed
        self.num_reward_cells = 1
        self.n_hd = 8
        self.timestep = 32 * 3
        self.tau_w = 10  # time constant for the window function
        if self.mode == RobotMode.EXPLOIT:
            self.tau_w = 3
        self.enable_multiscale = enable_multiscale

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

        self.hmap_x = np.zeros(self.num_steps)  # x-coordinates
        self.hmap_y = np.zeros(self.num_steps)  # y-coordinate

        self.hmap_h = np.zeros(
            (self.num_steps, self.n_hd)
        )  # head direction cell activations
        self.hmap_g = np.zeros(self.num_steps)  # goal estimates
        self.hmap_vis_density = np.zeros(self.num_steps)  # visual density

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
        self.collided = tf.Variable(np.zeros(2, np.int32))
        self.rotation_field = self.robot.getField("rotation")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice("right wheel sensor")
        self.right_position_sensor.enable(self.timestep)

        self.start_loc = start_loc
        self.stats_collector = stats_collector
        self.trial_id = trial_id

        if self.mode == RobotMode.LEARN_OJAS:
            self.clear()

        # Load or initialize PCN layer(s)
        self.load_pcn(
            num_small_place_cells=self.num_small_place_cells,
            num_large_place_cells=self.num_large_place_cells,
            n_hd=self.n_hd,
            timestep=self.timestep,
            enable_ojas=enable_ojas,
            enable_stdp=enable_stdp,
        )

        # Initialize history maps for both layers
        self.hmap_z_small = np.zeros((self.num_steps, self.num_small_place_cells))
        if self.enable_multiscale:
            self.hmap_z_large = np.zeros((self.num_steps, self.num_large_place_cells))

        self.load_rcn(
            num_reward_cells=self.num_reward_cells,
            num_small_place_cells=self.num_small_place_cells,
            num_large_place_cells=self.num_large_place_cells,
            num_replay=1,
        )

        print(f"SMALL: Oja's: {self.pcn_small.enable_ojas}, STDP: {self.pcn_small.enable_stdp}")
        if self.enable_multiscale:
            print(f"LARGE: Oja's: {self.pcn_large.enable_ojas}, STDP: {self.pcn_large.enable_stdp}")
            print()

        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd)

        # Initialize boundaries
        self.boundary_data = tf.Variable(tf.zeros((720, 1)))

        self.directional_reward_estimates = tf.zeros(self.n_hd)
        self.step(self.timestep)
        self.step_count += 1

        # Initialize goal
        self.goal_location = [-1, 1]
        self.expected_reward = 0
        self.last_reward = 0

        if randomize_start_loc:
            while True:
                INITIAL = [rng.uniform(-2.3, 2.3), 0.5, rng.uniform(-2.3, 2.3)]
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
                [self.start_loc[0], 0.5, self.start_loc[1]]
            )
            self.robot.resetPhysics()

        self.sense()
        self.compute_pcn_activations()
        self.update_hmaps()

    def load_pcn(
        self,
        num_small_place_cells: int,
        num_large_place_cells: int,
        n_hd: int,
        timestep: int,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
    ):
        """Loads the multi-scale or single-scale PCN(s), and conditionally enables Oja’s or STDP.

        If enable_multiscale == True:
        - pcn_small and pcn_large are loaded or created
        Else:
        - Only pcn_small is loaded or created
        Also toggles self.pcn_small.enable_ojas / enable_stdp depending on arguments or mode.
        Same for self.pcn_large if multiscale is on.
        """
        if self.enable_multiscale:
            # ------ SMALL PCN ------
            try:
                with open("pcn_small.pkl", "rb") as f:
                    self.pcn_small = pickle.load(f)
                    self.pcn_small.reset_activations()
                    print("Loaded existing Small Place Cell Network.")
            except FileNotFoundError:
                bvc = BoundaryVectorCellLayer(
                    max_dist=12,
                    input_dim=720,
                    n_hd=n_hd,
                    sigma_ang=90,
                    sigma_d=0.5,
                )
                self.pcn_small = PlaceCellLayer(
                    bvc_layer=bvc,
                    num_pc=num_small_place_cells,
                    timestep=timestep,
                    n_hd=n_hd,
                    enable_ojas=enable_ojas,  # We'll set this below
                    enable_stdp=enable_stdp,  # We'll set this below
                    modulate_by_vis_density=False,
                )
                print("Initialized new Small Place Cell Network.")

            # ------ LARGE PCN ------
            try:
                with open("pcn_large.pkl", "rb") as f:
                    self.pcn_large = pickle.load(f)
                    self.pcn_large.reset_activations()
                    print("Loaded existing Large Place Cell Network.")
            except FileNotFoundError:
                bvc = BoundaryVectorCellLayer(
                    max_dist=12,
                    input_dim=720,
                    n_hd=n_hd,
                    sigma_ang=90,
                    sigma_d=1.5,
                )
                self.pcn_large = PlaceCellLayer(
                    bvc_layer=bvc,
                    num_pc=num_large_place_cells,
                    timestep=timestep,
                    n_hd=n_hd,
                    enable_ojas=enable_ojas,  # set below
                    enable_stdp=enable_stdp,  # set below
                    modulate_by_vis_density=False,
                )
                print("Initialized new Large Place Cell Network.")

            if enable_ojas is not None:
                self.pcn_small.enable_ojas = enable_ojas
                self.pcn_large.enable_ojas = enable_ojas
            else: # Default
                if self.mode == RobotMode.LEARN_OJAS:
                    self.pcn_small.enable_ojas = True
                    self.pcn_large.enable_ojas = True
                else:
                    self.pcn_small.enable_ojas = False
                    self.pcn_large.enable_ojas = False

            if enable_stdp is not None:
                self.pcn_small.enable_stdp = enable_stdp
                self.pcn_large.enable_stdp = enable_stdp
            else: # Default
                if self.mode == RobotMode.LEARN_HEBB or self.mode == RobotMode.DMTP:
                    self.pcn_small.enable_stdp = True
                    self.pcn_large.enable_stdp = True
                else:
                    self.pcn_small.enable_stdp = False
                    self.pcn_large.enable_stdp = False

        else:
            # SINGLE-SCALE: Just pcn_small
            try:
                with open("pcn_small.pkl", "rb") as f:
                    self.pcn_small = pickle.load(f)
                    self.pcn_small.reset_activations()
                    print("Loaded existing Place Cell Network.")
            except FileNotFoundError:
                bvc = BoundaryVectorCellLayer(
                    max_dist=12,
                    input_dim=720,
                    n_hd=n_hd,
                    sigma_ang=90,
                    sigma_d=0.5,
                )
                self.pcn_small = PlaceCellLayer(
                    bvc_layer=bvc,
                    num_pc=num_small_place_cells,
                    timestep=timestep,
                    n_hd=n_hd,
                    enable_ojas=enable_ojas,  # We'll handle below
                    enable_stdp=enable_stdp,  # We'll handle below
                    modulate_by_vis_density=False,
                )
                print("Initialized new Place Cell Network.")

            # Mirror the original logic for Oja’s / STDP
            if enable_ojas is not None:
                self.pcn_small.enable_ojas = enable_ojas
            else: # Default
                if self.mode == RobotMode.LEARN_OJAS:
                    self.pcn_small.enable_ojas = True
                else:
                    self.pcn_small.enable_ojas = False

            if enable_stdp is not None:
                self.pcn_small.enable_stdp = enable_stdp
            else: # Default
                if self.mode == RobotMode.LEARN_HEBB or self.mode == RobotMode.DMTP:
                    self.pcn_small.enable_stdp = True
                else:
                    self.pcn_small.enable_stdp = False

    def load_rcn(
        self,
        num_reward_cells: int,
        num_small_place_cells: int,
        num_large_place_cells: int,
        num_replay: int,
    ):
        """
        Loads or initializes one or two Reward Cell Networks, depending on
        enable_multiscale.

        If enable_multiscale is True:
            - Loads/initializes self.rcn_small (for small scale)
            - Loads/initializes self.rcn_large (for large scale)
        Else:
            - Loads/initializes a single self.rcn
        """

        if self.enable_multiscale:
            # --- RCN SMALL ---
            try:
                with open("rcn_small.pkl", "rb") as f:
                    self.rcn_small = pickle.load(f)
                    print("Loaded existing Small Reward Cell Network.")
            except FileNotFoundError:
                self.rcn_small = RewardCellLayer(
                    num_reward_cells=num_reward_cells,
                    input_dim=num_small_place_cells,
                    num_replay=num_replay,
                )
                print("Initialized new Small Reward Cell Network.")

            # --- RCN LARGE ---
            try:
                with open("rcn_large.pkl", "rb") as f:
                    self.rcn_large = pickle.load(f)
                    print("Loaded existing Large Reward Cell Network.")
            except FileNotFoundError:
                self.rcn_large = RewardCellLayer(
                    num_reward_cells=num_reward_cells,
                    input_dim=num_large_place_cells,
                    num_replay=num_replay,
                )
                print("Initialized new Large Reward Cell Network.")

        else:
            # Single-scale approach
            try:
                with open("rcn_small.pkl", "rb") as f:
                    self.rcn_small = pickle.load(f)
                    print("Loaded existing Reward Cell Network.")
            except FileNotFoundError:
                self.rcn_small = RewardCellLayer(
                    num_reward_cells=num_reward_cells,
                    input_dim=num_small_place_cells,  # Or whichever you prefer
                    num_replay=num_replay,
                )
                print("Initialized new Reward Cell Network.")


    ########################################### RUN LOOP ###########################################

    def run(self):
        """Runs the main control loop of the robot.

        The method manages the robot's behavior based on its current mode:
        - MANUAL_CONTROL: Allows user keyboard control
        - LEARN_OJAS/LEARN_HEBB/DMTP/PLOTTING: Runs exploration behavior
        - EXPLOIT: Runs goal-directed navigation
        - RECORDING: Records sensor data
        """

        print(f"Starting robot in stage {self.mode}")
        print(f"Mutliscale: {self.enable_multiscale}")

        while True:
            # Handle the robot's state
            if self.mode == RobotMode.MANUAL_CONTROL:
                self.manual_control()

            if self.done:
                break

            elif (
                self.mode == RobotMode.LEARN_OJAS
                or self.mode == RobotMode.LEARN_HEBB
                or self.mode == RobotMode.DMTP
                or self.mode == RobotMode.PLOTTING
            ):
                self.explore()

            elif self.mode == RobotMode.EXPLOIT:
                self.exploit()

            elif self.mode == RobotMode.RECORDING:
                self.recording()

            else:
                print("Unknown state. Exiting...")
                break

    ########################################### EXPLORE ###########################################

    def explore(self) -> None:
        """
        Handles the exploration mode logic for the robot.

        If enable_multiscale is True, train small and large networks in parallel:
            - self.rcn_small + self.pcn_small
            - self.rcn_large + self.pcn_large
        If enable_multiscale is False, train single network:
            - self.rcn + self.pcn_large  (or whichever one you use)
        """
        for step in range(self.tau_w):
            self.sense()

            if self.enable_multiscale:
                if self.mode == RobotMode.DMTP:
                    actual_reward = self.get_actual_reward()
                    # 1) Update & TD for small scale
                    self.rcn_small.update_reward_cell_activations(
                        self.pcn_small.place_cell_activations
                    )
                    self.rcn_small.td_update(
                        self.pcn_small.place_cell_activations, next_reward=actual_reward
                    )

                    # 2) Update & TD for large scale
                    self.rcn_large.update_reward_cell_activations(
                        self.pcn_large.place_cell_activations
                    )
                    self.rcn_large.td_update(
                        self.pcn_large.place_cell_activations, next_reward=actual_reward
                    )
            else:
                if self.mode == RobotMode.DMTP:
                    actual_reward = self.get_actual_reward()
                    # Single-scale approach
                    self.rcn_small.update_reward_cell_activations(
                        self.pcn_small.place_cell_activations
                    )
                    self.rcn_small.td_update(
                        self.pcn_small.place_cell_activations, next_reward=actual_reward
                    )

            # -----------------------------
            # COLLISION / MOVEMENT LOGIC
            # -----------------------------
            if np.any(self.collided):
                # E.g. turn randomly, break from loop
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.turn(random_angle)
                break

            # Compute new PCN activations and update maps
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
            self.check_goal_reached()

        # Possibly turn randomly after finishing the for-loop
        self.turn(np.random.normal(0, np.deg2rad(30)))
    
    def get_actual_reward(self):
        """Determines the actual reward for the agent at the current state.

        Returns:
            float: The actual reward value (1.0 if at goal, 0.0 otherwise)
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        distance_to_goal = np.linalg.norm(
            [curr_pos[0] - self.goal_location[0], curr_pos[2] - self.goal_location[1]]
        )

        # Determine the correct goal radius based on the current mode
        if self.mode == RobotMode.EXPLOIT:
            goal_radius = self.goal_r["exploit"]
        else:  # Default to "explore" goal radius for all other modes
            goal_radius = self.goal_r["explore"]

        # Check if the robot is within the goal radius
        if distance_to_goal <= goal_radius:
            return 1.0  # Reward for reaching the goal
        else:
            return 0.0  # No reward otherwise

    ########################################### EXPLOIT ###########################################
    def exploit(self):
        """
        Enhanced goal-directed navigation using learned reward maps.
        Dynamically switches between small and large reward scales based on environmental context.
        """

        # 1) Stop movement and update sensor readings
        self.stop()
        self.sense()

        # 2) Always compute place cell activations
        self.compute_pcn_activations()

        # 3) Update history maps and check for goal reached
        self.update_hmaps()
        self.check_goal_reached()

        # 4) Proceed only if enough steps have been taken
        if self.step_count <= self.tau_w:
            return  # Not enough steps for an 'exploit' action yet

        # Number of steps to "preplay" or look ahead
        num_steps_preplay = 2  

        # Initialize directional reward estimates
        self.directional_reward_estimates = None

        # Initialize current scale if not set
        if not hasattr(self, "current_scale"):
            self.current_scale = "None"  # Initialize with a placeholder scale

        if not hasattr(self, "current_mode"):
            self.current_mode = "None"

        self.low_reward_state = False

        # -------------------------------------------------
        #  MULTISCALE REWARD COMPUTATION
        # -------------------------------------------------
        if self.enable_multiscale:
            # --- SMALL LAYER RCN ---
            self.rcn_small.update_reward_cell_activations(
                self.pcn_small.place_cell_activations, visit=True
            )
            max_reward_activation_small = tf.reduce_max(
                self.rcn_small.reward_cell_activations
            )

            # --- LARGE LAYER RCN ---
            self.rcn_large.update_reward_cell_activations(
                self.pcn_large.place_cell_activations, visit=True
            )
            max_reward_activation_large = tf.reduce_max(
                self.rcn_large.reward_cell_activations
            )

            # Compute context
            curr_pos = self.robot.getField("translation").getSFVec3f()
            distance_to_goal = np.linalg.norm(
                [curr_pos[0] - self.goal_location[0], curr_pos[2] - self.goal_location[1]]
            )

        else:
            # Single-scale reward
            self.rcn_small.update_reward_cell_activations(
                self.pcn_small.place_cell_activations, visit=True
            )
            max_reward_activation = tf.reduce_max(
                self.rcn_small.reward_cell_activations
            )

        # Log the reward detection
        if self.enable_multiscale:
            max_small = max_reward_activation_small.numpy()
            max_large = max_reward_activation_large.numpy()

            # Check if they are both above some threshold
            if self.low_reward_state and (max_small > 1e-4 or max_large > 1e-4):
                print("Reward signals picked back up.")
                self.low_reward_state = False

        else:
            max_single = max_reward_activation.numpy()
            if self.low_reward_state and (max_single > 1e-4):
                print("Reward signals picked back up.")
                self.low_reward_state = False


        # -------------------------------------------------
        # TRIGGER EXPLORE() IF NECESSARY
        # -------------------------------------------------
        should_explore = False

        if self.enable_multiscale:
            # Condition 1: Both small and large activations are too low
            if max_reward_activation_small <= 1e-6 and max_reward_activation_large <= 1e-6:
                self.low_reward_state = True
                should_explore = True

        else:
            # Condition 2: Single-scale reward is too low
            if max_reward_activation <= 1e-6:
                self.low_reward_state = True
                should_explore = True

        # Condition 3: Directional estimates are all near zero
        if self.directional_reward_estimates is not None and np.sum(self.directional_reward_estimates) < 1e-6:
            self.low_reward_state = True
            should_explore = True

        # Condition 4: Collision detected
        if np.any(self.collided):
            should_explore = True

        # Execute exploration if any condition is met
        if should_explore:
            self.stop()
            self.explore()
            return

        # -------------------------------------------------
        #  CALCULATE POTENTIAL REWARD FOR EACH DIRECTION
        # -------------------------------------------------
        angles = np.linspace(0, 2 * np.pi, self.n_hd, endpoint=False)
        if self.enable_multiscale:
            pot_rew_small = np.zeros(self.n_hd)
            pot_rew_large = np.zeros(self.n_hd)

            # Compute directional rewards for small and large scale
            for d in range(self.n_hd):
                # Preplay or predict small PCN activations
                pcn_small_future = self.pcn_small.preplay(d, num_steps=num_steps_preplay)
                self.rcn_small.update_reward_cell_activations(pcn_small_future)
                pot_rew_small[d] = tf.reduce_max(
                    np.nan_to_num(self.rcn_small.reward_cell_activations)
                )

                # Similarly for large PCN
                pcn_large_future = self.pcn_large.preplay(d, num_steps=num_steps_preplay)
                self.rcn_large.update_reward_cell_activations(pcn_large_future)
                pot_rew_large[d] = tf.reduce_max(
                    np.nan_to_num(self.rcn_large.reward_cell_activations)
                )

            # -------------------------------------------------
            # Dynamic scale selection based on context
            # -------------------------------------------------
            pot_rew_small /= np.max(pot_rew_small) + 1e-6
            pot_rew_large /= np.max(pot_rew_large) + 1e-6

            grad_small = np.sum(np.abs(np.diff(pot_rew_small, append=pot_rew_small[0])))
            grad_large = np.sum(np.abs(np.diff(pot_rew_large, append=pot_rew_large[0])))

            if self.vis_density < 0.5 or distance_to_goal > 2.0:
                selected_scale = "Large"
                combined_pot_rew = pot_rew_large
            elif grad_small > grad_large * 1.1:
                selected_scale = "Small"
                combined_pot_rew = pot_rew_small
            else:
                alpha = grad_small / (grad_small + grad_large + 1e-6)
                selected_scale = "Blend"
                combined_pot_rew = alpha * pot_rew_small + (1 - alpha) * pot_rew_large

            # Print scale switch
            if selected_scale != self.current_scale:
                print(f"Switching scale: {self.current_scale} -> {selected_scale}")
                self.current_scale = selected_scale

            self.directional_reward_estimates = combined_pot_rew

        else:
            # Single-scale
            pot_rew = np.zeros(self.n_hd)
            for d in range(self.n_hd):
                pcn_future = self.pcn_small.preplay(d, num_steps=num_steps_preplay)
                self.rcn_small.update_reward_cell_activations(pcn_future)
                pot_rew[d] = tf.reduce_max(
                    np.nan_to_num(self.rcn_small.reward_cell_activations)
                )
            self.directional_reward_estimates = pot_rew

        # -------------------------------------------------
        #  3) DETERMINE ACTION ANGLE
        # -------------------------------------------------
        # Smooth the reward estimates
        self.directional_reward_estimates = gaussian_filter1d(self.directional_reward_estimates, sigma=1.0)

        # Validate directional reward estimates
        if self.directional_reward_estimates is None or np.isnan(self.directional_reward_estimates).any() or np.sum(self.directional_reward_estimates) <= 1e-6:
            print("Directional reward estimates invalid. Triggering exploration.")
            self.low_reward_state = True
            self.explore()
            return

        # Calculate the action angle
        action_angle = circmean(angles, weights=self.directional_reward_estimates)

        # Handle NaN in action_angle
        if np.isnan(action_angle):
            print("Circmean resulted in NaN. Triggering exploration.")
            self.explore()
            return

        # Determine the index of the chosen direction
        index = int(action_angle // (2 * np.pi / self.n_hd))

        # Validate the reward for the chosen direction
        max_reward = self.directional_reward_estimates[index]
        if max_reward <= 1e-3:
            print("Chosen direction reward too low. Triggering exploration.")
            self.explore()
            return

        # Ensure action_angle is within valid range
        if abs(action_angle) > np.pi:
            action_angle -= np.sign(action_angle) * 2 * np.pi

        # Calculate the angle to turn
        angle_to_turn = -np.deg2rad(
            np.rad2deg(action_angle) - self.current_heading_deg
        )
        # Normalize to [-π, π]
        angle_to_turn = ((angle_to_turn + np.pi) % (2 * np.pi)) - np.pi
        self.turn(angle_to_turn)

        # -------------------------------------------------
        #  MOVE FORWARD
        # -------------------------------------------------
        for s in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
            self.check_goal_reached()
            if any(self.collided):
                self.stop()
                self.explore()
            if self.done:
                return

            actual_reward = self.get_actual_reward()
            if self.enable_multiscale:
                self.rcn_small.update_reward_cell_activations(
                    self.pcn_small.place_cell_activations
                )
                self.rcn_small.td_update(
                    self.pcn_small.place_cell_activations, next_reward=actual_reward
                )

                self.rcn_large.update_reward_cell_activations(
                    self.pcn_large.place_cell_activations
                )
                self.rcn_large.td_update(
                    self.pcn_large.place_cell_activations, next_reward=actual_reward
                )
            else:
                self.rcn_small.update_reward_cell_activations(
                    self.pcn_small.place_cell_activations
                )
                self.rcn_small.td_update(
                    self.pcn_small.place_cell_activations, next_reward=actual_reward
                )

    ########################################### RECORDING ###########################################

    def record_sensor_data(self):
        """Records sensor data during robot operation.

        Stores:
            - Current position coordinates
            - Heading angle in degrees
            - LiDAR range readings

        The data is stored in the sensor_data dictionary with keys:
        'positions', 'headings', and 'lidar'.

        Returns:
            None
        """
        # Get current position
        curr_pos = self.robot.getField("translation").getSFVec3f()
        self.sensor_data["positions"].append([curr_pos[0], curr_pos[2]])

        # Get current heading angle in degrees
        current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))
        self.sensor_data["headings"].append(current_heading_deg)

        # Get LiDAR readings
        lidar_readings = self.range_finder.getRangeImage()
        self.sensor_data["lidar"].append(
            lidar_readings.copy()
        )  # Copy to avoid referencing issues

    def save_sensor_data(self):
        """
        Saves the recorded sensor data to files for later use.
        """
        # Convert lists to NumPy arrays
        positions = np.array(self.sensor_data["positions"])
        headings = np.array(self.sensor_data["headings"])
        lidar_data = np.array(self.sensor_data["lidar"])

        # Save data to files
        np.save("recorded_positions.npy", positions)
        np.save("recorded_headings.npy", headings)
        np.save("recorded_lidar.npy", lidar_data)

        print("Sensor data saved.")

    def recording(self):
        """
        Handles the logic for recording sensor data without using place fields.
        """
        # Sense the environment
        self.sense()

        # Record sensor data
        self.record_sensor_data()

        # Move the robot forward
        if not np.any(self.collided):
            self.forward()
        else:
            # If collided, turn by a random angle to avoid obstacle
            self.turn(np.random.uniform(-np.pi / 2, np.pi / 2))
            self.collided.assign([0, 0])  # Reset collision status

        # Introduce a 5% chance to rotate randomly
        if rng.uniform(0, 1) < 0.05:  # 5% probability
            self.turn(
                np.random.normal(0, np.deg2rad(30))
            )  # Random turn with normal distribution

        # Increment the step count
        self.step_count += 1

        # Check if the maximum number of steps has been reached
        if self.step_count >= self.num_steps:
            self.save()
            print("Data recording complete.")

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
        self.boundaries = self.range_finder.getRangeImage()

        # 2. Get the robot's current heading in degrees using the compass and convert it to an integer.
        # Shape: scalar (int)
        self.current_heading_deg = int(
            self.get_bearing_in_degrees(self.compass.getValues())
        )

        # 3. Roll the LiDAR data based on the current heading to align the 'front' with index 0.
        # Shape: (720,) - LiDAR data remains 720 points, but shifted according to the robot's current heading.
        self.boundaries = np.roll(self.boundaries, 2 * self.current_heading_deg)

        # 4. Convert the current heading from degrees to radians.
        # Shape: scalar (float) - Current heading of the robot in radians.
        current_heading_rad = np.deg2rad(self.current_heading_deg)

        # 5. Define the anchor direction (theta_0) as 0 radians for now, meaning no offset is applied.
        theta_0 = 0

        # 6. Calculate the current heading vector from the heading in radians.
        # Shape: (2,) - A 2D vector representing the robot's current heading direction: [cos(theta), sin(theta)].
        v_in = np.array([np.cos(current_heading_rad), np.sin(current_heading_rad)])

        # 7. Compute the activations of the head direction cells based on the current heading vector (v_in).
        # Shape: (self.num_cells,) - A 1D array where each element represents the activation of a head direction cell.
        self.hd_activations = tf.cast(
            self.head_direction_layer.get_hd_activation(theta_0=theta_0, v_in=v_in), tf.float32
        )

        # 8. Update the collision status using the left bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the left bumper, 0 otherwise.
        self.collided.scatter_nd_update([[0]], [int(self.left_bumper.getValue())])

        # 9. Update the collision status using the right bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the right bumper, 0 otherwise.
        self.collided.scatter_nd_update([[1]], [int(self.right_bumper.getValue())])

        # Log collisions
        if np.any(self.collided):
            if self.stats_collector:
                self.stats_collector.update_stat("collision_count", self.stats_collector.stats["collision_count"] + 1)

        # Get environmental complexity
        # self.vis_density = self.compute_visual_density(self.boundaries, self.current_heading_deg)
        self.vis_density = self.compute_visual_density(self.boundaries)

        # 10. Proceed to the next timestep in the robot's control loop.
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

    ########################################### COMPUTE PCN ACTIVATIONS ###########################################

    def compute_pcn_activations(self):
        if self.enable_multiscale:
            # Compute activations for the small PCN (unaffected by complexity)
            self.pcn_small.get_place_cell_activations(
                input_data=[self.boundaries, np.linspace(0, 2 * np.pi, 720, False)],
                hd_activations=self.hd_activations,
                vis_density=self.vis_density,  # No modulation
            )

            # Compute activations for the large PCN (affected by complexity)
            self.pcn_large.get_place_cell_activations(
                input_data=[self.boundaries, np.linspace(0, 2 * np.pi, 720, False)],
                hd_activations=self.hd_activations,
                vis_density=self.vis_density,  # Modulated by complexity
            )
        else:
            self.pcn_small.get_place_cell_activations(
                input_data=[self.boundaries, np.linspace(0, 2 * np.pi, 720, False)],
                hd_activations=self.hd_activations,
                vis_density=0.0,  # No modulation
            )
        
        self.step(self.timestep)

    ########################################### UPDATE HMAPS ###########################################

    def update_hmaps(self):
        curr_pos = self.robot.getField("translation").getSFVec3f()

        if self.step_count < self.num_steps:
            # Record position and always record the small PCN
            self.hmap_x[self.step_count] = curr_pos[0]
            self.hmap_y[self.step_count] = curr_pos[2]
            # Always record the small PCN
            self.hmap_z_small[self.step_count] = self.pcn_small.place_cell_activations.numpy()
            
            self.hmap_h[self.step_count] = self.hd_activations
            self.hmap_vis_density[self.step_count] = self.vis_density
            
            if self.enable_multiscale:
                # If multiscale, also record the large PCN
                self.hmap_z_large[self.step_count] = self.pcn_large.place_cell_activations.numpy()
                self.hmap_g[self.step_count] = tf.reduce_sum(self.pcn_large.bvc_activations)
            else:
                # Single-scale => only small PCN, so record e.g. the small BVC sum
                self.hmap_g[self.step_count] = tf.reduce_sum(self.pcn_small.bvc_activations)

        self.step_count += 1

    ########################################### CHECK GOAL REACHED ###########################################

    def check_goal_reached(self):
        """
        Check if the robot has reached the goal and perform necessary actions when the goal is reached.
        """
        curr_pos = self.robot.getField("translation").getSFVec3f()
        curr_pos_2d = [curr_pos[0], curr_pos[2]]  # Use only x and z coordinates
        time_limit = 5 # minutes

        # In LEARN_OJAS or PLOTTING mode, save only after run_time is reached:
        if (self.mode == RobotMode.LEARN_OJAS or self.mode == RobotMode.LEARN_HEBB or self.mode == RobotMode.PLOTTING) \
        and self.getTime() >= 60 * self.run_time_minutes:
            self.save(include_pcn=self.mode != RobotMode.PLOTTING, include_rcn=self.mode != RobotMode.PLOTTING)

        # In DMTP mode, save immediately after the goal is reached
        elif self.mode == RobotMode.DMTP and np.allclose(
            self.goal_location, curr_pos_2d, atol=self.goal_r["explore"]
        ):
            self.auto_pilot()
            
            if self.enable_multiscale:
                self.rcn_small.update_reward_cell_activations(
                    self.pcn_small.place_cell_activations,
                    visit=True,
                )
                self.rcn_large.update_reward_cell_activations(
                    self.pcn_large.place_cell_activations,
                    visit=True,
                )
                print("Replay called for RCN Small")
                self.rcn_small.replay(pcn=self.pcn_small)
                print(f"small scale RCN activations: {self.rcn_small.reward_cell_activations}")
                print("Replay called for RCN Large")
                self.rcn_large.replay(pcn=self.pcn_large)
                print(f"large scale RCN activations: {self.rcn_large.reward_cell_activations}")
            else:
                self.rcn_small.update_reward_cell_activations(
                    self.pcn_small.place_cell_activations,
                    visit=True
                )                
                self.rcn_small.replay(pcn=self.pcn_small)
            self.save(include_hmaps=False)

        elif self.mode == RobotMode.EXPLOIT and (
            self.getTime() >= time_limit * 60 or 
            np.allclose(self.goal_location, curr_pos_2d, atol=self.goal_r["exploit"])
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
                return

    ########################################### AUTO PILOT ###########################################

    def auto_pilot(self):
        print("Auto-piloting to the goal...")
        step = 0
        curr_pos = self.robot.getField("translation").getSFVec3f()
        while not np.allclose(
            self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["explore"]
        ):
            curr_pos = self.robot.getField("translation").getSFVec3f()
            delta_x = curr_pos[0] - self.goal_location[0]
            delta_y = curr_pos[2] - self.goal_location[1]

            if delta_x >= 0:
                theta = tf.math.atan(abs(delta_y), abs(delta_x))
                desired = np.pi * 2 - theta if delta_y >= 0 else np.pi + theta
            elif delta_y >= 0:
                theta = tf.math.atan(abs(delta_y), abs(delta_x))
                desired = np.pi / 2 - theta
            else:
                theta = tf.math.atan(abs(delta_x), abs(delta_y))
                desired = np.pi - theta
            self.turn(-(desired - np.deg2rad(self.current_heading_deg)))
            self.sense()
            self.compute_pcn_activations()
            self.update_hmaps()
            self.forward()
            step += 1

    ########################################### ENVIRONMENTAL COMPLEXITY ###########################################
    
    def compute_visual_density(self, lidar_data):
        """
        Computes the visual density based on distances to walls and proximity to the goal,
        applying an exponential decay to each LiDAR reading and combining it with a
        radial goal gradient.

        Args:
            lidar_data (array): LiDAR readings indicating distances to obstacles.

        Returns:
            float: The computed visual density, emphasizing proximity to walls and the goal.
        """
        # Threshold distance for influence (e.g., max effective wall influence)
        max_influence_radius = 1.5  # Adjust based on environment size

        # Apply exponential decay to each LiDAR reading for wall density
        wall_densities = np.exp(-lidar_data / max_influence_radius)
        wall_density = np.sum(wall_densities) / len(lidar_data)

        # Compute distance to the goal
        curr_pos = self.robot.getField("translation").getSFVec3f()
        distance_to_goal = np.linalg.norm(
            [curr_pos[0] - self.goal_location[0], curr_pos[2] - self.goal_location[1]]
        )

        # Exponential decay for goal proximity
        goal_density = np.exp(-distance_to_goal / max_influence_radius)

        # Combine wall density and goal density
        vis_density = wall_density + goal_density

        # Clamp between 0 and 1
        vis_density = np.clip(vis_density, 0, 1)

        return vis_density

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
        # Log the turn count
        if self.stats_collector:
            self.stats_collector.update_stat("turn_count", self.stats_collector.stats["turn_count"] + 1)

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
        for i in range(self.hmap_x.shape[0] - 1):
            current_position = np.array([self.hmap_y[i], self.hmap_x[i]])
            next_position = np.array([self.hmap_y[i + 1], self.hmap_x[i + 1]])
            path_length += np.linalg.norm(next_position - current_position)

        return path_length

    def save(
        self,
        include_pcn: bool = True,
        include_rcn: bool = True,
        include_hmaps: bool = True,
    ):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), 
        history maps, and optionally navigation statistics.

        Parameters:
            include_pcn (bool): If True, saves both the small and large Place Cell Networks (PCN).
            include_rcn (bool): If True, saves the Reward Cell Network (RCN).
            include_hmaps (bool): If True, saves the history of the agent's path and activations.
        """
        files_saved = []

        # Save PCNs
        if include_pcn:
            with open("pcn_small.pkl", "wb") as output:
                pickle.dump(self.pcn_small, output)
                files_saved.append("pcn_small.pkl")
            
            # Save the large PCN only if it exists (multiscale mode)
            if self.enable_multiscale and hasattr(self, "pcn_large"):
                with open("pcn_large.pkl", "wb") as output:
                    pickle.dump(self.pcn_large, output)
                    files_saved.append("pcn_large.pkl")

        # Save RCNs
        if include_rcn:
            if self.enable_multiscale:
                if hasattr(self, "rcn_small"):
                    with open("rcn_small.pkl", "wb") as output:
                        pickle.dump(self.rcn_small, output)
                        files_saved.append("rcn_small.pkl")
                if hasattr(self, "rcn_large"):
                    with open("rcn_large.pkl", "wb") as output:
                        pickle.dump(self.rcn_large, output)
                        files_saved.append("rcn_large.pkl")
            else:
                with open("rcn_small.pkl", "wb") as output:
                    pickle.dump(self.rcn_small, output)
                    files_saved.append("rcn_small.pkl")

        # Save history maps
        if include_hmaps:
            with open("hmap_x.pkl", "wb") as output:
                pickle.dump(self.hmap_x[: self.step_count], output)
                files_saved.append("hmap_x.pkl")

            with open("hmap_y.pkl", "wb") as output:
                pickle.dump(self.hmap_y[: self.step_count], output)
                files_saved.append("hmap_y.pkl")

            with open("hmap_z_small.pkl", "wb") as output:
                pickle.dump(self.hmap_z_small[: self.step_count], output)
                files_saved.append("hmap_z_small.pkl")

            if self.enable_multiscale:
                with open("hmap_z_large.pkl", "wb") as output:
                    pickle.dump(self.hmap_z_large[: self.step_count], output)
                    files_saved.append("hmap_z_large.pkl")

            with open("hmap_g.pkl", "wb") as output:
                pickle.dump(self.hmap_g[: self.step_count], output)
                files_saved.append("hmap_g.pkl")

            with open("hmap_h.pkl", "wb") as output:
                pickle.dump(self.hmap_h[: self.step_count], output)
                files_saved.append("hmap_h.pkl")

            with open("hmap_vis_density.pkl", "wb") as output:
                pickle.dump(self.hmap_vis_density[: self.step_count], output)
                files_saved.append("hmap_vis_density.pkl")

        # User confirmation dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes("-topmost", True)  # Keep the dialog on top
        root.update()
        messagebox.showinfo("Information", "Press OK to save data")
        root.destroy()

        # Pause the simulation
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

        print(f"Files Saved: {files_saved}")
        print("Saving Done!")

    def clear(self):
        files_to_remove = [
            "pcn_small.pkl",
            "pcn_large.pkl",
            "rcn_small.pkl",
            "rcn_large.pkl",
            "hmap_x.pkl",
            "hmap_y.pkl",
            "hmap_g.pkl",
            "hmap_h.pkl",
            "hmap_vis_density.pkl",
            "hmap_z_small.pkl",
            "hmap_z_large.pkl",
        ]

        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass  # Ignore if the file does not exist

        print("State files cleared.")
