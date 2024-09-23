import numpy as np
import tensorflow as tf
tf.random.set_seed(5)
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from astropy.stats import circmean, circvar
import pickle
import os
import time
import tkinter as tk
from tkinter import messagebox
from controller import Supervisor, Robot

from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from layers.head_direction_layer import HeadDirectionLayer
from layers.place_cell_layer import PlaceCellLayer
from layers.reward_cell_layer import RewardCellLayer

np.set_printoptions(precision=2)
PI = tf.constant(np.pi) 
rng = default_rng() # random number generator
cmap = get_cmap('plasma')

try:
    with open('hmap_x.pkl', 'rb') as f:
        hmap_x = pickle.load(f)
    with open('hmap_y.pkl', 'rb') as f:
        hmap_y = pickle.load(f)
    with open('hmap_z.pkl', 'rb') as f:
        hmap_z = np.asarray(pickle.load(f))
except:
    pass

class Driver(Supervisor):
    """
    The Driver class controls the robot, manages its sensory inputs, and coordinates the activation of neural network layers 
    (place cells and reward cells) to simulate navigation and learning in an environment.

    Attributes:
        max_speed (float): The maximum speed of the robot.
        left_speed (float): The current speed of the left wheel.
        right_speed (float): The current speed of the right wheel.
        timestep (int): The timestep for each simulation step.
        wheel_radius (float): The radius of the robot's wheels.
        axle_length (float): The distance between the robot's wheels.
        run_time (int): The total run time for the simulation in seconds.
        num_steps (int): The number of simulation steps based on run time and timestep.
        sensor_data_x (ndarray): Array to store x-coordinates of sensor data.
        sensor_data_y (ndarray): Array to store y-coordinates of sensor data.
        place_cell_activations (ndarray): Array to store activations of place cells over time.
        head_direction_activations (ndarray): Array to store head direction cell activations over time.
        goal_estimates (ndarray): Array to store estimates of the goal location over time.
        robot (object): Placeholder for the robot instance.
        keyboard (object): Placeholder for the keyboard instance.
        compass (object): Placeholder for the compass sensor.
        range_finder (object): Placeholder for the range finder sensor.
        left_bumper (object): Placeholder for the left bumper sensor.
        right_bumper (object): Placeholder for the right bumper sensor.
        display (object): Placeholder for the display instance.
        rotation_field (object): Placeholder for the rotation field of the robot.
        left_motor (object): Placeholder for the left motor of the robot.
        right_motor (object): Placeholder for the right motor of the robot.
        left_position_sensor (object): Placeholder for the left wheel position sensor.
        right_position_sensor (object): Placeholder for the right wheel position sensor.
        pcn (PlaceCellLayer): Instance of the place cell network.
        rcn (RewardCellLayer): Instance of the reward cell network.
        boundary_data (Tensor): Tensor to store boundary data from sensors.
        goal_location (list): The coordinates of the goal location.
        expected_reward (float): The expected reward at the current state.
        last_reward (float): The reward received in the previous step.
        context (int): Index of the current context in the environment.
        current_pcn_state (Tensor): Tensor representing the current state.
        prev_pcn_state (Tensor): Tensor representing the previous state.
    """

    def initialization(self, stage="learning", run_mode='explore', randomize_start_loc=True, run_time_hours=1):
        """
        Initializes the Driver class with specified parameters and sets up the robot's sensors and neural networks.

        Parameters:
            stage: 'explore'-->'dmpt'-->'exploit'
            run_mode: 'explore' or 'exploit'
            randomize_start_loc: Randomize agent spawn location
            run_time_hours (int): Total run time for the simulation in hours.
        """
        # Stage and RunMode
        self.stage = stage
        self.run_mode = run_mode

        self.STATE_MANUAL_CONTROL = 0
        self.STATE_EXPLORE = 1
        self.STATE_EXPLOIT = 2

        # Model parameters
        self.num_place_cells = 1000
        self.num_reward_cells = 10
        self.n_hd = 8
        self.timestep = 32*3 # WorldInfo.basicTimeStep = 32ms
        self.tau_w = 5 # time constant for the window function
        self.context = 0  # TODO: Get rid of this

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60
        self.step_count = 0
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.goal_r = {"explore": .1, "exploit": .2}

        # Sensor data storage
        self.hmap_x = np.zeros(self.num_steps) 
        self.hmap_y = np.zeros(self.num_steps)
        self.hmap_z = np.zeros((self.num_steps, self.num_place_cells)) # place cell activations
        self.hmap_h = np.zeros((self.num_steps, self.n_hd)) # head direction cell activations
        self.hmap_g = np.zeros(self.num_steps) # goal estimates
        
        # Initialize hardware components and sensors
        self.robot = self.getFromDef('agent')  # Placeholder for robot instance
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        self.compass = self.getDevice('compass')
        self.compass.enable(self.timestep)
        self.range_finder = self.getDevice('range-finder')
        self.range_finder.enable(self.timestep)
        self.left_bumper = self.getDevice('bumper_left')
        self.left_bumper.enable(self.timestep)
        self.right_bumper = self.getDevice('bumper_right')
        self.right_bumper.enable(self.timestep)
        self.collided = tf.Variable(np.zeros(2, np.int32))
        self.rotation_field = self.robot.getField('rotation')
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_position_sensor = self.getDevice('left wheel sensor')
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice('right wheel sensor')
        self.right_position_sensor.enable(self.timestep)

        # Initialize layers
        self.load_pcn(self.num_place_cells, self.n_hd)
        self.load_rcn(self.num_reward_cells, self.num_place_cells)
        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd)

        # Initialize landmarks
        self.landmarks = np.inf * np.ones((5, 1))

        # Initialize boundaries
        self.boundary_data = tf.Variable(tf.zeros((720, 1)))

        self.act = tf.zeros(self.n_hd)
        self.step(self.timestep)
        self.step_count += 1

        # Initialize goal and context
        self.goal_location = [-1, 1]
        self.expected_reward = 0
        self.last_reward = 0
        self.current_pcn_state = tf.zeros_like(self.pcn.place_cell_activations)
        self.prev_pcn_state = tf.zeros_like(self.pcn.place_cell_activations)

        if randomize_start_loc:
            INITIAL = [rng.uniform(-2.4, 2.4), 0.5, rng.uniform(-2.4, 2.4)]
            self.robot.getField('translation').setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        
        self.sense()
        self.compute()

    def load_pcn(self, num_place_cells, n_hd):
        """
        Loads the place cell network from a file if available, or initializes a new one.
        """
        try:
            with open('pcn.pkl', "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
        except:
            self.pcn = PlaceCellLayer(num_place_cells, 720, self.timestep, 10, n_hd)
            print("Initialized new Place Cell Network.")
        return self.pcn

    def load_rcn(self, num_reward_cells, num_place_cells):
        """
        Loads the reward cell network from a file if available, or initializes a new one.
        """
        try:
            with open('rcn.pkl', 'rb') as f:
                self.rcn = pickle.load(f)
                print("Loaded existing Reward Cell Network.")
        except:
            self.rcn = RewardCellLayer(num_reward_cells=num_reward_cells, input_dim=num_place_cells, num_replay=3)
            print("Initialized new Reward Cell Network.")
        return self.rcn
    
########################################### RUN LOOP ###########################################
    
    def run(self):
        """
        Runs the main control loop of the robot, managing its behavior based on the current state.
        """
        # Initialize the state based on run_mode
        if self.run_mode == 'manual_control':
            self.state = self.STATE_MANUAL_CONTROL
        elif self.run_mode == 'explore':
            self.state = self.STATE_EXPLORE
        elif self.run_mode == 'exploit':
            self.state = self.STATE_EXPLOIT
        else:
            raise ValueError(f"Unknown run mode: {self.run_mode}")

        print(f"Starting {self.stage} stage in {self.run_mode} mode.")
        print(f"Goal at {self.goal_location}")

        while True:
            # Process keyboard input and check if we need to switch to manual control
            key = self.keyboard.getKey()
            if key in (ord('W'), ord('D'), ord('A'), ord('S')):
                print("Switching to manual control")
                self.state = self.STATE_MANUAL_CONTROL

            # Handle the robot's state
            if self.state == self.STATE_MANUAL_CONTROL:
                self.manual_control()
            elif self.state == self.STATE_EXPLORE:
                self.explore()
            elif self.state == self.STATE_EXPLOIT:
                self.exploit()
            else:
                print("Unknown state. Exiting...")
                break 
            
########################################### EXPLORE ###########################################

    def explore(self):
        """
        Handles the logic for the 'explore' mode.
        """
        self.prev_pcn_state = self.current_pcn_state
        self.current_pcn_state *= 0

        for s in range(self.tau_w):
            self.sense()

            if np.any(self.collided):
                self.turn(np.deg2rad(60))
                break

            if self.stage == "dmtp":
                self.current_pcn_state += self.pcn.place_cell_activations
                self.check_goal_reached()
                
            self.compute()
            self.forward()
            self.check_goal_reached()

        if self.stage == "dmtp":
            self.current_pcn_state /= self.tau_w # NOTE: 'self.tau_w' is 's' in Ade's code. Not sure how that would have worked...
        
        self.turn(np.random.normal(0, np.deg2rad(30))) # Choose a new random direction
        
########################################### EXPLOIT ###########################################

    def exploit(self):
        """
        Executes the exploitation routine.
        """
        self.current_pcn_state *= 0  # Reset the current place cell state
        self.stop()
        self.sense()
        self.compute()

        if self.step_count > self.tau_w:
            potential_rewards, potential_energy = self.compute_rewards_and_energy(num_steps=1)

            # Update action based on computed rewards
            self.act += 1 * (potential_rewards - self.act)
            act = np.nan_to_num(circmean(np.linspace(0, np.pi * 2, self.n_hd, endpoint=False), weights=self.act))
            max_reward = potential_rewards[int(act // (2 * np.pi / self.n_hd))]

            if max_reward <= 1e-3:
                print(f"Max reward: {max_reward}")
                print(f"potential_rewards: {potential_rewards}")
                print("Returning to exploration")
                self.state = self.STATE_EXPLORE
                return

            # Collision handling
            if np.any(self.collided):
                self.turn(np.deg2rad(60))
                self.stop()
                self.rcn.td_update(self.pcn.place_cell_activations, potential_energy[int(act // (2 * np.pi / self.n_hd))], max_reward, self.context)
                return
            else:
                self.adjust_turn_based_on_action(act)
            
            # Move forward
            for step in range(self.tau_w):
                self.sense()
                self.compute()
                self.forward()
                self.current_pcn_state += self.pcn.place_cell_activations
                self.check_goal_reached()

            # Normalize the accumulated place cell state over the window
            self.current_pcn_state /= self.tau_w
            self.expected_reward = max_reward / potential_energy[int(act // (2 * np.pi / self.n_hd))]
            self.last_reward = self.rcn.reward_cell_activations[self.context]

    def compute_rewards_and_energy(self, num_steps):
        potential_rewards = np.empty(self.n_hd)
        potential_energy = np.empty(self.n_hd)

        for direction in range(self.n_hd):
            pcn_activations = self.pcn.exploit(direction, num_steps=num_steps)
            self.rcn.update_reward_cell_activations(pcn_activations)

            potential_energy[direction] = tf.norm(pcn_activations, 1)
            potential_rewards[direction] = np.nan_to_num(self.rcn.reward_cell_activations[self.context])

        return potential_rewards, potential_energy

    def adjust_turn_based_on_action(self, act):
        """
        Adjusts the robot's turn based on the computed action.
        """
        if abs(act) > np.pi:
            act = act - np.sign(act) * 2 * np.pi
        self.turn(-np.deg2rad(np.rad2deg(act) - self.current_heading_deg) % (np.pi * 2))
        print(np.rad2deg(act), self.current_heading_deg, np.rad2deg(act) - self.current_heading_deg)

########################################### SENSE ###########################################

    def sense(self):
        """
        The 'sense' method updates the robot's perception of its environment, including its orientation, 
        distance to obstacles (boundaries), head direction cell activations, and collision detection.

        Steps:
        1. Capture the LiDAR (range finder) data, which provides distances to obstacles in all directions.
        2. Get the robot's current heading using the compass, convert it to radians, and adjust the LiDAR data 
        using np.roll() to align it with the robot's heading.
        3. Compute the current head direction vector and update the activations of the head direction cells.
        4. Update the collision status by checking the bumper sensors.
        5. Proceed to the next timestep in the robot's control loop.
        """

        # 1. Capture distance data from the range finder (LiDAR), which provides 720 points around the robot.
        # Shape: (720,)
        self.boundaries = self.range_finder.getRangeImage()

        # 2. Get the robot's current heading in degrees using the compass and convert it to an integer.
        # Shape: scalar (int)
        self.current_heading_deg = int(self.get_bearing_in_degrees(self.compass.getValues()))

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
        self.hd_activations = self.head_direction_layer.get_hd_activation(theta_0=theta_0, v_in=v_in)

        # 8. Update the collision status using the left bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the left bumper, 0 otherwise.
        self.collided.scatter_nd_update([[0]], [int(self.left_bumper.getValue())])

        # 9. Update the collision status using the right bumper sensor.
        # Shape: scalar (int) - 1 if collision detected on the right bumper, 0 otherwise.
        self.collided.scatter_nd_update([[1]], [int(self.right_bumper.getValue())])

        # 10. Proceed to the next timestep in the robot's control loop.
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north):
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
        # Compute the place cell network activations
        self.pcn.get_place_cell_activations(input_data=[self.boundaries, np.linspace(0, 2 * np.pi, 720, False)], 
                hd_activations=self.hd_activations, mode=self.stage, collided=np.any(self.collided))
        
        # Advance the timestep and update position
        self.step(self.timestep)
        curr_pos = self.robot.getField('translation').getSFVec3f()

        # Update place cell and sensor maps
        if self.step_count < self.num_steps:
            self.hmap_x[self.step_count] = curr_pos[0]
            self.hmap_y[self.step_count] = curr_pos[2]
            self.hmap_z[self.step_count] = self.pcn.place_cell_activations
            self.hmap_h[self.step_count] = self.hd_activations
            self.hmap_g[self.step_count] = tf.reduce_sum(self.pcn.bvc_activations)

        # Increment timestep
        self.step_count += 1

########################################### CHECK GOAL REACHED ###########################################

    def check_goal_reached(self):
        """
        Check if the robot has reached the goal and perform necessary actions when the goal is reached.
        """
        curr_pos = self.robot.getField('translation').getSFVec3f()
        if (self.stage == "dmtp" and np.allclose(self.goal_location, [curr_pos[0], curr_pos[2]], 0, self.goal_r["exploit"])):
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Started at: {np.array([self.hmap_x[0], self.hmap_y[0]])}")
            print(f"Current position: {np.array([curr_pos[0], curr_pos[2]])}")
            distance_to_goal = np.linalg.norm(np.array([self.hmap_x[0], self.hmap_y[0]]) - self.goal_location) - self.goal_r["exploit"]
            print(f"Distance to goal: {distance_to_goal}")
            print(f"Time taken: {self.getTime()}")
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes("-topmost", True)  # Always keep the window on top
            root.update()
            self.save(include_hmaps=False)
            messagebox.showinfo("Information", "Goal reached - Press OK save the data")
            root.destroy()  # Destroy the main window
            self.worldReload()  
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE) 
        elif ((self.stage == "learning" or self.stage == "explore") and (self.getTime() >= 60 * self.run_time_minutes)):
            if self.stage == "learning":
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes("-topmost", True)  # Always keep the window on top
                root.update()
                self.save(include_hmaps=True)
                messagebox.showinfo("Information", "Press OK to save the data")
                root.destroy()  # Destroy the main window
                self.worldReload()
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)            

########################################### HELPER METHODS ###########################################

    def manual_control(self):
        """
        Allows for manual control of the robot using keyboard inputs.
        """
        k = self.keyboard.getKey()
        if k != -1:
            print("Before:", self.hd_activations.argmax(), self.current_heading)
        if k == ord('W'):
            self.forward()
        elif k == ord('D'):
            self.turn(-np.deg2rad(90))
        elif k == ord('A'):
            self.turn(np.deg2rad(90))
        elif k == ord('S'):
            self.stop()
        if k != -1:
            print("After:", self.hd_activations.argmax(), self.current_heading)

    def forward(self):
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()
    
    def turn(self, angle, circle=False):
        self.stop()
        self.move()
        l_offset = self.left_position_sensor.getValue()
        r_offset = self.right_position_sensor.getValue()
        self.sense()
        neg = -1.0 if (angle < 0.0) else 1.0
        if circle:
            self.left_motor.setVelocity(0)
        else:
            self.left_motor.setVelocity(neg * self.max_speed/2)
        self.right_motor.setVelocity(-neg * self.max_speed/2)
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
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
    
    def move(self):
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
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
            next_position = np.array([self.hmap_y[i+1], self.hmap_x[i+1]])
            path_length += np.linalg.norm(next_position - current_position)
        
        return path_length

    def save(self, include_hmaps=True):
        """
        Saves the state of the PCN (Place Cell Network), RCN (Reward Cell Network), and optionally
        the maps that store the agent's movement and activations.

        Parameters:
            include_maps (bool): If True, saves the history of the agent's path and activations.
        """
        # Save the Place Cell Network (PCN)
        with open('pcn.pkl', 'wb') as output:
            pickle.dump(self.pcn, output)

        # Save the Reward Cell Network (RCN)
        with open('rcn.pkl', 'wb') as output:
            pickle.dump(self.rcn, output)

        # Save the history maps if specified
        if include_hmaps:
            with open('hmap_x.pkl', 'wb') as output:
                pickle.dump(self.hmap_x[:self.step_count], output)
            with open('hmap_y.pkl', 'wb') as output:
                pickle.dump(self.hmap_y[:self.step_count], output)
            with open('hmap_z.pkl', 'wb') as output:
                pickle.dump(self.hmap_z[:self.step_count], output)
            with open('hmap_g.pkl', 'wb') as output:
                pickle.dump(self.hmap_g[:self.step_count], output)
            with open('hmap_h.pkl', 'wb') as output:
                pickle.dump(self.hmap_h[:self.step_count], output)
        
        print("Saved!")

    def clear(self):
        """
        Clears the saved state files for the Place Cell Network (PCN), Reward Cell Network (RCN),
        and the history maps by removing their corresponding pickle files.
        """
        files_to_remove = ['pcn.pkl', 'rcn.pkl', 'hmap_x.pkl', 'hmap_y.pkl', 'hmap_z.pkl', 'hmap_g.pkl']
        
        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass  # Ignore if the file does not exist

        print("State files cleared.")