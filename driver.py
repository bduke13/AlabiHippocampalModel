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
import warnings
from controller import Supervisor, Robot

from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from layers.head_direction_layer import HeadDirectionLayer
from layers.place_cell_layer import PlaceCellLayer
from layers.reward_cell_layer import RewardCellLayer

np.set_printoptions(precision=2)
num_pc = 1000 # number of PC
input_dim = 720 # BVC input size (720 bc RPLidar spits out a 720-point array)
timestep = 32 * 3
max_dist = 12 # max distance of LiDAR
tau_w = 10 # time constant for the window function
PI = tf.constant(np.pi) 
rng = default_rng() # random number generator
cmap = get_cmap('plasma')
goal_r = {"explore": .1, "exploit": .6}

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

    def initialization(self, mode="learning", randomize_start_loc=True, run_time_hours=2):
        """
        Initializes the Driver class with specified parameters and sets up the robot's sensors and neural networks.

        Parameters:
            mode: 'explore'-->'dmpt'-->'exploit'
            randomize_start_loc: Randomize agent spawn location
            run_time_hours (int): Total run time for the simulation in hours.
        """
        # Mode
        self.mode = mode

        # Model parameters
        self.num_place_cells = 1000
        self.num_reward_cells = 10
        self.n_hd = 8
        self.timestep = 32*3
        self.context = 0  # TODO: Get rid of this

        # Robot parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.timestep = timestep
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time_minutes * 60 // (2 * self.timestep / 1000))
        self.ts = 0

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
        self.display = self.getDevice('display')
        self.rotation_field = self.robot.getField('rotation')
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_position_sensor = self.getDevice('left wheel sensor')
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor = self.getDevice('right wheel sensor')
        self.right_position_sensor.enable(self.timestep)

        # Initialize PCN and RCN layers
        self.place_cell_network = self.load_pcn(self.num_place_cells, self.n_hd)
        self.reward_cell_network = self.load_rcn(self.num_reward_cells, self.num_place_cells)

        # Initialize landmarks
        self.landmarks = np.inf * np.ones((5, 1))

        # Initialize boundaries
        self.boundary_data = tf.Variable(tf.zeros((720, 1)))

        self.act = tf.zeros(self.n_hd)
        self.step(self.timestep)

        # Initialize goal and context
        self.goal_location = [-3, 3]
        self.expected_reward = 0
        self.last_reward = 0
        self.current_pcn_state = tf.zeros_like(self.pcn.place_cell_activations)
        self.prev_pcn_state = tf.zeros_like(self.pcn.place_cell_activations)

        if randomize_start_loc:
            INITIAL = [rng.uniform(-5, 5), 0.5, rng.uniform(-5, 5)]
            self.robot.getField('translation').setSFVec3f(INITIAL)
            self.robot.resetPhysics()
        
        self.sense()
        self.compute()

    def load_pcn(self, num_place_cells, n_hd):
        """
        Loads the place cell network from a file if available, or initializes a new one.

        Parameters:
            num_place_cells (int): Number of place cells in the place cell network.
            n_hd (int): Number of head direction cells.

        Returns:
            PlaceCellLayer: The loaded or newly initialized place cell network.
        """
        try:
            with open('pcn.pkl', "rb") as f:
                self.pcn = pickle.load(f)
                self.pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
        except:
            self.pcn = PlaceCellLayer(num_place_cells, 720, self.timestep, 12, n_hd)
            print("Initialized new Place Cell Network.")
        return self.pcn

    def load_rcn(self, num_reward_cells, num_place_cells):
        """
        Loads the reward cell network from a file if available, or initializes a new one.

        Parameters:
            num_reward_cells (int): Number of reward cells in the reward cell network.
            num_place_cells (int): Number of place cells in the place cell network.

        Returns:
            RewardCellLayer: The loaded or newly initialized reward cell network.
        """
        try:
            with open('reward_cell_network.pkl', 'rb') as f:
                self.rcn = pickle.load(f)
                print("Loaded existing Reward Cell Network.")
        except:
            self.rcn = RewardCellLayer(num_reward_cells=num_reward_cells, input_dim=num_place_cells, num_replay=3)
            print("Initialized new Reward Cell Network.")
        return self.rcn
    
    def run(self, run_mode):
        """
        Runs the main control loop of the robot. It will handle different modes like "dmtp", "explore", and "exploit".
        """
        while True:
            if run_mode == "exploit":
                self.exploit()
            else:
                self.explore()
            # Switch to manual control if a key is pressed.
            if self.keyboard.getKey() in (ord('W'), ord('D'), ord('A'), ord('S')):
                print("Switching to manual control")
                while True:
                    self.manual_control()

    def explore(self):
        """
        Handles the logic for the 'explore' mode.
        """
        self.prev_pcn_state = self.current_pcn_state
        self.current_pcn_state *= 0

        for s in range(tau_w):
            self.sense()

            if np.any(self.collided):
                self.turn(np.deg2rad(60))
                break

            if self.mode == "dmtp":
                self.current_pcn_state += self.place_cell_network.place_cell_activations
                self.check_goal_reached()
                
            self.compute()
            self.forward()
            self.check_goal_reached()

        if self.mode == "dmtp":
            self.current_pcn_state /= tau_w # NOTE: 'tau_w' is 's' in Ade's code. Not sure how that would have worked...
        
        self.turn(np.random.normal(0, np.deg2rad(30)))

    def exploit(self):
        """
        Executes an exploitation routine where the agent uses its internal models (PCN and RCN) to
        navigate towards a goal and update its internal reward and place cell activations. This method
        is called continuously in run() and reloads the sim when the goal is found.
        """
        self.current_pcn_state *= 0  # Reset the current place cell state
        self.stop()  
        self.sense()
        self.compute()  # Compute activations in the PCN and RCN
        self.check_goal_reached()  # Check if the robot is at the goal

        if self.ts > tau_w:
            act, max_reward, num_steps = 0, 0, 1
            potential_rewards = np.empty(self.n_hd)  # Array to store potential rewards
            potential_energy = np.empty(self.n_hd)  # Array to store potential energy

            # Compute the reward cell activations based on the current PCN state
            self.reward_cell_network.update_reward_cell_activations(self.place_cell_network.place_cell_activations, visit=True, context=self.context)
            # print("Reward:", self.reward_cell_network.reward_cell_activations[self.context], 
            #     "Most Active:", self.place_cell_network.place_cell_activations.numpy().argsort()[-3:])

            # If the reward is negligible, return to exploration
            if self.reward_cell_network.reward_cell_activations[self.context] <= 1e-6:
                self.explore()
                return

            # Iterate over all possible head directions and compute their potential rewards and energy
            for direction in range(self.n_hd):
                # Exploit the current place cell state for each direction
                pcn_activations = self.place_cell_network.exploit(direction, num_steps=num_steps)
                self.reward_cell_network.update_reward_cell_activations(pcn_activations)
                
                potential_energy[direction] = tf.norm(pcn_activations, 1)
                potential_rewards[direction] = np.nan_to_num(self.reward_cell_network.reward_cell_activations[self.context])

            # Update action based on computed rewards
            self.act += 1 * (potential_rewards - self.act)  # Update internal action state
            act = np.nan_to_num(circmean(np.linspace(0, np.pi * 2, self.n_hd, endpoint=False), weights=self.act))
            # var = np.nan_to_num(circvar(np.linspace(0, np.pi * 2, self.n_hd, endpoint=False), weights=self.act)) # NOTE: Not used
            max_reward = potential_rewards[int(act // (2 * np.pi / self.n_hd))] # NOTE: May get a ValueError here since there's a chance of division by 0. I think you can ignore it.

            # If the maximum reward is negligible or variance is high, return to exploration
            if max_reward <= 1e-3:
                print("Returning to exploration")
                self.explore()
                return

            # # Plot the current action space (polar plot)
            # fig = plt.figure(2)
            # fig.clf()
            # ax = fig.add_subplot(projection='polar')
            # ax.set_theta_zero_location("N")
            # ax.set_theta_direction(-1)
            # ax.plot(np.linspace(0, np.pi * 2, self.n_hd, endpoint=False), self.act)
            # title = f"{np.rad2deg(act)}, {np.rad2deg(var)}, {tf.reduce_max(self.act).numpy()}"
            # plt.title(title)
            # plt.pause(.01)

            # If a collision is detected, update the reward cell network and turn the robot
            if np.any(self.collided):
                self.turn(np.deg2rad(60))
                self.stop()
                self.reward_cell_network.td_update(self.place_cell_network.place_cell_activations, potential_energy[int(act // (2 * np.pi / self.n_hd))], max_reward, self.context)
                return
            else:
                # Adjust the turning angle based on the action and head direction
                if abs(act) > np.pi:
                    act = act - np.sign(act) * 2 * np.pi
                self.turn(-np.deg2rad(np.rad2deg(act) - self.n_index) % (np.pi * 2))
                print(np.rad2deg(act), self.n_index, np.rad2deg(act) - self.n_index)

            # Move forward and continue sensing and computing activations over the time window
            for step in range(tau_w):
                self.sense()
                self.compute()
                self.forward()
                self.current_pcn_state += self.place_cell_network.place_cell_activations
                self.check_goal_reached(False, step)

            # Normalize the accumulated place cell state over the window
            self.current_pcn_state /= tau_w

            # Update internal expected reward for the next step
            self.expected_reward = max_reward / potential_energy[int(act // (2 * np.pi / self.n_hd))]
            self.last_reward = self.reward_cell_network.reward_cell_activations[self.context]

    def compute(self):
        """
        Compute the activations of place cells and handle the environment interactions.
        """
        # Compute the place cell network activations
        self.pcn.get_place_cell_activations([self.boundaries, np.linspace(0, 2 * np.pi, 720, False)], 
                self.head_direction_vector, self.mode, np.any(self.collided))
        
        # Advance the timestep and update position
        self.step(self.timestep)
        curr_pos = self.robot.getField('translation').getSFVec3f()

        # Update place cell and sensor maps if within timestep bounds
        if self.ts < self.hmap_x.size:
            self.hmap_x[self.ts] = curr_pos[0]
            self.hmap_y[self.ts] = curr_pos[2]
            self.hmap_z[self.ts] = self.pcn.place_cell_activations
            self.hmap_h[self.ts] = self.head_direction_vector
            self.hmap_g[self.ts] = tf.reduce_sum(self.pcn.bvc_activations)

        # Increment timestep
        self.ts += 1

    def check_goal_reached(self):
        """
        Check if the robot has reached the goal and perform necessary actions when the goal is reached.
        """
        curr_pos = self.robot.getField('translation').getSFVec3f()

        if self.getTime() >= 60*self.run_time_minutes:
            return
        if (self.mode=="dmtp" and np.allclose(self.goalLocation, [curr_pos[0], curr_pos[2]], 0, goal_r["exploit"])) or ((self.mode=="cleanup" or self.mode=="learning") and (self.getTime() >=60*self.run_time_minutes)):
            print("Goal reached")
            print(f"Total distance traveled: {self.compute_path_length()}")
            print(f"Started at: {np.array([self.hmap_x[0], self.hmap_y[0]])}")
            print(f"Current position: {np.array([curr_pos[0], curr_pos[2]])}")
            distance_to_goal = np.linalg.norm(np.array([self.hmap_x[0], self.hmap_y[0]]) - self.goal_location) - goal_r["exploit"]
            print(f"Distance to goal: {distance_to_goal}")
            print(f"Time taken: {self.getTime()}")

            # Get a unique timestamp for file naming
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Generate image for place cell network activations
            place_cell_img_path = os.path.join("images", f"place_cell_activations_{timestamp}.png")
            plt.figure()
            plt.imshow(tf.reduce_max(self.pcn.w_rec_hd_place, 0))
            plt.title("Place Cell Network Activations")
            plt.savefig(place_cell_img_path)
            plt.close()

            # Generate image for reward cell network activations
            reward_cell_img_path = os.path.join("images", f"reward_cell_activations_{timestamp}.png")
            plt.figure()
            plt.stem(self.rcn.w_in[self.context].numpy())
            goal_angle = tf.math.atan2(curr_pos[2] - self.goal_location[1], curr_pos[0] - self.goal_location[0]).numpy()
            plt.title(f"Reward Cell Network | Goal Angle: {goal_angle}")
            plt.savefig(reward_cell_img_path)
            plt.close()

            # Save current state and reload the world
            self.save(True)
            self.worldReload()

    def manual_control(self):
        """
        Allows for manual control of the robot using keyboard inputs.
        """
        k = self.keyboard.getKey()
        if k != -1:
            print("Before:", self.head_direction_vector.argmax(), self.n_index)
        if k == ord('W'):
            self.forward()
        elif k == ord('D'):
            self.turn(-np.deg2rad(90))
        elif k == ord('A'):
            self.turn(np.deg2rad(90))
        elif k == ord('S'):
            self.stop()
        if k != -1:
            print("After:", self.head_direction_vector.argmax(), self.n_index)

    def forward(self):
        self.leftSpeed = self.max_speed
        self.rightSpeed = self.max_speed
        self.move()
        self.sense()
    
    def turn(self, angle, circle=False):
        self.stop()
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
        self.left_motor.setVelocity(self.leftSpeed)
        self.right_motor.setVelocity(self.rightSpeed)

    def sense(self):
        self.boundaries = self.range_finder.getRangeImage()
        self.n_index = int(self.get_bearing_in_degrees(self.compass.getValues()))
        self.boundaries = np.roll(self.boundaries, 2*self.n_index)
        rad = np.deg2rad(self.n_index)
        v = np.array([np.cos(rad), np.sin(rad)])
        self.head_direction_vector = self.head_direction(0, v)
        self.collided.scatter_nd_update([[0]], [int(self.left_bumper.getValue())])
        self.collided.scatter_nd_update([[1]], [int(self.right_bumper.getValue())])
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north):
        rad = np.arctan2(north[0], north[2])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0
        return bearing
    
    def head_direction(self, theta_0, v_in=[1, 1]):
        """
        Computes the head direction activation based on the input vector `v_in` and 
        the angular difference between the current direction `theta_0` and preferred 
        directions of the head direction cells.
        
        Parameters:
        - theta_0: The current heading direction (in radians).
        - v_in: Input vector representing the current direction, default is [1, 1].

        Returns:
        - np.ndarray: The activation values of the head direction cells.
        """
        # Calculate the preferred directions (tuning kernel)
        theta_i = np.arange(0, 2 * np.pi, np.deg2rad(360 // self.n_hd))
        D = np.empty(2, dtype=np.ndarray)
        D[0] = np.cos(np.add(theta_i, theta_0))
        D[1] = np.sin(np.add(theta_i, theta_0))

        # Compute the dot product between the input vector `v_in` and the tuning kernel
        activation = np.dot(v_in, D)

        # Return the activation values
        return activation

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
                pickle.dump(self.hmap_x[:self.ts], output)
            with open('hmap_y.pkl', 'wb') as output:
                pickle.dump(self.hmap_y[:self.ts], output)
            with open('hmap_z.pkl', 'wb') as output:
                pickle.dump(self.hmap_z[:self.ts], output)
            with open('hmap_g.pkl', 'wb') as output:
                pickle.dump(self.hmap_g[:self.ts], output)
            with open('hmap_h.pkl', 'wb') as output:
                pickle.dump(self.hmap_h[:self.ts], output)

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