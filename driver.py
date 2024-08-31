# %%
import numpy as np
import tensorflow as tf
tf.random.set_seed(5)
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from astropy.stats import circmean, circvar
import _pickle as pickle
import os

from networks import *

# %%
np.set_printoptions(precision=2)
num_pc = 1000 # number of PC
input_dim = 720 # BVC input size (720 bc RPLidar spits out a 720-point array)
timestep = 32 * 3
max_dist = 12 # max distance of LiDAR
tau_w = 10 # time constant for the window function
PI = tf.constant(np.pi) 
rng = default_rng() # random number generator
cmap = get_cmap('plasma')
goal_r = {"explore":.1, "exploit":.6}

# %%
try:
    with open('hmap_x.pkl', 'rb') as f:
        hmap_x = pickle.load(f)
    with open('hmap_y.pkl', 'rb') as f:
        hmap_y = pickle.load(f)
    with open('hmap_z.pkl', 'rb') as f:
        hmap_z = np.asarray(pickle.load(f))
except:
    pass

# %%
class Driver:
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
        current_step (int): The current step in the simulation.
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
        s (Tensor): Tensor representing the current state.
        s_prev (Tensor): Tensor representing the previous state.
    """

    def __init__(self, num_place_cells=1000, num_reward_cells=10, num_head_directions=8, run_time_hours=2, timestep=96):
        """
        Initializes the Driver class with specified parameters and sets up the robot's sensors and neural networks.

        Parameters:
            num_place_cells (int): Number of place cells in the place cell network.
            num_reward_cells (int): Number of reward cells in the reward cell network.
            num_head_directions (int): Number of head direction cells.
            run_time_hours (int): Total run time for the simulation in hours.
            timestep (int): The time step duration for each simulation step.
        """
        # Robot parameters
        self.max_speed = 4
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.timestep = timestep
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.run_time_minutes = run_time_hours * 60
        self.num_steps = int(self.run_time * 60 // (2 * self.timestep / 1000))

        # Sensor data storage
        self.sensor_data_x = np.zeros(self.num_steps)
        self.sensor_data_y = np.zeros(self.num_steps)
        self.place_cell_activations = np.zeros((self.num_steps, num_place_cells))
        self.head_direction_activations = np.zeros((self.num_steps, num_head_directions))
        self.goal_estimates = np.zeros(self.num_steps)

        # Initialize timestep
        self.current_step = 0

        # Initialize hardware components and sensors. Some of these won't be used.
        self.robot = None  # Placeholder for robot instance
        self.keyboard = None
        self.compass = None
        self.range_finder = None
        self.left_bumper = None
        self.right_bumper = None
        self.display = None
        self.rotation_field = None
        self.left_motor = None
        self.right_motor = None
        self.left_position_sensor = None
        self.right_position_sensor = None

        # Initialize neural network layers
        self.pcn = self.load_pcn(num_place_cells, num_head_directions)
        self.rcn = self.load_rcn(num_reward_cells, num_place_cells)

        # Initialize boundaries
        self.boundary_data = tf.Variable(tf.zeros((720, 1)))

        # Initialize goal and context
        self.goal_location = None
        self.expected_reward = 0
        self.last_reward = 0
        self.context = None
        self.s = tf.zeros_like(self.pcn.place_cell_activations)
        self.s_prev = tf.zeros_like(self.pcn.place_cell_activations)

    def load_pcn(self, num_place_cells, num_head_directions):
        """
        Loads the place cell network from a file if available, or initializes a new one.

        Parameters:
            num_place_cells (int): Number of place cells in the place cell network.
            num_head_directions (int): Number of head direction cells.

        Returns:
            PlaceCellLayer: The loaded or newly initialized place cell network.
        """
        try:
            with open('place_cell_network.pkl', "rb") as f:
                pcn = pickle.load(f)
                pcn.reset_activations()
                print("Loaded existing Place Cell Network.")
        except:
            pcn = PlaceCellLayer(num_place_cells, 720, self.timestep, 12, num_head_directions)
            print("Initialized new Place Cell Network.")
        return pcn

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
                rcn = pickle.load(f)
                print("Loaded existing Reward Cell Network.")
        except:
            rcn = RewardCellLayer(num_reward_cells, num_place_cells, 3)
            print("Initialized new Reward Cell Network.")
        return rcn

    def startup(self, context, mode, randomize=False):
        """
        Starts up the driver by initializing hardware, resetting the robot position, and setting the goal location.

        Parameters:
            context (int): The context or scenario index in the environment.
            mode (str): The mode of operation (e.g., "learn", "explore").
            randomize (bool): Whether to randomize the robot's starting position.
        """
        self.context = context
        self.mode = mode
        self.initialize_hardware()
        self.reset_robot_position(randomize)
        self.set_goal_location(context)
        self.sense()
        self.compute()



