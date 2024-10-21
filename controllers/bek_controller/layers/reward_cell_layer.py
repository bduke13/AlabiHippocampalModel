import numpy as np
import tensorflow as tf
tf.random.set_seed(5)
from copy import deepcopy

class RewardCellLayer:
    def __init__(self, num_reward_cells=10, input_dim=1000, num_replay=3):
        '''
        Initializes the Reward Cell Layer.

        Parameters:
        num_reward_cells: Number of reward cells in the layer.
        input_dim: Dimension of the input vector to the layer (== num_pc).
        num_replay: Number of replay iterations for the reward learning process.
        '''
        # Number of reward cells
        self.num_reward_cells = num_reward_cells
        
        # Activation values for reward cells
        # Shape: (num_reward_cells, 1)
        self.reward_cell_activations = tf.zeros((num_reward_cells, 1), dtype=tf.float32)
        
        # Number of replay iterations
        self.num_replay = num_replay

        # Initialize weights with small random values
        self.w_in = tf.Variable(tf.zeros((num_reward_cells, input_dim)), dtype=tf.float32)
        
        # Effective input weight matrix, used for updating during visits
        self.w_in_effective = tf.Variable(tf.zeros((num_reward_cells, input_dim)), dtype=tf.float32)

    def update_reward_cell_activations(self, input_data, visit=False, context=0):
        '''
        Computes the activations of reward cells based on input data.

        Parameters:
        input_data: The input data for the reward cells.
        visit: Boolean flag indicating if the cell is being visited.
        context: Context index for selecting specific input weights.
        '''
        # Calculate the norm of the input data
        input_norm = tf.linalg.norm(input_data, ord=1)
        # Add a small epsilon to prevent division by zero
        safe_denominator = tf.maximum(input_norm, 1e-6)
        # Calculate reward cell activations using effective weights
        self.reward_cell_activations = tf.tensordot(self.w_in_effective, input_data, axes=1) / safe_denominator

        if visit:
            # Update weights directly based on input data
            learning_rate = 0.1  # Increase the learning rate for significant updates
            updated_weights = self.w_in_effective[context] + learning_rate * input_data
            self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[context]], [updated_weights])

    def new_reward(self, pc_net, context=0):
        '''
        Replays the place cell activations and updates the reward cell weights.

        Parameters:
        pc_net: The place cell network.
        context: Context index for selecting specific input weights.
        '''
        # Create a copy of the place cell network
        pc_net_copy = deepcopy(pc_net)
        dw = tf.zeros_like(self.w_in)

        for t in range(10):
            # Update dw
            exp_factor = tf.math.exp(-t / 6)
            pc_net_v_norm = tf.linalg.normalize(pc_net_copy.place_cell_activations, ord=np.inf)[0]
            dw = tf.tensor_scatter_nd_add(dw, [[context]], [exp_factor * pc_net_v_norm])

            # Update pc_net_copy.place_cell_activations
            v = tf.identity(pc_net_copy.place_cell_activations)
            w_rec_max = tf.reduce_max(pc_net_copy.w_rec_hd_place, axis=0)
            v = tf.nn.relu(tf.tensordot(tf.cast(w_rec_max, tf.float32), pc_net.place_cell_activations, axes=1) + v)
            pc_net_copy.place_cell_activations = tf.tanh(v)

        # Update weights
        dw_norm = tf.linalg.normalize(dw, ord=np.inf)[0]
        self.w_in.assign_add(dw_norm)
        self.w_in_effective = tf.identity(self.w_in)

    def td_update(self, input_data, next_reward, context=0):
        '''
        Temporal difference (TD) update method for reward learning.

        Parameters:
        input_data: Input to the reward cell layer.
        next_reward: Reward at the next time step.
        context: Context index for selecting specific input weights.
        '''
        print("Next reward:", next_reward)
        print("Before update:", tf.tensordot(self.w_in_effective[context], input_data, axes=1))

        # Calculate the prediction error (delta)
        prediction = tf.tensordot(self.w_in_effective[context], input_data, axes=1)
        delta = next_reward - prediction

        # Update weights based on the TD learning rule
        learning_rate = 0.01  # Adjust the learning rate as needed
        updated_weights = self.w_in_effective[context] + learning_rate * delta * input_data
        self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[context]], [updated_weights])

        print("After update:", tf.tensordot(self.w_in_effective[context], input_data, axes=1))
