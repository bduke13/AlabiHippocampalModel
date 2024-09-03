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
        
        # Input weight matrix connecting reward cells to input data
        # Shape: (num_reward_cells, input_dim)
        self.w_in = tf.Variable(np.zeros((num_reward_cells, input_dim)), dtype=tf.float32)
        
        # Activation values for reward cells
        # Shape: (num_reward_cells, 1)
        self.reward_cell_activations = tf.zeros((num_reward_cells, 1), dtype=tf.float32)
        
        # Number of replay iterations
        self.num_replay = num_replay
        
        # Effective input weight matrix, used for updating during visits
        self.w_in_effective = tf.identity(self.w_in)

    def compute_reward_cell_activations(self, input_data, visit=False, context=1):
        '''
        Computes the activations of reward cells based on input data.

        Parameters:
        input_data: The input data for the reward cells.
        visit: Boolean flag indicating if the cell is being visited.
        context: Context index for selecting specific input weights.
        '''
        self.reward_cell_activations = tf.tensordot(self.w_in_effective, input_data, 1) / tf.linalg.norm(input_data, 1)
        if visit:
            updated_weights = self.w_in_effective[context] - 0.2 * input_data * self.w_in_effective[context]
            self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[context]], [updated_weights])

    def td_update(self, input_data, next_reward, context=0):
        '''
        Temporal difference (TD) update method for reward learning.

        Parameters:
        input_data: Input to the reward cell layer.
        next_expected_input: Expected input at the next time step.
        next_reward: Reward at the next time step.
        context: Context index for selecting specific input weights.
        '''
        print("Next reward:", next_reward)
        print("Before update:", tf.tensordot(self.w_in, input_data, 1))

        # Perform the TD update on the effective weight matrix
        v_prime = (0.6 * self.w_in[context] * input_data) - self.w_in[context] * input_data
        self.w_in_effective = tf.tensor_scatter_nd_add(self.w_in_effective, [[context]], [v_prime])

        print("After update:", tf.tensordot(self.w_in[context], input_data, 1))