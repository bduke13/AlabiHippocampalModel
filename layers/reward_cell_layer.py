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

    def __call__(self, input_data, visit=False, context_index=1):
        '''
        Computes the activations of reward cells based on input data.

        Parameters:
        input_data: The input data for the reward cells.
        visit: Boolean flag indicating if the cell is being visited.
        context_index: Context index for selecting specific input weights.
        '''
        self.reward_cell_activations = tf.tensordot(self.w_in_effective, input_data, 1) / tf.linalg.norm(input_data, 1)

        if visit:
            updated_weights = self.w_in_effective[context_index] - 0.2 * input_data * self.w_in_effective[context_index]
            self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[context_index]], [updated_weights])

    def new_reward(self, place_cell_layer, context_index=0, target=None):
        '''
        Updates the input weights based on place cell network activations through replay.

        Parameters:
        place_cell_layer: The place cell network whose activations influence the reward cell weights.
        context_index: Context index for selecting specific input weights.
        target: Target vector for unlearning, if provided.
        '''
        place_cell_layer = deepcopy(place_cell_layer)
        delta_weights = tf.zeros_like(self.w_in)

        for t in range(10):
            # Update weights if a target is specified for unlearning
            if target is not None:
                update_value = tf.maximum(0, self.w_in_effective[context_index] - 0.6 * place_cell_layer.place_cell_activations)
                self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[context_index]], [update_value])
                print("Unlearned weights minimum:", tf.reduce_min(self.w_in_effective))
                return

            # Accumulate weight changes over time, modulated by the replay steps
            delta_weights = tf.tensor_scatter_nd_add(delta_weights, [[context_index]], [
                tf.math.exp(-t / 6) * tf.linalg.normalize(place_cell_layer.place_cell_activations, np.inf)[0]
            ])

            # Update the place cell network activations
            previous_activations = tf.identity(place_cell_layer.place_cell_activations)
            new_activations = tf.nn.relu(
                tf.tensordot(tf.cast(tf.reduce_max(place_cell_layer.w_rec_place_to_place, 0), tf.float32), previous_activations, 1) + previous_activations
            )
            place_cell_layer.place_cell_activations = tf.tanh(new_activations)

        # Normalize and update input weights
        self.w_in.assign_add(tf.linalg.normalize(delta_weights, np.inf)[0])
        self.w_in_effective = tf.identity(self.w_in)