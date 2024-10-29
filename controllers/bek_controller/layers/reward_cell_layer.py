import numpy as np
import tensorflow as tf
from copy import deepcopy

tf.random.set_seed(5)


class RewardCellLayer:
    def __init__(self, num_reward_cells=1, input_dim=200, num_replay=3, context=0):
        """
        Initializes the Reward Cell Layer.

        Parameters:
        num_reward_cells: Number of reward cells in the model. One reward cell tracks an individual reward map or context.
        input_dim: Dimension of the input vector to the layer (== num_pc).
        num_replay: Number of replay iterations for the reward learning process.
        context: Decides which context to use for activations. Context 0 selects the first reward cell.
        """
        # Given n reward cells, each reward cell has a corresponding weight vector, accessed by the index of the cell. Just use 0 for now.
        self.context = context

        # Number of reward cells
        self.num_reward_cells = num_reward_cells

        # Activation values for reward cells
        # Shape: (num_reward_cells, 1)
        self.reward_cell_activations = tf.zeros((num_reward_cells, 1), dtype=tf.float32)

        # Number of replay iterations
        self.num_replay = num_replay

        # Initialize weights with small random values
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    
        self.w_in = tf.Variable(
            initializer(shape=(num_reward_cells, input_dim)), dtype=tf.float32
        )
    
        self.w_in_effective = tf.Variable(
            tf.identity(self.w_in), dtype=tf.float32
        )

    def update_reward_cell_activations(self, input_data, visit=False):
        """
        Computes the activations of reward cells based on input data.

        Parameters:
        input_data: The input data for the reward cells.
        visit: Boolean flag indicating if the cell is being visited.
        """
        # Calculate the norm of the input data
        input_norm = tf.linalg.norm(input_data, ord=1)
        # Add a small epsilon to prevent division by zero
        safe_denominator = tf.maximum(input_norm, 1e-6)
        # Calculate reward cell activations using effective weights
        self.reward_cell_activations = tf.tensordot(self.w_in_effective, input_data, axes=1) / safe_denominator

        if visit:
            # Update weights directly based on input data
            learning_rate = 0.1  # Adjust as needed
            updated_weights = self.w_in_effective[self.context] + learning_rate * input_data
            self.w_in_effective = tf.tensor_scatter_nd_update(
                self.w_in_effective, [[self.context]], [updated_weights]
            )
            # # Synchronize weights
            # self.w_in = tf.identity(self.w_in_effective)

    def replay(self, pcn):
        """
        Replays the place cell activations and updates the reward cell weights.

        Parameters:
        pcn: The place cell network.
        """
        # Create a copy of the place cell network
        pcn_copy = deepcopy(pcn)
        weight_update = tf.zeros_like(self.w_in)

        for time_step in range(10):
            # Update weight_update
            exponential_decay_factor = tf.math.exp(-time_step / 6)

            # Normalize the place cell activations using the infinity norm (max absolute value).
            # This scales the activations so that the largest activation becomes 1, helping to prevent any
            # single activation from dominating the update and improving numerical stability.
            normalized_place_cell_activations = tf.linalg.normalize(
                pcn_copy.place_cell_activations, ord=2
            )[0]

            # tf.tensor_scatter_nd_add() modifies the weight_update tensor at the specified index (self.context),
            # adding the normalized activations weighted by an exponential decay factor.
            # This gradually decreases the contribution of place cell activations over the iterations.
            weight_update = tf.tensor_scatter_nd_add(
                weight_update,
                [[self.context]],
                [exponential_decay_factor * normalized_place_cell_activations],
            )

            # Update pcn_copy.place_cell_activations
            recurrent_weights_max = tf.reduce_max(pcn_copy.w_rec_hd_place, axis=0)
            updated_place_cell_activations = tf.nn.relu(
                tf.tensordot(
                    tf.cast(recurrent_weights_max, tf.float32),
                    pcn_copy.place_cell_activations,
                    axes=1,
                )
                + pcn_copy.place_cell_activations
            )
            pcn_copy.place_cell_activations = tf.tanh(updated_place_cell_activations)

        # Normalize and update weights
        normalized_weight_update = tf.linalg.normalize(weight_update, ord=np.inf)[0]
        self.w_in.assign_add(normalized_weight_update)
        self.w_in_effective = tf.identity(self.w_in)

    def td_update(self, input_data, next_reward):
        """
        Temporal difference (TD) update method for reward learning.

        Parameters:
        input_data: Input to the reward cell layer.
        next_reward: Reward at the next time step.
        """
        # Calculate the prediction error (delta)
        prediction = tf.tensordot(self.w_in_effective[self.context], input_data, axes=1)
        delta = next_reward - prediction

        # Update weights based on the TD learning rule
        learning_rate = 0.2  # Adjust the learning rate as needed
        updated_weights = self.w_in_effective[self.context] + learning_rate * delta * input_data
        self.w_in_effective = tf.tensor_scatter_nd_update(self.w_in_effective, [[self.context]], [updated_weights])
