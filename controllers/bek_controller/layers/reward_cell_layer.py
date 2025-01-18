import numpy as np
import tensorflow as tf
from copy import deepcopy

tf.random.set_seed(5)

class RewardCellLayer:
    def __init__(self, num_reward_cells=1, input_dim=200, num_replay=3, learning_rate=0.1):
        """Initialize the Reward Cell Layer.

        Args:
            num_reward_cells: Number of reward cells in the model. Defaults to 1.
            input_dim: Dimension of the input vector to the layer (== num_pc). Defaults to 200.
            num_replay: Number of replay iterations for the reward learning process. Defaults to 3.
        """
        # Number of reward cells
        self.num_reward_cells = num_reward_cells

        # Activation values for reward cells
        # Shape: (num_reward_cells, 1)
        self.reward_cell_activations = tf.zeros((num_reward_cells, 1), dtype=tf.float32)

        # Number of replay iterations
        self.num_replay = num_replay

        self.learning_rate = learning_rate

        # Initialize weights with small random values
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # Input weights for reward cells
        self.w_in = tf.Variable(
            initializer(shape=(num_reward_cells, input_dim)), dtype=tf.float32
        )

        # Effective weights used for calculations (could differ if weights are frozen or modified)
        self.w_in_effective = tf.Variable(tf.identity(self.w_in), dtype=tf.float32)

    def update_reward_cell_activations(self, input_data, visit=False):
        """Compute the activations of reward cells based on input data.
        Args:
            input_data: array-like, the input data for the reward cells.
            visit: Flag indicating if the cell is being visited. Defaults to False.
        """
        # Calculate the norm of the input data
        input_norm = tf.linalg.norm(input_data, ord=1)
        # Add a small epsilon to prevent division by zero
        safe_denominator = tf.maximum(input_norm, 1e-4)
        # Calculate reward cell activations using effective weights and normalize
        activations = (
            tf.tensordot(self.w_in_effective, input_data, axes=1) / safe_denominator
        )
        # Clip values to prevent extreme values
        self.reward_cell_activations = tf.clip_by_value(activations, 0, 1e6)

        if visit:
            # Update weights directly based on input data
            updated_weights = self.w_in_effective + self.learning_rate * input_data
            self.w_in_effective.assign(updated_weights)

    def replay(self, pcn):
        """Replay the place cell activations and update reward cell weights.
        Args:
            pcn: PlaceCellNetwork instance for replay processing.
        """
        # Create a copy of the place cell network
        pcn_copy = deepcopy(pcn)
        weight_update = tf.zeros_like(self.w_in)
        for time_step in range(10):
            # Exponential decay factor for the current time step
            exponential_decay_factor = tf.math.exp(-time_step / 6)
            # Normalize the place cell activations using L2 norm with added stability
            pc_activations = pcn_copy.place_cell_activations
            norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(pc_activations)), 1e-12))
            normalized_place_cell_activations = pc_activations / norm
            # Ensure no NaN values
            normalized_place_cell_activations = tf.where(
                tf.math.is_nan(normalized_place_cell_activations),
                tf.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations,
            )

            # Update weight_update cumulatively
            weight_update += (
                exponential_decay_factor * normalized_place_cell_activations
            )

            # Update pcn_copy.place_cell_activations
            recurrent_weights_max = tf.reduce_max(pcn_copy.w_rec_tripartite, axis=0)
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
        self.w_in_effective.assign(self.w_in)

    def td_update(self, input_data, next_reward):
        """Perform temporal difference (TD) update for reward learning.

        Args:
            input_data: array-like, input to the reward cell layer.
            next_reward: float, reward at the next time step.
        """
        # Calculate the prediction error (delta)
        prediction = tf.tensordot(self.w_in_effective, input_data, axes=1)
        delta = next_reward - prediction

        # Update weights based on the TD learning rule
        learning_rate = 0.1  # Adjust the learning rate as needed
        updated_weights = self.w_in_effective + learning_rate * delta * input_data
        self.w_in_effective.assign(updated_weights)
