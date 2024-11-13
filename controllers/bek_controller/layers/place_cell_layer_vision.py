import numpy as np
import tensorflow as tf
import os
from numpy.random import default_rng
from tensorflow.keras.models import load_model

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
tf.get_logger().setLevel("ERROR")

tf.random.set_seed(5)


class PlaceCellLayer:
    """Model a layer of place cells receiving input from an encoder model.

    Place cells develop spatially localized receptive fields (place fields) through
    competitive learning and synaptic plasticity.

    This implementation is based on the model described in Chapter 3 of the
    dissertation, specifically Equations (3.2a), (3.2b), and (3.3).
    """

    def __init__(
        self,
        encoder_path: str = "encoder_model.keras",
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_hebb: bool = False,
    ):
        """Initialize the Place Cell Layer.

        Args:
            encoder_path: Path to the pre-trained encoder model.
            num_pc: Number of place cells in the layer.
            timestep: Time step for simulation/learning updates in milliseconds.
            n_hd: Number of head direction cells.
            enable_ojas: Enable weight updates via competition.
            enable_hebb: Enable tripartite synapse weight updates.
            input_height: Height of the input images.
            input_width: Width of the input images.
        """
        rng = default_rng()

        # Number of place cells
        self.num_pc = num_pc

        # Load the encoder model
        self.encoder = load_model(
            encoder_path
        )  # compile=False reduces startup messages
        # Get the output shape from the encoder model
        self.num_features = self.encoder.output_shape[
            -1
        ]  # Gets the last dimension of output shape

        self.input_width = 96
        self.input_height = 96
        # Input weight matrix connecting place cells to encoder features
        # Initialized with a 20% probability of connection
        # Shape: (num_pc, num_features)
        self.w_in = tf.Variable(
            rng.binomial(n=1, p=0.1, size=(num_pc, self.num_features)), dtype=tf.float32
        )

        # Recurrent weight matrix for head direction and place cell interactions
        # Shape: (n_hd, num_pc, num_pc)
        self.w_rec_tripartite = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)

        # Activation values for place cells
        # Shape: (num_pc,)
        self.place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        # Time constant for updating place cell activations
        self.tau = timestep / 1000  # Convert timestep to seconds

        # Activation values for encoder features
        # Shape: (num_features,)
        self.encoder_features = tf.zeros(self.num_features, dtype=tf.float32)

        # Coefficient to modify effect of place cell recurrent inhibition (Γ_pp in Equation 3.2a)
        self.gamma_pp = 0.8

        # Coefficient to modify effect of encoder feature afferent inhibition (Γ_pb in Equation 3.2a)
        self.gamma_pb = 0.03

        # Time constant for the membrane potential dynamics of place cells (τ_p in Equation 3.2a)
        self.tau_p = 0.05

        # Normalization factor for synaptic weight updates (α_pb in Equation 3.3)
        self.alpha_pb = np.sqrt(0.5)

        # Previous place cell activations
        # Shape: (num_pc,)
        self.prev_place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        # Initial weights for the input connections from encoder features to place cells
        self.initial_w_in = tf.Variable(self.w_in)

        # Temporary variable for the current activation update step
        # Shape: (num_pc,)
        self.activation_update = tf.zeros_like(
            self.place_cell_activations, dtype=tf.float32
        )

        # Head direction modulation (if applicable)
        self.head_direction_modulation = None

        # Trace of place cell activations for eligibility tracking
        self.place_cell_trace = tf.zeros_like(self.place_cell_activations)

        # Trace of head direction cells for eligibility tracking
        # Shape: (n_hd, 1, 1)
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), tf.float32)

        # Enables/disables updating weights to spread place cells through environment via competition
        self.enable_ojas = enable_ojas

        # Enables/disables updating weights in the tripartite synapses to track adjacencies between cells
        self.enable_hebb = enable_hebb

    def get_place_cell_activations(
        self, input_data, hd_activations, collided: bool = False
    ):
        """Compute place cell activations from image input and head direction inputs.

        Args:
            input_data: Input image data as a numpy array or tensor.
            hd_activations: Head direction cell activations.
            collided: Whether the agent has collided with an obstacle.
        """
        # Ensure input_data has a batch dimension
        if len(input_data.shape) == 3:
            input_data = tf.expand_dims(input_data, 0)  # Add batch dimension

        # Get encoder features directly from the input
        self.encoder_features = tf.squeeze(self.encoder(input_data))

        # Set bvc_activations to encoder_features (if you use this terminology elsewhere)
        self.bvc_activations = self.encoder_features

        # Compute the input to place cells using encoder features
        afferent_excitation = tf.tensordot(self.w_in, self.encoder_features, axes=1)

        # Compute the total encoder feature activity for afferent inhibition
        afferent_inhibition = self.gamma_pb * tf.reduce_sum(self.encoder_features)

        # Compute the total place cell activity for recurrent inhibition
        # Recurrent inhibition term: Γ^{pp} ∑_j v_j^p (Equation 3.2a)
        recurrent_inhibition = self.gamma_pp * tf.reduce_sum(
            self.place_cell_activations
        )

        # Update the activation variable `activation_update` with the new input, applying the membrane potential dynamics
        # Equation (3.2a): τ_p (ds_i^p/dt) = -s_i^p + afferent_excitation - afferent_inhibition - recurrent_inhibition
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )

        # Apply ReLU followed by tanh to compute the new place cell activations
        # Equation (3.2b): v_i^p = tanh([ψ s_i^p]_+)
        # Here, ψ is implicitly set to 1
        self.place_cell_activations = tf.tanh(tf.nn.relu(self.activation_update))

        # Update the eligibility trace and weights
        if (
            self.enable_hebb
            and tf.reduce_any(tf.not_equal(self.place_cell_activations, 0))
            and not collided
        ):
            # Update the eligibility trace for place cells and head direction cells
            # Eligibility traces are used for temporal difference learning and sequence encoding
            if self.place_cell_trace is None:
                self.place_cell_trace = tf.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (
                self.tau / 3 * (self.place_cell_activations - self.place_cell_trace)
            )
            # Convert hd_activations to float32 and expand dimensions
            hd_expanded = tf.cast(
                tf.expand_dims(tf.expand_dims(hd_activations, -1), -1), dtype=tf.float32
            )
            self.hd_cell_trace += self.tau / 3 * (hd_expanded - self.hd_cell_trace)

            # Update recurrent weights for place cell interactions modulated by head direction
            # This implements sequence learning and is similar to STDP
            delta_w_rec = self.hd_cell_trace * (
                tf.tensordot(
                    self.place_cell_activations[:, tf.newaxis],
                    self.place_cell_trace[tf.newaxis, :],
                    axes=1,
                )
                - tf.tensordot(
                    self.place_cell_trace[:, tf.newaxis],
                    self.place_cell_activations[tf.newaxis, :],
                    axes=1,
                )
            )
            self.w_rec_tripartite += tf.cast(delta_w_rec, tf.float32)

        # Update the input weights based on the current activations and encoder features
        # This is the competitive learning rule from Equation (3.3)
        if self.enable_ojas and tf.reduce_any(
            tf.not_equal(self.place_cell_activations, 0)
        ):
            # Compute the weight update according to Oja's rule (Equation 3.3)
            weight_update = self.tau * (
                self.place_cell_activations[:, tf.newaxis]
                * (
                    self.encoder_features[tf.newaxis, :]
                    - (1 / self.alpha_pb)
                    * self.place_cell_activations[:, tf.newaxis]
                    * self.w_in
                )
            )
            # Update the input weights from encoder features to place cells
            self.w_in.assign_add(weight_update)

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations.assign(tf.zeros_like(self.place_cell_activations))
        self.activation_update.assign(tf.zeros_like(self.activation_update))
        self.place_cell_trace = None

    def preplay(self, direction, num_steps=1):
        """Simulate preplay of place cell activations using recurrent weights.

        Used to predict future states without actual movement.

        Args:
            direction: Index of head direction for exploiting recurrent weights.
            num_steps: Number of exploitation steps to simulate looking ahead.

        Returns:
            Updated place cell activations after exploitation.
        """
        # Copy the current place cell activations
        # Shape: (num_pc,)
        place_cell_activations = tf.identity(self.place_cell_activations)

        # Iterate to update the place cell activations
        for _ in range(num_steps):
            # Store previous activations
            previous_activations = tf.identity(place_cell_activations)

            # Compute new activations based on recurrent weights and previous activations
            # The recurrent weights are modulated by the specified head direction
            place_cell_activations = tf.tanh(
                tf.nn.relu(
                    tf.tensordot(
                        self.w_rec_tripartite[direction], previous_activations, axes=1
                    )
                    - previous_activations
                )
            )

        # Return the updated place cell activations
        return place_cell_activations
