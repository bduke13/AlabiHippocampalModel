import numpy as np
import tensorflow as tf
from numpy.random import default_rng
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

tf.random.set_seed(5)

class PlaceCellLayer:
    def __init__(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        modulate_by_vis_density: bool = True,  # Flag for modulation
    ):
        rng = default_rng()

        # Number of place cells
        self.num_pc = num_pc

        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvc_layer = bvc_layer

        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvc_layer.num_bvc

        # Enables/disables updating weights via Oja's rule and STDP
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp

        # Flag for modulating place cell activations by visual density
        self.modulate_by_vis_density = modulate_by_vis_density

        # Initialize weights for input connections from BVCs to place cells
        self.w_in = tf.Variable(
            rng.binomial(n=1, p=0.2, size=(self.num_pc, self.num_bvc)),
            dtype=tf.float32
        )

        # Recurrent weight matrix for head direction and place cell interactions
        self.w_rec_tripartite = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)

        # Activation values for place cells
        self.place_cell_activations = tf.Variable(tf.zeros(num_pc, dtype=tf.float32))

        # Time constant for updating place cell activations
        self.tau = timestep / 1000  # Convert timestep to seconds

        # Activation values for boundary vector cells (BVCs)
        self.bvc_activations = tf.zeros(self.num_bvc, dtype=tf.float32)

        # Coefficient to modify effect of place cell recurrent inhibition
        self.gamma_pp = 0.5

        # Coefficient to modify effect of boundary vector cell afferent inhibition
        self.gamma_pb = 0.3

        # Time constant for the membrane potential dynamics of place cells
        self.tau_p = 0.5

        # Normalization factor for synaptic weight updates
        self.alpha_pb = np.sqrt(0.5)

        # Previous place cell activations
        self.prev_place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        # Initial weights for the input connections from BVCs to place cells
        self.initial_w_in = tf.Variable(self.w_in)

        # Temporary variable for the current activation update step
        self.activation_update = tf.zeros_like(self.place_cell_activations, dtype=tf.float32)

        # Trace of place cell activations for eligibility tracking
        self.place_cell_trace = None

        # Trace of head direction cells for eligibility tracking
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), tf.float32)

    def get_place_cell_activations(self, input_data, hd_activations, vis_density, collided=False):
        # Store the previous place cell activations
        self.prev_place_cell_activations = tf.identity(self.place_cell_activations)

        # Compute BVC activations
        self.bvc_activations = self.bvc_layer.get_bvc_activation(input_data[0], input_data[1])

        # Compute afferent excitation
        afferent_excitation = tf.matmul(
            self.w_in, tf.expand_dims(self.bvc_activations, -1)
        )
        afferent_excitation = tf.reshape(afferent_excitation, [-1])

        # Compute afferent inhibition
        afferent_inhibition = self.gamma_pb * tf.reduce_sum(self.bvc_activations)

        # Compute recurrent inhibition
        recurrent_inhibition = self.gamma_pp * tf.reduce_sum(self.place_cell_activations)

        # Update activation variable
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )

        # Apply activation function
        self.place_cell_activations = tf.tanh(tf.nn.relu(self.activation_update))

        # Modulate activations based on visual density if enabled
        if self.modulate_by_vis_density:
            self.place_cell_activations *= 1-vis_density  # Modulate by vis_density
        else:
            pass  # Do not modulate activations

        # Update eligibility trace and weights
        if self.enable_stdp and tf.reduce_any(self.place_cell_activations > 0) and not collided:
            # Update eligibility traces
            if self.place_cell_trace is None:
                self.place_cell_trace = tf.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (
                self.tau / 3 * (self.place_cell_activations - self.place_cell_trace)
            )
            self.hd_cell_trace += (
                self.tau / 3 * (
                    tf.expand_dims(tf.expand_dims(hd_activations, axis=-1), axis=-1)
                    - self.hd_cell_trace
                )
            )

            # Update recurrent weights
            delta_w_rec = tf.einsum(
                'i,j->ij', self.place_cell_activations, self.place_cell_trace
            ) - tf.einsum(
                'i,j->ij', self.place_cell_trace, self.place_cell_activations
            )
            delta_w_rec = tf.expand_dims(delta_w_rec, axis=0)
            delta_w_rec *= tf.expand_dims(
                tf.expand_dims(hd_activations, axis=-1), axis=-1
            )
            self.w_rec_tripartite += delta_w_rec

        # Update input weights via Oja's rule
        if self.enable_ojas and tf.reduce_any(self.place_cell_activations > 0):
            weight_update = self.tau * (
                self.place_cell_activations[:, tf.newaxis] * (
                    self.bvc_activations[tf.newaxis, :]
                    - (1 / self.alpha_pb) * self.place_cell_activations[:, tf.newaxis] * self.w_in
                )
            )
            self.w_in.assign_add(weight_update)

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations = tf.zeros_like(self.place_cell_activations, dtype=tf.float32)
        self.activation_update = tf.zeros_like(self.activation_update, dtype=tf.float32)
        self.place_cell_trace = None

    def preplay(self, direction, num_steps=1):
        """Simulate preplay of place cell activations using recurrent weights."""
        place_cell_activations = tf.identity(self.place_cell_activations)
        for _ in range(num_steps):
            previous_activations = tf.identity(place_cell_activations)
            place_cell_activations = tf.tanh(
                tf.nn.relu(
                    tf.tensordot(
                        tf.cast(self.w_rec_tripartite[direction], tf.float32),
                        previous_activations,
                        axes=1,
                    ) - previous_activations
                )
            )
        return place_cell_activations
