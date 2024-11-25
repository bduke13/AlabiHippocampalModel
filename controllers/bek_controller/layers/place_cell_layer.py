import numpy as np
import tensorflow as tf
from numpy.random import default_rng
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

tf.random.set_seed(5)

class PlaceCellLayer:
    """Model a layer of place cells receiving input from Boundary Vector Cells."""

    def __init__(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        num_scales: int = 1,
    ):
        """Initialize the Place Cell Layer."""
        rng = default_rng()

        # Number of place cells
        self.num_pc = num_pc

        # Enables/disables multiscale functionality
        self.num_scales = num_scales

        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvc_layer = bvc_layer

        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvc_layer.num_bvc

        # Enables/disables updating weights via Oja's rule and STDP
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp

        ############### MULTISCALE ###############
        if self.num_scales > 1:
            # Divide place cells dynamically into scales
            num_cells_per_scale = self.divide_place_cells(self.num_pc, self.num_scales)
            self.place_cell_scales = tf.constant(
                [f"scale_{i}" for i, count in enumerate(num_cells_per_scale) for _ in range(count)]
            )

            # Scale indices and weight initialization
            self.scale_indices = []
            self.w_in_scales = []
            start_idx = 0

            for i, num_cells in enumerate(num_cells_per_scale):
                end_idx = start_idx + num_cells
                indices = tf.range(start_idx, end_idx, dtype=tf.int32)
                self.scale_indices.append(indices)

                # Initialize weights for this scale
                w_in_scale = tf.Variable(
                    rng.binomial(n=1, p=0.2, size=(num_cells, self.num_bvc)),
                    dtype=tf.float32
                )
                self.w_in_scales.append(w_in_scale)
                start_idx = end_idx

            # Combine all weights into the main weight matrix
            self.w_in = tf.concat(self.w_in_scales, axis=0)
        else:
            # Single-scale logic
            self.w_in = tf.Variable(
                rng.binomial(n=1, p=0.2, size=(self.num_pc, self.num_bvc)),
                dtype=tf.float32
            )
            self.scale_indices = [tf.range(0, self.num_pc, dtype=tf.int32)]
        ##########################################

        # Recurrent weight matrix for head direction and place cell interactions
        self.w_rec_tripartite = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)

        # Activation values for place cells
        self.place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

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

        # Head direction modulation (if applicable)
        self.head_direction_modulation = None

        # Trace of place cell activations for eligibility tracking
        self.place_cell_trace = None

        # Trace of head direction cells for eligibility tracking
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), tf.float32)

    def divide_place_cells(self, num_place_cells, num_scales, first_scale_ratio=0.75):
        """
        Divide num_place_cells into num_scales with descending percentages.
        """
        if num_scales < 1:
            raise ValueError("num_scales must be at least 1.")
        if num_scales == 1:
            return [num_place_cells]

        remaining_ratio = 1.0 - first_scale_ratio
        remaining_ratios = [
            remaining_ratio / sum(1 / (i + 1) for i in range(num_scales - 1)) / (j + 1)
            for j in range(num_scales - 1)
        ]
        all_ratios = [first_scale_ratio] + remaining_ratios
        num_cells_per_scale = [int(round(ratio * num_place_cells)) for ratio in all_ratios]
        num_cells_per_scale[-1] += num_place_cells - sum(num_cells_per_scale)
        return num_cells_per_scale

    def get_place_cell_activations(self, input_data, hd_activations, vis_density, collided=False):
        # Store the previous place cell activations
        self.prev_place_cell_activations = tf.identity(self.place_cell_activations)

        # Compute BVC activations
        self.bvc_activations = self.bvc_layer.get_bvc_activation(input_data[0], input_data[1])

        if self.num_scales > 1:
            afferent_excitation = tf.zeros_like(self.place_cell_activations)

            # Compute afferent excitation for each scale
            for i, indices in enumerate(self.scale_indices):
                w_in_scale = tf.gather(self.w_in, indices)
                afferent_excitation_scale = tf.matmul(
                    w_in_scale, tf.expand_dims(self.bvc_activations, -1)
                )
                afferent_excitation_scale = tf.reshape(afferent_excitation_scale, [-1])

                # Scatter the excitation into the full tensor
                afferent_excitation = tf.tensor_scatter_nd_update(
                    afferent_excitation,
                    tf.reshape(indices, [-1, 1]),
                    afferent_excitation_scale
                )
        else:
            afferent_excitation = tf.matmul(
                self.w_in, tf.expand_dims(self.bvc_activations, -1)
            )
            afferent_excitation = tf.reshape(afferent_excitation, [-1])

        # Compute afferent inhibition
        afferent_inhibition = self.gamma_pb * tf.reduce_sum(self.bvc_activations)

        # Compute recurrent inhibition
        if self.num_scales > 1:
            recurrent_inhibition = tf.zeros_like(self.place_cell_activations)
            for indices in self.scale_indices:
                scale_inhibition = self.gamma_pp * tf.reduce_sum(
                    tf.gather(self.place_cell_activations, indices)
                )
                recurrent_inhibition = tf.tensor_scatter_nd_update(
                    recurrent_inhibition,
                    tf.reshape(indices, [-1, 1]),
                    tf.fill([len(indices)], scale_inhibition)
                )
        else:
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

        # Modulate activations based on visual density and scale
        if self.num_scales > 1:
            modulation_factors = tf.TensorArray(dtype=tf.float32, size=self.num_pc)
            for i, indices in enumerate(self.scale_indices):
                # Define scale factor and ensure it's a tf.float32
                scale_factor = tf.cast(
                    vis_density if i == 0 else (1.0 - vis_density) * (1 / (i + 1)),
                    tf.float32
                )
                modulation_factors = modulation_factors.scatter(
                    indices, tf.fill([len(indices)], scale_factor)
                )
            modulation_factors = modulation_factors.stack()
            self.place_cell_activations *= modulation_factors

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
            if self.num_scales > 1:
                for i, indices in enumerate(self.scale_indices):
                    activations_scale = tf.gather(self.place_cell_activations, indices)
                    w_in_scale = tf.gather(self.w_in, indices)

                    # Modulate learning rate
                    scale_factor = vis_density if i == 0 else (1.0 - vis_density) * (1 / (i + 1))
                    weight_update = self.tau * scale_factor * (
                        activations_scale[:, tf.newaxis] * (
                            self.bvc_activations[tf.newaxis, :]
                            - (1 / self.alpha_pb) * activations_scale[:, tf.newaxis] * w_in_scale
                        )
                    )
                    self.w_in = tf.tensor_scatter_nd_update(
                        self.w_in, tf.reshape(indices, [-1, 1]), w_in_scale + weight_update
                    )
            else:
                weight_update = self.tau * (
                    self.place_cell_activations[:, tf.newaxis] * (
                        self.bvc_activations[tf.newaxis, :]
                        - (1 / self.alpha_pb) * self.place_cell_activations[:, tf.newaxis] * self.w_in
                    )
                )
                self.w_in.assign_add(weight_update)

    def get_activations_by_scale(self, scale: str):
        """Retrieve place cell activations for a specific scale."""
        if self.num_scales <= 1:
            raise ValueError("Multiscale is not enabled or num_scales is 1.")

        scale_idx = int(scale.split("_")[-1])  # Extract scale index from the name
        if scale_idx >= len(self.scale_indices):
            raise ValueError(f"Scale {scale} is out of bounds.")

        indices = self.scale_indices[scale_idx]
        return tf.gather(self.place_cell_activations, indices)

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations.assign(tf.zeros_like(self.place_cell_activations))
        self.activation_update.assign(tf.zeros_like(self.activation_update))
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
