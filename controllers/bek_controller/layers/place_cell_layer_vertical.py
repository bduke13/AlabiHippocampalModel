import tensorflow as tf
from layers.boundary_vector_cell_layer_vertical import BoundaryVectorCellLayer

tf.random.set_seed(5)


class PlaceCellLayer:
    """Model a layer of place cells receiving input from Boundary Vector Cells.

    Place cells develop spatially localized receptive fields (place fields) through
    competitive learning and synaptic plasticity.
    """

    def __init__(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
    ):
        """Initialize the Place Cell Layer.

        Args:
            bvc_layer: The BVC layer used as input to place cell activations.
            num_pc: Number of place cells in the layer.
            timestep: Time step for simulation/learning updates in milliseconds.
            n_hd: Number of head direction cells.
            enable_ojas: Enable weight updates via competition.
            enable_stdp: Enable tripartite synapse weight updates via STDP.
        """

        self.num_pc = num_pc
        self.bvc_layer = bvc_layer
        self.num_bvc = self.bvc_layer.num_bvc

        # Instead of NumPy rng.binomial, use TF random ops:
        # Create a mask of shape (num_pc, num_bvc) with True if <0.2, else False
        w_init = tf.cast(tf.random.uniform([num_pc, self.num_bvc]) < 0.25, tf.float32)

        self.w_in = tf.Variable(w_init, dtype=tf.float32)
        self.w_rec_tripartite = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)

        self.place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        self.tau = timestep / 1000.0
        self.bvc_activations = tf.zeros(self.num_bvc, dtype=tf.float32)
        self.gamma_pp = 0.5
        self.gamma_pb = 0.3
        self.tau_p = 0.5
        self.alpha_pb = tf.sqrt(tf.constant(0.5, dtype=tf.float32))

        self.prev_place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)
        self.initial_w_in = tf.Variable(self.w_in)

        self.activation_update = tf.zeros_like(
            self.place_cell_activations, dtype=tf.float32
        )

        self.head_direction_modulation = None
        self.boundary_cell_activations = tf.zeros((n_hd, num_pc))
        self.place_cell_trace = tf.zeros_like(self.place_cell_activations)
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), dtype=tf.float64)

        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp

    def get_place_cell_activations(
        self, input_data, hd_activations, collided: bool = False
    ):
        """Compute place cell activations from BVC and head direction inputs."""

        self.prev_place_cell_activations = tf.identity(self.place_cell_activations)

        self.bvc_activations = self.bvc_layer.get_bvc_activation(input_data)

        # Afferent excitation: ∑_j W_ij^{pb} v_j^b
        afferent_excitation = tf.tensordot(self.w_in, self.bvc_activations, axes=1)

        # Afferent inhibition: Γ^{pb} ∑_j v_j^b
        afferent_inhibition = self.gamma_pb * tf.reduce_sum(self.bvc_activations)

        # Recurrent inhibition: Γ^{pp} ∑_j v_j^p
        recurrent_inhibition = self.gamma_pp * tf.reduce_sum(
            self.place_cell_activations
        )

        # Update activations: τ_p ds_i^p/dt = -s_i^p + afferent_excitation - afferent_inhibition - recurrent_inhibition
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )

        self.place_cell_activations = tf.tanh(tf.nn.relu(self.activation_update))

        # STDP updates if enabled
        if (
            self.enable_stdp
            and not collided
            and tf.reduce_any(tf.not_equal(self.place_cell_activations, 0.0))
        ):
            if self.place_cell_trace is None:
                self.place_cell_trace = tf.zeros_like(self.place_cell_activations)

            self.place_cell_trace += (
                self.tau / 3.0 * (self.place_cell_activations - self.place_cell_trace)
            )

            # Replace np.nan_to_num with TF equivalent:
            hd_activations_no_nan = tf.where(
                tf.math.is_nan(hd_activations),
                tf.zeros_like(hd_activations),
                hd_activations,
            )

            # Expand dimensions to match shapes:
            # hd_activations[:, np.newaxis, np.newaxis] -> expand dims twice
            hd_expanded = tf.expand_dims(tf.expand_dims(hd_activations_no_nan, 1), 2)

            self.hd_cell_trace += self.tau / 3.0 * (hd_expanded - self.hd_cell_trace)

            # STDP: w_rec_tripartite += hd * (v^p(t)*v^p(t-1)^T - v^p(t-1)*v^p(t)^T)
            # Expand dimensions for outer products:
            current_acts_expanded = tf.expand_dims(
                self.place_cell_activations, 1
            )  # (num_pc,1)
            trace_expanded = tf.expand_dims(self.place_cell_trace, 0)  # (1, num_pc)

            forward_term = tf.tensordot(
                self.place_cell_activations, self.place_cell_trace, axes=0
            )  # (num_pc, num_pc)
            backward_term = tf.tensordot(
                self.place_cell_trace, self.place_cell_activations, axes=0
            )  # (num_pc, num_pc)

            # Apply head direction modulation
            # w_rec_tripartite: (n_hd, num_pc, num_pc)
            # Add hd_expanded * (forward_term - backward_term)
            # hd_expanded: (n_hd, 1, 1), broadcast to (n_hd, num_pc, num_pc)
            self.w_rec_tripartite.assign_add(
                tf.cast(hd_expanded, tf.float32) * (forward_term - backward_term)
            )

        # Oja's rule updates if enabled
        if self.enable_ojas and tf.reduce_any(
            tf.not_equal(self.place_cell_activations, 0.0)
        ):
            # Expand dimensions for broadcasting
            pc_acts_expanded = tf.expand_dims(
                self.place_cell_activations, 1
            )  # (num_pc, 1)
            bvc_acts_expanded = tf.expand_dims(self.bvc_activations, 0)  # (1, num_bvc)

            # weight_update = τ * [v_i^p (v_j^b - (1/α_pb)*v_i^p*w_ij^{pb})]
            weight_update = self.tau * (
                pc_acts_expanded
                * (
                    bvc_acts_expanded
                    - (1.0 / self.alpha_pb) * pc_acts_expanded * self.w_in
                )
            )
            self.w_in.assign_add(weight_update)

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
            # place_cell_activations = tanh(ReLU(W_rec_tripartite[direction]*previous - previous))
            # tensordot shape: (num_pc,) <- (num_pc, num_pc) x (num_pc,)
            rec_input = tf.tensordot(
                self.w_rec_tripartite[direction], previous_activations, axes=1
            )
            place_cell_activations = tf.tanh(
                tf.nn.relu(rec_input - previous_activations)
            )

        return place_cell_activations
