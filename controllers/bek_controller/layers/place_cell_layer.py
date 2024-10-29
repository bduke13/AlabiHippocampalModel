import numpy as np
import tensorflow as tf
from numpy.random import default_rng
from enums import RobotStage, RobotMode
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

tf.random.set_seed(5)


class PlaceCellLayer:
    """
    The PlaceCellLayer class models a layer of place cells that receive input from a Boundary Vector Cell (BVC) layer.
    Place cells develop spatially localized receptive fields (place fields) through competitive learning and synaptic plasticity.

    This implementation is based on the model described in Chapter 3 of the dissertation, specifically Equations (3.2a), (3.2b), and (3.3).
    """

    def __init__(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
    ):
        """
        Initializes the Place Cell Layer.

        Parameters:
        - num_pc (int): Number of place cells in the layer.
        - input_dim (int): Dimension of the input vector to the layer (e.g., 720 for RPLidar).
        - timestep (int): Time step for simulation or learning updates (in milliseconds).
        - max_dist (float): Maximum distance that the boundary vector cells (BVCs) respond to.
        - n_hd (int): Number of head direction cells.
        """
        rng = default_rng()

        # Number of place cells
        self.num_pc = num_pc

        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvc_layer = bvc_layer

        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvc_layer.num_bvc

        # Input weight matrix connecting place cells to BVCs
        # Initialized with a 20% probability of connection
        # Shape: (num_pc, num_bvc)
        self.w_in = tf.Variable(
            rng.binomial(1, 0.2, (num_pc, self.num_bvc)), dtype=tf.float32
        )

        # Recurrent weight matrix for head direction and place cell interactions
        # Shape: (n_hd, num_pc, num_pc)
        self.w_rec_tripartite = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)

        # Activation values for place cells
        # Shape: (num_pc,)
        self.place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        # Time constant for updating place cell activations
        self.tau = timestep / 1000  # Convert timestep to seconds

        # Activation values for boundary vector cells (BVCs)
        # Shape: (num_bvc,)
        self.bvc_activations = tf.zeros(self.num_bvc, dtype=tf.float32)

        # Coefficient to modify effect of place cell recurrent inhibition (Γ_pp in Equation 3.2a)
        self.gamma_pp = 0.5

        # Coefficient to modify effect of boundary vector cell afferent inhibition (Γ_pb in Equation 3.2a)
        self.gamma_pb = 0.3

        # Time constant for the membrane potential dynamics of place cells (τ_p in Equation 3.2a)
        self.tau_p = 0.1

        # Normalization factor for synaptic weight updates (α_pb in Equation 3.3)
        self.alpha_pb = np.sqrt(0.5)

        # Previous place cell activations
        # Shape: (num_pc,)
        self.prev_place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)

        # Initial weights for the input connections from BVCs to place cells
        self.initial_w_in = tf.Variable(self.w_in)

        # Temporary variable for the current activation update step
        # Shape: (num_pc,)
        self.activation_update = tf.zeros_like(
            self.place_cell_activations, dtype=tf.float32
        )

        # Placeholder for recurrent inputs (not used in this simplified model)
        self.recurrent_input = 0

        # Head direction modulation (if applicable)
        self.head_direction_modulation = None

        # Boundary cell activation values (if any boundary cells are used)
        # Shape: (n_hd, num_pc)
        self.boundary_cell_activations = tf.zeros((n_hd, num_pc))

        # Trace of place cell activations for eligibility tracking
        self.place_cell_trace = tf.zeros_like(self.place_cell_activations)

        # Trace of head direction cells for eligibility tracking
        # Shape: (n_hd, 1, 1)
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), tf.float64)

        # Enables/disables updating weights to spread place cells through environment via competition
        self.enable_ojas = False

        # Enables/disables updating weights in the tripartite synapses to track adjacencies between cells
        self.enable_hebb = False

    def get_place_cell_activations(
        self, input_data, hd_activations, mode=RobotMode.LEARNING, collided=False
    ):
        """
        Computes the activation of place cells based on the input from boundary vector cells (BVCs) and head direction activations.

        Parameters:
        - input_data (tuple): Input to the BVC layer.
            - input_data[0]: Array of distances (e.g., from LiDAR or range finder).
            - input_data[1]: Array of angles corresponding to those distances.
        - hd_activations (array): Head direction activations.
        - mode (str): Operation mode, typically "learning" or "dmtp".
        - collided (bool): Indicates if the agent has collided with an obstacle.
        """
        # Store the previous place cell activations
        self.prev_place_cell_activations = tf.identity(self.place_cell_activations)

        # Compute BVC activations based on the input distances and angles
        self.bvc_activations = self.bvc_layer.get_bvc_activation(
            input_data[0], input_data[1]
        )

        # Compute the input to place cells by taking the dot product of the input weights and BVC activations
        # Afferent excitation term: ∑_j W_ij^{pb} v_j^b (Equation 3.2a)
        afferent_excitation = tf.tensordot(self.w_in, self.bvc_activations, axes=1)

        # Compute the total BVC activity for afferent inhibition
        # Afferent inhibition term: Γ^{pb} ∑_j v_j^b (Equation 3.2a)
        afferent_inhibition = self.gamma_pb * tf.reduce_sum(self.bvc_activations)

        # Compute the total place cell activity for recurrent inhibition
        # Recurrent inhibition term: Γ^{pp} ∑_j v_j^p (Equation 3.2a)
        recurrent_inhibition = self.gamma_pp * tf.reduce_sum(
            tf.cast(self.place_cell_activations, tf.float32)
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

        # Update the eligibility trace and weights if in "dmtp" mode and no collision
        if self.enable_hebb and np.any(self.place_cell_activations):
            # Update the eligibility trace for place cells and head direction cells
            # Eligibility traces are used for temporal difference learning and sequence encoding
            if self.place_cell_trace is None:
                self.place_cell_trace = tf.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (
                self.tau / 3 * (self.place_cell_activations - self.place_cell_trace)
            )
            self.hd_cell_trace += (
                self.tau
                / 3
                * (
                    np.nan_to_num(hd_activations)[:, np.newaxis, np.newaxis]
                    - self.hd_cell_trace
                )
            )

            # Update recurrent weights for place cell interactions modulated by head direction
            # This implements sequence learning and is similar to STDP
            self.w_rec_tripartite += tf.cast(
                np.nan_to_num(hd_activations)[:, np.newaxis, np.newaxis], tf.float32
            ) * (
                tf.tensordot(
                    self.place_cell_activations[:, np.newaxis],
                    self.place_cell_trace[np.newaxis, :],
                    axes=1,
                )
                - tf.tensordot(
                    self.place_cell_trace[:, np.newaxis],
                    self.place_cell_activations[np.newaxis, :],
                    axes=1,
                )
            )

        # Update the input weights based on the current activations and BVC activations
        # This is the competitive learning rule from Equation (3.3)
        if self.enable_ojas and np.any(self.place_cell_activations):
            # Compute the weight update according to Oja's rule (Equation 3.3)
            weight_update = self.tau * (
                self.place_cell_activations[:, np.newaxis]
                * (
                    self.bvc_activations[np.newaxis, :]
                    - (1 / self.alpha_pb)
                    * self.place_cell_activations[:, np.newaxis]
                    * self.w_in
                )
            )
            # Update the input weights from BVCs to place cells
            self.w_in.assign_add(weight_update)

    def reset_activations(self):
        """
        Resets the place cell activations and related variables to zero.
        """
        self.place_cell_activations *= 0
        self.activation_update *= 0
        self.place_cell_trace = None

    def preplay(self, direction, num_steps=1):
        """
        Simulates the exploitation or preplay of place cell activations based on the recurrent weights.
        This is used to predict future states without actual movement.

        Parameters:
        - direction (int): The index of the head direction in which to exploit the recurrent weights.
        - num_steps (int): Number of exploitation steps to perform (simulates looking ahead).

        Returns:
        - place_cell_activations (tf.Tensor): The updated place cell activations after exploitation.
        """
        # Copy the current place cell activations
        # Shape: (num_pc,)
        place_cell_activations = tf.identity(self.place_cell_activations)

        # Iterate to update the place cell activations
        for step in range(num_steps):
            # Store previous activations
            previous_activations = tf.identity(place_cell_activations)

            # Compute new activations based on recurrent weights and previous activations
            # The recurrent weights are modulated by the specified head direction
            place_cell_activations = tf.tanh(
                tf.nn.relu(
                    tf.tensordot(
                        tf.cast(self.w_rec_tripartite[direction], tf.float32),
                        previous_activations,
                        axes=1,
                    )
                    - previous_activations
                )
            )

        # Return the updated place cell activations
        return place_cell_activations

    def __getitem__(self, index):
        """
        Retrieves the place cell activation at the specified index.
        Example: activation = place_cell_layer[5]  # Activation of the place cell at index 5

        Parameters:
        - index (int): The index of the place cell activation to retrieve.

        Returns:
        - activation (float): The activation value of the place cell at the specified index.
        """
