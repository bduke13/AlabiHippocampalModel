import numpy as np
import torch
from numpy.random import default_rng
from typing import Optional

# Set a fixed seed similar to tf.random.set_seed(5)
torch.manual_seed(5)


class PlaceCellLayer:
    """Model a layer of place cells receiving input from Boundary Vector Cells.

    Place cells develop spatially localized receptive fields (place fields) through
    competitive learning and synaptic plasticity.

    This implementation is based on the model described in Chapter 3 of the
    dissertation, specifically Equations (3.2a), (3.2b), and (3.3).
    """

    def __init__(
        self,
        bvc_layer,
        num_pc: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.25,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
    ):
        """Initialize the Place Cell Layer.

        Args:
            bvc_layer: The BVC layer used as input to place cell activations.
            num_pc: Number of place cells in the layer.
            timestep: Time step for simulation/learning updates in milliseconds.
            n_hd: Number of head direction cells.
            enable_ojas: Enable weight updates via competition.
            enable_stdp: Enable tripartite synapse weight updates via Spike-Timing-Dependent Plasticity.
            w_in_init_ratio: What proportion of the weights of BVC -> PCN are active initially
            device: Which device to place the tensors on (e.g., "cpu" or "cuda").
            dtype: PyTorch data type (e.g., torch.float32).
        """
        # Set up random generator for binomial initialization
        rng = default_rng()

        self.device = device
        self.dtype = dtype

        # Number of place cells
        self.num_pc = num_pc

        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvc_layer = bvc_layer

        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvc_layer.num_bvc

        # Input weight matrix connecting place cells to BVCs
        # Shape: (num_pc, num_bvc)
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, self.num_bvc))
        w_in_init = torch.tensor(w_in_init, dtype=self.dtype, device=self.device)
        # We wrap in nn.Parameter so the weights can be learnable if needed
        self.w_in = torch.nn.Parameter(w_in_init, requires_grad=False)

        # Recurrent weight matrix for head direction and place cell interactions
        # Shape: (n_hd, num_pc, num_pc)
        self.w_rec_tripartite = torch.zeros(
            (n_hd, num_pc, num_pc), dtype=self.dtype, device=self.device
        )

        # Activation values for place cells
        # Shape: (num_pc,)
        self.place_cell_activations = torch.zeros(
            num_pc, dtype=self.dtype, device=self.device
        )

        # Time constant for updating place cell activations
        self.tau = timestep / 1000.0  # Convert timestep to seconds

        # Activation values for boundary vector cells (BVCs)
        # Shape: (num_bvc,)
        self.bvc_activations = torch.zeros(
            self.num_bvc, dtype=self.dtype, device=self.device
        )

        # Coefficient to modify effect of place cell recurrent inhibition (Γ_pp in Equation 3.2a)
        self.gamma_pp = gamma_pp

        # Coefficient to modify effect of boundary vector cell afferent inhibition (Γ_pb in Equation 3.2a)
        self.gamma_pb = gamma_pb

        # Time constant for the membrane potential dynamics of place cells (τ_p in Equation 3.2a)
        self.tau_p = 0.5

        # Normalization factor for synaptic weight updates (α_pb in Equation 3.3)
        self.alpha_pb = np.sqrt(0.5)

        # Initial weights for the input connections from BVCs to place cells
        self.initial_w_in = torch.clone(self.w_in.data)

        # Temporary variable for the current activation update step
        # Shape: (num_pc,)
        self.activation_update = torch.zeros_like(
            self.place_cell_activations, dtype=self.dtype, device=self.device
        )

        # Head direction modulation (if applicable)
        self.head_direction_modulation = None

        # Boundary cell activation values (if any boundary cells are used)
        # Shape: (n_hd, num_pc)
        self.boundary_cell_activations = torch.zeros(
            (n_hd, num_pc), dtype=self.dtype, device=self.device
        )

        # Trace of place cell activations for eligibility tracking
        self.place_cell_trace = torch.zeros_like(self.place_cell_activations)

        # Trace of head direction cells for eligibility tracking
        # Shape: (n_hd, 1, 1)
        self.hd_cell_trace = torch.zeros(
            (n_hd, 1, 1), dtype=torch.float64, device=self.device
        )

        # Enables/disables updating weights to spread place cells through environment via competition
        self.enable_ojas = enable_ojas

        # Enables/disables updating weights in the tripartite synapses to track adjacencies between cells
        self.enable_stdp = enable_stdp

    def get_place_cell_activations(
        self,
        distances: torch.Tensor,
        angles: torch.Tensor,
        hd_activations: Optional[torch.Tensor] = None,
        collided: bool = False,
    ):
        """Compute place cell activations from BVC and head direction inputs.

        Args:
            distances: 1D tensor of distance readings (from the LiDAR).
            angles: 1D tensor of corresponding angles for each distance point.
            hd_activations: 1D tensor of head direction cell activations.
            collided: Whether the agent has collided with an obstacle.
        """
        # Ensure distances and angles are tensors
        if isinstance(distances, np.ndarray):
            distances = torch.tensor(distances, dtype=self.dtype, device=self.device)
        if isinstance(angles, np.ndarray):
            angles = torch.tensor(angles, dtype=self.dtype, device=self.device)

        # Compute BVC activations based on distances and angles
        self.bvc_activations = self.bvc_layer.get_bvc_activation(
            distances=distances, angles=angles
        )

        # Compute afferent excitation (BVC-to-place cell input)
        afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)

        # Compute inhibitory terms
        afferent_inhibition = self.gamma_pb * torch.sum(self.bvc_activations)
        recurrent_inhibition = self.gamma_pp * torch.sum(self.place_cell_activations)

        # Update activation based on the model equation
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )

        # Apply activation function
        self.place_cell_activations = torch.tanh(torch.relu(self.activation_update))

        # Handle STDP updates if enabled and no collision occurred
        if self.enable_stdp and not collided and torch.any(self.place_cell_activations > 0):
            if self.place_cell_trace is None:
                self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (self.tau / 3) * (
                self.place_cell_activations - self.place_cell_trace
            )

            if hd_activations is not None:
                hd_activations = torch.nan_to_num(hd_activations)
                hd_activations = hd_activations.unsqueeze(1).unsqueeze(2)

                pc_act_mat = torch.ger(self.place_cell_activations, self.place_cell_trace)
                pc_trace_mat = torch.ger(self.place_cell_trace, self.place_cell_activations)
                self.w_rec_tripartite += hd_activations.to(self.device) * (pc_act_mat.to(self.device) - pc_trace_mat.to(self.device))

        # Oja's rule for weight updates (if enabled)
        if self.enable_ojas and torch.any(self.place_cell_activations > 0):
            pc_activations_col = self.place_cell_activations.unsqueeze(1)
            bvc_activations_row = self.bvc_activations.unsqueeze(0)
            weight_update = self.tau * (
                pc_activations_col
                * (bvc_activations_row - (1 / self.alpha_pb) * pc_activations_col * self.w_in)
            )
            with torch.no_grad():
                self.w_in += weight_update

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations.zero_()
        self.activation_update.zero_()
        self.place_cell_trace = None  # As in original code

    def preplay(self, direction: int, num_steps: int = 1) -> torch.Tensor:
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
        place_cell_activations = self.place_cell_activations.clone()

        # Iterate to update the place cell activations
        for _ in range(num_steps):
            # Store previous activations
            previous_activations = place_cell_activations.clone()

            # Compute new activations based on recurrent weights and previous activations
            # The recurrent weights are modulated by the specified head direction
            # shape: (num_pc,)
            # w_rec_tripartite[direction] shape: (num_pc, num_pc)
            # previous_activations shape: (num_pc,)
            # torch.matmul -> (num_pc,)
            updated = torch.matmul(
                self.w_rec_tripartite[direction], previous_activations
            )

            # Subtractive term: - previous_activations
            updated = updated - previous_activations

            # Apply ReLU and then tanh
            place_cell_activations = torch.tanh(torch.relu(updated))

        return place_cell_activations
