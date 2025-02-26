import torch
from numpy.random import default_rng
from typing import Optional
from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

class PlaceCellLayerWithGrid:
    def __init__(
        self,
        bvc_layer: BoundaryVectorCellLayer,
        num_pc: int = 500,
        num_grid_cells: int = 400,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.25,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
        tau: float = 0.005,
        alpha_pb: float = 10,
        w_max: float = 2,
        w_min: float = 0,
    ):
        """Initialize the place cell layer with BVC and grid cell inputs.

        Args:
            bvc_layer (BoundaryVectorCellLayer): BVC layer instance.
            num_pc (int): Number of place cells.
            num_grid_cells (int): Number of grid cells.
            timestep (int): Simulation timestep in ms.
            n_hd (int): Number of head direction cells.
            enable_ojas (bool): Enable Oja's rule learning.
            enable_stdp (bool): Enable STDP learning.
            w_in_init_ratio (float): Initial connection probability for weights.
            device (torch.device): Computation device.
            dtype (torch.dtype): Data type for tensors.
            gamma_pp (float): Place-to-place inhibition strength.
            gamma_pb (float): Place-to-BVC/grid inhibition strength.
            tau (float): Learning rate for Oja's rule.
            alpha_pb (float): Oja's rule normalization factor.
            w_max (float): Maximum weight value.
            w_min (float): Minimum weight value.
        """
        rng = default_rng()
        self.device = device
        self.dtype = dtype
        self.num_pc = num_pc
        self.bvc_layer = bvc_layer
        self.num_bvc = bvc_layer.num_bvc
        self.num_grid_cells = num_grid_cells
        self.timestep = timestep
        self.n_hd = n_hd
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp
        self.gamma_pp = gamma_pp
        self.gamma_pb = gamma_pb
        self.tau = tau
        self.alpha_pb = alpha_pb
        self.w_max = w_max
        self.w_min = w_min

        # Initialize weights from BVCs to place cells
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, self.num_bvc))
        self.w_in = torch.nn.Parameter(
            torch.tensor(w_in_init, dtype=dtype, device=device), requires_grad=False
        )
        # Initialize weights from grid cells to place cells
        w_grid_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, num_grid_cells))
        self.w_grid = torch.nn.Parameter(
            torch.tensor(w_grid_init, dtype=dtype, device=device), requires_grad=False
        )
        # Recurrent weights (place-to-place)
        self.w_rec = torch.nn.Parameter(
            torch.zeros((num_pc, num_pc), dtype=dtype, device=device), requires_grad=False
        )
        # Head direction modulation
        self.w_hd = torch.ones((num_pc, n_hd), dtype=dtype, device=device) / n_hd
        # Place cell activations
        self.place_cell_activations = torch.zeros(num_pc, dtype=dtype, device=device)
        self.bvc_activations = torch.zeros(self.num_bvc, dtype=dtype, device=device)
        # Short-term memory for STDP (assuming same structure as original)
        self.stdp_short_term_pc = torch.zeros(num_pc, dtype=dtype, device=device)

    def get_place_cell_activations(
        self,
        distances: torch.Tensor,
        hd_activations: torch.Tensor,
        grid_activations: torch.Tensor,
        collided: bool = False,
    ) -> torch.Tensor:
        """Compute place cell activations with BVC and grid cell inputs.

        Args:
            distances (torch.Tensor): Boundary distances from BVC layer.
            hd_activations (torch.Tensor): Head direction activations.
            grid_activations (torch.Tensor): Grid cell activations.
            collided (bool): Whether the robot has collided.

        Returns:
            torch.Tensor: Place cell activations.
        """
        self.bvc_activations = self.bvc_layer.get_bvc_activation(distances)
        # Combine excitations from BVCs and grid cells
        """
        afferent_excitation = (
            torch.matmul(self.w_in, self.bvc_activations) +
            torch.matmul(self.w_grid, grid_activations)
        )
        """
        afferent_excitation = (
            torch.matmul(self.w_in, self.bvc_activations) +
            torch.matmul(self.w_grid, grid_activations)
        )
        # Recurrent excitation
        recurrent_excitation = torch.matmul(self.w_rec, self.place_cell_activations)
        # Head direction modulation
        hd_modulation = torch.matmul(self.w_hd, hd_activations)
        # Total excitation with inhibition
        total_excitation = (
            afferent_excitation +
            recurrent_excitation -
            self.gamma_pp * torch.sum(self.place_cell_activations)
        ) * hd_modulation
        # Update activations (assuming leaky integration as in original)
        self.place_cell_activations = (
            (1 - self.timestep / 1000) * self.place_cell_activations +
            (self.timestep / 1000) * torch.relu(total_excitation)
        )

        # Apply Oja's rule for both w_in and w_grid if enabled
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            pc_activations_col = self.place_cell_activations.unsqueeze(1)
            # Update BVC weights
            bvc_activations_row = self.bvc_activations.unsqueeze(0)
            weight_update_in = self.tau * (
                pc_activations_col * (
                    bvc_activations_row -
                    (1 / self.alpha_pb) * pc_activations_col * self.w_in
                )
            )
            with torch.no_grad():
                self.w_in += weight_update_in
                self.w_in.clamp_(self.w_min, self.w_max)
            # Update grid cell weights
            """
            grid_activations_row = grid_activations.unsqueeze(0)
            weight_update_grid = self.tau * (
                pc_activations_col * (
                    grid_activations_row -
                    (1 / self.alpha_pb) * pc_activations_col * self.w_grid
                )
            )
            with torch.no_grad():
                self.w_grid += weight_update_grid
                self.w_grid.clamp_(self.w_min, self.w_max)
            """

        return self.place_cell_activations

    def reset_activations(self):
        """Reset place cell activations to zero."""
        self.place_cell_activations.zero_()
        self.bvc_activations.zero_()
        self.stdp_short_term_pc.zero_()

    # Assume other methods (e.g., STDP) remain unchanged from PlaceCellLayer