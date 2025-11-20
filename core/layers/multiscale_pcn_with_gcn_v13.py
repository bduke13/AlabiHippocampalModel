import numpy as np
import torch
from numpy.random import default_rng
from typing import Optional

# Set a fixed seed similar to tf.random.set_seed(5)
torch.manual_seed(5)


class MultiscalePlaceCellWithGrid:
    """Model a layer of place cells receiving input from both Boundary Vector Cells and Grid Cells.

    Place cells develop spatially localized receptive fields (place fields) through
    competitive learning and synaptic plasticity.

    This implementation extends the original model to incorporate grid cell inputs with
    a controllable percentage of influence between BVCs and grid cells, while supporting
    the multiscale approach.
    """
    def __init__(
        self,
        bvc_layer,
        num_pc: int = 200,
        num_grid_cells: int = 0,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.2,
        w_grid_init_ratio: float = 0.2,
        # Optional GC->PC init strategy
        w_grid_init_strategy: str = 'global',  # 'global' | 'balanced_modules'
        gc_num_modules: int = None,
        gc_cells_per_module: int = None,
        grid_influence: float = 0.5,  # Default to 50% grid influence
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,  # Parameter for grid cell inhibition
        # Oja's learning normalization parameters
        alpha_pb: float = None,  # Weight decay factor for BVC->PC learning (default: sqrt(0.5))
        alpha_pg: float = None,  # Weight decay factor for GC->PC learning (default: sqrt(0.5))
        # Correlation-based weighting parameters
        enable_correlation_weighting: bool = True,
        correlation_window: int = 100,
        correlation_update_freq: int = 10,
        correlation_scaling: float = 2.0,
        min_correlation_weight: float = 0.1,
        correlation_threshold: float = 0.01,
        # Activation threshold parameters
        activation_threshold: float = 0.15,  # Minimum activation for place cell firing
        enable_wall_aware_threshold: bool = False,  # Dynamic threshold based on wall proximity
        wall_threshold: float = 0.3,  # Higher threshold near walls
        wall_distance_cutoff: float = 1,  # Distance (m) where wall effects start
    ):
        """Initialize the Place Cell Layer with Grid cell input.

        Args:
            bvc_layer: The BVC layer used as input to place cell activations.
            num_pc: Number of place cells in the layer.
            num_grid_cells: Number of grid cells providing input.
            timestep: Time step for simulation/learning updates in milliseconds.
            n_hd: Number of head direction cells.
            enable_ojas: Enable weight updates via competition.
            enable_stdp: Enable tripartite synapse weight updates via Spike-Timing-Dependent Plasticity.
            w_in_init_ratio: What proportion of BVC->PC connections are active initially.
            w_grid_init_ratio: What proportion of GC->PC connections are active initially.
            w_grid_init_strategy: 'global' for Bernoulli over all GCs; 'balanced_modules' to distribute
                                  initial GC connections evenly across modules.
            gc_num_modules: Number of GC modules (required if using 'balanced_modules').
            gc_cells_per_module: Cells per GC module (required if using 'balanced_modules').
            grid_influence: Percentage (0.0 to 1.0) of grid cell influence on place cells.
                           0.0 means only BVC input, 1.0 means only grid cell input.
            device: Which device to place the tensors on (e.g., "cpu" or "cuda").
            dtype: PyTorch data type (e.g., torch.float32).
            gamma_pp: Coefficient for place cell recurrent inhibition.
            gamma_pb: Coefficient for boundary vector cell afferent inhibition.
            gamma_pg: Coefficient for grid cell afferent inhibition.
            alpha_pb: Weight decay factor for BVC->PC learning (Oja's rule). Lower = stronger decay/more selective.
            alpha_pg: Weight decay factor for GC->PC learning (Oja's rule). Lower = stronger decay/more selective.
            enable_correlation_weighting: Whether to enable correlation-based connection weighting.
            correlation_window: Window size for correlation tracking.
            correlation_update_freq: How often to update correlation matrix.
            correlation_scaling: Scaling factor for correlation-based weights.
            min_correlation_weight: Minimum weight for weakly correlated connections.
            correlation_threshold: Threshold for considering activations in correlation.
            activation_threshold: Minimum activation required for place cell firing (prevents weak activations).
            enable_wall_aware_threshold: If True, increase threshold near walls to prevent cross-wall learning.
            wall_threshold: Higher threshold used when robot is very close to walls.
            wall_distance_cutoff: Distance (meters) at which wall-aware thresholding begins.
        """
        # Set up random generator for binomial initialization
        rng = default_rng()

        self.device = device
        self.dtype = dtype
        self.grid_gain = 0.5
        self.n_hd = n_hd

        # Number of place cells
        self.num_pc = num_pc

        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvc_layer = bvc_layer

        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvc_layer.num_bvc

        # Number of Grid Cells
        self.num_grid_cells = num_grid_cells

        # Grid influence parameter (0.0 to 1.0)
        self.grid_influence = grid_influence

        # Correlation-based weighting parameters
        self.enable_correlation_weighting = enable_correlation_weighting
        self.correlation_window = correlation_window
        self.correlation_update_freq = correlation_update_freq
        self.correlation_scaling = correlation_scaling
        self.min_correlation_weight = min_correlation_weight
        self.correlation_threshold = correlation_threshold

        # Activation threshold parameters
        self.activation_threshold = activation_threshold
        self.enable_wall_aware_threshold = enable_wall_aware_threshold
        self.wall_threshold = wall_threshold
        self.wall_distance_cutoff = wall_distance_cutoff

        # Connection decay parameters (from v21)
        self.enable_connection_decay = True
        self.connection_decay_rate = 0.0001

        # Adaptive STDP learning rate parameters (from v21)
        self.enable_adaptive_stdp = False
        self.adaptive_initial_lr = 0.1  # Starting learning rate
        self.adaptive_final_lr = 0.03    # Final (minimum) learning rate
        self.adaptive_decay_rate = 3.0   # Rate of exponential decay
        self.stdp_learning_rate = 0.01   # Fixed rate when adaptive disabled
        self.tau_hd = 0.1                # HD eligibility trace time constant

        # Performance optimization - cache connection strengths
        self.connection_strength_cache = torch.zeros(n_hd, dtype=dtype, device=device)
        self.strength_update_counter = 0
        self.strength_update_frequency = 10  # Update strengths every 10 STDP steps

        # Correlation tracking data structures
        self.activation_history = []
        self.correlation_matrix = torch.ones(num_pc, num_pc, dtype=dtype, device=device) * 0.5
        self.correlation_step_counter = 0

        # Input weight matrix connecting place cells to BVCs
        # Shape: (num_pc, num_bvc)
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, self.num_bvc))
        w_in_init = torch.tensor(w_in_init, dtype=self.dtype, device=self.device)
        # We wrap in nn.Parameter so the weights can be learnable if needed
        self.w_in = torch.nn.Parameter(w_in_init, requires_grad=False)

        # Input weight matrix connecting place cells to Grid Cells
        # Shape: (num_pc, num_grid_cells)
        if num_grid_cells > 0:
            use_balanced = (
                isinstance(w_grid_init_strategy, str)
                and w_grid_init_strategy.lower() == 'balanced_modules'
                and isinstance(gc_num_modules, int) and gc_num_modules is not None and gc_num_modules > 0
                and isinstance(gc_cells_per_module, int) and gc_cells_per_module is not None and gc_cells_per_module > 0
            )

            if use_balanced and (gc_num_modules * gc_cells_per_module >= self.num_grid_cells):
                # Balanced per-module initialization to encourage diverse module inputs per PC
                total_gc = self.num_grid_cells
                K = int(round(w_grid_init_ratio * total_gc))
                K = max(0, min(K, total_gc))
                w_grid_np = np.zeros((num_pc, total_gc), dtype=np.int8)
                base_q = (K // gc_num_modules)
                remainder = K - base_q * gc_num_modules

                for pc_idx in range(num_pc):
                    for m in range(gc_num_modules):
                        m_start = m * gc_cells_per_module
                        if m_start >= total_gc:
                            break
                        m_size = min(gc_cells_per_module, total_gc - m_start)
                        quota = base_q + (1 if m < remainder else 0)
                        quota = min(quota, m_size)
                        if quota > 0:
                            chosen = rng.choice(m_size, size=quota, replace=False)
                            w_grid_np[pc_idx, m_start + chosen] = 1
                w_grid_init = torch.tensor(w_grid_np, dtype=self.dtype, device=self.device)
            else:
                # Default global Bernoulli mask
                w_grid_init = rng.binomial(n=1, p=w_grid_init_ratio, size=(num_pc, self.num_grid_cells))
                w_grid_init = torch.tensor(w_grid_init, dtype=self.dtype, device=self.device)

            # We wrap in nn.Parameter so the weights can be learnable if needed
            self.w_grid = torch.nn.Parameter(w_grid_init, requires_grad=False)
        else:
            self.w_grid = None

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

        ##################################################################################################
        self.eta_stdp = 0.3 # .3
        ##################################################################################################

        # Activation values for boundary vector cells (BVCs)
        # Shape: (num_bvc,)
        self.bvc_activations = torch.zeros(
            self.num_bvc, dtype=self.dtype, device=self.device
        )

        # Activation values for grid cells
        # Shape: (num_grid_cells,)
        if num_grid_cells > 0:
            self.grid_cell_activations = torch.zeros(
                self.num_grid_cells, dtype=self.dtype, device=self.device
            )
        else:
            self.grid_cell_activations = None

        # Coefficient to modify effect of place cell recurrent inhibition (Γ_pp in Equation 3.2a)
        self.gamma_pp = gamma_pp

        # Coefficient to modify effect of boundary vector cell afferent inhibition (Γ_pb in Equation 3.2a)
        self.gamma_pb = gamma_pb

        # Coefficient to modify effect of grid cell afferent inhibition
        self.gamma_pg = gamma_pg

        # Time constant for the membrane potential dynamics of place cells (τ_p in Equation 3.2a)
        self.tau_p = 0.5

        # Normalization factor for synaptic weight updates (α_pb in Equation 3.3)
        # Use provided values or default to sqrt(0.5) ≈ 0.707
        self.alpha_pb = alpha_pb if alpha_pb is not None else np.sqrt(0.5)

        # Normalization factor for grid cell synaptic weight updates
        # Use provided values or default to sqrt(0.5) ≈ 0.707
        self.alpha_pg = alpha_pg if alpha_pg is not None else np.sqrt(0.5)

        # Initial weights for the input connections from BVCs to place cells
        self.initial_w_in = torch.clone(self.w_in.data)

        # Initial weights for the input connections from Grid Cells to place cells
        if self.w_grid is not None:
            self.initial_w_grid = torch.clone(self.w_grid.data)

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
        distances: np.ndarray,
        grid_activations: Optional[torch.Tensor] = None,
        hd_activations: Optional[np.ndarray] = None,
        collided: bool = False,
    ):
        """Compute place cell activations from BVC, grid cell, and head direction inputs.

        Args:
            distances: 1D NumPy array of distance readings (to be fed into the BVC layer).
            grid_activations: 1D tensor of grid cell activations.
            hd_activations: 1D NumPy array of head direction cell activations.
            collided: Whether the agent has collided with an obstacle.
        """
        # Apply connection decay at the beginning of each timestep (from v21)
        if self.enable_stdp:
            self.apply_connection_decay()

        # Convert distances to torch tensor if not already
        if isinstance(distances, torch.Tensor):
            distances_torch = (
                distances.clone().detach().to(dtype=self.dtype, device=self.device)
            )
        else:
            distances_torch = torch.tensor(
                distances, dtype=self.dtype, device=self.device
            )

        # Process head direction activations if provided
        if hd_activations is not None:
            if isinstance(hd_activations, torch.Tensor):
                hd_activations_torch = (
                    hd_activations.clone()
                    .detach()
                    .to(dtype=self.dtype, device=self.device)
                )
            else:
                hd_activations_torch = torch.as_tensor(
                    hd_activations, dtype=self.dtype, device=self.device
                )
        else:
            hd_activations_torch = None

        # Update HD eligibility traces (from v21)
        self.update_hd_eligibility_trace(hd_activations_torch)

        # Compute BVC activations based on the input distances
        self.bvc_activations = self.bvc_layer.get_bvc_activation(
            distances=distances_torch
        )

        # Store grid cell activations if provided
        if grid_activations is not None and self.num_grid_cells > 0:
            self.grid_cell_activations = grid_activations.to(dtype=self.dtype, device=self.device)

        # Compute the BVC input to place cells by taking the dot product of input weights and BVC activations
        # Afferent excitation term: ∑_j W_ij^{pb} v_j^b
        bvc_afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)

        # Compute the Grid Cell input to place cells (if grid cells are used)
        grid_afferent_excitation = torch.zeros_like(bvc_afferent_excitation)
        if self.grid_cell_activations is not None and self.w_grid is not None:
            grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)

        # Combine BVC and Grid Cell inputs based on the grid_influence parameter
        if self.grid_influence == 0.0:
            # Only use BVC input
            afferent_excitation = bvc_afferent_excitation
        elif self.grid_influence == 1.0:
            # Only use Grid Cell input
            afferent_excitation = grid_afferent_excitation
        else:
            # Combine BVC and Grid Cell inputs based on grid_influence
            afferent_excitation = (1.0 - self.grid_influence) * bvc_afferent_excitation + \
                                 self.grid_influence * grid_afferent_excitation

        # Compute total BVC activity for afferent inhibition
        # Afferent inhibition term from BVCs: Γ^{pb} ∑_j v_j^b
        bvc_afferent_inhibition = self.gamma_pb * torch.sum(self.bvc_activations)

        # Compute total Grid Cell activity for afferent inhibition (if grid cells are used)
        grid_afferent_inhibition = 0.0
        if self.grid_cell_activations is not None:
            grid_afferent_inhibition = self.gamma_pg * torch.sum(self.grid_cell_activations)

        # Combine BVC and Grid Cell inhibition based on the grid_influence parameter
        if self.grid_influence == 0.0:
            # Only use BVC inhibition
            afferent_inhibition = bvc_afferent_inhibition
        elif self.grid_influence == 1.0:
            # Only use Grid Cell inhibition
            afferent_inhibition = grid_afferent_inhibition
        else:
            # Combine BVC and Grid Cell inhibition based on grid_influence
            afferent_inhibition = (1.0 - self.grid_influence) * bvc_afferent_inhibition + \
                                 self.grid_influence * grid_afferent_inhibition

        # Compute total place cell activity for recurrent inhibition
        # Recurrent inhibition term: Γ^{pp} ∑_j v_j^p
        recurrent_inhibition = self.gamma_pp * torch.sum(self.place_cell_activations)

        # Update the activation_update variable
        # Equation (3.2a): τ_p (ds_i^p/dt) = -s_i^p + afferent_excitation - afferent_inhibition - recurrent_inhibition
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )

        # Apply ReLU then tanh to compute new place cell activations
        # Equation (3.2b): v_i^p = tanh([ψ s_i^p]_+)
        # Here, ψ is implicitly set to 1
        # Apply hard threshold to prevent weak activations from forming place cells

        # Determine effective threshold (wall-aware or fixed)
        if self.enable_wall_aware_threshold and distances_torch is not None:
            # Wall-aware threshold: higher near walls to prevent cross-wall learning
            min_wall_dist = torch.min(distances_torch).item()

            if min_wall_dist < self.wall_distance_cutoff:
                # Interpolate between base and wall threshold based on distance
                alpha = min_wall_dist / self.wall_distance_cutoff
                effective_threshold = self.wall_threshold * (1 - alpha) + self.activation_threshold * alpha
            else:
                # Far from walls: use base threshold
                effective_threshold = self.activation_threshold
        else:
            # Fixed threshold mode
            effective_threshold = self.activation_threshold

        # Apply threshold and activation function
        thresholded = torch.relu(self.activation_update)
        self.place_cell_activations = torch.tanh(thresholded)

        # Update correlation tracking
        if self.enable_correlation_weighting:
            self.update_correlation_tracking(self.place_cell_activations)

        # Check STDP updates if enabled and no collision occurred
        if (
            self.enable_stdp
            and torch.any(self.place_cell_activations != 0)
            and not collided
        ):
            # Update eligibility trace for place cells
            if self.place_cell_trace is None:
                self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (self.tau / 3) * (
                self.place_cell_activations - self.place_cell_trace
            )

            # Update recurrent weights for place cell interactions modulated by head direction
            # STDP-like update with adaptive learning rates (from v21)
            if hd_activations_torch is not None:
                # Use HD eligibility traces instead of current HD activations
                hd_contrib = self.hd_cell_trace  # Shape: (n_hd, 1, 1)

                # Compute raw STDP update
                pc_act_mat = torch.ger(self.place_cell_activations, self.place_cell_trace)
                pc_trace_mat = torch.ger(self.place_cell_trace, self.place_cell_activations)

                # Raw update for all HD slices (before learning rate scaling)
                raw_update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)

                # Apply correlation-based weighting to raw update
                weighted_update_rec = self.apply_correlation_weighting_to_stdp(raw_update_rec)

                # Apply adaptive learning rate scaling per direction
                if self.enable_adaptive_stdp:
                    # Update connection strengths periodically for performance
                    self.strength_update_counter += 1
                    if self.strength_update_counter % self.strength_update_frequency == 0:
                        for direction in range(self.n_hd):
                            self.connection_strength_cache[direction] = self.compute_connection_strength(
                                self.w_rec_tripartite[direction]
                            )

                    adaptive_scaled_update = torch.zeros_like(weighted_update_rec)

                    for direction in range(self.n_hd):
                        # Use cached connection strength for performance
                        connection_strength = self.connection_strength_cache[direction].item()

                        # Get adaptive learning rate for this direction
                        adaptive_lr = self.get_adaptive_learning_rate(connection_strength)

                        # Apply direction-specific learning rate
                        adaptive_scaled_update[direction] = adaptive_lr * weighted_update_rec[direction]

                    scaled_update_rec = adaptive_scaled_update
                else:
                    # Apply fixed learning rate scaling
                    scaled_update_rec = self.eta_stdp * weighted_update_rec

                # Apply final update
                self.w_rec_tripartite += scaled_update_rec.type(self.dtype)

        # Check Oja's rule for input weights if enabled
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            # Apply Oja's rule to BVC weights if not fully using grid cells
            if self.grid_influence < 1.0:
                # (num_pc, 1)
                pc_activations_col = self.place_cell_activations.unsqueeze(1)
                # (1, num_bvc)
                bvc_activations_row = self.bvc_activations.unsqueeze(0)

                # Modified Oja's rule with grid_influence factor
                # The weight update is scaled by (1.0 - grid_influence) to account for the reduced influence of BVCs
                weight_update_bvc = self.tau * (
                    pc_activations_col
                    * (
                        bvc_activations_row
                        - (1 / self.alpha_pb) * pc_activations_col * self.w_in
                    )
                ) * (1.0 - self.grid_influence)

                # In PyTorch, we can update the data directly or reassign
                with torch.no_grad():
                    self.w_in += weight_update_bvc

            # Apply Oja's rule to Grid Cell weights if using grid cells
            if self.grid_influence > 0.0 and self.grid_cell_activations is not None and self.w_grid is not None:
                # (num_pc, 1)
                pc_activations_col = self.place_cell_activations.unsqueeze(1)
                # (1, num_grid_cells)
                grid_activations_row = self.grid_cell_activations.unsqueeze(0)

                # Oja's rule for grid cell weights
                # The weight update is scaled by grid_influence to account for the influence of grid cells
                weight_update_grid = self.tau * (
                    pc_activations_col
                    * (
                        grid_activations_row
                        - (1 / self.alpha_pg) * pc_activations_col * self.w_grid
                    )
                ) * self.grid_influence

                # Update grid cell weights
                with torch.no_grad():
                    self.w_grid += weight_update_grid

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations.zero_()
        self.activation_update.zero_()
        self.place_cell_trace = None  # As in original code

        # Reset HD eligibility traces (from v21)
        self.hd_cell_trace.zero_()

        # Reset adaptive learning cache (from v21)
        if hasattr(self, 'connection_strength_cache'):
            self.connection_strength_cache.zero_()
            self.strength_update_counter = 0

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

    def preplay_from_state(self, starting_activations: torch.Tensor, direction: int, num_steps: int = 1) -> torch.Tensor:
        """Perform preplay starting from an arbitrary place cell activation state.

        Args:
            starting_activations: Place cell activations to start from (shape: num_pc)
            direction: Index of head direction for exploiting recurrent weights (0-7)
            num_steps: Number of preplay steps to simulate

        Returns:
            Updated place cell activations after preplay
        """
        # Copy starting activations
        place_cell_activations = starting_activations.clone()

        # Iterate preplay steps
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()

            # Use directional connections from the specified HD slice
            updated = torch.matmul(
                self.w_rec_tripartite[direction], previous_activations
            )

            updated = updated - previous_activations
            place_cell_activations = torch.tanh(torch.relu(updated))

        return place_cell_activations

    def _compute_learned_turn_probabilities_batched(self, activations: torch.Tensor, current_directions: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute turn probabilities for batched trajectories using learned W_rec weights.

        Args:
            activations: Current place cell activations (shape: batch_size, num_pc)
            current_directions: Current direction for each trajectory (shape: batch_size,)
            temperature: Softmax temperature (higher = more random)

        Returns:
            turn_probs: Probabilities for each turn option [-1, 0, +1] (shape: batch_size, 3)
        """
        batch_size = activations.shape[0]
        turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)

        # Compute next directions for all turn options
        # Shape: (batch_size, 3) where 3 is [left, straight, right]
        next_dirs = (current_directions.unsqueeze(1) + turn_options.unsqueeze(0)) % self.n_hd

        # Extract w_rec for all next directions
        # w_rec_tripartite: (n_hd, num_pc, num_pc)
        # next_dirs: (batch_size, 3)
        # We need to flatten, index, then reshape
        next_dirs_flat = next_dirs.flatten()  # (batch_size * 3,)
        w_rec_flat = self.w_rec_tripartite[next_dirs_flat]  # (batch_size * 3, num_pc, num_pc)
        w_rec_batch = w_rec_flat.view(batch_size, 3, self.num_pc, self.num_pc)  # (batch_size, 3, num_pc, num_pc)

        # Compute transition strengths for each turn option
        # activations: (batch_size, num_pc) -> (batch_size, 1, num_pc, 1) for broadcasting
        activations_expanded = activations.unsqueeze(1).unsqueeze(3)  # (batch_size, 1, num_pc, 1)

        # Batch matrix-vector multiply: (batch_size, 3, num_pc, num_pc) @ (batch_size, 1, num_pc, 1)
        # Result: (batch_size, 3, num_pc, 1)
        transition_activations = torch.matmul(w_rec_batch, activations_expanded)  # (batch_size, 3, num_pc, 1)

        # Sum over place cells to get transition strength for each turn option
        # Shape: (batch_size, 3)
        turn_logits = transition_activations.squeeze(3).sum(dim=2) / temperature

        # Softmax to get probabilities
        turn_probs = torch.softmax(turn_logits, dim=1)  # (batch_size, 3)

        return turn_probs

    def preplay_from_state_batched(self, starting_activations: torch.Tensor, directions: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """Perform batched preplay starting from multiple activation states.

        This is an optimized version that processes multiple trajectories in parallel,
        significantly improving GPU utilization and reducing computation time.

        Args:
            starting_activations: Place cell activations to start from (shape: batch_size, num_pc)
            directions: Direction indices for each trajectory (shape: batch_size,)
            num_steps: Number of preplay steps to simulate for each trajectory

        Returns:
            Updated place cell activations after preplay (shape: batch_size, num_pc)
        """
        # starting_activations: (batch_size, num_pc)
        # directions: (batch_size,) - integer indices
        batch_size = starting_activations.shape[0]

        # Copy starting activations
        place_cell_activations = starting_activations.clone()

        # Iterate preplay steps
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()

            # Extract directional connections for each trajectory in the batch
            # w_rec_tripartite: (n_hd, num_pc, num_pc)
            # directions: (batch_size,) -> index into first dimension
            # Result: (batch_size, num_pc, num_pc)
            w_rec_batch = self.w_rec_tripartite[directions]

            # Batch matrix multiplication
            # w_rec_batch: (batch_size, num_pc, num_pc)
            # previous_activations: (batch_size, num_pc)
            # Need to expand to (batch_size, num_pc, 1) for bmm, then squeeze
            updated = torch.bmm(w_rec_batch, previous_activations.unsqueeze(2)).squeeze(2)

            updated = updated - previous_activations
            place_cell_activations = torch.tanh(torch.relu(updated))

        return place_cell_activations

    def _evaluate_activations_for_reward(self, activations: torch.Tensor) -> float:
        """Helper method to evaluate reward for a given place cell activation pattern.

        This is a placeholder - the actual implementation should use the reward cell network.
        This method gets temporarily overridden in exploit_v2() to use the real RCN.

        Args:
            activations: Place cell activations to evaluate

        Returns:
            float: Estimated reward value
        """
        # Default implementation: sum of activations as a proxy
        return activations.sum().item()

    def multi_step_preplay_constrained_weighted(self, forced_first_direction: int, max_steps: int = 3, decay_factor: float = 0.6, debug: bool = False) -> float:
        """Multi-step preplay constrained to start in a specific direction with exponential step weighting.

        Args:
            forced_first_direction: Direction (0-7) that the first step must take
            max_steps: Number of steps to look ahead
            decay_factor: Exponential decay for step weighting (0.6 = each step worth 60% of previous)
            debug: Whether to print debug information

        Returns:
            float: Best weighted total reward for paths starting in the forced direction
        """
        if debug:
            print(f"    Weighted constrained preplay: forced_dir={forced_first_direction}({forced_first_direction*45}°), steps={max_steps}, decay={decay_factor}")

        current_activations = self.place_cell_activations.clone()
        all_paths_evaluated = []

        def evaluate_path_recursive_weighted(activations, remaining_steps, path_so_far, depth=0, accumulated_reward=0.0):
            """Recursively evaluate paths with exponential step weighting."""

            # Calculate weight for this step (exponential decay)
            step_weight = decay_factor ** depth

            # Evaluate reward at current state and apply step weight
            current_step_reward = self._evaluate_activations_for_reward(activations)
            weighted_step_reward = step_weight * current_step_reward
            total_reward_so_far = accumulated_reward + weighted_step_reward

            if debug and len(path_so_far) <= 2:  # Only debug shorter paths
                print(f"  {'  ' * depth}Step {depth+1}: raw_reward={current_step_reward:.4f}, weight={step_weight:.3f}, weighted={weighted_step_reward:.4f}, total={total_reward_so_far:.4f}")

            if remaining_steps == 0:
                # Base case: no more steps to evaluate
                all_paths_evaluated.append((path_so_far.copy(), total_reward_so_far))
                return total_reward_so_far

            # Determine directions to explore at this step
            if len(path_so_far) == 0:
                # First step: use forced direction only
                directions_to_try = [forced_first_direction]
            else:
                # Subsequent steps: limited branching
                last_direction = path_so_far[-1]
                directions_to_try = [
                    last_direction,                           # Continue straight
                    (last_direction - 1) % self.n_hd,        # Turn counter-clockwise
                    (last_direction + 1) % self.n_hd         # Turn clockwise
                ]

            best_reward_from_here = -float('inf')

            for direction in directions_to_try:
                # Simulate one step in this direction
                next_activations = self.preplay_from_state(activations, direction, num_steps=1)

                # Recursively evaluate the rest of the path with weighted accumulation
                new_path = path_so_far + [direction]
                reward = evaluate_path_recursive_weighted(next_activations, remaining_steps - 1, new_path, depth + 1, total_reward_so_far)

                best_reward_from_here = max(best_reward_from_here, reward)

            return best_reward_from_here

        # Start evaluation from forced direction
        best_reward = evaluate_path_recursive_weighted(current_activations, max_steps, [], 0, 0.0)

        if debug:
            print(f"    Weighted result: best_reward={best_reward:.4f}")

        return best_reward

    def check_preplay_validity_at_depth(self, starting_activations: torch.Tensor, direction: int, depth: int, threshold: float = 0.1) -> bool:
        """Check if place cells remain valid (above threshold) at a given preplay depth.

        Args:
            starting_activations: Initial place cell activations
            direction: Head direction index (0-7)
            depth: Number of preplay steps to simulate
            threshold: Minimum activation value to consider valid

        Returns:
            bool: True if max place cell activation > threshold at this depth
        """
        # Simulate preplay to the specified depth
        activations = self.preplay_from_state(starting_activations, direction, num_steps=depth)

        # Check if any place cell exceeds threshold
        max_activation = torch.max(activations).item()
        return max_activation > threshold

    def determine_valid_preplay_depth(self, max_steps: int = 3, threshold: float = 0.1) -> int:
        """Determine maximum valid preplay depth by checking place cell activations.

        Tests all possible preplay branches at each depth to find the maximum depth
        where place cells still have meaningful activations.

        Args:
            max_steps: Maximum number of steps to test
            threshold: Minimum activation value to consider valid

        Returns:
            int: Maximum valid depth (1 to max_steps), or 0 if no valid preplay possible
        """
        current_activations = self.place_cell_activations.clone()

        # Check if we even have valid place cells at the start
        if torch.max(current_activations).item() <= threshold:
            return 0

        valid_depth = 0

        # Test each depth incrementally
        for depth in range(1, max_steps + 1):
            found_valid = False

            # Test all 8 initial directions
            for initial_direction in range(self.n_hd):
                # For depth 1, just check the single direction
                if depth == 1:
                    if self.check_preplay_validity_at_depth(current_activations, initial_direction, depth, threshold):
                        found_valid = True
                        break
                else:
                    # For depth > 1, need to check branches
                    # Simulate to depth-1 first
                    intermediate_activations = self.preplay_from_state(current_activations, initial_direction, num_steps=depth-1)

                    # Now check the 3 possible branches at this depth
                    # Get the last direction from the path (for simplicity, use initial_direction)
                    last_direction = initial_direction
                    branch_directions = [
                        last_direction,                        # Straight
                        (last_direction - 1) % self.n_hd,     # Counter-clockwise
                        (last_direction + 1) % self.n_hd      # Clockwise
                    ]

                    # Check if any branch has valid place cells
                    for branch_dir in branch_directions:
                        if self.check_preplay_validity_at_depth(intermediate_activations, branch_dir, 1, threshold):
                            found_valid = True
                            break

                    if found_valid:
                        break

            if found_valid:
                valid_depth = depth
            else:
                # No valid place cells at this depth, stop searching
                break

        return valid_depth

    def multi_step_preplay_adaptive_validity(self, forced_first_direction: int, max_steps: int = 3, decay_factor: float = 0.6, activation_threshold: float = 0.1, debug: bool = False) -> tuple:
        """Multi-step preplay with adaptive depth based on place cell validity.

        Similar to multi_step_preplay_constrained_weighted, but determines valid depth
        based on whether place cells remain active during preplay rather than distance.

        Args:
            forced_first_direction: Direction (0-7) that the first step must take
            max_steps: Maximum number of steps to look ahead
            decay_factor: Exponential decay for step weighting
            activation_threshold: Minimum activation to consider place cells valid
            debug: Whether to print debug information

        Returns:
            tuple: (best_weighted_reward, actual_steps_used)
        """
        # First, determine the valid preplay depth
        valid_depth = self.determine_valid_preplay_depth(max_steps, activation_threshold)

        if valid_depth == 0:
            # No valid preplay possible
            if debug:
                print(f"    Adaptive preplay: No valid depth found (all PCs below threshold {activation_threshold})")
            return 0.0, 0

        if debug:
            print(f"    Adaptive preplay: forced_dir={forced_first_direction}({forced_first_direction*45}°), valid_depth={valid_depth}/{max_steps}, threshold={activation_threshold}")

        # Use the existing weighted preplay with the validated depth
        current_activations = self.place_cell_activations.clone()
        all_paths_evaluated = []

        def evaluate_path_recursive_weighted(activations, remaining_steps, path_so_far, depth=0, accumulated_reward=0.0):
            """Recursively evaluate paths with exponential step weighting."""

            # Calculate weight for this step (exponential decay)
            step_weight = decay_factor ** depth

            # Evaluate reward at current state and apply step weight
            current_step_reward = self._evaluate_activations_for_reward(activations)
            weighted_step_reward = step_weight * current_step_reward
            total_reward_so_far = accumulated_reward + weighted_step_reward

            if debug and len(path_so_far) <= 2:
                print(f"  {'  ' * depth}Step {depth+1}: raw_reward={current_step_reward:.4f}, weight={step_weight:.3f}, weighted={weighted_step_reward:.4f}, total={total_reward_so_far:.4f}")

            if remaining_steps == 0:
                # Base case: no more steps to evaluate
                all_paths_evaluated.append((path_so_far.copy(), total_reward_so_far))
                return total_reward_so_far

            # Determine directions to explore at this step
            if len(path_so_far) == 0:
                # First step: use forced direction only
                directions_to_try = [forced_first_direction]
            else:
                # Subsequent steps: limited branching
                last_direction = path_so_far[-1]
                directions_to_try = [
                    last_direction,                           # Continue straight
                    (last_direction - 1) % self.n_hd,        # Turn counter-clockwise
                    (last_direction + 1) % self.n_hd         # Turn clockwise
                ]

            best_reward_from_here = -float('inf')

            for direction in directions_to_try:
                # Simulate one step in this direction
                next_activations = self.preplay_from_state(activations, direction, num_steps=1)

                # Recursively evaluate the rest of the path with weighted accumulation
                new_path = path_so_far + [direction]
                reward = evaluate_path_recursive_weighted(next_activations, remaining_steps - 1, new_path, depth + 1, total_reward_so_far)

                best_reward_from_here = max(best_reward_from_here, reward)

            return best_reward_from_here

        # Start evaluation from forced direction with validated depth
        best_reward = evaluate_path_recursive_weighted(current_activations, valid_depth, [], 0, 0.0)

        if debug:
            print(f"    Adaptive result: best_reward={best_reward:.4f}, used_depth={valid_depth}")

        return best_reward, valid_depth

    def multi_step_preplay_improvement_weighted(self, forced_first_direction: int, max_steps: int = 3,
                                                 decay_factor: float = 0.6, activation_threshold: float = 0.1,
                                                 improvement_boost: float = 1.5, debug: bool = False) -> tuple:
        """Multi-step preplay with reward-improvement-based weighting.

        Similar to multi_step_preplay_adaptive_validity, but modulates the decay weight
        based on whether future rewards represent an improvement over previous steps.
        This helps the system "see" improving trajectories toward goals more clearly.

        Args:
            forced_first_direction: Direction (0-7) that the first step must take
            max_steps: Maximum number of steps to look ahead
            decay_factor: Base exponential decay for step weighting
            activation_threshold: Minimum activation to consider place cells valid
            improvement_boost: Maximum multiplier for improving rewards (e.g., 1.5 = 50% boost)
            debug: Whether to print debug information

        Returns:
            tuple: (best_weighted_reward, actual_steps_used)
        """
        # First, determine the valid preplay depth
        valid_depth = self.determine_valid_preplay_depth(max_steps, activation_threshold)

        if valid_depth == 0:
            # No valid preplay possible
            if debug:
                print(f"    Improvement-weighted preplay: No valid depth found (all PCs below threshold {activation_threshold})")
            return 0.0, 0

        if debug:
            print(f"    Improvement-weighted preplay: forced_dir={forced_first_direction}({forced_first_direction*45}°), valid_depth={valid_depth}/{max_steps}, threshold={activation_threshold}, boost={improvement_boost}")

        # Use the existing weighted preplay with the validated depth
        current_activations = self.place_cell_activations.clone()
        all_paths_evaluated = []

        def evaluate_path_recursive_improvement_weighted(activations, remaining_steps, path_so_far, depth=0, accumulated_reward=0.0, prev_reward=None):
            """Recursively evaluate paths with improvement-aware weighting.

            Weight future steps more heavily if they represent reward improvements,
            and less heavily if rewards are flat or declining.
            """

            # Evaluate reward at current state
            current_step_reward = self._evaluate_activations_for_reward(activations)

            # Calculate base weight (time-based decay)
            base_weight = decay_factor ** depth

            # Modulate weight based on reward improvement
            if prev_reward is not None and depth > 0:
                improvement = current_step_reward - prev_reward

                if improvement > 0:
                    # Reward is improving: boost the weight
                    # Map improvement [0, 1] to multiplier [1.0, improvement_boost]
                    improvement_multiplier = 1.0 + (improvement * (improvement_boost - 1.0))
                else:
                    # Reward is flat or declining: reduce the weight
                    # Map negative improvement to multiplier [0.6, 1.0]
                    improvement_multiplier = max(0.6, 1.0 + improvement)

                step_weight = base_weight * improvement_multiplier

                if debug and len(path_so_far) <= 2:
                    print(f"  {'  ' * depth}Depth {depth}: prev_r={prev_reward:.3f}, curr_r={current_step_reward:.3f}, "
                          f"imp={improvement:+.3f}, mult={improvement_multiplier:.2f}, weight={step_weight:.4f}")
            else:
                step_weight = base_weight

            # Apply weight to reward
            weighted_step_reward = step_weight * current_step_reward
            total_reward_so_far = accumulated_reward + weighted_step_reward

            if debug and len(path_so_far) <= 2:
                print(f"  {'  ' * depth}Step {depth+1}: raw_reward={current_step_reward:.4f}, "
                      f"weight={step_weight:.3f}, weighted={weighted_step_reward:.4f}, total={total_reward_so_far:.4f}")

            if remaining_steps == 0:
                # Base case: no more steps to evaluate
                all_paths_evaluated.append((path_so_far.copy(), total_reward_so_far))
                return total_reward_so_far

            # Determine directions to explore at this step
            if len(path_so_far) == 0:
                # First step: use forced direction only
                directions_to_try = [forced_first_direction]
            else:
                # Subsequent steps: limited branching
                last_direction = path_so_far[-1]
                directions_to_try = [
                    last_direction,                           # Continue straight
                    (last_direction - 1) % self.n_hd,        # Turn counter-clockwise
                    (last_direction + 1) % self.n_hd         # Turn clockwise
                ]

            best_reward_from_here = -float('inf')

            for direction in directions_to_try:
                # Simulate one step in this direction
                next_activations = self.preplay_from_state(activations, direction, num_steps=1)

                # Recursively evaluate the rest of the path with weighted accumulation
                # Pass current reward as prev_reward for next level
                new_path = path_so_far + [direction]
                reward = evaluate_path_recursive_improvement_weighted(
                    next_activations,
                    remaining_steps - 1,
                    new_path,
                    depth + 1,
                    total_reward_so_far,
                    prev_reward=current_step_reward  # Track previous step reward for improvement calculation
                )

                best_reward_from_here = max(best_reward_from_here, reward)

            return best_reward_from_here

        # Start evaluation from forced direction with validated depth
        # No prev_reward for first step (depth 0)
        best_reward = evaluate_path_recursive_improvement_weighted(
            current_activations,
            valid_depth,
            [],
            0,
            0.0,
            prev_reward=None
        )

        if debug:
            print(f"    Improvement-weighted result: best_reward={best_reward:.4f}, used_depth={valid_depth}")

        return best_reward, valid_depth

    def update_correlation_tracking(self, pc_activations):
        """Update correlation tracking with current place cell activations.
        
        Args:
            pc_activations: Current place cell activations tensor
        """
        if not self.enable_correlation_weighting:
            return
            
        # Apply threshold to activations before adding to history
        # This prevents weak noise from influencing correlations
        thresholded_activations = torch.where(
            pc_activations > self.correlation_threshold,
            pc_activations,
            torch.zeros_like(pc_activations)
        )
        
        # Add to activation history
        self.activation_history.append(thresholded_activations.clone().detach())
        
        # Maintain window size
        if len(self.activation_history) > self.correlation_window:
            self.activation_history.pop(0)
        
        # Update correlation matrix periodically for efficiency
        self.correlation_step_counter += 1
        if (self.correlation_step_counter % self.correlation_update_freq == 0 and 
            len(self.activation_history) >= 10):
            self.compute_correlation_matrix()

    def compute_correlation_matrix(self):
        """Compute correlation matrix from activation history."""
        if len(self.activation_history) < 10:  # Need minimum data
            return
            
        try:
            # Stack history into matrix: (timesteps, num_pc)
            history_matrix = torch.stack(self.activation_history, dim=0)
            
            # Compute correlation coefficient matrix
            self.correlation_matrix = torch.corrcoef(history_matrix.T)
            
            # Handle NaN values (cells that never activate or have constant activation)
            self.correlation_matrix = torch.nan_to_num(
                self.correlation_matrix, 
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )
            
            # Ensure diagonal is 1.0 (self-correlation) - compatible with older PyTorch
            diagonal_indices = torch.arange(self.num_pc, device=self.device)
            self.correlation_matrix[diagonal_indices, diagonal_indices] = 1.0
            
        except Exception as e:
            # Fallback to neutral correlations if computation fails
            print(f"[PCN] Correlation computation failed: {e}")
            self.correlation_matrix = torch.ones(
                self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
            ) * 0.5

    def get_correlation_weights(self):
        """Convert correlation matrix to connection weights.
        
        Returns:
            torch.Tensor: Weight matrix (num_pc, num_pc) with values in [min_weight, 1.0]
        """
        if not self.enable_correlation_weighting:
            return torch.ones(self.num_pc, self.num_pc, dtype=self.dtype, device=self.device)
        
        # Sigmoid transformation for smooth mapping from correlation to weights
        # Maps correlation range [-1, 1] to sigmoid range [0, 1]
        sigmoid_corr = torch.sigmoid(self.correlation_scaling * self.correlation_matrix)
        
        # Scale to [min_correlation_weight, 1.0] range
        weights = (self.min_correlation_weight + 
                  (1.0 - self.min_correlation_weight) * sigmoid_corr)
        
        return weights

    def apply_correlation_weighting_to_stdp(self, connection_update):
        """Apply correlation-based weighting to STDP updates.

        Args:
            connection_update: Raw STDP update tensor (n_hd, num_pc, num_pc)

        Returns:
            Weighted connection update tensor
        """
        if not self.enable_correlation_weighting:
            return connection_update

        # Get correlation weights
        correlation_weights = self.get_correlation_weights()

        # Apply weights to all head direction slices
        weighted_update = torch.zeros_like(connection_update)
        for hd_idx in range(connection_update.shape[0]):
            weighted_update[hd_idx] = connection_update[hd_idx] * correlation_weights

        return weighted_update

    ###############################################################################
    # ADAPTIVE LEARNING RATE & CONNECTION DECAY (from v21)
    ###############################################################################

    def apply_connection_decay(self):
        """Apply synaptic decay to recurrent connections for homeostasis."""
        if not self.enable_connection_decay:
            return

        # Simple exponential decay: w *= (1 - decay_rate)
        self.w_rec_tripartite *= (1 - self.connection_decay_rate)

    def update_hd_eligibility_trace(self, hd_activations):
        """Update head direction eligibility traces with proper time constant.

        Args:
            hd_activations: Current head direction activations tensor (n_hd,)
        """
        if hd_activations is None:
            return

        # Convert to proper shape and handle NaN values
        hd_activations_clean = torch.nan_to_num(hd_activations)
        hd_activations_expanded = hd_activations_clean.unsqueeze(1).unsqueeze(2)  # (n_hd, 1, 1)

        # Update eligibility trace with proper time constant
        # dΥ^h_k/dt = (-Υ^h_k + v^h_k) / τ_hd
        self.hd_cell_trace += (self.tau / self.tau_hd) * (
            hd_activations_expanded - self.hd_cell_trace
        )

    def compute_connection_strength(self, direction_connections):
        """Compute connection development measure for adaptive learning rate.

        Args:
            direction_connections: Connection matrix for a single direction (num_pc, num_pc)

        Returns:
            float: Connection strength measure for determining learning rate
        """
        abs_connections = torch.abs(direction_connections)

        # Use a low threshold to capture early learning
        significance_threshold = 0.0001
        significant_mask = abs_connections > significance_threshold

        if torch.sum(significant_mask) > 0:
            # Return mean strength of significant connections
            return torch.mean(abs_connections[significant_mask]).item()
        else:
            # For completely zero connections, return 0.0 (will use initial_lr)
            return 0.0

    def get_adaptive_learning_rate(self, connection_strength):
        """Compute adaptive learning rate based on connection strength.

        Args:
            connection_strength: Connection development measure from compute_connection_strength

        Returns:
            float: Learning rate for this direction (between final_lr and initial_lr)
        """
        if not self.enable_adaptive_stdp:
            return self.stdp_learning_rate

        # Exponential decay from initial_lr to final_lr
        # lr = final_lr + (initial_lr - final_lr) * exp(-connection_strength * decay_rate)
        import math
        decay_factor = math.exp(-connection_strength * self.adaptive_decay_rate)
        lr = self.adaptive_final_lr + (self.adaptive_initial_lr - self.adaptive_final_lr) * decay_factor

        return lr

    def boltzmann_multiscale_preplay(
        self,
        scales_data: list,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        inverse_temperature: float = 5.0,
        debug: bool = False
    ) -> tuple:
        """Perform Boltzmann-weighted integration of multi-scale preplay trajectories.

        This function evaluates imagined preplay paths across multiple spatial scales
        (small, medium, large) and eight possible head directions. Each trajectory follows
        a fixed heading (no within-trajectory turning) for the specified number of steps.
        Step rewards and direction vectors are combined with temporal discounting, and a
        Boltzmann distribution weights all trajectories to produce a final movement direction
        and expected value.

        If the combined vector magnitude falls below epsilon (near-zero from cancellation),
        the function falls back to the max-return trajectory, setting both the direction and
        expected value to ensure full consistency between the chosen action and its value.

        All operations are fully vectorized using PyTorch tensors with no .item() calls
        until the final output conversion.

        Args:
            scales_data: List of tuples (scale_name, pcn, rcn) for each spatial scale
            num_steps: Number of preplay steps per trajectory (each continues straight)
            discount_factor: Temporal discount factor (gamma), typically 0.9
            inverse_temperature: Boltzmann inverse temperature (beta), controls selectivity
            debug: Whether to print debug information

        Returns:
            tuple: (final_direction_deg, expected_value, combined_vector,
                    discounted_returns, direction_vectors, boltzmann_probs, trajectory_metadata)
            - final_direction_deg: Final movement direction in degrees (0-360)
            - expected_value: Expected reward value (tensor), consistent with chosen direction
            - combined_vector: Underlying combined direction vector [x, y] (tensor)
            - discounted_returns: Per-trajectory discounted returns (tensor, shape [N_traj])
            - direction_vectors: Per-trajectory direction vectors (tensor, shape [N_traj, 2])
            - boltzmann_probs: Per-trajectory Boltzmann probabilities (tensor, shape [N_traj])
            - trajectory_metadata: List of dicts with scale_idx, scale_name, direction for each trajectory
        """

        if debug:
            print(f"[BOLTZMANN] Preplay: {len(scales_data)} scales × {self.n_hd} dirs × {num_steps} steps | γ={discount_factor} β={inverse_temperature}")

        # Build discount weights for all steps: [gamma^0, gamma^1, ..., gamma^(num_steps-1)]
        discount_weights = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)

        # Store trajectory metadata (scale info and direction indices)
        trajectory_metadata = []

        # Accumulate per-trajectory data as lists of tensors
        all_discounted_returns = []
        all_direction_vectors = []

        # Loop through each scale
        for scale_idx, (scale_name, pcn, rcn) in enumerate(scales_data):
            # Loop through each of the 8 head directions
            for initial_direction in range(self.n_hd):
                # We will accumulate all micro-trajectories into a single macro-vector
                macro_discounted_return = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                macro_direction_vector = torch.zeros(2, dtype=self.dtype, device=self.device)

                micro_trajectory_count = 0

                def explore_microtrajectory(activations, remaining_steps, path_so_far, depth, accumulated_reward, accumulated_vector):
                    """Recursively explore all micro-trajectories starting from initial_direction.

                    Args:
                        activations: Current place cell activations
                        remaining_steps: Number of steps remaining
                        path_so_far: List of directions taken so far
                        depth: Current depth in the trajectory (0-indexed)
                        accumulated_reward: Sum of discounted rewards so far
                        accumulated_vector: Sum of discounted direction vectors so far
                    """
                    nonlocal macro_discounted_return, macro_direction_vector, micro_trajectory_count

                    # Determine which directions to explore at this step
                    if len(path_so_far) == 0:
                        # First step: use initial direction only (forced)
                        directions_to_try = [initial_direction]
                    else:
                        # Subsequent steps: branch to straight, left (-1), right (+1)
                        last_direction = path_so_far[-1]
                        directions_to_try = [
                            last_direction,                    # Continue straight
                            (last_direction - 1) % self.n_hd,  # Turn left (counter-clockwise)
                            (last_direction + 1) % self.n_hd   # Turn right (clockwise)
                        ]

                    for direction in directions_to_try:
                        # Perform one preplay step in this direction
                        next_activations = pcn.preplay_from_state(activations, direction, num_steps=1)

                        # Evaluate reward at this step (keep as tensor)
                        rcn.update_reward_cell_activations(next_activations, visit=False)
                        step_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))

                        # Compute discounted reward for this step
                        step_weight = discount_weights[depth]
                        weighted_reward = step_weight * step_reward

                        # Compute direction vector for this step
                        step_angle = direction * (2 * np.pi / self.n_hd)
                        step_vector = torch.tensor([np.cos(step_angle), np.sin(step_angle)],
                                                   dtype=self.dtype, device=self.device)
                        weighted_vector = step_weight * step_vector

                        # Accumulate for this branch
                        branch_reward = accumulated_reward + weighted_reward
                        branch_vector = accumulated_vector + weighted_vector

                        if remaining_steps == 1:
                            # Leaf node: this micro-trajectory is complete
                            # Add to macro-vector
                            macro_discounted_return += branch_reward
                            macro_direction_vector += branch_vector
                            micro_trajectory_count += 1
                        else:
                            # Continue exploring this branch
                            new_path = path_so_far + [direction]
                            explore_microtrajectory(
                                next_activations,
                                remaining_steps - 1,
                                new_path,
                                depth + 1,
                                branch_reward,
                                branch_vector
                            )

                # Start recursive exploration from initial direction
                starting_activations = pcn.place_cell_activations.clone()
                explore_microtrajectory(
                    starting_activations,
                    num_steps,
                    [],
                    0,
                    torch.tensor(0.0, dtype=self.dtype, device=self.device),
                    torch.zeros(2, dtype=self.dtype, device=self.device)
                )

                # Store the macro-vector for this (scale, initial_direction) pair
                all_discounted_returns.append(macro_discounted_return)
                all_direction_vectors.append(macro_direction_vector)
                trajectory_metadata.append({
                    'scale_idx': scale_idx,
                    'scale_name': scale_name,
                    'direction': initial_direction
                })

        # Stack all trajectory results into tensors
        # Shape: [N_trajectories] and [N_trajectories, 2]
        discounted_returns = torch.stack(all_discounted_returns)
        direction_vectors = torch.stack(all_direction_vectors)

        # Compute Boltzmann probabilities using numerically stable softmax
        # P(trajectory) ∝ exp(beta * discounted_return)
        # Subtract max for numerical stability before exp
        rewards_normalized = discounted_returns - torch.max(discounted_returns)
        boltzmann_weights = torch.exp(inverse_temperature * rewards_normalized)

        # Normalize to get probabilities
        total_weight = torch.sum(boltzmann_weights)
        boltzmann_probs = boltzmann_weights / torch.clamp(total_weight, min=1e-9)

        if debug:
            # Top 3 trajectories
            top_indices = torch.topk(boltzmann_probs, min(3, len(boltzmann_probs))).indices
            print("[BOLTZMANN] Top 3:")
            for idx in top_indices:
                traj = trajectory_metadata[idx.item()]
                scale_abbrev = traj['scale_name'][0]  # S/M/L
                print(f"  {scale_abbrev}-{traj['direction']*45:3d}° P={boltzmann_probs[idx].item():.3f} R={discounted_returns[idx].item():.3f}")

            # Summary statistics
            print(f"[BOLTZMANN] Rewards: min={torch.min(discounted_returns).item():.2f} "
                  f"max={torch.max(discounted_returns).item():.2f} mean={torch.mean(discounted_returns).item():.2f}")

            # Scale contribution breakdown
            scale_probs = []
            for i in range(len(scales_data)):
                scale_prob = boltzmann_probs[i*self.n_hd:(i+1)*self.n_hd].sum().item()
                scale_probs.append(scale_prob)
            scale_str = ' '.join([f"{scales_data[i][0][0]}:{p:.2f}" for i, p in enumerate(scale_probs)])
            print(f"[BOLTZMANN] Scale Mass: {scale_str}")

        # Form combined movement vector: weighted sum of direction vectors
        # Shape: [2] = sum over trajectories of (prob * direction_vector)
        combined_vector = torch.sum(boltzmann_probs.unsqueeze(1) * direction_vectors, dim=0)

        # Compute expected value: weighted sum of discounted returns (may be updated in fallback)
        expected_value = torch.sum(boltzmann_probs * discounted_returns)

        # Robust fallback: if combined vector magnitude is near zero, use max-return trajectory
        # and update expected value to match the chosen trajectory for consistency
        combined_magnitude = torch.norm(combined_vector)
        epsilon = 1e-6
        if combined_magnitude < epsilon:
            # Pick trajectory with highest discounted return
            max_idx = torch.argmax(discounted_returns)
            combined_vector = direction_vectors[max_idx]
            # Set expected value to match the chosen trajectory (full consistency)
            expected_value = discounted_returns[max_idx]
            if debug:
                print(f"[BOLTZMANN] Fallback: vector near zero (mag={combined_magnitude.item():.2e}), using max-return traj")

        # Compute final direction angle from combined vector
        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])

        # Convert to degrees [0, 360) - keep as tensor until final return
        final_direction_deg_tensor = final_direction_rad * (180.0 / np.pi)
        final_direction_deg_tensor = torch.where(
            final_direction_deg_tensor < 0,
            final_direction_deg_tensor + 360.0,
            final_direction_deg_tensor
        )

        if debug:
            print(f"[BOLTZMANN] Result: θ={final_direction_deg_tensor.item():.1f}° V={expected_value.item():.3f} "
                  f"vec=[{combined_vector[0].item():.2f},{combined_vector[1].item():.2f}]")

        # Return all tensors (no .item() calls), plus metadata
        return (final_direction_deg_tensor.item(), expected_value, combined_vector,
                discounted_returns, direction_vectors, boltzmann_probs, trajectory_metadata)

    def hierarchical_multiscale_preplay(
        self,
        scales_data: list,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        within_scale_beta: float = 2.0,
        scale_selection_beta: float = 1.0,
        ema_lambda: float = 0.1,
        prev_scale_weights: Optional[torch.Tensor] = None,
        scale_reliability: Optional[torch.Tensor] = None,
        safety_mask: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> tuple:
        """Perform hierarchical multi-scale preplay with entropy-based scale selection.

        This method introduces a two-level normalization approach:
        1. Within each scale: Compute Boltzmann distribution over directions
        2. Across scales: Score scales by mean return and entropy, apply softmax + EMA

        The key innovation is entropy-based scale quality assessment:
        - Low entropy within a scale → clear directional signal → trust this scale
        - High entropy within a scale → confused/ambiguous → distrust this scale

        Args:
            scales_data: List of tuples (scale_name, pcn, rcn) for each spatial scale
            num_steps: Number of preplay steps per trajectory
            discount_factor: Temporal discount factor (gamma)
            within_scale_beta: Inverse temperature for within-scale Boltzmann (beta)
            scale_selection_beta: Inverse temperature for scale selection softmax (higher = more decisive)
            ema_lambda: EMA decay for scale weights (0.1 = 10% new, 90% old)
            prev_scale_weights: Previous timestep's scale weights for EMA (None on first call)
            scale_reliability: Per-scale reliability scores [num_scales] in [0,1] (None = disabled)
            safety_mask: Optional boolean tensor [n_hd]; False marks directions that are unsafe and should be
                         ignored in within-scale scoring (weights ~0 via large negative return)
            debug: Whether to print debug information

        Returns:
            tuple: (final_direction_deg, expected_value, combined_vector,
                    discounted_returns_per_traj, direction_vectors_per_traj,
                    joint_probs, trajectory_metadata, scale_weights)
            - final_direction_deg: Final movement direction in degrees (0-360)
            - expected_value: Expected reward value (tensor)
            - combined_vector: Combined direction vector [x, y] (tensor)
            - discounted_returns_per_traj: Per-trajectory discounted returns (tensor, [N_traj])
            - direction_vectors_per_traj: Per-trajectory direction vectors (tensor, [N_traj, 2])
            - joint_probs: P(s,d) joint probabilities (tensor, [N_traj])
            - trajectory_metadata: List of dicts with scale_idx, scale_name, direction
            - scale_weights: P(s) scale weights after EMA (tensor, [num_scales])
        """

        num_scales = len(scales_data)

        if debug:
            print(f"[HIERARCHICAL] Preplay: {num_scales} scales × {self.n_hd} dirs × {num_steps} steps")
            print(f"[HIERARCHICAL] Params: β={within_scale_beta} λ={ema_lambda}")

        # Build discount weights for all steps: [gamma^0, gamma^1, ..., gamma^(num_steps-1)]
        discount_weights = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)
        # Precompute unit step vectors for each head direction to avoid per-sample trig
        angles = torch.arange(self.n_hd, device=self.device, dtype=self.dtype) * (2 * np.pi / self.n_hd)
        step_vectors = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

        # Store trajectory metadata
        trajectory_metadata = []

        # Accumulate per-trajectory data
        all_discounted_returns = []
        all_direction_vectors = []

        # ------------------------------------------------------------------
        # STAGE 1: Compute discounted returns and direction vectors
        #          for all (scale, direction) pairs
        # ------------------------------------------------------------------

        for scale_idx, (scale_name, pcn, rcn) in enumerate(scales_data):
            for initial_direction in range(self.n_hd):
                # If a safety mask is provided and this direction is unsafe, skip exploration and down-weight it.
                if safety_mask is not None and not bool(safety_mask[initial_direction]):
                    all_discounted_returns.append(torch.tensor(-1e9, dtype=self.dtype, device=self.device))
                    all_direction_vectors.append(torch.zeros(2, dtype=self.dtype, device=self.device))
                    trajectory_metadata.append({
                        'scale_idx': scale_idx,
                        'scale_name': scale_name,
                        'direction': initial_direction
                    })
                    continue

                # Accumulate macro-trajectory from all micro-trajectories
                macro_discounted_return = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                macro_direction_vector = torch.zeros(2, dtype=self.dtype, device=self.device)

                def explore_microtrajectory(activations, remaining_steps, path_so_far, depth, accumulated_reward, accumulated_vector):
                    """Recursively explore all micro-trajectories starting from initial_direction."""
                    nonlocal macro_discounted_return, macro_direction_vector

                    # Determine which directions to explore at this step
                    if len(path_so_far) == 0:
                        # First step: use initial direction only (forced)
                        directions_to_try = [initial_direction]
                    else:
                        # Subsequent steps: branch to straight, left (-1), right (+1)
                        last_direction = path_so_far[-1]
                        directions_to_try = [
                            last_direction,                    # Continue straight
                            (last_direction - 1) % self.n_hd,  # Turn left (counter-clockwise)
                            (last_direction + 1) % self.n_hd   # Turn right (clockwise)
                        ]

                    for direction in directions_to_try:
                        # Perform one preplay step in this direction
                        next_activations = pcn.preplay_from_state(activations, direction, num_steps=1)

                        # Evaluate reward at this step (keep as tensor)
                        rcn.update_reward_cell_activations(next_activations, visit=False)
                        step_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))

                        # Compute discounted reward for this step
                        step_weight = discount_weights[depth]
                        weighted_reward = step_weight * step_reward

                        # Compute direction vector for this step
                        step_angle = direction * (2 * np.pi / self.n_hd)
                        step_vector = torch.tensor([np.cos(step_angle), np.sin(step_angle)],
                                                   dtype=self.dtype, device=self.device)
                        weighted_vector = step_weight * step_vector

                        # Accumulate for this branch
                        branch_reward = accumulated_reward + weighted_reward
                        branch_vector = accumulated_vector + weighted_vector

                        if remaining_steps == 1:
                            # Leaf node: this micro-trajectory is complete
                            macro_discounted_return += branch_reward
                            macro_direction_vector += branch_vector
                        else:
                            # Continue exploring this branch
                            new_path = path_so_far + [direction]
                            explore_microtrajectory(
                                next_activations,
                                remaining_steps - 1,
                                new_path,
                                depth + 1,
                                branch_reward,
                                branch_vector
                            )

                # Start recursive exploration from initial direction
                starting_activations = pcn.place_cell_activations.clone()
                explore_microtrajectory(
                    starting_activations,
                    num_steps,
                    [],
                    0,
                    torch.tensor(0.0, dtype=self.dtype, device=self.device),
                    torch.zeros(2, dtype=self.dtype, device=self.device)
                )

                # Store the macro-vector for this (scale, initial_direction) pair
                all_discounted_returns.append(macro_discounted_return)
                all_direction_vectors.append(macro_direction_vector)
                trajectory_metadata.append({
                    'scale_idx': scale_idx,
                    'scale_name': scale_name,
                    'direction': initial_direction
                })

        # Stack all trajectory results into tensors
        # Shape: [N_trajectories] and [N_trajectories, 2]
        discounted_returns = torch.stack(all_discounted_returns)
        direction_vectors = torch.stack(all_direction_vectors)

        # ------------------------------------------------------------------
        # STAGE 2: Hierarchical Normalization
        # ------------------------------------------------------------------

        # Initialize storage for per-scale statistics
        scale_mean_returns = torch.zeros(num_scales, dtype=self.dtype, device=self.device)
        scale_entropies = torch.zeros(num_scales, dtype=self.dtype, device=self.device)
        scale_direction_probs = []  # List of tensors, one per scale

        for scale_idx in range(num_scales):
            # Extract returns for this scale (8 directions)
            scale_start = scale_idx * self.n_hd
            scale_end = scale_start + self.n_hd
            scale_returns = discounted_returns[scale_start:scale_end]

            # Compute Boltzmann distribution over directions within this scale
            # P(d|s) = exp(beta * R_{s,d}) / Σ_d' exp(beta * R_{s,d'})
            scale_returns_normalized = scale_returns - torch.max(scale_returns)  # Numerical stability
            boltzmann_weights = torch.exp(within_scale_beta * scale_returns_normalized)
            direction_probs = boltzmann_weights / torch.clamp(torch.sum(boltzmann_weights), min=1e-9)
            scale_direction_probs.append(direction_probs)

            # Compute mean return: M_s = Σ_d P(d|s) × R_{s,d}
            mean_return = torch.sum(direction_probs * scale_returns)
            scale_mean_returns[scale_idx] = mean_return
            

            # Compute entropy: H_s = -Σ_d P(d|s) log P(d|s)
            # Clamp probabilities to avoid log(0)
            safe_probs = torch.clamp(direction_probs, min=1e-9)
            entropy = -torch.sum(direction_probs * torch.log(safe_probs))
            scale_entropies[scale_idx] = entropy
            print(f"For {scale_idx}: Mean: {mean_return}, Entropy: {entropy}")

        # NEW SIMPLEX-BASED SCALE SCORING WITH SOFTMAX
        eps = 1e-9

        # Step 1: Calculate inverse entropy (lower entropy = higher value)
        inverse_entropies = 1.0 / (scale_entropies + eps)

        # Step 2: Get reliability scores (or use uniform if not provided)
        if scale_reliability is not None:
            reliability_scores = scale_reliability
            print(f"Reliability (raw): {scale_reliability}")
        else:
            # If no reliability provided, use uniform (all 1s, will cancel out in product)
            reliability_scores = torch.ones(num_scales, dtype=self.dtype, device=self.device)

        # Step 3: Compute scale scores as product: mean_return * inverse_entropy * reliability
        scale_scores = scale_mean_returns * inverse_entropies * reliability_scores
        print(f"Scale scores (raw product): {scale_scores}")

        # Step 4: Apply softmax with beta parameter
        scale_scores_normalized = scale_scores - torch.max(scale_scores)  # Numerical stability
        scale_weights_raw = torch.exp(scale_selection_beta * scale_scores_normalized)
        scale_weights_current = scale_weights_raw / torch.clamp(torch.sum(scale_weights_raw), min=eps)
        print(f"Scale weights (after softmax with β={scale_selection_beta}): {scale_weights_current}")

        # Apply EMA for temporal stability
        if prev_scale_weights is not None:
            scale_weights = ema_lambda * scale_weights_current + (1.0 - ema_lambda) * prev_scale_weights
            # Renormalize after EMA
            scale_weights = scale_weights / torch.clamp(torch.sum(scale_weights), min=1e-9)
        else:
            # First timestep: no previous weights, use current
            scale_weights = scale_weights_current

        # ------------------------------------------------------------------
        # STAGE 3: Form Joint Distribution P(s, d) = P(s) × P(d|s)
        # ------------------------------------------------------------------

        joint_probs = torch.zeros(num_scales * self.n_hd, dtype=self.dtype, device=self.device)

        for scale_idx in range(num_scales):
            scale_start = scale_idx * self.n_hd
            scale_end = scale_start + self.n_hd
            # P(s, d) = P(s) × P(d|s)
            joint_probs[scale_start:scale_end] = scale_weights[scale_idx] * scale_direction_probs[scale_idx]

        # ------------------------------------------------------------------
        # STAGE 4: Compute Combined Vector and Expected Value
        # ------------------------------------------------------------------

        # Form combined movement vector: weighted sum of direction vectors
        combined_vector = torch.sum(joint_probs.unsqueeze(1) * direction_vectors, dim=0)

        # Compute expected value: weighted sum of discounted returns
        expected_value = torch.sum(joint_probs * discounted_returns)

        # Robust fallback: if combined vector magnitude is near zero, use max-return trajectory
        combined_magnitude = torch.norm(combined_vector)
        epsilon = 1e-6
        if combined_magnitude < epsilon:
            max_idx = torch.argmax(discounted_returns)
            combined_vector = direction_vectors[max_idx]
            expected_value = discounted_returns[max_idx]
            if debug:
                print(f"[HIERARCHICAL] Fallback: vector near zero (mag={combined_magnitude.item():.2e}), using max-return traj")

        # Compute final direction angle from combined vector
        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])

        # Convert to degrees [0, 360)
        final_direction_deg_tensor = final_direction_rad * (180.0 / np.pi)
        final_direction_deg_tensor = torch.where(
            final_direction_deg_tensor < 0,
            final_direction_deg_tensor + 360.0,
            final_direction_deg_tensor
        )

        # ------------------------------------------------------------------
        # STAGE 5: Debug Output
        # ------------------------------------------------------------------

        if debug:
            print(f"\n[HIERARCHICAL] Scale Statistics:")
            for scale_idx, (scale_name, _, _) in enumerate(scales_data):
                print(f"  {scale_name}: M={scale_mean_returns[scale_idx].item():.3f} "
                      f"H={scale_entropies[scale_idx].item():.3f} "
                      f"Q={scale_scores[scale_idx].item():.3f} "
                      f"P={scale_weights[scale_idx].item():.3f}")

            # Top 3 trajectories by joint probability
            top_indices = torch.topk(joint_probs, min(3, len(joint_probs))).indices
            print(f"\n[HIERARCHICAL] Top 3 Trajectories:")
            for idx in top_indices:
                traj = trajectory_metadata[idx.item()]
                scale_abbrev = traj['scale_name'][0]  # S/M/L
                print(f"  {scale_abbrev}-{traj['direction']*45:3d}° P={joint_probs[idx].item():.3f} "
                      f"R={discounted_returns[idx].item():.3f}")

            print(f"\n[HIERARCHICAL] Result: θ={final_direction_deg_tensor.item():.1f}° "
                  f"V={expected_value.item():.3f} "
                  f"vec=[{combined_vector[0].item():.2f},{combined_vector[1].item():.2f}]")

        # Return all data including scale_weights for next timestep's EMA
        return (final_direction_deg_tensor.item(), expected_value, combined_vector,
                discounted_returns, direction_vectors, joint_probs, trajectory_metadata, scale_weights)

    def hierarchical_multiscale_preplay_sampling(
        self,
        scales_data: list,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        within_scale_beta: float = 2.0,
        scale_selection_beta: float = 1.0,
        ema_lambda: float = 0.1,
        prev_scale_weights: Optional[torch.Tensor] = None,
        scale_reliability: Optional[torch.Tensor] = None,
        num_samples: int = 10,
        sampling_strategy: str = "uniform",
        sampling_temperature: float = 1.0,
        debug: bool = False
    ) -> tuple:
        """Perform hierarchical multi-scale preplay with STOCHASTIC TRAJECTORY SAMPLING.

        This is a biologically plausible alternative to exhaustive branching that samples
        a fixed number of trajectories instead of enumerating all possible futures.

        Key differences from hierarchical_multiscale_preplay:
        1. Samples K trajectories per (scale, direction) instead of exhaustive branching
        2. Aggregates via MEAN not SUM (unbiased Monte Carlo estimate)
        3. Returns variance diagnostics for each trajectory
        4. More efficient for deeper planning (num_steps > 2)

        Biological motivation:
        - Hippocampal replay shows variable, sampled trajectories, not exhaustive search
        - Grid cells and head direction cells have inherent noise
        - Each mental simulation is slightly different
        - Matches experimental data on theta sequences

        Args:
            scales_data: List of tuples (scale_name, pcn, rcn) for each spatial scale
            num_steps: Number of preplay steps per trajectory
            discount_factor: Temporal discount factor (gamma)
            within_scale_beta: Inverse temperature for within-scale Boltzmann (beta)
            scale_selection_beta: Inverse temperature for scale selection softmax (higher = more decisive)
            ema_lambda: EMA decay for scale weights (0.1 = 10% new, 90% old)
            prev_scale_weights: Previous timestep's scale weights for EMA (None on first call)
            scale_reliability: Per-scale reliability scores [num_scales] in [0,1] (None = disabled)
            num_samples: Number of trajectories to sample per (scale, direction) pair
            sampling_strategy: "uniform" (random turns) or "learned" (use W_rec probabilities)
            sampling_temperature: Softmax temperature for learned strategy (higher = more random)
            debug: Whether to print debug information

        Returns:
            tuple: (final_direction_deg, expected_value, combined_vector,
                    discounted_returns_per_traj, direction_vectors_per_traj,
                    joint_probs, trajectory_metadata, scale_weights, sampling_variances)
            - final_direction_deg: Final movement direction in degrees (0-360)
            - expected_value: Expected reward value (tensor)
            - combined_vector: Combined direction vector [x, y] (tensor)
            - discounted_returns_per_traj: Per-trajectory discounted returns (tensor, [N_traj])
            - direction_vectors_per_traj: Per-trajectory direction vectors (tensor, [N_traj, 2])
            - joint_probs: P(s,d) joint probabilities (tensor, [N_traj])
            - trajectory_metadata: List of dicts with scale_idx, scale_name, direction
            - scale_weights: P(s) scale weights after EMA (tensor, [num_scales])
            - sampling_variances: Per-(scale,direction) return variance (tensor, [num_scales, n_hd])
        """

        num_scales = len(scales_data)

        if debug:
            print(f"[HIERARCHICAL-SAMPLING] Preplay: {num_scales} scales × {self.n_hd} dirs × {num_samples} samples × {num_steps} steps")
            print(f"[HIERARCHICAL-SAMPLING] Params: β={within_scale_beta} λ={ema_lambda}")
            print(f"[HIERARCHICAL-SAMPLING] Sampling: strategy={sampling_strategy} temp={sampling_temperature} K={num_samples}")
            print(f"[HIERARCHICAL-SAMPLING] OPTIMIZED: Batched processing with batch_size={self.n_hd * num_samples} per scale")

        # Build discount weights for all steps: [gamma^0, gamma^1, ..., gamma^(num_steps-1)]
        discount_weights = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)

        # Store trajectory metadata
        trajectory_metadata = []

        # Accumulate per-trajectory data
        all_discounted_returns = []
        all_direction_vectors = []

        # Track sampling variance per (scale, direction)
        sampling_variances = torch.zeros((num_scales, self.n_hd), dtype=self.dtype, device=self.device)

        # ------------------------------------------------------------------
        # STAGE 1: Sample trajectories for all (scale, direction) pairs (OPTIMIZED BATCHED VERSION)
        # ------------------------------------------------------------------

        for scale_idx, (scale_name, pcn, rcn) in enumerate(scales_data):
            # OPTIMIZATION: Process all 8 directions × num_samples trajectories in a single batch
            # Total batch size: n_hd * num_samples (e.g., 8 * 10 = 80 trajectories)
            batch_size = self.n_hd * num_samples

            # Initialize all trajectories for this scale
            # Shape: (batch_size, num_pc)
            # Each direction has num_samples copies of the same initial activation
            initial_activations = pcn.place_cell_activations.unsqueeze(0).expand(batch_size, -1).clone()

            # Initial directions: [0, 0, ..., 0, 1, 1, ..., 1, ..., 7, 7, ..., 7]
            # Shape: (batch_size,)
            initial_directions = torch.arange(self.n_hd, device=self.device).repeat_interleave(num_samples)
            current_directions = initial_directions.clone()

            # Initialize return and vector accumulators
            # Shape: (batch_size,)
            trajectory_returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            # Shape: (batch_size, 2)
            trajectory_vectors = torch.zeros(batch_size, 2, dtype=self.dtype, device=self.device)

            # Current activations for all trajectories
            activations_batch = initial_activations.clone()

            # Simulate num_steps forward for all trajectories in parallel
            for step in range(num_steps):
                # First: Sample direction for this step (if not first step)
                if step > 0:  # First step uses initial_direction
                    if sampling_strategy == "uniform":
                        # Uniform random: left, straight, right with equal probability
                        # Sample for all trajectories at once
                        turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
                        # Random choice for each trajectory
                        chosen_turns = turn_options[torch.randint(0, 3, (batch_size,), device=self.device)]
                        current_directions = (current_directions + chosen_turns) % self.n_hd

                    elif sampling_strategy == "learned":
                        # OPTIMIZED: Use batched turn probability computation
                        turn_probs_batch = pcn._compute_learned_turn_probabilities_batched(
                            activations_batch, current_directions, sampling_temperature
                        )
                        # turn_probs_batch: (batch_size, 3)

                        # Sample from categorical distribution for each trajectory
                        turn_indices = torch.multinomial(turn_probs_batch, num_samples=1).squeeze(1)  # (batch_size,)
                        turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
                        chosen_turns = turn_options[turn_indices]
                        current_directions = (current_directions + chosen_turns) % self.n_hd

                    else:
                        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

                # Second: Preplay one step for all trajectories in parallel
                activations_batch = pcn.preplay_from_state_batched(
                    activations_batch, current_directions, num_steps=1
                )

                # Third: Evaluate reward for all trajectories in parallel
                step_rewards = rcn.compute_reward_activations_batched(activations_batch)  # (batch_size,)
                step_rewards = torch.nan_to_num(step_rewards)

                # Fourth: Accumulate discounted returns
                step_weight = discount_weights[step]
                trajectory_returns += step_weight * step_rewards

                # Fifth: Accumulate direction vectors
                # Convert directions to angles: (batch_size,)
                step_angles = current_directions.float() * (2 * np.pi / self.n_hd)
                # Compute unit vectors: (batch_size, 2)
                step_vectors = torch.stack([torch.cos(step_angles), torch.sin(step_angles)], dim=1)
                trajectory_vectors += step_weight * step_vectors

            # Reshape results by (direction, sample)
            # trajectory_returns: (batch_size,) -> (n_hd, num_samples)
            returns_by_dir = trajectory_returns.view(self.n_hd, num_samples)
            # trajectory_vectors: (batch_size, 2) -> (n_hd, num_samples, 2)
            vectors_by_dir = trajectory_vectors.view(self.n_hd, num_samples, 2)

            # Compute mean and variance across samples for each direction
            # Means: (n_hd,) and (n_hd, 2)
            macro_returns = torch.mean(returns_by_dir, dim=1)  # (n_hd,)
            macro_vectors = torch.mean(vectors_by_dir, dim=1)  # (n_hd, 2)

            # Variances: (n_hd,)
            return_variances = torch.var(returns_by_dir, dim=1)  # (n_hd,)
            sampling_variances[scale_idx, :] = return_variances
            print(f"For Scale: {scale_idx}, Variance: {return_variances}")

            # Store macro-trajectories for each direction
            for initial_direction in range(self.n_hd):
                all_discounted_returns.append(macro_returns[initial_direction])
                all_direction_vectors.append(macro_vectors[initial_direction])
                trajectory_metadata.append({
                    'scale_idx': scale_idx,
                    'scale_name': scale_name,
                    'direction': initial_direction
                })

        # Stack all trajectory results into tensors
        # Shape: [N_trajectories] and [N_trajectories, 2]
        discounted_returns = torch.stack(all_discounted_returns)
        direction_vectors = torch.stack(all_direction_vectors)

        # ------------------------------------------------------------------
        # STAGE 2: Hierarchical Normalization (identical to deterministic version)
        # ------------------------------------------------------------------

        # Initialize storage for per-scale statistics
        scale_mean_returns = torch.zeros(num_scales, dtype=self.dtype, device=self.device)
        scale_entropies = torch.zeros(num_scales, dtype=self.dtype, device=self.device)
        scale_direction_probs = []  # List of tensors, one per scale

        for scale_idx in range(num_scales):
            # Extract returns for this scale (8 directions)
            scale_start = scale_idx * self.n_hd
            scale_end = scale_start + self.n_hd
            scale_returns = discounted_returns[scale_start:scale_end]

            # Compute Boltzmann distribution over directions within this scale
            # P(d|s) = exp(beta * R_{s,d}) / Σ_d' exp(beta * R_{s,d'})
            scale_returns_normalized = scale_returns - torch.max(scale_returns)  # Numerical stability
            boltzmann_weights = torch.exp(within_scale_beta * scale_returns_normalized)
            direction_probs = boltzmann_weights / torch.clamp(torch.sum(boltzmann_weights), min=1e-9)
            scale_direction_probs.append(direction_probs)

            # Compute mean return: M_s = Σ_d P(d|s) × R_{s,d}
            mean_return = torch.sum(direction_probs * scale_returns)
            scale_mean_returns[scale_idx] = mean_return

            # Compute entropy: H_s = -Σ_d P(d|s) log P(d|s)
            # Clamp probabilities to avoid log(0)
            safe_probs = torch.clamp(direction_probs, min=1e-9)
            entropy = -torch.sum(direction_probs * torch.log(safe_probs))
            scale_entropies[scale_idx] = entropy

            print(f"For Scale: {scale_idx}, Mean: {mean_return}, Entropy: {entropy}")

        # NEW SIMPLEX-BASED SCALE SCORING WITH SOFTMAX
        eps = 1e-9

        # Step 1: Calculate inverse entropy (lower entropy = higher value)
        inverse_entropies = 1.0 / (scale_entropies + eps)

        # Step 2: Get reliability scores (or use uniform if not provided)
        if scale_reliability is not None:
            reliability_scores = scale_reliability
            print(f"Reliability (raw): {scale_reliability}")
        else:
            # If no reliability provided, use uniform (all 1s, will cancel out in product)
            reliability_scores = torch.ones(num_scales, dtype=self.dtype, device=self.device)

        # Step 3: Compute scale scores as product: mean_return * inverse_entropy * reliability
        scale_scores = scale_mean_returns * inverse_entropies * reliability_scores
        print(f"Scale scores (raw product): {scale_scores}")

        # Step 4: Apply softmax with beta parameter
        scale_scores_normalized = scale_scores - torch.max(scale_scores)  # Numerical stability
        scale_weights_raw = torch.exp(scale_selection_beta * scale_scores_normalized)
        scale_weights_current = scale_weights_raw / torch.clamp(torch.sum(scale_weights_raw), min=eps)
        print(f"Scale weights (after softmax with β={scale_selection_beta}): {scale_weights_current}")

        # Apply EMA for temporal stability
        if prev_scale_weights is not None:
            scale_weights = ema_lambda * scale_weights_current + (1.0 - ema_lambda) * prev_scale_weights
            # Renormalize after EMA
            scale_weights = scale_weights / torch.clamp(torch.sum(scale_weights), min=1e-9)
        else:
            # First timestep: no previous weights, use current
            scale_weights = scale_weights_current

        # ------------------------------------------------------------------
        # STAGE 3: Form Joint Distribution P(s, d) = P(s) × P(d|s)
        # ------------------------------------------------------------------

        joint_probs = torch.zeros(num_scales * self.n_hd, dtype=self.dtype, device=self.device)

        for scale_idx in range(num_scales):
            scale_start = scale_idx * self.n_hd
            scale_end = scale_start + self.n_hd
            # P(s, d) = P(s) × P(d|s)
            joint_probs[scale_start:scale_end] = scale_weights[scale_idx] * scale_direction_probs[scale_idx]

        # ------------------------------------------------------------------
        # STAGE 4: Compute Combined Vector and Expected Value
        # ------------------------------------------------------------------

        # Form combined movement vector: weighted sum of direction vectors
        combined_vector = torch.sum(joint_probs.unsqueeze(1) * direction_vectors, dim=0)

        # Compute expected value: weighted sum of discounted returns
        expected_value = torch.sum(joint_probs * discounted_returns)

        # Robust fallback: if combined vector magnitude is near zero, use max-return trajectory
        combined_magnitude = torch.norm(combined_vector)
        epsilon = 1e-6
        if combined_magnitude < epsilon:
            max_idx = torch.argmax(discounted_returns)
            combined_vector = direction_vectors[max_idx]
            expected_value = discounted_returns[max_idx]
            if debug:
                print(f"[HIERARCHICAL-SAMPLING] Fallback: vector near zero (mag={combined_magnitude.item():.2e}), using max-return traj")

        # Compute final direction angle from combined vector
        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])

        # Convert to degrees [0, 360)
        final_direction_deg_tensor = final_direction_rad * (180.0 / np.pi)
        final_direction_deg_tensor = torch.where(
            final_direction_deg_tensor < 0,
            final_direction_deg_tensor + 360.0,
            final_direction_deg_tensor
        )

        # ------------------------------------------------------------------
        # STAGE 5: Debug Output
        # ------------------------------------------------------------------

        if debug:
            print(f"\n[HIERARCHICAL-SAMPLING] Scale Statistics:")
            for scale_idx, (scale_name, _, _) in enumerate(scales_data):
                avg_variance = torch.mean(sampling_variances[scale_idx]).item()
                max_variance = torch.max(sampling_variances[scale_idx]).item()
                print(f"  {scale_name}: M={scale_mean_returns[scale_idx].item():.3f} "
                      f"H={scale_entropies[scale_idx].item():.3f} "
                      f"Q={scale_scores[scale_idx].item():.3f} "
                      f"P={scale_weights[scale_idx].item():.3f} "
                      f"Var_avg={avg_variance:.4f} Var_max={max_variance:.4f}")

            # Variance diagnostics: warn if variance is high (K may be too low)
            overall_variance = torch.mean(sampling_variances).item()
            max_overall_variance = torch.max(sampling_variances).item()

            # Theoretical standard error: σ/√K, so for K=10: σ/3.16
            # High variance (>0.1) suggests K is too low for this trajectory complexity
            if max_overall_variance > 0.1:
                print(f"[HIERARCHICAL-SAMPLING] WARNING: High sampling variance detected (max={max_overall_variance:.4f})")
                print(f"  Suggested action: Increase num_samples (current K={num_samples}) or reduce num_steps")

            print(f"[HIERARCHICAL-SAMPLING] Variance summary: mean={overall_variance:.4f} max={max_overall_variance:.4f} "
                  f"(K={num_samples} samples per trajectory)")

            # Top 3 trajectories by joint probability with variance info
            top_indices = torch.topk(joint_probs, min(3, len(joint_probs))).indices
            print(f"\n[HIERARCHICAL-SAMPLING] Top 3 Trajectories:")
            for idx in top_indices:
                traj = trajectory_metadata[idx.item()]
                scale_abbrev = traj['scale_name'][0]  # S/M/L
                traj_variance = sampling_variances[traj['scale_idx'], traj['direction']].item()
                print(f"  {scale_abbrev}-{traj['direction']*45:3d}° P={joint_probs[idx].item():.3f} "
                      f"R={discounted_returns[idx].item():.3f} Var={traj_variance:.4f}")

            print(f"\n[HIERARCHICAL-SAMPLING] Result: θ={final_direction_deg_tensor.item():.1f}° "
                  f"V={expected_value.item():.3f} "
                  f"vec=[{combined_vector[0].item():.2f},{combined_vector[1].item():.2f}]")

        # Return all data including scale_weights for next timestep's EMA and sampling variances
        return (final_direction_deg_tensor.item(), expected_value, combined_vector,
                discounted_returns, direction_vectors, joint_probs, trajectory_metadata, scale_weights, sampling_variances)
