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
        self.alpha_pb = np.sqrt(0.5)

        # Normalization factor for grid cell synaptic weight updates
        self.alpha_pg = np.sqrt(0.5)

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

    def multi_step_preplay_selective_softmax_global(
        self,
        max_steps: int = 3,
        gradient_threshold: float = 0.2,
        softmax_temperature: float = 0.5,
        decay_factor: float = 0.8,
        debug: bool = False
    ) -> tuple:
        """Selective global multistep preplay with softmax branch selection.

        This method implements a novel approach to multistep preplay:
        1. Computes single-step rewards for all 8 directions
        2. Calculates gradient to determine if reward landscape is informative
        3. If gradient < threshold: performs global multistep preplay exploring all branches
        4. Takes MAX reward across branches for each initial direction
        5. Applies softmax to the 8 directional max rewards for amplification
        6. Returns probability distribution over directions (softmax output)

        Key advantages:
        - Only performs expensive multistep when needed (low gradient = flat landscape)
        - Explores entire reward landscape once rather than per-direction
        - Uses softmax with temperature to intelligently weight promising branches
        - More efficient than always doing multistep preplay
        - Always uses full depth when triggered (no premature cutoff from weak activations)

        Args:
            max_steps: Maximum number of preplay steps (depth of exploration)
            gradient_threshold: Trigger multistep if gradient below this value
            softmax_temperature: Temperature for softmax (<1 amplifies differences)
            decay_factor: Temporal decay for future rewards (0.8 = light decay)
            debug: Whether to print debug information

        Returns:
            tuple: (directional_rewards, multistep_triggered)
            - directional_rewards: Tensor of 8 rewards (one per direction)
            - multistep_triggered: Boolean indicating if multistep was used
        """

        # Stage 1: Single-step evaluation for all 8 directions
        single_step_rewards = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)

        for direction in range(self.n_hd):
            activations = self.preplay_from_state(
                self.place_cell_activations, direction, num_steps=1
            )
            single_step_rewards[direction] = self._evaluate_activations_for_reward(activations)

        # Compute gradient across all directions
        gradient = torch.sum(torch.abs(torch.diff(single_step_rewards))).item()

        if debug:
            print(f"    [SELECTIVE_SOFTMAX] Single-step gradient: {gradient:.4f}, threshold: {gradient_threshold}")

        # If gradient is sufficient, return single-step results
        if gradient >= gradient_threshold:
            if debug:
                print(f"    [SELECTIVE_SOFTMAX] Using single-step (gradient sufficient)")
            return single_step_rewards, False

        # Stage 2: Global multistep preplay
        if debug:
            print(f"    [SELECTIVE_SOFTMAX] Gradient insufficient, triggering global multistep preplay")

        # Always use max depth when multistep is triggered
        # The gradient threshold already determines when we need deeper exploration
        # Softmax will handle weak signals and find relative differences
        valid_depth = max_steps

        if debug:
            print(f"    [SELECTIVE_SOFTMAX] Using full depth: {valid_depth} steps")

        # Explore all branches starting from all 8 directions
        all_branches = []  # [(initial_dir, path, cumulative_reward), ...]

        def explore_branch(initial_dir, activations, remaining_depth, path, cumulative_reward, depth):
            """Recursively explore branches from a starting direction."""

            # Calculate decay weight for this depth
            step_weight = decay_factor ** depth

            # Evaluate current state with decay
            current_reward = self._evaluate_activations_for_reward(activations)
            weighted_reward = step_weight * current_reward
            total_reward = cumulative_reward + weighted_reward

            if remaining_depth == 0:
                # Leaf node - store this branch
                all_branches.append((initial_dir, path.copy(), total_reward))
                return

            # Determine branching directions
            if len(path) == 0:
                # First step - use the initial direction
                directions = [initial_dir]
            else:
                # Subsequent steps - branch to straight/left/right
                last_dir = path[-1]
                directions = [
                    last_dir,                    # Straight
                    (last_dir - 1) % self.n_hd,  # Left (counter-clockwise)
                    (last_dir + 1) % self.n_hd   # Right (clockwise)
                ]

            for direction in directions:
                next_activations = self.preplay_from_state(activations, direction, num_steps=1)
                new_path = path + [direction]
                explore_branch(initial_dir, next_activations, remaining_depth - 1,
                              new_path, total_reward, depth + 1)

        # Start exploration from all 8 initial directions
        current_activations = self.place_cell_activations.clone()
        for initial_dir in range(self.n_hd):
            explore_branch(initial_dir, current_activations, valid_depth, [], 0.0, 0)

        if debug:
            print(f"    [SELECTIVE_SOFTMAX] Explored {len(all_branches)} total branches")

        # Group by initial direction - take MAX reward for each direction
        # This finds the best possible future for each initial direction
        directional_max_rewards = torch.full((self.n_hd,), -float('inf'),
                                             dtype=self.dtype, device=self.device)

        for initial_dir, _, reward in all_branches:
            # Keep the best (maximum) reward for each initial direction
            directional_max_rewards[initial_dir] = torch.max(
                directional_max_rewards[initial_dir],
                torch.tensor(reward, dtype=self.dtype, device=self.device)
            )

        # Apply softmax with temperature to the 8 directional max rewards
        # This amplifies differences between DIRECTIONS (not individual branches)
        # Result is a probability distribution over the 8 directions
        directional_rewards = torch.nn.functional.softmax(
            directional_max_rewards / softmax_temperature, dim=0
        )

        if debug:
            print(f"    [SELECTIVE_SOFTMAX] Final directional rewards (normalized): {directional_rewards.cpu().numpy()}")

        return directional_rewards, True

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
            print(f"[BOLTZMANN_PREPLAY] Starting multi-scale preplay")
            print(f"[BOLTZMANN_PREPLAY] Scales: {len(scales_data)}, Directions: {self.n_hd}, Steps: {num_steps}")
            print(f"[BOLTZMANN_PREPLAY] Discount: {discount_factor}, Inv-Temp: {inverse_temperature}")

        # Build discount weights for all steps: [gamma^0, gamma^1, ..., gamma^(num_steps-1)]
        discount_weights = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)

        # Store trajectory metadata (scale info and direction indices)
        trajectory_metadata = []

        # Accumulate per-trajectory data as lists of tensors
        all_discounted_returns = []
        all_direction_vectors = []

        # Loop through each scale
        for scale_idx, (scale_name, pcn, rcn) in enumerate(scales_data):
            if debug:
                print(f"\n[BOLTZMANN_PREPLAY] Processing scale {scale_idx}: {scale_name}")

            # Loop through each of the 8 head directions
            for direction in range(self.n_hd):
                # Store per-step rewards and angles for this trajectory
                step_rewards = []
                step_angles = []

                # Simulate preplay for this direction
                current_activations = pcn.place_cell_activations.clone()
                current_direction = direction

                for step in range(num_steps):
                    # Perform one preplay step
                    next_activations = pcn.preplay_from_state(
                        current_activations, current_direction, num_steps=1
                    )

                    # Evaluate reward at this step (keep as tensor)
                    rcn.update_reward_cell_activations(next_activations, visit=False)
                    step_reward = torch.max(torch.nan_to_num(rcn.reward_cell_activations))
                    step_rewards.append(step_reward)

                    # Store step angle (in radians) - trajectory continues in same fixed direction
                    # (no within-trajectory turning in current implementation)
                    step_angle = current_direction * (2 * np.pi / self.n_hd)
                    step_angles.append(step_angle)

                    # Update for next step - continue straight in same direction
                    current_activations = next_activations

                # Stack step rewards into tensor: shape [num_steps]
                step_rewards_tensor = torch.stack(step_rewards)

                # Compute discounted return: sum(gamma^t * r_t)
                discounted_return = torch.sum(discount_weights * step_rewards_tensor)

                # Convert step angles to unit vectors: [cos(angle), sin(angle)]
                step_angles_tensor = torch.tensor(step_angles, dtype=self.dtype, device=self.device)
                step_vectors = torch.stack([torch.cos(step_angles_tensor), torch.sin(step_angles_tensor)], dim=1)  # [num_steps, 2]

                # Compute discounted direction vector: sum(gamma^t * [cos(theta_t), sin(theta_t)])
                direction_vector = torch.sum(discount_weights.unsqueeze(1) * step_vectors, dim=0)  # [2]

                # Store trajectory results
                all_discounted_returns.append(discounted_return)
                all_direction_vectors.append(direction_vector)
                trajectory_metadata.append({
                    'scale_idx': scale_idx,
                    'scale_name': scale_name,
                    'direction': direction
                })

                if debug and direction % 4 == 0:
                    print(f"  Dir {direction} ({direction*45}°): reward={discounted_return.item():.4f}, "
                          f"vector=[{direction_vector[0].item():.3f}, {direction_vector[1].item():.3f}]")

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
            print(f"\n[BOLTZMANN_PREPLAY] Boltzmann probability range: "
                  f"[{torch.min(boltzmann_probs).item():.6f}, {torch.max(boltzmann_probs).item():.6f}]")
            top_indices = torch.topk(boltzmann_probs, min(3, len(boltzmann_probs))).indices
            print("[BOLTZMANN_PREPLAY] Top 3 trajectories:")
            for idx in top_indices:
                traj = trajectory_metadata[idx.item()]
                print(f"  Scale={traj['scale_name']}, Dir={traj['direction']}({traj['direction']*45}°), "
                      f"P={boltzmann_probs[idx].item():.4f}, R={discounted_returns[idx].item():.4f}")

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
                print(f"[BOLTZMANN_PREPLAY] Combined vector near zero (mag={combined_magnitude.item():.2e}), "
                      f"using max-return trajectory {max_idx.item()}")
                print(f"[BOLTZMANN_PREPLAY] Updated expected value to match: {expected_value.item():.4f}")

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
            print(f"\n[BOLTZMANN_PREPLAY] Final direction: {final_direction_deg_tensor.item():.1f}°")
            print(f"[BOLTZMANN_PREPLAY] Expected value: {expected_value.item():.4f}")
            print(f"[BOLTZMANN_PREPLAY] Combined vector: [{combined_vector[0].item():.3f}, {combined_vector[1].item():.3f}]")

        # Return all tensors (no .item() calls), plus metadata
        return (final_direction_deg_tensor.item(), expected_value, combined_vector,
                discounted_returns, direction_vectors, boltzmann_probs, trajectory_metadata)
