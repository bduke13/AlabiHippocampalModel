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
        w_in_init_ratio: float = 0.25,
        w_grid_init_ratio: float = 0.25,
        grid_influence: float = 0.5,  # Default to 50% grid influence
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,  # Parameter for grid cell inhibition
        enable_proximity_suppression: bool = True,
        proximity_threshold_factor: float = 2.0,
        proximity_suppression_steepness: float = 10.0,
        proximity_suppression_midpoint: float = 0.5,
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
            w_in_init_ratio: What proportion of the weights of BVC -> PCN are active initially.
            w_grid_init_ratio: What proportion of the weights of GC -> PCN are active initially.
            grid_influence: Percentage (0.0 to 1.0) of grid cell influence on place cells.
                           0.0 means only BVC input, 1.0 means only grid cell input.
            device: Which device to place the tensors on (e.g., "cpu" or "cuda").
            dtype: PyTorch data type (e.g., torch.float32).
            gamma_pp: Coefficient for place cell recurrent inhibition.
            gamma_pb: Coefficient for boundary vector cell afferent inhibition.
            gamma_pg: Coefficient for grid cell afferent inhibition.
            enable_proximity_suppression: Whether to enable grid cell suppression near walls.
            proximity_threshold_factor: Factor to multiply sigma_r to get suppression threshold.
            proximity_suppression_steepness: Steepness of the sigmoid suppression curve.
            proximity_suppression_midpoint: Midpoint of the sigmoid suppression curve.
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

        # Proximity suppression parameters
        self.enable_proximity_suppression = enable_proximity_suppression
        self.proximity_threshold_factor = proximity_threshold_factor
        self.proximity_suppression_steepness = proximity_suppression_steepness
        self.proximity_suppression_midpoint = proximity_suppression_midpoint

        # For debugging/monitoring proximity effects
        self.last_wall_distance = None
        self.last_proximity_suppression = None
        self.last_effective_grid_influence = None

        # Input weight matrix connecting place cells to BVCs
        # Shape: (num_pc, num_bvc)
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, self.num_bvc))
        w_in_init = torch.tensor(w_in_init, dtype=self.dtype, device=self.device)
        # We wrap in nn.Parameter so the weights can be learnable if needed
        self.w_in = torch.nn.Parameter(w_in_init, requires_grad=False)

        # Input weight matrix connecting place cells to Grid Cells
        # Shape: (num_pc, num_grid_cells)
        if num_grid_cells > 0:
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
        grid_afferent_excitation = self.grid_gain * grid_afferent_excitation
        if self.grid_cell_activations is not None and self.w_grid is not None:
            grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)

        # Compute proximity-based grid influence suppression
        proximity_suppression = self.compute_proximity_suppression(distances_torch)
        effective_grid_influence = self.grid_influence * (1.0 - proximity_suppression)

        # Store for debugging
        self.last_effective_grid_influence = effective_grid_influence

        # Combine BVC and Grid Cell inputs based on the EFFECTIVE grid_influence parameter
        if effective_grid_influence == 0.0:
            # Only use BVC input
            afferent_excitation = bvc_afferent_excitation
        elif effective_grid_influence == 1.0:
            # Only use Grid Cell input
            afferent_excitation = grid_afferent_excitation
        else:
            # Combine BVC and Grid Cell inputs based on effective_grid_influence
            afferent_excitation = (1.0 - effective_grid_influence) * bvc_afferent_excitation + \
                                 effective_grid_influence * grid_afferent_excitation

        # Compute total BVC activity for afferent inhibition
        # Afferent inhibition term from BVCs: Γ^{pb} ∑_j v_j^b
        bvc_afferent_inhibition = self.gamma_pb * torch.sum(self.bvc_activations)

        # Compute total Grid Cell activity for afferent inhibition (if grid cells are used)
        grid_afferent_inhibition = 0.0
        if self.grid_cell_activations is not None:
            grid_afferent_inhibition = self.gamma_pg * torch.sum(self.grid_cell_activations)

        # Combine BVC and Grid Cell inhibition based on the EFFECTIVE grid_influence parameter
        if effective_grid_influence == 0.0:
            # Only use BVC inhibition
            afferent_inhibition = bvc_afferent_inhibition
        elif effective_grid_influence == 1.0:
            # Only use Grid Cell inhibition
            afferent_inhibition = grid_afferent_inhibition
        else:
            # Combine BVC and Grid Cell inhibition based on effective_grid_influence
            afferent_inhibition = (1.0 - effective_grid_influence) * bvc_afferent_inhibition + \
                                 effective_grid_influence * grid_afferent_inhibition

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
        self.place_cell_activations = torch.tanh(torch.relu(self.activation_update))

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

            # Update eligibility trace for head direction cells (if provided)
            if hd_activations is not None:
                # Convert hd_activations to avoid NaNs if needed
                hd_activations_no_nan = torch.nan_to_num(hd_activations_torch)

                # Expand to shape (n_hd, 1, 1)
                hd_activations_no_nan = hd_activations_no_nan.unsqueeze(1).unsqueeze(2)
                self.hd_cell_trace += (self.tau / 3) * (
                    hd_activations_no_nan - self.hd_cell_trace
                )

                # Update recurrent weights for place cell interactions modulated by head direction
                # STDP-like update
                # For simplicity, we handle all HD cells in one go, shape (n_hd, num_pc, num_pc):
                hd_contrib = (
                    torch.nan_to_num(hd_activations_torch).unsqueeze(-1).unsqueeze(-1)
                )

                # place_cell_activations shape: (num_pc,)
                # place_cell_trace shape:       (num_pc,)
                # Outer products: (num_pc x num_pc)
                # We'll replicate across the head direction dimension
                pc_act_mat = torch.ger(self.place_cell_activations, self.place_cell_trace)
                pc_trace_mat = torch.ger(self.place_cell_trace, self.place_cell_activations)

                # Each head direction is used as a scalar factor per slice
                # shape broadcast: (n_hd, 1, 1) * (num_pc, num_pc) => (n_hd, num_pc, num_pc)
                update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)
                self.w_rec_tripartite += update_rec.type(self.dtype)

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

    def compute_proximity_suppression(self, distances):
        """Calculate how much to suppress grid cell influence based on wall proximity.

        Args:
            distances: Tensor of distance readings from sensors

        Returns:
            float: Suppression factor (0.0 = no suppression, 1.0 = full suppression)
        """
        if not self.enable_proximity_suppression:
            return 0.0

        # Find minimum distance to any wall
        wall_distance = torch.min(distances).item()

        # Scale-aware proximity threshold
        proximity_threshold = self.proximity_threshold_factor * self.bvc_layer.sigma_r.item()

        # Store for debugging
        self.last_wall_distance = wall_distance

        if wall_distance >= proximity_threshold:
            # Far from walls - no suppression
            suppression = 0.0
        else:
            # Close to walls - apply smooth sigmoid suppression
            normalized_distance = wall_distance / proximity_threshold
            sigmoid_input = self.proximity_suppression_steepness * (normalized_distance - self.proximity_suppression_midpoint)
            suppression = 1.0 - torch.sigmoid(torch.tensor(sigmoid_input, dtype=self.dtype, device=self.device)).item()

        # Store for debugging
        self.last_proximity_suppression = suppression
        return suppression

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
