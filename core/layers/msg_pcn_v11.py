import numpy as np
import torch
from numpy.random import default_rng
from typing import Optional
import math
import torch.nn.functional as F

# Set a fixed seed for reproducibility
torch.manual_seed(5)


class PlaceCellLayer:
    """
    Place cell network for spatial representation learning.
    
    Integrates boundary vector cells and grid cells to learn place fields
    through competitive learning and spike-timing dependent plasticity.
    Supports adaptive learning rates, correlation-based connection weighting,
    and proximity-based grid cell suppression for robust spatial mapping.
    """
    
    def __init__(self, bvc_layer, **config):
        """
        Initialize place cell layer with configuration dictionary.
        
        Args:
            bvc_layer: Boundary vector cell layer instance
            **config: Configuration dictionary containing all parameters
                     from the driver's scale configuration
        """
        # Set up random generator for initialization
        rng = default_rng()
        
        # Store configuration and extract device/dtype
        self.config = config
        self.device = config.get('device', torch.device("cpu"))
        self.dtype = config.get('dtype', torch.float32)
        
        # Core network parameters
        self.num_pc = config['num_pc']  # Required parameter
        self.bvc_layer = bvc_layer
        self.num_bvc = self.bvc_layer.num_bvc
        self.num_grid_cells = config.get('num_grid_cells', 0)
        self.n_hd = config.get('num_hd', 8)
        
        if self.n_hd % 2 != 0:
            raise ValueError(f"num_hd must be even for bidirectional connections, got {self.n_hd}")
        
        # Timing parameters
        timestep = config.get('timestep', 96)
        self.tau = timestep / 1000.0  # Convert to seconds
        self.tau_p = 0.5
        
        # Network dynamics parameters
        self.grid_influence = config.get('grid_influence', 0.3)
        self.gamma_pp = config.get('gamma_pp', 0.9)
        self.gamma_pb = config.get('gamma_pb', 0.2)
        self.gamma_pg = config.get('gamma_pg', 0.3)
        
        # Learning parameters
        self.enable_ojas = config.get('enable_ojas', False)
        self.enable_stdp = config.get('enable_stdp', False)
        self.stdp_learning_rate = config.get('stdp_lr', 0.05)
        self.tau_hd = config.get('tau_hd', 0.5)
        
        # Adaptive learning parameters
        self.enable_adaptive_stdp = config.get('enable_adaptive_stdp', True)
        self.adaptive_initial_lr = config.get('adaptive_initial_lr', 0.10)
        self.adaptive_final_lr = config.get('adaptive_final_lr', 0.03)
        self.adaptive_decay_rate = config.get('adaptive_decay_rate', 60)
        
        # Connection decay parameters
        self.enable_connection_decay = config.get('enable_connection_decay', True)
        self.connection_decay_rate = config.get('connection_decay_rate', 0.00005)
        
        # Correlation-based weighting parameters
        self.enable_correlation_weighting = config.get('enable_correlation_weighting', True)
        self.correlation_window = config.get('correlation_window', 12)
        self.correlation_update_freq = config.get('correlation_update_freq', 10)
        self.correlation_scaling = config.get('correlation_scaling', 2.0)
        self.min_correlation_weight = config.get('min_correlation_weight', 0.1)
        self.correlation_threshold = config.get('correlation_threshold', 0.01)
        
        # Proximity suppression parameters
        self.enable_proximity_suppression = config.get('enable_proximity_suppression', True)
        self.proximity_threshold_factor = config.get('proximity_threshold_factor', 1.0)
        self.proximity_suppression_steepness = config.get('proximity_suppression_steepness', 10.0)
        self.proximity_suppression_midpoint = config.get('proximity_suppression_midpoint', 0.5)
        
        # Scale identification for debugging
        self.scale_name = config.get('name', 'unknown')
        self.enable_debug_prints = config.get('enable_debug_prints', False)
        
        # Initialize network weights
        self._initialize_weights(rng, config)
        
        # Initialize state variables
        self._initialize_state_variables()
        
        # Initialize learning mechanisms
        self._initialize_learning_mechanisms()
        
        print(f"[PCN] Initialized {self.scale_name} scale: {self.num_pc} place cells")
        if self.enable_adaptive_stdp:
            print(f"[PCN] Adaptive STDP: {self.adaptive_initial_lr} → {self.adaptive_final_lr}")

    def _initialize_weights(self, rng, config):
        """Initialize synaptic weight matrices."""
        # Input weights from BVCs to place cells
        w_in_init_ratio = config.get('w_in_init_ratio', 0.25)
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(self.num_pc, self.num_bvc))
        self.w_in = torch.nn.Parameter(
            torch.tensor(w_in_init, dtype=self.dtype, device=self.device),
            requires_grad=False
        )
        
        # Input weights from grid cells to place cells (if grid cells exist)
        if self.num_grid_cells > 0:
            w_grid_init_ratio = config.get('w_grid_init_ratio', 0.25)
            w_grid_init = rng.binomial(n=1, p=w_grid_init_ratio, size=(self.num_pc, self.num_grid_cells))
            self.w_grid = torch.nn.Parameter(
                torch.tensor(w_grid_init, dtype=self.dtype, device=self.device),
                requires_grad=False
            )
        else:
            self.w_grid = None
        
        # Recurrent weight matrix for head direction and place cell interactions
        self.w_rec_tripartite = torch.zeros(
            (self.n_hd, self.num_pc, self.num_pc), dtype=self.dtype, device=self.device
        )
        
        # Store initial weights for reference
        self.initial_w_in = torch.clone(self.w_in.data)
        if self.w_grid is not None:
            self.initial_w_grid = torch.clone(self.w_grid.data)

    def _initialize_state_variables(self):
        """Initialize activation and state variables."""
        # Place cell activations
        self.place_cell_activations = torch.zeros(
            self.num_pc, dtype=self.dtype, device=self.device
        )
        
        # Activation update variable
        self.activation_update = torch.zeros_like(self.place_cell_activations)
        
        # BVC activations
        self.bvc_activations = torch.zeros(
            self.num_bvc, dtype=self.dtype, device=self.device
        )
        
        # Grid cell activations (if applicable)
        if self.num_grid_cells > 0:
            self.grid_cell_activations = torch.zeros(
                self.num_grid_cells, dtype=self.dtype, device=self.device
            )
        else:
            self.grid_cell_activations = None
        
        # Eligibility traces
        self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
        self.hd_cell_trace = torch.zeros(
            (self.n_hd, 1, 1), dtype=self.dtype, device=self.device
        )
        
        # Proximity state variables
        self.last_wall_distance = None
        self.last_proximity_suppression = None
        self.last_effective_grid_influence = None

    def _initialize_learning_mechanisms(self):
        """Initialize learning-related data structures."""
        # Normalization factors for synaptic updates
        self.alpha_pb = np.sqrt(0.5)
        self.alpha_pg = np.sqrt(0.5)
        
        # Correlation tracking
        self.activation_history = []
        self.correlation_matrix = torch.ones(
            self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
        ) * 0.5
        self.correlation_step_counter = 0
        
        # Adaptive learning tracking
        if self.enable_adaptive_stdp:
            self.connection_strength_cache = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
            self.strength_update_counter = 0
            self.strength_update_frequency = 10

    def get_place_cell_activations(
        self,
        distances: np.ndarray,
        angles: Optional[np.ndarray] = None,
        grid_activations: Optional[torch.Tensor] = None,
        hd_activations: Optional[np.ndarray] = None,
        collided: bool = False,
    ):
        """
        Compute place cell activations based on sensory input and learning state.

        Integrates boundary vector cell and grid cell inputs, applies inhibition,
        and updates synaptic weights through competitive learning and STDP.

        Args:
            distances: 1D array of LiDAR distance readings
            angles: 1D array of angles corresponding to LiDAR readings (optional, auto-generated if None)
            grid_activations: Grid cell activations (optional)
            hd_activations: Head direction cell activations (optional)
            collided: Whether the agent has collided with an obstacle
        """
        # Apply connection decay at the beginning of each timestep
        if self.enable_stdp:
            self.apply_connection_decay()
        
        # Convert inputs to torch tensors
        if isinstance(distances, torch.Tensor):
            distances_torch = distances.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            distances_torch = torch.tensor(distances, dtype=self.dtype, device=self.device)

        # Handle angles - generate if not provided
        if angles is None:
            # Auto-generate angles based on distance array length (lidar resolution)
            angles = np.linspace(0, 2 * np.pi, len(distances), endpoint=False)

        if isinstance(angles, torch.Tensor):
            angles_torch = angles.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            angles_torch = torch.tensor(angles, dtype=self.dtype, device=self.device)

        if hd_activations is not None:
            if isinstance(hd_activations, torch.Tensor):
                hd_activations_torch = hd_activations.clone().detach().to(dtype=self.dtype, device=self.device)
            else:
                hd_activations_torch = torch.as_tensor(hd_activations, dtype=self.dtype, device=self.device)
        else:
            hd_activations_torch = None

        # Update head direction eligibility traces
        self.update_hd_eligibility_trace(hd_activations_torch)

        # Compute BVC activations
        self.bvc_activations = self.bvc_layer.get_bvc_activation(distances=distances_torch, angles=angles_torch)
        
        # Store grid cell activations if provided
        if grid_activations is not None and self.num_grid_cells > 0:
            self.grid_cell_activations = grid_activations.to(dtype=self.dtype, device=self.device)
        
        # Compute afferent excitation from BVCs
        bvc_afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)
        
        # Compute afferent excitation from grid cells
        grid_afferent_excitation = torch.zeros_like(bvc_afferent_excitation)
        if self.grid_cell_activations is not None and self.w_grid is not None:
            grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)
        
        # Compute proximity-based grid influence suppression
        proximity_suppression = self.compute_proximity_suppression(distances_torch)
        effective_grid_influence = self.grid_influence * (1.0 - proximity_suppression)
        self.last_effective_grid_influence = effective_grid_influence
        
        # Combine BVC and grid cell inputs based on effective grid influence
        if effective_grid_influence == 0.0:
            afferent_excitation = bvc_afferent_excitation
        elif effective_grid_influence == 1.0:
            afferent_excitation = grid_afferent_excitation
        else:
            afferent_excitation = ((1.0 - effective_grid_influence) * bvc_afferent_excitation + 
                                 effective_grid_influence * grid_afferent_excitation)
        
        # Compute afferent inhibition
        bvc_afferent_inhibition = self.gamma_pb * torch.sum(self.bvc_activations)
        grid_afferent_inhibition = 0.0
        if self.grid_cell_activations is not None:
            grid_afferent_inhibition = self.gamma_pg * torch.sum(self.grid_cell_activations)
        
        if effective_grid_influence == 0.0:
            afferent_inhibition = bvc_afferent_inhibition
        elif effective_grid_influence == 1.0:
            afferent_inhibition = grid_afferent_inhibition
        else:
            afferent_inhibition = ((1.0 - effective_grid_influence) * bvc_afferent_inhibition + 
                                 effective_grid_influence * grid_afferent_inhibition)
        
        # Compute recurrent inhibition
        recurrent_inhibition = self.gamma_pp * torch.sum(self.place_cell_activations)
        
        # Update activations using membrane potential dynamics
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )
        
        # Apply ReLU then tanh activation
        self.place_cell_activations = torch.tanh(torch.relu(self.activation_update))
        
        # Update correlation tracking
        self.update_correlation_tracking(self.place_cell_activations)
        
        # Apply STDP learning if enabled and conditions are met
        if (self.enable_stdp and 
            torch.any(self.place_cell_activations != 0) and 
            not collided and 
            hd_activations_torch is not None):
            
            self._apply_stdp_learning(hd_activations_torch)
        
        # Apply Oja's rule if enabled
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            self._apply_ojas_learning(effective_grid_influence)

    def _apply_stdp_learning(self, hd_activations_torch):
        """Apply spike-timing dependent plasticity learning."""
        # Update place cell eligibility traces
        if self.place_cell_trace is None:
            self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
        self.place_cell_trace += (self.tau / 3) * (
            self.place_cell_activations - self.place_cell_trace
        )
        
        # Use HD eligibility traces for STDP
        hd_contrib = self.hd_cell_trace  # Shape: (n_hd, 1, 1)
        
        # Compute raw STDP update
        pc_act_mat = torch.outer(self.place_cell_activations, self.place_cell_trace)
        pc_trace_mat = torch.outer(self.place_cell_trace, self.place_cell_activations)
        raw_update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)
        
        # Apply correlation-based weighting
        weighted_update_rec = self.apply_correlation_weighting_to_stdp(raw_update_rec)
        
        # Apply learning rate scaling (adaptive or fixed)
        if self.enable_adaptive_stdp:
            scaled_update_rec = self._apply_adaptive_learning_rates(weighted_update_rec)
        else:
            scaled_update_rec = self.stdp_learning_rate * weighted_update_rec
        
        # Apply final update
        self.w_rec_tripartite += scaled_update_rec.type(self.dtype)

    def _apply_adaptive_learning_rates(self, weighted_update_rec):
        """Apply adaptive learning rates per head direction."""
        # Update connection strengths periodically for performance
        self.strength_update_counter += 1
        if self.strength_update_counter % self.strength_update_frequency == 0:
            for direction in range(self.n_hd):
                self.connection_strength_cache[direction] = self._compute_connection_strength(
                    self.w_rec_tripartite[direction]
                )
        
        adaptive_scaled_update = torch.zeros_like(weighted_update_rec)
        
        for direction in range(self.n_hd):
            # Use cached connection strength for performance
            connection_strength = self.connection_strength_cache[direction].item()
            
            # Get adaptive learning rate for this direction
            adaptive_lr = self._get_adaptive_learning_rate(connection_strength)
            
            # Apply direction-specific learning rate
            adaptive_scaled_update[direction] = adaptive_lr * weighted_update_rec[direction]
        
        return adaptive_scaled_update

    def _apply_ojas_learning(self, effective_grid_influence):
        """Apply Oja's competitive learning rule."""
        # BVC weight updates
        if effective_grid_influence < 1.0:
            pc_activations_col = self.place_cell_activations.unsqueeze(1)
            bvc_activations_row = self.bvc_activations.unsqueeze(0)
            
            weight_update_bvc = self.tau * (
                pc_activations_col * (
                    bvc_activations_row - 
                    (1 / self.alpha_pb) * pc_activations_col * self.w_in
                )
            ) * (1.0 - effective_grid_influence)
            
            with torch.no_grad():
                self.w_in += weight_update_bvc
        
        # Grid cell weight updates
        if (effective_grid_influence > 0.0 and 
            self.grid_cell_activations is not None and 
            self.w_grid is not None):
            
            pc_activations_col = self.place_cell_activations.unsqueeze(1)
            grid_activations_row = self.grid_cell_activations.unsqueeze(0)
            
            weight_update_grid = self.tau * (
                pc_activations_col * (
                    grid_activations_row - 
                    (1 / self.alpha_pg) * pc_activations_col * self.w_grid
                )
            ) * effective_grid_influence
            
            with torch.no_grad():
                self.w_grid += weight_update_grid

    def reset_activations(self):
        """Reset all activation states and eligibility traces."""
        self.place_cell_activations.zero_()
        self.activation_update.zero_()
        self.place_cell_trace = None
        self.hd_cell_trace.zero_()
        
        # Reset adaptive learning cache
        if hasattr(self, 'connection_strength_cache'):
            self.connection_strength_cache.zero_()
            self.strength_update_counter = 0
        
        # Reset correlation tracking
        self.activation_history = []
        self.correlation_matrix = torch.ones(
            self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
        ) * 0.5
        self.correlation_step_counter = 0

    def preplay(self, direction: int, num_steps: int = 1) -> torch.Tensor:
        """
        Simulate preplay using learned recurrent connections.
        
        Args:
            direction: Head direction index for exploiting connections
            num_steps: Number of preplay steps to simulate
        
        Returns:
            Updated place cell activations after preplay
        """
        place_cell_activations = self.place_cell_activations.clone()
        
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()
            
            # Use directional connections from the specified HD slice
            updated = torch.matmul(
                self.w_rec_tripartite[direction], previous_activations
            )
            
            updated = updated - previous_activations
            place_cell_activations = torch.tanh(torch.relu(updated))
        
        return place_cell_activations

    def preplay_from_state(self, starting_activations: torch.Tensor, direction: int, num_steps: int = 1) -> torch.Tensor:
        """
        Perform preplay starting from arbitrary place cell activation state.
        
        Args:
            starting_activations: Place cell activations to start from
            direction: Head direction index for connections
            num_steps: Number of preplay steps
        
        Returns:
            Final place cell activations after preplay
        """
        place_cell_activations = starting_activations.clone()
        
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()
            
            updated = torch.matmul(
                self.w_rec_tripartite[direction], previous_activations
            )
            
            updated = updated - previous_activations
            place_cell_activations = torch.tanh(torch.relu(updated))
        
        return place_cell_activations

    def multi_step_preplay(self, forced_first_direction: int, max_steps: int = 3, decay_factor: float = 0.6) -> float:
        """
        Multi-step preplay with exponential step weighting and limited branching.
        
        Args:
            forced_first_direction: Direction for first step
            max_steps: Maximum number of preplay steps
            decay_factor: Exponential decay for step weighting
        
        Returns:
            Best weighted total reward for paths starting in forced direction
        """
        current_activations = self.place_cell_activations.clone()
        all_paths_evaluated = []
        
        def evaluate_path_recursive(activations, remaining_steps, path_so_far, depth=0, accumulated_reward=0.0):
            """Recursively evaluate paths with exponential step weighting."""
            # Calculate weight for this step
            step_weight = decay_factor ** depth
            
            # Evaluate reward at current state and apply step weight
            current_step_reward = self._evaluate_activations_for_reward(activations)
            weighted_step_reward = step_weight * current_step_reward
            total_reward_so_far = accumulated_reward + weighted_step_reward
            
            if remaining_steps == 0:
                all_paths_evaluated.append((path_so_far.copy(), total_reward_so_far))
                return total_reward_so_far
            
            # Determine directions to explore
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
                
                # Recursively evaluate the rest of the path
                new_path = path_so_far + [direction]
                reward = evaluate_path_recursive(
                    next_activations, remaining_steps - 1, new_path, depth + 1, total_reward_so_far
                )
                
                best_reward_from_here = max(best_reward_from_here, reward)
            
            return best_reward_from_here
        
        # Start evaluation from forced direction
        best_reward = evaluate_path_recursive(current_activations, max_steps, [], 0, 0.0)
        return best_reward

    def multi_step_preplay_final_state(self, direction: int, max_steps: int = 3) -> torch.Tensor:
        """
        Perform multi-step preplay and return final place cell activations.
        Used for goal cell detection in exploitation algorithms.
        
        Args:
            direction: Starting direction for preplay
            max_steps: Number of steps to look ahead
            
        Returns:
            Final place cell activations after multi-step preplay
        """
        current_activations = self.place_cell_activations.clone()
        
        def simulate_best_path(activations, remaining_steps, forced_first_dir=None):
            """Recursively find and simulate the best path."""
            if remaining_steps == 0:
                return activations
            
            if forced_first_dir is not None:
                # First step: use forced direction
                directions_to_try = [forced_first_dir]
            else:
                # Continue in same direction (simplified for efficiency)
                directions_to_try = [direction]
            
            best_reward = -float('inf')
            best_final_state = activations
            
            for next_dir in directions_to_try:
                # Simulate one step
                step_activations = self.preplay_from_state(activations, next_dir, num_steps=1)
                
                # Recursively evaluate remaining steps
                final_state = simulate_best_path(step_activations, remaining_steps - 1)
                
                # Evaluate this path's final reward
                path_reward = self._evaluate_activations_for_reward(final_state)
                
                if path_reward > best_reward:
                    best_reward = path_reward
                    best_final_state = final_state
            
            return best_final_state
        
        # Start simulation with forced first direction
        final_activations = simulate_best_path(current_activations, max_steps, direction)
        return final_activations

    def _evaluate_activations_for_reward(self, activations: torch.Tensor) -> float:
        """
        Placeholder method for evaluating reward from place cell activations.
        This gets overridden by the driver during exploitation to use the RCN.
        
        Args:
            activations: Place cell activations to evaluate
            
        Returns:
            Estimated reward value
        """
        # Default implementation: sum of activations as a proxy
        return activations.sum().item()

    def apply_connection_decay(self):
        """Apply synaptic decay to recurrent connections for homeostasis."""
        if not self.enable_connection_decay:
            return
        # Exponential decay: w *= (1 - decay_rate)
        self.w_rec_tripartite *= (1 - self.connection_decay_rate)

    def update_correlation_tracking(self, pc_activations):
        """
        Update correlation tracking with current place cell activations.
        
        Args:
            pc_activations: Current place cell activations tensor
        """
        if not self.enable_correlation_weighting:
            return
        
        # Apply threshold to activations before adding to history
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
            
            # Handle NaN values
            self.correlation_matrix = torch.nan_to_num(
                self.correlation_matrix, 
                nan=0.0, posinf=1.0, neginf=-1.0
            )
            
            # Ensure diagonal is 1.0 (self-correlation)
            diagonal_indices = torch.arange(self.num_pc, device=self.device)
            self.correlation_matrix[diagonal_indices, diagonal_indices] = 1.0
            
        except Exception as e:
            # Fallback to neutral correlations if computation fails
            if self.enable_debug_prints:
                print(f"[PCN] Correlation computation failed: {e}")
            self.correlation_matrix = torch.ones(
                self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
            ) * 0.5

    def get_correlation_weights(self):
        """
        Convert correlation matrix to connection weights.
        
        Returns:
            Weight matrix (num_pc, num_pc) with values in [min_weight, 1.0]
        """
        if not self.enable_correlation_weighting:
            return torch.ones(self.num_pc, self.num_pc, dtype=self.dtype, device=self.device)
        
        # Sigmoid transformation for smooth mapping from correlation to weights
        sigmoid_corr = torch.sigmoid(self.correlation_scaling * self.correlation_matrix)
        
        # Scale to [min_correlation_weight, 1.0] range
        weights = (self.min_correlation_weight + 
                  (1.0 - self.min_correlation_weight) * sigmoid_corr)
        
        return weights

    def apply_correlation_weighting_to_stdp(self, connection_update):
        """
        Apply correlation-based weighting to STDP updates.
        
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

    def update_hd_eligibility_trace(self, hd_activations):
        """
        Update head direction eligibility traces with proper time constant.
        
        Args:
            hd_activations: Current head direction activations tensor (n_hd,)
        """
        if hd_activations is None:
            return
        
        # Convert to proper shape and handle NaN values
        hd_activations_clean = torch.nan_to_num(hd_activations)
        hd_activations_expanded = hd_activations_clean.unsqueeze(1).unsqueeze(2)  # (n_hd, 1, 1)
        
        # Update eligibility trace with proper time constant
        self.hd_cell_trace += (self.tau / self.tau_hd) * (
            hd_activations_expanded - self.hd_cell_trace
        )

    def compute_proximity_suppression(self, distances):
        """
        Calculate how much to suppress grid cell influence based on wall proximity.
        
        Grid cells provide global spatial context but can be inaccurate near boundaries.
        This method detects wall proximity and reduces grid influence accordingly.
        
        Args:
            distances: LiDAR distance readings
            
        Returns:
            Suppression factor [0, 1] where 1 = full suppression
        """
        if not self.enable_proximity_suppression:
            return 0.0
        
        # Find minimum distance to any wall
        wall_distance = torch.min(distances).item()
        self.last_wall_distance = wall_distance
        
        # Scale-aware proximity threshold  
        sigma_r_val = self.bvc_layer.sigma_r
        if isinstance(sigma_r_val, torch.Tensor):
            sigma_r_val = sigma_r_val.item()
        proximity_threshold = self.proximity_threshold_factor * float(sigma_r_val)
        
        if wall_distance >= proximity_threshold:
            # Far from walls - no suppression
            suppression = 0.0
        else:
            # Close to walls - apply smooth sigmoid suppression
            normalized_distance = wall_distance / proximity_threshold
            sigmoid_input = self.proximity_suppression_steepness * (
                normalized_distance - self.proximity_suppression_midpoint
            )
            suppression = 1.0 - torch.sigmoid(
                torch.tensor(sigmoid_input, dtype=self.dtype, device=self.device)
            ).item()
        
        self.last_proximity_suppression = suppression
        return suppression

    def _compute_connection_strength(self, direction_connections):
        """
        Compute connection development measure for adaptive learning rate.
        
        Args:
            direction_connections: Connection matrix for single direction (num_pc, num_pc)
            
        Returns:
            Connection strength measure for determining learning rate
        """
        abs_connections = torch.abs(direction_connections)
        significance_threshold = 0.0001  # Threshold for early detection
        significant_mask = abs_connections > significance_threshold
        
        if torch.sum(significant_mask) > 0:
            return torch.mean(abs_connections[significant_mask]).item()
        else:
            return 0.0

    def _get_adaptive_learning_rate(self, connection_strength):
        """
        Compute adaptive learning rate based on connection strength.
        
        Args:
            connection_strength: Connection development measure
            
        Returns:
            Learning rate for this direction (between final_lr and initial_lr)
        """
        if not self.enable_adaptive_stdp:
            return self.stdp_learning_rate
        
        # Exponential decay from initial_lr to final_lr
        import math
        decay_factor = math.exp(-connection_strength * self.adaptive_decay_rate)
        lr = self.adaptive_final_lr + (self.adaptive_initial_lr - self.adaptive_final_lr) * decay_factor
        
        return lr

    def get_basic_stats(self):
        """
        Get essential statistics about the place cell network state.
        
        Returns:
            Dictionary with basic network statistics
        """
        stats = {
            "scale_name": self.scale_name,
            "num_place_cells": self.num_pc,
            "current_activation_sum": torch.sum(self.place_cell_activations).item(),
            "current_activation_max": torch.max(self.place_cell_activations).item(),
            "enable_adaptive_stdp": self.enable_adaptive_stdp,
            "grid_influence": self.grid_influence,
            "effective_grid_influence": getattr(self, 'last_effective_grid_influence', None),
        }
        
        if self.enable_adaptive_stdp and hasattr(self, 'connection_strength_cache'):
            # Add adaptive learning statistics
            avg_strength = torch.mean(self.connection_strength_cache).item()
            max_strength = torch.max(self.connection_strength_cache).item()
            
            stats.update({
                "avg_connection_strength": avg_strength,
                "max_connection_strength": max_strength,
                "avg_learning_rate": self._get_adaptive_learning_rate(avg_strength),
                "max_learning_rate": self._get_adaptive_learning_rate(max_strength),
            })
        
        if self.enable_correlation_weighting:
            stats.update({
                "correlation_history_length": len(self.activation_history),
                "correlation_matrix_mean": self.correlation_matrix.mean().item(),
            })
        
        return stats