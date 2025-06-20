import numpy as np
import torch
from numpy.random import default_rng
from typing import Optional

# Set a fixed seed similar to tf.random.set_seed(5)
torch.manual_seed(5)


class MultiscalePlaceCellWithGridV65:
    """Model a layer of place cells receiving input from both Boundary Vector Cells and Grid Cells.
    
    V6.5 Changes (Adaptive Learning Rate Implementation):
    - NEW: Adaptive STDP learning rate that starts high and naturally decays as connections develop
    - NEW: Simple exponential decay mechanism with minimal parameters (initial_lr, final_lr, decay_rate)
    - NEW: Direction-specific learning rate adaptation based on connection strength
    - MAINTAINED: All V6 improvements (HD eligibility traces, proper learning rate scaling)
    - MAINTAINED: All V5 correlation-based weighting functionality
    - MAINTAINED: All V4 connection decay functionality
    
    This version provides natural learning rate adaptation without fixed thresholds or phases,
    making it robust across different environments, timescales, and place cell configurations.
    
    Place cells develop spatially localized receptive fields (place fields) through
    competitive learning and synaptic plasticity with adaptive connection learning.
    """
    def __init__(
        self,
        bvc_layer,
        num_pc: int = 200,
        num_grid_cells: int = 200,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.25,
        w_grid_init_ratio: float = 0.25,
        grid_influence: float = 0.5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,
        # V4 Connection decay parameters (kept)
        enable_connection_decay: bool = True,
        connection_decay_rate: float = 0.002,
        # V5 Correlation-based weighting parameters (kept)
        enable_correlation_weighting: bool = True,
        correlation_window: int = 100,
        correlation_update_freq: int = 10,
        correlation_scaling: float = 2.0,
        min_correlation_weight: float = 0.1,
        correlation_threshold: float = 0.01,
        # V6 STDP learning rate parameter (kept for backward compatibility)
        stdp_learning_rate: float = 0.01,
        # V6 HD eligibility trace time constant (kept)
        tau_hd: float = 0.1,
        # V6.5 NEW: Adaptive learning rate parameters
        enable_adaptive_stdp: bool = False,
        adaptive_initial_lr: float = 0.15,
        adaptive_final_lr: float = 0.02,
        adaptive_decay_rate: float = 3.0,
        # V6.5 NEW: Scale name for debugging
        scale_name: str = "unknown",
        # V6.5 NEW: Logging and debugging toggles
        enable_debug_prints: bool = False,
        enable_adaptive_logging: bool = False,
    ):
        """Initialize the Place Cell Layer V6.5 with adaptive STDP learning rates.
        
        Args:
            bvc_layer: The BVC layer used as input to place cell activations.
            num_pc: Number of place cells in the layer.
            num_grid_cells: Number of grid cells providing input.
            timestep: Time step for simulation/learning updates in milliseconds.
            n_hd: Number of head direction cells.
            enable_ojas: Enable weight updates via competition.
            enable_stdp: Enable tripartite synapse weight updates via STDP.
            w_in_init_ratio: What proportion of the weights of BVC -> PCN are active initially.
            w_grid_init_ratio: What proportion of the weights of GC -> PCN are active initially.
            grid_influence: Percentage (0.0 to 1.0) of grid cell influence on place cells.
            device: Which device to place the tensors on.
            dtype: PyTorch data type.
            gamma_pp: Coefficient for place cell recurrent inhibition.
            gamma_pb: Coefficient for boundary vector cell afferent inhibition.
            gamma_pg: Coefficient for grid cell afferent inhibition.
            enable_connection_decay: Whether to enable synaptic decay mechanism.
            connection_decay_rate: Rate of synaptic decay per timestep.
            enable_correlation_weighting: Whether to use correlation-based connection weighting.
            correlation_window: Number of timesteps of activation history to track.
            correlation_update_freq: How often to update correlation matrix (every N timesteps).
            correlation_scaling: Scaling factor for sigmoid transformation of correlations.
            min_correlation_weight: Minimum connection weight for uncorrelated cells.
            correlation_threshold: Minimum activation level to include in correlation tracking.
            stdp_learning_rate: Fixed learning rate for STDP updates (used when adaptive disabled).
            tau_hd: Time constant for HD eligibility trace updates.
            enable_adaptive_stdp: Whether to use adaptive learning rate instead of fixed rate.
            adaptive_initial_lr: Starting learning rate for adaptive STDP.
            adaptive_final_lr: Final (minimum) learning rate for adaptive STDP.
            adaptive_decay_rate: Rate of exponential decay from initial to final learning rate.
            scale_name: Name of the scale for debugging purposes.
            enable_debug_prints: Whether to print debug information during STDP updates.
            enable_adaptive_logging: Whether to log connection strengths and learning rates over time.
        """
        # Set up random generator for binomial initialization
        rng = default_rng()
        
        self.device = device
        self.dtype = dtype
        self.n_hd = n_hd

        if self.n_hd % 2 != 0:
            raise ValueError(f"n_hd must be even for bidirectional connections, got {self.n_hd}")
        
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
        
        # V4 Connection decay parameters (kept from V4)
        self.enable_connection_decay = enable_connection_decay
        self.connection_decay_rate = connection_decay_rate
        
        # V5 Correlation-based weighting parameters (kept from V5)
        self.enable_correlation_weighting = enable_correlation_weighting
        self.correlation_window = correlation_window
        self.correlation_update_freq = correlation_update_freq
        self.correlation_scaling = correlation_scaling
        self.min_correlation_weight = min_correlation_weight
        self.correlation_threshold = correlation_threshold
        
        # V6 STDP learning parameters (kept from V6)
        self.stdp_learning_rate = stdp_learning_rate
        self.tau_hd = tau_hd
        
        # V6.5 NEW: Adaptive learning rate parameters
        self.enable_adaptive_stdp = enable_adaptive_stdp
        self.adaptive_initial_lr = adaptive_initial_lr
        self.adaptive_final_lr = adaptive_final_lr
        self.adaptive_decay_rate = adaptive_decay_rate
        
        # V6.5 NEW: Store scale name for debugging
        self.scale_name = scale_name
        
        # V6.5 NEW: Debugging and logging controls
        self.enable_debug_prints = enable_debug_prints
        self.enable_adaptive_logging = enable_adaptive_logging
        
        # V6.5 NEW: Data logging structures
        if self.enable_adaptive_logging:
            self.connection_strength_log = []  # List of [timestep, direction_strengths]
            self.learning_rate_log = []        # List of [timestep, direction_learning_rates] 
            self.weight_magnitude_log = []     # List of [timestep, total_weight_magnitude]
            self.logging_step_counter = 0
        
        # V6.5 NEW: Performance optimization - cache connection strengths
        self.connection_strength_cache = torch.zeros(n_hd, dtype=dtype, device=device)
        self.strength_update_counter = 0
        self.strength_update_frequency = 10  # Update strengths every 10 STDP steps
        
        # Validate parameters
        if adaptive_initial_lr <= 0 or adaptive_initial_lr > 1:
            raise ValueError(f"adaptive_initial_lr must be in (0, 1], got {adaptive_initial_lr}")
        if adaptive_final_lr <= 0 or adaptive_final_lr > 1:
            raise ValueError(f"adaptive_final_lr must be in (0, 1], got {adaptive_final_lr}")
        if adaptive_final_lr >= adaptive_initial_lr:
            raise ValueError(f"adaptive_final_lr ({adaptive_final_lr}) must be < adaptive_initial_lr ({adaptive_initial_lr})")
        if adaptive_decay_rate <= 0:
            raise ValueError(f"adaptive_decay_rate must be positive, got {adaptive_decay_rate}")
        if tau_hd <= 0:
            raise ValueError(f"tau_hd must be positive, got {tau_hd}")
        
        print(f"[PCN_V65] Initialized with adaptive STDP: {self.enable_adaptive_stdp}")
        if self.enable_adaptive_stdp:
            print(f"[PCN_V65] Adaptive learning rate: {self.adaptive_initial_lr} → {self.adaptive_final_lr} (decay: {self.adaptive_decay_rate})")
        else:
            print(f"[PCN_V65] Fixed STDP learning rate: {self.stdp_learning_rate}")
        print(f"[PCN_V65] HD eligibility trace time constant: {self.tau_hd}")
        
        # V5 Correlation tracking data structures (kept from V5)
        self.activation_history = []
        self.correlation_matrix = torch.ones(num_pc, num_pc, dtype=dtype, device=device) * 0.5
        self.correlation_step_counter = 0
        
        # Input weight matrix connecting place cells to BVCs
        # Shape: (num_pc, num_bvc)
        w_in_init = rng.binomial(n=1, p=w_in_init_ratio, size=(num_pc, self.num_bvc))
        w_in_init = torch.tensor(w_in_init, dtype=self.dtype, device=self.device)
        self.w_in = torch.nn.Parameter(w_in_init, requires_grad=False)
        
        # Input weight matrix connecting place cells to Grid Cells
        # Shape: (num_pc, num_grid_cells)
        if num_grid_cells > 0:
            w_grid_init = rng.binomial(n=1, p=w_grid_init_ratio, size=(num_pc, self.num_grid_cells))
            w_grid_init = torch.tensor(w_grid_init, dtype=self.dtype, device=self.device)
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
        
        # Coefficients for inhibition
        self.gamma_pp = gamma_pp
        self.gamma_pb = gamma_pb
        self.gamma_pg = gamma_pg
        
        # Time constant for membrane potential dynamics
        self.tau_p = 0.5
        
        # Normalization factors for synaptic weight updates
        self.alpha_pb = np.sqrt(0.5)
        self.alpha_pg = np.sqrt(0.5) 
        
        # Initial weights backup
        self.initial_w_in = torch.clone(self.w_in.data)
        if self.w_grid is not None:
            self.initial_w_grid = torch.clone(self.w_grid.data)
        
        # Temporary variable for activation updates
        self.activation_update = torch.zeros_like(
            self.place_cell_activations, dtype=self.dtype, device=self.device
        )
        
        # Head direction modulation
        self.head_direction_modulation = None
        
        # Boundary cell activations
        self.boundary_cell_activations = torch.zeros(
            (n_hd, num_pc), dtype=self.dtype, device=self.device
        )
        
        # Eligibility traces
        self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
        
        # V6 HD eligibility traces with proper initialization (kept from V6)
        self.hd_cell_trace = torch.zeros(
            (n_hd, 1, 1), dtype=self.dtype, device=self.device
        )
        
        # Learning enables/disables
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp

    def apply_connection_decay(self):
        """V4: Apply synaptic decay to recurrent connections for homeostasis."""
        if not self.enable_connection_decay:
            return
            
        # Simple exponential decay: w *= (1 - decay_rate)
        self.w_rec_tripartite *= (1 - self.connection_decay_rate)

    def update_correlation_tracking(self, pc_activations):
        """V5: Update correlation tracking with current place cell activations.
        
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
        """V5: Compute correlation matrix from activation history."""
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
            print(f"[PCN_V65] Correlation computation failed: {e}")
            self.correlation_matrix = torch.ones(
                self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
            ) * 0.5

    def get_correlation_weights(self):
        """V5: Convert correlation matrix to connection weights.
        
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
        """V5: Apply correlation-based weighting to STDP updates.
        
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
        """V6: Update head direction eligibility traces with proper time constant.
        
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
        """V6.5 NEW: Compute connection development measure for adaptive learning rate.
        
        Args:
            direction_connections: Connection matrix for a single direction (num_pc, num_pc)
            
        Returns:
            float: Connection strength measure for determining learning rate
        """
        abs_connections = torch.abs(direction_connections)
        
        # Use a much lower threshold to capture early learning
        # This should be roughly equivalent to 1-2 STDP updates
        significance_threshold = 0.0001  # Much lower threshold for early detection
        significant_mask = abs_connections > significance_threshold
        
        if torch.sum(significant_mask) > 0:
            # Return mean strength of significant connections
            return torch.mean(abs_connections[significant_mask]).item()
        else:
            # For completely zero connections, return 0.0 (will use initial_lr)
            return 0.0

    def get_adaptive_learning_rate(self, connection_strength):
        """V6.5 NEW: Compute adaptive learning rate based on connection strength.
        
        Args:
            connection_strength: Connection development measure from compute_connection_strength
            
        Returns:
            float: Learning rate for this direction (between final_lr and initial_lr)
        """
        if not self.enable_adaptive_stdp:
            return self.stdp_learning_rate
        
        # Exponential decay from initial_lr to final_lr
        # lr = final_lr + (initial_lr - final_lr) * exp(-connection_strength * decay_rate)
        # Use math.exp for scalar operations to avoid tensor/scalar mixing
        import math
        decay_factor = math.exp(-connection_strength * self.adaptive_decay_rate)
        lr = self.adaptive_final_lr + (self.adaptive_initial_lr - self.adaptive_final_lr) * decay_factor
        
        return lr

    def get_place_cell_activations(
        self,
        distances: np.ndarray,
        grid_activations: Optional[torch.Tensor] = None,
        hd_activations: Optional[np.ndarray] = None,
        collided: bool = False,
    ):
        """Compute place cell activations with V6.5 adaptive STDP learning rates.
        
        V6.5 Changes:
        - NEW: Adaptive learning rate computation based on connection strength per direction
        - NEW: Smooth exponential decay from high to low learning rates as connections develop
        - MAINTAINED: All V6 improvements (HD eligibility traces, proper learning rate scaling)
        - MAINTAINED: All V5 correlation-based weighting functionality
        - MAINTAINED: All V4 connection decay functionality
        
        Args:
            distances: 1D NumPy array of distance readings.
            grid_activations: 1D tensor of grid cell activations.
            hd_activations: 1D NumPy array of head direction cell activations.
            collided: Whether the agent has collided with an obstacle.
        """
        # V4: Apply connection decay at the beginning of each timestep
        if self.enable_stdp:
            self.apply_connection_decay()
        
        # Convert inputs to torch tensors
        if isinstance(distances, torch.Tensor):
            distances_torch = distances.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            distances_torch = torch.tensor(distances, dtype=self.dtype, device=self.device)
        
        if hd_activations is not None:
            if isinstance(hd_activations, torch.Tensor):
                hd_activations_torch = hd_activations.clone().detach().to(dtype=self.dtype, device=self.device)
            else:
                hd_activations_torch = torch.as_tensor(hd_activations, dtype=self.dtype, device=self.device)
        else:
            hd_activations_torch = None
        
        # V6: Update HD eligibility traces
        self.update_hd_eligibility_trace(hd_activations_torch)
        
        # Compute BVC activations
        self.bvc_activations = self.bvc_layer.get_bvc_activation(distances=distances_torch)
        
        # Store grid cell activations if provided
        if grid_activations is not None and self.num_grid_cells > 0:
            self.grid_cell_activations = grid_activations.to(dtype=self.dtype, device=self.device)
        
        # Compute afferent excitation from BVCs
        bvc_afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)
        
        # Compute afferent excitation from Grid Cells
        grid_afferent_excitation = torch.zeros_like(bvc_afferent_excitation)
        if self.grid_cell_activations is not None and self.w_grid is not None:
            grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)
        
        # Combine BVC and Grid Cell inputs based on grid_influence
        if self.grid_influence == 0.0:
            afferent_excitation = bvc_afferent_excitation
        elif self.grid_influence == 1.0:
            afferent_excitation = grid_afferent_excitation
        else:
            afferent_excitation = (1.0 - self.grid_influence) * bvc_afferent_excitation + \
                                 self.grid_influence * grid_afferent_excitation
        
        # Compute afferent inhibition
        bvc_afferent_inhibition = self.gamma_pb * torch.sum(self.bvc_activations)
        grid_afferent_inhibition = 0.0
        if self.grid_cell_activations is not None:
            grid_afferent_inhibition = self.gamma_pg * torch.sum(self.grid_cell_activations)
        
        if self.grid_influence == 0.0:
            afferent_inhibition = bvc_afferent_inhibition
        elif self.grid_influence == 1.0:
            afferent_inhibition = grid_afferent_inhibition
        else:
            afferent_inhibition = (1.0 - self.grid_influence) * bvc_afferent_inhibition + \
                                 self.grid_influence * grid_afferent_inhibition
        
        # Compute recurrent inhibition
        recurrent_inhibition = self.gamma_pp * torch.sum(self.place_cell_activations)
        
        # Update activations
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )
        
        # Apply ReLU then tanh
        self.place_cell_activations = torch.tanh(torch.relu(self.activation_update))
        
        # V5: Update correlation tracking
        self.update_correlation_tracking(self.place_cell_activations)
        
        # Check STDP updates if enabled and no collision occurred
        if (
            self.enable_stdp
            and torch.any(self.place_cell_activations != 0)
            and not collided
            and hd_activations_torch is not None
        ):
            # Update place cell eligibility traces
            if self.place_cell_trace is None:
                self.place_cell_trace = torch.zeros_like(self.place_cell_activations)
            self.place_cell_trace += (self.tau / 3) * (
                self.place_cell_activations - self.place_cell_trace
            )
            
            # V6: Use HD eligibility traces instead of current HD activations
            hd_contrib = self.hd_cell_trace  # Shape: (n_hd, 1, 1)
            
            # Compute raw STDP update
            pc_act_mat = torch.ger(self.place_cell_activations, self.place_cell_trace)
            pc_trace_mat = torch.ger(self.place_cell_trace, self.place_cell_activations)
            
            # Raw update for all HD slices (before learning rate scaling)
            raw_update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)
            
            # V5: Apply correlation-based weighting to raw update
            weighted_update_rec = self.apply_correlation_weighting_to_stdp(raw_update_rec)
            
            # V6.5 NEW: Apply adaptive learning rate scaling per direction
            if self.enable_adaptive_stdp:
                # Update connection strengths periodically for performance
                self.strength_update_counter += 1
                if self.strength_update_counter % self.strength_update_frequency == 0:
                    for direction in range(self.n_hd):
                        self.connection_strength_cache[direction] = self.compute_connection_strength(
                            self.w_rec_tripartite[direction]
                        )
                
                adaptive_scaled_update = torch.zeros_like(weighted_update_rec)
                
                # Store current learning rates for logging
                current_learning_rates = torch.zeros(self.n_hd)
                
                for direction in range(self.n_hd):
                    # Use cached connection strength for performance
                    connection_strength = self.connection_strength_cache[direction].item()
                    
                    # Get adaptive learning rate for this direction
                    adaptive_lr = self.get_adaptive_learning_rate(connection_strength)
                    current_learning_rates[direction] = adaptive_lr
                    
                    # Apply direction-specific learning rate
                    adaptive_scaled_update[direction] = adaptive_lr * weighted_update_rec[direction]
                
                # V6.5 NEW: Log data if enabled (every 10 steps to reduce storage)
                if self.enable_adaptive_logging and self.strength_update_counter % 10 == 0:
                    self.logging_step_counter += 1
                    
                    # Log connection strengths per direction
                    direction_strengths = self.connection_strength_cache.cpu().numpy().copy()
                    self.connection_strength_log.append([self.logging_step_counter, direction_strengths])
                    
                    # Log learning rates per direction
                    direction_lrs = current_learning_rates.cpu().numpy().copy()
                    self.learning_rate_log.append([self.logging_step_counter, direction_lrs])
                    
                    # Log total weight magnitude
                    total_weight_magnitude = torch.norm(self.w_rec_tripartite).item()
                    self.weight_magnitude_log.append([self.logging_step_counter, total_weight_magnitude])
                
                scaled_update_rec = adaptive_scaled_update
            else:
                # V6: Apply fixed learning rate scaling
                scaled_update_rec = self.stdp_learning_rate * weighted_update_rec
            
            # Apply final update
            self.w_rec_tripartite += scaled_update_rec.type(self.dtype)
            
            # Debug output for first few updates
            if hasattr(self, '_debug_update_count'):
                self._debug_update_count += 1
            else:
                self._debug_update_count = 1
                
            # Debug output every 100 updates (if enabled)
            if self.enable_debug_prints:
                if hasattr(self, '_debug_update_count'):
                    self._debug_update_count += 1
                else:
                    self._debug_update_count = 1
                    
                # Print debug info every 100 steps
                if self._debug_update_count % 100 == 0:
                    print(f"[PCN_V65-{self.scale_name}] STDP Update #{self._debug_update_count}:")
                    if self.enable_adaptive_stdp:
                        # Show adaptive learning rates per direction using cached values
                        for direction in range(self.n_hd):
                            # Ensure cache is initialized for debug output
                            if self.strength_update_counter == 1 or torch.all(self.connection_strength_cache == 0):
                                strength = self.compute_connection_strength(self.w_rec_tripartite[direction])
                                self.connection_strength_cache[direction] = strength
                            else:
                                strength = self.connection_strength_cache[direction].item()
                            lr = self.get_adaptive_learning_rate(strength)
                            print(f"  - Direction {direction}: strength={strength:.4f}, lr={lr:.4f}")
                    else:
                        print(f"  - Fixed learning rate: {self.stdp_learning_rate}")
                    print(f"  - HD trace max: {torch.max(self.hd_cell_trace).item():.4f}")
                    print(f"  - Raw update magnitude: {torch.norm(raw_update_rec).item():.4f}")
                    print(f"  - Weighted update magnitude: {torch.norm(weighted_update_rec).item():.4f}")
                    print(f"  - Scaled update magnitude: {torch.norm(scaled_update_rec).item():.4f}")
        
        # Apply Oja's rule if enabled (unchanged from V6)
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            # BVC weight updates
            if self.grid_influence < 1.0:
                pc_activations_col = self.place_cell_activations.unsqueeze(1)
                bvc_activations_row = self.bvc_activations.unsqueeze(0)
                
                weight_update_bvc = self.tau * (
                    pc_activations_col
                    * (
                        bvc_activations_row
                        - (1 / self.alpha_pb) * pc_activations_col * self.w_in
                    )
                ) * (1.0 - self.grid_influence)
                
                with torch.no_grad():
                    self.w_in += weight_update_bvc
            
            # Grid Cell weight updates
            if self.grid_influence > 0.0 and self.grid_cell_activations is not None and self.w_grid is not None:
                pc_activations_col = self.place_cell_activations.unsqueeze(1)
                grid_activations_row = self.grid_cell_activations.unsqueeze(0)
                
                weight_update_grid = self.tau * (
                    pc_activations_col
                    * (
                        grid_activations_row
                        - (1 / self.alpha_pg) * pc_activations_col * self.w_grid
                    )
                ) * self.grid_influence
                
                with torch.no_grad():
                    self.w_grid += weight_update_grid

    def reset_activations(self):
        """Reset place cell activations and related variables to zero."""
        self.place_cell_activations.zero_()
        self.activation_update.zero_()
        self.place_cell_trace = None
        
        # V6: Reset HD eligibility traces
        self.hd_cell_trace.zero_()
        
        # V6.5: Reset adaptive learning cache
        if hasattr(self, 'connection_strength_cache'):
            self.connection_strength_cache.zero_()
            self.strength_update_counter = 0
            
        # V6.5: Reset logging data
        if hasattr(self, 'enable_adaptive_logging') and self.enable_adaptive_logging:
            self.connection_strength_log = []
            self.learning_rate_log = []
            self.weight_magnitude_log = []
            self.logging_step_counter = 0
        
        # V5: Reset correlation tracking
        self.activation_history = []
        self.correlation_matrix = torch.ones(
            self.num_pc, self.num_pc, dtype=self.dtype, device=self.device
        ) * 0.5
        self.correlation_step_counter = 0
        
        # Reset debug counter
        if hasattr(self, '_debug_update_count'):
            self._debug_update_count = 0

    def preplay(self, direction: int, num_steps: int = 1) -> torch.Tensor:
        """Simulate preplay using learned connections.
        
        V6.5: Uses connections learned through adaptive STDP with direction-specific learning rates.
        These connections should be more stable and biologically plausible than previous versions.
        
        Args:
            direction: Index of head direction for exploiting recurrent weights.
            num_steps: Number of exploitation steps to simulate.
        
        Returns:
            Updated place cell activations after preplay.
        """
        # Copy current activations
        place_cell_activations = self.place_cell_activations.clone()
        
        # Iterate preplay steps
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()
            
            # Use directional connections from the specified HD slice
            # V6.5: These connections are now learned with adaptive learning rates,
            # making them more stable and predictable across different environments
            updated = torch.matmul(
                self.w_rec_tripartite[direction], previous_activations
            )
            
            updated = updated - previous_activations
            place_cell_activations = torch.tanh(torch.relu(updated))
        
        return place_cell_activations
    
    def get_opposite_direction(self, direction: int) -> int:
        """Get the opposite head direction for bidirectional connections."""
        return (direction + self.n_hd // 2) % self.n_hd

    def get_correlation_stats(self):
        """V5: Get statistics about current correlation matrix for debugging."""
        if not self.enable_correlation_weighting:
            return {}
            
        stats = {
            "correlation_matrix_mean": self.correlation_matrix.mean().item(),
            "correlation_matrix_std": self.correlation_matrix.std().item(),
            "correlation_matrix_min": self.correlation_matrix.min().item(),
            "correlation_matrix_max": self.correlation_matrix.max().item(),
            "activation_history_length": len(self.activation_history),
            "correlation_step_counter": self.correlation_step_counter,
        }
        
        # Compute weight statistics
        weights = self.get_correlation_weights()
        stats.update({
            "correlation_weights_mean": weights.mean().item(),
            "correlation_weights_std": weights.std().item(),
            "correlation_weights_min": weights.min().item(),
            "correlation_weights_max": weights.max().item(),
        })
        
        return stats

    def get_adaptive_learning_logs(self):
        """V6.5 NEW: Get logged adaptive learning data for saving/analysis."""
        if not hasattr(self, 'enable_adaptive_logging') or not self.enable_adaptive_logging:
            return None
            
        return {
            'connection_strength_log': self.connection_strength_log,
            'learning_rate_log': self.learning_rate_log,
            'weight_magnitude_log': self.weight_magnitude_log,
            'scale_name': self.scale_name,
            'adaptive_params': {
                'initial_lr': self.adaptive_initial_lr,
                'final_lr': self.adaptive_final_lr,
                'decay_rate': self.adaptive_decay_rate,
            },
            'total_logged_steps': self.logging_step_counter
        }

    def get_stdp_stats(self):
        """V6.5 NEW: Get comprehensive statistics about STDP learning for debugging."""
        stats = {
            "enable_stdp": self.enable_stdp,
            "enable_adaptive_stdp": self.enable_adaptive_stdp,
            "tau_hd": self.tau_hd,
            "hd_trace_magnitude": torch.norm(self.hd_cell_trace).item(),
            "hd_trace_max": torch.max(self.hd_cell_trace).item(),
            "hd_trace_mean": torch.mean(self.hd_cell_trace).item(),
            "connection_magnitude": torch.norm(self.w_rec_tripartite).item(),
            "connection_max": torch.max(self.w_rec_tripartite).item(),
            "connection_min": torch.min(self.w_rec_tripartite).item(),
        }
        
        if self.enable_adaptive_stdp:
            # Add adaptive learning rate statistics
            stats.update({
                "adaptive_initial_lr": self.adaptive_initial_lr,
                "adaptive_final_lr": self.adaptive_final_lr,
                "adaptive_decay_rate": self.adaptive_decay_rate,
            })
            
            # Add per-direction statistics
            direction_stats = {}
            for direction in range(self.n_hd):
                strength = self.compute_connection_strength(self.w_rec_tripartite[direction])
                lr = self.get_adaptive_learning_rate(strength)
                direction_stats[f"direction_{direction}_strength"] = strength
                direction_stats[f"direction_{direction}_lr"] = lr
            
            stats.update(direction_stats)
        else:
            stats["fixed_stdp_learning_rate"] = self.stdp_learning_rate
        
        if hasattr(self, '_debug_update_count'):
            stats["debug_update_count"] = self._debug_update_count
        
        return stats

    def get_adaptive_learning_stats(self):
        """V6.5 NEW: Get detailed adaptive learning rate statistics for analysis."""
        if not self.enable_adaptive_stdp:
            return {"adaptive_learning_enabled": False}
        
        stats = {
            "adaptive_learning_enabled": True,
            "parameters": {
                "initial_lr": self.adaptive_initial_lr,
                "final_lr": self.adaptive_final_lr,
                "decay_rate": self.adaptive_decay_rate,
            },
            "directions": {}
        }
        
        for direction in range(self.n_hd):
            strength = self.compute_connection_strength(self.w_rec_tripartite[direction])
            lr = self.get_adaptive_learning_rate(strength)
            
            stats["directions"][direction] = {
                "connection_strength": strength,
                "current_lr": lr,
                "lr_ratio": lr / self.adaptive_initial_lr,  # How much decay has occurred
            }
        
        # Summary statistics
        all_strengths = [stats["directions"][d]["connection_strength"] for d in range(self.n_hd)]
        all_lrs = [stats["directions"][d]["current_lr"] for d in range(self.n_hd)]
        
        stats["summary"] = {
            "avg_connection_strength": np.mean(all_strengths),
            "max_connection_strength": np.max(all_strengths),
            "min_connection_strength": np.min(all_strengths),
            "avg_learning_rate": np.mean(all_lrs),
            "max_learning_rate": np.max(all_lrs),
            "min_learning_rate": np.min(all_lrs),
        }
        
        return stats