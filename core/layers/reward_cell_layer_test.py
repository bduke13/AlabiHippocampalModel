import torch
import math
import copy

# Global constants for scale-dependent reward propagation
# C_REWARD_PER_CELL now acts as a fixed reward budget per scale (not per cell)
# Optionally override per-scale via k_reward in the caller if needed.
# C_LAMBDA controls the time scale of propagation
C_REWARD_PER_CELL = 5.0  # Reward budget per scale
C_LAMBDA = 20.0  # Decay length scaling constant


class RewardCellLayerTest:
    def __init__(
        self,
        num_place_cells=200,
        num_replay=3,
        learning_rate=0.1,
        replay_timesteps=20,
        replay_decay_factor=6,
        custom_replay_timesteps=40,
        initial_value_multiplier=1.0,
        sigma_pc_s=None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Reward Cell Layer (Test version with scale-dependent reward propagation).

        Args:
            num_place_cells: Dimension of the input vector (number of place cells).
            num_replay: Number of replay iterations.
            learning_rate: Learning rate for weight updates.
            replay_timesteps: Number of timesteps for regular replay (default: 20).
            replay_decay_factor: DEPRECATED - use sigma_pc_s instead.
            custom_replay_timesteps: Number of timesteps for custom activations replay (default: 40).
            initial_value_multiplier: DEPRECATED - use sigma_pc_s instead.
            sigma_pc_s: Place field size for this scale. If provided, overrides decay_factor and
                       initial_value_multiplier with scale-dependent values:
                       lambda_s = C_LAMBDA * sigma_pc_s (propagation length)
                       C_REWARD = C_REWARD_PER_CELL (fixed reward budget per scale)
                       A_s = C_REWARD / lambda_s (initial amplitude)
            device: Device to run computations on ("cpu" or "cuda").
        """
        self.device = device
        self.num_replay = num_replay
        self.learning_rate = learning_rate
        self.replay_timesteps = replay_timesteps
        self.custom_replay_timesteps = custom_replay_timesteps

        # Scale-dependent reward propagation mechanism
        if sigma_pc_s is not None:
            # Calculate scale-dependent parameters from place field size
            self.sigma_pc_s = sigma_pc_s
            self.lambda_s = C_LAMBDA * sigma_pc_s  # Decay length (propagation distance)

            # Fixed reward budget per scale (no longer scaled by place cell count)
            self.C_REWARD = C_REWARD_PER_CELL
            self.A_s = self.C_REWARD / self.lambda_s     # Amplitude (initial reward strength)

            print(f"[RCN] Scale-dependent reward: sigma_pc_s={sigma_pc_s:.2f}, "
                  f"lambda_s={self.lambda_s:.2f}, C_REWARD={self.C_REWARD:.1f}, A_s={self.A_s:.4f}")
        else:
            # Legacy mode: use provided parameters directly
            self.sigma_pc_s = None
            self.lambda_s = replay_decay_factor
            self.A_s = initial_value_multiplier
            self.C_REWARD = None
            print(f"[RCN] Legacy mode: lambda_s={self.lambda_s:.2f}, A_s={self.A_s:.4f}")

        # Store legacy parameters for compatibility
        self.replay_decay_factor = self.lambda_s
        self.initial_value_multiplier = self.A_s

        # Initialize reward cell activation as a tensor of shape (1, 1)
        self.reward_cell_activations = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Initialize weights with small random values (stddev=0.01)
        self.w_in = (
            torch.randn((1, num_place_cells), dtype=torch.float32, device=self.device)
            * 0.0001
        )
        self.w_in_effective = self.w_in.clone()

    def update_reward_cell_activations(self, input_data, visit=False):
        """
        Compute the activations of the reward cell based on input data.

        Args:
            input_data: A torch.Tensor (shape: [num_place_cells]) representing the input.
            visit: If True, update the weights based on the input.
        """
        # Ensure input_data is on the correct device
        input_data = input_data.to(self.device)

        # Compute L1 norm and a safe denominator to avoid division by zero
        input_norm = torch.norm(input_data, p=1)
        safe_denominator = torch.max(
            input_norm, torch.tensor(1e-4, dtype=torch.float32, device=self.device)
        )

        # Compute activations via matrix multiplication and normalization
        activations = torch.matmul(self.w_in_effective, input_data) / safe_denominator

        # Clamp activations to be between 0 and 1e6 and reshape to (1, 1)
        self.reward_cell_activations = torch.clamp(activations, 0, 1e6).view(1, 1)

        if visit:
            # Update weights directly based on the input (broadcast input_data over the row)
            updated_weights = self.w_in_effective + self.learning_rate * input_data
            self.w_in_effective = updated_weights

    def compute_reward_activations_batched(self, input_data_batch):
        """
        Compute reward activations for a batch of place cell activation patterns.

        This is an optimized version for preplay that processes multiple activation
        patterns in parallel without updating internal state.

        Args:
            input_data_batch: A torch.Tensor (shape: [batch_size, num_place_cells])

        Returns:
            reward_activations: A torch.Tensor (shape: [batch_size,]) with reward values
        """
        # Ensure input is on correct device
        input_data_batch = input_data_batch.to(self.device)

        # Compute L1 norms for each sample in the batch
        # Shape: (batch_size,)
        input_norms = torch.norm(input_data_batch, p=1, dim=1)
        safe_denominators = torch.maximum(
            input_norms, torch.tensor(1e-4, dtype=torch.float32, device=self.device)
        )

        # Batch matrix-vector multiply
        # w_in_effective: (1, num_place_cells)
        # input_data_batch: (batch_size, num_place_cells)
        # Result: (batch_size,)
        activations = torch.matmul(input_data_batch, self.w_in_effective.T).squeeze(1) / safe_denominators

        # Clamp activations
        activations = torch.clamp(activations, 0, 1e6)

        return activations

    def replay(self, pcn):
        """
        Replay the place cell activations and update reward cell weights.

        Args:
            pcn: A place cell network instance expected to have attributes:
                 - place_cell_activations: torch.Tensor of shape (num_pc,)
                 - w_rec_tripartite: torch.Tensor of shape (n_hd, num_pc, num_pc)
        """
        if not getattr(pcn, "reward_replay_supported", True):
            return

        # Use a deep copy to prevent external modifications
        pcn_copy = copy.deepcopy(pcn)

        # Move tensors to the correct device
        place_cell_activations = pcn_copy.place_cell_activations.to(
            self.device
        )  # shape: (num_pc,)
        w_rec_tripartite = pcn_copy.w_rec_tripartite.to(
            self.device
        )  # shape: (n_hd, num_pc, num_pc)

        # Initialize the weight update tensor with the same shape as w_in: (1, num_place_cells)
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        # Perform replay for a configurable number of time steps
        for time_step in range(self.replay_timesteps):
            exponential_decay_factor = math.exp(-time_step / self.lambda_s)

            # Normalize place cell activations (L2 norm with stability)
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(
                        1e-12, dtype=place_cell_activations.dtype, device=self.device
                    ),
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val

            # Replace any NaN values with zeros
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations,
            )

            # Cumulatively update the weight update
            # Apply A_s (scale-dependent amplitude) to modulate reward strength
            weight_update = (
                weight_update
                + exponential_decay_factor * normalized_place_cell_activations * self.A_s
            )

            # Update place cell activations:
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[
                0
            ]  # shape: (num_pc, num_pc)

            # Use torch.matmul to compute activations
            dot_val = torch.matmul(
                recurrent_weights_max.float(), place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # Light stability guard: only rescale if values explode, otherwise preserve amplitude ordering
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val

        # Update the weights and synchronize the effective weights.
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

    def replay_with_custom_activations(self, pcn, custom_activations):
        """
        **CRITICAL METHOD for multi-goal reward map creation!**

        Replay with custom place cell activations instead of PCN's current state.
        This allows creating distinct reward maps for each goal location.

        Without this method, all goals would get identical reward maps because
        the robot is at the same location when generating all maps.

        Args:
            pcn: A place cell network instance expected to have attributes:
                 - w_rec_tripartite: torch.Tensor of shape (n_hd, num_pc, num_pc)
            custom_activations: torch.Tensor of shape (num_pc,) - the activations to use for replay
                              (typically an artificial activation pattern with one PC set to 1.0)
        """
        if not getattr(pcn, "reward_replay_supported", True):
            return

        # Use a deep copy to prevent external modifications
        pcn_copy = copy.deepcopy(pcn)

        # Move tensors to the correct device - USE CUSTOM ACTIVATIONS
        place_cell_activations = custom_activations.to(self.device)  # ← KEY: Use custom activations
        w_rec_tripartite = pcn_copy.w_rec_tripartite.to(
            self.device
        )  # shape: (n_hd, num_pc, num_pc)

        # Initialize the weight update tensor with the same shape as w_in: (1, num_place_cells)
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        # Perform replay for a configurable number of time steps (custom activations version)
        for time_step in range(self.custom_replay_timesteps):
            exponential_decay_factor = math.exp(-time_step / self.lambda_s)

            # Normalize place cell activations (L2 norm with stability)
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(
                        1e-12, dtype=place_cell_activations.dtype, device=self.device
                    ),
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val

            # Replace any NaN values with zeros
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations,
            )

            # Cumulatively update the weight update
            # Apply A_s (scale-dependent amplitude) to modulate reward strength
            weight_update = (
                weight_update
                + exponential_decay_factor * normalized_place_cell_activations * self.A_s
            )

            # Update place cell activations using recurrent connections
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[
                0
            ]  # shape: (num_pc, num_pc)

            # Use torch.matmul to compute activations
            dot_val = torch.matmul(
                recurrent_weights_max.float(), place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # Light stability guard: only rescale if values explode, otherwise preserve amplitude ordering
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val

        # Update the weights and synchronize the effective weights.
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

    def td_update(self, input_data, next_reward):
        """
        Perform a temporal difference (TD) update for reward learning.

        Args:
            input_data: A torch.Tensor representing the input vector (shape: [num_place_cells]).
            next_reward: A float representing the reward at the next timestep.
        """
        input_data = input_data.to(self.device)

        # Compute prediction using the full weight vector (a single reward cell)
        prediction = torch.matmul(self.w_in_effective, input_data)  # shape: (1,)
        delta = next_reward - prediction.squeeze()

        # Update weights based on the TD learning rule.
        updated_weights = self.w_in_effective + self.learning_rate * delta * input_data
        self.w_in_effective = updated_weights
