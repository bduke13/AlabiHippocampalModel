"""
Unified Multi-Scale Reward Cell Network with cross-scale replay.

This implementation creates a single reward map across all scales (3250 cells total)
and uses the unified adjacency matrix for hierarchical replay. Reward propagates
through cross-scale connections, allowing coarse-to-fine information flow.

Key features:
- Single unified reward cell receiving input from all 3250 place cells
- Cross-scale replay through unified W_rec matrix
- Scale-dependent decay lengths: λ_s = C_LAMBDA × σ_r^s
- Biologically plausible hierarchical replay (large → medium → small)
"""

import torch
import math
import copy
from typing import List, Dict

# Global constants for scale-dependent reward propagation
C_REWARD_PER_CELL = 5.0   # Reward budget per scale
C_LAMBDA = 20.0           # Decay length scaling constant


class UnifiedMultiScaleRCN:
    """
    Unified reward cell network handling all scales together.

    Instead of separate reward maps per scale, maintains a single unified reward
    that respects scale-appropriate spatial structure through the learned adjacency
    matrix.
    """

    def __init__(
        self,
        num_place_cells_total: int = 3250,  # 2000 + 1000 + 250
        scale_configs: List[Dict] = None,   # List of scale configurations
        num_replay: int = 3,
        learning_rate: float = 0.1,
        replay_timesteps: int = 40,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize unified multi-scale reward cell layer.

        Args:
            num_place_cells_total: Total number of place cells across all scales
            scale_configs: List of scale configurations with sigma_r values
            num_replay: Number of replay iterations
            learning_rate: Learning rate for weight updates
            replay_timesteps: Number of timesteps for replay
            device: Computation device
        """
        self.device = device
        self.num_place_cells_total = num_place_cells_total
        self.num_replay = num_replay
        self.learning_rate = learning_rate
        self.replay_timesteps = replay_timesteps
        # Replay diffusion controls (locality-preserving defaults).
        # These reduce over-global reward spreading when unified recurrent weights are dense/high-gain.
        self.replay_row_normalize = True
        self.replay_transition_topk = 64

        # Store scale configurations
        self.scale_configs = scale_configs if scale_configs else []
        self.num_scales = len(self.scale_configs)

        # Calculate scale-dependent parameters
        self.lambda_per_scale = []
        self.A_per_scale = []

        for cfg in self.scale_configs:
            sigma_r = cfg.get("sigma_r", 1.0)
            lambda_s = C_LAMBDA * sigma_r
            A_s = C_REWARD_PER_CELL / lambda_s

            self.lambda_per_scale.append(lambda_s)
            self.A_per_scale.append(A_s)

            print(f"[UnifiedRCN] Scale {cfg.get('name', '?')}: "
                  f"σ_r={sigma_r:.2f}m, λ_s={lambda_s:.2f}, A_s={A_s:.4f}")

        # Default lambda for overall replay (use medium scale's value)
        if len(self.lambda_per_scale) > 1:
            self.lambda_s = self.lambda_per_scale[1]  # Medium scale
            self.A_s = self.A_per_scale[1]
        elif len(self.lambda_per_scale) == 1:
            self.lambda_s = self.lambda_per_scale[0]
            self.A_s = self.A_per_scale[0]
        else:
            # Fallback
            self.lambda_s = 40.0
            self.A_s = 0.125

        # Initialize unified reward cell activation
        self.reward_cell_activations = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Initialize unified weights: (1, num_place_cells_total)
        # Connects single reward cell to all place cells across scales
        self.w_in = (
            torch.randn(
                1, num_place_cells_total, dtype=torch.float32, device=self.device
            )
            * 0.01
        )

        # Effective weights (used for forward pass)
        self.w_in_effective = self.w_in.clone()

        # Scale boundaries for indexing
        if self.scale_configs:
            self.scale_boundaries = [0]
            cumsum = 0
            for cfg in self.scale_configs:
                cumsum += cfg["num_pc"]
                self.scale_boundaries.append(cumsum)
        else:
            self.scale_boundaries = [0, num_place_cells_total]

        print(f"[UnifiedRCN] Initialized with {num_place_cells_total} total place cells")
        print(f"  Scale boundaries: {self.scale_boundaries}")
        print(f"  Default replay: λ={self.lambda_s:.2f}, A={self.A_s:.4f}")

    def reconfigure_from_scale_configs(self, scale_configs: List[Dict]):
        """
        Refresh scale-dependent replay parameters from current controller scale configs.
        """
        self.scale_configs = scale_configs if scale_configs else []
        self.num_scales = len(self.scale_configs)
        self.num_place_cells_total = sum(cfg["num_pc"] for cfg in self.scale_configs) if self.scale_configs else self.num_place_cells_total

        self.lambda_per_scale = []
        self.A_per_scale = []
        for cfg in self.scale_configs:
            sigma_r = cfg.get("sigma_r", 1.0)
            lambda_s = C_LAMBDA * sigma_r
            A_s = C_REWARD_PER_CELL / lambda_s
            self.lambda_per_scale.append(lambda_s)
            self.A_per_scale.append(A_s)

        if len(self.lambda_per_scale) > 1:
            self.lambda_s = self.lambda_per_scale[1]
            self.A_s = self.A_per_scale[1]
        elif len(self.lambda_per_scale) == 1:
            self.lambda_s = self.lambda_per_scale[0]
            self.A_s = self.A_per_scale[0]

        if self.scale_configs:
            self.scale_boundaries = [0]
            cumsum = 0
            for cfg in self.scale_configs:
                cumsum += cfg["num_pc"]
                self.scale_boundaries.append(cumsum)
        else:
            self.scale_boundaries = [0, self.num_place_cells_total]

    def update_reward_cell_activations(
        self,
        place_cell_activations: torch.Tensor,
        visit: bool = False,
    ) -> torch.Tensor:
        """
        Update unified reward cell activations.

        Args:
            place_cell_activations: Unified activation vector (num_place_cells_total,)
            visit: Whether currently at goal location

        Returns:
            reward_activations: Scalar reward value
        """
        # Reshape to batch format if needed
        if place_cell_activations.dim() == 1:
            place_cell_activations = place_cell_activations.unsqueeze(0)

        # Move to correct device
        place_cell_activations = place_cell_activations.to(self.device)

        # Compute reward: r = W_in @ v^p
        safe_denominators = torch.clamp(
            torch.sum(torch.abs(self.w_in_effective), dim=1, keepdim=True),
            min=1e-12
        )

        # Result: (batch_size,)
        activations = torch.matmul(
            place_cell_activations,
            self.w_in_effective.T
        ).squeeze(1) / safe_denominators

        # Clamp activations
        activations = torch.clamp(activations, 0, 1e6)

        # Update weights if at goal
        if visit:
            # Simple increment for goal location
            normalized_pc = place_cell_activations / (
                torch.norm(place_cell_activations) + 1e-12
            )
            self.w_in.data += self.learning_rate * normalized_pc
            self.w_in_effective = self.w_in.clone()

        self.reward_cell_activations = activations.unsqueeze(1)

        return activations

    def replay(self, unified_pcn, use_scale_gate: bool = False, min_gate: float = 0.0):
        """
        Unified replay through cross-scale adjacency matrix.

        Uses the unified W_rec matrix that connects all scales together,
        allowing reward to propagate from large scales to small scales
        naturally through learned transitions.

        Args:
            unified_pcn: Unified multi-scale PCN instance with:
                - place_cell_activations: torch.Tensor (num_pc_total,)
                - w_rec_unified: torch.Tensor (n_hd, num_pc_total, num_pc_total)
        """
        # Deep copy to avoid external modifications
        pcn_copy = copy.deepcopy(unified_pcn)

        # Move tensors to device
        place_cell_activations = pcn_copy.place_cell_activations.to(self.device)
        w_rec_unified = pcn_copy.w_rec_unified.to(self.device)
        if use_scale_gate:
            replay_gate = self._get_replay_gate(pcn_copy)
            if min_gate > 0.0:
                replay_gate = torch.clamp(replay_gate, min=float(min_gate), max=1.0)
        else:
            replay_gate = torch.ones(
                self.num_place_cells_total, dtype=torch.float32, device=self.device
            )

        # Build a replay transition matrix once for stability/locality.
        recurrent_weights_max = torch.max(w_rec_unified, dim=0)[0]
        transition = self._prepare_replay_transition(recurrent_weights_max)

        # Initialize weight update
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        # Perform unified replay
        for time_step in range(self.replay_timesteps):
            # Exponential decay
            exponential_decay_factor = math.exp(-time_step / self.lambda_s)

            # Normalize activations
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(1e-12, dtype=place_cell_activations.dtype, device=self.device)
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val

            # Replace NaNs with zeros
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations
            )

            # Accumulate weight update (gated by latest scale preference).
            weight_update = (
                weight_update
                + exponential_decay_factor * (normalized_place_cell_activations * replay_gate) * self.A_s
            )

            # Propagate activations: v(t+1) = tanh([T @ v(t) + v(t)]_+)
            dot_val = torch.matmul(
                transition.float(),
                place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)
            place_cell_activations = place_cell_activations * replay_gate

        # Stability guard
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val

        # Update weights
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

    def replay_with_custom_activations(
        self,
        unified_pcn,
        custom_activations,
        use_scale_gate: bool = False,
        min_gate: float = 0.0,
    ):
        """
        Replay with custom place cell activations (for multi-goal scenarios).

        Args:
            unified_pcn: Unified PCN instance
            custom_activations: torch.Tensor (num_pc_total,) - custom activation pattern
        """
        # Deep copy to avoid modifications
        pcn_copy = copy.deepcopy(unified_pcn)

        # Move tensors to device
        place_cell_activations = custom_activations.to(self.device)
        w_rec_unified = pcn_copy.w_rec_unified.to(self.device)
        if use_scale_gate:
            replay_gate = self._get_replay_gate(pcn_copy)
            if min_gate > 0.0:
                replay_gate = torch.clamp(replay_gate, min=float(min_gate), max=1.0)
        else:
            replay_gate = torch.ones(
                self.num_place_cells_total, dtype=torch.float32, device=self.device
            )

        # Build a replay transition matrix once for stability/locality.
        recurrent_weights_max = torch.max(w_rec_unified, dim=0)[0]
        transition = self._prepare_replay_transition(recurrent_weights_max)

        # Initialize weight update
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        # Perform replay with custom starting activations
        for time_step in range(self.replay_timesteps):
            exponential_decay_factor = math.exp(-time_step / self.lambda_s)

            # Normalize
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(1e-12, dtype=place_cell_activations.dtype, device=self.device)
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations
            )

            # Accumulate (gated by latest scale preference).
            weight_update = (
                weight_update
                + exponential_decay_factor * (normalized_place_cell_activations * replay_gate) * self.A_s
            )

            # Propagate
            dot_val = torch.matmul(
                transition.float(),
                place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)
            place_cell_activations = place_cell_activations * replay_gate

        # Stability guard
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val

        # Update weights
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

    def _get_replay_gate(self, unified_pcn) -> torch.Tensor:
        """
        Fetch per-PC replay gate from the most recent PCN scale preference.
        Falls back to all-ones when no preference is available.
        """
        gate = None
        if hasattr(unified_pcn, "get_last_scale_preference_per_pc"):
            gate = unified_pcn.get_last_scale_preference_per_pc()
        if gate is None:
            gate = torch.ones(
                self.num_place_cells_total, dtype=torch.float32, device=self.device
            )
        else:
            gate = gate.to(device=self.device, dtype=torch.float32)
        return gate

    def _prepare_replay_transition(self, recurrent_weights_max: torch.Tensor) -> torch.Tensor:
        """
        Prepare a stable replay transition matrix from recurrent weights.
        Steps:
        1) Keep only excitatory edges.
        2) Keep top-k outgoing edges per row (optional) to preserve locality.
        3) Row-normalize (optional) to prevent global blow-up.
        """
        transition = torch.clamp(recurrent_weights_max, min=0.0).to(self.device)

        topk = int(getattr(self, "replay_transition_topk", 0) or 0)
        if topk > 0 and topk < transition.shape[1]:
            vals, idx = torch.topk(transition, k=topk, dim=1)
            sparse = torch.zeros_like(transition)
            sparse.scatter_(1, idx, vals)
            transition = sparse

        if bool(getattr(self, "replay_row_normalize", True)):
            row_sum = torch.sum(transition, dim=1, keepdim=True)
            transition = torch.where(
                row_sum > 1e-12,
                transition / row_sum,
                torch.zeros_like(transition),
            )
        return transition

    def get_activations_per_scale(self, place_cell_activations: torch.Tensor) -> List[float]:
        """
        Compute per-scale reward contributions for analysis.

        Args:
            place_cell_activations: Unified activation vector

        Returns:
            List of reward values per scale
        """
        rewards_per_scale = []

        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]

            # Extract scale-specific weights and activations
            w_scale = self.w_in_effective[:, start:end]
            pc_scale = place_cell_activations[start:end]

            # Compute reward for this scale
            reward_scale = torch.sum(w_scale * pc_scale).item()
            rewards_per_scale.append(reward_scale)

        return rewards_per_scale

    def compute_reward_activations_batched(self, place_cell_activations_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute reward activations for a batch of place cell states.

        Used for stochastic preplay sampling where multiple trajectories
        are evaluated in parallel.

        Args:
            place_cell_activations_batch: Batch of activation vectors
                                          Shape: (batch_size, num_place_cells_total)

        Returns:
            Reward values for each state in the batch
            Shape: (batch_size,)
        """
        # Ensure batch format
        if place_cell_activations_batch.dim() == 1:
            place_cell_activations_batch = place_cell_activations_batch.unsqueeze(0)

        # Move to correct device
        place_cell_activations_batch = place_cell_activations_batch.to(self.device)

        # Compute rewards: r = W_in @ v^p
        # w_in_effective: (1, num_pc_total)
        # place_cell_activations_batch: (batch_size, num_pc_total)
        # Result: (batch_size, 1)
        safe_denominators = torch.clamp(
            torch.sum(torch.abs(self.w_in_effective), dim=1, keepdim=True),
            min=1e-12
        )

        activations = torch.matmul(
            place_cell_activations_batch,
            self.w_in_effective.T
        ).squeeze(1) / safe_denominators.squeeze()

        # Clamp activations
        activations = torch.clamp(activations, 0, 1e6)

        return activations
