"""
Unified Multi-Scale Reward Cell Network with cross-scale replay.

This implementation creates a single reward map across all scales (1750 cells total)
and uses the unified adjacency matrix for hierarchical replay. Reward propagates
through cross-scale connections, allowing coarse-to-fine information flow.

Key features:
- Single unified reward cell receiving input from all 1750 place cells
- Cross-scale replay through unified W_rec matrix
- Scale-dependent decay lengths: λ_s = C_LAMBDA × σ_r^s
- Biologically plausible hierarchical replay (large → medium → small)
"""

import torch
import math
import copy
from typing import List, Dict

# Global constants for unified reward propagation
C_REWARD_UNIFIED = 5.0    # Single reward budget shared across all scales
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
        num_place_cells_total: int = 1750,  # 1000 + 500 + 250
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
        # Experience-conditioned replay controls.
        # Hybrid mode combines:
        # 1) reverse experienced path replay, and
        # 2) local graph diffusion.
        self.use_experience_replay = True
        self.experience_topk = 8
        self.enable_hybrid_replay = True
        self.path_replay_weight = 0.8
        self.diffusion_replay_weight = 0.2
        # True unified decay: one global decay profile across all scales.
        self.use_global_decay = True
        # Backward-compat alias from older checkpoints/scripts.
        self.experience_mix_eta = 0.2
        self.experience_transition_topk = 12

        # Store scale configurations
        self.scale_configs = scale_configs if scale_configs else []
        self.num_scales = len(self.scale_configs)

        # Calculate scale-dependent decay lengths.
        # Unified replay uses one shared reward budget across all scales.
        self.lambda_per_scale = []
        self.C_REWARD = float(C_REWARD_UNIFIED)

        for cfg in self.scale_configs:
            sigma_r = cfg.get("sigma_r", 1.0)
            lambda_s = C_LAMBDA * sigma_r
            self.lambda_per_scale.append(lambda_s)

            print(f"[UnifiedRCN] Scale {cfg.get('name', '?')}: "
                  f"sigma_r={sigma_r:.2f}m, lambda_s={lambda_s:.2f}")

        # Default lambda for overall replay (use medium scale's value)
        if len(self.lambda_per_scale) > 1:
            self.lambda_s = self.lambda_per_scale[1]  # Medium scale
        elif len(self.lambda_per_scale) == 1:
            self.lambda_s = self.lambda_per_scale[0]
        else:
            # Fallback
            self.lambda_s = 40.0
            
        # Initialize unified reward cell activation
        self.reward_cell_activations = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Initialize unified weights: (1, num_place_cells_total)
        # Connects single reward cell to all place cells across scales
        self.w_in = torch.zeros(
            1, num_place_cells_total, dtype=torch.float32, device=self.device
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
        self.lambda_per_pc = self._build_lambda_per_pc()
        self.lambda_global = self._build_global_lambda()
        self._ensure_experience_buffers()

        print(f"[UnifiedRCN] Initialized with {num_place_cells_total} total place cells")
        print(f"  Scale boundaries: {self.scale_boundaries}")
        print(f"  Default replay: lambda={self.lambda_s:.2f}, C_total={self.C_REWARD:.4f}")

    def reconfigure_from_scale_configs(self, scale_configs: List[Dict]):
        """
        Refresh scale-dependent replay parameters from current controller scale configs.
        """
        self.scale_configs = scale_configs if scale_configs else []
        self.num_scales = len(self.scale_configs)
        self.num_place_cells_total = sum(cfg["num_pc"] for cfg in self.scale_configs) if self.scale_configs else self.num_place_cells_total

        self.lambda_per_scale = []
        self.C_REWARD = float(C_REWARD_UNIFIED)
        for cfg in self.scale_configs:
            sigma_r = cfg.get("sigma_r", 1.0)
            lambda_s = C_LAMBDA * sigma_r
            self.lambda_per_scale.append(lambda_s)

        if len(self.lambda_per_scale) > 1:
            self.lambda_s = self.lambda_per_scale[1]
        elif len(self.lambda_per_scale) == 1:
            self.lambda_s = self.lambda_per_scale[0]
        else:
            self.lambda_s = 40.0

        if self.scale_configs:
            self.scale_boundaries = [0]
            cumsum = 0
            for cfg in self.scale_configs:
                cumsum += cfg["num_pc"]
                self.scale_boundaries.append(cumsum)
        else:
            self.scale_boundaries = [0, self.num_place_cells_total]
        self.lambda_per_pc = self._build_lambda_per_pc()
        self.lambda_global = self._build_global_lambda()
        self._ensure_experience_buffers()

    def _ensure_experience_buffers(self):
        """Ensure transition-count buffers exist and match current PC dimensionality."""
        expected = int(self.num_place_cells_total)
        counts = getattr(self, "experience_transition_counts", None)
        if counts is None or int(getattr(counts, "shape", [0, 0])[0]) != expected:
            self.experience_transition_counts = torch.zeros(
                (expected, expected), dtype=torch.float32, device=torch.device("cpu")
            )

    def observe_transition(
        self,
        prev_place_cell_activations: torch.Tensor,
        curr_place_cell_activations: torch.Tensor,
    ):
        """
        Accumulate experienced transitions (prev -> curr) from trajectory data.
        Uses top-k activations per step for efficiency and robustness.
        """
        if not bool(getattr(self, "use_experience_replay", True)):
            return
        self._ensure_experience_buffers()
        if prev_place_cell_activations is None or curr_place_cell_activations is None:
            return

        prev = prev_place_cell_activations.detach().to(dtype=torch.float32)
        curr = curr_place_cell_activations.detach().to(dtype=torch.float32)
        if prev.numel() != self.num_place_cells_total or curr.numel() != self.num_place_cells_total:
            return

        k_global = int(max(1, min(self.experience_topk, self.num_place_cells_total)))
        boundaries = getattr(self, "scale_boundaries", None)

        if boundaries is not None and len(boundaries) >= 2:
            # Per-scale top-k ensures small-field scales (0, 1) are recorded even when
            # large-scale cells dominate the global top-k activation ranking.
            k_per_scale = max(1, k_global // (len(boundaries) - 1))
            all_prev_idx, all_curr_idx = [], []
            for s_start, s_end in zip(boundaries[:-1], boundaries[1:]):
                k_s = min(k_per_scale, s_end - s_start)
                _, pi = torch.topk(prev[s_start:s_end], k=k_s)
                _, ci = torch.topk(curr[s_start:s_end], k=k_s)
                all_prev_idx.append(pi + s_start)
                all_curr_idx.append(ci + s_start)
            # Also include global top-k for cross-scale connections.
            _, gi_p = torch.topk(prev, k=k_global)
            _, gi_c = torch.topk(curr, k=k_global)
            all_prev_idx.append(gi_p)
            all_curr_idx.append(gi_c)
            prev_idx = torch.cat(all_prev_idx).unique()
            curr_idx = torch.cat(all_curr_idx).unique()
            prev_vals = prev[prev_idx].clamp(min=0.0)
            curr_vals = curr[curr_idx].clamp(min=0.0)
        else:
            # Fallback: original global top-k.
            prev_vals, prev_idx = torch.topk(prev, k=k_global)
            curr_vals, curr_idx = torch.topk(curr, k=k_global)
            prev_vals = prev_vals.clamp(min=0.0)
            curr_vals = curr_vals.clamp(min=0.0)

        prev_mass = float(prev_vals.sum().item())
        curr_mass = float(curr_vals.sum().item())
        if prev_mass <= 1e-12 or curr_mass <= 1e-12:
            return

        prev_vals = prev_vals / max(prev_mass, 1e-12)
        curr_vals = curr_vals / max(curr_mass, 1e-12)
        prev_idx = prev_idx.detach().cpu().long()
        curr_idx = curr_idx.detach().cpu().long()
        prev_vals = prev_vals.detach().cpu()
        curr_vals = curr_vals.detach().cpu()

        # Weighted outer-product update on sparse index set.
        for i, v_i in zip(prev_idx, prev_vals):
            self.experience_transition_counts[i, curr_idx] += v_i * curr_vals

    def reset_experience_transitions(self):
        """Clear accumulated experienced transition counts."""
        self._ensure_experience_buffers()
        self.experience_transition_counts.zero_()

    def _prepare_experience_replay_transition(
        self,
        local_transition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build reverse replay transition from experienced trajectories and optionally
        blend with local graph diffusion for side-branch/global support.
        """
        if not bool(getattr(self, "use_experience_replay", True)):
            return local_transition
        self._ensure_experience_buffers()

        counts = self.experience_transition_counts
        total_counts = float(torch.sum(counts).item())
        if total_counts <= 1e-9:
            return local_transition

        counts_dev = counts.to(device=self.device, dtype=torch.float32)
        row_sum = torch.sum(counts_dev, dim=1, keepdim=True)
        forward = torch.where(
            row_sum > 1e-12,
            counts_dev / row_sum,
            torch.zeros_like(counts_dev),
        )
        # Keep strongest experienced outgoing transitions per node.
        topk = int(max(0, getattr(self, "experience_transition_topk", 0)))
        if topk > 0 and topk < forward.shape[1]:
            vals, idx = torch.topk(forward, k=topk, dim=1)
            sparse = torch.zeros_like(forward)
            sparse.scatter_(1, idx, vals)
            sparse_row_sum = torch.sum(sparse, dim=1, keepdim=True)
            forward = torch.where(
                sparse_row_sum > 1e-12,
                sparse / sparse_row_sum,
                torch.zeros_like(sparse),
            )
        reverse_exp = torch.transpose(forward, 0, 1)

        use_hybrid = bool(getattr(self, "enable_hybrid_replay", True))
        if use_hybrid:
            w_path = float(max(0.0, getattr(self, "path_replay_weight", 0.8)))
            w_diff = float(max(0.0, getattr(self, "diffusion_replay_weight", 0.2)))
            w_sum = w_path + w_diff
            if w_sum <= 1e-12:
                w_path, w_diff = 0.8, 0.2
                w_sum = 1.0
            w_path /= w_sum
            w_diff /= w_sum
        else:
            eta = float(min(1.0, max(0.0, getattr(self, "experience_mix_eta", 0.2))))
            w_path = 1.0 - eta
            w_diff = eta
        reverse_local = torch.transpose(local_transition, 0, 1)
        transition = (w_path * reverse_exp) + (w_diff * reverse_local)

        if bool(getattr(self, "replay_row_normalize", True)):
            trans_row_sum = torch.sum(transition, dim=1, keepdim=True)
            transition = torch.where(
                trans_row_sum > 1e-12,
                transition / trans_row_sum,
                torch.zeros_like(transition),
            )
        return transition

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
        local_transition = self._prepare_replay_transition(recurrent_weights_max)
        transition = self._prepare_experience_replay_transition(local_transition)

        # Initialize weight update
        weight_update = torch.zeros_like(self.w_in, device=self.device)
        replay_amplitude = self._compute_unified_amplitude(replay_gate)

        # Perform unified replay
        for time_step in range(self.replay_timesteps):
            exponential_decay_factor = self._decay_factor_at_step(time_step)

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
                + exponential_decay_factor * (normalized_place_cell_activations * replay_gate) * replay_amplitude
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
        local_transition = self._prepare_replay_transition(recurrent_weights_max)
        transition = self._prepare_experience_replay_transition(local_transition)

        # Initialize weight update
        weight_update = torch.zeros_like(self.w_in, device=self.device)
        replay_amplitude = self._compute_unified_amplitude(replay_gate)

        # Perform replay with custom starting activations
        for time_step in range(self.replay_timesteps):
            exponential_decay_factor = self._decay_factor_at_step(time_step)

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
                + exponential_decay_factor * (normalized_place_cell_activations * replay_gate) * replay_amplitude
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

    def _compute_unified_amplitude(self, replay_gate: torch.Tensor) -> float:
        """
        Compute replay amplitude from one shared reward budget.

        Shared-budget rule:
        - One total budget C_REWARD is distributed across scales according to gate mass.
        - Effective amplitude is sum_s(share_s * C_REWARD / lambda_s).
        """
        if bool(getattr(self, "use_global_decay", True)):
            lam = float(max(1e-6, getattr(self, "lambda_global", self.lambda_s)))
            return float(self.C_REWARD / lam)

        if self.num_scales <= 0 or len(self.lambda_per_scale) != self.num_scales:
            return float(self.C_REWARD / max(float(self.lambda_s), 1e-6))

        masses = []
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            masses.append(torch.sum(replay_gate[start:end]))
        masses_t = torch.stack(masses)
        total_mass = torch.sum(masses_t)

        if (not torch.isfinite(total_mass)) or total_mass.item() <= 1e-12:
            shares = torch.full(
                (self.num_scales,),
                1.0 / float(self.num_scales),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            shares = masses_t / total_mass

        lambda_t = torch.tensor(self.lambda_per_scale, dtype=torch.float32, device=self.device)
        inv_lambda_t = 1.0 / torch.clamp(lambda_t, min=1e-6)
        amp = self.C_REWARD * torch.sum(shares * inv_lambda_t)
        return float(amp.item())

    def _build_global_lambda(self) -> float:
        """
        Build one global decay length for unified replay.
        Uses mean scale lambda to avoid biasing reward spread to one scale.
        """
        if self.lambda_per_scale and len(self.lambda_per_scale) > 0:
            return float(sum(self.lambda_per_scale) / float(len(self.lambda_per_scale)))
        return float(self.lambda_s)

    def _decay_factor_at_step(self, time_step: int) -> torch.Tensor:
        """
        Decay factor for replay step t.
        - Global mode: one scalar schedule shared by all PCs.
        - Legacy mode: per-PC scale-aware schedule.
        """
        if bool(getattr(self, "use_global_decay", True)):
            lam = float(max(1e-6, getattr(self, "lambda_global", self.lambda_s)))
            return torch.full(
                (self.num_place_cells_total,),
                math.exp(-float(time_step) / lam),
                dtype=torch.float32,
                device=self.device,
            )

        if self.lambda_per_pc is not None:
            return torch.exp(
                torch.tensor(-float(time_step), dtype=torch.float32, device=self.device)
                / self.lambda_per_pc
            )

        return torch.full(
            (self.num_place_cells_total,),
            math.exp(-float(time_step) / self.lambda_s),
            dtype=torch.float32,
            device=self.device,
        )

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

    def _build_lambda_per_pc(self) -> torch.Tensor:
        """
        Expand scale-specific replay decay lengths to a per-PC vector.
        """
        if self.num_scales <= 0 or len(self.lambda_per_scale) != self.num_scales:
            return torch.full(
                (self.num_place_cells_total,),
                float(self.lambda_s),
                dtype=torch.float32,
                device=self.device,
            )

        blocks = []
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            n = int(max(0, end - start))
            lam = float(self.lambda_per_scale[scale_idx])
            blocks.append(torch.full((n,), lam, dtype=torch.float32, device=self.device))

        if not blocks:
            return torch.full(
                (self.num_place_cells_total,),
                float(self.lambda_s),
                dtype=torch.float32,
                device=self.device,
            )
        return torch.cat(blocks, dim=0)

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
