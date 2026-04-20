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
from typing import List, Dict, Optional

# Global constants for unified reward propagation
C_REWARD_UNIFIED = 5.0    # Single reward budget shared across all scales
C_LAMBDA = 20.0           # Reference spatial decay constant


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
        # Reward readout uses the paper-style input-L1 normalization.
        self.reward_normalization_mode = "input_l1"
        # Wavefront-style replay: by default do not carry previous activation
        # mass forward, otherwise long replay collapses toward a plateau.
        self.replay_residual_mix = 0.0
        # Replay decay calibration:
        # - physical_scale (default): equalize spread in physical space by
        #   shortening lambda for larger fields and lengthening it for smaller fields.
        # - legacy_sigma: old rule lambda_s = C_LAMBDA * sigma_r.
        self.replay_decay_mode = "physical_scale"
        self.replay_decay_scale_key = "sigma_pc_s"
        self.replay_decay_fallback_key = "sigma_r"
        self.replay_decay_reference_scale_idx = 1
        self.replay_decay_min_lambda = 4.0
        # Default to per-scale decay rather than one global profile.
        self.use_global_decay = False
        # Backward-compat alias from older checkpoints/scripts.
        self.experience_mix_eta = 0.2
        self.experience_transition_topk = 12
        # The model now uses the paper-room-local goal-map path only.
        self.goal_map_generation_mode = "paper_room_local_replay"
        self.goal_map_path_topk = 16
        self.goal_map_path_decay = 0.96
        self.goal_map_path_frontier_only = True
        self.goal_map_neighbor_topk = 12
        self.goal_map_neighbor_steps = 6
        self.goal_map_neighbor_decay = 0.84
        self.goal_map_neighbor_frontier_only = True
        self.goal_map_neighbor_seed_scale = 0.60
        self.goal_map_checkpoint_mode = "implicit_cascade"
        self.goal_map_checkpoint_scale = 0.25
        self.goal_map_checkpoint_support_mode = "threshold_normalized"
        self.goal_map_checkpoint_stop_parent_replay = True
        self.goal_map_seed_gain = 1.0
        # Store scale configurations
        self.scale_configs = scale_configs if scale_configs else []
        self.num_scales = len(self.scale_configs)

        # Calculate scale-dependent decay lengths.
        # Unified replay uses one shared reward budget across all scales.
        self.lambda_per_scale = []
        self.C_REWARD = float(C_REWARD_UNIFIED)
        self._refresh_scale_decay_lengths(log_prefix="[UnifiedRCN]")

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
        self._refresh_scale_decay_lengths(log_prefix="[UnifiedRCN]")

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

    def _resolve_scale_field_extent(self, cfg: Dict) -> float:
        """
        Estimate the physical hop length represented by one replay step on this scale.

        Prefer the place-field width used for reward propagation when available,
        and fall back to sigma_r otherwise.
        """
        primary_key = str(getattr(self, "replay_decay_scale_key", "sigma_pc_s"))
        fallback_key = str(getattr(self, "replay_decay_fallback_key", "sigma_r"))
        raw_extent = cfg.get(primary_key, cfg.get(fallback_key, 1.0))
        try:
            extent = float(raw_extent)
        except (TypeError, ValueError):
            extent = 1.0
        return max(1e-6, extent)

    def _compute_lambda_per_scale(self) -> List[float]:
        """
        Build one replay decay length per scale.

        physical_scale:
            keep reward decay approximately consistent in meters by setting
            lambda_s proportional to the inverse of field extent.
        legacy_sigma:
            old rule lambda_s = C_LAMBDA * sigma_r.
        """
        if not self.scale_configs:
            return []

        mode = str(getattr(self, "replay_decay_mode", "physical_scale")).strip().lower()
        min_lambda = float(max(1e-3, getattr(self, "replay_decay_min_lambda", 4.0)))

        if mode in {"legacy_sigma", "legacy", "sigma"}:
            return [
                max(min_lambda, C_LAMBDA * max(1e-6, float(cfg.get("sigma_r", 1.0))))
                for cfg in self.scale_configs
            ]

        hop_lengths = [self._resolve_scale_field_extent(cfg) for cfg in self.scale_configs]
        ref_idx = int(getattr(self, "replay_decay_reference_scale_idx", 1))
        if not (0 <= ref_idx < len(hop_lengths)):
            ref_idx = min(len(hop_lengths) - 1, max(0, len(hop_lengths) // 2))
        tau_space = C_LAMBDA * hop_lengths[ref_idx]
        return [
            max(min_lambda, tau_space / max(1e-6, hop_len))
            for hop_len in hop_lengths
        ]

    def _refresh_scale_decay_lengths(self, log_prefix: str = "[UnifiedRCN]"):
        """Recompute scale-specific replay lambdas and emit a compact summary."""
        self.lambda_per_scale = self._compute_lambda_per_scale()
        mode = str(getattr(self, "replay_decay_mode", "physical_scale")).strip().lower()
        for cfg, lambda_s in zip(self.scale_configs, self.lambda_per_scale):
            sigma_r = float(cfg.get("sigma_r", 1.0))
            field_extent = self._resolve_scale_field_extent(cfg)
            print(
                f"{log_prefix} Scale {cfg.get('name', '?')}: "
                f"sigma_r={sigma_r:.2f}m, field_extent={field_extent:.2f}m, "
                f"lambda_s={lambda_s:.2f} ({mode})"
            )

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

        # Weighted outer-product update on the sparse index set.
        # This is mathematically identical to the old row-wise Python loop, but
        # avoids repeated interpreter overhead during large deferred rebuilds.
        outer = prev_vals.unsqueeze(1) * curr_vals.unsqueeze(0)
        self.experience_transition_counts[prev_idx[:, None], curr_idx[None, :]] += outer

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
        # experience_transition_counts stores experienced edges as prev -> curr.
        # For backward reward replay with transition @ value, rows must index
        # predecessor states and columns successor states, so the forward-normalized
        # matrix already has the correct orientation. Transposing here would push
        # reward toward successors instead of back along experienced trajectories.
        reverse_exp = forward

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

    def _row_max_normalize(self, transition: torch.Tensor) -> torch.Tensor:
        """
        Normalize each row by its strongest outgoing edge.

        Competitive replay uses max-backups rather than mass-conserving sums, so
        row-max normalization preserves the best local successor strength instead
        of shrinking value in high-outdegree regions.
        """
        row_max = torch.max(transition, dim=1, keepdim=True).values
        return torch.where(
            row_max > 1e-12,
            transition / row_max,
            torch.zeros_like(transition),
        )

    def _prepare_goal_map_path_transition(
        self,
        local_transition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a backward path graph for goal-map construction.

        Rows index predecessor cells and columns successor cells, so a max-backup
        over rows propagates value backward along experienced routes.
        """
        self._ensure_experience_buffers()
        counts = self.experience_transition_counts
        total_counts = float(torch.sum(counts).item())
        if total_counts <= 1e-9:
            reverse_local = torch.transpose(local_transition, 0, 1)
            return self._row_max_normalize(reverse_local)

        path_transition = counts.to(device=self.device, dtype=torch.float32)
        topk = int(max(1, getattr(self, "goal_map_path_topk", self.experience_transition_topk)))
        if topk > 0 and topk < path_transition.shape[1]:
            vals, idx = torch.topk(path_transition, k=topk, dim=1)
            sparse = torch.zeros_like(path_transition)
            sparse.scatter_(1, idx, vals)
            path_transition = sparse

        path_constraint = getattr(self, "goal_map_path_constraint", None)
        if path_constraint is None:
            path_constraint = getattr(self, "goal_map_spatial_constraint", None)
        if path_constraint is not None:
            path_constraint = path_constraint.to(
                device=self.device, dtype=torch.float32
            )
            if path_constraint.shape == path_transition.shape:
                path_transition = path_transition * self._row_max_normalize(
                    torch.clamp(path_constraint, min=0.0)
                )

        return self._row_max_normalize(path_transition)

    def _prepare_goal_map_neighbor_transition(
        self,
        recurrent_weights_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a symmetric local-neighbor graph from recurrent place-cell weights.

        This is intentionally separate from the experienced path graph: reward
        should first follow traversed routes, then bleed weakly into nearby
        fields that overlap those routes.
        """
        neighbor_constraint = getattr(self, "goal_map_neighbor_constraint", None)
        if neighbor_constraint is None:
            neighbor_constraint = getattr(self, "goal_map_spatial_constraint", None)
        if neighbor_constraint is not None:
            neighbor_constraint = neighbor_constraint.to(
                device=self.device, dtype=torch.float32
            )
            if neighbor_constraint.shape == recurrent_weights_max.shape:
                return self._row_max_normalize(torch.clamp(neighbor_constraint, min=0.0))

        excit = torch.clamp(recurrent_weights_max, min=0.0).to(self.device)
        symmetric = torch.maximum(excit, excit.T)
        topk = int(max(1, getattr(self, "goal_map_neighbor_topk", 12)))
        if topk > 0 and topk < symmetric.shape[1]:
            vals, idx = torch.topk(symmetric, k=topk, dim=1)
            sparse = torch.zeros_like(symmetric)
            sparse.scatter_(1, idx, vals)
            symmetric = sparse
        return self._row_max_normalize(symmetric)

    def _competitive_backup_step(
        self,
        transition: torch.Tensor,
        wave: torch.Tensor,
    ) -> torch.Tensor:
        """One max-backup step on a predecessor->successor graph."""
        if wave.dim() != 1:
            wave = wave.view(-1)
        edge_supported = transition * wave.unsqueeze(0)
        return torch.max(edge_supported, dim=1).values

    def _competitive_neighbor_from_source(
        self,
        transition: torch.Tensor,
        source_wave: torch.Tensor,
        num_steps: int,
        step_decay: float,
        base_values: torch.Tensor,
        frontier_only: bool = True,
    ) -> torch.Tensor:
        """
        Spread reward locally from one path-source wave.

        The source wave itself is not part of the neighbor branch. The first
        neighbor shell appears after one transition step, which matches the
        intended semantics of "start from the path value, then decay once as it
        spreads laterally".
        """
        if num_steps <= 0:
            return torch.zeros_like(source_wave)

        wave = torch.clamp(source_wave.to(self.device, dtype=torch.float32), min=0.0)
        neighbor_values = torch.zeros_like(wave)
        decay = float(min(0.9999, max(0.0, step_decay)))
        eps = 1e-8

        for _ in range(int(max(0, num_steps))):
            wave = decay * self._competitive_backup_step(transition, wave)
            if frontier_only:
                ceiling = torch.maximum(base_values, neighbor_values)
                wave = torch.where(wave > (ceiling + eps), wave, torch.zeros_like(wave))
            neighbor_values = torch.maximum(neighbor_values, wave)
            if float(torch.max(wave).item()) <= eps:
                break

        return neighbor_values

    def _resolve_checkpoint_relay_payload(self):
        """
        Load optional checkpoint relay seeds for sequential backbone replenishment.

        Checkpoints are represented as place-cell seed vectors. They are not added
        into the initial seed. Instead, a checkpoint fires once when the path
        backbone first reaches it strongly enough.
        """
        relay_seeds = getattr(self, "goal_map_checkpoint_seeds", None)
        if relay_seeds is None:
            return None

        relay_seeds = torch.as_tensor(
            relay_seeds, device=self.device, dtype=torch.float32
        )
        if relay_seeds.dim() == 1:
            relay_seeds = relay_seeds.unsqueeze(0)
        if relay_seeds.numel() == 0:
            return None
        if relay_seeds.shape[1] != self.num_place_cells_total:
            fixed = torch.zeros(
                (relay_seeds.shape[0], self.num_place_cells_total),
                dtype=torch.float32,
                device=self.device,
            )
            n_copy = min(self.num_place_cells_total, relay_seeds.shape[1])
            fixed[:, :n_copy] = relay_seeds[:, :n_copy]
            relay_seeds = fixed

        relay_peak = torch.amax(torch.abs(relay_seeds), dim=1, keepdim=True).clamp_min(1e-8)
        relay_seeds = torch.clamp(relay_seeds / relay_peak, min=0.0)

        relay_presence = getattr(self, "goal_map_checkpoint_presence", None)
        if relay_presence is None:
            relay_presence = relay_seeds
        relay_presence = torch.as_tensor(
            relay_presence, device=self.device, dtype=torch.float32
        )
        if relay_presence.dim() == 1:
            relay_presence = relay_presence.unsqueeze(0)
        if relay_presence.shape[0] != relay_seeds.shape[0]:
            relay_presence = relay_seeds
        if relay_presence.shape[1] != self.num_place_cells_total:
            fixed_presence = torch.zeros_like(relay_seeds)
            n_copy = min(self.num_place_cells_total, relay_presence.shape[1])
            fixed_presence[:, :n_copy] = relay_presence[:, :n_copy]
            relay_presence = fixed_presence
        presence_peak = torch.amax(torch.abs(relay_presence), dim=1, keepdim=True).clamp_min(1e-8)
        relay_presence = torch.clamp(relay_presence / presence_peak, min=0.0)

        relay_weights = getattr(self, "goal_map_checkpoint_weights", None)
        if relay_weights is None:
            relay_weights = torch.ones(
                relay_seeds.shape[0], dtype=torch.float32, device=self.device
            )
        else:
            relay_weights = torch.as_tensor(
                relay_weights, device=self.device, dtype=torch.float32
            ).view(-1)
            if relay_weights.numel() != relay_seeds.shape[0]:
                fixed_weights = torch.ones(
                    relay_seeds.shape[0], dtype=torch.float32, device=self.device
                )
                n_copy = min(relay_weights.numel(), relay_seeds.shape[0])
                fixed_weights[:n_copy] = relay_weights[:n_copy]
                relay_weights = fixed_weights
        relay_weights = torch.clamp(relay_weights, min=0.0)

        relay_threshold = float(
            max(0.0, getattr(self, "goal_map_checkpoint_threshold", 0.01))
        )
        return relay_seeds, relay_presence, relay_weights, relay_threshold

    def _checkpoint_relay_emission(
        self,
        backbone_support: torch.Tensor,
        relay_seeds: torch.Tensor,
        relay_presence: torch.Tensor,
        relay_weights: torch.Tensor,
        relay_triggered: torch.Tensor,
        relay_threshold: float,
    ):
        """
        Fire one-shot checkpoint relays when the backbone reaches them.

        Detection uses the strongest overlap between the current accumulated
        backbone support and each checkpoint presence template. Fired
        checkpoints emit a new checkpoint-centered backbone seed scaled by the
        downstream value that has already reached them and the configured relay
        weight. By default the arriving support is normalized by the firing
        threshold, so a checkpoint that barely crosses threshold still launches
        a meaningful subordinate relay instead of dying out immediately.
        """
        if backbone_support.dim() != 1:
            backbone_support = backbone_support.view(-1)
        available = ~relay_triggered
        if not bool(torch.any(available).item()):
            return torch.zeros_like(backbone_support), relay_triggered, 0

        emission_support = torch.max(
            relay_presence[available] * backbone_support.unsqueeze(0),
            dim=1,
        ).values
        fired_local = emission_support >= float(relay_threshold)
        if not bool(torch.any(fired_local).item()):
            return torch.zeros_like(backbone_support), relay_triggered, 0

        available_idx = torch.nonzero(available, as_tuple=False).squeeze(1)
        fired_idx = available_idx[fired_local]
        fired_support = emission_support[fired_local]
        fired_weights = relay_weights[fired_idx]
        support_mode = str(
            getattr(self, "goal_map_checkpoint_support_mode", "threshold_normalized")
        ).strip().lower()
        if support_mode in {"threshold_normalized", "threshold", "normalized"}:
            support_scale = fired_support / max(float(relay_threshold), 1e-6)
            support_scale = torch.clamp(support_scale, min=0.0, max=1.0)
            amplitudes = torch.clamp(fired_weights * support_scale, min=0.0)
        else:
            amplitudes = torch.clamp(fired_support * fired_weights, min=0.0)
        relay_emit = torch.max(
            relay_seeds[fired_idx] * amplitudes.unsqueeze(1),
            dim=0,
        ).values
        relay_triggered = relay_triggered.clone()
        relay_triggered[fired_idx] = True
        return relay_emit, relay_triggered, int(fired_idx.numel())

    def _competitive_relay_fill_unrewarded(
        self,
        path_transition: torch.Tensor,
        neighbor_transition: torch.Tensor,
        relay_seed: torch.Tensor,
        existing_path_values: torch.Tensor,
        existing_neighbor_values: torch.Tensor,
        path_steps_remaining: int,
        path_decay: float,
        neighbor_steps: int,
        neighbor_decay: float,
        neighbor_seed_scale: float,
        neighbor_frontier_only: bool,
        eps: float = 1e-8,
    ):
        """
        Expand one checkpoint relay only into currently unrewarded territory.

        The checkpoint itself can already lie on the goal-side backbone. Its
        replay should therefore seed upstream expansion without being allowed to
        strengthen or rewrite cells that already carry backbone reward.
        Downstream neighbor blur should not count as owned territory here,
        otherwise weak spill from the goal segment can block checkpoint relay
        from establishing an upstream room-local gradient.
        """
        relay_seed = torch.clamp(
            relay_seed.to(self.device, dtype=torch.float32), min=0.0
        ).view(-1)
        if relay_seed.numel() != self.num_place_cells_total:
            fixed_seed = torch.zeros(
                self.num_place_cells_total, dtype=torch.float32, device=self.device
            )
            n_copy = min(self.num_place_cells_total, relay_seed.numel())
            fixed_seed[:n_copy] = relay_seed[:n_copy]
            relay_seed = fixed_seed

        occupied = existing_path_values > eps
        relay_path_values = torch.zeros_like(existing_path_values)
        relay_neighbor_values = torch.zeros_like(existing_neighbor_values)
        frontier_source = relay_seed

        max_steps = int(max(0, path_steps_remaining))
        if max_steps <= 0 or float(torch.max(frontier_source).item()) <= eps:
            return relay_path_values, relay_neighbor_values

        for _ in range(max_steps):
            relay_wave = path_decay * self._competitive_backup_step(
                path_transition, frontier_source
            )
            relay_frontier = torch.where(
                (~occupied) & (relay_wave > (relay_path_values + eps)),
                relay_wave,
                torch.zeros_like(relay_wave),
            )
            relay_path_values = torch.maximum(relay_path_values, relay_frontier)

            if neighbor_steps > 0 and neighbor_seed_scale > 0.0:
                emitted_neighbors = self._competitive_neighbor_from_source(
                    transition=neighbor_transition,
                    source_wave=neighbor_seed_scale * relay_frontier,
                    num_steps=neighbor_steps,
                    step_decay=neighbor_decay,
                    base_values=torch.maximum(
                        torch.maximum(existing_path_values, existing_neighbor_values),
                        torch.maximum(relay_path_values, relay_neighbor_values),
                    ),
                    frontier_only=neighbor_frontier_only,
                )
                relay_neighbor_values = torch.maximum(
                    relay_neighbor_values, emitted_neighbors
                )

            if float(torch.max(relay_frontier).item()) <= eps:
                break
            frontier_source = relay_frontier

        return relay_path_values, relay_neighbor_values

    def _build_goal_map_path_and_neighbor_values(
        self,
        path_transition: torch.Tensor,
        neighbor_transition: torch.Tensor,
        seed_activations: torch.Tensor,
    ):
        """
        Build the path backbone and local neighbor field together.

        Unlike the older implementation, the neighbor branch is emitted from
        each newly reached path frontier, not from the final union of all path
        values. This keeps lateral spread subordinate to the traversed backbone
        and avoids open-area blobs dominating the map.
        """
        seed = torch.clamp(
            seed_activations.to(self.device, dtype=torch.float32), min=0.0
        ).view(-1)
        if seed.numel() != self.num_place_cells_total:
            fixed_seed = torch.zeros(
                self.num_place_cells_total, dtype=torch.float32, device=self.device
            )
            n_copy = min(self.num_place_cells_total, seed.numel())
            fixed_seed[:n_copy] = seed[:n_copy]
            seed = fixed_seed

        seed_peak = float(torch.max(seed).item())
        if seed_peak > 1e-12:
            seed = seed / seed_peak
        seed = seed * float(max(0.0, getattr(self, "goal_map_seed_gain", 1.0)))

        path_values = seed.clone()
        path_wave = seed.clone()
        path_frontier = seed.clone()
        neighbor_values = torch.zeros_like(path_values)

        path_steps = int(max(1, self.replay_timesteps))
        path_decay = float(getattr(self, "goal_map_path_decay", 0.96))
        path_frontier_only = bool(getattr(self, "goal_map_path_frontier_only", True))
        neighbor_steps = int(max(0, getattr(self, "goal_map_neighbor_steps", 0)))
        neighbor_decay = float(getattr(self, "goal_map_neighbor_decay", 0.84))
        neighbor_frontier_only = bool(
            getattr(self, "goal_map_neighbor_frontier_only", True)
        )
        neighbor_seed_scale = float(
            min(1.0, max(0.0, getattr(self, "goal_map_neighbor_seed_scale", 0.60)))
        )
        eps = 1e-8
        relay_payload = self._resolve_checkpoint_relay_payload()
        relay_trigger_count = 0
        stop_parent_at_checkpoint = bool(
            getattr(self, "goal_map_checkpoint_stop_parent_replay", True)
        )
        parent_handed_off = False
        if relay_payload is not None:
            relay_seeds, relay_presence, relay_weights, relay_threshold = relay_payload
            relay_triggered = torch.zeros(
                relay_seeds.shape[0], dtype=torch.bool, device=self.device
            )
        else:
            relay_triggered = None

        if neighbor_steps > 0 and neighbor_seed_scale > 0.0:
            initial_neighbors = self._competitive_neighbor_from_source(
                transition=neighbor_transition,
                source_wave=neighbor_seed_scale * path_frontier,
                num_steps=neighbor_steps,
                step_decay=neighbor_decay,
                base_values=path_values,
                frontier_only=neighbor_frontier_only,
            )
            neighbor_values = torch.maximum(neighbor_values, initial_neighbors)

        for step_idx in range(1, path_steps):
            if parent_handed_off:
                path_wave = torch.zeros_like(path_wave)
                path_frontier = torch.zeros_like(path_frontier)
                emission = torch.zeros_like(path_values)
            else:
                source_path = path_frontier if path_frontier_only else path_wave
                path_wave = path_decay * self._competitive_backup_step(
                    path_transition, source_path
                )
                if path_frontier_only:
                    path_frontier = torch.where(
                        path_wave > (path_values + eps),
                        path_wave,
                        torch.zeros_like(path_wave),
                    )
                    path_values = torch.maximum(path_values, path_frontier)
                    emission = path_frontier
                else:
                    path_values = torch.maximum(path_values, path_wave)
                    emission = path_wave

            if relay_triggered is not None:
                relay_added_total = torch.zeros_like(path_values)
                relay_fired_any = False
                while True:
                    relay_emission, relay_triggered, fired_now = self._checkpoint_relay_emission(
                        backbone_support=path_values,
                        relay_seeds=relay_seeds,
                        relay_presence=relay_presence,
                        relay_weights=relay_weights,
                        relay_triggered=relay_triggered,
                        relay_threshold=relay_threshold,
                    )
                    if fired_now <= 0:
                        break
                    relay_trigger_count += fired_now
                    relay_fired_any = True
                    relay_path_fill, relay_neighbor_fill = (
                        self._competitive_relay_fill_unrewarded(
                            path_transition=path_transition,
                            neighbor_transition=neighbor_transition,
                            relay_seed=relay_emission,
                            existing_path_values=path_values,
                            existing_neighbor_values=neighbor_values,
                            path_steps_remaining=max(1, path_steps - step_idx),
                            path_decay=path_decay,
                            neighbor_steps=neighbor_steps,
                            neighbor_decay=neighbor_decay,
                            neighbor_seed_scale=neighbor_seed_scale,
                            neighbor_frontier_only=neighbor_frontier_only,
                            eps=eps,
                        )
                    )
                    path_values = torch.maximum(path_values, relay_path_fill)
                    neighbor_values = torch.maximum(
                        neighbor_values, relay_neighbor_fill
                    )
                    relay_added = torch.maximum(relay_path_fill, relay_neighbor_fill)
                    relay_added_total = torch.maximum(relay_added_total, relay_added)

                if relay_fired_any:
                    if stop_parent_at_checkpoint:
                        parent_handed_off = True
                        path_wave = torch.zeros_like(path_wave)
                        path_frontier = torch.zeros_like(path_frontier)
                        emission = relay_added_total
                    else:
                        emission = torch.maximum(emission, relay_added_total)

            if float(torch.max(emission).item()) <= eps:
                break

            if neighbor_steps > 0 and neighbor_seed_scale > 0.0:
                emitted_neighbors = self._competitive_neighbor_from_source(
                    transition=neighbor_transition,
                    source_wave=neighbor_seed_scale * emission,
                    num_steps=neighbor_steps,
                    step_decay=neighbor_decay,
                    base_values=torch.maximum(path_values, neighbor_values),
                    frontier_only=neighbor_frontier_only,
                )
                neighbor_values = torch.maximum(neighbor_values, emitted_neighbors)

        return path_values, neighbor_values, relay_trigger_count

    def build_goal_map_from_custom_activations(
        self,
        unified_pcn,
        custom_activations: torch.Tensor,
    ) -> str:
        """
        Build unified reward weights for a goal seed.

        The new default mode keeps the computation in place-cell space:
        1. Backward value propagation on experienced place-cell transitions.
        2. Weak local spread on symmetric place-cell adjacency.
        """
        pcn_copy = copy.deepcopy(unified_pcn)
        recurrent_weights_max = torch.max(
            pcn_copy.w_rec_unified.to(self.device), dim=0
        )[0]
        local_transition = self._prepare_replay_transition(recurrent_weights_max)
        path_transition = self._prepare_goal_map_path_transition(local_transition)
        neighbor_transition = self._prepare_goal_map_neighbor_transition(recurrent_weights_max)
        path_values, neighbor_values, relay_trigger_count = self._build_goal_map_path_and_neighbor_values(
            path_transition=path_transition,
            neighbor_transition=neighbor_transition,
            seed_activations=custom_activations,
        )
        final_values = torch.maximum(path_values, neighbor_values)

        peak = float(torch.max(final_values).item())
        if peak > 1e-12:
            final_values = final_values / peak

        self.w_in = final_values.unsqueeze(0)
        self.w_in_effective = self.w_in.clone()
        self.reward_cell_activations = torch.zeros_like(self.reward_cell_activations)
        seed_gain = float(getattr(self, "goal_map_seed_gain", 1.0))
        relay_seed_count = int(
            0
            if getattr(self, "goal_map_checkpoint_weights", None) is None
            else torch.as_tensor(getattr(self, "goal_map_checkpoint_weights")).numel()
        )
        self.goal_map_checkpoint_seeds = None
        self.goal_map_checkpoint_presence = None
        self.goal_map_checkpoint_weights = None
        self.goal_map_seed_gain = 1.0
        neighbor_steps = int(max(0, getattr(self, "goal_map_neighbor_steps", 0)))
        return (
            "goal_map=competitive_path_neighbor("
            f"path_topk={int(getattr(self, 'goal_map_path_topk', 16))},"
            f"path_decay={float(getattr(self, 'goal_map_path_decay', 0.96)):.2f},"
            f"neighbor_topk={int(getattr(self, 'goal_map_neighbor_topk', 12))},"
            f"neighbor_steps={neighbor_steps},"
            f"neighbor_decay={float(getattr(self, 'goal_map_neighbor_decay', 0.84)):.2f},"
            f"neighbor_seed_scale={float(getattr(self, 'goal_map_neighbor_seed_scale', 0.60)):.2f},"
            f"checkpoint_support_mode={str(getattr(self, 'goal_map_checkpoint_support_mode', 'threshold_normalized'))},"
            f"checkpoint_stop_parent={int(bool(getattr(self, 'goal_map_checkpoint_stop_parent_replay', True)))},"
            "checkpoint_preserve_existing=1,"
            f"seed_gain={seed_gain:.2f},"
            f"checkpoint_relays={relay_trigger_count}/{relay_seed_count})"
        )

    def _reward_denominators(self, place_cell_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute one denominator per replay/query state.

        Reward readout uses paper-style normalization by imagined PC activity mass.
        """
        return torch.sum(torch.abs(place_cell_activations), dim=1).clamp(min=1e-4)

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

        # Compute reward: r = W_in @ v^p, normalized by imagined activity mass.
        safe_denominators = self._reward_denominators(place_cell_activations)

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
            replay_gate = self._get_replay_gate(
                pcn_copy,
                activations=place_cell_activations,
            )
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

            place_cell_activations = self._advance_replay_wave(
                transition=transition,
                place_cell_activations=place_cell_activations,
                replay_gate=replay_gate,
            )

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
            replay_gate = self._get_replay_gate(
                pcn_copy,
                activations=place_cell_activations,
            )
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

            place_cell_activations = self._advance_replay_wave(
                transition=transition,
                place_cell_activations=place_cell_activations,
                replay_gate=replay_gate,
            )

        # Stability guard
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and max_val > 1e3:
            weight_update = weight_update / max_val

        # Update weights
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

    def _advance_replay_wave(
        self,
        transition: torch.Tensor,
        place_cell_activations: torch.Tensor,
        replay_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the replay state by one step.

        `replay_residual_mix=0` gives a pure wavefront update driven by the
        replay transition graph. Larger values reintroduce the old residual
        carry and produce flatter long-range maps.
        """
        dot_val = torch.matmul(transition.float(), place_cell_activations)
        residual_mix = float(
            min(1.0, max(0.0, getattr(self, "replay_residual_mix", 0.0)))
        )
        if residual_mix > 0.0:
            dot_val = dot_val + (residual_mix * place_cell_activations)
        updated_place_cell_activations = torch.relu(dot_val)
        next_activations = torch.tanh(updated_place_cell_activations)
        return next_activations * replay_gate

    def _get_replay_gate(
        self,
        unified_pcn,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fetch the segmented replay gate from the unified PCN.

        Falls back to all-ones when no segmented expression state is available.
        """
        gate = None
        if hasattr(unified_pcn, "get_segmented_expression_gate_per_pc"):
            gate = unified_pcn.get_segmented_expression_gate_per_pc(
                activations=activations
            )
        elif hasattr(unified_pcn, "get_last_scale_preference_per_pc"):
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

    def compute_reward_contribution_for_scale_batched(
        self,
        place_cell_activations_batch: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        """
        Compute the reward contribution for a single scale over a batch.

        Used by scale-specific preplay scoring when the caller only needs one
        scale's contribution.
        """
        if place_cell_activations_batch.dim() == 1:
            place_cell_activations_batch = place_cell_activations_batch.unsqueeze(0)

        place_cell_activations_batch = place_cell_activations_batch.to(self.device)
        scale_idx = int(scale_idx)
        if scale_idx < 0 or scale_idx >= self.num_scales:
            return torch.zeros(
                place_cell_activations_batch.shape[0],
                dtype=place_cell_activations_batch.dtype,
                device=self.device,
            )

        start = self.scale_boundaries[scale_idx]
        end = self.scale_boundaries[scale_idx + 1]
        scale_weights = self.w_in_effective[:, start:end]
        scale_acts = place_cell_activations_batch[:, start:end]
        numerators = torch.matmul(scale_acts, scale_weights.T).squeeze(1)
        denominators = self._reward_denominators(place_cell_activations_batch)
        return numerators / denominators

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

        # Compute rewards using the same input-L1 normalization rule as the single-state path.
        safe_denominators = self._reward_denominators(place_cell_activations_batch)

        activations = torch.matmul(
            place_cell_activations_batch,
            self.w_in_effective.T
        ).squeeze(1) / safe_denominators

        # Clamp activations
        activations = torch.clamp(activations, 0, 1e6)

        return activations
