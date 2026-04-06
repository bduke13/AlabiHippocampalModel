"""Multiscale Controller Test - Enhanced with LEARN_LOCATIONS_COVERAGE and EXPLOIT_LOCATIONS_RANDOM modes"""

import sys
import os
import re
import gc
import copy
import json
import math
import torch
from pathlib import Path
from datetime import datetime

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from core.layers.reward_cell_layer_v11 import C_LAMBDA
from msg_driver import Driver
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector
from path_planning import generate_spawn_locations

#################################
# Utility Functions
#################################

def get_highest_trial_id(stats_folder, corner):
    """
    Reads the stats directory and determines the highest trial ID for the given corner.
    """
    trial_ids = []
    if os.path.exists(stats_folder):
        for file_name in os.listdir(stats_folder):
            match = re.match(rf"trial_(\d+)_corner_{corner[0]}_{corner[1]}", file_name)
            if match:
                trial_ids.append(int(match.group(1)))
    return max(trial_ids) if trial_ids else 0

def get_world_name(bot):
    """
    Determines the current world name dynamically from the .wbt file.
    """
    world_path = bot.getWorldPath()
    return os.path.basename(world_path).replace('.wbt', '')

def save_trial_parameters(world_name, trial_id, mode, **kwargs):
    """
    Saves trial parameters to a JSON file in the pkl folder (alongside trial data).

    Args:
        world_name: Name of the world
        trial_id: Unique identifier for the trial
        mode: Robot mode being used
        **kwargs: All parameters passed to the trial
    """
    # Determine the correct folder based on mode (matching msg_driver_v10.py logic)
    auto_trial_name = kwargs.get("auto_trial_name")
    current_auto_trial = kwargs.get("current_auto_trial")

    if mode in {RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
                RobotMode.PLOTTING_AUTO, RobotMode.PLOTTING_COVERAGE_AUTO} and auto_trial_name:
        # Auto trial mode: pkl_{trial_name}/{world_name}_{trial_num}/
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{current_auto_trial}"
        save_folder = os.path.join(PROJECT_ROOT, "webots", "controllers", "msg_controller", base_folder, trial_folder)
    else:
        # Standard mode: pkl/{world_name}/
        save_folder = os.path.join(PROJECT_ROOT, "webots", "controllers", "msg_controller", "pkl", world_name)

    os.makedirs(save_folder, exist_ok=True)

    # Resolve effective model configuration after ablation overrides.
    prepared_kwargs, scales_list, ablation_log = prepare_trial_configuration(kwargs)
    kwargs = prepared_kwargs
    scale_names = kwargs.get("scale_names", [])

    # Prepare parameters to save
    trial_params = {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "world_name": world_name,
        "mode": str(mode),
        "scale_names": scale_names,
        "scales_definitions": scales_list,  # Full scale definitions
        "robot_parameters": {
            "start_location": kwargs.get("start_loc", [0, 0]),
            "randomize_start_loc": kwargs.get("randomize_start_loc", False),
            "run_time_hours": kwargs.get("run_time_hours", 2),
            "max_dist": kwargs.get("max_dist", 25),
            "enable_ojas": kwargs.get("enable_ojas", None),
            "enable_stdp": kwargs.get("enable_stdp", None),
            "td_learning": kwargs.get("td_learning", False),
            "use_prox_mod": kwargs.get("use_prox_mod", False),
            "plot_bvc": kwargs.get("plot_bvc", False),
            "use_unified_multiscale": kwargs.get("use_unified_multiscale", False),
            "gcn_scale_invariant": kwargs.get("gcn_scale_invariant", True),
            "gcn_use_adaptive_minmax": kwargs.get("gcn_use_adaptive_minmax", False),
            "goal_assoc_unique_topk": kwargs.get("goal_assoc_unique_topk", 16),
            "goal_assoc_max_activation_drop": kwargs.get("goal_assoc_max_activation_drop", 0.08),
            "goal_assoc_locality_radius": kwargs.get("goal_assoc_locality_radius", 2.0),
            "goal_assoc_locality_max_activation_drop": kwargs.get("goal_assoc_locality_max_activation_drop", 0.65),
            "goal_map_use_compact_pc_gate": kwargs.get("goal_map_use_compact_pc_gate", True),
            "goal_map_pc_compactness_peak_fraction": kwargs.get("goal_map_pc_compactness_peak_fraction", 0.50),
            "goal_map_pc_compactness_min_peak": kwargs.get("goal_map_pc_compactness_min_peak", 0.05),
            "goal_map_pc_compactness_max_radius": kwargs.get("goal_map_pc_compactness_max_radius", 4.0),
            "goal_map_fill_source_envelope_tau": kwargs.get("goal_map_fill_source_envelope_tau", 6.0),
            "goal_map_fill_source_envelope_floor": kwargs.get("goal_map_fill_source_envelope_floor", 0.10),
            "goal_assoc_require_all_scales": kwargs.get("goal_assoc_require_all_scales", False),
            "record_experience_transitions": kwargs.get("record_experience_transitions", True),
            "two_phase_learning": kwargs.get("two_phase_learning", False),
            "phase1_min_steps": kwargs.get("phase1_min_steps", 2500),
            "phase1_bin_size": kwargs.get("phase1_bin_size", 0.5),
            "phase1_min_revisit_bins": kwargs.get("phase1_min_revisit_bins", 15),
            "phase1_revisit_cosine_threshold": kwargs.get("phase1_revisit_cosine_threshold", 0.90),
            "phase1_revisit_window": kwargs.get("phase1_revisit_window", 200),
            "defer_experience_build_until_phase2_end": kwargs.get("defer_experience_build_until_phase2_end", True),
            "goal_map_replay_timesteps": kwargs.get("goal_map_replay_timesteps", 12),
            "goal_map_paper_replay_tau": kwargs.get("goal_map_paper_replay_tau", 8.0),
            "goal_map_goal_replay_event_count": kwargs.get(
                "goal_map_goal_replay_event_count", 4
            ),
            "goal_map_checkpoint_replay_event_count": kwargs.get(
                "goal_map_checkpoint_replay_event_count", 3
            ),
            "goal_map_replay_event_merge_gap_steps": kwargs.get(
                "goal_map_replay_event_merge_gap_steps", 3
            ),
            "goal_map_checkpoint_room_support_min_fraction": kwargs.get(
                "goal_map_checkpoint_room_support_min_fraction", 0.98
            ),
            "hybrid_path_replay_weight": kwargs.get("hybrid_path_replay_weight", 0.8),
            "hybrid_diffusion_replay_weight": kwargs.get("hybrid_diffusion_replay_weight", 0.2),
            "goal_map_path_replay_weight": kwargs.get("goal_map_path_replay_weight", 1.0),
            "goal_map_diffusion_replay_weight": kwargs.get("goal_map_diffusion_replay_weight", 0.0),
            "goal_map_experience_transition_topk": kwargs.get("goal_map_experience_transition_topk", 16),
            "unified_reward_normalization_mode": kwargs.get("unified_reward_normalization_mode", "input_l1"),
            "unified_goal_map_mode": kwargs.get("unified_goal_map_mode", "paper_room_local_replay"),
            "goal_map_room_normalization_mode": kwargs.get("goal_map_room_normalization_mode", "per_room_peak"),
            "goal_map_path_topk": kwargs.get("goal_map_path_topk", 16),
            "goal_map_path_decay": kwargs.get("goal_map_path_decay", 0.97),
            "goal_map_path_frontier_only": kwargs.get("goal_map_path_frontier_only", True),
            "goal_map_neighbor_topk": kwargs.get("goal_map_neighbor_topk", 12),
            "goal_map_neighbor_steps": kwargs.get("goal_map_neighbor_steps", 6),
            "goal_map_neighbor_decay": kwargs.get("goal_map_neighbor_decay", 0.84),
            "goal_map_neighbor_frontier_only": kwargs.get("goal_map_neighbor_frontier_only", True),
            "goal_map_neighbor_seed_scale": kwargs.get("goal_map_neighbor_seed_scale", 0.60),
            "goal_map_goal_seed_mode": kwargs.get("goal_map_goal_seed_mode", "contact_ensemble"),
            "goal_map_goal_replay_retention_mode": kwargs.get(
                "goal_map_goal_replay_retention_mode", "baseline"
            ),
            "goal_map_goal_seed_post_contact_steps": kwargs.get(
                "goal_map_goal_seed_post_contact_steps", 2
            ),
            "goal_map_goal_seed_within_visit_aggregation": kwargs.get(
                "goal_map_goal_seed_within_visit_aggregation", "max"
            ),
            "goal_map_goal_seed_across_visit_aggregation": kwargs.get(
                "goal_map_goal_seed_across_visit_aggregation", "mean"
            ),
            "goal_map_goal_seed_state_normalization": kwargs.get(
                "goal_map_goal_seed_state_normalization", "peak"
            ),
            "goal_map_goal_seed_self_retention_alpha": kwargs.get(
                "goal_map_goal_seed_self_retention_alpha", 0.15
            ),
            "goal_map_path_constraint_sigma": kwargs.get("goal_map_path_constraint_sigma", 4.0),
            "goal_map_path_constraint_topk": kwargs.get("goal_map_path_constraint_topk", 128),
            "goal_map_local_constraint_sigma": kwargs.get("goal_map_local_constraint_sigma", 2.6),
            "goal_map_local_constraint_topk": kwargs.get("goal_map_local_constraint_topk", 36),
            "goal_map_goal_seed_center_radius_factor": kwargs.get("goal_map_goal_seed_center_radius_factor", 0.0),
            "goal_map_goal_seed_center_min_radius": kwargs.get("goal_map_goal_seed_center_min_radius", 0.0),
            "goal_map_checkpoint_seed_center_radius_factor": kwargs.get("goal_map_checkpoint_seed_center_radius_factor", 0.0),
            "goal_map_checkpoint_seed_center_min_radius": kwargs.get("goal_map_checkpoint_seed_center_min_radius", 0.6),
            "goal_map_spatial_obstacle_block": kwargs.get("goal_map_spatial_obstacle_block", True),
            "goal_map_obstacle_margin": kwargs.get("goal_map_obstacle_margin", 0.05),
            "goal_map_use_checkpoint_relays": kwargs.get("goal_map_use_checkpoint_relays", False),
            "goal_map_checkpoint_mode": kwargs.get("goal_map_checkpoint_mode", "multi_route_memory_graph"),
            "goal_map_checkpoint_scale": kwargs.get("goal_map_checkpoint_scale", 0.25),
            "goal_map_checkpoint_support_mode": kwargs.get("goal_map_checkpoint_support_mode", "threshold_normalized"),
            "goal_map_checkpoint_stop_parent_replay": kwargs.get("goal_map_checkpoint_stop_parent_replay", True),
            "checkpoint_boost_threshold": kwargs.get("checkpoint_boost_threshold", 0.01),
            "checkpoint_target_arrival": kwargs.get("checkpoint_target_arrival", 0.15),
            "checkpoint_min_relay_weight": kwargs.get("checkpoint_min_relay_weight", 1.0),
            "goal_map_route_selection_support_fraction": kwargs.get("goal_map_route_selection_support_fraction", 0.15),
            "goal_map_route_selection_overlap_threshold": kwargs.get("goal_map_route_selection_overlap_threshold", 0.08),
            "goal_map_use_denoised_experience_graph": kwargs.get("goal_map_use_denoised_experience_graph", True),
            "goal_map_use_segmented_checkpoint_histories": kwargs.get("goal_map_use_segmented_checkpoint_histories", True),
            "unified_preplay_scale_arbitration": kwargs.get("unified_preplay_scale_arbitration", True),
            "unified_preplay_scale_prior_mix": kwargs.get("unified_preplay_scale_prior_mix", 0.0),
            "goal_map_smoothing_sigma": kwargs.get("goal_map_smoothing_sigma", 1.0),
            "goal_map_smoothing_topk": kwargs.get("goal_map_smoothing_topk", 24),
            "goal_map_smoothing_passes": kwargs.get("goal_map_smoothing_passes", 0),
            "goal_map_smoothing_mix": kwargs.get("goal_map_smoothing_mix", 0.45),
            "unified_preplay_commit_threshold": kwargs.get("unified_preplay_commit_threshold", 0.30),
            "preplay_heading_selection_mode": kwargs.get("preplay_heading_selection_mode", "trajectory_vector"),
            "unified_exploit_action_mode": kwargs.get("unified_exploit_action_mode", "local_reward_gradient_primary"),
            "unified_transition_preplay_enabled": kwargs.get("unified_transition_preplay_enabled", False),
            "unified_spatial_fallback_enabled": kwargs.get("unified_spatial_fallback_enabled", True),
            "unified_spatial_override_margin": kwargs.get("unified_spatial_override_margin", 0.0),
            "unified_spatial_lookahead_distance": kwargs.get("unified_spatial_lookahead_distance", 1.0),
            "unified_spatial_lookahead_sigma": kwargs.get("unified_spatial_lookahead_sigma", 0.9),
            "unified_exploit_step_fraction": kwargs.get("unified_exploit_step_fraction", 0.25),
            "exploit_direction_clearance_percentile": kwargs.get("exploit_direction_clearance_percentile", 20.0),
            "prune_experience_loops": kwargs.get("prune_experience_loops", True),
            "loop_prune_min_top1": kwargs.get("loop_prune_min_top1", 0.08),
            "loop_prune_min_top1_to_top2_ratio": kwargs.get("loop_prune_min_top1_to_top2_ratio", 1.15),
        },
        "unified_ablation": {
            "flags": prepared_kwargs.get("unified_ablation_flags", {}),
            "applied_log": ablation_log,
        },
        "goal_config": kwargs.get("goal_config"),
        "trial_config": kwargs.get("trial_config"),
        "environment_parameters": {
            "environment_size": kwargs.get("environment_size"),
            "grid_size": kwargs.get("grid_size"),
            "coverage_percentage": kwargs.get("coverage_percentage"),
            "minimum_learning_duration_hours": kwargs.get("minimum_learning_duration_hours", 4.0),
            "min_goal_visits": kwargs.get("min_goal_visits", 3),
            "min_checkpoint_visits": kwargs.get("min_checkpoint_visits", 3),
            "checkpoint_visit_radius": kwargs.get("checkpoint_visit_radius", 0.8),
            "goal_visit_cooldown_seconds": kwargs.get("goal_visit_cooldown_seconds", 6.0),
            "goal_exit_hysteresis": kwargs.get("goal_exit_hysteresis", 0.1),
            "proximity_mode": kwargs.get("proximity_mode", "min"),
            "proximity_trimmed_sigma": kwargs.get("proximity_trimmed_sigma", 2.5),
            "proximity_pair_percentile": kwargs.get("proximity_pair_percentile", 25.0),
            "pcn_gate_mode": kwargs.get("pcn_gate_mode", "normal"),
            "pcn_learning_adaptation_mode": kwargs.get("pcn_learning_adaptation_mode", "gaussian_post_competition_expression"),
            "pcn_learning_stdp_start_steps": kwargs.get("pcn_learning_stdp_start_steps", 8000),
            "pcn_learning_stop_ojas_on_stabilization": kwargs.get("pcn_learning_stop_ojas_on_stabilization", False),
            "pcn_learning_cross_scale_coupling_start_steps": kwargs.get("pcn_learning_cross_scale_coupling_start_steps", 8000),
            "pcn_learning_cross_scale_coupling_ramp_steps": kwargs.get("pcn_learning_cross_scale_coupling_ramp_steps", 12000),
            "pcn_learning_cross_scale_coupling_min": kwargs.get("pcn_learning_cross_scale_coupling_min", 0.0),
            "pcn_soft_scale_overlap": kwargs.get("pcn_soft_scale_overlap", True),
            "pcn_soft_scale_overlap_in_learning": kwargs.get("pcn_soft_scale_overlap_in_learning", True),
            "pcn_soft_scale_gate_floor": kwargs.get("pcn_soft_scale_gate_floor", 0.20),
            "pcn_soft_scale_gate_floor_in_learning": kwargs.get("pcn_soft_scale_gate_floor_in_learning", 0.12),
            "pcn_soft_cross_inhibition_scale": kwargs.get("pcn_soft_cross_inhibition_scale", 0.35),
            "pcn_soft_cross_inhibition_scale_in_learning": kwargs.get("pcn_soft_cross_inhibition_scale_in_learning", 0.25),
            "pcn_soft_cross_inhibition_cap": kwargs.get("pcn_soft_cross_inhibition_cap", 0.75),
        },
        "path_planning_parameters": {
            "optimal_path_distance": kwargs.get("optimal_path_distance"),
            "path_failure_ratio": kwargs.get("path_failure_ratio"),
            "min_spawn_distance": kwargs.get("min_spawn_distance"),
            "wall_clearance": kwargs.get("wall_clearance"),
        },
        "auto_trial_parameters": {
            "auto_trial_name": kwargs.get("auto_trial_name"),
            "num_auto_trials": kwargs.get("num_auto_trials"),
            "current_auto_trial": kwargs.get("current_auto_trial"),
        }
    }

    # Save to JSON file
    filename = f"{trial_id}_parameters.json"
    filepath = os.path.join(save_folder, filename)

    with open(filepath, 'w') as f:
        json.dump(trial_params, f, indent=2)

    print(f"[INFO] Trial parameters saved to: {filepath}")

    return filepath

#################################
# Scale Definitions
#################################

# Replay step budget is proportional to lambda_s to normalize spread per time constant
STEPS_PER_LAMBDA = 8  # Adjust to push farther (higher) or be more local (lower)

def _steps_for_sigma(sigma_pc_s: float) -> int:
    """Derive custom replay timesteps proportional to lambda_s; sigma drives lambda_s."""
    return int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA))

# Per-scale tuning multipliers for unified scale selection:
# sigma_tune_s = sigma_tune_k_s * sigma_r_s
SIGMA_TUNE_K_SMALL = 1.3
SIGMA_TUNE_K_MEDIUM = 0.8
SIGMA_TUNE_K_LARGE = 0.6
SIGMA_TUNE_K_XLARGE = 1.0

SCALES_DEFS_GRID = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "sigma_pc_s": 1.0,  # Place field size for scale-dependent reward propagation
        "num_pc": 1000,
        "sigma_r": 0.5, #0.5
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 100,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.3,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        "w_grid_init_strategy": "balanced_modules",
        # Place cell recurrent inhibition parameters
        "gamma_pp": 1.0,  # Place-to-place recurrent inhibition strength
        "gamma_pb": 0.35,  # BVC-to-place afferent inhibition strength
        # Unified multi-scale parameters
        "d_opt": 0.7,  # Optimal boundary distance for this scale
        "gamma_cross": 0.6,  # Reduced to ease scale starvation
        "sigma_tune_k": SIGMA_TUNE_K_SMALL,  # sigma_tune = sigma_tune_k * sigma_r
        # Optional per-cell d_opt jitter (biological heterogeneity around scale default)
        "d_opt_jitter_std": 0.15,
        "d_opt_jitter_range": 0.3,
        "d_opt_jitter_seed": 5000,
        # Grid cell parameters
        "grid_influence": 0.3, # 0.3
        "learning_grid_influence_scale": 1.0,
        "gamma_pg": 0.35, # 0.3
        "grid_balance_modalities": False,
        "grid_balance_ema": 0.95,
        "grid_balance_min_gain": 0.1,
        "grid_balance_max_gain": 8.0,
        "num_grid_cells": 800,
        "num_modules": 8,
        "cells_per_module": 100,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 4,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 256,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        # Oja's learning normalization parameters
        "alpha_pb": 0.447,  # np.sqrt(0.5) - BVC weight decay factor (default)
        "alpha_pg": 0.447,  # np.sqrt(0.3) - Grid weight decay factor (stronger decay = more selective)
        # Correlation-based weighting parameters
        "enable_correlation_weighting": False,
        "correlation_window": 8, #12, 10
        "correlation_update_freq": 1, #2
        "correlation_scaling": 10, #2.5
        "min_correlation_weight": 0.10, #0.03
        "correlation_threshold": 0.005, #0.015
        # Reward cell replay parameters
        "replay_timesteps": 100,  # Number of timesteps for regular replay
        "replay_decay_factor": 3,  # FAST decay - rewards diminish quickly with distance
        "custom_replay_timesteps": _steps_for_sigma(1.0),  # Spread scaled to lambda_s
        "initial_value_multiplier": 2.0,  # HIGHEST starting value - strongest near goal

    },
    "medium": {
        "scale_index": 1,
        "name": "medium",
        "sigma_pc_s": 1.5,  # Place field size for scale-dependent reward propagation
        "num_pc": 500,
        "sigma_r": 1.0, #2
        "sigma_theta": 3,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 50,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.25,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        "w_grid_init_strategy": "balanced_modules",
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_pb": 0.3,  # BVC-to-place afferent inhibition strength
        # Unified multi-scale parameters
        "d_opt": 2.5,
        "gamma_cross": 0.65,
        "sigma_tune_k": SIGMA_TUNE_K_MEDIUM,  # sigma_tune = sigma_tune_k * sigma_r
        # Optional per-cell d_opt jitter (biological heterogeneity around scale default)
        "d_opt_jitter_std": 0.35,
        "d_opt_jitter_range": 0.7,
        "d_opt_jitter_seed": 5001,
        # Grid cell parameters
        "grid_influence": 0.35, #0.35
        "learning_grid_influence_scale": 1.0,
        "gamma_pg": 0.35,
        "grid_balance_modalities": False,
        "grid_balance_ema": 0.95,
        "grid_balance_min_gain": 0.1,
        "grid_balance_max_gain": 8.0,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 5.5,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        # Oja's learning normalization parameters
        "alpha_pb": 0.632,
        "alpha_pg": 0.632,
        # Correlation-based weighting parameters
        "enable_correlation_weighting": False,
        "correlation_window": 12, #18, 15
        "correlation_update_freq": 1, #4
        "correlation_scaling": 10, #3.0
        "min_correlation_weight": 0.10, #0.06
        "correlation_threshold": 0.05, #0.03
        # Reward cell replay parameters
        "replay_timesteps": 40,  # Number of timesteps for regular replay
        "replay_decay_factor": 8,  # Medium decay - balanced propagation
        "custom_replay_timesteps": _steps_for_sigma(1.5),  # Spread scaled to lambda_s
        "initial_value_multiplier": 1.5,  # Medium starting value - strongest at medium distances

    },
    "large": {
        "scale_index": 2,
        "name": "large",
        "sigma_pc_s": 3.0,  # Place field size for scale-dependent reward propagation
        "num_pc": 250,
        "sigma_r": 1.5, # 2.0
        "sigma_theta": 5,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 75,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.2,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.2,  # Proportion of GC->PC weights active initially
        "w_grid_init_strategy": "balanced_modules",
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_pb": 0.25,  # BVC-to-place afferent inhibition strength
        # Unified multi-scale parameters
        "d_opt": 5.0,
        "gamma_cross": 0.7,
        "sigma_tune_k": SIGMA_TUNE_K_LARGE,  # sigma_tune = sigma_tune_k * sigma_r
        # Optional per-cell d_opt jitter (biological heterogeneity around scale default)
        "d_opt_jitter_std": 0.45,
        "d_opt_jitter_range": 0.9,
        "d_opt_jitter_seed": 5002,
        # Keep largest scale recruited in genuinely deep open space via a delayed plateau ramp.
        "large_scale_one_sided": True,
        "large_scale_plateau": 1.0,
        "large_scale_plateau_onset_sigma": 1.0,
        "large_scale_plateau_full_sigma": 2.0,
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "learning_grid_influence_scale": 1.0,
        "gamma_pg": 0.25,
        "grid_balance_modalities": False,
        "grid_balance_ema": 0.95,
        "grid_balance_min_gain": 0.1,
        "grid_balance_max_gain": 8.0,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 7,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        # Correlation-based weighting parameters
        "enable_correlation_weighting": False,
        "correlation_window": 18, #24, 20
        "correlation_update_freq": 1, #6
        "correlation_scaling": 10, #2.5
        "min_correlation_weight": 0.10, #0.12
        "correlation_threshold": 0.05, #0.05
        # Reward cell replay parameters
        "replay_timesteps": 40,  # Number of timesteps for regular replay
        "replay_decay_factor": 20,  # SLOW decay - rewards persist over long distances
        "custom_replay_timesteps": _steps_for_sigma(3.0),  # Spread scaled to lambda_s
        "initial_value_multiplier": 1.0,  # LOWEST starting value - strongest far from goal
    },
    "xlarge": {
        "scale_index": 3,
        "name": "xlarge",
        "sigma_pc_s": 5.0,  # Place field size for scale-dependent reward propagation
        "num_pc": 200,
        "sigma_r": 4,
        "sigma_theta": 8,
        "rcn_learning_rate": 0.005,
        # BVC parameters
        "num_bvc_per_dir": 50,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.25,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.25,  # Proportion of GC->PC weights active initially
        "w_grid_init_strategy": "balanced_modules",
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.5,  # Place-to-place recurrent inhibition strength
        "gamma_pb": 0.3,  # BVC-to-place afferent inhibition strength
        # Unified multi-scale parameters
        "d_opt": 7.5,
        "gamma_cross": 0.33,
        "sigma_tune_k": SIGMA_TUNE_K_XLARGE,  # sigma_tune = sigma_tune_k * sigma_r
        # Optional per-cell d_opt jitter (biological heterogeneity around scale default)
        "d_opt_jitter_std": 0.45,
        "d_opt_jitter_range": 0.9,
        "d_opt_jitter_seed": 5003,
        # Keep largest scale recruited in genuinely deep open space via a delayed plateau ramp.
        "large_scale_one_sided": True,
        "large_scale_plateau": 1.0,
        "large_scale_plateau_onset_sigma": 1.0,
        "large_scale_plateau_full_sigma": 2.0,
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "learning_grid_influence_scale": 1.15,
        "gamma_pg": 0.32,
        "grid_balance_modalities": False,
        "grid_balance_ema": 0.95,
        "grid_balance_min_gain": 0.1,
        "grid_balance_max_gain": 8.0,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.0, 1.0),
        "scale_multiplier": 5.0,
        "module_scale_ratio": 1.8,
        "translation_scale": 1.0,
        "mask_resolution": 96,
        "smooth_sigma": 1.5,
        # Correlation-based weighting parameters
        "enable_correlation_weighting": True,
        "correlation_window": 30,
        "correlation_update_freq": 8,
        "correlation_scaling": 2.0,
        "min_correlation_weight": 0.18,
        "correlation_threshold": 0.08,
        # Reward cell replay parameters
        "replay_timesteps": 20,  # Number of timesteps for regular replay
        "replay_decay_factor": 25,  # Very slow decay - maximum persistence
        "custom_replay_timesteps": _steps_for_sigma(5.0),  # Spread scaled to lambda_s
        "initial_value_multiplier": 0.8,  # Very low starting value - only dominant at extreme distances
    }
}

UNIFIED_ABLATION_FLAG_DEFAULTS = {
    "ablation_force_nonunified_baseline": False,
    "ablation_legacy_grid_runtime": False,
    "ablation_disable_unified_scale_selection": False,
    "ablation_disable_unified_cross_scale_learning": False,
    "ablation_disable_unified_learning_adaptation": False,
    "ablation_disable_unified_dopt_jitter": False,
    "ablation_disable_unified_large_scale_plateau": False,
}


def compile_scales(scale_names, grid_balance_modalities=None):
    """
    Convert a list of scale names (e.g. ["small", "large"]) into a list of
    actual scale definitions from SCALES_DEFS.
    """
    scales = [copy.deepcopy(SCALES_DEFS_GRID[name]) for name in scale_names]
    if grid_balance_modalities is not None:
        enabled = bool(grid_balance_modalities)
        for scale in scales:
            scale["grid_balance_modalities"] = enabled
    return scales


def _extract_unified_ablation_flags(kwargs):
    return {
        name: bool(kwargs.get(name, default))
        for name, default in UNIFIED_ABLATION_FLAG_DEFAULTS.items()
    }


def _apply_unified_ablation_overrides(kwargs, scales):
    prepared_kwargs = copy.deepcopy(kwargs)
    prepared_scales = copy.deepcopy(scales)
    flags = _extract_unified_ablation_flags(prepared_kwargs)
    log = []

    if flags["ablation_force_nonunified_baseline"]:
        prepared_kwargs["use_unified_multiscale"] = False
        prepared_kwargs["minimum_learning_duration_hours"] = 0.0
        prepared_kwargs["goal_assoc_require_all_scales"] = True
        prepared_kwargs["gcn_use_adaptive_minmax"] = True
        log.append("force_nonunified_baseline")

    if flags["ablation_legacy_grid_runtime"]:
        prepared_kwargs["gcn_scale_invariant"] = False
        prepared_kwargs["pcn_grid_balance_modalities"] = False
        for scale in prepared_scales:
            scale["module_scale_ratio"] = 1.0
            scale["activation_cache_size"] = 1024
            scale["activation_cache_quantization"] = 1e-4
            scale["learning_grid_influence_scale"] = 1.0
            scale["grid_balance_modalities"] = False
        log.append("legacy_grid_runtime")

    if flags["ablation_disable_unified_scale_selection"]:
        prepared_kwargs["pcn_gate_mode"] = "no_gate_no_inhibition"
        prepared_kwargs["pcn_soft_scale_overlap"] = False
        prepared_kwargs["pcn_soft_scale_overlap_in_learning"] = False
        prepared_kwargs["pcn_soft_scale_gate_floor"] = 0.0
        prepared_kwargs["pcn_soft_scale_gate_floor_in_learning"] = 0.0
        prepared_kwargs["pcn_soft_cross_inhibition_scale"] = 0.0
        prepared_kwargs["pcn_soft_cross_inhibition_scale_in_learning"] = 0.0
        prepared_kwargs["pcn_soft_cross_inhibition_cap"] = 0.0
        for scale in prepared_scales:
            scale["gamma_cross"] = 0.0
        log.append("disable_unified_scale_selection")

    if flags["ablation_disable_unified_cross_scale_learning"]:
        prepared_kwargs["pcn_learning_cross_scale_coupling_start_steps"] = int(1e9)
        prepared_kwargs["pcn_learning_cross_scale_coupling_ramp_steps"] = 1
        prepared_kwargs["pcn_learning_cross_scale_coupling_min"] = 0.0
        log.append("disable_unified_cross_scale_learning")

    if flags["ablation_disable_unified_learning_adaptation"]:
        prepared_kwargs["pcn_learning_adaptation_mode"] = "gaussian_post_competition_expression"
        log.append("disable_unified_learning_adaptation_deprecated")

    if flags["ablation_disable_unified_dopt_jitter"]:
        for scale in prepared_scales:
            scale["d_opt_jitter_std"] = 0.0
            scale["d_opt_jitter_range"] = 0.0
        log.append("disable_unified_dopt_jitter")

    if flags["ablation_disable_unified_large_scale_plateau"]:
        for scale in prepared_scales:
            scale["large_scale_one_sided"] = False
            scale["large_scale_plateau"] = 0.0
            scale["large_scale_plateau_onset_sigma"] = 0.0
            scale["large_scale_plateau_full_sigma"] = 0.0
        log.append("disable_unified_large_scale_plateau")

    prepared_kwargs["unified_ablation_flags"] = flags
    prepared_kwargs["unified_ablation_log"] = list(log)
    return prepared_kwargs, prepared_scales, log


def prepare_trial_configuration(kwargs):
    prepared_kwargs = copy.deepcopy(kwargs)
    scales = compile_scales(
        prepared_kwargs.get("scale_names", []),
        grid_balance_modalities=prepared_kwargs.get("pcn_grid_balance_modalities", None),
    )
    return _apply_unified_ablation_overrides(prepared_kwargs, scales)


# Per-environment single-goal positions (match the Goal proto in each .wbt file).
WORLD_SINGLE_GOALS = {
    "environment_1": {"name": "goal", "location": [-9.0, -9.0], "radius": 0.8},
    "environment_2": {"name": "goal", "location": [ 9.0, -9.0], "radius": 0.8},
    "environment_3": {"name": "goal", "location": [ 9.0,  9.0], "radius": 0.8},
    "environment_4": {"name": "goal", "location": [-9.0,  9.0], "radius": 0.8},
    "environment_5": {"name": "goal", "location": [ 9.0, -9.0], "radius": 0.8},
    "environment_6": {"name": "goal", "location": [-9.0, -9.0], "radius": 0.8},
}

WORLD_MULTI_GOALS = {
    "20x20_multi_goal": {
        "explore_radius": 0.7,
        "exploit_radius": 1.0,
        "goals": [
            {"name": "red", "location": [7.0, 7.0]},
            {"name": "green", "location": [-7.0, 7.0]},
            {"name": "blue", "location": [7.0, -7.0]},
            {"name": "yellow", "location": [-7.0, -7.0]},
        ],
    },
    "20x20_cross_multi_goal": {
        "explore_radius": 0.7,
        "exploit_radius": 1.0,
        "goals": [
            {"name": "red", "location": [7.0, 7.0]},
            {"name": "green", "location": [-7.0, 7.0]},
            {"name": "blue", "location": [7.0, -7.0]},
            {"name": "yellow", "location": [-7.0, -7.0]},
        ],
    },
    "20x20_maze_multi_goal": {
        "explore_radius": 0.7,
        "exploit_radius": 1.0,
        "goals": [
            {"name": "red", "location": [7.0, 7.0]},
            {"name": "green", "location": [-7.0, 7.0]},
            {"name": "blue", "location": [7.0, -7.0]},
            {"name": "yellow", "location": [-7.0, -7.0]},
        ],
    },
}


def _build_goal_config_for_world(world_name, exploit=False):
    """Return a world-matched goal config for single- or multi-goal worlds."""
    multi_entry = WORLD_MULTI_GOALS.get(world_name)
    if multi_entry is not None:
        radius_key = "exploit_radius" if exploit else "explore_radius"
        radius = float(multi_entry[radius_key])
        goals = []
        for goal in multi_entry["goals"]:
            goals.append(
                {
                    "name": goal["name"],
                    "location": list(goal["location"]),
                    "radius": radius,
                }
            )
        return {"type": "multi", "goals": goals}

    entry = WORLD_SINGLE_GOALS.get(world_name)
    if entry is None:
        print(f"[WARNING] No goal config found for world '{world_name}', using origin fallback")
        entry = {"name": "goal", "location": [0.0, 0.0], "radius": 0.8}
    return {"type": "multi", "goals": [entry]}


def _build_single_goal_config(world_name):
    """Backward-compatible wrapper for callers that expect dynamic goal config."""
    return _build_goal_config_for_world(world_name, exploit=False)


#################################
# Trial Execution Functions
#################################

def _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs):
    """Run a single trial"""

    # Update goal config for this trial if target_goal specified
    trial_kwargs = kwargs.copy()
    if target_goal and "goal_config" in trial_kwargs:
        trial_kwargs["goal_config"]["target_goal"] = target_goal

    # Resolve effective configuration after ablation overrides.
    trial_kwargs, scales_list, ablation_log = prepare_trial_configuration(trial_kwargs)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

    world_name = get_world_name(bot)

    if ablation_log:
        print(f"[ABLATION] Applied: {', '.join(ablation_log)}")

    # Save trial parameters to JSON file in world folder
    save_trial_parameters(
        world_name=world_name,
        trial_id=trial_id,
        mode=mode,
        start_loc=start_loc,
        target_goal=target_goal,
        **trial_kwargs
    )

    # Initialize bot for this trial
    bot.initialization(
        mode=mode,
        run_time_hours=trial_kwargs.get("run_time_hours", 2),
        randomize_start_loc=False,  # We're setting specific start location
        start_loc=start_loc,
        enable_ojas=trial_kwargs.get("enable_ojas", None),
        enable_stdp=trial_kwargs.get("enable_stdp", None),
        scales=scales_list,
        rcn_learning_rates=rcn_learning_rates,
        stats_collector=stats_collector_instance,
        trial_id=trial_id,
        world_name=world_name,
        goal_config=trial_kwargs.get("goal_config"),
        trial_config=trial_kwargs.get("trial_config"),
        max_dist=trial_kwargs.get("max_dist", 25),
        plot_bvc=trial_kwargs.get("plot_bvc", False),
        td_learning=trial_kwargs.get("td_learning", False),
        use_prox_mod=trial_kwargs.get("use_prox_mod", False),
        use_unified_multiscale=trial_kwargs.get("use_unified_multiscale", False),
        environment_size=trial_kwargs.get("environment_size", None),
        grid_size=trial_kwargs.get("grid_size", None),
        coverage_percentage=trial_kwargs.get("coverage_percentage", None),
        minimum_learning_duration_hours=trial_kwargs.get("minimum_learning_duration_hours", 4.0),
        min_goal_visits=trial_kwargs.get("min_goal_visits", 3),
        min_checkpoint_visits=trial_kwargs.get("min_checkpoint_visits", 3),
        checkpoint_visit_radius=trial_kwargs.get("checkpoint_visit_radius", 0.8),
        goal_assoc_require_all_scales=trial_kwargs.get("goal_assoc_require_all_scales", False),
        goal_visit_cooldown_seconds=trial_kwargs.get("goal_visit_cooldown_seconds", 6.0),
        goal_exit_hysteresis=trial_kwargs.get("goal_exit_hysteresis", 0.1),
        proximity_mode=trial_kwargs.get("proximity_mode", "min"),
        proximity_trimmed_sigma=trial_kwargs.get("proximity_trimmed_sigma", 2.5),
        proximity_pair_percentile=trial_kwargs.get("proximity_pair_percentile", 25.0),
        pcn_gate_mode=trial_kwargs.get("pcn_gate_mode", "normal"),
        pcn_learning_adaptation_mode=trial_kwargs.get(
            "pcn_learning_adaptation_mode", "gaussian_post_competition_expression"
        ),
        pcn_learning_stdp_start_steps=trial_kwargs.get(
            "pcn_learning_stdp_start_steps", 8000
        ),
        pcn_learning_stop_ojas_on_stabilization=trial_kwargs.get(
            "pcn_learning_stop_ojas_on_stabilization", False
        ),
        pcn_learning_cross_scale_coupling_start_steps=trial_kwargs.get(
            "pcn_learning_cross_scale_coupling_start_steps", 8000
        ),
        pcn_learning_cross_scale_coupling_ramp_steps=trial_kwargs.get(
            "pcn_learning_cross_scale_coupling_ramp_steps", 12000
        ),
        pcn_learning_cross_scale_coupling_min=trial_kwargs.get(
            "pcn_learning_cross_scale_coupling_min", 0.0
        ),
        pcn_soft_scale_overlap=trial_kwargs.get("pcn_soft_scale_overlap", True),
        pcn_soft_scale_overlap_in_learning=trial_kwargs.get("pcn_soft_scale_overlap_in_learning", True),
        pcn_soft_scale_gate_floor=trial_kwargs.get("pcn_soft_scale_gate_floor", 0.20),
        pcn_soft_scale_gate_floor_in_learning=trial_kwargs.get(
            "pcn_soft_scale_gate_floor_in_learning",
            pcn_soft_scale_gate_floor_in_learning,
        ),
        pcn_soft_cross_inhibition_scale=trial_kwargs.get("pcn_soft_cross_inhibition_scale", 0.35),
        pcn_soft_cross_inhibition_scale_in_learning=trial_kwargs.get("pcn_soft_cross_inhibition_scale_in_learning", 0.25),
        pcn_soft_cross_inhibition_cap=trial_kwargs.get("pcn_soft_cross_inhibition_cap", 0.75),
        use_bvc_context_modulation=trial_kwargs.get("use_bvc_context_modulation", True),
        bvc_context_gain_floor=trial_kwargs.get("bvc_context_gain_floor", 0.0),
        bvc_context_gain_strength=trial_kwargs.get("bvc_context_gain_strength", 1.0),
        optimal_path_distance=trial_kwargs.get("optimal_path_distance", None),
        path_failure_ratio=trial_kwargs.get("path_failure_ratio", None),
        paths_folder=trial_kwargs.get("paths_folder", None),
        hmaps_folder=trial_kwargs.get("hmaps_folder", None),
        auto_trial_name=trial_kwargs.get("auto_trial_name", None),
        num_auto_trials=trial_kwargs.get("num_auto_trials", 5),
        current_auto_trial=trial_kwargs.get("current_auto_trial", 1),
        lightweight_hmaps=trial_kwargs.get("lightweight_hmaps", False),
        hmap_sample_stride=trial_kwargs.get("hmap_sample_stride", 10),
        hmap_topk=trial_kwargs.get("hmap_topk", 8),
        gcn_scale_invariant=trial_kwargs.get("gcn_scale_invariant", True),
        gcn_use_adaptive_minmax=trial_kwargs.get("gcn_use_adaptive_minmax", False),
        goal_assoc_unique_topk=trial_kwargs.get("goal_assoc_unique_topk", 16),
        goal_assoc_max_activation_drop=trial_kwargs.get("goal_assoc_max_activation_drop", 0.08),
        goal_assoc_locality_radius=trial_kwargs.get("goal_assoc_locality_radius", 2.0),
        goal_assoc_locality_max_activation_drop=trial_kwargs.get("goal_assoc_locality_max_activation_drop", 0.65),
        goal_map_use_compact_pc_gate=trial_kwargs.get("goal_map_use_compact_pc_gate", True),
        goal_map_pc_compactness_peak_fraction=trial_kwargs.get("goal_map_pc_compactness_peak_fraction", 0.50),
        goal_map_pc_compactness_min_peak=trial_kwargs.get("goal_map_pc_compactness_min_peak", 0.05),
        goal_map_pc_compactness_max_radius=trial_kwargs.get("goal_map_pc_compactness_max_radius", 4.0),
        goal_map_fill_source_envelope_tau=trial_kwargs.get("goal_map_fill_source_envelope_tau", 6.0),
        goal_map_fill_source_envelope_floor=trial_kwargs.get("goal_map_fill_source_envelope_floor", 0.10),
        record_experience_transitions=trial_kwargs.get("record_experience_transitions", True),
        two_phase_learning=trial_kwargs.get("two_phase_learning", False),
        phase1_min_steps=trial_kwargs.get("phase1_min_steps", 2500),
        phase1_bin_size=trial_kwargs.get("phase1_bin_size", 0.5),
        phase1_min_revisit_bins=trial_kwargs.get("phase1_min_revisit_bins", 15),
        phase1_revisit_cosine_threshold=trial_kwargs.get("phase1_revisit_cosine_threshold", 0.90),
        phase1_revisit_window=trial_kwargs.get("phase1_revisit_window", 200),
        defer_experience_build_until_phase2_end=trial_kwargs.get("defer_experience_build_until_phase2_end", True),
        goal_map_replay_timesteps=trial_kwargs.get(
            "goal_map_replay_timesteps", goal_map_replay_timesteps
        ),
        goal_map_paper_replay_tau=trial_kwargs.get(
            "goal_map_paper_replay_tau", goal_map_paper_replay_tau
        ),
        goal_map_goal_replay_event_count=trial_kwargs.get(
            "goal_map_goal_replay_event_count", goal_map_goal_replay_event_count
        ),
        goal_map_checkpoint_replay_event_count=trial_kwargs.get(
            "goal_map_checkpoint_replay_event_count",
            goal_map_checkpoint_replay_event_count,
        ),
        goal_map_replay_event_merge_gap_steps=trial_kwargs.get(
            "goal_map_replay_event_merge_gap_steps",
            goal_map_replay_event_merge_gap_steps,
        ),
        goal_map_checkpoint_room_support_min_fraction=trial_kwargs.get(
            "goal_map_checkpoint_room_support_min_fraction",
            goal_map_checkpoint_room_support_min_fraction,
        ),
        hybrid_path_replay_weight=trial_kwargs.get("hybrid_path_replay_weight", 0.8),
        hybrid_diffusion_replay_weight=trial_kwargs.get("hybrid_diffusion_replay_weight", 0.2),
        goal_map_path_replay_weight=trial_kwargs.get("goal_map_path_replay_weight", 1.0),
        goal_map_diffusion_replay_weight=trial_kwargs.get("goal_map_diffusion_replay_weight", 0.0),
        goal_map_experience_transition_topk=trial_kwargs.get("goal_map_experience_transition_topk", 16),
        unified_reward_normalization_mode=trial_kwargs.get("unified_reward_normalization_mode", "input_l1"),
        unified_goal_map_mode=trial_kwargs.get("unified_goal_map_mode", "paper_room_local_replay"),
        goal_map_room_normalization_mode=trial_kwargs.get(
            "goal_map_room_normalization_mode",
            goal_map_room_normalization_mode,
        ),
        goal_map_path_topk=trial_kwargs.get("goal_map_path_topk", 16),
        goal_map_path_decay=trial_kwargs.get("goal_map_path_decay", goal_map_path_decay),
        goal_map_path_frontier_only=trial_kwargs.get("goal_map_path_frontier_only", True),
        goal_map_neighbor_topk=trial_kwargs.get("goal_map_neighbor_topk", 12),
        goal_map_neighbor_steps=trial_kwargs.get(
            "goal_map_neighbor_steps", goal_map_neighbor_steps
        ),
        goal_map_neighbor_decay=trial_kwargs.get("goal_map_neighbor_decay", goal_map_neighbor_decay),
        goal_map_neighbor_frontier_only=trial_kwargs.get("goal_map_neighbor_frontier_only", True),
        goal_map_neighbor_seed_scale=trial_kwargs.get(
            "goal_map_neighbor_seed_scale", goal_map_neighbor_seed_scale
        ),
        goal_map_goal_seed_mode=trial_kwargs.get(
            "goal_map_goal_seed_mode", goal_map_goal_seed_mode
        ),
        goal_map_goal_replay_retention_mode=trial_kwargs.get(
            "goal_map_goal_replay_retention_mode",
            goal_map_goal_replay_retention_mode,
        ),
        goal_map_goal_seed_post_contact_steps=trial_kwargs.get(
            "goal_map_goal_seed_post_contact_steps",
            goal_map_goal_seed_post_contact_steps,
        ),
        goal_map_goal_seed_within_visit_aggregation=trial_kwargs.get(
            "goal_map_goal_seed_within_visit_aggregation",
            goal_map_goal_seed_within_visit_aggregation,
        ),
        goal_map_goal_seed_across_visit_aggregation=trial_kwargs.get(
            "goal_map_goal_seed_across_visit_aggregation",
            goal_map_goal_seed_across_visit_aggregation,
        ),
        goal_map_goal_seed_state_normalization=trial_kwargs.get(
            "goal_map_goal_seed_state_normalization",
            goal_map_goal_seed_state_normalization,
        ),
        goal_map_goal_seed_self_retention_alpha=trial_kwargs.get(
            "goal_map_goal_seed_self_retention_alpha",
            goal_map_goal_seed_self_retention_alpha,
        ),
        goal_map_path_constraint_sigma=trial_kwargs.get(
            "goal_map_path_constraint_sigma", goal_map_path_constraint_sigma
        ),
        goal_map_path_constraint_topk=trial_kwargs.get(
            "goal_map_path_constraint_topk", goal_map_path_constraint_topk
        ),
        goal_map_local_constraint_sigma=trial_kwargs.get("goal_map_local_constraint_sigma", goal_map_local_constraint_sigma),
        goal_map_local_constraint_topk=trial_kwargs.get("goal_map_local_constraint_topk", goal_map_local_constraint_topk),
        goal_map_goal_seed_center_radius_factor=trial_kwargs.get(
            "goal_map_goal_seed_center_radius_factor",
            goal_map_goal_seed_center_radius_factor,
        ),
        goal_map_goal_seed_center_min_radius=trial_kwargs.get(
            "goal_map_goal_seed_center_min_radius",
            goal_map_goal_seed_center_min_radius,
        ),
        goal_map_checkpoint_seed_center_radius_factor=trial_kwargs.get(
            "goal_map_checkpoint_seed_center_radius_factor",
            goal_map_checkpoint_seed_center_radius_factor,
        ),
        goal_map_checkpoint_seed_center_min_radius=trial_kwargs.get(
            "goal_map_checkpoint_seed_center_min_radius",
            goal_map_checkpoint_seed_center_min_radius,
        ),
        goal_map_spatial_obstacle_block=trial_kwargs.get("goal_map_spatial_obstacle_block", True),
        goal_map_obstacle_margin=trial_kwargs.get("goal_map_obstacle_margin", 0.05),
        goal_map_use_checkpoint_relays=trial_kwargs.get(
            "goal_map_use_checkpoint_relays", goal_map_use_checkpoint_relays
        ),
        goal_map_checkpoint_mode=trial_kwargs.get(
            "goal_map_checkpoint_mode", goal_map_checkpoint_mode
        ),
        goal_map_checkpoint_scale=trial_kwargs.get(
            "goal_map_checkpoint_scale", goal_map_checkpoint_scale
        ),
        goal_map_checkpoint_support_mode=trial_kwargs.get(
            "goal_map_checkpoint_support_mode", goal_map_checkpoint_support_mode
        ),
        goal_map_checkpoint_stop_parent_replay=trial_kwargs.get(
            "goal_map_checkpoint_stop_parent_replay", goal_map_checkpoint_stop_parent_replay
        ),
        checkpoint_boost_threshold=trial_kwargs.get(
            "checkpoint_boost_threshold", checkpoint_boost_threshold
        ),
        checkpoint_target_arrival=trial_kwargs.get(
            "checkpoint_target_arrival", checkpoint_target_arrival
        ),
        checkpoint_min_relay_weight=trial_kwargs.get(
            "checkpoint_min_relay_weight", checkpoint_min_relay_weight
        ),
        goal_map_route_selection_support_fraction=trial_kwargs.get(
            "goal_map_route_selection_support_fraction",
            goal_map_route_selection_support_fraction,
        ),
        goal_map_route_selection_overlap_threshold=trial_kwargs.get(
            "goal_map_route_selection_overlap_threshold",
            goal_map_route_selection_overlap_threshold,
        ),
        goal_map_use_denoised_experience_graph=trial_kwargs.get(
            "goal_map_use_denoised_experience_graph", goal_map_use_denoised_experience_graph
        ),
        goal_map_use_segmented_checkpoint_histories=trial_kwargs.get(
            "goal_map_use_segmented_checkpoint_histories",
            goal_map_use_segmented_checkpoint_histories,
        ),
        unified_preplay_scale_arbitration=trial_kwargs.get("unified_preplay_scale_arbitration", True),
        unified_preplay_scale_prior_mix=trial_kwargs.get("unified_preplay_scale_prior_mix", 0.0),
        goal_map_smoothing_sigma=trial_kwargs.get("goal_map_smoothing_sigma", 1.0),
        goal_map_smoothing_topk=trial_kwargs.get("goal_map_smoothing_topk", 24),
        goal_map_smoothing_passes=trial_kwargs.get("goal_map_smoothing_passes", 0),
        goal_map_smoothing_mix=trial_kwargs.get("goal_map_smoothing_mix", 0.45),
        unified_preplay_commit_threshold=trial_kwargs.get("unified_preplay_commit_threshold", 0.30),
        preplay_heading_selection_mode=trial_kwargs.get("preplay_heading_selection_mode", "trajectory_vector"),
        unified_exploit_action_mode=trial_kwargs.get("unified_exploit_action_mode", "local_reward_gradient_primary"),
        unified_transition_preplay_enabled=trial_kwargs.get("unified_transition_preplay_enabled", False),
        unified_spatial_fallback_enabled=trial_kwargs.get("unified_spatial_fallback_enabled", True),
        unified_spatial_override_margin=trial_kwargs.get("unified_spatial_override_margin", 0.0),
        unified_spatial_lookahead_distance=trial_kwargs.get("unified_spatial_lookahead_distance", 1.0),
        unified_spatial_lookahead_sigma=trial_kwargs.get("unified_spatial_lookahead_sigma", 0.9),
        unified_exploit_step_fraction=trial_kwargs.get("unified_exploit_step_fraction", 0.25),
        exploit_direction_clearance_percentile=trial_kwargs.get("exploit_direction_clearance_percentile", 20.0),
        prune_experience_loops=trial_kwargs.get("prune_experience_loops", True),
        loop_prune_min_top1=trial_kwargs.get("loop_prune_min_top1", 0.08),
        loop_prune_min_top1_to_top2_ratio=trial_kwargs.get("loop_prune_min_top1_to_top2_ratio", 1.15),
    )

    bot.trial_id = trial_id

    # Run the trial
    bot.run()


def _run_simple_trials(mode, trial_config, **kwargs):
    """Handle simple trial execution"""
    # Use start_locations from trial_config if available, otherwise fall back to start_loc from kwargs
    if "start_locations" in trial_config:
        start_locations = trial_config["start_locations"]
    else:
        start_loc = kwargs.get("start_loc", [0, 0])
        start_locations = [start_loc]

    trials_per_start = trial_config["count"]
    save_data = kwargs.get("save_data", False)

    # Remove start_loc from kwargs to avoid conflicts when passing to _run_single_trial
    trial_kwargs = {k: v for k, v in kwargs.items() if k != "start_loc"}

    bot = Driver()
    world_name = get_world_name(bot)
    if trial_kwargs.get("goal_config") is None:
        trial_kwargs["goal_config"] = _build_goal_config_for_world(world_name, exploit=(mode == RobotMode.EXPLOIT))

    # Setup stats if needed
    if save_data and mode == RobotMode.EXPLOIT:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)
        stats_collector_instance = stats_collector(output_dir=stats_folder)
    else:
        stats_collector_instance = None

    for start_loc in start_locations:
        for trial_num in range(1, trials_per_start + 1):

            # Get existing trial count
            if save_data:
                current_trial_id = get_highest_trial_id(stats_folder, start_loc)
                trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}"
            else:
                trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}"

            print(f"[INFO] Running trial: {trial_id}")

            # Run single trial
            _run_single_trial(bot, mode, trial_id, start_loc, None, stats_collector_instance, **trial_kwargs)


def _run_combination_trials(mode, trial_config, **kwargs):
    """Handle combination trial execution (start locations x target goals)"""
    # Use start_locations from trial_config if available, otherwise fall back to start_loc from kwargs
    if "start_locations" in trial_config:
        start_locations = trial_config["start_locations"]
    else:
        start_loc = kwargs.get("start_loc", [0, 0])
        start_locations = [start_loc]

    target_goals = trial_config.get("target_goals", [None])
    trials_per_combo = trial_config["count"]
    save_data = kwargs.get("save_data", False)

    # Remove start_loc from kwargs to avoid conflicts when passing to _run_single_trial
    trial_kwargs = {k: v for k, v in kwargs.items() if k != "start_loc"}

    bot = Driver()
    world_name = get_world_name(bot)
    if trial_kwargs.get("goal_config") is None:
        trial_kwargs["goal_config"] = _build_goal_config_for_world(
            world_name,
            exploit=(mode == RobotMode.EXPLOIT),
        )

    # Setup stats if needed
    if save_data:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)
        stats_collector_instance = stats_collector(output_dir=stats_folder)
    else:
        stats_collector_instance = None

    for start_loc in start_locations:
        for target_goal in target_goals:
            # Skip if goal and start are the same (for goal-specific trials)
            if target_goal:
                goal_config = kwargs.get("goal_config", {})
                goals = goal_config.get("goals", [])
                goal_location = next((g["location"] for g in goals if g["name"] == target_goal), None)
                if goal_location and goal_location == start_loc:
                    print(f"[INFO] Skipping trial - start location {start_loc} equals goal {target_goal} location")
                    continue

            for trial_num in range(1, trials_per_combo + 1):
                # Get existing trial count
                if save_data:
                    current_trial_id = get_highest_trial_id(stats_folder, start_loc)
                    if target_goal:
                        trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}_goal_{target_goal}"
                    else:
                        trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}"
                else:
                    if target_goal:
                        trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}_goal_{target_goal}"
                    else:
                        trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}"

                print(f"[INFO] Running trial: {trial_id}")

                # Run single trial
                _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **trial_kwargs)


def _run_random_spawn_trials(mode, trial_config, **kwargs):
    """Handle random spawn trial execution without precomputed path planning."""
    trials_per_goal = trial_config["trials_per_goal"]
    save_data = kwargs.get("save_data", False)

    # Get path planning parameters
    min_spawn_distance = kwargs.get("min_spawn_distance", 6.0)
    wall_clearance = kwargs.get("wall_clearance", 0.5)
    path_failure_ratio = kwargs.get("path_failure_ratio", 10.0)

    bot = Driver()
    world_name = get_world_name(bot)
    if kwargs.get("goal_config") is None:
        kwargs["goal_config"] = _build_goal_config_for_world(
            world_name,
            exploit=(mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM),
        )

    # Get goals from goal config
    goal_config = kwargs["goal_config"]
    goals = goal_config.get("goals", [])

    print(f"[RANDOM_SPAWN] Generating spawn locations for {len(goals)} goals")

    # Generate spawn locations using path planning module
    spawn_locations = generate_spawn_locations(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance
    )

    # Create spawn/goal trial combinations
    combinations = []
    for goal in goals:
        goal_name = goal["name"]
        goal_spawns = spawn_locations.get(goal_name, [])
        for i, spawn_pos in enumerate(goal_spawns):
            combinations.append({
                "start": spawn_pos,
                "goal": goal,
                "trial_number": i + 1,
                "goal_name": goal_name
            })

    # Setup stats collection
    if save_data:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        # Use stats_random instead of stats
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)

        # Create hmaps folder
        hmaps_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "hmaps"
        )
        os.makedirs(hmaps_folder, exist_ok=True)

        stats_collector_instance = stats_collector(output_dir=stats_folder)
    else:
        stats_collector_instance = None

    print(f"[RANDOM_SPAWN] Running {len(combinations)} trials")

    # Execute trials using single-trial pattern
    successful_trials = 0
    for i, combination in enumerate(combinations):
        start_pos = combination["start"]
        goal_name = combination["goal_name"]
        trial_number = combination["trial_number"]

        # Create trial ID
        trial_id = f"trial_{trial_number}_goal_{goal_name}_random"

        print(f"[RANDOM_SPAWN] Running {trial_id}: Start {start_pos} -> Goal {goal_name} ({successful_trials + 1}/{len(combinations)})")

        # Prepare trial-specific kwargs
        trial_kwargs = kwargs.copy()
        trial_kwargs["goal_config"] = {
            "type": "multi",
            "goals": goals,
            "target_goal": goal_name  # Set target goal for this trial
        }
        trial_kwargs["optimal_path_distance"] = None
        trial_kwargs["path_failure_ratio"] = path_failure_ratio
        trial_kwargs["paths_folder"] = None
        trial_kwargs["hmaps_folder"] = hmaps_folder if save_data else None

        # Run single trial (fresh driver instance for each trial)
        _run_single_trial(bot, mode, trial_id, start_pos, goal_name, stats_collector_instance, **trial_kwargs)
        successful_trials += 1

    print(f"[RANDOM_SPAWN] Completed {successful_trials} trials successfully")


def _check_trial_exists(auto_trial_name, world_name, trial_num):
    """
    Check if a learning trial folder already exists and has completion data.
    Returns True if the trial is complete, False otherwise.
    """
    base_folder = f"pkl_{auto_trial_name}"
    trial_folder = f"{world_name}_{trial_num}"
    trial_path = os.path.join(base_folder, trial_folder)

    if not os.path.exists(trial_path):
        return False

    # Check if completion time JSON exists (indicates completed trial)
    completion_file = os.path.join(trial_path, "trial_completion_time.json")
    return os.path.exists(completion_file)


def _check_exploit_trial_complete(auto_trial_name, world_name, trial_num, scale_name_str):
    """
    Check if an exploitation trial is already complete.
    Returns True if the trial has stats data, False otherwise.
    """
    stats_base = f"stats_{auto_trial_name}"
    world_trial_folder = f"{world_name}_{trial_num}"
    stats_folder = os.path.join(
        PROJECT_ROOT, "analysis", stats_base, world_trial_folder, scale_name_str, "JSON"
    )

    if not os.path.exists(stats_folder):
        return False

    # Check if there are any JSON files (indicates completed trials)
    json_files = [f for f in os.listdir(stats_folder) if f.endswith('.json')]
    return len(json_files) > 0


def _check_plotting_trial_complete(auto_trial_name, world_name, trial_num, mode_suffix=""):
    """
    Check if a plotting trial is already complete.
    Returns True if the trial folder exists (plotting doesn't save data, just runs), False otherwise.
    Since plotting mode doesn't save data, we create a marker file to indicate completion.

    Args:
        auto_trial_name: Name of the auto trial set
        world_name: Name of the world
        trial_num: Trial number
        mode_suffix: Optional suffix to differentiate between plotting modes (e.g., "_coverage")
    """
    base_folder = f"pkl_{auto_trial_name}"
    trial_folder = f"{world_name}_{trial_num}"
    trial_path = os.path.join(base_folder, trial_folder)

    if not os.path.exists(trial_path):
        return False

    # Check for plotting completion marker file with optional suffix
    marker_filename = f"plotting_complete{mode_suffix}.txt"
    plotting_marker = os.path.join(trial_path, marker_filename)
    return os.path.exists(plotting_marker)


def _run_learn_coverage_auto_trials(mode, **kwargs):
    """
    Handle automated multi-trial learning with coverage tracking.
    Each trial starts from scratch with fresh networks and saves to separate folders.
    Skips trials that already have completion data.

    Uses the same pattern as _run_random_spawn_trials: single Driver instance,
    multiple initialization() + run() cycles without worldReload().
    """
    # Extract auto trial parameters
    auto_trial_name = kwargs.get("auto_trial_name", "default_trial")
    num_auto_trials = kwargs.get("num_auto_trials", 5)

    print(f"[AUTO_TRIAL] Starting automated learning trials")
    print(f"[AUTO_TRIAL] Trial name: {auto_trial_name}")
    print(f"[AUTO_TRIAL] Number of trials: {num_auto_trials}")

    # Create single Driver instance that will be reused for all trials
    bot = Driver()
    world_name = get_world_name(bot)
    if kwargs.get("goal_config") is None:
        kwargs["goal_config"] = _build_goal_config_for_world(world_name, exploit=False)

    # Check for existing trials and find starting point
    first_incomplete_trial = None
    for trial_num in range(1, num_auto_trials + 1):
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            print(f"[AUTO_TRIAL] Trial {trial_num} already exists, skipping...")
        else:
            if first_incomplete_trial is None:
                first_incomplete_trial = trial_num

    if first_incomplete_trial is None:
        print(f"[AUTO_TRIAL] All {num_auto_trials} trials already complete!")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_TRIAL] Starting from trial {first_incomplete_trial}")

    # Get scale info for RCN setup (same for all trials)
    prepared_kwargs, scales_list, ablation_log = prepare_trial_configuration(kwargs)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]
    if ablation_log:
        print(f"[ABLATION] Applied: {', '.join(ablation_log)}")

    for trial_num in range(first_incomplete_trial, num_auto_trials + 1):
        # Double-check in case trial was completed between checks
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            print(f"[AUTO_TRIAL] Trial {trial_num} already complete, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"[AUTO_TRIAL] Starting trial {trial_num}/{num_auto_trials}")
        print(f"{'='*60}\n")

        # Save trial parameters to JSON file
        trial_id = f"auto_trial_{trial_num}"
        save_trial_parameters(
            world_name=world_name,
            trial_id=trial_id,
            mode=mode,
            **prepared_kwargs
        )

        # Initialize bot for this trial (creates fresh networks, resets robot position)
        bot.initialization(
            mode=mode,
            run_time_hours=prepared_kwargs.get("run_time_hours", 2),
            randomize_start_loc=prepared_kwargs.get("randomize_start_loc", False),
            start_loc=prepared_kwargs.get("start_loc", [0, 0]),
            enable_ojas=prepared_kwargs.get("enable_ojas", True),
            enable_stdp=prepared_kwargs.get("enable_stdp", True),
            scales=scales_list,
            rcn_learning_rates=rcn_learning_rates,
            stats_collector=None,  # No stats collector for learning
            trial_id=f"auto_trial_{trial_num}",
            world_name=world_name,
            goal_config=prepared_kwargs.get("goal_config"),
            trial_config=prepared_kwargs.get("trial_config"),
            max_dist=prepared_kwargs.get("max_dist", 25),
            plot_bvc=prepared_kwargs.get("plot_bvc", False),
            td_learning=prepared_kwargs.get("td_learning", False),
            use_prox_mod=prepared_kwargs.get("use_prox_mod", False),
            use_unified_multiscale=prepared_kwargs.get("use_unified_multiscale", False),
            environment_size=prepared_kwargs.get("environment_size", None),
            grid_size=prepared_kwargs.get("grid_size", None),
            coverage_percentage=prepared_kwargs.get("coverage_percentage", None),
            minimum_learning_duration_hours=prepared_kwargs.get("minimum_learning_duration_hours", 4.0),
            min_goal_visits=prepared_kwargs.get("min_goal_visits", 3),
            min_checkpoint_visits=prepared_kwargs.get("min_checkpoint_visits", 3),
            checkpoint_visit_radius=prepared_kwargs.get("checkpoint_visit_radius", 0.8),
            goal_assoc_require_all_scales=prepared_kwargs.get("goal_assoc_require_all_scales", False),
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
            pcn_learning_adaptation_mode=prepared_kwargs.get(
                "pcn_learning_adaptation_mode", "gaussian_post_competition_expression"
            ),
            pcn_learning_stdp_start_steps=prepared_kwargs.get(
                "pcn_learning_stdp_start_steps", 8000
            ),
            pcn_learning_stop_ojas_on_stabilization=prepared_kwargs.get(
                "pcn_learning_stop_ojas_on_stabilization", False
            ),
            pcn_learning_cross_scale_coupling_start_steps=prepared_kwargs.get(
                "pcn_learning_cross_scale_coupling_start_steps", 8000
            ),
            pcn_learning_cross_scale_coupling_ramp_steps=prepared_kwargs.get(
                "pcn_learning_cross_scale_coupling_ramp_steps", 12000
            ),
            pcn_learning_cross_scale_coupling_min=prepared_kwargs.get(
                "pcn_learning_cross_scale_coupling_min", 0.0
            ),
            pcn_soft_scale_overlap=prepared_kwargs.get("pcn_soft_scale_overlap", True),
            pcn_soft_scale_overlap_in_learning=prepared_kwargs.get("pcn_soft_scale_overlap_in_learning", True),
            pcn_soft_scale_gate_floor=prepared_kwargs.get("pcn_soft_scale_gate_floor", 0.20),
            pcn_soft_scale_gate_floor_in_learning=prepared_kwargs.get(
                "pcn_soft_scale_gate_floor_in_learning",
                pcn_soft_scale_gate_floor_in_learning,
            ),
            pcn_soft_cross_inhibition_scale=prepared_kwargs.get("pcn_soft_cross_inhibition_scale", 0.35),
            pcn_soft_cross_inhibition_scale_in_learning=prepared_kwargs.get("pcn_soft_cross_inhibition_scale_in_learning", 0.25),
            pcn_soft_cross_inhibition_cap=prepared_kwargs.get("pcn_soft_cross_inhibition_cap", 0.75),
            gcn_scale_invariant=prepared_kwargs.get("gcn_scale_invariant", True),
            gcn_use_adaptive_minmax=prepared_kwargs.get("gcn_use_adaptive_minmax", False),
            goal_assoc_unique_topk=prepared_kwargs.get("goal_assoc_unique_topk", 16),
            goal_assoc_max_activation_drop=prepared_kwargs.get("goal_assoc_max_activation_drop", 0.08),
            goal_assoc_locality_radius=prepared_kwargs.get("goal_assoc_locality_radius", 2.0),
            goal_assoc_locality_max_activation_drop=prepared_kwargs.get("goal_assoc_locality_max_activation_drop", 0.65),
            goal_map_use_compact_pc_gate=prepared_kwargs.get("goal_map_use_compact_pc_gate", True),
            goal_map_pc_compactness_peak_fraction=prepared_kwargs.get("goal_map_pc_compactness_peak_fraction", 0.50),
            goal_map_pc_compactness_min_peak=prepared_kwargs.get("goal_map_pc_compactness_min_peak", 0.05),
            goal_map_pc_compactness_max_radius=prepared_kwargs.get("goal_map_pc_compactness_max_radius", 4.0),
            goal_map_fill_source_envelope_tau=prepared_kwargs.get("goal_map_fill_source_envelope_tau", 6.0),
            goal_map_fill_source_envelope_floor=prepared_kwargs.get("goal_map_fill_source_envelope_floor", 0.10),
            record_experience_transitions=prepared_kwargs.get("record_experience_transitions", True),
            two_phase_learning=prepared_kwargs.get("two_phase_learning", False),
            phase1_min_steps=prepared_kwargs.get("phase1_min_steps", 2500),
            defer_experience_build_until_phase2_end=prepared_kwargs.get("defer_experience_build_until_phase2_end", True),
            goal_map_replay_timesteps=prepared_kwargs.get(
                "goal_map_replay_timesteps", goal_map_replay_timesteps
            ),
            goal_map_paper_replay_tau=prepared_kwargs.get(
                "goal_map_paper_replay_tau", goal_map_paper_replay_tau
            ),
            goal_map_goal_replay_event_count=prepared_kwargs.get(
                "goal_map_goal_replay_event_count", goal_map_goal_replay_event_count
            ),
            goal_map_checkpoint_replay_event_count=prepared_kwargs.get(
                "goal_map_checkpoint_replay_event_count",
                goal_map_checkpoint_replay_event_count,
            ),
            goal_map_replay_event_merge_gap_steps=prepared_kwargs.get(
                "goal_map_replay_event_merge_gap_steps",
                goal_map_replay_event_merge_gap_steps,
            ),
            goal_map_checkpoint_room_support_min_fraction=prepared_kwargs.get(
                "goal_map_checkpoint_room_support_min_fraction",
                goal_map_checkpoint_room_support_min_fraction,
            ),
            hybrid_path_replay_weight=prepared_kwargs.get("hybrid_path_replay_weight", 0.8),
            hybrid_diffusion_replay_weight=prepared_kwargs.get("hybrid_diffusion_replay_weight", 0.2),
            goal_map_path_replay_weight=prepared_kwargs.get("goal_map_path_replay_weight", 1.0),
            goal_map_diffusion_replay_weight=prepared_kwargs.get("goal_map_diffusion_replay_weight", 0.0),
            goal_map_experience_transition_topk=prepared_kwargs.get("goal_map_experience_transition_topk", 16),
            unified_reward_normalization_mode=prepared_kwargs.get("unified_reward_normalization_mode", "input_l1"),
            unified_goal_map_mode=prepared_kwargs.get("unified_goal_map_mode", "paper_room_local_replay"),
            goal_map_room_normalization_mode=prepared_kwargs.get(
                "goal_map_room_normalization_mode",
                goal_map_room_normalization_mode,
            ),
            goal_map_path_topk=prepared_kwargs.get("goal_map_path_topk", 16),
            goal_map_path_decay=prepared_kwargs.get("goal_map_path_decay", goal_map_path_decay),
            goal_map_path_frontier_only=prepared_kwargs.get("goal_map_path_frontier_only", True),
            goal_map_neighbor_topk=prepared_kwargs.get("goal_map_neighbor_topk", 12),
            goal_map_neighbor_steps=prepared_kwargs.get(
                "goal_map_neighbor_steps", goal_map_neighbor_steps
            ),
            goal_map_neighbor_decay=prepared_kwargs.get("goal_map_neighbor_decay", goal_map_neighbor_decay),
            goal_map_neighbor_frontier_only=prepared_kwargs.get("goal_map_neighbor_frontier_only", True),
            goal_map_neighbor_seed_scale=prepared_kwargs.get(
                "goal_map_neighbor_seed_scale", goal_map_neighbor_seed_scale
            ),
            goal_map_goal_seed_mode=prepared_kwargs.get(
                "goal_map_goal_seed_mode", goal_map_goal_seed_mode
            ),
            goal_map_goal_replay_retention_mode=prepared_kwargs.get(
                "goal_map_goal_replay_retention_mode",
                goal_map_goal_replay_retention_mode,
            ),
            goal_map_goal_seed_post_contact_steps=prepared_kwargs.get(
                "goal_map_goal_seed_post_contact_steps",
                goal_map_goal_seed_post_contact_steps,
            ),
            goal_map_goal_seed_within_visit_aggregation=prepared_kwargs.get(
                "goal_map_goal_seed_within_visit_aggregation",
                goal_map_goal_seed_within_visit_aggregation,
            ),
            goal_map_goal_seed_across_visit_aggregation=prepared_kwargs.get(
                "goal_map_goal_seed_across_visit_aggregation",
                goal_map_goal_seed_across_visit_aggregation,
            ),
            goal_map_goal_seed_state_normalization=prepared_kwargs.get(
                "goal_map_goal_seed_state_normalization",
                goal_map_goal_seed_state_normalization,
            ),
            goal_map_goal_seed_self_retention_alpha=prepared_kwargs.get(
                "goal_map_goal_seed_self_retention_alpha",
                goal_map_goal_seed_self_retention_alpha,
            ),
            goal_map_path_constraint_sigma=prepared_kwargs.get(
                "goal_map_path_constraint_sigma", goal_map_path_constraint_sigma
            ),
            goal_map_path_constraint_topk=prepared_kwargs.get(
                "goal_map_path_constraint_topk", goal_map_path_constraint_topk
            ),
            goal_map_local_constraint_sigma=prepared_kwargs.get("goal_map_local_constraint_sigma", goal_map_local_constraint_sigma),
            goal_map_local_constraint_topk=prepared_kwargs.get("goal_map_local_constraint_topk", goal_map_local_constraint_topk),
            goal_map_goal_seed_center_radius_factor=prepared_kwargs.get(
                "goal_map_goal_seed_center_radius_factor",
                goal_map_goal_seed_center_radius_factor,
            ),
            goal_map_goal_seed_center_min_radius=prepared_kwargs.get(
                "goal_map_goal_seed_center_min_radius",
                goal_map_goal_seed_center_min_radius,
            ),
            goal_map_checkpoint_seed_center_radius_factor=prepared_kwargs.get(
                "goal_map_checkpoint_seed_center_radius_factor",
                goal_map_checkpoint_seed_center_radius_factor,
            ),
            goal_map_checkpoint_seed_center_min_radius=prepared_kwargs.get(
                "goal_map_checkpoint_seed_center_min_radius",
                goal_map_checkpoint_seed_center_min_radius,
            ),
            goal_map_spatial_obstacle_block=prepared_kwargs.get("goal_map_spatial_obstacle_block", True),
            goal_map_obstacle_margin=prepared_kwargs.get("goal_map_obstacle_margin", 0.05),
            goal_map_use_checkpoint_relays=kwargs.get(
                "goal_map_use_checkpoint_relays", goal_map_use_checkpoint_relays
            ),
            goal_map_checkpoint_mode=kwargs.get(
                "goal_map_checkpoint_mode", goal_map_checkpoint_mode
            ),
            goal_map_checkpoint_scale=kwargs.get(
                "goal_map_checkpoint_scale", goal_map_checkpoint_scale
            ),
            goal_map_checkpoint_support_mode=kwargs.get(
                "goal_map_checkpoint_support_mode", goal_map_checkpoint_support_mode
            ),
            goal_map_checkpoint_stop_parent_replay=kwargs.get(
                "goal_map_checkpoint_stop_parent_replay", goal_map_checkpoint_stop_parent_replay
            ),
            checkpoint_boost_threshold=kwargs.get(
                "checkpoint_boost_threshold", checkpoint_boost_threshold
            ),
            checkpoint_target_arrival=kwargs.get(
                "checkpoint_target_arrival", checkpoint_target_arrival
            ),
            checkpoint_min_relay_weight=kwargs.get(
                "checkpoint_min_relay_weight", checkpoint_min_relay_weight
            ),
            goal_map_route_selection_support_fraction=kwargs.get(
                "goal_map_route_selection_support_fraction",
                goal_map_route_selection_support_fraction,
            ),
            goal_map_route_selection_overlap_threshold=kwargs.get(
                "goal_map_route_selection_overlap_threshold",
                goal_map_route_selection_overlap_threshold,
            ),
            goal_map_use_denoised_experience_graph=kwargs.get(
                "goal_map_use_denoised_experience_graph", goal_map_use_denoised_experience_graph
            ),
            goal_map_use_segmented_checkpoint_histories=kwargs.get(
                "goal_map_use_segmented_checkpoint_histories",
                goal_map_use_segmented_checkpoint_histories,
            ),
            unified_preplay_scale_arbitration=kwargs.get("unified_preplay_scale_arbitration", True),
            unified_preplay_scale_prior_mix=kwargs.get("unified_preplay_scale_prior_mix", 0.0),
            goal_map_smoothing_sigma=kwargs.get("goal_map_smoothing_sigma", 1.0),
            goal_map_smoothing_topk=kwargs.get("goal_map_smoothing_topk", 24),
            exploit_direction_clearance_percentile=kwargs.get("exploit_direction_clearance_percentile", 20.0),
            goal_map_smoothing_passes=kwargs.get("goal_map_smoothing_passes", 0),
            goal_map_smoothing_mix=kwargs.get("goal_map_smoothing_mix", 0.45),
            unified_preplay_commit_threshold=kwargs.get("unified_preplay_commit_threshold", 0.30),
            preplay_heading_selection_mode=kwargs.get("preplay_heading_selection_mode", "trajectory_vector"),
            unified_exploit_action_mode=kwargs.get("unified_exploit_action_mode", "local_reward_gradient_primary"),
            unified_transition_preplay_enabled=kwargs.get("unified_transition_preplay_enabled", False),
            unified_spatial_fallback_enabled=kwargs.get("unified_spatial_fallback_enabled", True),
            unified_spatial_override_margin=kwargs.get("unified_spatial_override_margin", 0.0),
            unified_spatial_lookahead_distance=kwargs.get("unified_spatial_lookahead_distance", 1.0),
            unified_spatial_lookahead_sigma=kwargs.get("unified_spatial_lookahead_sigma", 0.9),
            unified_exploit_step_fraction=kwargs.get("unified_exploit_step_fraction", 0.25),
            prune_experience_loops=kwargs.get("prune_experience_loops", True),
            loop_prune_min_top1=kwargs.get("loop_prune_min_top1", 0.08),
            loop_prune_min_top1_to_top2_ratio=kwargs.get("loop_prune_min_top1_to_top2_ratio", 1.15),
        )

        # Run the trial (initialization() already reset self.done = False)
        print(f"[AUTO_TRIAL] Running trial {trial_num}...")
        bot.run()
        print(f"[AUTO_TRIAL] Trial {trial_num} complete!")

        # Print learned checkpoint route edges
        cp_cp = getattr(bot, "checkpoint_to_checkpoint_edges", {})
        cp_goal = getattr(bot, "checkpoint_to_goal_edges", {})
        if cp_cp or cp_goal:
            print(f"\n[ROUTE_EDGES] Learned checkpoint connectivity (trial {trial_num}):")
            print(f"  Checkpoint-to-checkpoint edges ({len(cp_cp)}):")
            for (src, src_dir, dst, dst_dir), count in sorted(
                cp_cp.items(), key=lambda x: -x[1]
            ):
                print(f"    cp{src} ({src_dir}) -> cp{dst} ({dst_dir}): {count}x")
            print(f"  Checkpoint-to-goal edges ({len(cp_goal)}):")
            for (src, src_dir, goal_name), count in sorted(
                cp_goal.items(), key=lambda x: -x[1]
            ):
                print(f"    cp{src} ({src_dir}) -> {goal_name}: {count}x")

    print(f"\n[AUTO_TRIAL] All {num_auto_trials} trials complete!")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


def _run_exploit_random_auto_trials(mode, **kwargs):
    """
    Handle automated multi-trial exploitation with random spawns.
    Loads networks from each LEARN_LOCATIONS_COVERAGE_AUTO trial folder and runs
    full exploitation trials, saving results to stats_{auto_trial_name}.

    Uses the same pattern as _run_learn_coverage_auto_trials: single Driver instance,
    multiple initialization() + run() cycles.
    """
    # Extract auto trial parameters
    auto_trial_name = kwargs.get("auto_trial_name", "default_trial")
    num_auto_trials = kwargs.get("num_auto_trials", 5)

    print(f"[AUTO_EXPLOIT] Starting automated exploitation trials")
    print(f"[AUTO_EXPLOIT] Trial name: {auto_trial_name}")
    print(f"[AUTO_EXPLOIT] Number of learning trials to exploit: {num_auto_trials}")

    # Create single Driver instance that will be reused for all trials
    bot = Driver()
    world_name = get_world_name(bot)
    if kwargs.get("goal_config") is None:
        kwargs["goal_config"] = _build_goal_config_for_world(world_name, exploit=True)

    # Check which learning trials exist
    available_trials = []
    for trial_num in range(1, num_auto_trials + 1):
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            available_trials.append(trial_num)
        else:
            print(f"[AUTO_EXPLOIT] Learning trial {trial_num} not found, skipping...")

    if not available_trials:
        print(f"[AUTO_EXPLOIT] No learning trials found! Run LEARN_LOCATIONS_COVERAGE_AUTO first.")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_EXPLOIT] Found {len(available_trials)} learning trials to exploit: {available_trials}")

    # Get exploitation parameters
    trials_per_goal = kwargs.get("trials_per_goal", 20)
    min_spawn_distance = kwargs.get("min_spawn_distance", 6.0)
    wall_clearance = kwargs.get("wall_clearance", 0.5)
    path_failure_ratio = kwargs.get("path_failure_ratio", 15.0)

    prepared_kwargs, scales_list, ablation_log = prepare_trial_configuration(kwargs)
    if ablation_log:
        print(f"[ABLATION] Applied: {', '.join(ablation_log)}")

    # Get goal config
    goal_config = prepared_kwargs.get("goal_config", {})
    goals = goal_config.get("goals", [])

    # Get scale info
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]
    scale_name_str = "_".join(prepared_kwargs.get("scale_names", []))

    # Check which exploitation trials are already complete and can be skipped
    trials_to_run = []
    for trial_num in available_trials:
        if _check_exploit_trial_complete(auto_trial_name, world_name, trial_num, scale_name_str):
            print(f"[AUTO_EXPLOIT] Exploitation trial {trial_num} already complete, skipping...")
        else:
            trials_to_run.append(trial_num)

    if not trials_to_run:
        print(f"[AUTO_EXPLOIT] All exploitation trials already complete!")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_EXPLOIT] Running exploitation for trials: {trials_to_run}")

    # Generate spawn locations (same for all trials)
    print(f"[AUTO_EXPLOIT] Generating spawn locations for {len(goals)} goals, {trials_per_goal} trials per goal")
    spawn_locations = generate_spawn_locations(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance
    )

    # Create spawn/goal combinations
    combinations = []
    for goal in goals:
        goal_name = goal["name"]
        goal_spawns = spawn_locations.get(goal_name, [])
        for i, spawn_pos in enumerate(goal_spawns):
            combinations.append({
                "start": spawn_pos,
                "goal": goal,
                "trial_number": i + 1,
                "goal_name": goal_name
            })

    # Loop through each learning trial that needs exploitation
    for trial_num in trials_to_run:
        print(f"\n{'='*60}")
        print(f"[AUTO_EXPLOIT] Exploiting learning trial {trial_num}/{num_auto_trials}")
        print(f"{'='*60}\n")

        # Setup stats folder for this learning trial
        # Use same naming scheme as pkl folders: world_name_{trial_num}
        stats_base = f"stats_{auto_trial_name}"
        world_trial_folder = f"{world_name}_{trial_num}"
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", stats_base, world_trial_folder, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)

        hmaps_folder = os.path.join(
            PROJECT_ROOT, "analysis", stats_base, world_trial_folder, scale_name_str, "hmaps"
        )
        os.makedirs(hmaps_folder, exist_ok=True)

        # Create stats collector for this learning trial
        stats_collector_instance = stats_collector(output_dir=stats_folder)

        # Run all exploitation trials for this learning trial
        successful_trials = 0
        for i, combination in enumerate(combinations):
            start_pos = combination["start"]
            goal_name = combination["goal_name"]
            trial_number = combination["trial_number"]

            # Create trial ID
            trial_id = f"trial_{trial_number}_goal_{goal_name}_random"

            print(f"[AUTO_EXPLOIT] Learning trial {trial_num}, Exploit trial {trial_id}: {start_pos} -> {goal_name} ({successful_trials + 1}/{len(combinations)})")

            # Prepare trial-specific kwargs
            trial_kwargs = prepared_kwargs.copy()
            trial_kwargs["goal_config"] = {
                "type": "multi",
                "goals": goals,
                "target_goal": goal_name
            }
            trial_kwargs["optimal_path_distance"] = None
            trial_kwargs["path_failure_ratio"] = path_failure_ratio
            trial_kwargs["paths_folder"] = None
            trial_kwargs["hmaps_folder"] = hmaps_folder
            trial_kwargs["auto_trial_name"] = auto_trial_name
            trial_kwargs["current_auto_trial"] = trial_num

            # Run single trial with networks from this learning trial
            _run_single_trial(bot, mode, trial_id, start_pos, goal_name, stats_collector_instance, **trial_kwargs)
            successful_trials += 1

        print(f"[AUTO_EXPLOIT] Learning trial {trial_num} complete: {successful_trials} exploitation trials")

        # Clear GPU cache and run garbage collection to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[AUTO_EXPLOIT] Cleared GPU cache after learning trial {trial_num}")
        gc.collect()

    print(f"\n[AUTO_EXPLOIT] All exploitation complete! Ran {len(trials_to_run)}/{len(available_trials)} learning trials (skipped {len(available_trials) - len(trials_to_run)} already complete).")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


def _run_plotting_auto_trials(mode, **kwargs):
    """
    Handle automated multi-trial plotting mode.
    Loads networks from each LEARN_LOCATIONS_COVERAGE_AUTO trial folder and runs
    plotting mode for the specified run_time_hours.

    Uses the same pattern as other auto modes: single Driver instance,
    multiple initialization() + run() cycles.
    """
    # Extract auto trial parameters
    auto_trial_name = kwargs.get("auto_trial_name", "default_trial")
    num_auto_trials = kwargs.get("num_auto_trials", 5)

    print(f"[AUTO_PLOTTING] Starting automated plotting trials")
    print(f"[AUTO_PLOTTING] Trial name: {auto_trial_name}")
    print(f"[AUTO_PLOTTING] Number of learning trials to plot: {num_auto_trials}")

    # Create single Driver instance that will be reused for all trials
    bot = Driver()
    world_name = get_world_name(bot)
    if kwargs.get("goal_config") is None:
        kwargs["goal_config"] = _build_goal_config_for_world(world_name, exploit=False)

    # Check which learning trials exist
    available_trials = []
    for trial_num in range(1, num_auto_trials + 1):
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            available_trials.append(trial_num)
        else:
            print(f"[AUTO_PLOTTING] Learning trial {trial_num} not found, skipping...")

    if not available_trials:
        print(f"[AUTO_PLOTTING] No learning trials found! Run LEARN_LOCATIONS_COVERAGE_AUTO first.")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_PLOTTING] Found {len(available_trials)} learning trials to plot: {available_trials}")

    # Get scale info
    prepared_kwargs, scales_list, ablation_log = prepare_trial_configuration(kwargs)
    if ablation_log:
        print(f"[ABLATION] Applied: {', '.join(ablation_log)}")
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

    # Check which plotting trials are already complete and can be skipped
    trials_to_run = []
    for trial_num in available_trials:
        if _check_plotting_trial_complete(auto_trial_name, world_name, trial_num):
            print(f"[AUTO_PLOTTING] Plotting trial {trial_num} already complete, skipping...")
        else:
            trials_to_run.append(trial_num)

    if not trials_to_run:
        print(f"[AUTO_PLOTTING] All plotting trials already complete!")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_PLOTTING] Running plotting for trials: {trials_to_run}")

    # Loop through each learning trial that needs plotting
    for trial_num in trials_to_run:
        print(f"\n{'='*60}")
        print(f"[AUTO_PLOTTING] Plotting learning trial {trial_num}/{num_auto_trials}")
        print(f"{'='*60}\n")

        # Save trial parameters to JSON file
        trial_id = f"auto_plotting_{trial_num}"
        save_trial_parameters(
            world_name=world_name,
            trial_id=trial_id,
            mode=mode,
            **kwargs
        )

        # Initialize bot for this trial (loads networks from the trial folder)
        bot.initialization(
            mode=mode,
            run_time_hours=kwargs.get("run_time_hours", 2),
            randomize_start_loc=kwargs.get("randomize_start_loc", False),
            start_loc=kwargs.get("start_loc", [0, 0]),
            enable_ojas=False,
            enable_stdp=False,
            scales=scales_list,
            rcn_learning_rates=rcn_learning_rates,
            stats_collector=None,  # No stats collector for plotting
            trial_id=f"auto_plotting_{trial_num}",
            world_name=world_name,
            goal_config=kwargs.get("goal_config"),
            trial_config=kwargs.get("trial_config"),
            max_dist=kwargs.get("max_dist", 25),
            plot_bvc=kwargs.get("plot_bvc", False),
            td_learning=False,
            use_prox_mod=False,
            use_unified_multiscale=kwargs.get("use_unified_multiscale", False),
            environment_size=kwargs.get("environment_size", None),
            grid_size=kwargs.get("grid_size", None),
            coverage_percentage=kwargs.get("coverage_percentage", None),
            minimum_learning_duration_hours=kwargs.get("minimum_learning_duration_hours", 4.0),
            min_goal_visits=kwargs.get("min_goal_visits", 3),
            min_checkpoint_visits=kwargs.get("min_checkpoint_visits", 3),
            checkpoint_visit_radius=kwargs.get("checkpoint_visit_radius", 0.8),
            goal_assoc_require_all_scales=kwargs.get("goal_assoc_require_all_scales", False),
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
            pcn_learning_adaptation_mode=kwargs.get(
                "pcn_learning_adaptation_mode", "gaussian_post_competition_expression"
            ),
            pcn_learning_stdp_start_steps=kwargs.get(
                "pcn_learning_stdp_start_steps", 8000
            ),
            pcn_learning_stop_ojas_on_stabilization=kwargs.get(
                "pcn_learning_stop_ojas_on_stabilization", False
            ),
            pcn_learning_cross_scale_coupling_start_steps=kwargs.get(
                "pcn_learning_cross_scale_coupling_start_steps", 8000
            ),
            pcn_learning_cross_scale_coupling_ramp_steps=kwargs.get(
                "pcn_learning_cross_scale_coupling_ramp_steps", 12000
            ),
            pcn_learning_cross_scale_coupling_min=kwargs.get(
                "pcn_learning_cross_scale_coupling_min", 0.0
            ),
            pcn_soft_scale_overlap=kwargs.get("pcn_soft_scale_overlap", True),
            pcn_soft_scale_overlap_in_learning=kwargs.get("pcn_soft_scale_overlap_in_learning", True),
            pcn_soft_scale_gate_floor=kwargs.get("pcn_soft_scale_gate_floor", 0.20),
            pcn_soft_scale_gate_floor_in_learning=kwargs.get(
                "pcn_soft_scale_gate_floor_in_learning",
                pcn_soft_scale_gate_floor_in_learning,
            ),
            pcn_soft_cross_inhibition_scale=kwargs.get("pcn_soft_cross_inhibition_scale", 0.35),
            pcn_soft_cross_inhibition_scale_in_learning=kwargs.get("pcn_soft_cross_inhibition_scale_in_learning", 0.25),
            pcn_soft_cross_inhibition_cap=kwargs.get("pcn_soft_cross_inhibition_cap", 0.75),
            gcn_scale_invariant=kwargs.get("gcn_scale_invariant", True),
            gcn_use_adaptive_minmax=kwargs.get("gcn_use_adaptive_minmax", False),
            goal_assoc_unique_topk=kwargs.get("goal_assoc_unique_topk", 16),
            goal_assoc_max_activation_drop=kwargs.get("goal_assoc_max_activation_drop", 0.08),
            goal_assoc_locality_radius=kwargs.get("goal_assoc_locality_radius", 2.0),
            goal_assoc_locality_max_activation_drop=kwargs.get("goal_assoc_locality_max_activation_drop", 0.65),
            goal_map_use_compact_pc_gate=kwargs.get("goal_map_use_compact_pc_gate", True),
            goal_map_pc_compactness_peak_fraction=kwargs.get("goal_map_pc_compactness_peak_fraction", 0.50),
            goal_map_pc_compactness_min_peak=kwargs.get("goal_map_pc_compactness_min_peak", 0.05),
            goal_map_pc_compactness_max_radius=kwargs.get("goal_map_pc_compactness_max_radius", 4.0),
            goal_map_fill_source_envelope_tau=kwargs.get("goal_map_fill_source_envelope_tau", 6.0),
            goal_map_fill_source_envelope_floor=kwargs.get("goal_map_fill_source_envelope_floor", 0.10),
            record_experience_transitions=kwargs.get("record_experience_transitions", True),
            two_phase_learning=kwargs.get("two_phase_learning", False),
            phase1_min_steps=kwargs.get("phase1_min_steps", 2500),
            defer_experience_build_until_phase2_end=kwargs.get("defer_experience_build_until_phase2_end", True),
            goal_map_replay_timesteps=kwargs.get(
                "goal_map_replay_timesteps", goal_map_replay_timesteps
            ),
            goal_map_paper_replay_tau=kwargs.get(
                "goal_map_paper_replay_tau", goal_map_paper_replay_tau
            ),
            goal_map_goal_replay_event_count=kwargs.get(
                "goal_map_goal_replay_event_count", goal_map_goal_replay_event_count
            ),
            goal_map_checkpoint_replay_event_count=kwargs.get(
                "goal_map_checkpoint_replay_event_count",
                goal_map_checkpoint_replay_event_count,
            ),
            goal_map_replay_event_merge_gap_steps=kwargs.get(
                "goal_map_replay_event_merge_gap_steps",
                goal_map_replay_event_merge_gap_steps,
            ),
            goal_map_checkpoint_room_support_min_fraction=kwargs.get(
                "goal_map_checkpoint_room_support_min_fraction",
                goal_map_checkpoint_room_support_min_fraction,
            ),
            hybrid_path_replay_weight=kwargs.get("hybrid_path_replay_weight", 0.8),
            hybrid_diffusion_replay_weight=kwargs.get("hybrid_diffusion_replay_weight", 0.2),
            goal_map_path_replay_weight=kwargs.get("goal_map_path_replay_weight", 1.0),
            goal_map_diffusion_replay_weight=kwargs.get("goal_map_diffusion_replay_weight", 0.0),
            goal_map_experience_transition_topk=kwargs.get("goal_map_experience_transition_topk", 16),
            unified_reward_normalization_mode=kwargs.get("unified_reward_normalization_mode", "input_l1"),
            unified_goal_map_mode=kwargs.get("unified_goal_map_mode", "paper_room_local_replay"),
            goal_map_room_normalization_mode=kwargs.get(
                "goal_map_room_normalization_mode",
                goal_map_room_normalization_mode,
            ),
            goal_map_path_topk=kwargs.get("goal_map_path_topk", 16),
            goal_map_path_decay=kwargs.get("goal_map_path_decay", goal_map_path_decay),
            goal_map_path_frontier_only=kwargs.get("goal_map_path_frontier_only", True),
            goal_map_neighbor_topk=kwargs.get("goal_map_neighbor_topk", 12),
            goal_map_neighbor_steps=kwargs.get(
                "goal_map_neighbor_steps", goal_map_neighbor_steps
            ),
            goal_map_neighbor_decay=kwargs.get("goal_map_neighbor_decay", goal_map_neighbor_decay),
            goal_map_neighbor_frontier_only=kwargs.get("goal_map_neighbor_frontier_only", True),
            goal_map_neighbor_seed_scale=kwargs.get(
                "goal_map_neighbor_seed_scale", goal_map_neighbor_seed_scale
            ),
            goal_map_goal_seed_mode=kwargs.get(
                "goal_map_goal_seed_mode", goal_map_goal_seed_mode
            ),
            goal_map_goal_replay_retention_mode=kwargs.get(
                "goal_map_goal_replay_retention_mode",
                goal_map_goal_replay_retention_mode,
            ),
            goal_map_goal_seed_post_contact_steps=kwargs.get(
                "goal_map_goal_seed_post_contact_steps",
                goal_map_goal_seed_post_contact_steps,
            ),
            goal_map_goal_seed_within_visit_aggregation=kwargs.get(
                "goal_map_goal_seed_within_visit_aggregation",
                goal_map_goal_seed_within_visit_aggregation,
            ),
            goal_map_goal_seed_across_visit_aggregation=kwargs.get(
                "goal_map_goal_seed_across_visit_aggregation",
                goal_map_goal_seed_across_visit_aggregation,
            ),
            goal_map_goal_seed_state_normalization=kwargs.get(
                "goal_map_goal_seed_state_normalization",
                goal_map_goal_seed_state_normalization,
            ),
            goal_map_goal_seed_self_retention_alpha=kwargs.get(
                "goal_map_goal_seed_self_retention_alpha",
                goal_map_goal_seed_self_retention_alpha,
            ),
            goal_map_path_constraint_sigma=kwargs.get(
                "goal_map_path_constraint_sigma", goal_map_path_constraint_sigma
            ),
            goal_map_path_constraint_topk=kwargs.get(
                "goal_map_path_constraint_topk", goal_map_path_constraint_topk
            ),
            goal_map_local_constraint_sigma=kwargs.get("goal_map_local_constraint_sigma", goal_map_local_constraint_sigma),
            goal_map_local_constraint_topk=kwargs.get("goal_map_local_constraint_topk", goal_map_local_constraint_topk),
            goal_map_goal_seed_center_radius_factor=kwargs.get(
                "goal_map_goal_seed_center_radius_factor",
                goal_map_goal_seed_center_radius_factor,
            ),
            goal_map_goal_seed_center_min_radius=kwargs.get(
                "goal_map_goal_seed_center_min_radius",
                goal_map_goal_seed_center_min_radius,
            ),
            goal_map_checkpoint_seed_center_radius_factor=kwargs.get(
                "goal_map_checkpoint_seed_center_radius_factor",
                goal_map_checkpoint_seed_center_radius_factor,
            ),
            goal_map_checkpoint_seed_center_min_radius=kwargs.get(
                "goal_map_checkpoint_seed_center_min_radius",
                goal_map_checkpoint_seed_center_min_radius,
            ),
            goal_map_spatial_obstacle_block=kwargs.get("goal_map_spatial_obstacle_block", True),
            goal_map_obstacle_margin=kwargs.get("goal_map_obstacle_margin", 0.05),
            goal_map_use_checkpoint_relays=kwargs.get(
                "goal_map_use_checkpoint_relays", goal_map_use_checkpoint_relays
            ),
            goal_map_checkpoint_mode=kwargs.get(
                "goal_map_checkpoint_mode", goal_map_checkpoint_mode
            ),
            goal_map_checkpoint_scale=kwargs.get(
                "goal_map_checkpoint_scale", goal_map_checkpoint_scale
            ),
            goal_map_checkpoint_support_mode=kwargs.get(
                "goal_map_checkpoint_support_mode", goal_map_checkpoint_support_mode
            ),
            goal_map_checkpoint_stop_parent_replay=kwargs.get(
                "goal_map_checkpoint_stop_parent_replay", goal_map_checkpoint_stop_parent_replay
            ),
            checkpoint_boost_threshold=kwargs.get(
                "checkpoint_boost_threshold", checkpoint_boost_threshold
            ),
            checkpoint_target_arrival=kwargs.get(
                "checkpoint_target_arrival", checkpoint_target_arrival
            ),
            checkpoint_min_relay_weight=kwargs.get(
                "checkpoint_min_relay_weight", checkpoint_min_relay_weight
            ),
            goal_map_route_selection_support_fraction=kwargs.get(
                "goal_map_route_selection_support_fraction",
                goal_map_route_selection_support_fraction,
            ),
            goal_map_route_selection_overlap_threshold=kwargs.get(
                "goal_map_route_selection_overlap_threshold",
                goal_map_route_selection_overlap_threshold,
            ),
            goal_map_use_denoised_experience_graph=kwargs.get(
                "goal_map_use_denoised_experience_graph", goal_map_use_denoised_experience_graph
            ),
            goal_map_use_segmented_checkpoint_histories=kwargs.get(
                "goal_map_use_segmented_checkpoint_histories",
                goal_map_use_segmented_checkpoint_histories,
            ),
            unified_preplay_scale_arbitration=kwargs.get("unified_preplay_scale_arbitration", True),
            unified_preplay_scale_prior_mix=kwargs.get("unified_preplay_scale_prior_mix", 0.0),
            goal_map_smoothing_sigma=kwargs.get("goal_map_smoothing_sigma", 1.0),
            goal_map_smoothing_topk=kwargs.get("goal_map_smoothing_topk", 24),
            exploit_direction_clearance_percentile=kwargs.get("exploit_direction_clearance_percentile", 20.0),
            goal_map_smoothing_passes=kwargs.get("goal_map_smoothing_passes", 0),
            goal_map_smoothing_mix=kwargs.get("goal_map_smoothing_mix", 0.45),
            unified_preplay_commit_threshold=kwargs.get("unified_preplay_commit_threshold", 0.30),
            preplay_heading_selection_mode=kwargs.get("preplay_heading_selection_mode", "trajectory_vector"),
            unified_exploit_action_mode=kwargs.get("unified_exploit_action_mode", "local_reward_gradient_primary"),
            unified_transition_preplay_enabled=kwargs.get("unified_transition_preplay_enabled", False),
            unified_spatial_fallback_enabled=kwargs.get("unified_spatial_fallback_enabled", True),
            unified_spatial_override_margin=kwargs.get("unified_spatial_override_margin", 0.0),
            unified_spatial_lookahead_distance=kwargs.get("unified_spatial_lookahead_distance", 1.0),
            unified_spatial_lookahead_sigma=kwargs.get("unified_spatial_lookahead_sigma", 0.9),
            unified_exploit_step_fraction=kwargs.get("unified_exploit_step_fraction", 0.25),
            prune_experience_loops=kwargs.get("prune_experience_loops", True),
            loop_prune_min_top1=kwargs.get("loop_prune_min_top1", 0.08),
            loop_prune_min_top1_to_top2_ratio=kwargs.get("loop_prune_min_top1_to_top2_ratio", 1.15),
        )

        # Run the plotting trial
        print(f"[AUTO_PLOTTING] Running plotting for trial {trial_num}...")
        bot.run()
        print(f"[AUTO_PLOTTING] Plotting trial {trial_num} complete!")

        # Create marker file to indicate this plotting trial is complete
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{trial_num}"
        trial_path = os.path.join(base_folder, trial_folder)
        plotting_marker = os.path.join(trial_path, "plotting_complete.txt")

        with open(plotting_marker, 'w') as f:
            f.write(f"Plotting completed for trial {trial_num}\n")
            f.write(f"Run time: {kwargs.get('run_time_hours', 2)} hours\n")

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[AUTO_PLOTTING] Cleared GPU cache after trial {trial_num}")
        gc.collect()

    print(f"\n[AUTO_PLOTTING] All plotting complete! Ran {len(trials_to_run)}/{len(available_trials)} learning trials (skipped {len(available_trials) - len(trials_to_run)} already complete).")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


def _run_plotting_coverage_auto_trials(mode, **kwargs):
    """
    Handle automated multi-trial plotting mode with coverage-based stopping condition.
    Loads networks from each LEARN_LOCATIONS_COVERAGE_AUTO trial folder and runs
    plotting mode until the specified coverage percentage is reached.

    Uses the same pattern as other auto modes: single Driver instance,
    multiple initialization() + run() cycles.
    """
    # Extract auto trial parameters
    auto_trial_name = kwargs.get("auto_trial_name", "default_trial")
    num_auto_trials = kwargs.get("num_auto_trials", 5)

    print(f"[AUTO_PLOTTING_COVERAGE] Starting automated plotting trials with coverage stopping condition")
    print(f"[AUTO_PLOTTING_COVERAGE] Trial name: {auto_trial_name}")
    print(f"[AUTO_PLOTTING_COVERAGE] Number of learning trials to plot: {num_auto_trials}")
    print(f"[AUTO_PLOTTING_COVERAGE] Coverage target: {kwargs.get('coverage_percentage', 0.95) * 100}%")

    # Create single Driver instance that will be reused for all trials
    bot = Driver()
    world_name = get_world_name(bot)
    if kwargs.get("goal_config") is None:
        kwargs["goal_config"] = _build_goal_config_for_world(world_name, exploit=False)

    # Check which learning trials exist
    available_trials = []
    for trial_num in range(1, num_auto_trials + 1):
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            available_trials.append(trial_num)
        else:
            print(f"[AUTO_PLOTTING_COVERAGE] Learning trial {trial_num} not found, skipping...")

    if not available_trials:
        print(f"[AUTO_PLOTTING_COVERAGE] No learning trials found! Run LEARN_LOCATIONS_COVERAGE_AUTO first.")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_PLOTTING_COVERAGE] Found {len(available_trials)} learning trials to plot: {available_trials}")

    # Get scale info
    prepared_kwargs, scales_list, ablation_log = prepare_trial_configuration(kwargs)
    if ablation_log:
        print(f"[ABLATION] Applied: {', '.join(ablation_log)}")
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

    # Check which plotting trials are already complete and can be skipped
    trials_to_run = []
    for trial_num in available_trials:
        if _check_plotting_trial_complete(auto_trial_name, world_name, trial_num, mode_suffix="_coverage"):
            print(f"[AUTO_PLOTTING_COVERAGE] Plotting trial {trial_num} already complete, skipping...")
        else:
            trials_to_run.append(trial_num)

    if not trials_to_run:
        print(f"[AUTO_PLOTTING_COVERAGE] All plotting trials already complete!")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_PLOTTING_COVERAGE] Running plotting for trials: {trials_to_run}")

    # Loop through each learning trial that needs plotting
    for trial_num in trials_to_run:
        print(f"\n{'='*60}")
        print(f"[AUTO_PLOTTING_COVERAGE] Plotting learning trial {trial_num}/{num_auto_trials}")
        print(f"{'='*60}\n")

        # Save trial parameters to JSON file
        trial_id = f"auto_plotting_coverage_{trial_num}"
        save_trial_parameters(
            world_name=world_name,
            trial_id=trial_id,
            mode=mode,
            **kwargs
        )

        # Initialize bot for this trial (loads networks from the trial folder)
        # Use PLOTTING_COVERAGE_AUTO mode to ensure correct path loading
        bot.initialization(
            mode=RobotMode.PLOTTING_COVERAGE_AUTO,  # Use PLOTTING_COVERAGE_AUTO mode for correct path
            run_time_hours=kwargs.get("run_time_hours", 24),  # High fallback time limit
            randomize_start_loc=kwargs.get("randomize_start_loc", False),
            start_loc=kwargs.get("start_loc", [0, 0]),
            enable_ojas=False,
            enable_stdp=False,
            scales=scales_list,
            rcn_learning_rates=rcn_learning_rates,
            stats_collector=None,  # No stats collector for plotting
            trial_id=f"auto_plotting_coverage_{trial_num}",
            world_name=world_name,
            goal_config=kwargs.get("goal_config"),
            trial_config=kwargs.get("trial_config"),
            max_dist=kwargs.get("max_dist", 25),
            plot_bvc=kwargs.get("plot_bvc", False),
            td_learning=False,
            use_prox_mod=False,
            use_unified_multiscale=kwargs.get("use_unified_multiscale", False),
            environment_size=kwargs.get("environment_size", None),
            grid_size=kwargs.get("grid_size", None),
            coverage_percentage=kwargs.get("coverage_percentage", None),
            minimum_learning_duration_hours=kwargs.get("minimum_learning_duration_hours", 4.0),
            min_goal_visits=kwargs.get("min_goal_visits", 3),
            min_checkpoint_visits=kwargs.get("min_checkpoint_visits", 3),
            checkpoint_visit_radius=kwargs.get("checkpoint_visit_radius", 0.8),
            goal_assoc_require_all_scales=kwargs.get("goal_assoc_require_all_scales", False),
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
            pcn_learning_adaptation_mode=kwargs.get(
                "pcn_learning_adaptation_mode", "gaussian_post_competition_expression"
            ),
            pcn_learning_stdp_start_steps=kwargs.get(
                "pcn_learning_stdp_start_steps", 8000
            ),
            pcn_learning_stop_ojas_on_stabilization=kwargs.get(
                "pcn_learning_stop_ojas_on_stabilization", False
            ),
            pcn_learning_cross_scale_coupling_start_steps=kwargs.get(
                "pcn_learning_cross_scale_coupling_start_steps", 8000
            ),
            pcn_learning_cross_scale_coupling_ramp_steps=kwargs.get(
                "pcn_learning_cross_scale_coupling_ramp_steps", 12000
            ),
            pcn_learning_cross_scale_coupling_min=kwargs.get(
                "pcn_learning_cross_scale_coupling_min", 0.0
            ),
            pcn_soft_scale_overlap=kwargs.get("pcn_soft_scale_overlap", True),
            pcn_soft_scale_overlap_in_learning=kwargs.get("pcn_soft_scale_overlap_in_learning", True),
            pcn_soft_scale_gate_floor=kwargs.get("pcn_soft_scale_gate_floor", 0.20),
            pcn_soft_scale_gate_floor_in_learning=kwargs.get(
                "pcn_soft_scale_gate_floor_in_learning",
                pcn_soft_scale_gate_floor_in_learning,
            ),
            pcn_soft_cross_inhibition_scale=kwargs.get("pcn_soft_cross_inhibition_scale", 0.35),
            pcn_soft_cross_inhibition_scale_in_learning=kwargs.get("pcn_soft_cross_inhibition_scale_in_learning", 0.25),
            pcn_soft_cross_inhibition_cap=kwargs.get("pcn_soft_cross_inhibition_cap", 0.75),
            gcn_scale_invariant=kwargs.get("gcn_scale_invariant", True),
            gcn_use_adaptive_minmax=kwargs.get("gcn_use_adaptive_minmax", False),
            goal_assoc_unique_topk=kwargs.get("goal_assoc_unique_topk", 16),
            goal_assoc_max_activation_drop=kwargs.get("goal_assoc_max_activation_drop", 0.08),
            goal_assoc_locality_radius=kwargs.get("goal_assoc_locality_radius", 2.0),
            goal_assoc_locality_max_activation_drop=kwargs.get("goal_assoc_locality_max_activation_drop", 0.65),
            goal_map_use_compact_pc_gate=kwargs.get("goal_map_use_compact_pc_gate", True),
            goal_map_pc_compactness_peak_fraction=kwargs.get("goal_map_pc_compactness_peak_fraction", 0.50),
            goal_map_pc_compactness_min_peak=kwargs.get("goal_map_pc_compactness_min_peak", 0.05),
            goal_map_pc_compactness_max_radius=kwargs.get("goal_map_pc_compactness_max_radius", 4.0),
            goal_map_fill_source_envelope_tau=kwargs.get("goal_map_fill_source_envelope_tau", 6.0),
            goal_map_fill_source_envelope_floor=kwargs.get("goal_map_fill_source_envelope_floor", 0.10),
            record_experience_transitions=kwargs.get("record_experience_transitions", True),
            two_phase_learning=kwargs.get("two_phase_learning", False),
            phase1_min_steps=kwargs.get("phase1_min_steps", 2500),
            defer_experience_build_until_phase2_end=kwargs.get("defer_experience_build_until_phase2_end", True),
            goal_map_replay_timesteps=kwargs.get(
                "goal_map_replay_timesteps", goal_map_replay_timesteps
            ),
            goal_map_paper_replay_tau=kwargs.get(
                "goal_map_paper_replay_tau", goal_map_paper_replay_tau
            ),
            goal_map_goal_replay_event_count=kwargs.get(
                "goal_map_goal_replay_event_count", goal_map_goal_replay_event_count
            ),
            goal_map_checkpoint_replay_event_count=kwargs.get(
                "goal_map_checkpoint_replay_event_count",
                goal_map_checkpoint_replay_event_count,
            ),
            goal_map_replay_event_merge_gap_steps=kwargs.get(
                "goal_map_replay_event_merge_gap_steps",
                goal_map_replay_event_merge_gap_steps,
            ),
            goal_map_checkpoint_room_support_min_fraction=kwargs.get(
                "goal_map_checkpoint_room_support_min_fraction",
                goal_map_checkpoint_room_support_min_fraction,
            ),
            hybrid_path_replay_weight=kwargs.get("hybrid_path_replay_weight", 0.8),
            hybrid_diffusion_replay_weight=kwargs.get("hybrid_diffusion_replay_weight", 0.2),
            goal_map_path_replay_weight=kwargs.get("goal_map_path_replay_weight", 1.0),
            goal_map_diffusion_replay_weight=kwargs.get("goal_map_diffusion_replay_weight", 0.0),
            goal_map_experience_transition_topk=kwargs.get("goal_map_experience_transition_topk", 16),
            unified_reward_normalization_mode=kwargs.get("unified_reward_normalization_mode", "input_l1"),
            unified_goal_map_mode=kwargs.get("unified_goal_map_mode", "paper_room_local_replay"),
            goal_map_room_normalization_mode=kwargs.get(
                "goal_map_room_normalization_mode",
                goal_map_room_normalization_mode,
            ),
            goal_map_path_topk=kwargs.get("goal_map_path_topk", 16),
            goal_map_path_decay=kwargs.get("goal_map_path_decay", goal_map_path_decay),
            goal_map_path_frontier_only=kwargs.get("goal_map_path_frontier_only", True),
            goal_map_neighbor_topk=kwargs.get("goal_map_neighbor_topk", 12),
            goal_map_neighbor_steps=kwargs.get(
                "goal_map_neighbor_steps", goal_map_neighbor_steps
            ),
            goal_map_neighbor_decay=kwargs.get("goal_map_neighbor_decay", goal_map_neighbor_decay),
            goal_map_neighbor_frontier_only=kwargs.get("goal_map_neighbor_frontier_only", True),
            goal_map_neighbor_seed_scale=kwargs.get(
                "goal_map_neighbor_seed_scale", goal_map_neighbor_seed_scale
            ),
            goal_map_goal_seed_mode=kwargs.get(
                "goal_map_goal_seed_mode", goal_map_goal_seed_mode
            ),
            goal_map_goal_replay_retention_mode=kwargs.get(
                "goal_map_goal_replay_retention_mode",
                goal_map_goal_replay_retention_mode,
            ),
            goal_map_goal_seed_post_contact_steps=kwargs.get(
                "goal_map_goal_seed_post_contact_steps",
                goal_map_goal_seed_post_contact_steps,
            ),
            goal_map_goal_seed_within_visit_aggregation=kwargs.get(
                "goal_map_goal_seed_within_visit_aggregation",
                goal_map_goal_seed_within_visit_aggregation,
            ),
            goal_map_goal_seed_across_visit_aggregation=kwargs.get(
                "goal_map_goal_seed_across_visit_aggregation",
                goal_map_goal_seed_across_visit_aggregation,
            ),
            goal_map_goal_seed_state_normalization=kwargs.get(
                "goal_map_goal_seed_state_normalization",
                goal_map_goal_seed_state_normalization,
            ),
            goal_map_goal_seed_self_retention_alpha=kwargs.get(
                "goal_map_goal_seed_self_retention_alpha",
                goal_map_goal_seed_self_retention_alpha,
            ),
            goal_map_path_constraint_sigma=kwargs.get(
                "goal_map_path_constraint_sigma", goal_map_path_constraint_sigma
            ),
            goal_map_path_constraint_topk=kwargs.get(
                "goal_map_path_constraint_topk", goal_map_path_constraint_topk
            ),
            goal_map_local_constraint_sigma=kwargs.get("goal_map_local_constraint_sigma", goal_map_local_constraint_sigma),
            goal_map_local_constraint_topk=kwargs.get("goal_map_local_constraint_topk", goal_map_local_constraint_topk),
            goal_map_goal_seed_center_radius_factor=kwargs.get(
                "goal_map_goal_seed_center_radius_factor",
                goal_map_goal_seed_center_radius_factor,
            ),
            goal_map_goal_seed_center_min_radius=kwargs.get(
                "goal_map_goal_seed_center_min_radius",
                goal_map_goal_seed_center_min_radius,
            ),
            goal_map_checkpoint_seed_center_radius_factor=kwargs.get(
                "goal_map_checkpoint_seed_center_radius_factor",
                goal_map_checkpoint_seed_center_radius_factor,
            ),
            goal_map_checkpoint_seed_center_min_radius=kwargs.get(
                "goal_map_checkpoint_seed_center_min_radius",
                goal_map_checkpoint_seed_center_min_radius,
            ),
            goal_map_spatial_obstacle_block=kwargs.get("goal_map_spatial_obstacle_block", True),
            goal_map_obstacle_margin=kwargs.get("goal_map_obstacle_margin", 0.05),
            goal_map_use_checkpoint_relays=kwargs.get(
                "goal_map_use_checkpoint_relays", goal_map_use_checkpoint_relays
            ),
            goal_map_checkpoint_mode=kwargs.get(
                "goal_map_checkpoint_mode", goal_map_checkpoint_mode
            ),
            goal_map_checkpoint_scale=kwargs.get(
                "goal_map_checkpoint_scale", goal_map_checkpoint_scale
            ),
            goal_map_checkpoint_support_mode=kwargs.get(
                "goal_map_checkpoint_support_mode", goal_map_checkpoint_support_mode
            ),
            goal_map_checkpoint_stop_parent_replay=kwargs.get(
                "goal_map_checkpoint_stop_parent_replay", goal_map_checkpoint_stop_parent_replay
            ),
            checkpoint_boost_threshold=kwargs.get(
                "checkpoint_boost_threshold", checkpoint_boost_threshold
            ),
            checkpoint_target_arrival=kwargs.get(
                "checkpoint_target_arrival", checkpoint_target_arrival
            ),
            checkpoint_min_relay_weight=kwargs.get(
                "checkpoint_min_relay_weight", checkpoint_min_relay_weight
            ),
            goal_map_route_selection_support_fraction=kwargs.get(
                "goal_map_route_selection_support_fraction",
                goal_map_route_selection_support_fraction,
            ),
            goal_map_route_selection_overlap_threshold=kwargs.get(
                "goal_map_route_selection_overlap_threshold",
                goal_map_route_selection_overlap_threshold,
            ),
            goal_map_use_denoised_experience_graph=kwargs.get(
                "goal_map_use_denoised_experience_graph", goal_map_use_denoised_experience_graph
            ),
            goal_map_use_segmented_checkpoint_histories=kwargs.get(
                "goal_map_use_segmented_checkpoint_histories",
                goal_map_use_segmented_checkpoint_histories,
            ),
            unified_preplay_scale_arbitration=kwargs.get("unified_preplay_scale_arbitration", True),
            unified_preplay_scale_prior_mix=kwargs.get("unified_preplay_scale_prior_mix", 0.0),
            goal_map_smoothing_sigma=kwargs.get("goal_map_smoothing_sigma", 1.0),
            goal_map_smoothing_topk=kwargs.get("goal_map_smoothing_topk", 24),
            exploit_direction_clearance_percentile=kwargs.get("exploit_direction_clearance_percentile", 20.0),
            goal_map_smoothing_passes=kwargs.get("goal_map_smoothing_passes", 0),
            goal_map_smoothing_mix=kwargs.get("goal_map_smoothing_mix", 0.45),
            unified_preplay_commit_threshold=kwargs.get("unified_preplay_commit_threshold", 0.30),
            preplay_heading_selection_mode=kwargs.get("preplay_heading_selection_mode", "trajectory_vector"),
            unified_exploit_action_mode=kwargs.get("unified_exploit_action_mode", "local_reward_gradient_primary"),
            unified_transition_preplay_enabled=kwargs.get("unified_transition_preplay_enabled", False),
            unified_spatial_fallback_enabled=kwargs.get("unified_spatial_fallback_enabled", True),
            unified_spatial_override_margin=kwargs.get("unified_spatial_override_margin", 0.0),
            unified_spatial_lookahead_distance=kwargs.get("unified_spatial_lookahead_distance", 1.0),
            unified_spatial_lookahead_sigma=kwargs.get("unified_spatial_lookahead_sigma", 0.9),
            unified_exploit_step_fraction=kwargs.get("unified_exploit_step_fraction", 0.25),
            prune_experience_loops=kwargs.get("prune_experience_loops", True),
            loop_prune_min_top1=kwargs.get("loop_prune_min_top1", 0.08),
            loop_prune_min_top1_to_top2_ratio=kwargs.get("loop_prune_min_top1_to_top2_ratio", 1.15),
        )

        # Run the plotting trial with coverage-based stopping
        print(f"[AUTO_PLOTTING_COVERAGE] Running plotting for trial {trial_num} until coverage target reached...")
        bot.run()
        print(f"[AUTO_PLOTTING_COVERAGE] Plotting trial {trial_num} complete!")

        # Create marker file to indicate this plotting trial is complete (with _coverage suffix)
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{trial_num}"
        trial_path = os.path.join(base_folder, trial_folder)
        plotting_marker = os.path.join(trial_path, "plotting_complete_coverage.txt")

        with open(plotting_marker, 'w') as f:
            f.write(f"Plotting (coverage-based) completed for trial {trial_num}\n")
            f.write(f"Coverage target: {kwargs.get('coverage_percentage', 0.95) * 100}%\n")
            f.write(f"Environment size: {kwargs.get('environment_size', [20.0, 20.0])}\n")
            f.write(f"Grid size: {kwargs.get('grid_size', 0.5)}\n")

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[AUTO_PLOTTING_COVERAGE] Cleared GPU cache after trial {trial_num}")
        gc.collect()

    print(f"\n[AUTO_PLOTTING_COVERAGE] All plotting complete! Ran {len(trials_to_run)}/{len(available_trials)} learning trials (skipped {len(available_trials) - len(trials_to_run)} already complete).")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


#################################
# run_bot
#################################

def run_bot(mode, **kwargs):
    """
    Runs the bot in the specified mode with the given parameters.
    Supports both simple and combination trial configurations.
    """
    # Handle AUTO modes separately
    if mode == RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO:
        _run_learn_coverage_auto_trials(mode, **kwargs)
        return
    elif mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO:
        _run_exploit_random_auto_trials(mode, **kwargs)
        return
    elif mode == RobotMode.PLOTTING_AUTO:
        _run_plotting_auto_trials(mode, **kwargs)
        return
    elif mode == RobotMode.PLOTTING_COVERAGE_AUTO:
        _run_plotting_coverage_auto_trials(mode, **kwargs)
        return

    # Get trial configuration and remove it from kwargs to avoid conflict
    trial_config = kwargs.pop("trial_config", {
        "type": "simple",
        "count": 1
    })

    print(f"[INFO] Starting run with mode: {mode}")
    print(f"[INFO] Trial type: {trial_config['type']}")

    if trial_config["type"] == "simple":
        _run_simple_trials(mode, trial_config, **kwargs)
    elif trial_config["type"] == "combinations":
        _run_combination_trials(mode, trial_config, **kwargs)
    elif trial_config["type"] == "random_spawns":
        _run_random_spawn_trials(mode, trial_config, **kwargs)
    else:
        raise ValueError(f"Unknown trial type: {trial_config['type']}")

    # Pause the simulation
    bot = Driver()
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

#################################
# Main Controller Entry Point
#################################

if __name__ == "__main__":

    # We'll map string to the actual RobotMode enum
    MODES_MAP = {
        "LEARN_OJAS": RobotMode.LEARN_OJAS,
        "LEARN_HEBB": RobotMode.LEARN_HEBB,
        "LEARN_LOCATIONS_TWO_PHASE": RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        "LEARN_LOCATIONS_ADAPTIVE": RobotMode.LEARN_LOCATIONS_ADAPTIVE,
        "DMTP": RobotMode.DMTP,
        "EXPLOIT": RobotMode.EXPLOIT,
        "EXPLOIT_SAVE": RobotMode.EXPLOIT,
        "LEARNING_SAVE": RobotMode.EXPLOIT,
        "PLOTTING": RobotMode.PLOTTING,
        "LEARN_LOCATIONS_COVERAGE": RobotMode.LEARN_LOCATIONS_COVERAGE,
        "LEARN_LOCATIONS_COVERAGE_AUTO": RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        "EXPLOIT_LOCATIONS_RANDOM": RobotMode.EXPLOIT_LOCATIONS_RANDOM,
        "EXPLOIT_LOCATIONS_RANDOM_AUTO": RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        "PLOTTING_AUTO": RobotMode.PLOTTING_AUTO,
        "PLOTTING_COVERAGE_AUTO": RobotMode.PLOTTING_COVERAGE_AUTO,
        "REBUILD_REWARD_MAP": RobotMode.REBUILD_REWARD_MAP,
    }

    SELECTED_MODE = "LEARN_LOCATIONS_COVERAGE"
    td_learning = False # keep off
    corners = [[8,-8]] # start point
    dmtp_start = [-9,9]
    exploit_start = corners[0]
    start_loc = [1, 1]

    goal_location = [-7, 7]
    randomize_start_loc = False
    use_prox_mod = False

    # ========================================================================
    # UNIFIED MULTI-SCALE FLAG
    # Set to True to enable adaptive cross-scale inhibition and unified replay
    # Set to False to use original independent scales (for rollback)
    # ========================================================================
    use_unified_multiscale = True
    # Make GCN coding shared/scale-invariant across all active scales.
    gcn_scale_invariant = False
    # Match old non-unified per-cell grid normalization during ablations.
    gcn_use_adaptive_minmax = True
    # Goal association uniqueness: prefer near-top PCs that are not already reused by other goals.
    goal_assoc_unique_topk = 16
    goal_assoc_max_activation_drop = 0.08
    # Reject aliased goal PCs whose field peaks are far from the actual goal.
    goal_assoc_locality_radius = 2.0
    goal_assoc_locality_max_activation_drop = 0.65
    # Reward should only use compact, single-field PCs; broad/split fields create detached peaks.
    goal_map_use_compact_pc_gate = True
    goal_map_pc_compactness_peak_fraction = 0.50
    goal_map_pc_compactness_min_peak = 0.05
    goal_map_pc_compactness_max_radius = 4.0
    goal_map_fill_source_envelope_tau = 6.0
    goal_map_fill_source_envelope_floor = 0.10
    # Context-dependent modulation of boundary-driven input (alternative to hard eligibility masking)
    use_bvc_context_modulation = True
    bvc_context_gain_floor = 0.0
    bvc_context_gain_strength = 1.0
    # Whole-model unified ablation flags.
    # Current production path: post-competition Gaussian expression.
    ablation_force_nonunified_baseline = False
    ablation_legacy_grid_runtime = True
    ablation_disable_unified_scale_selection = False
    ablation_disable_unified_cross_scale_learning = False
    ablation_disable_unified_learning_adaptation = False
    ablation_disable_unified_dopt_jitter = False
    ablation_disable_unified_large_scale_plateau = False
    record_experience_transitions = True
    # Two-phase learning settings: OJAS stabilize -> STDP learn coverage/goals.
    # Stability is measured by revisit consistency: the same spatial location
    # must reliably produce the same place cell activation pattern.
    two_phase_learning = False
    phase1_min_steps = 2500
    phase1_max_steps = 20000           # force phase-2 transition after this many steps
    phase1_bin_size = 0.5              # spatial bin size in metres
    phase1_min_revisit_bins = 15       # unique bins that must have been revisited
    phase1_revisit_cosine_threshold = 0.90   # mean revisit cosine similarity required
    phase1_revisit_window = 200        # rolling window size for revisit cosine history
    defer_experience_build_until_phase2_end = True
    goal_map_replay_timesteps = 12
    goal_map_paper_replay_tau = 8.0
    goal_map_goal_replay_event_count = 4
    goal_map_checkpoint_replay_event_count = 3
    goal_map_replay_event_merge_gap_steps = 3
    goal_map_checkpoint_room_support_min_fraction = 0.98
    hybrid_path_replay_weight = 0.8
    hybrid_diffusion_replay_weight = 0.2
    goal_map_path_replay_weight = 1.0
    goal_map_diffusion_replay_weight = 0.0
    goal_map_experience_transition_topk = 16
    unified_reward_normalization_mode = "input_l1"
    unified_goal_map_mode = "paper_room_local_replay"
    goal_map_room_normalization_mode = "per_room_peak"
    goal_map_path_topk = 16
    goal_map_path_decay = 0.985
    goal_map_path_frontier_only = True
    goal_map_neighbor_topk = 12
    goal_map_neighbor_steps = 6
    goal_map_neighbor_decay = 0.84
    goal_map_neighbor_frontier_only = True
    goal_map_neighbor_seed_scale = 0.60
    goal_map_goal_seed_mode = "contact_ensemble"
    goal_map_goal_replay_retention_mode = "baseline"
    goal_map_goal_seed_post_contact_steps = 2
    goal_map_goal_seed_within_visit_aggregation = "max"
    goal_map_goal_seed_across_visit_aggregation = "mean"
    goal_map_goal_seed_state_normalization = "peak"
    goal_map_goal_seed_self_retention_alpha = 0.15
    goal_map_path_constraint_sigma = 4.0
    goal_map_path_constraint_topk = 128
    goal_map_local_constraint_sigma = 2.6
    goal_map_local_constraint_topk = 36
    goal_map_goal_seed_center_radius_factor = 0.0
    goal_map_goal_seed_center_min_radius = 0.0
    goal_map_checkpoint_seed_center_radius_factor = 0.0
    goal_map_checkpoint_seed_center_min_radius = 0.6
    goal_map_use_checkpoint_relays = False
    unified_preplay_scale_arbitration = True
    unified_preplay_scale_prior_mix = 0.0
    goal_map_smoothing_sigma = 1.0
    goal_map_smoothing_topk = 24
    goal_map_smoothing_passes = 0
    goal_map_smoothing_mix = 0.45
    unified_preplay_commit_threshold = 0.30
    preplay_heading_selection_mode = "trajectory_vector"
    unified_exploit_action_mode = "local_reward_gradient_primary"
    unified_transition_preplay_enabled = False
    unified_spatial_fallback_enabled = True
    unified_spatial_override_margin = 0.0
    unified_spatial_lookahead_distance = 1.0
    unified_spatial_lookahead_sigma = 0.9
    unified_exploit_step_fraction = 0.25
    exploit_direction_clearance_percentile = 20.0
    prune_experience_loops = True
    loop_prune_min_top1 = 0.08
    loop_prune_min_top1_to_top2_ratio = 1.15
    # Proximity mode controls scale-selection distance signal.
    # "local_minima": finds local minima in the LiDAR profile (parameter-free,
    #   handles doorways/corners/open space; returns harmonic mean of two closest)
    # "opposite_pair_percentile": corridor-aware opposite-pair half-width
    # "min": classic nearest-wall minimum distance
    proximity_mode = "local_minima"
    proximity_pair_percentile = 25.0  # used only when proximity_mode == "opposite_pair_percentile"
    # Gate mode for unified PCN: "normal", "no_gate_no_inhibition", "no_gate_with_inhibition"
    pcn_gate_mode = "normal"
    # Unified learning-time scale adaptation:
    # each scale competes internally first, then a smoothed Gaussian gain
    # modulates the final expressed output and learning uses that expressed code.
    pcn_learning_adaptation_mode = "gaussian_post_competition_expression"
    pcn_learning_stdp_start_steps = 0
    pcn_learning_stop_ojas_on_stabilization = False
    pcn_learning_cross_scale_coupling_start_steps = 0
    pcn_learning_cross_scale_coupling_ramp_steps = 0
    pcn_learning_cross_scale_coupling_min = 1.0
    # Re-enable unified grid/BVC balancing after the aliasing ablation pass.
    pcn_grid_balance_modalities = False
    # Toggle for safer adaptive overlap in the unified forward path.
    pcn_soft_scale_overlap = True
    pcn_soft_scale_overlap_in_learning = True
    pcn_soft_scale_gate_floor = 0.20
    pcn_soft_scale_gate_floor_in_learning = 0.12
    pcn_soft_cross_inhibition_scale = 0.35
    pcn_soft_cross_inhibition_scale_in_learning = 0.25
    pcn_soft_cross_inhibition_cap = 0.75
    goal_map_spatial_obstacle_block = True
    goal_map_obstacle_margin = 0.05
    goal_map_checkpoint_mode = "multi_route_memory_graph"
    goal_map_checkpoint_scale = 0.25
    goal_map_checkpoint_support_mode = "threshold_normalized"
    goal_map_checkpoint_stop_parent_replay = True
    checkpoint_boost_threshold = 0.01
    checkpoint_target_arrival = 0.15
    checkpoint_min_relay_weight = 1.0
    goal_map_route_selection_support_fraction = 0.15
    goal_map_route_selection_overlap_threshold = 0.08
    goal_map_use_denoised_experience_graph = True
    goal_map_use_segmented_checkpoint_histories = True

    multiscale = ["small", "medium", "large"]
    small = ["small"]
    medium = ["medium"]
    large = ["large"]

    scale_names = multiscale # what scales you are using
    run_time_hours = 15
    max_dist = 25
    plot_bvc = False

    enable_ojas = False
    enable_stdp = False

    # Multi-goal configuration for learning and exploitation
    multi_goal_config = {
        "type": "multi",
        "goals": [
            {"name": "red", "location": [7, 7], "radius": 0.7},
            {"name": "green", "location": [-7, 7], "radius": 0.7},
            {"name": "blue", "location": [7, -7], "radius": 0.7},
            {"name": "yellow", "location": [-7, -7], "radius": 0.7}
        ]
    }

    # Coverage parameters for LEARN_LOCATIONS_COVERAGE
    environment_size = [20.0, 20.0]  # 20x20 meter environment
    grid_size = 0.5  # 0.5 meter grid cells
    coverage_percentage = 0.95  # 95% coverage target
    min_goal_visits = 10  # Minimum visits to the single goal before reward map is built
    min_checkpoint_visits = 3  # Require a few directed crossings per checkpoint direction before ending training
    checkpoint_visit_radius = 0.8
    minimum_learning_duration_hours = 4.0
    goal_assoc_require_all_scales = False

    # Random spawn parameters for EXPLOIT_LOCATIONS_RANDOM
    min_spawn_distance = 6.0  # 6 meters from goal
    wall_clearance = 0.5  # 0.5 meter clearance from walls
    generate_path_plots = False  # Whether to generate path visualization plots
    trials_per_goal = 20  # 20 trials per goal
    path_failure_ratio = 10.0  # Fail if robot travels 2x optimal distance

    # Auto trial parameters (shared by both LEARN and EXPLOIT AUTO modes)
    auto_trial_name = "GC_FIXED"  # Trial set name (creates pkl_{name} and stats_{name} folders)
    num_auto_trials = 5  # Number of trials to run automatically


    MODE_PARAMS = {
        "LEARN_OJAS": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": plot_bvc,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "LEARN_HEBB": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_config": None,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": plot_bvc,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            "min_checkpoint_visits": min_checkpoint_visits,
            "checkpoint_visit_radius": checkpoint_visit_radius,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "DMTP": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": plot_bvc,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "EXPLOIT": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "td_learning": td_learning,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "EXPLOIT_SAVE": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "num_loops": 20,
            "save_data": True,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "LEARNING_SAVE": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 51,
            "save_data": True,
            "td_learning": td_learning,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "PLOTTING": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": plot_bvc,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "REBUILD_REWARD_MAP": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_config": None,
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": 0.01,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": False,
            "use_unified_multiscale": use_unified_multiscale,
            "two_phase_learning": two_phase_learning,
            "defer_experience_build_until_phase2_end": defer_experience_build_until_phase2_end,
        },
        "LEARN_LOCATIONS_COVERAGE": {
            "corners": [[0, 0]],  # Single starting location for learning
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            "min_checkpoint_visits": min_checkpoint_visits,
            "checkpoint_visit_radius": checkpoint_visit_radius,
            "use_unified_multiscale": use_unified_multiscale,
            "two_phase_learning": True,
            "phase1_min_steps": phase1_min_steps,
            "phase1_max_steps": phase1_max_steps,
            "phase1_bin_size": phase1_bin_size,
            "phase1_min_revisit_bins": phase1_min_revisit_bins,
            "phase1_revisit_cosine_threshold": phase1_revisit_cosine_threshold,
            "phase1_revisit_window": phase1_revisit_window,
            "defer_experience_build_until_phase2_end": defer_experience_build_until_phase2_end,
        },
        "LEARN_LOCATIONS_ADAPTIVE": {
            "corners": [[0, 0]],
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            "min_checkpoint_visits": min_checkpoint_visits,
            "checkpoint_visit_radius": checkpoint_visit_radius,
            "use_unified_multiscale": use_unified_multiscale,
        },
        "LEARN_LOCATIONS_TWO_PHASE": {
            "corners": [[0, 0]],
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            "min_checkpoint_visits": min_checkpoint_visits,
            "checkpoint_visit_radius": checkpoint_visit_radius,
            "use_unified_multiscale": use_unified_multiscale,
            "two_phase_learning": True,
            "phase1_min_steps": phase1_min_steps,
            "phase1_max_steps": phase1_max_steps,
            "phase1_bin_size": phase1_bin_size,
            "phase1_min_revisit_bins": phase1_min_revisit_bins,
            "phase1_revisit_cosine_threshold": phase1_revisit_cosine_threshold,
            "phase1_revisit_window": phase1_revisit_window,
            "defer_experience_build_until_phase2_end": defer_experience_build_until_phase2_end,
        },
        "LEARN_LOCATIONS_COVERAGE_AUTO": {
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            "min_checkpoint_visits": min_checkpoint_visits,
            "checkpoint_visit_radius": checkpoint_visit_radius,
            "use_unified_multiscale": use_unified_multiscale,
            "two_phase_learning": True,
            "phase1_min_steps": phase1_min_steps,
            "phase1_max_steps": phase1_max_steps,
            "phase1_bin_size": phase1_bin_size,
            "phase1_min_revisit_bins": phase1_min_revisit_bins,
            "phase1_revisit_cosine_threshold": phase1_revisit_cosine_threshold,
            "phase1_revisit_window": phase1_revisit_window,
            "defer_experience_build_until_phase2_end": defer_experience_build_until_phase2_end,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        },
        "EXPLOIT_LOCATIONS_RANDOM": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": 5.0,  # High fallback time limit
            "max_dist": max_dist,
            "td_learning": td_learning,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": "exploit_locations",
            "save_data": True,
            "goal_config": None,  # set dynamically from world name at runtime
            "trial_config": {
                "type": "random_spawns",
                "trials_per_goal": trials_per_goal
            },
            # Random spawn specific parameters
            "trials_per_goal": trials_per_goal,
            "min_spawn_distance": min_spawn_distance,
            "wall_clearance": wall_clearance,
            "path_failure_ratio": path_failure_ratio,
            "generate_path_plots": generate_path_plots,
            "use_unified_multiscale": use_unified_multiscale,
            # Lightweight logging for fast exploit initialization + goal-save
            "lightweight_hmaps": True,
            "hmap_sample_stride": 10,
            "hmap_topk": 8,
        },
        "EXPLOIT_LOCATIONS_RANDOM_AUTO": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": 5.0,  # High fallback time limit
            "max_dist": max_dist,
            "td_learning": td_learning,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "goal_config": None,  # set dynamically from world name at runtime
            # Random spawn specific parameters
            "trials_per_goal": trials_per_goal,
            "min_spawn_distance": min_spawn_distance,
            "wall_clearance": wall_clearance,
            "path_failure_ratio": path_failure_ratio,
            "generate_path_plots": generate_path_plots,
            "use_unified_multiscale": use_unified_multiscale,
            # Lightweight logging for fast exploit initialization + goal-save
            "lightweight_hmaps": True,
            "hmap_sample_stride": 10,
            "hmap_topk": 8,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        },
        "PLOTTING_AUTO": {
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": plot_bvc,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "use_unified_multiscale": use_unified_multiscale,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        },
        "PLOTTING_COVERAGE_AUTO": {
            "start_loc": start_loc,
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": 24,  # High fallback time limit for coverage-based stopping
            "td_learning": False,
            "use_prox_mod": False,
            "plot_bvc": plot_bvc,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "use_unified_multiscale": use_unified_multiscale,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        }
    }

    # Apply shared defaults across all modes from one place.
    # Do not overwrite mode-specific two-phase settings.
    for _mode_name in MODE_PARAMS:
        MODE_PARAMS[_mode_name].setdefault("min_goal_visits", min_goal_visits)
        MODE_PARAMS[_mode_name].setdefault("min_checkpoint_visits", min_checkpoint_visits)
        MODE_PARAMS[_mode_name].setdefault("checkpoint_visit_radius", checkpoint_visit_radius)
        MODE_PARAMS[_mode_name].setdefault("minimum_learning_duration_hours", minimum_learning_duration_hours)
        MODE_PARAMS[_mode_name].setdefault("goal_assoc_require_all_scales", goal_assoc_require_all_scales)
        MODE_PARAMS[_mode_name]["proximity_mode"] = proximity_mode
        MODE_PARAMS[_mode_name]["proximity_pair_percentile"] = proximity_pair_percentile
        MODE_PARAMS[_mode_name]["pcn_gate_mode"] = pcn_gate_mode
        MODE_PARAMS[_mode_name]["pcn_learning_adaptation_mode"] = pcn_learning_adaptation_mode
        MODE_PARAMS[_mode_name]["pcn_learning_stdp_start_steps"] = pcn_learning_stdp_start_steps
        MODE_PARAMS[_mode_name]["pcn_learning_stop_ojas_on_stabilization"] = pcn_learning_stop_ojas_on_stabilization
        MODE_PARAMS[_mode_name]["pcn_learning_cross_scale_coupling_start_steps"] = pcn_learning_cross_scale_coupling_start_steps
        MODE_PARAMS[_mode_name]["pcn_learning_cross_scale_coupling_ramp_steps"] = pcn_learning_cross_scale_coupling_ramp_steps
        MODE_PARAMS[_mode_name]["pcn_learning_cross_scale_coupling_min"] = pcn_learning_cross_scale_coupling_min
        MODE_PARAMS[_mode_name]["pcn_grid_balance_modalities"] = pcn_grid_balance_modalities
        MODE_PARAMS[_mode_name]["pcn_soft_scale_overlap"] = pcn_soft_scale_overlap
        MODE_PARAMS[_mode_name]["pcn_soft_scale_overlap_in_learning"] = pcn_soft_scale_overlap_in_learning
        MODE_PARAMS[_mode_name]["pcn_soft_scale_gate_floor"] = pcn_soft_scale_gate_floor
        MODE_PARAMS[_mode_name]["pcn_soft_scale_gate_floor_in_learning"] = pcn_soft_scale_gate_floor_in_learning
        MODE_PARAMS[_mode_name]["pcn_soft_cross_inhibition_scale"] = pcn_soft_cross_inhibition_scale
        MODE_PARAMS[_mode_name]["pcn_soft_cross_inhibition_scale_in_learning"] = pcn_soft_cross_inhibition_scale_in_learning
        MODE_PARAMS[_mode_name]["pcn_soft_cross_inhibition_cap"] = pcn_soft_cross_inhibition_cap
        MODE_PARAMS[_mode_name]["gcn_scale_invariant"] = gcn_scale_invariant
        MODE_PARAMS[_mode_name]["gcn_use_adaptive_minmax"] = gcn_use_adaptive_minmax
        MODE_PARAMS[_mode_name]["goal_assoc_unique_topk"] = goal_assoc_unique_topk
        MODE_PARAMS[_mode_name]["goal_assoc_max_activation_drop"] = goal_assoc_max_activation_drop
        MODE_PARAMS[_mode_name]["goal_assoc_locality_radius"] = goal_assoc_locality_radius
        MODE_PARAMS[_mode_name]["goal_assoc_locality_max_activation_drop"] = goal_assoc_locality_max_activation_drop
        MODE_PARAMS[_mode_name]["goal_map_use_compact_pc_gate"] = goal_map_use_compact_pc_gate
        MODE_PARAMS[_mode_name]["goal_map_pc_compactness_peak_fraction"] = goal_map_pc_compactness_peak_fraction
        MODE_PARAMS[_mode_name]["goal_map_pc_compactness_min_peak"] = goal_map_pc_compactness_min_peak
        MODE_PARAMS[_mode_name]["goal_map_pc_compactness_max_radius"] = goal_map_pc_compactness_max_radius
        MODE_PARAMS[_mode_name]["goal_map_fill_source_envelope_tau"] = goal_map_fill_source_envelope_tau
        MODE_PARAMS[_mode_name]["goal_map_fill_source_envelope_floor"] = goal_map_fill_source_envelope_floor
        MODE_PARAMS[_mode_name]["ablation_force_nonunified_baseline"] = ablation_force_nonunified_baseline
        MODE_PARAMS[_mode_name]["ablation_legacy_grid_runtime"] = ablation_legacy_grid_runtime
        MODE_PARAMS[_mode_name]["ablation_disable_unified_scale_selection"] = ablation_disable_unified_scale_selection
        MODE_PARAMS[_mode_name]["ablation_disable_unified_cross_scale_learning"] = ablation_disable_unified_cross_scale_learning
        MODE_PARAMS[_mode_name]["ablation_disable_unified_learning_adaptation"] = ablation_disable_unified_learning_adaptation
        MODE_PARAMS[_mode_name]["ablation_disable_unified_dopt_jitter"] = ablation_disable_unified_dopt_jitter
        MODE_PARAMS[_mode_name]["ablation_disable_unified_large_scale_plateau"] = ablation_disable_unified_large_scale_plateau
        MODE_PARAMS[_mode_name]["record_experience_transitions"] = record_experience_transitions
        MODE_PARAMS[_mode_name].setdefault("two_phase_learning", two_phase_learning)
        MODE_PARAMS[_mode_name].setdefault("phase1_min_steps", phase1_min_steps)
        MODE_PARAMS[_mode_name].setdefault("phase1_max_steps", phase1_max_steps)
        MODE_PARAMS[_mode_name].setdefault("phase1_bin_size", phase1_bin_size)
        MODE_PARAMS[_mode_name].setdefault("phase1_min_revisit_bins", phase1_min_revisit_bins)
        MODE_PARAMS[_mode_name].setdefault("phase1_revisit_cosine_threshold", phase1_revisit_cosine_threshold)
        MODE_PARAMS[_mode_name].setdefault("phase1_revisit_window", phase1_revisit_window)
        MODE_PARAMS[_mode_name].setdefault("defer_experience_build_until_phase2_end", defer_experience_build_until_phase2_end)
        MODE_PARAMS[_mode_name]["goal_map_replay_timesteps"] = goal_map_replay_timesteps
        MODE_PARAMS[_mode_name]["goal_map_paper_replay_tau"] = goal_map_paper_replay_tau
        MODE_PARAMS[_mode_name]["goal_map_goal_replay_event_count"] = (
            goal_map_goal_replay_event_count
        )
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_replay_event_count"] = (
            goal_map_checkpoint_replay_event_count
        )
        MODE_PARAMS[_mode_name]["goal_map_replay_event_merge_gap_steps"] = (
            goal_map_replay_event_merge_gap_steps
        )
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_room_support_min_fraction"] = (
            goal_map_checkpoint_room_support_min_fraction
        )
        MODE_PARAMS[_mode_name]["hybrid_path_replay_weight"] = hybrid_path_replay_weight
        MODE_PARAMS[_mode_name]["hybrid_diffusion_replay_weight"] = hybrid_diffusion_replay_weight
        MODE_PARAMS[_mode_name]["goal_map_path_replay_weight"] = goal_map_path_replay_weight
        MODE_PARAMS[_mode_name]["goal_map_diffusion_replay_weight"] = goal_map_diffusion_replay_weight
        MODE_PARAMS[_mode_name]["goal_map_experience_transition_topk"] = goal_map_experience_transition_topk
        MODE_PARAMS[_mode_name]["unified_reward_normalization_mode"] = unified_reward_normalization_mode
        MODE_PARAMS[_mode_name]["unified_goal_map_mode"] = unified_goal_map_mode
        MODE_PARAMS[_mode_name]["goal_map_room_normalization_mode"] = goal_map_room_normalization_mode
        MODE_PARAMS[_mode_name]["goal_map_path_topk"] = goal_map_path_topk
        MODE_PARAMS[_mode_name]["goal_map_path_decay"] = goal_map_path_decay
        MODE_PARAMS[_mode_name]["goal_map_path_frontier_only"] = goal_map_path_frontier_only
        MODE_PARAMS[_mode_name]["goal_map_neighbor_topk"] = goal_map_neighbor_topk
        MODE_PARAMS[_mode_name]["goal_map_neighbor_steps"] = goal_map_neighbor_steps
        MODE_PARAMS[_mode_name]["goal_map_neighbor_decay"] = goal_map_neighbor_decay
        MODE_PARAMS[_mode_name]["goal_map_neighbor_frontier_only"] = goal_map_neighbor_frontier_only
        MODE_PARAMS[_mode_name]["goal_map_neighbor_seed_scale"] = goal_map_neighbor_seed_scale
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_mode"] = goal_map_goal_seed_mode
        MODE_PARAMS[_mode_name]["goal_map_goal_replay_retention_mode"] = goal_map_goal_replay_retention_mode
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_post_contact_steps"] = goal_map_goal_seed_post_contact_steps
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_within_visit_aggregation"] = goal_map_goal_seed_within_visit_aggregation
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_across_visit_aggregation"] = goal_map_goal_seed_across_visit_aggregation
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_state_normalization"] = goal_map_goal_seed_state_normalization
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_self_retention_alpha"] = goal_map_goal_seed_self_retention_alpha
        MODE_PARAMS[_mode_name]["goal_map_path_constraint_sigma"] = goal_map_path_constraint_sigma
        MODE_PARAMS[_mode_name]["goal_map_path_constraint_topk"] = goal_map_path_constraint_topk
        MODE_PARAMS[_mode_name]["goal_map_local_constraint_sigma"] = goal_map_local_constraint_sigma
        MODE_PARAMS[_mode_name]["goal_map_local_constraint_topk"] = goal_map_local_constraint_topk
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_center_radius_factor"] = goal_map_goal_seed_center_radius_factor
        MODE_PARAMS[_mode_name]["goal_map_goal_seed_center_min_radius"] = goal_map_goal_seed_center_min_radius
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_seed_center_radius_factor"] = goal_map_checkpoint_seed_center_radius_factor
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_seed_center_min_radius"] = goal_map_checkpoint_seed_center_min_radius
        MODE_PARAMS[_mode_name]["goal_map_spatial_obstacle_block"] = goal_map_spatial_obstacle_block
        MODE_PARAMS[_mode_name]["goal_map_obstacle_margin"] = goal_map_obstacle_margin
        MODE_PARAMS[_mode_name]["goal_map_use_checkpoint_relays"] = goal_map_use_checkpoint_relays
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_mode"] = goal_map_checkpoint_mode
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_scale"] = goal_map_checkpoint_scale
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_support_mode"] = goal_map_checkpoint_support_mode
        MODE_PARAMS[_mode_name]["goal_map_checkpoint_stop_parent_replay"] = goal_map_checkpoint_stop_parent_replay
        MODE_PARAMS[_mode_name]["checkpoint_boost_threshold"] = checkpoint_boost_threshold
        MODE_PARAMS[_mode_name]["checkpoint_target_arrival"] = checkpoint_target_arrival
        MODE_PARAMS[_mode_name]["checkpoint_min_relay_weight"] = checkpoint_min_relay_weight
        MODE_PARAMS[_mode_name]["goal_map_route_selection_support_fraction"] = goal_map_route_selection_support_fraction
        MODE_PARAMS[_mode_name]["goal_map_route_selection_overlap_threshold"] = goal_map_route_selection_overlap_threshold
        MODE_PARAMS[_mode_name]["goal_map_use_denoised_experience_graph"] = goal_map_use_denoised_experience_graph
        MODE_PARAMS[_mode_name]["goal_map_use_segmented_checkpoint_histories"] = goal_map_use_segmented_checkpoint_histories
        MODE_PARAMS[_mode_name]["unified_preplay_scale_arbitration"] = unified_preplay_scale_arbitration
        MODE_PARAMS[_mode_name]["unified_preplay_scale_prior_mix"] = unified_preplay_scale_prior_mix
        MODE_PARAMS[_mode_name]["goal_map_smoothing_sigma"] = goal_map_smoothing_sigma
        MODE_PARAMS[_mode_name]["goal_map_smoothing_topk"] = goal_map_smoothing_topk
        MODE_PARAMS[_mode_name]["goal_map_smoothing_passes"] = goal_map_smoothing_passes
        MODE_PARAMS[_mode_name]["goal_map_smoothing_mix"] = goal_map_smoothing_mix
        MODE_PARAMS[_mode_name]["unified_preplay_commit_threshold"] = unified_preplay_commit_threshold
        MODE_PARAMS[_mode_name]["preplay_heading_selection_mode"] = preplay_heading_selection_mode
        MODE_PARAMS[_mode_name]["unified_exploit_action_mode"] = unified_exploit_action_mode
        MODE_PARAMS[_mode_name]["unified_transition_preplay_enabled"] = unified_transition_preplay_enabled
        MODE_PARAMS[_mode_name]["unified_spatial_fallback_enabled"] = unified_spatial_fallback_enabled
        MODE_PARAMS[_mode_name]["unified_spatial_override_margin"] = unified_spatial_override_margin
        MODE_PARAMS[_mode_name]["unified_spatial_lookahead_distance"] = unified_spatial_lookahead_distance
        MODE_PARAMS[_mode_name]["unified_spatial_lookahead_sigma"] = unified_spatial_lookahead_sigma
        MODE_PARAMS[_mode_name]["unified_exploit_step_fraction"] = unified_exploit_step_fraction
        MODE_PARAMS[_mode_name]["exploit_direction_clearance_percentile"] = exploit_direction_clearance_percentile
        MODE_PARAMS[_mode_name]["prune_experience_loops"] = prune_experience_loops
        MODE_PARAMS[_mode_name]["loop_prune_min_top1"] = loop_prune_min_top1
        MODE_PARAMS[_mode_name]["loop_prune_min_top1_to_top2_ratio"] = loop_prune_min_top1_to_top2_ratio

    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_adaptation_mode"] = "gaussian_post_competition_expression"
    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_stop_ojas_on_stabilization"] = True
    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_stdp_start_steps"] = 0
    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_cross_scale_coupling_start_steps"] = 0
    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_cross_scale_coupling_ramp_steps"] = 0
    MODE_PARAMS["LEARN_LOCATIONS_ADAPTIVE"]["pcn_learning_cross_scale_coupling_min"] = 1.0

    if SELECTED_MODE not in MODE_PARAMS or SELECTED_MODE not in MODES_MAP:
        print("Invalid mode selected.")
        sys.exit(1)

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[SELECTED_MODE]
    params = MODE_PARAMS[SELECTED_MODE]

    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)
