"""Model/trial configuration for multiscale_grid_controller_v2."""

import math
from typing import Any, Dict, List

from core.layers.reward_cell_layer_test import C_LAMBDA
from robot_modes_v2 import RobotModeV2

STEPS_PER_LAMBDA = 8


def _steps_for_sigma(sigma_pc_s: float) -> int:
    return int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA))


# ---------------------------------------------------------------------------
# QUICK-EDIT RUN SETTINGS (frequently changed)
# ---------------------------------------------------------------------------
MODES_MAP = {
    "LEARN_OJAS": RobotModeV2.LEARN_OJAS,
    "LEARN_HEBB": RobotModeV2.LEARN_HEBB,
    "LEARN_LOCATIONS": RobotModeV2.LEARN_LOCATIONS,
    "PLOTTING": RobotModeV2.PLOTTING,
    "BENCHMARK": RobotModeV2.BENCHMARK,
}

SUPPORTED_V2_MODE_KEYS = list(MODES_MAP.keys())

# Main mode selection
DEFAULT_SELECTED_MODE = "BENCHMARK"
# Execution is configured separately from behavior mode.
# Example batch config:
# {"type": "auto", "series_name": "trial_batch_a", "num_trials": 5, "resume_completed": True}
DEFAULT_EXECUTION_CONFIG = {"type": "single"}

# Shared defaults used by most modes
DEFAULT_START_LOC = [1, 1]
DEFAULT_SCALE_NAMES = ["small", "medium", "large"]
DEFAULT_RUN_TIME_HOURS = 15
DEFAULT_MAX_DIST = 25
DEFAULT_RUNTIME_STEP_MODE = "hybrid"
DEFAULT_ENVIRONMENT_SIZE = [20.0, 20.0]
DEFAULT_GRID_SIZE = 0.5
DEFAULT_COVERAGE_PERCENTAGE = 0.95

# Global non-exploit stop toggle:
# - "time": stop by run_time_hours
# - "coverage": stop by target coverage
# - "time_or_coverage": either condition
GLOBAL_NON_EXPLOIT_STOP_CONDITION = "coverage"


# ---------------------------------------------------------------------------
# SCALE PRESETS (lower-frequency edits)
# ---------------------------------------------------------------------------
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "sigma_pc_s": 1.0,
        "num_pc": 2000,
        "sigma_r": 0.5,
        "sigma_theta": 1.0,
        "rcn_learning_rate": 0.1,
        "num_bvc_per_dir": 100,
        "w_in_init_ratio": 0.3,
        "w_grid_init_ratio": 0.3,
        "gamma_pp": 1.0,
        "gamma_pb": 0.35,
        "grid_influence": 0.3,
        "gamma_pg": 0.35,
        "num_grid_cells": 800,
        "num_modules": 8,
        "cells_per_module": 100,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 4.0,
        "translation_scale": 2.0,
        "mask_resolution": 256,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "alpha_pb": 0.447,
        "alpha_pg": 0.447,
        "enable_correlation_weighting": False,
        "correlation_window": 8,
        "correlation_update_freq": 1,
        "correlation_scaling": 10.0,
        "min_correlation_weight": 0.10,
        "correlation_threshold": 0.005,
        "adjacency_learning_mode": "topk",
        "adjacency_topk": 256,
        "adjacency_activity_floor": 0.02,
        "replay_timesteps": 100,
        "replay_decay_factor": 3,
        "custom_replay_timesteps": _steps_for_sigma(1.0),
        "initial_value_multiplier": 2.0,
    },
    "medium": {
        "scale_index": 1,
        "name": "medium",
        "sigma_pc_s": 1.5,
        "num_pc": 1000,
        "sigma_r": 1.0,
        "sigma_theta": 3.0,
        "rcn_learning_rate": 0.1,
        "num_bvc_per_dir": 50,
        "w_in_init_ratio": 0.25,
        "w_grid_init_ratio": 0.3,
        "gamma_pp": 0.7,
        "gamma_pb": 0.35,
        "grid_influence": 0.35,
        "gamma_pg": 0.35,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 5.5,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "alpha_pb": 0.632,
        "alpha_pg": 0.632,
        "enable_correlation_weighting": False,
        "correlation_window": 12,
        "correlation_update_freq": 1,
        "correlation_scaling": 10.0,
        "min_correlation_weight": 0.10,
        "correlation_threshold": 0.05,
        "adjacency_learning_mode": "topk",
        "adjacency_topk": 128,
        "adjacency_activity_floor": 0.02,
        "replay_timesteps": 40,
        "replay_decay_factor": 8,
        "custom_replay_timesteps": _steps_for_sigma(1.5),
        "initial_value_multiplier": 1.5,
    },
    "large": {
        "scale_index": 2,
        "name": "large",
        "sigma_pc_s": 3.0,
        "num_pc": 500,
        "sigma_r": 1.5,
        "sigma_theta": 5.0,
        "rcn_learning_rate": 0.1,
        "num_bvc_per_dir": 75,
        "w_in_init_ratio": 0.2,
        "w_grid_init_ratio": 0.2,
        "gamma_pp": 0.7,
        "gamma_pb": 0.25,
        "grid_influence": 0.35,
        "gamma_pg": 0.25,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 7.0,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "alpha_pb": 0.632,
        "alpha_pg": 0.632,
        "enable_correlation_weighting": False,
        "correlation_window": 18,
        "correlation_update_freq": 1,
        "correlation_scaling": 10.0,
        "min_correlation_weight": 0.10,
        "correlation_threshold": 0.05,
        "adjacency_learning_mode": "topk",
        "adjacency_topk": 64,
        "adjacency_activity_floor": 0.02,
        "replay_timesteps": 20,
        "replay_decay_factor": 15,
        "custom_replay_timesteps": _steps_for_sigma(3.0),
        "initial_value_multiplier": 1.2,
    },
}


def compile_scales(scale_names: List[str], use_grid: bool = True) -> List[Dict[str, Any]]:
    scales: List[Dict[str, Any]] = []
    for name in scale_names:
        if name not in SCALE_PRESETS:
            raise ValueError(f"Unknown scale '{name}'. Available: {list(SCALE_PRESETS.keys())}")
        scale = dict(SCALE_PRESETS[name])
        if not use_grid:
            scale["num_grid_cells"] = 0
            scale["grid_influence"] = 0.0
        scales.append(scale)
    return scales


def default_goal_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "explore": {
            "type": "multi",
            "goals": [
                {"name": "red", "location": [7, 7], "radius": 0.7},
                {"name": "green", "location": [-7, 7], "radius": 0.7},
                {"name": "blue", "location": [7, -7], "radius": 0.7},
                {"name": "yellow", "location": [-7, -7], "radius": 0.7},
            ],
        },
        "exploit": {
            "type": "multi",
            "goals": [
                {"name": "red", "location": [7, 7], "radius": 1.0},
                {"name": "green", "location": [-7, 7], "radius": 1.0},
                {"name": "blue", "location": [7, -7], "radius": 1.0},
                {"name": "yellow", "location": [-7, -7], "radius": 1.0},
            ],
        },
    }


def default_mode_params() -> Dict[str, Dict[str, Any]]:
    goals = default_goal_configs()

    common = {
        "start_loc": list(DEFAULT_START_LOC),
        "scale_names": list(DEFAULT_SCALE_NAMES),
        "run_time_hours": DEFAULT_RUN_TIME_HOURS,
        "max_dist": DEFAULT_MAX_DIST,
        "plot_bvc": False,
        "td_learning": False,
        "use_prox_mod": False,
        # Runtime stepping policy:
        # - "legacy": preserves old multi-step-per-iteration behavior
        # - "hybrid": keeps legacy step budget but avoids extra sensor-processing on forward() path
        "runtime_step_mode": DEFAULT_RUNTIME_STEP_MODE,
        "non_exploit_stop_condition": GLOBAL_NON_EXPLOIT_STOP_CONDITION,
        "coverage_free_space_only": True,
        "goal_constraint_mode": "none",
        "execution_config": dict(DEFAULT_EXECUTION_CONFIG),
    }

    return {
        "LEARN_OJAS": {
            **common,
            "enable_ojas": True,
            "enable_stdp": False,
            "goal_constraint_mode": "none",
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "environment_size": list(DEFAULT_ENVIRONMENT_SIZE),
            "grid_size": DEFAULT_GRID_SIZE,
            "coverage_percentage": DEFAULT_COVERAGE_PERCENTAGE,
            "non_exploit_stop_condition": "time_or_coverage",
            "save_data": False,
        },
        "LEARN_HEBB": {
            **common,
            "enable_ojas": False,
            "enable_stdp": True,
            "goal_constraint_mode": "none",
            "goal_config": None,
            "trial_config": {"type": "simple", "count": 1},
            "environment_size": list(DEFAULT_ENVIRONMENT_SIZE),
            "grid_size": DEFAULT_GRID_SIZE,
            "coverage_percentage": DEFAULT_COVERAGE_PERCENTAGE,
            "non_exploit_stop_condition": "time_or_coverage",
            "save_data": False,
        },
        "LEARN_LOCATIONS": {
            **common,
            "enable_ojas": True,
            "enable_stdp": True,
            "goal_constraint_mode": "learning",
            "goal_config": goals["explore"],
            "trial_config": {"type": "simple", "count": 1},
            "environment_size": list(DEFAULT_ENVIRONMENT_SIZE),
            "grid_size": DEFAULT_GRID_SIZE,
            "coverage_percentage": DEFAULT_COVERAGE_PERCENTAGE,
            "min_goal_visits": 10,
            "save_data": False,
        },
        "PLOTTING": {
            **common,
            "enable_ojas": False,
            "enable_stdp": False,
            "trial_config": {"type": "simple", "count": 1},
            "goal_constraint_mode": "none",
            "goal_config": goals["explore"],
            "save_data": False,
        },
        "BENCHMARK": {
            **common,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": 2,
            "trial_config": {"type": "simple", "count": 1},
            "goal_config": None,
            "non_exploit_stop_condition": "time",
            "goal_constraint_mode": "none",
            "ignore_goals": True,
            "track_runtime_metrics": True,
            "save_data": False,
        },
    }
