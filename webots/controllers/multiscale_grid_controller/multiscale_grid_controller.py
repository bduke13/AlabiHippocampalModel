"""Multiscale Controller Test - Enhanced with LEARN_LOCATIONS_COVERAGE and EXPLOIT_LOCATIONS_RANDOM modes"""

import sys
import os
import re
import gc
import json
import hashlib
import math
import copy
import shutil
import torch
from pathlib import Path
from datetime import datetime

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHARED_SPAWN_MANIFEST_VERSION = 1
SHARED_SPAWN_MANIFEST_ROOT = PROJECT_ROOT.parent / "shared_navigation_spawns"
sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from core.layers.reward_cell_layer_test import C_LAMBDA
from multiscale_grid_driver import Driver
from core.robot.robot_mode import RobotMode
from core.robot.webots_worlds import (
    build_goal_config_from_world,
    get_world_agent_start,
    get_world_size,
)
from analysis.stats.stats_collector import stats_collector
from path_planning import generate_spawn_locations, calculate_optimal_paths, save_path_visualizations
from preplay_defaults import UNIFIED_PREPLAY_DEFAULTS

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


def _goal_entries_from_config(goal_config):
    """Return a normalized goal list from either single- or multi-goal config."""
    if not goal_config:
        return []

    goal_type = goal_config.get("type")
    if goal_type == "single":
        if "location" not in goal_config:
            return []
        return [
            {
                "name": goal_config.get("name", "default"),
                "location": goal_config["location"],
                "radius": goal_config.get("radius", 1.0),
            }
        ]

    if goal_type == "multi":
        return [
            {
                "name": goal["name"],
                "location": goal["location"],
                "radius": goal.get("radius", 1.0),
            }
            for goal in goal_config.get("goals", [])
        ]

    return [
        {
            "name": goal["name"],
            "location": goal["location"],
            "radius": goal.get("radius", 1.0),
        }
        for goal in goal_config.get("goals", [])
    ]


def _clone_goal_config(goal_config):
    return copy.deepcopy(goal_config) if goal_config else goal_config


def _probe_runtime_world_defaults(bot):
    world_name = get_world_name(bot)
    goal_config = build_goal_config_from_world(world_name)
    agent_start = get_world_agent_start(world_name)
    environment_size = get_world_size(world_name)
    return {
        "world_name": world_name,
        "goal_config": goal_config,
        "agent_start": agent_start,
        "environment_size": environment_size,
    }


def _mode_requires_goals(mode):
    return mode in {
        RobotMode.LEARN_LOCATIONS_COVERAGE,
        RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        RobotMode.EXPLOIT_LOCATIONS_RANDOM,
        RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        RobotMode.PLOTTING_AUTO,
        RobotMode.PLOTTING_COVERAGE_AUTO,
    }


def _mode_supports_multi_goal_world(mode):
    return mode in {
        RobotMode.LEARN_LOCATIONS_COVERAGE,
        RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        RobotMode.EXPLOIT_LOCATIONS_RANDOM,
        RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        RobotMode.PLOTTING_AUTO,
        RobotMode.PLOTTING_COVERAGE_AUTO,
    }


def _collapse_to_single_goal(goal_config):
    goals = _goal_entries_from_config(goal_config)
    if not goals:
        return None
    goal = goals[0]
    return {
        "type": "single",
        "name": goal["name"],
        "location": goal["location"],
        "radius": goal["radius"],
    }


def _apply_world_defaults(mode, kwargs, bot):
    """Fill in world-derived defaults so controller configs need less manual editing."""
    normalized = dict(kwargs)
    inferred = None

    def ensure_inferred():
        nonlocal inferred
        if inferred is None:
            inferred = _probe_runtime_world_defaults(bot)
        return inferred

    if normalized.get("goal_config") is None:
        inferred_goal_config = ensure_inferred().get("goal_config")
        if inferred_goal_config is not None:
            if not _mode_supports_multi_goal_world(mode):
                inferred_goal_config = _collapse_to_single_goal(inferred_goal_config)
            normalized["goal_config"] = inferred_goal_config

    if normalized.get("environment_size") is None:
        normalized["environment_size"] = ensure_inferred().get("environment_size")

    if normalized.get("start_loc") is None:
        inferred_start = ensure_inferred().get("agent_start")
        if inferred_start is not None:
            normalized["start_loc"] = inferred_start

    if normalized.get("goal_location") is None:
        inferred_goals = _goal_entries_from_config(normalized.get("goal_config"))
        if inferred_goals:
            normalized["goal_location"] = inferred_goals[0]["location"]

    if _mode_requires_goals(mode) and not _goal_entries_from_config(normalized.get("goal_config")):
        world_name = ensure_inferred().get("world_name", "unknown")
        raise ValueError(
            f"No goals could be inferred from world '{world_name}'. "
            f"Add goal markers to the .wbt or provide goal_config explicitly."
        )

    return normalized

def _sanitize_run_component(value):
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return text or "default"

def _rounded_trial_float(value):
    return round(float(value), 6)

def _goal_signature(goals):
    return [
        {
            "name": str(goal["name"]),
            "location": [_rounded_trial_float(coord) for coord in goal["location"]],
            "radius": _rounded_trial_float(goal.get("radius", 1.0)),
        }
        for goal in sorted(goals, key=lambda entry: str(entry["name"]))
    ]

def _json_safe_trial_data(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_trial_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_trial_data(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value

def _spawn_manifest_settings(world_name, goals, trials_per_goal, min_spawn_distance, wall_clearance):
    return {
        "world_name": world_name,
        "trials_per_goal": int(trials_per_goal),
        "min_spawn_distance": _rounded_trial_float(min_spawn_distance),
        "wall_clearance": _rounded_trial_float(wall_clearance),
        "goals": _goal_signature(goals),
    }

def _spawn_manifest_path(settings, manifest_path=None):
    if manifest_path:
        return Path(manifest_path)

    signature_text = json.dumps(settings, sort_keys=True)
    digest = hashlib.sha1(signature_text.encode("utf-8")).hexdigest()[:10]
    filename = (
        f"{_sanitize_run_component(settings['world_name'])}_"
        f"trials-{settings['trials_per_goal']}_spawns-{digest}.json"
    )
    return SHARED_SPAWN_MANIFEST_ROOT / filename

def _rebuild_manifest_trials(manifest, goals, trials_per_goal):
    goal_by_name = {str(goal["name"]): goal for goal in goals}
    selected_by_goal = {str(goal["name"]): {} for goal in goals}

    for combination, path_result in zip(
        manifest.get("combinations", []),
        manifest.get("path_results", []),
    ):
        goal_name = str(combination.get("goal_name", ""))
        trial_number = int(combination.get("trial_number", 0))
        if goal_name not in selected_by_goal:
            continue
        if trial_number < 1 or trial_number > trials_per_goal:
            continue
        if not path_result.get("success", False):
            continue
        selected_by_goal[goal_name][trial_number] = (combination, path_result)

    combinations = []
    path_results = []
    for goal in goals:
        goal_name = str(goal["name"])
        for trial_number in range(1, trials_per_goal + 1):
            pair = selected_by_goal[goal_name].get(trial_number)
            if pair is None:
                return None

            combination, path_result = pair
            start = [_rounded_trial_float(coord) for coord in combination["start"]]
            rebuilt_combination = {
                "start": start,
                "goal": goal_by_name[goal_name],
                "trial_number": trial_number,
                "goal_name": goal_name,
            }
            rebuilt_result = {
                "start": start,
                "goal": goal_by_name[goal_name],
                "path": path_result.get("path", []),
                "distance": float(path_result["distance"]),
                "success": True,
            }
            combinations.append(rebuilt_combination)
            path_results.append(rebuilt_result)

    return combinations, path_results

def _load_shared_spawn_manifest(manifest_path, settings, goals):
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[SPAWN_MANIFEST] Could not read {manifest_path}: {exc}. Regenerating.")
        return None

    if manifest.get("version") != SHARED_SPAWN_MANIFEST_VERSION:
        print(f"[SPAWN_MANIFEST] Version mismatch in {manifest_path}. Regenerating.")
        return None
    if manifest.get("settings") != settings:
        print(f"[SPAWN_MANIFEST] Settings mismatch in {manifest_path}. Regenerating.")
        return None

    rebuilt = _rebuild_manifest_trials(
        manifest,
        goals,
        settings["trials_per_goal"],
    )
    if rebuilt is None:
        print(f"[SPAWN_MANIFEST] Incomplete manifest {manifest_path}. Regenerating.")
        return None

    combinations, path_results = rebuilt
    print(
        f"[SPAWN_MANIFEST] Loaded {len(combinations)} shared trials "
        f"from {manifest_path}"
    )
    return combinations, path_results

def _save_shared_spawn_manifest(manifest_path, settings, combinations, path_results):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": SHARED_SPAWN_MANIFEST_VERSION,
        "generated_at": datetime.now().isoformat(),
        "settings": settings,
        "combinations": _json_safe_trial_data(combinations),
        "path_results": _json_safe_trial_data(path_results),
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"[SPAWN_MANIFEST] Saved shared trials to {manifest_path}")

def _generate_path_validated_spawn_trials(
    world_name,
    goals,
    trials_per_goal,
    min_spawn_distance,
    wall_clearance,
    max_refill_rounds,
):
    selected_by_goal = {str(goal["name"]): [] for goal in goals}
    seen_starts_by_goal = {str(goal["name"]): set() for goal in goals}

    for goal in goals:
        goal_name = str(goal["name"])
        refill_round = 0
        while (
            len(selected_by_goal[goal_name]) < trials_per_goal
            and refill_round < max_refill_rounds
        ):
            refill_round += 1
            needed = trials_per_goal - len(selected_by_goal[goal_name])
            request_count = max(needed * 2, needed + 4)

            print(
                f"[SPAWN_MANIFEST] Goal {goal_name}: collecting {needed} more "
                f"path-valid starts (round {refill_round}/{max_refill_rounds})"
            )
            spawn_locations = generate_spawn_locations(
                world_name=world_name,
                goals=[goal],
                trials_per_goal=request_count,
                min_spawn_distance=min_spawn_distance,
                wall_clearance=wall_clearance,
            )

            candidate_combinations = []
            for spawn_pos in spawn_locations.get(goal_name, []):
                start = [_rounded_trial_float(coord) for coord in spawn_pos]
                start_key = tuple(start)
                if start_key in seen_starts_by_goal[goal_name]:
                    continue
                seen_starts_by_goal[goal_name].add(start_key)
                candidate_combinations.append({
                    "start": start,
                    "goal": goal,
                    "trial_number": len(selected_by_goal[goal_name]) + len(candidate_combinations) + 1,
                    "goal_name": goal_name,
                })

            if not candidate_combinations:
                continue

            candidate_results = calculate_optimal_paths(
                world_name,
                candidate_combinations,
                wall_clearance,
            )
            accepted_this_round = 0
            for combination, path_result in zip(candidate_combinations, candidate_results):
                if not path_result.get("success", False):
                    continue

                trial_number = len(selected_by_goal[goal_name]) + 1
                accepted_combination = {
                    "start": combination["start"],
                    "goal": goal,
                    "trial_number": trial_number,
                    "goal_name": goal_name,
                }
                accepted_result = {
                    "start": combination["start"],
                    "goal": goal,
                    "path": path_result["path"],
                    "distance": float(path_result["distance"]),
                    "success": True,
                }
                selected_by_goal[goal_name].append((accepted_combination, accepted_result))
                accepted_this_round += 1
                if len(selected_by_goal[goal_name]) >= trials_per_goal:
                    break

            print(
                f"[SPAWN_MANIFEST] Goal {goal_name}: accepted {accepted_this_round}; "
                f"{len(selected_by_goal[goal_name])}/{trials_per_goal} ready"
            )

        if len(selected_by_goal[goal_name]) < trials_per_goal:
            raise RuntimeError(
                f"Unable to generate {trials_per_goal} path-valid starts for goal "
                f"'{goal_name}' after {max_refill_rounds} refill rounds. "
                f"Generated {len(selected_by_goal[goal_name])}."
            )

    combinations = []
    path_results = []
    for goal in goals:
        goal_name = str(goal["name"])
        for combination, path_result in selected_by_goal[goal_name]:
            combinations.append(combination)
            path_results.append(path_result)

    return combinations, path_results

def _get_shared_spawn_trials(
    world_name,
    goals,
    trials_per_goal,
    min_spawn_distance,
    wall_clearance,
    **kwargs,
):
    settings = _spawn_manifest_settings(
        world_name,
        goals,
        trials_per_goal,
        min_spawn_distance,
        wall_clearance,
    )
    manifest_path = _spawn_manifest_path(
        settings,
        kwargs.get("spawn_manifest_path") or kwargs.get("shared_spawn_manifest_path"),
    )

    if not kwargs.get("force_regenerate_spawn_manifest", False):
        loaded = _load_shared_spawn_manifest(manifest_path, settings, goals)
        if loaded is not None:
            return loaded

    max_refill_rounds = int(kwargs.get("spawn_manifest_max_refill_rounds", 12))
    combinations, path_results = _generate_path_validated_spawn_trials(
        world_name,
        goals,
        int(trials_per_goal),
        min_spawn_distance,
        wall_clearance,
        max_refill_rounds,
    )
    _save_shared_spawn_manifest(manifest_path, settings, combinations, path_results)
    return combinations, path_results

def _spawn_manifest_kwargs(params):
    manifest_kwargs = dict(params)
    for key in ("trials_per_goal", "min_spawn_distance", "wall_clearance"):
        manifest_kwargs.pop(key, None)
    return manifest_kwargs

def save_trial_parameters(world_name, trial_id, mode, **kwargs):
    """
    Saves trial parameters to a JSON file in the pkl folder (alongside trial data).

    Args:
        world_name: Name of the world
        trial_id: Unique identifier for the trial
        mode: Robot mode being used
        **kwargs: All parameters passed to the trial
    """
    # Determine the correct folder based on mode (matching multiscale_grid_driver.py logic)
    auto_trial_name = kwargs.get("auto_trial_name")
    current_auto_trial = kwargs.get("current_auto_trial")

    if mode in {RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
                RobotMode.PLOTTING_AUTO, RobotMode.PLOTTING_COVERAGE_AUTO} and auto_trial_name:
        # Auto trial mode: pkl_{trial_name}/{world_name}_{trial_num}/
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{current_auto_trial}"
        save_folder = os.path.join(PROJECT_ROOT, "webots", "controllers", "multiscale_grid_controller", base_folder, trial_folder)
    else:
        # Standard mode: pkl/{world_name}/
        save_folder = os.path.join(PROJECT_ROOT, "webots", "controllers", "multiscale_grid_controller", "pkl", world_name)

    os.makedirs(save_folder, exist_ok=True)

    # Get scale names and compile scale definitions
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)

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
        },
        "goal_config": kwargs.get("goal_config"),
        "trial_config": kwargs.get("trial_config"),
        "environment_parameters": {
            "environment_size": kwargs.get("environment_size"),
            "grid_size": kwargs.get("grid_size"),
            "coverage_percentage": kwargs.get("coverage_percentage"),
            "min_goal_visits": kwargs.get("min_goal_visits", 3),
        },
        "preplay_parameters": {
            "unified_recurrent_preplay_horizon": kwargs.get("unified_recurrent_preplay_horizon"),
            "unified_preplay_discount_factor": kwargs.get("unified_preplay_discount_factor"),
            "unified_preplay_within_direction_beta": kwargs.get("unified_preplay_within_direction_beta"),
            "unified_preplay_num_samples": kwargs.get("unified_preplay_num_samples"),
            "unified_preplay_sampling_temperature": kwargs.get("unified_preplay_sampling_temperature"),
            "unified_preplay_turn_offsets": kwargs.get("unified_preplay_turn_offsets"),
            "unified_preplay_normalize_transitions": kwargs.get("unified_preplay_normalize_transitions"),
            "unified_preplay_global_score_normalization": kwargs.get("unified_preplay_global_score_normalization"),
            "unified_preplay_executable_rollouts": kwargs.get("unified_preplay_executable_rollouts"),
            "unified_preplay_boundary_mode": kwargs.get(
                "unified_preplay_boundary_mode",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_boundary_mode"],
            ),
            "unified_preplay_microtrajectory_safety_margin": kwargs.get(
                "unified_preplay_microtrajectory_safety_margin"
            ),
            "unified_preplay_blocked_return_penalty": kwargs.get(
                "unified_preplay_blocked_return_penalty"
            ),
            "unified_preplay_no_reward_threshold": kwargs.get(
                "unified_preplay_no_reward_threshold",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_no_reward_threshold"],
            ),
            "unified_preplay_decision_diagnostics": kwargs.get(
                "unified_preplay_decision_diagnostics"
            ),
            "unified_preplay_decision_diagnostics_stride": kwargs.get(
                "unified_preplay_decision_diagnostics_stride"
            ),
            "exploit_loop_recovery": kwargs.get(
                "exploit_loop_recovery",
                UNIFIED_PREPLAY_DEFAULTS["exploit_loop_recovery"],
            ),
        },
        "stdp_parameters": {
            "stdp_rectified_scale_centering": kwargs.get("stdp_rectified_scale_centering", False),
            "stdp_rectified_hd_gate": kwargs.get("stdp_rectified_hd_gate", True),
            "stdp_winner_hd_gate": kwargs.get("stdp_winner_hd_gate", False),
            "stdp_min_input_mass": kwargs.get("stdp_min_input_mass", 0.0),
        },
        "exploit_activation_parameters": {
            "exploit_bvc_context_gaussian_modulation": kwargs.get(
                "exploit_bvc_context_gaussian_modulation", True
            ),
            "exploit_cross_scale_inhibition": kwargs.get(
                "exploit_cross_scale_inhibition", False
            ),
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
        },
        "debug_parameters": {
            "debug_heading_convention_test": kwargs.get("debug_heading_convention_test", False),
            "debug_heading_test_forward_steps": kwargs.get("debug_heading_test_forward_steps", 4),
            "debug_heading_test_start_loc": kwargs.get("debug_heading_test_start_loc"),
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

SCALES_DEFS_GRID = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "sigma_pc_s": 1.0,  # Place field size for scale-dependent reward propagation
        "num_pc": 2000,
        "sigma_r": 0.5, #0.5
        "mod_d_opt": 0.0,
        "mod_sigma": 0.5,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 100,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.3,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 1.0,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.6,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.35,  # BVC-to-place afferent inhibition strength, 0.35 default
        # Grid cell parameters
        "grid_influence": 0.3, # 0.3
        "gamma_pg": 0.35, # 0.3
        "num_grid_cells": 800,
        "num_modules": 8,
        "cells_per_module": 100,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 4,
        "translation_scale": 2.0,
        "mask_resolution": 256,
        "smooth_sigma": 0.5,
        # Oja's learning normalization parameters
        "alpha_pb": 0.447,  # np.sqrt(0.5) = 0.447 - BVC weight decay factor (default)
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
        "num_pc": 1000,
        "sigma_r": 1.0, #2
        "mod_d_opt": 1.3,
        "mod_sigma": 0.45,
        "sigma_theta": 3,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 50,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.25,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 1.2,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.35,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.35, #0.35
        "gamma_pg": 0.35,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 5.5,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        # Oja's learning normalization parameters
        "alpha_pb": 0.632, # 0.632 DEFAULT
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
        "num_pc": 500,
        "sigma_r": 1.5, # 2.0
        "mod_d_opt": 0.0,
        "mod_sigma": 2.0,
        "sigma_theta": 5,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 75,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.2,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.2,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 1.5,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.25,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "gamma_pg": 0.25,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 7,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
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
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.5,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.0,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.3,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "gamma_pg": 0.32,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.0, 1.0),
        "scale_multiplier": 5.0,
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

SCALES_DEFS_NO_GRID = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "sigma_pc_s": 1.0,  # Place field size for scale-dependent reward propagation
        "num_pc": 2000,
        "sigma_r": 0.5, #0.5
        "mod_d_opt": 0.0,
        "mod_sigma": 0.5,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 100,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.3,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 1.0,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.0,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.35,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.0, # 0.3
        "gamma_pg": 0.35, # 0.3
        "num_grid_cells": 800,
        "num_modules": 8,
        "cells_per_module": 100,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 4,
        "translation_scale": 2.0,
        "mask_resolution": 256,
        "smooth_sigma": 1,
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
        "num_pc": 1000,
        "sigma_r": 1.0, #2
        "mod_d_opt": 1.35,
        "mod_sigma": 0.4,
        "sigma_theta": 3,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 50,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.25,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.3,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.0,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.35,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.0, #0.35
        "gamma_pg": 0.35,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 5.5,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 1.5,
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
        "num_pc": 500,
        "sigma_r": 1.5, # 2.0
        "mod_d_opt": 0.0,
        "mod_sigma": 2.1,
        "sigma_theta": 5,
        "rcn_learning_rate": 0.1,
        # BVC parameters
        "num_bvc_per_dir": 75,  # Number of BVCs per head direction
        # Weight initialization parameters
        "w_in_init_ratio": 0.2,  # Proportion of BVC->PC weights active initially
        "w_grid_init_ratio": 0.2,  # Proportion of GC->PC weights active initially
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.7,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.0,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.25,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.0,  # 0.35
        "gamma_pg": 0.25,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 7,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 1.5,
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
        # Place cell recurrent inhibition parameters
        "gamma_pp": 0.5,  # Place-to-place recurrent inhibition strength
        "gamma_cross": 0.0,  # Cross-scale recurrent inhibition strength
        "gamma_pb": 0.3,  # BVC-to-place afferent inhibition strength
        # Grid cell parameters
        "grid_influence": 0.0,  # 0.35
        "gamma_pg": 0.32,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.0, 1.0),
        "scale_multiplier": 5.0,
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

def compile_scales(scale_names):
    """
    Convert a list of scale names (e.g. ["small", "large"]) into a list of
    actual scale definitions from SCALES_DEFS.
    """
    return [SCALES_DEFS_GRID[name] for name in scale_names]


#################################
# Trial Execution Functions
#################################

def _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs):
    """Run a single trial"""

    # Update goal config for this trial if target_goal specified
    trial_kwargs = kwargs.copy()
    if "goal_config" in trial_kwargs:
        trial_kwargs["goal_config"] = _clone_goal_config(trial_kwargs["goal_config"])
    if target_goal and "goal_config" in trial_kwargs:
        trial_kwargs["goal_config"]["target_goal"] = target_goal

    # Get scale info for RCN setup
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

    world_name = get_world_name(bot)

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
        environment_size=trial_kwargs.get("environment_size", None),
        grid_size=trial_kwargs.get("grid_size", None),
        coverage_percentage=trial_kwargs.get("coverage_percentage", None),
        min_goal_visits=trial_kwargs.get("min_goal_visits", 3),
        optimal_path_distance=trial_kwargs.get("optimal_path_distance", None),
        path_failure_ratio=trial_kwargs.get("path_failure_ratio", None),
        paths_folder=trial_kwargs.get("paths_folder", None),
        hmaps_folder=trial_kwargs.get("hmaps_folder", None),
        unified_recurrent_preplay_horizon=trial_kwargs.get(
            "unified_recurrent_preplay_horizon",
            UNIFIED_PREPLAY_DEFAULTS["unified_recurrent_preplay_horizon"],
        ),
        unified_preplay_discount_factor=trial_kwargs.get(
            "unified_preplay_discount_factor",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
        ),
        unified_preplay_within_direction_beta=trial_kwargs.get(
            "unified_preplay_within_direction_beta",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
        ),
        unified_preplay_num_samples=trial_kwargs.get(
            "unified_preplay_num_samples",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
        ),
        unified_preplay_sampling_temperature=trial_kwargs.get(
            "unified_preplay_sampling_temperature",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
        ),
        unified_preplay_turn_offsets=trial_kwargs.get(
            "unified_preplay_turn_offsets",
            list(UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"]),
        ),
        unified_preplay_normalize_transitions=trial_kwargs.get(
            "unified_preplay_normalize_transitions",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_normalize_transitions"],
        ),
        unified_preplay_global_score_normalization=trial_kwargs.get(
            "unified_preplay_global_score_normalization",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_global_score_normalization"],
        ),
        unified_preplay_executable_rollouts=trial_kwargs.get(
            "unified_preplay_executable_rollouts",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_executable_rollouts"],
        ),
        unified_preplay_boundary_mode=trial_kwargs.get(
            "unified_preplay_boundary_mode",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_boundary_mode"],
        ),
        unified_preplay_microtrajectory_safety_margin=trial_kwargs.get(
            "unified_preplay_microtrajectory_safety_margin",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_microtrajectory_safety_margin"],
        ),
        unified_preplay_blocked_return_penalty=trial_kwargs.get(
            "unified_preplay_blocked_return_penalty",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_blocked_return_penalty"],
        ),
        unified_preplay_no_reward_threshold=trial_kwargs.get(
            "unified_preplay_no_reward_threshold",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_no_reward_threshold"],
        ),
        unified_preplay_decision_diagnostics=trial_kwargs.get(
            "unified_preplay_decision_diagnostics",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_decision_diagnostics"],
        ),
        unified_preplay_decision_diagnostics_stride=trial_kwargs.get(
            "unified_preplay_decision_diagnostics_stride",
            UNIFIED_PREPLAY_DEFAULTS["unified_preplay_decision_diagnostics_stride"],
        ),
        stdp_rectified_scale_centering=trial_kwargs.get("stdp_rectified_scale_centering", False),
        stdp_rectified_hd_gate=trial_kwargs.get("stdp_rectified_hd_gate", True),
        stdp_winner_hd_gate=trial_kwargs.get("stdp_winner_hd_gate", False),
        stdp_min_input_mass=trial_kwargs.get("stdp_min_input_mass", 0.0),
        exploit_bvc_context_gaussian_modulation=trial_kwargs.get(
            "exploit_bvc_context_gaussian_modulation", True
        ),
        exploit_cross_scale_inhibition=trial_kwargs.get(
            "exploit_cross_scale_inhibition", False
        ),
        auto_trial_name=trial_kwargs.get("auto_trial_name", None),
        num_auto_trials=trial_kwargs.get("num_auto_trials", 5),
        current_auto_trial=trial_kwargs.get("current_auto_trial", 1),
        exploit_debug_logging=trial_kwargs.get("exploit_debug_logging", False),
        exploit_hd_score_logging=trial_kwargs.get("exploit_hd_score_logging", False),
        exploit_loop_recovery=trial_kwargs.get(
            "exploit_loop_recovery",
            UNIFIED_PREPLAY_DEFAULTS["exploit_loop_recovery"],
        ),
        debug_heading_convention_test=trial_kwargs.get("debug_heading_convention_test", False),
        debug_heading_test_forward_steps=trial_kwargs.get("debug_heading_test_forward_steps", 4),
        debug_heading_test_start_loc=trial_kwargs.get("debug_heading_test_start_loc", None),
    )

    bot.trial_id = trial_id

    # Run the trial
    bot.run()


def _run_simple_trials(bot, mode, trial_config, **kwargs):
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

    world_name = get_world_name(bot)

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
            # Skip if goal and start are the same
            goal_config = kwargs.get("goal_config", {})
            if goal_config.get("type") == "single" and goal_config["location"] == start_loc:
                print(f"[INFO] Skipping trial - start location {start_loc} equals goal location")
                continue

            # Get existing trial count
            if save_data:
                current_trial_id = get_highest_trial_id(stats_folder, start_loc)
                trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}"
            else:
                trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}"

            print(f"[INFO] Running trial: {trial_id}")

            # Run single trial
            _run_single_trial(bot, mode, trial_id, start_loc, None, stats_collector_instance, **trial_kwargs)


def _run_combination_trials(bot, mode, trial_config, **kwargs):
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

    world_name = get_world_name(bot)

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
                goals = _goal_entries_from_config(goal_config)
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


def _run_random_spawn_trials(bot, mode, trial_config, **kwargs):
    """Handle random spawn trial execution with path planning"""
    trials_per_goal = trial_config["trials_per_goal"]
    save_data = kwargs.get("save_data", False)

    # Get path planning parameters
    min_spawn_distance = kwargs.get("min_spawn_distance", 6.0)
    wall_clearance = kwargs.get("wall_clearance", 0.5)
    path_failure_ratio = kwargs.get("path_failure_ratio", 10.0)

    world_name = get_world_name(bot)

    # Get goals from goal config
    goal_config = kwargs.get("goal_config", {})
    goals = _goal_entries_from_config(goal_config)

    print(f"[RANDOM_SPAWN] Generating spawn locations and paths for {len(goals)} goals")

    # Load or generate path-validated spawn locations shared with the original model.
    combinations, path_results = _get_shared_spawn_trials(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance,
        **_spawn_manifest_kwargs(kwargs),
    )

    # Setup stats collection
    if save_data:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        # Use stats_random instead of stats
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)

        # Create paths visualization folder and generate plots if enabled
        generate_plots = kwargs.get("generate_path_plots", True)
        if generate_plots:
            paths_folder = os.path.join(
                PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "paths"
            )
            os.makedirs(paths_folder, exist_ok=True)

            # Generate and save path visualizations
            print("[RANDOM_SPAWN] Generating path visualizations...")
            save_path_visualizations(
                world_name=world_name,
                combinations=combinations,
                path_results=path_results,
                output_dir=paths_folder,
                wall_clearance=wall_clearance,
                min_spawn_distance=min_spawn_distance
            )
        else:
            print("[RANDOM_SPAWN] Path plot generation disabled, skipping visualizations...")
            paths_folder = None

        # Create hmaps folder
        hmaps_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "hmaps"
        )
        os.makedirs(hmaps_folder, exist_ok=True)

        stats_collector_instance = stats_collector(output_dir=stats_folder)
    else:
        stats_collector_instance = None
        paths_folder = None

    print(f"[RANDOM_SPAWN] Running {len(combinations)} trials")

    # Execute trials using single-trial pattern
    successful_trials = 0
    for i, (combination, path_result) in enumerate(zip(combinations, path_results)):
        if not path_result["success"]:
            raise RuntimeError(
                f"Shared spawn manifest contains a failed path for "
                f"{combination['goal_name']} trial {combination['trial_number']}"
            )

        start_pos = combination["start"]
        goal_name = combination["goal_name"]
        trial_number = combination["trial_number"]
        optimal_distance = path_result["distance"]

        # Create trial ID
        trial_id = f"trial_{trial_number}_goal_{goal_name}_random"

        print(f"[RANDOM_SPAWN] Running {trial_id}: Start {start_pos} -> Goal {goal_name} ({successful_trials + 1}/{len(path_results)})")

        # Prepare trial-specific kwargs
        trial_kwargs = kwargs.copy()
        trial_kwargs.pop("start_loc", None)
        trial_kwargs["goal_config"] = {
            "type": "multi",
            "goals": goals,
            "target_goal": goal_name  # Set target goal for this trial
        }
        trial_kwargs["optimal_path_distance"] = optimal_distance
        trial_kwargs["path_failure_ratio"] = path_failure_ratio
        trial_kwargs["paths_folder"] = paths_folder
        trial_kwargs["hmaps_folder"] = hmaps_folder if save_data else None
        trial_kwargs["path_visualization"] = path_result.get("visualization_path", None)

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


def _regular_pkl_training_network_dir(world_name):
    return os.path.join("pkl", world_name, "networks")


def _check_regular_pkl_training_exists(world_name):
    network_dir = _regular_pkl_training_network_dir(world_name)
    required_files = [
        "pcn_unified.pkl",
        "unified_rcn_goal.pkl",
        "gcn_scale_0.pkl",
        "gcn_scale_1.pkl",
        "gcn_scale_2.pkl",
    ]
    return os.path.isdir(network_dir) and all(
        os.path.exists(os.path.join(network_dir, filename))
        for filename in required_files
    )


def _available_regular_pkl_training_worlds():
    pkl_root = os.path.join("pkl")
    if not os.path.isdir(pkl_root):
        return []
    worlds = []
    for entry in sorted(os.listdir(pkl_root)):
        if _check_regular_pkl_training_exists(entry):
            worlds.append(entry)
    return worlds


def _clone_hmaps_for_plotting(source_hmap_dir, suffix="plotting_coverage"):
    """Clone existing hmap files into a plotting-specific folder and return it."""
    normalized_source = os.path.normpath(source_hmap_dir)
    parent_dir = os.path.dirname(normalized_source)
    source_name = os.path.basename(normalized_source)
    plotting_hmap_dir = os.path.join(parent_dir, f"{source_name}_{suffix}")
    os.makedirs(plotting_hmap_dir, exist_ok=True)

    copied = 0
    if os.path.isdir(normalized_source):
        for filename in os.listdir(normalized_source):
            if not filename.startswith("hmap_"):
                continue
            src = os.path.join(normalized_source, filename)
            dst = os.path.join(plotting_hmap_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied += 1

    print(
        f"[PLOTTING_HMAPS] Using cloned hmap folder: {plotting_hmap_dir} "
        f"(copied {copied} existing hmap item(s) from {normalized_source})"
    )
    return plotting_hmap_dir


def _run_regular_pkl_plotting_trial(bot, mode, **kwargs):
    """Run time-based plotting against the current world's regular pkl/{world} folder."""
    world_name = get_world_name(bot)
    if not _check_regular_pkl_training_exists(world_name):
        available = _available_regular_pkl_training_worlds()
        print(f"[PLOTTING] No regular training folder found for current world: {world_name}")
        print(f"[PLOTTING] Available regular pkl training worlds: {available}")
        bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]
    trial_id = "plotting_existing_pkl"
    plotting_run_time_hours = kwargs.get("plotting_run_time_hours", 10)
    source_hmap_dir = os.path.join("pkl", world_name, "hmaps")
    plotting_hmap_dir = _clone_hmaps_for_plotting(source_hmap_dir, suffix="plotting")

    print(f"[PLOTTING] Using regular training folder: pkl/{world_name}/networks")
    print(
        "[PLOTTING] Learned networks will be loaded read-only; "
        f"plotting hmaps will be saved to {plotting_hmap_dir}."
    )
    print(f"[PLOTTING] Time-based plotting duration: {plotting_run_time_hours} hours")

    parameter_kwargs = dict(kwargs)
    parameter_kwargs["auto_trial_name"] = None
    parameter_kwargs["current_auto_trial"] = None
    parameter_kwargs["run_time_hours"] = plotting_run_time_hours
    parameter_kwargs["plotting_stop_condition"] = "time_limit"
    parameter_kwargs["plotting_hmap_dir"] = plotting_hmap_dir
    save_trial_parameters(
        world_name=world_name,
        trial_id=trial_id,
        mode=mode,
        **parameter_kwargs,
    )

    bot.initialization(
        mode=RobotMode.PLOTTING_AUTO,
        run_time_hours=plotting_run_time_hours,
        randomize_start_loc=kwargs.get("randomize_start_loc", False),
        start_loc=kwargs.get("start_loc", [0, 0]),
        enable_ojas=False,
        enable_stdp=False,
        scales=scales_list,
        rcn_learning_rates=rcn_learning_rates,
        stats_collector=None,
        trial_id=trial_id,
        world_name=world_name,
        goal_config=kwargs.get("goal_config"),
        trial_config=kwargs.get("trial_config"),
        max_dist=kwargs.get("max_dist", 25),
        plot_bvc=kwargs.get("plot_bvc", False),
        td_learning=False,
        use_prox_mod=False,
        environment_size=kwargs.get("environment_size", None),
        grid_size=kwargs.get("grid_size", None),
        coverage_percentage=None,
        min_goal_visits=kwargs.get("min_goal_visits", 3),
        hmaps_folder=plotting_hmap_dir,
        auto_trial_name=None,
        num_auto_trials=1,
        current_auto_trial=1,
    )

    print(f"[PLOTTING] Running time-based plotting for {world_name}...")
    bot.run()
    print(f"[PLOTTING] Time-based plotting complete for {world_name}.")

    marker_path = os.path.join("pkl", world_name, "plotting_complete.txt")
    with open(marker_path, "w") as f:
        f.write(f"Plotting (time-based) completed for {world_name}\n")
        f.write(f"Run time: {plotting_run_time_hours} hours\n")
        f.write(f"Hmaps folder: {plotting_hmap_dir}\n")
        f.write(f"Environment size: {kwargs.get('environment_size', [20.0, 20.0])}\n")
        f.write(f"Grid size: {kwargs.get('grid_size', 0.5)}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[PLOTTING] Cleared GPU cache")
    gc.collect()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


def _run_learn_coverage_auto_trials(bot, mode, **kwargs):
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

    world_name = get_world_name(bot)

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
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

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
            **kwargs
        )

        # Initialize bot for this trial (creates fresh networks, resets robot position)
        bot.initialization(
            mode=mode,
            run_time_hours=kwargs.get("run_time_hours", 2),
            randomize_start_loc=kwargs.get("randomize_start_loc", False),
            start_loc=kwargs.get("start_loc", [0, 0]),
            enable_ojas=kwargs.get("enable_ojas", True),
            enable_stdp=kwargs.get("enable_stdp", True),
            scales=scales_list,
            rcn_learning_rates=rcn_learning_rates,
            stats_collector=None,  # No stats collector for learning
            trial_id=f"auto_trial_{trial_num}",
            world_name=world_name,
            goal_config=kwargs.get("goal_config"),
            trial_config=kwargs.get("trial_config"),
            max_dist=kwargs.get("max_dist", 25),
            plot_bvc=kwargs.get("plot_bvc", False),
            td_learning=kwargs.get("td_learning", False),
            use_prox_mod=kwargs.get("use_prox_mod", False),
            environment_size=kwargs.get("environment_size", None),
            grid_size=kwargs.get("grid_size", None),
            coverage_percentage=kwargs.get("coverage_percentage", None),
            min_goal_visits=kwargs.get("min_goal_visits", 3),
            unified_recurrent_preplay_horizon=kwargs.get(
                "unified_recurrent_preplay_horizon",
                UNIFIED_PREPLAY_DEFAULTS["unified_recurrent_preplay_horizon"],
            ),
            unified_preplay_discount_factor=kwargs.get(
                "unified_preplay_discount_factor",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
            ),
            unified_preplay_within_direction_beta=kwargs.get(
                "unified_preplay_within_direction_beta",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
            ),
            unified_preplay_num_samples=kwargs.get(
                "unified_preplay_num_samples",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
            ),
            unified_preplay_sampling_temperature=kwargs.get(
                "unified_preplay_sampling_temperature",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
            ),
            unified_preplay_turn_offsets=kwargs.get(
                "unified_preplay_turn_offsets",
                list(UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"]),
            ),
            unified_preplay_normalize_transitions=kwargs.get(
                "unified_preplay_normalize_transitions",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_normalize_transitions"],
            ),
            unified_preplay_global_score_normalization=kwargs.get(
                "unified_preplay_global_score_normalization",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_global_score_normalization"],
            ),
            unified_preplay_executable_rollouts=kwargs.get(
                "unified_preplay_executable_rollouts",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_executable_rollouts"],
            ),
            unified_preplay_microtrajectory_safety_margin=kwargs.get(
                "unified_preplay_microtrajectory_safety_margin",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_microtrajectory_safety_margin"],
            ),
            unified_preplay_blocked_return_penalty=kwargs.get(
                "unified_preplay_blocked_return_penalty",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_blocked_return_penalty"],
            ),
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
        )

        # Run the trial (initialization() already reset self.done = False)
        print(f"[AUTO_TRIAL] Running trial {trial_num}...")
        bot.run()
        print(f"[AUTO_TRIAL] Trial {trial_num} complete!")

    print(f"\n[AUTO_TRIAL] All {num_auto_trials} trials complete!")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)


def _run_exploit_random_auto_trials(bot, mode, **kwargs):
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

    world_name = get_world_name(bot)

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
    generate_plots = kwargs.get("generate_path_plots", True)

    # Get goal config
    goal_config = kwargs.get("goal_config", {})
    goals = _goal_entries_from_config(goal_config)

    # Get scale info
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]
    scale_name_str = "_".join(scale_names)

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

    # Load or generate path-validated spawn locations (same for all trials and models).
    print(f"[AUTO_EXPLOIT] Generating spawn locations for {len(goals)} goals, {trials_per_goal} trials per goal")
    combinations, path_results = _get_shared_spawn_trials(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance,
        **_spawn_manifest_kwargs(kwargs),
    )

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

        # Create paths and hmaps folders
        if generate_plots:
            paths_folder = os.path.join(
                PROJECT_ROOT, "analysis", stats_base, world_trial_folder, scale_name_str, "paths"
            )
            os.makedirs(paths_folder, exist_ok=True)

            # Generate path visualizations (once per learning trial)
            print(f"[AUTO_EXPLOIT] Generating path visualizations for trial {trial_num}...")
            save_path_visualizations(
                world_name=world_name,
                combinations=combinations,
                path_results=path_results,
                output_dir=paths_folder,
                wall_clearance=wall_clearance,
                min_spawn_distance=min_spawn_distance
            )
        else:
            paths_folder = None

        hmaps_folder = os.path.join(
            PROJECT_ROOT, "analysis", stats_base, world_trial_folder, scale_name_str, "hmaps"
        )
        os.makedirs(hmaps_folder, exist_ok=True)

        # Create stats collector for this learning trial
        stats_collector_instance = stats_collector(output_dir=stats_folder)

        # Run all exploitation trials for this learning trial
        successful_trials = 0
        for i, (combination, path_result) in enumerate(zip(combinations, path_results)):
            if not path_result["success"]:
                raise RuntimeError(
                    f"Shared spawn manifest contains a failed path for "
                    f"{combination['goal_name']} trial {combination['trial_number']}"
                )

            start_pos = combination["start"]
            goal_name = combination["goal_name"]
            trial_number = combination["trial_number"]
            optimal_distance = path_result["distance"]

            # Create trial ID
            trial_id = f"trial_{trial_number}_goal_{goal_name}_random"

            print(f"[AUTO_EXPLOIT] Learning trial {trial_num}, Exploit trial {trial_id}: {start_pos} -> {goal_name} ({successful_trials + 1}/{len(path_results)})")

            # Prepare trial-specific kwargs
            trial_kwargs = kwargs.copy()
            trial_kwargs.pop("start_loc", None)
            trial_kwargs["goal_config"] = {
                "type": "multi",
                "goals": goals,
                "target_goal": goal_name
            }
            trial_kwargs["optimal_path_distance"] = optimal_distance
            trial_kwargs["path_failure_ratio"] = path_failure_ratio
            trial_kwargs["paths_folder"] = paths_folder
            trial_kwargs["hmaps_folder"] = hmaps_folder
            trial_kwargs["path_visualization"] = path_result.get("visualization_path", None)
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


def _run_plotting_auto_trials(bot, mode, **kwargs):
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

    world_name = get_world_name(bot)

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
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
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
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{trial_num}"
        trial_path = os.path.join(base_folder, trial_folder)
        source_hmap_dir = os.path.join(trial_path, "hmaps")
        plotting_hmap_dir = _clone_hmaps_for_plotting(source_hmap_dir, suffix="plotting")

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
            environment_size=kwargs.get("environment_size", None),
            grid_size=kwargs.get("grid_size", None),
            coverage_percentage=kwargs.get("coverage_percentage", None),
            min_goal_visits=kwargs.get("min_goal_visits", 3),
            unified_recurrent_preplay_horizon=kwargs.get(
                "unified_recurrent_preplay_horizon",
                UNIFIED_PREPLAY_DEFAULTS["unified_recurrent_preplay_horizon"],
            ),
            unified_preplay_discount_factor=kwargs.get(
                "unified_preplay_discount_factor",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
            ),
            unified_preplay_within_direction_beta=kwargs.get(
                "unified_preplay_within_direction_beta",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
            ),
            unified_preplay_num_samples=kwargs.get(
                "unified_preplay_num_samples",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
            ),
            unified_preplay_sampling_temperature=kwargs.get(
                "unified_preplay_sampling_temperature",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
            ),
            unified_preplay_turn_offsets=kwargs.get(
                "unified_preplay_turn_offsets",
                list(UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"]),
            ),
            unified_preplay_normalize_transitions=kwargs.get(
                "unified_preplay_normalize_transitions",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_normalize_transitions"],
            ),
            unified_preplay_global_score_normalization=kwargs.get(
                "unified_preplay_global_score_normalization",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_global_score_normalization"],
            ),
            unified_preplay_executable_rollouts=kwargs.get(
                "unified_preplay_executable_rollouts",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_executable_rollouts"],
            ),
            unified_preplay_microtrajectory_safety_margin=kwargs.get(
                "unified_preplay_microtrajectory_safety_margin",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_microtrajectory_safety_margin"],
            ),
            unified_preplay_blocked_return_penalty=kwargs.get(
                "unified_preplay_blocked_return_penalty",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_blocked_return_penalty"],
            ),
            hmaps_folder=plotting_hmap_dir,
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
        )

        # Run the plotting trial
        print(f"[AUTO_PLOTTING] Running plotting for trial {trial_num}...")
        bot.run()
        print(f"[AUTO_PLOTTING] Plotting trial {trial_num} complete!")

        # Create marker file to indicate this plotting trial is complete
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


def _run_plotting_coverage_auto_trials(bot, mode, **kwargs):
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
    print(f"[AUTO_PLOTTING_COVERAGE] Coverage target: {kwargs.get('coverage_percentage', 0.90) * 100}%")

    world_name = get_world_name(bot)

    # Check which learning trials exist
    available_trials = []
    for trial_num in range(1, num_auto_trials + 1):
        if _check_trial_exists(auto_trial_name, world_name, trial_num):
            available_trials.append(trial_num)
        else:
            print(f"[AUTO_PLOTTING_COVERAGE] Learning trial {trial_num} not found, skipping...")

    if not available_trials:
        if _check_regular_pkl_training_exists(world_name):
            print(
                "[AUTO_PLOTTING_COVERAGE] No auto learning trials found; "
                f"falling back to time-based plotting for regular pkl/{world_name}/networks."
            )
            _run_regular_pkl_plotting_trial(bot, mode, **kwargs)
        else:
            print(
                "[AUTO_PLOTTING_COVERAGE] No auto learning trials found, and no "
                f"regular pkl/{world_name}/networks training folder exists."
            )
            print(
                "[AUTO_PLOTTING_COVERAGE] Available regular pkl training worlds: "
                f"{_available_regular_pkl_training_worlds()}"
            )
            bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
        return

    print(f"[AUTO_PLOTTING_COVERAGE] Found {len(available_trials)} learning trials to plot: {available_trials}")

    # Get scale info
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
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
        base_folder = f"pkl_{auto_trial_name}"
        trial_folder = f"{world_name}_{trial_num}"
        trial_path = os.path.join(base_folder, trial_folder)
        source_hmap_dir = os.path.join(trial_path, "hmaps")
        plotting_hmap_dir = _clone_hmaps_for_plotting(source_hmap_dir, suffix="plotting_coverage")

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
            environment_size=kwargs.get("environment_size", None),
            grid_size=kwargs.get("grid_size", None),
            coverage_percentage=kwargs.get("coverage_percentage", None),
            min_goal_visits=kwargs.get("min_goal_visits", 3),
            unified_recurrent_preplay_horizon=kwargs.get(
                "unified_recurrent_preplay_horizon",
                UNIFIED_PREPLAY_DEFAULTS["unified_recurrent_preplay_horizon"],
            ),
            unified_preplay_discount_factor=kwargs.get(
                "unified_preplay_discount_factor",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_discount_factor"],
            ),
            unified_preplay_within_direction_beta=kwargs.get(
                "unified_preplay_within_direction_beta",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_within_direction_beta"],
            ),
            unified_preplay_num_samples=kwargs.get(
                "unified_preplay_num_samples",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_num_samples"],
            ),
            unified_preplay_sampling_temperature=kwargs.get(
                "unified_preplay_sampling_temperature",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_sampling_temperature"],
            ),
            unified_preplay_turn_offsets=kwargs.get(
                "unified_preplay_turn_offsets",
                list(UNIFIED_PREPLAY_DEFAULTS["unified_preplay_turn_offsets"]),
            ),
            unified_preplay_normalize_transitions=kwargs.get(
                "unified_preplay_normalize_transitions",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_normalize_transitions"],
            ),
            unified_preplay_global_score_normalization=kwargs.get(
                "unified_preplay_global_score_normalization",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_global_score_normalization"],
            ),
            unified_preplay_executable_rollouts=kwargs.get(
                "unified_preplay_executable_rollouts",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_executable_rollouts"],
            ),
            unified_preplay_microtrajectory_safety_margin=kwargs.get(
                "unified_preplay_microtrajectory_safety_margin",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_microtrajectory_safety_margin"],
            ),
            unified_preplay_blocked_return_penalty=kwargs.get(
                "unified_preplay_blocked_return_penalty",
                UNIFIED_PREPLAY_DEFAULTS["unified_preplay_blocked_return_penalty"],
            ),
            hmaps_folder=plotting_hmap_dir,
            auto_trial_name=auto_trial_name,
            num_auto_trials=num_auto_trials,
            current_auto_trial=trial_num,
        )

        # Run the plotting trial with coverage-based stopping
        print(f"[AUTO_PLOTTING_COVERAGE] Running plotting for trial {trial_num} until coverage target reached...")
        bot.run()
        print(f"[AUTO_PLOTTING_COVERAGE] Plotting trial {trial_num} complete!")

        # Create marker file to indicate this plotting trial is complete (with _coverage suffix)
        plotting_marker = os.path.join(trial_path, "plotting_complete_coverage.txt")

        with open(plotting_marker, 'w') as f:
            f.write(f"Plotting (coverage-based) completed for trial {trial_num}\n")
            f.write(f"Coverage target: {kwargs.get('coverage_percentage', 0.90) * 100}%\n")
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
    bot = Driver()
    kwargs = _apply_world_defaults(mode, kwargs, bot)

    # Handle AUTO modes separately
    if mode == RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO:
        _run_learn_coverage_auto_trials(bot, mode, **kwargs)
        return
    elif mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO:
        _run_exploit_random_auto_trials(bot, mode, **kwargs)
        return
    elif mode == RobotMode.PLOTTING_AUTO:
        _run_plotting_auto_trials(bot, mode, **kwargs)
        return
    elif mode == RobotMode.PLOTTING_COVERAGE_AUTO:
        _run_plotting_coverage_auto_trials(bot, mode, **kwargs)
        return

    # Get trial configuration and remove it from kwargs to avoid conflict
    trial_config = kwargs.pop("trial_config", {
        "type": "simple",
        "count": 1
    })

    print(f"[INFO] Starting run with mode: {mode}")
    print(f"[INFO] Trial type: {trial_config['type']}")

    if trial_config["type"] == "simple":
        _run_simple_trials(bot, mode, trial_config, **kwargs)
    elif trial_config["type"] == "combinations":
        _run_combination_trials(bot, mode, trial_config, **kwargs)
    elif trial_config["type"] == "random_spawns":
        _run_random_spawn_trials(bot, mode, trial_config, **kwargs)
    else:
        raise ValueError(f"Unknown trial type: {trial_config['type']}")

    # Pause the simulation
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

#################################
# Main Controller Entry Point
#################################

if __name__ == "__main__":

    # We'll map string to the actual RobotMode enum
    MODES_MAP = {
        "LEARN_OJAS": RobotMode.LEARN_OJAS,
        "LEARN_HEBB": RobotMode.LEARN_HEBB,
        "DMTP": RobotMode.LEARN_LOCATIONS_COVERAGE,
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
    }

    SELECTED_MODE = "PLOTTING_COVERAGE_AUTO"
    td_learning = False # keep off
    corners = [[8,-8]] # start point
    exploit_start = corners[0]
    start_loc = None

    goal_location = None
    randomize_start_loc = False
    use_prox_mod = False

    multiscale = ["small", "medium", "large"]
    small = ["small"]
    medium = ["medium"]
    large = ["large"]

    scale_names = multiscale # what scales you are using
    run_time_hours = 4
    max_dist = 25
    plot_bvc = False

    enable_ojas = False
    enable_stdp = False

    # Goal and start metadata are inferred from the active .wbt world unless
    # explicitly overridden per mode.
    learning_goal_config = None
    exploit_goal_config = None

    

    # Coverage parameters for LEARN_LOCATIONS_COVERAGE
    environment_size = None
    grid_size = 0.5  # 0.5 meter grid cells
    coverage_percentage = 0.95  # 90% coverage target
    min_goal_visits = 10  # Minimum number of visits required per goal

    # Random spawn parameters for EXPLOIT_LOCATIONS_RANDOM
    min_spawn_distance = 6.0  # 6 meters from goal
    wall_clearance = 0.5  # 0.5 meter clearance from walls
    generate_path_plots = False  # Whether to generate path visualization plots
    trials_per_goal = 20  # 20 trials per goal
    path_failure_ratio = 10.0  # Fail if robot travels 2x optimal distance

    # Unified exploit/preplay configuration
    unified_preplay_config = copy.deepcopy(UNIFIED_PREPLAY_DEFAULTS)
    unified_stdp_config = {
        "stdp_rectified_scale_centering": False,
        "stdp_rectified_hd_gate": True,
        "stdp_winner_hd_gate": False,
        "stdp_min_input_mass": 0.0,
    }
    heading_convention_debug_config = {
        "exploit_debug_logging": False,
        "exploit_hd_score_logging": False,
        "debug_heading_convention_test": False,
        "debug_heading_test_forward_steps": 4,
        "debug_heading_test_start_loc": None,
    }
    exploit_activation_config = {
        "exploit_bvc_context_gaussian_modulation": True,
        "exploit_cross_scale_inhibition": False,
    }

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
            "plot_bvc": plot_bvc
        },
        "LEARN_HEBB": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "num_loops": 1,
            "save_data": False,
            "plot_bvc": plot_bvc,
            **unified_stdp_config,
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
            **unified_preplay_config,
            **exploit_activation_config,
            **heading_convention_debug_config,
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
            **exploit_activation_config,
            **heading_convention_debug_config,
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
            **unified_stdp_config,
            **heading_convention_debug_config,
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
            "plot_bvc": plot_bvc
        },
        "LEARN_LOCATIONS_COVERAGE": {
            "corners": [[0, 0]],  # Single starting location for learning
            "start_loc": start_loc,
            "goal_config": learning_goal_config,
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
            "use_prox_mod": use_prox_mod,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            **unified_stdp_config,
        },
        "LEARN_LOCATIONS_COVERAGE_AUTO": {
            "start_loc": start_loc,
            "goal_config": learning_goal_config,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "td_learning": False,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": False,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            "min_goal_visits": min_goal_visits,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
            **unified_stdp_config,
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
            "goal_config": exploit_goal_config,
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
            **unified_preplay_config,
            **exploit_activation_config,
            **heading_convention_debug_config,
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
            "goal_config": exploit_goal_config,
            # Random spawn specific parameters
            "trials_per_goal": trials_per_goal,
            "min_spawn_distance": min_spawn_distance,
            "wall_clearance": wall_clearance,
            "path_failure_ratio": path_failure_ratio,
            "generate_path_plots": generate_path_plots,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
            **unified_preplay_config,
            **exploit_activation_config,
            **heading_convention_debug_config,
        },
        "PLOTTING_AUTO": {
            "start_loc": start_loc,
            "goal_config": learning_goal_config,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "td_learning": False,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        },
        "PLOTTING_COVERAGE_AUTO": {
            "start_loc": start_loc,
            "goal_config": learning_goal_config,
            "trial_config": {"type": "simple", "count": 1},
            "max_dist": max_dist,
            "randomize_start_loc": False,
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": 24,  # High fallback time limit for coverage-based auto plotting
            "plotting_run_time_hours": 10,  # Used by regular pkl/{world} fallback plotting
            "td_learning": False,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "environment_size": environment_size,
            "grid_size": grid_size,
            "coverage_percentage": coverage_percentage,
            # Auto trial specific parameters (use shared variables)
            "auto_trial_name": auto_trial_name,
            "num_auto_trials": num_auto_trials,
        }
    }

    mode_aliases = {
        "DMTP": "LEARN_LOCATIONS_COVERAGE",
    }
    selected_mode_key = mode_aliases.get(SELECTED_MODE, SELECTED_MODE)

    if selected_mode_key not in MODE_PARAMS or selected_mode_key not in MODES_MAP:
        print("Invalid mode selected.")
        sys.exit(1)

    if selected_mode_key != SELECTED_MODE:
        print(f"[INFO] {SELECTED_MODE} is an alias for {selected_mode_key}.")

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[selected_mode_key]
    params = MODE_PARAMS[selected_mode_key]

    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)
