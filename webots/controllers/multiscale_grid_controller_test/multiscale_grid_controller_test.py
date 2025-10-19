"""Multiscale Controller Test - Enhanced with LEARN_LOCATIONS_COVERAGE and EXPLOIT_LOCATIONS_RANDOM modes"""

import sys
import os
import re
from pathlib import Path

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from multiscale_grid_driver_test import Driver
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector
from path_planning import generate_spawn_locations, calculate_optimal_paths, save_path_visualizations

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

#################################
# Scale Definitions
#################################

SCALES_DEFS = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "num_pc": 2000,
        "sigma_r": 0.5,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # Grid cell parameters
        "grid_influence": 0.25, # 0.25
        "gamma_pg": 0.3, # 0.3
        "num_grid_cells": 800,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 100.0,
        "frequency_divisor": 0.25,
        # Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 0.25,  # 2.0 * sigma_r = 1.0m threshold
        "proximity_suppression_steepness": 10.0,
        "proximity_suppression_midpoint": 0.5,

    },
    "medium": {
        "scale_index": 1,
        "name": "medium",
        "num_pc": 1000,
        "sigma_r": 2,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # Grid cell parameters
        "grid_influence": 0.25, #0.25
        "gamma_pg": 0.32,
        "num_grid_cells": 600,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 200.0,
        "frequency_divisor": 0.5,
        # Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 0.25,  # 1.2 * sigma_r = 2.4m threshold
        "proximity_suppression_steepness": 9.0,
        "proximity_suppression_midpoint": 0.45,

    },
    "large": {
        "scale_index": 2,
        "name": "large",
        "num_pc": 250,
        "sigma_r": 4,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
        # Grid cell parameters
        "grid_influence": 0.3,  # 0.3
        "gamma_pg": 0.32,
        "num_grid_cells": 500,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 400.0,
        "frequency_divisor": 0.75,
        # Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 0.25,  # 2.0 * sigma_r = 8.0m threshold
        "proximity_suppression_steepness": 8.0,
        "proximity_suppression_midpoint": 0.4,
    },
    "xlarge": {
        "scale_index": 3,
        "name": "xlarge",
        "num_pc": 200,
        "sigma_r": 4,
        "sigma_theta": 8,
        "rcn_learning_rate": 0.005,
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "gamma_pg": 0.32,
        "num_grid_cells": 400,
        "rotation_range": (0, 360),
        "spread_range": (1.0, 1.0),
        "translation_factor": 800.0,
        "frequency_divisor": 0.7,
        # Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 2.0,  # 2.0 * sigma_r = 8.0m threshold
        "proximity_suppression_steepness": 8.0,
        "proximity_suppression_midpoint": 0.4,
    }
}

def compile_scales(scale_names):
    """
    Convert a list of scale names (e.g. ["small", "large"]) into a list of
    actual scale definitions from SCALES_DEFS.
    """
    return [SCALES_DEFS[name] for name in scale_names]


#################################
# Trial Execution Functions
#################################

def _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs):
    """Run a single trial"""

    # Update goal config for this trial if target_goal specified
    trial_kwargs = kwargs.copy()
    if target_goal and "goal_config" in trial_kwargs:
        trial_kwargs["goal_config"]["target_goal"] = target_goal

    # Get scale info for RCN setup
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

    world_name = get_world_name(bot)

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
        optimal_path_distance=trial_kwargs.get("optimal_path_distance", None),
        path_failure_ratio=trial_kwargs.get("path_failure_ratio", None),
        paths_folder=trial_kwargs.get("paths_folder", None),
        hmaps_folder=trial_kwargs.get("hmaps_folder", None),
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
    """Handle random spawn trial execution with path planning"""
    trials_per_goal = trial_config["trials_per_goal"]
    save_data = kwargs.get("save_data", False)

    # Get path planning parameters
    min_spawn_distance = kwargs.get("min_spawn_distance", 6.0)
    wall_clearance = kwargs.get("wall_clearance", 0.5)
    path_failure_ratio = kwargs.get("path_failure_ratio", 10.0)

    bot = Driver()
    world_name = get_world_name(bot)

    # Get goals from goal config
    goal_config = kwargs.get("goal_config", {})
    goals = goal_config.get("goals", [])

    print(f"[RANDOM_SPAWN] Generating spawn locations and paths for {len(goals)} goals")

    # Generate spawn locations using path planning module
    spawn_locations = generate_spawn_locations(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance
    )

    # Create spawn/goal combinations for path calculation
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

    # Calculate optimal paths for all combinations
    path_results = calculate_optimal_paths(world_name, combinations, wall_clearance)

    # Setup stats collection
    if save_data:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        # Use stats_random instead of stats
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)

        # Create paths visualization folder
        paths_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats_random", world_name, scale_name_str, "paths"
        )
        os.makedirs(paths_folder, exist_ok=True)

        # Generate and save path visualizations
        save_path_visualizations(
            world_name=world_name,
            combinations=combinations,
            path_results=path_results,
            output_dir=paths_folder,
            wall_clearance=wall_clearance,
            min_spawn_distance=min_spawn_distance
        )

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
            print(f"[RANDOM_SPAWN] Skipping trial with failed path: {combination['goal_name']} trial {combination['trial_number']}")
            continue

        start_pos = combination["start"]
        goal_name = combination["goal_name"]
        trial_number = combination["trial_number"]
        optimal_distance = path_result["distance"]

        # Create trial ID
        trial_id = f"trial_{trial_number}_goal_{goal_name}_random"

        print(f"[RANDOM_SPAWN] Running {trial_id}: Start {start_pos} -> Goal {goal_name} ({successful_trials + 1}/{len([p for p in path_results if p['success']])})")

        # Prepare trial-specific kwargs
        trial_kwargs = kwargs.copy()
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


#################################
# run_bot
#################################

def run_bot(mode, **kwargs):
    """
    Runs the bot in the specified mode with the given parameters.
    Supports both simple and combination trial configurations.
    """
    # Get trial configuration and remove it from kwargs to avoid conflict
    trial_config = kwargs.pop("trial_config", {
        "type": "simple",
        "count": 1,
        "start_locations": [[0, 0]]
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
        "DMTP": RobotMode.DMTP,
        "EXPLOIT": RobotMode.EXPLOIT,
        "EXPLOIT_SAVE": RobotMode.EXPLOIT,
        "LEARNING_SAVE": RobotMode.EXPLOIT,
        "PLOTTING": RobotMode.PLOTTING,
        "LEARN_LOCATIONS_COVERAGE": RobotMode.LEARN_LOCATIONS_COVERAGE,
        "EXPLOIT_LOCATIONS_RANDOM": RobotMode.EXPLOIT_LOCATIONS_RANDOM,
    }

    SELECTED_MODE = "EXPLOIT_LOCATIONS_RANDOM"
    td_learning = False # keep off
    corners = [[8,-8]] # start point
    dmtp_start = [-9,9]
    exploit_start = corners[0]
    start_loc = [5, 5]

    goal_location = [-7, 7]
    randomize_start_loc = False
    use_prox_mod = False

    multiscale = ["small", "medium", "large"]
    small = ["small"]
    medium = ["medium"]
    large = ["large"]

    scale_names = multiscale # what scales you are using
    run_time_hours = 8
    max_dist = 25
    plot_bvc = False

    enable_ojas = False
    enable_stdp = False

    # Multi-goal configuration for learning and exploitation
    multi_goal_config = {
        "type": "multi",
        "goals": [
            {"name": "red", "location": [7, 7], "radius": 1.0},
            {"name": "green", "location": [-7, 7], "radius": 1.0},
            {"name": "blue", "location": [7, -7], "radius": 1.0},
            {"name": "yellow", "location": [-7, -7], "radius": 1.0}
        ]
    }

    # Coverage parameters for LEARN_LOCATIONS_COVERAGE
    environment_size = [20.0, 20.0]  # 20x20 meter environment
    grid_size = 0.25  # 0.5 meter grid cells
    coverage_percentage = 0.90  # 90% coverage target

    # Random spawn parameters for EXPLOIT_LOCATIONS_RANDOM
    min_spawn_distance = 6.0  # 6 meters from goal
    wall_clearance = 0.5  # 0.5 meter clearance from walls
    trials_per_goal = 20  # 20 trials per goal
    path_failure_ratio = 10.0  # Fail if robot travels 2x optimal distance


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
            "plot_bvc": plot_bvc
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
            "plot_bvc": plot_bvc
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
            "plot_bvc": plot_bvc
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
            "goal_config": multi_goal_config,
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
            "goal_config": {
                "type": "multi",
                "goals": multi_goal_config["goals"]
                # No target_goal - will be set per trial
            },
            "trial_config": {
                "type": "random_spawns",
                "trials_per_goal": trials_per_goal
            },
            # Random spawn specific parameters
            "trials_per_goal": trials_per_goal,
            "min_spawn_distance": min_spawn_distance,
            "wall_clearance": wall_clearance,
            "path_failure_ratio": path_failure_ratio
        }
    }

    if SELECTED_MODE not in MODE_PARAMS or SELECTED_MODE not in MODES_MAP:
        print("Invalid mode selected.")
        sys.exit(1)

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[SELECTED_MODE]
    params = MODE_PARAMS[SELECTED_MODE]

    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)
