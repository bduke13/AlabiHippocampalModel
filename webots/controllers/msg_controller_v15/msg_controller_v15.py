import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from msg_driver_v15 import MultiscaleDriverWithGrid
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

#################################
# Utility Functions
#################################

def save_run_parameters(world_name, selected_mode, mode_params, scales_defs, scale_names, trial_id=None):
    """
    Save all run parameters to a JSON file in the controller's world directory.
    
    Args:
        world_name: Name of the current world
        selected_mode: The selected mode string
        mode_params: Dictionary of parameters for the selected mode
        scales_defs: Dictionary of all scale definitions
        scale_names: List of scale names being used
        trial_id: Optional trial ID for this run
    """
    # Get the controller directory (where this file is located)
    controller_dir = Path(__file__).resolve().parent
    
    # Create the directory structure in controller/pkl/<world_name>/
    world_dir = os.path.join(controller_dir, "pkl", world_name)
    os.makedirs(world_dir, exist_ok=True)
    
    # Collect all parameters
    run_parameters = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "selected_mode": selected_mode,
            "trial_id": trial_id,
            "world_name": world_name
        },
        "mode_parameters": mode_params,
        "active_scales": {
            name: scales_defs[name] for name in scale_names
        },
        "all_available_scales": scales_defs,
        "scale_configuration": {
            "selected_scale_names": scale_names,
            "scale_combination_string": "_".join(scale_names)
        }
    }
    
    # Save to JSON file (overwrites existing file)
    json_path = os.path.join(world_dir, "run_parameters.json")
    with open(json_path, 'w') as f:
        json.dump(run_parameters, f, indent=2, default=str)
    
    print(f"[INFO] Saved run parameters to: {json_path}")
    return json_path

def get_highest_trial_id(stats_folder, start_loc, target_goal=None):
    """
    Reads the stats directory and determines the highest trial ID for the given start location and optional target goal.
    """
    trial_ids = []
    if os.path.exists(stats_folder):
        for file_name in os.listdir(stats_folder):
            if target_goal:
                # Pattern: trial_X_start_Y_Z_goal_GOALNAME
                pattern = rf"trial_(\d+)_start_{start_loc[0]}_{start_loc[1]}_goal_{target_goal}"
            else:
                # Pattern: trial_X_start_Y_Z (no goal suffix)
                pattern = rf"trial_(\d+)_start_{start_loc[0]}_{start_loc[1]}(?!_goal)"
            
            match = re.match(pattern, file_name)
            if match:
                trial_ids.append(int(match.group(1)))
    return max(trial_ids) if trial_ids else 0

def get_world_name(bot):
    """
    Determines the current world name dynamically from the .wbt file.
    """
    world_path = bot.getWorldPath()
    return os.path.basename(world_path).replace('.wbt', '')

def compile_scales(scale_names):
    """
    Convert a list of scale names (e.g. ["small", "large"]) into a list of 
    actual scale definitions from SCALES_DEFS.
    """
    return [SCALES_DEFS[name] for name in scale_names]

def _run_simple_trials(mode, trial_config, **kwargs):
    """Handle simple trial execution"""
    start_locations = trial_config["start_locations"]
    trials_per_start = trial_config["count"]
    save_data = kwargs.get("save_data", False)
    
    bot = MultiscaleDriverWithGrid()
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
            _run_single_trial(bot, mode, trial_id, start_loc, None, stats_collector_instance, **kwargs)

def _run_combination_trials(mode, trial_config, **kwargs):
    """Handle combination trial execution (start locations x target goals)"""
    start_locations = trial_config["start_locations"]
    target_goals = trial_config.get("target_goals", [None])
    trials_per_combo = trial_config["count"]
    save_data = kwargs.get("save_data", False)
    
    bot = MultiscaleDriverWithGrid()
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
                    current_trial_id = get_highest_trial_id(stats_folder, start_loc, target_goal)
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
                _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs)

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
    replay_timesteps = [scale["replay_timesteps"] for scale in scales_list]
    replay_decay_constants = [scale["replay_decay_constant"] for scale in scales_list]
    
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
        replay_timesteps=replay_timesteps,
        replay_decay_constants=replay_decay_constants,
        stats_collector=stats_collector_instance,
        trial_id=trial_id,
        world_name=world_name,
        goal_config=trial_kwargs.get("goal_config"),
        trial_config=trial_kwargs.get("trial_config"),
        max_dist=trial_kwargs.get("max_dist", 25),
        plot_bvc=trial_kwargs.get("plot_bvc", False),
        td_learning=trial_kwargs.get("td_learning", False),
        use_prox_mod=trial_kwargs.get("use_prox_mod", False),
        clear_files=trial_kwargs.get("clear_files", False),
        action_mode=trial_kwargs.get("action_mode", None),
    )
    
    bot.trial_id = trial_id
    
    # Save parameters for this trial
    save_run_parameters(
        world_name=world_name,
        selected_mode=CURRENT_SELECTED_MODE,
        mode_params=CURRENT_MODE_PARAMS,
        scales_defs=SCALES_DEFS,
        scale_names=scale_names,
        trial_id=trial_id
    )
    
    # Run the trial
    bot.run()
    
    # Reload world if needed for next trial
    if mode == RobotMode.EXPLOIT and trial_kwargs.get("save_data", False):
        bot.worldReload()

#################################
# Scale Definitions
#################################

SCALES_DEFS = {
    "small": {
        "scale_index": 0,
        "name": "small",
        "num_pc": 2000, #2000
        "sigma_r": 0.4,
        "sigma_theta": 1.0,
        "rcn_learning_rate": 0.1,
        "gamma_pp": 0.9, # 0.5
        "gamma_pb": 0.2, # 0.3
        # Grid cell parameters
        "grid_influence": 0.25,  # 0.2
        "gamma_pg": 0.3, # 0.3
        "num_grid_cells": 800,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 100.0,
        "frequency_divisor": 0.25,  # Smallest grid scale (high frequency)
        # Replay Params
        "replay_timesteps": 20, # 35
        "replay_decay_constant": 6, # 6
        # V4 Connection decay parameters (kept)
        "enable_connection_decay": True,
        "connection_decay_rate": 0.00005,
        # V5 NEW: Correlation-based weighting parameters
        "enable_correlation_weighting": True,
        "correlation_window": 12,  # Longer for fine scale (more data needed)
        "correlation_update_freq": 2,  # Less frequent due to computational cost
        "correlation_scaling": 3.5,  # More selective for fine resolution
        "min_correlation_weight": 0.03,  # Lower baseline due to many cells
        "correlation_threshold": 0.015,  # Match fine scale sensitivity
        # V6 NEW: STDP learning rate (slower for fine scale stability)
        "stdp_learning_rate": 0.05,  # Conservative for high-resolution, noisy activations
        "tau_hd": .5,  # HD eligibility trace time constant
        # V6.5 NEW: Adaptive learning parameters for small scale
        "enable_adaptive_stdp": True,
        "adaptive_initial_lr": 0.10,  # High initial learning rate
        "adaptive_final_lr": 0.03,    # Low final learning rate
        "adaptive_decay_rate": 300,   # Faster decay for fine spatial resolution
        "enable_debug_prints": False,     # Set to True to enable debug output
        "enable_adaptive_logging": True,  # Set to True to log adaptive learning data
        # NEW: Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 2.0,  # 2.0 * sigma_r = 0.8m threshold
        "proximity_suppression_steepness": 10.0,
        "proximity_suppression_midpoint": 0.5,

    },
    "medium": {
        "scale_index": 1,
        "name": "medium",
        "num_pc": 1250,
        "sigma_r": 0.6,
        "sigma_theta": 0.7,
        "rcn_learning_rate": 0.1,
        "gamma_pp": 1.0,
        "gamma_pb": 0.2,
        # Grid cell parameters
        "grid_influence": 0.25,  # 0.25
        "gamma_pg": 0.32,
        "num_grid_cells": 600,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 200.0,
        "frequency_divisor": 0.35,  # Medium grid scale 0.5
        # Replay Params
        "replay_timesteps": 20, # 30
        "replay_decay_constant": 6,# 6
        # V4 Connection decay parameters (kept)
        "enable_connection_decay": True,
        "connection_decay_rate": 0.00005,
        # V5 NEW: Correlation-based weighting parameters
        "enable_correlation_weighting": True,
        "correlation_window": 18,  # Moderate for medium scale
        "correlation_update_freq": 4,  # Moderate frequency
        "correlation_scaling": 3.0,  # Moderate selectivity
        "min_correlation_weight": 0.06,  # Moderate baseline
        "correlation_threshold": 0.03,  # Match medium scale sensitivity
        # V6 NEW: STDP learning rate (moderate for medium scale)
        "stdp_learning_rate": 0.05,  # Balanced learning rate for medium resolution
        "tau_hd": .5,  # HD eligibility trace time constant
        # V6.5 NEW: Adaptive learning parameters for medium scale
        "enable_adaptive_stdp": True,
        "adaptive_initial_lr": 0.10,  # High initial learning rate
        "adaptive_final_lr": 0.03,    # Low final learning rate
        "adaptive_decay_rate": 200,   # Balanced decay for medium spatial resolution
        "enable_debug_prints": False,     # Set to True to enable debug output
        "enable_adaptive_logging": True,  # Set to True to log adaptive learning data
        # NEW: Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 1.2,  # 2.0 * sigma_r = 1.6m threshold
        "proximity_suppression_steepness": 9.0,   # Slightly smoother for large scale
        "proximity_suppression_midpoint": 0.45,
    },
    "large": {
        "scale_index": 2,
        "name": "large",
        "num_pc": 500,
        "sigma_r": 0.8,
        "sigma_theta": 1.0,
        "rcn_learning_rate": 0.1,
        "gamma_pp": 1.5,
        "gamma_pb": 0.3,
        # Grid cell parameters
        "grid_influence": 0.3,  # 0.3
        "gamma_pg": 0.32,
        "num_grid_cells": 500,
        "rotation_range": (0, 180),
        "spread_range": (1.0, 1.0),
        "translation_factor": 400.0,
        "frequency_divisor": 0.55,  # Larger grid scale (lower frequency) # 1.0
        # Replay Params
        "replay_timesteps": 10, # 20
        "replay_decay_constant": 6,
        # V4 Connection decay parameters (kept)
        "enable_connection_decay": True,
        "connection_decay_rate": 0.00005,
        # V5 NEW: Correlation-based weighting parameters
        "enable_correlation_weighting": True,
        "correlation_window": 24,  # Shorter for coarse scale
        "correlation_update_freq": 6,  # More frequent updates possible
        "correlation_scaling": 2.5,  # Less selective for coarse resolution
        "min_correlation_weight": 0.12,  # Higher baseline for robustness
        "correlation_threshold": 0.05,  # Match large scale sensitivity
        # V6 NEW: STDP learning rate (faster for large scale)
        "stdp_learning_rate": 0.05,  # Higher learning rate for sparse, coarse activations
        "tau_hd": .5,  # HD eligibility trace time constant
        # V6.5 NEW: Adaptive learning parameters for large scale
        "enable_adaptive_stdp": True,
        "adaptive_initial_lr": 0.10,  # High initial learning rate
        "adaptive_final_lr": 0.03,    # Low final learning rate
        "adaptive_decay_rate": 60,   # Slower decay for coarse spatial resolution
        "enable_debug_prints": False,     # Set to True to enable debug output
        "enable_adaptive_logging": True,  # Set to True to log adaptive learning data
        # NEW: Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 2.0,  # 2.0 * sigma_r = 1.6m threshold
        "proximity_suppression_steepness": 8.0,   # Slightly smoother for large scale
        "proximity_suppression_midpoint": 0.4,
    },
    "xlarge": {
        "scale_index": 3,
        "name": "xlarge",
        "num_pc": 250,
        "sigma_r": 1.0,
        "sigma_theta": 8.0,
        "rcn_learning_rate": 0.1,
        "gamma_pp": 2.0,
        "gamma_pb": 0.30,
        # Grid cell parameters
        "grid_influence": 0.35,  # 0.35
        "gamma_pg": 0.32,
        "num_grid_cells": 400,
        "rotation_range": (0, 360),
        "spread_range": (1.0, 1.0),
        "translation_factor": 800.0,
        "frequency_divisor": 0.7,  # Largest grid scale (lowest frequency)
        # Replay Params
        "replay_timesteps": 20, # 15
        "replay_decay_constant": 6,
        # V4 Connection decay parameters (kept)
        "enable_connection_decay": True,
        "connection_decay_rate": 0.00005,
        # V5 NEW: Correlation-based weighting parameters
        "enable_correlation_weighting": True,
        "correlation_window": 30,  # Shortest for very coarse scale
        "correlation_update_freq": 8,  # Most frequent updates
        "correlation_scaling": 2.0,  # Least selective for global scale
        "min_correlation_weight": 0.18,  # Highest baseline for connectivity
        "correlation_threshold": 0.08,  # Reasonable threshold for xlarge scale
        # V6 NEW: STDP learning rate (fastest for xlarge scale)
        "stdp_learning_rate": 0.05,  # Highest learning rate for very sparse activations
        "tau_hd": .5,  # HD eligibility trace time constant
        # V6.5 NEW: Adaptive learning parameters for xlarge scale
        "enable_adaptive_stdp": True,
        "adaptive_initial_lr": 0.10,  # High initial learning rate
        "adaptive_final_lr": 0.03,    # Low final learning rate
        "adaptive_decay_rate": 30,   # Slowest decay for global spatial patterns
        "enable_debug_prints": False,     # Set to True to enable debug output
        "enable_adaptive_logging": True,  # Set to True to log adaptive learning data
        # NEW: Proximity suppression parameters
        "enable_proximity_suppression": True,
        "proximity_threshold_factor": 2.0,  # 2.0 * sigma_r = 1.6m threshold
        "proximity_suppression_steepness": 8.0,   # Slightly smoother for large scale
        "proximity_suppression_midpoint": 0.4,
    }
}

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
    else:
        raise ValueError(f"Unknown trial type: {trial_config['type']}")
    
    # Pause the simulation
    bot = MultiscaleDriverWithGrid()
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

#################################
# Main Controller Entry Point
#################################

# Global variables to hold current run info (for parameter saving)
CURRENT_SELECTED_MODE = None
CURRENT_MODE_PARAMS = None

if __name__ == "__main__":

    # We'll map string to the actual RobotMode enum
    MODES_MAP = {
        "OJAS": RobotMode.LEARNING,
        "HEBB": RobotMode.LEARNING,
        "LEARNING": RobotMode.LEARNING,
        "DMTP": RobotMode.DMTP,
        "DMTP_EXPLOIT": RobotMode.DMTP,
        "EXPLOIT": RobotMode.EXPLOIT,
        "EXPLOIT_SAVE": RobotMode.EXPLOIT,
        "LEARNING_SAVE": RobotMode.EXPLOIT,
        "PLOTTING": RobotMode.PLOTTING,
        "LEARN_LOCATIONS": RobotMode.LEARN_LOCATIONS,
        "EXPLOIT_LOCATIONS": RobotMode.EXPLOIT_LOCATIONS  
    }
    
    SELECTED_MODE = "EXPLOIT_LOCATIONS"
    td_learning = False 
    start_loc = [1, 1] # goal -7, 7
    randomize_start_loc = False
    use_prox_mod = False

    # Scale combinations to choose from:
    multiscale = ["small", "medium", "large"]  # Use all 3 scales
    small = ["small"]                         # Just small scale
    medium = ["medium"]                       # Just medium scale
    large = ["large"]                         # Just large scale
    xlarge = ["xlarge"]
    
    scale_names = multiscale  # what scales you are using
    run_time_hours = .1
    max_dist = 25
    plot_bvc = False

    enable_ojas = True
    enable_stdp = False

    # Define common goals for multi-goal modes
    goals = [
        {"name": "red", "location": [7, 7], "radius": 1.5},
        {"name": "green", "location": [-7, 7], "radius": 1.5},
        {"name": "blue", "location": [7, -7], "radius": 1.5},
        {"name": "yellow", "location": [-7, -7], "radius": 1.5}
    ]
    goals_exp = [
        {"name": "red", "location": [7, 7], "radius": 0.5},
        {"name": "green", "location": [-7, 7], "radius": 0.5},
        {"name": "blue", "location": [7, -7], "radius": 0.5},
        {"name": "yellow", "location": [-7, -7], "radius": 0.5}
    ]

    MODE_PARAMS = {
        "OJAS": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc, 
            "clear_files": True,
            "action_mode": 'explore',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "HEBB": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'explore',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "LEARNING": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc,
            "clear_files": True,
            "action_mode": 'explore',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "DMTP": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'explore',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "DMTP_EXPLOIT": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "td_learning": True,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'exploit_timed_fallback',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "EXPLOIT": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "td_learning": td_learning,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'exploit_v1',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "EXPLOIT_SAVE": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "clear_files": False,
            "action_mode": 'exploit_v0',
            "save_data": True,
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 20,
                "start_locations": [start_loc]
            }
        },
        "LEARNING_SAVE": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "td_learning": True,
            "clear_files": False,
            "action_mode": 'exploit_v2',
            "save_data": True,
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 51,
                "start_locations": [start_loc]
            }
        },
        "PLOTTING": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'explore',
            "goal_config": {
                "type": "single",
                "location": [-7, 7],
                "radius": 0.7,
                "name": "default"
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "LEARN_LOCATIONS": {
            "scale_names": scale_names,
            "enable_ojas": True,
            "enable_stdp": True,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": 'explore',
            "goal_config": {
                "type": "multi",
                "goals": goals_exp
            },
            "trial_config": {
                "type": "simple",
                "count": 1,
                "start_locations": [start_loc]
            }
        },
        "EXPLOIT_LOCATIONS": {
            "scale_names": scale_names,
            "enable_ojas": False,
            "enable_stdp": False,
            "run_time_hours": run_time_hours,
            "max_dist": max_dist,
            "td_learning": td_learning,
            "use_prox_mod": use_prox_mod,
            "plot_bvc": plot_bvc,
            "clear_files": False,
            "action_mode": "exploit_locations",
            "save_data": True,
            "goal_config": {
                "type": "multi",
                "goals": goals,
                "target_goal": "red"  # Which goal to navigate to
            },
            "trial_config": {
                "type": "combinations",
                "count": 5,
                "start_locations": [[7, -7], [-7, -7], [7, 7], [-7, 7], [0,0]],
                "target_goals": ["red", "green", "blue", "yellow"]
            }
        }
    }
    
    if SELECTED_MODE not in MODE_PARAMS or SELECTED_MODE not in MODES_MAP:
        print("Invalid mode selected.")
        sys.exit(1)

    # Set global variables for parameter saving
    CURRENT_SELECTED_MODE = SELECTED_MODE
    CURRENT_MODE_PARAMS = MODE_PARAMS[SELECTED_MODE]

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[SELECTED_MODE]
    params = MODE_PARAMS[SELECTED_MODE]
    
    print(f"[INFO] Starting run with mode: {SELECTED_MODE}")
    print(f"[INFO] Scale names: {params['scale_names']}")
    
    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)