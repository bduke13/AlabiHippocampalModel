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
from msg_driver_v65 import MultiscaleDriverWithGrid
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
        "sigma_r": 0.4,
        "sigma_theta": 1.0,
        "rcn_learning_rate": 0.1,
        "gamma_pp": 0.5, # 0.5
        "gamma_pb": 0.2, # 0.3
        # Grid cell parameters
        "grid_influence": 0.25,  # 0.2
        "gamma_pg": 0.3, # 0.3
        "num_grid_cells": 800,
        "rotation_range": (0, 360),
        "spread_range": (1.0, 1.0),
        "translation_factor": 100.0,
        "frequency_divisor": 0.25,  # Smallest grid scale (high frequency)
        # Replay Params
        "replay_timesteps": 35, # 35
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
        "rotation_range": (0, 360),
        "spread_range": (1.0, 1.0),
        "translation_factor": 200.0,
        "frequency_divisor": 0.35,  # Medium grid scale 0.5
        # Replay Params
        "replay_timesteps": 30, # 30
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
        "rotation_range": (0, 360),
        "spread_range": (1.0, 1.0),
        "translation_factor": 400.0,
        "frequency_divisor": 0.55,  # Larger grid scale (lower frequency) # 1.0
        # Replay Params
        "replay_timesteps": 20, # 20
        "replay_decay_constant": 5,
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
        "replay_timesteps": 15, # 15
        "replay_decay_constant": 5,
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
    }
}

def compile_scales(scale_names):
    """
    Convert a list of scale names (e.g. ["small", "large"]) into a list of 
    actual scale definitions from SCALES_DEFS.
    """
    return [SCALES_DEFS[name] for name in scale_names]


#################################
# run_bot
#################################

def run_bot(mode, corners=None, save_data=False, **kwargs):
    """
    Runs the bot in the specified mode with the given parameters.
    - If `mode` == EXPLOIT and `save_data=True`, multiple loops may run.
    - If `enable_multiscale=True`, we load more than one scale if scale_names 
      has more than one entry, or we can do single-scale if it has exactly one.
    """
    bot = MultiscaleDriverWithGrid()
    world_name = get_world_name(bot)
    print(f"[INFO] Current world: {world_name}")

    # Grab scale names from kwargs (with fallback to empty list)
    scale_names = kwargs.get("scale_names", [])

    # Convert scale names => scale definitions
    scales_list = compile_scales(scale_names)

    # Build a string like "small_medium_large"
    # This will be used instead of "3_scales"
    scale_name_str = "_".join(scale_names)

    # Decide how many loops/trials to run
    num_loops = kwargs.get("num_loops", 1)

    # If we're in exploit mode, optionally enable data saving
    if mode == RobotMode.EXPLOIT:
        if save_data:
            # Build path: analysis/stats/<world_name>/<scale_name_str>/JSON
            stats_folder = os.path.join(
                PROJECT_ROOT, 
                "analysis", 
                "stats", 
                world_name,
                scale_name_str,   # <--- Use the joined scale names here
                "JSON"
            )
            os.makedirs(stats_folder, exist_ok=True)
            stats_collector_instance = stats_collector(output_dir=stats_folder)
        else:
            stats_folder = None
            stats_collector_instance = None
    else:
        stats_folder = None
        stats_collector_instance = None

    # If no corners provided, default to a single corner
    if corners is None:
        corners = [[0, 0]]

    for corner in corners:
        corner_tuple = tuple(corner)
        bot.trial_indices = {}

        # Check how many times we might run for this corner
        if save_data and mode == RobotMode.EXPLOIT:
            # If saving data, figure out the latest trial # for this corner
            current_trial_id = get_highest_trial_id(stats_folder, corner)
        else:
            current_trial_id = 0

        # If we've already run enough trials, skip
        if current_trial_id >= num_loops:
            continue

        for _ in range(num_loops):
            current_trial_id += 1
            bot.trial_indices[corner_tuple] = current_trial_id
            trial_id = f"trial_{current_trial_id}_corner_{corner[0]}_{corner[1]}"

            if save_data:
                print(f"[INFO] Running trial: {trial_id}")

            # Extract RCN learning rates from the scale definitions
            rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]
            replay_timesteps = [scale["replay_timesteps"] for scale in scales_list]
            replay_decay_constants = [scale["replay_decay_constant"] for scale in scales_list]

            bot.initialization(
                mode=mode,
                run_time_hours=kwargs.get("run_time_hours", 2),
                randomize_start_loc=kwargs.get("randomize_start_loc", True),
                start_loc=kwargs.get("start_loc", corner),
                enable_ojas=kwargs.get("enable_ojas", None),
                enable_stdp=kwargs.get("enable_stdp", None),
                scales=scales_list,
                rcn_learning_rates=rcn_learning_rates,
                replay_timesteps=replay_timesteps,
                replay_decay_constants=replay_decay_constants,
                stats_collector=stats_collector_instance,
                trial_id=trial_id,
                world_name=world_name,
                goal_location=kwargs.get("goal_location", None),
                max_dist=kwargs.get("max_dist", 25),  # Use default value
                plot_bvc=kwargs.get("plot_bvc", False),
                td_learning=kwargs.get("td_learning", False),
                use_prox_mod=kwargs.get("use_prox_mod", False),
            )

            bot.trial_id = trial_id
            
            # NEW: Save parameters right before starting the run
            # At this point we have all the information we need
            print(f"[INFO] About to save run parameters for trial: {trial_id}")
            save_run_parameters(
                world_name=world_name,
                selected_mode=CURRENT_SELECTED_MODE,  # This will be set globally
                mode_params=CURRENT_MODE_PARAMS,      # This will be set globally  
                scales_defs=SCALES_DEFS,
                scale_names=scale_names,
                trial_id=trial_id
            )
            
            # Start the actual run
            bot.run()

            # If in exploit mode, reload the world between runs
            if mode == RobotMode.EXPLOIT and save_data:
                bot.worldReload()

    # Pause the sim
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
        "LEARN_OJAS": RobotMode.LEARN_OJAS,
        "LEARN_HEBB": RobotMode.LEARN_HEBB,
        "LEARN_COMBINED": RobotMode.LEARN_HEBB,
        "DMTP": RobotMode.DMTP,
        "EXPLOIT": RobotMode.EXPLOIT,
        "EXPLOIT_SAVE": RobotMode.EXPLOIT,
        "LEARNING_SAVE": RobotMode.EXPLOIT,
        "PLOTTING": RobotMode.PLOTTING  
    }
    
    SELECTED_MODE = "DMTP"
    td_learning = True 
    corners = [[-8,8]] # start point
    dmtp_start = [-9,9]
    exploit_start = corners[0]
    start_loc = [-7, 7]
    
    goal_location = [-7, 7]    
    randomize_start_loc = False
    use_prox_mod = False

    # Scale combinations to choose from:
    multiscale = ["small", "medium", "large", "xlarge"]  # Use all 3 scales
    small = ["small"]                         # Just small scale
    medium = ["medium"]                       # Just medium scale
    large = ["large"]                         # Just large scale
    xlarge = ["xlarge"]
    
    scale_names = multiscale  # what scales you are using
    run_time_hours = 4
    max_dist = 25
    plot_bvc = False

    enable_ojas = True
    enable_stdp = False

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
        # NEW: Combined learning mode
        "LEARN_COMBINED": {
            "corners": corners,
            "start_loc": start_loc,
            "goal_location": goal_location,
            "max_dist": max_dist,
            "randomize_start_loc": randomize_start_loc,
            "scale_names": scale_names,
            "enable_ojas": True,      # Enable place field formation
            "enable_stdp": True,      # Enable connection formation
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
            "num_loops": 1, 
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