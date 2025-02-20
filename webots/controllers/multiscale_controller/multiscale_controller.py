"""my_controller_iCreate controller."""

import sys
import os
import re
from pathlib import Path

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from driver import Driver
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

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
        "name": "small",
        "num_pc": 2000,
        "sigma_r": 0.5,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
    },
    # "medium": {
        # "name": "medium",
        # "num_pc": 1000,
        # "sigma_r": 3,
        # "sigma_theta": 2,
        # "rcn_learning_rate": 0.2,
    # },
    "large": {
        "name": "large",
        "num_pc": 500,
        "sigma_r": 1.5,
        "sigma_theta": 1,
        "rcn_learning_rate": 0.1,
    },
    # "xlarge": {
    #     "name": "xlarge",
    #     "num_pc": 200,
    #     "sigma_r": 4,
    #     "sigma_theta": 4,
    #     "rcn_learning_rate": 0.1,
    # }
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
    bot = Driver()
    world_name = get_world_name(bot)
    print(f"[INFO] Current world: {world_name}")

    stats_collector_instance = None
    num_loops = kwargs.get("num_loops", 1)

    # If we're in exploit mode, optionally enable data saving
    if mode == RobotMode.EXPLOIT:
        if save_data:
            stats_folder = os.path.join(PROJECT_ROOT, "analysis", "stats", world_name, "JSON")
            os.makedirs(stats_folder, exist_ok=True)
            stats_collector_instance = stats_collector(output_dir=stats_folder)
        else:
            stats_folder = None
    else:
        stats_folder = None

    # If no corners provided, default to a single corner
    if corners is None:
        corners = [[0, 0]]

    for corner in corners:
        corner_tuple = tuple(corner)
        bot.trial_indices = {}

        # Check how many times we might run for this corner
        if save_data and mode == RobotMode.EXPLOIT:
            # If data is saved, see how many are already done
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

            # Convert scale names => scale definitions
            scale_names = kwargs.get("scale_names", [])
            scales_list = compile_scales(scale_names)

            rcn_learning_rates = [scale["rcn_learning_rate"] for scale in scales_list]

            bot.initialization(
                mode=mode,
                run_time_hours=kwargs.get("run_time_hours", 2),
                randomize_start_loc=kwargs.get("randomize_start_loc", True),
                start_loc=start_loc,
                enable_ojas=kwargs.get("enable_ojas", None),
                enable_stdp=kwargs.get("enable_stdp", None),
                scales=scales_list,
                rcn_learning_rates=rcn_learning_rates,  # <-- NEW: Pass learning rates list
                stats_collector=stats_collector_instance,
                trial_id=trial_id,
                world_name=world_name,
                goal_location=kwargs.get("goal_location", None),
                max_dist=kwargs.get("max_dist", 10)
            )

            bot.trial_id = trial_id
            print(f"[INFO] Using scales: {scale_names}")
            bot.run()

            # If in exploit mode, reload the world between runs
            if mode == RobotMode.EXPLOIT:
                bot.worldReload()

    # Pause the sim
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
        "PLOTTING": RobotMode.PLOTTING  
    }
    
    SELECTED_MODE = "LEARN_OJAS"
    corners = [[8,-8], [-8,-8], [8,8]]
    start_loc = [0,0]
    goal_location = [-7, 7]    
    randomize_start_loc = False
    # ["small", "medium", "large"]
    scale_names = ["small","large"]
    run_time_hours = 2
    max_dist = 15

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
        },
        "DMTP": {
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
        },
        "EXPLOIT": {
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
            "num_loops": 3, 
            "save_data": True,
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
