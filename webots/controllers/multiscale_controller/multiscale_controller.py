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
        "scale_index": 0,
        "name": "small",
        "num_pc": 2000,
        "sigma_r": 0.5,
        "sigma_theta": 90,
        "rcn_learning_rate": 0.1,
    },
    "medium": {
        "scale_index": 1,
        "name": "medium",
        "num_pc": 500,
        "sigma_r": 2,
        "sigma_theta": 90,
        "rcn_learning_rate": 0.1,
    },
    "large": {
        "scale_index": 2,
        "name": "large",
        "num_pc": 250,
        "sigma_r": 4,
        "sigma_theta": 90,
        "rcn_learning_rate": 0.1,
    },
    "xlarge": {
        "scale_index": 3,
        "name": "xlarge",
        "num_pc": 200,
        "sigma_r": 4,
        "sigma_theta": 8,
        "rcn_learning_rate": 0.005,
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
    bot = Driver()
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

            bot.initialization(
                mode=mode,
                run_time_hours=kwargs.get("run_time_hours", 2),
                randomize_start_loc=kwargs.get("randomize_start_loc", True),
                start_loc=kwargs.get("start_loc", corner),
                enable_ojas=kwargs.get("enable_ojas", None),
                enable_stdp=kwargs.get("enable_stdp", None),
                scales=scales_list,
                rcn_learning_rates=rcn_learning_rates, 
                stats_collector=stats_collector_instance,
                trial_id=trial_id,
                world_name=world_name,
                goal_location=kwargs.get("goal_location", None),
                max_dist=kwargs.get("max_dist", 10),
                plot_bvc=kwargs.get("plot_bvc", False),
                td_learning=kwargs.get("td_learning", False),
                use_prox_mod=kwargs.get("use_prox_mod", False),
            )

            bot.trial_id = trial_id
            bot.run()

            # If in exploit mode, reload the world between runs
            if mode == RobotMode.EXPLOIT and save_data:
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
    
    SELECTED_MODE = "DMTP"
    td_learning = False
    corners = [[8,-8]]
    dmtp_start = [-9,9]
    exploit_start = corners[0]
    
    start_loc = dmtp_start
    
    goal_location = [-7, 7]    
    randomize_start_loc = False
    use_prox_mod = False

    multiscale = ["small", "medium", "large"]
    small = ["small"]
    medium = ["medium"]
    large = ["large"]
    
    scale_names = large
    run_time_hours = 4
    max_dist = 25
    plot_bvc = False

    enable_ojas = False
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
            "enable_ojas": False,
            "enable_stdp": False,
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

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[SELECTED_MODE]
    params = MODE_PARAMS[SELECTED_MODE]
    
    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)
