""""create3_MT for iCreate controller."""
import sys
from pathlib import Path

# Set the project root (adjusted for the new folder structure)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.robot.robot_mode import RobotMode
from driver_multi_tiral import MultiTrialDriver

# ----- Global flag to choose multi-trial mode -----
MULTI_TRIAL_MODE = False  # Set to True to run multiple trials

if MULTI_TRIAL_MODE:
    # Delegate control to the multi-trial routine.
    from multi_trial_controller import run_trials, trial_list
    run_trials(trial_list)
else:
    # Single-trial mode.
    trial_params = {
        # Environment parameters:
        "environment_label": "20Maze",
        "max_runtime_hours": 6,  # maximum run time in hours
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        # Model/layer parameters:
        "mode": RobotMode.LEARN_OJAS,
        "movement_method": "default",  # placeholder
        "sigma_ang": 5,
        "sigma_d": .7,
        "max_dist": 10,
        "num_bvc_per_dir": 200,
        "num_place_cells": 500,
        "n_hd": 8,
        # Runtime and data-saving parameters:
        "save_trial_data": True,
        "trial_name": "10x10_Test",  # if "none", a trial folder name will be auto-generated
        "run_multiple_trials": False  # always false in single-trial mode
    }
    bot = MultiTrialDriver(**trial_params)
    bot.run()  # run the single trial as before
