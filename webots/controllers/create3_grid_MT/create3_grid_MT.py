"""create3_grid_MT.py - Controller for iCreate robot with grid cells and multi-trial support."""

# Add root directory to python path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves three levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode
from driver_with_grid_MT import DriverGridMT

# ----- Global flag to choose multi-trial mode -----
MULTI_TRIAL_MODE = False  # Set to True to run multiple trials

if MULTI_TRIAL_MODE:
    # Delegate control to the multi-trial routine
    from multi_trial_controller import run_trials, trial_list
    run_trials(trial_list)
else:
    # Single-trial mode with grid cell parameters
    trial_params = {
        # Environment parameters
        "environment_label": "GridEnv",
        "max_runtime_hours": 3,
        "randomize_start_loc": False,
        "start_loc": [0, 0],
        "goal_location": [-3, 3],
        
        # Model/layer standard parameters
        "mode": RobotMode.LEARN_OJAS,
        "movement_method": "default",
        "sigma_ang": 1.0,
        "sigma_d": 1.0,
        "max_dist": 25,
        "num_bvc_per_dir": 50,
        "num_place_cells": 500,
        "n_hd": 8,
        
        # Grid cell parameters
        "num_grid_cells": 400,
        "grid_influence": 0.5,
        "rotation_range": (0, 90),
        "spread_range": (1.2, 1.2),
        "x_trans_range": (-1.0, 1.0),
        "y_trans_range": (-1.0, 1.0),
        "scale_multiplier": 4.0,
        "frequency_divisor": 1.0,
        
        # Runtime and data-saving parameters
        "save_trial_data": True,
        "trial_name": "GridTest_1",
    }
    
    # Create and run the driver
    bot = DriverGridMT(**trial_params)
    bot.run()