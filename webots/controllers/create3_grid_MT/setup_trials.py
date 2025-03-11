"""setup_trials.py - Configuration for specific grid cell experiments."""

from trial_generator import generate_trials
from core.robot.robot_mode import RobotMode

# Define base parameters (matching create3_grid_MT.py)
base_params = {
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
    "sigma_d": 0.7,
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
    "scale_multiplier": 5.0,
    "frequency_divisor": 1.0,
    
    # Runtime and data-saving parameters
    "save_trial_data": True,
    "disable_save_popup": True,  # For automation in multi-trial mode
}

# Define parameter variations as requested
param_variations = {
    "sigma_d": [1.2, 1.5],
    "grid_influence": [0.3, 0.5, 0.7],
    "scale_multiplier": [3, 3.5]
    
}

# Generate trials using full grid search
# This will create 6 × 4 × 6 = 144 trials
trial_list = generate_trials(base_params, param_variations, strategy="grid")

# Alternative: Generate fewer trials by varying one parameter at a time
# trial_list = generate_trials(base_params, param_variations, strategy="one-at-a-time")

# Print summary
print(f"Generated {len(trial_list)} trials")
for i, trial in enumerate(trial_list[:5]):  # Print first 5 trials
    print(f"Trial {i+1}: {trial['trial_name']}")
print("...")

# The trial_list can be imported by create3_grid_MT.py and multi_trial_controller.py