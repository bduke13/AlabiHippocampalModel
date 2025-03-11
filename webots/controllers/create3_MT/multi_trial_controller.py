""""multi_trial_controller for running multiple trials in succession."""
import sys
from pathlib import Path
import itertools
import numpy as np

# Remove popups.
# (No tkinter imports are needed.)

# Set project root and add it to sys.path so that modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.robot.robot_mode import RobotMode
from driver_multi_tiral import MultiTrialDriver

def generate_trial_list(grid_params, fixed_params, name_func=None):
    """
    Generate a list of trial dictionaries based on a grid search over parameters.
    
    Args:
        grid_params (dict): Dictionary where each key is a parameter name and the value is a list of values to try.
            For example: {"sigma_ang": [0.01, 0.02, 0.03, 0.04], "sigma_d": [0.1, 0.2, 0.3]}
        fixed_params (dict): Dictionary of parameters that remain fixed for all trials.
        name_func (callable, optional): A function that accepts a dictionary of grid parameters (the variation)
            and returns a string to be used as the trial name. If None, a default naming function is used.
    
    Returns:
        list: A list of dictionaries, one for each trial. Each dictionary is a merger of fixed_params
              and one combination of grid_params, plus a key "trial_name" with the name generated.
    """
    # Default naming function: creates a string by sorting keys alphabetically.
    if name_func is None:
        def name_func(variation):
            sorted_items = sorted(variation.items())
            return "_".join(f"{key}_{value}" for key, value in sorted_items)
    
    trial_list = []
    # Create a product of all grid parameter values.
    keys = list(grid_params.keys())
    for combo in itertools.product(*(grid_params[key] for key in keys)):
        variation = dict(zip(keys, combo))
        trial_name = name_func(variation)
        # Add the trial name to the variation.
        variation["trial_name"] = trial_name
        # Merge fixed and variable parameters.
        trial = fixed_params.copy()
        trial.update(variation)
        trial_list.append(trial)
    
    return trial_list

# --- Example Usage ---

# Grid search parameters. You can define them in a format like:
grid_params = {
    "sigma_ang": [round(val, 2) for val in np.arange(1, 3.1, 0.5)],  # 1.0, 1.5, 2.0, 2.5, 3.0
    "sigma_d": [round(val, 2) for val in np.arange(0.5, 4.1, 0.5)]   # 0.5, 1.0, 1.5, ..., 5.0
    #"sigma_ang": [round(val, 2) for val in np.arange(1, 3.1, 0.5)],  # 1.0, 1.5, 2.0, 2.5, 3.0
    #"sigma_d": [round(val, 2) for val in np.arange(0.5, 1.1, 0.2)]  # 0.5, 0.7, 0.9, 1.0
}
#ang 1 - 5
#d .5 - 5

# Fixed parameters that remain the same for all trials.
fixed_params = {
    "environment_label": "TestEnvironment",
    "max_runtime_hours": 15,           # 10 hours per trial
    "randomize_start_loc": False,      # random start is off
    "start_loc": [4, -4],
    "goal_location": [-3, 3],
    "mode": RobotMode.LEARN_OJAS,              # or use an appropriate enum from RobotMode
    "movement_method": "default",
    "max_dist": 30,
    "num_bvc_per_dir": 50,
    "num_place_cells": 1500,
    "n_hd": 8,
    "save_trial_data": True,
    "disable_save_popup": True         # For multi-trial mode
}

# Custom naming function for more control over trial names.
def custom_trial_name(variation):
    return f"sAng_{variation['sigma_ang']}_sD_{variation['sigma_d']}"

# Generate the trial list.
trial_list = generate_trial_list(grid_params, fixed_params, name_func=custom_trial_name)

print(f"Total trials generated: {len(trial_list)}")
# Now trial_list is ready to be passed to your multi-trial controller.


def run_trials(trial_list):
    # default_params provides a baseline configuration to create the initial Driver instance.
    # Since only one Robot instance is allowed per controller process, we create a single driver.
    # Then, before each trial, we call bot.trial_setup(trial_params) to update its parameters and reset its state.
    default_params = {
        "environment_label": "default",
        "max_runtime_hours": 0.5,
        "randomize_start_loc": True,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "mode": trial_list[0].get("mode"),
        "movement_method": "default",
        "sigma_ang": 0.01,
        "sigma_d": 0.5,
        "max_dist": 15,
        "num_bvc_per_dir": 50,
        "num_place_cells": 500,
        "n_hd": 8,
        "save_trial_data": True,
        "trial_name": "default_trial",
        "run_multiple_trials": False,  # Always false internally.
        "disable_save_popup": True
    }
    bot = MultiTrialDriver(**default_params)
    
    for trial_params in trial_list:
        print(f"Starting trial with parameters: {trial_params}")
        bot.trial_setup(trial_params)
        # Record the trial's start time so that runtime is measured relative to it.
        bot.trial_start_time = bot.getTime()
        bot.run_trial()
        # Resume simulation mode between trials.
        bot.simulationSetMode(bot.SIMULATION_MODE_FAST)
        print("Trial complete. Proceeding to next trial...\n")

    # After all trials are complete, pause the simulation.
    print("All trials have been completed. Pausing simulation.")
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

if __name__ == "__main__":
    run_trials(trial_list)
    print("All trials have been completed.")

