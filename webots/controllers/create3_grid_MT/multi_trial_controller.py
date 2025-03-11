"""multi_trial_controller.py - Controller for multi-trial experiments with grid cells."""
import sys
from pathlib import Path

# Set project root and add it to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from driver_with_grid_MT import DriverGridMT

# Import your trial configuration
from setup_trials import trial_list

def run_trials(trial_list):
    """Run multiple trials in sequence using the provided configurations."""
    if not trial_list:
        print("No trials to run - empty trial list")
        return
        
    # Use the first trial's parameters as defaults
    default_params = trial_list[0].copy()
    default_params["disable_save_popup"] = True
    
    # Create a single driver instance that will be reused for all trials
    print(f"Creating driver with default parameters from first trial")
    bot = DriverGridMT(**default_params)
    
    # Run each trial in sequence
    for i, trial_params in enumerate(trial_list):
        print(f"\nRunning trial {i+1} of {len(trial_list)}: {trial_params['trial_name']}")
        
        # Set up the robot and environment for this trial
        bot.trial_setup(trial_params)
        
        # Record the trial's start time
        bot.trial_start_time = bot.getTime()
        
        # Execute the trial until completion
        bot.run_trial()
        
        # Resume fast simulation mode between trials
        bot.simulationSetMode(bot.SIMULATION_MODE_FAST)
        
        print(f"Trial {i+1} completed")
    
    # After all trials complete, pause the simulation
    print("\nAll trials have been completed. Pausing simulation.")
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

if __name__ == "__main__":
    print(f"Running {len(trial_list)} trials...")
    run_trials(trial_list)