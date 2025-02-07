"""my_controller_iCreate controller."""

from math import e
import os
import pickle
from driver import Driver, RobotMode
from analysis.stats_collector import stats_collector
import re

def get_highest_trial_id(stats_folder, corner):
    """
    Reads the stats directory and determines the highest trial ID for the given corner.
    """
    corner_pattern = f"corner_{corner[0]}_{corner[1]}"
    trial_ids = []

    if os.path.exists(stats_folder):
        for file_name in os.listdir(stats_folder):
            match = re.match(rf"trial_(\d+)_corner_{corner[0]}_{corner[1]}", file_name)
            if match:
                trial_ids.append(int(match.group(1)))

    return max(trial_ids) if trial_ids else 0

# Function to detect the current world name dynamically
def get_world_name(bot):
    world_name = bot.getWorldPath()  # This returns the full path
    world_name = world_name.split('/')[-1].replace('.wbt', '')  # Extract just the world name
    return world_name

# Function to handle LEARN_OJAS or DMTP modes
def run_learn_or_dmtp(mode, run_time_hours, start_loc, randomize_start_loc, enable_multiscale):
    print(f"Running in mode: {mode.name}")
    bot = Driver()
    bot.initialization(
        mode=mode,
        run_time_hours=run_time_hours,
        start_loc=start_loc,
        randomize_start_loc=randomize_start_loc,
        enable_multiscale=enable_multiscale,
    )
    bot.trial_id = f"{start_loc[0]}_{start_loc[1]}"
    bot.run()

def run_exploit(corner, enable_multiscale, large_scale_only, save_data=False):
    bot = Driver()
    corner_tuple = tuple(corner)
    bot.trial_indices = {}

    # Dynamically determine the world name
    world_name = get_world_name(bot)
    print(f"Current world: {world_name}")

    # Define the stats folder path based on the world name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if enable_multiscale:
        stats_folder = os.path.join(current_dir, "analysis", "stats", "multiscale", world_name, "JSON")
    else:
        stats_folder = os.path.join(current_dir, "analysis", "stats", "vanilla", world_name, "JSON")
    os.makedirs(stats_folder, exist_ok=True)

    # Read the current highest trial ID for the given corner
    current_trial_id = get_highest_trial_id(stats_folder, corner)
    bot.trial_indices[corner_tuple] = current_trial_id

    # Increment trial index for the given corner
    bot.trial_indices[corner_tuple] += 1
    trial_id = f"trial_{bot.trial_indices[corner_tuple]}_corner_{corner[0]}_{corner[1]}"
    print(f"Running trial: {trial_id}")

    # Initialize stats collector if saving data
    stats_collector_instance = None
    if save_data:
        stats_collector_instance = stats_collector(output_dir=stats_folder)

    # Initialize the bot for the trial
    bot.initialization(
        mode=RobotMode.EXPLOIT,
        run_time_hours=1,
        start_loc=corner,
        enable_multiscale=enable_multiscale,
        large_scale_only=large_scale_only,
        randomize_start_loc=False,
        stats_collector=stats_collector_instance,
    )
    bot.trial_id = trial_id

    # Run the bot
    bot.run()

    # Pause simulation (manual reload expected)
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

def run_exploit_save(corners, enable_multiscale, large_scale_only=False, num_loops=10, save_data=True):
    bot = Driver()

    # Dynamically determine the world name
    world_name = get_world_name(bot)
    print(f"Current world: {world_name}")

    # Define the stats folder path based on the world name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if large_scale_only:
        stats_folder = os.path.join(current_dir, "analysis", "stats", "large_scale", world_name, "JSON")
    else:
        if enable_multiscale:
            stats_folder = os.path.join(current_dir, "analysis", "stats", "multiscale", world_name, "JSON")
        else:
            stats_folder = os.path.join(current_dir, "analysis", "stats", "vanilla", world_name, "JSON")
        os.makedirs(stats_folder, exist_ok=True)

    # Initialize stats collector if saving data
    stats_collector_instance = None
    if save_data:
        stats_collector_instance = stats_collector(output_dir=stats_folder)

    # Loop through corners
    for corner in corners:
        corner_tuple = tuple(corner)
        bot.trial_indices = {}

        # Read the current highest trial ID for the given corner
        current_trial_id = get_highest_trial_id(stats_folder, corner)

        # If the trial count exceeds num_loops, skip this corner
        if current_trial_id >= num_loops:
            continue

        while current_trial_id < num_loops:
            current_trial_id += 1
            bot.trial_indices[corner_tuple] = current_trial_id
            trial_id = f"trial_{current_trial_id}_corner_{corner[0]}_{corner[1]}"
            print(f"Running trial: {trial_id}")

            # Initialize the bot for the trial
            bot.initialization(
                mode=RobotMode.EXPLOIT,
                run_time_hours=1,
                start_loc=corner,
                enable_multiscale=enable_multiscale,
                large_scale_only=large_scale_only,
                randomize_start_loc=False,
                stats_collector=stats_collector_instance,
            )
            bot.trial_id = trial_id

            # Run the bot
            bot.run()
            bot.worldReload()

    # Pause simulation (manual reload expected)
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

################################# START HERE ##################################
    
# Define the mode to run
SELECTED_MODE = "EXPLOIT_SAVE"  # Options: LEARN_OJAS, DMTP, EXPLOIT, EXPLOIT_SAVE, PLOTTING
run_time_hours = 1
num_loops = 10
start_loc = [-9, 9]
randomize_start_loc = False
large_scale_only = False
enable_multiscale = False

# world0_20x20-goalBehindWall
# corners = [[8, -8], [0, -8], [-8, -8], [8, 0], [8, 8], [2, -2]]

# world0_20x20-obstacles
# corners = [[8, -8], [0, -8], [8, 0], [2, -2]]

# world0_20x20
# corners = [[8, -8], [-8, -8], [8, 8]]

# world0_20x20-2obstacles
# corners = [[8, -8], [0, -8], [8, 0]]

corners = [[8,-8]]

# Main function
if __name__ == "__main__":
    selected_corner = corners[0]
    # selected_corner = [4.6, 0.63]
    # selected_corner = [1.45, 0.96]
    if SELECTED_MODE == "LEARN_OJAS":
        run_learn_or_dmtp(mode=RobotMode.LEARN_OJAS, 
                          run_time_hours=run_time_hours,
                          start_loc=corners[1], 
                          randomize_start_loc=randomize_start_loc,
                          enable_multiscale=enable_multiscale)
    elif SELECTED_MODE == "LEARN_HEBB":
        run_learn_or_dmtp(mode=RobotMode.LEARN_HEBB, 
                        run_time_hours=run_time_hours,
                        start_loc=start_loc, 
                        randomize_start_loc=randomize_start_loc,
                        enable_multiscale=enable_multiscale)
    elif SELECTED_MODE == "DMTP":
        run_learn_or_dmtp(mode=RobotMode.DMTP, 
                          run_time_hours=run_time_hours,
                          start_loc=start_loc, 
                          randomize_start_loc=randomize_start_loc,
                          enable_multiscale=enable_multiscale)
    elif SELECTED_MODE == "EXPLOIT":
        run_exploit(corner=selected_corner, 
                    enable_multiscale=enable_multiscale,
                    large_scale_only=large_scale_only,
                    save_data=False)
    elif SELECTED_MODE == "EXPLOIT_SAVE":
        run_exploit_save(corners=corners, 
                         enable_multiscale=enable_multiscale,
                         large_scale_only=large_scale_only, 
                         num_loops=num_loops,
                         save_data=True)
    elif SELECTED_MODE == "PLOTTING":
        run_learn_or_dmtp(mode=RobotMode.PLOTTING, 
                          run_time_hours=run_time_hours,
                          start_loc=start_loc, 
                          randomize_start_loc=randomize_start_loc,
                          enable_multiscale=enable_multiscale)
    else:
        print("Invalid mode selected.")
