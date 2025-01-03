"""my_controller_iCreate controller."""

import os
from driver import Driver, RobotMode
from analysis.stats_collector import stats_collector

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
    bot.run()

# Function to handle EXPLOIT modes
def run_exploit(corners, enable_multiscale, save_data=False):
    bot = Driver()
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

    for corner in corners:
        corner_tuple = tuple(corner)
        bot.trial_indices[corner_tuple] = 0

    for corner in corners:
        corner_tuple = tuple(corner)
        for run in range(10):
            print(f"Running trial at corner {corner}, run {run + 1}...")
            bot.trial_indices[corner_tuple] = run + 1
            trial_id = f"trial_{run + 1}_corner_{corner[0]}_{corner[1]}"

            # Initialize stats collector only if saving is enabled
            stats_collector_instance = None
            if save_data:
                stats_collector_instance = stats_collector(output_dir=stats_folder)

            bot.initialization(
                mode=RobotMode.EXPLOIT,
                run_time_hours=1,
                start_loc=corner,
                randomize_start_loc=False,
                stats_collector=stats_collector_instance,
            )
            bot.trial_id = trial_id

            bot.run()
            bot.simulationReset()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

################################# START HERE ##################################
    
# Define the mode to run
SELECTED_MODE = "EXPLOIT_SAVE"  # Options: LEARN_OJAS, DMTP, EXPLOIT, EXPLOIT_SAVE, PLOTTING
run_time_hours = 1
start_loc = [2, 2]
randomize_start_loc = True
enable_multiscale = True

corners = [[2, -2], [-2, -2], [2, 2], [-2, 2]]

# Main function
if __name__ == "__main__":

    if SELECTED_MODE == "LEARN_OJAS":
        run_learn_or_dmtp(mode=RobotMode.LEARN_OJAS, 
                    run_time_hours=run_time_hours,
                    start_loc=corners[1], 
                    randomize_start_loc=randomize_start_loc,
                    enable_multiscale=enable_multiscale)
    if SELECTED_MODE == "DMTP":  # LEARN_OJAS
        run_learn_or_dmtp(mode=RobotMode.DMTP, 
                          run_time_hours=run_time_hours,
                          start_loc=corners[3], 
                          randomize_start_loc=False,
                          enable_multiscale=enable_multiscale)
    elif SELECTED_MODE == "EXPLOIT":  # Non-data-saving EXPLOIT
        run_exploit(corners, enable_multiscale, save_data=False)
    elif SELECTED_MODE == "EXPLOIT_SAVE":  # Data-saving EXPLOIT
        run_exploit(corners, enable_multiscale, save_data=True)
    elif SELECTED_MODE == "PLOTTING":  # PLOTTING
        run_learn_or_dmtp(mode=RobotMode.PLOTTING, start_loc=start_loc, enable_multiscale=enable_multiscale)
    else:
        print("Invalid mode selected.")
