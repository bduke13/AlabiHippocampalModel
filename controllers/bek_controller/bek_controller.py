"""my_controller_iCreate controller."""

import os
from driver import Driver, RobotMode
from analysis.stats_collector import stats_collector

# Define the mode to run
SELECTED_MODE = "EXPLOIT_SAVE"  # Options: LEARN_OJAS, DMTP, EXPLOIT, EXPLOIT_SAVE, PLOTTING

# Function to detect the current world name dynamically
def get_world_name(bot):
    world_name = bot.getWorldPath()  # This returns the full path
    world_name = world_name.split('/')[-1].replace('.wbt', '')  # Extract just the world name
    return world_name

# Function to handle LEARN_OJAS or DMTP modes
def run_learn_or_dmtp(mode, start_loc, run_time_hours=1, dmtp_run_time_hours=0.2, enable_multiscale=False):
    print(f"Running in mode: {mode.name}")
    bot = Driver()

    stats_collector_instance = stats_collector(output_dir="controllers\\bek_controller\\analysis\\stats")
    bot.initialization(
        mode=mode,
        randomize_start_loc=False,
        run_time_hours=run_time_hours,
        dmtp_run_time_hours=dmtp_run_time_hours,
        start_loc=start_loc,
        enable_multiscale=enable_multiscale,
        stats_collector=stats_collector_instance
    )
    bot.run()

# Function to handle EXPLOIT modes
def run_exploit(corners, save_data=False):
    bot = Driver()
    bot.trial_indices = {}

    # Dynamically determine the world name
    world_name = get_world_name(bot)
    print(f"Current world: {world_name}")

    # Define the stats folder path based on the world name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_folder = os.path.join(current_dir, "analysis", "stats", world_name)
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

# Main function
if __name__ == "__main__":

    if SELECTED_MODE == "LEARN_OJAS":  # LEARN_OJAS
        run_learn_or_dmtp(mode=RobotMode.LEARN_OJAS, start_loc=[2, 2])
    elif SELECTED_MODE == "DMTP":  # DMTP
        run_learn_or_dmtp(mode=RobotMode.DMTP, start_loc=[2, 2], dmtp_run_time_hours=0.2, enable_multiscale=True)
    elif SELECTED_MODE == "EXPLOIT":  # Non-data-saving EXPLOIT
        # Define the corners
        corners = [[2, -2], [-2, -2], [2, 2], [-2, 2]]
        run_exploit(corners, save_data=False)
    elif SELECTED_MODE == "EXPLOIT_SAVE":  # Data-saving EXPLOIT
        # Define the corners
        corners = [[2, -2], [-2, -2], [2, 2], [-2, 2]]
        run_exploit(corners, save_data=True)
    elif SELECTED_MODE == "PLOTTING":  # PLOTTING
        run_learn_or_dmtp(mode=RobotMode.PLOTTING, start_loc=[2, 2])
    else:
        print("Invalid mode selected.")
