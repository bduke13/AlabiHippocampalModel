"""my_controller_iCreate controller."""

# from driver import Driver, RobotMode
# from analysis.stats_collector import stats_collector

# # 1. LEARN_OJAS
# # 2. DMTP
# # 3. EXPLOIT
# # 4. (optional) PLOTTING

# bot = Driver()
# stats_collector = stats_collector(output_dir="controllers\\bek_controller\\analysis\\stats")

# bot.initialization(
#     mode=RobotMode.EXPLOIT,
#     randomize_start_loc=False,
#     run_time_hours=1,
#     dmtp_run_time_hours=0.2,
#     start_loc=[2, 2],
#     enable_multiscale=True,
#     stats_collector=stats_collector
# )

# bot.run()

import os
from driver import Driver, RobotMode
from analysis.stats_collector import stats_collector

# Define the corners
corners = [[2, -2], [-2, -2], [2, 2], [-2, 2]]

# Create one Driver instance for the robot
bot = Driver()
bot.trial_indices = {}

# Path to stats folder
current_dir = os.path.dirname(os.path.abspath(__file__))
stats_folder = os.path.join(current_dir, "analysis", "stats")

# Initialize trial indices for all corners
for corner in corners:
    corner_tuple = tuple(corner)  # Use tuple as key for immutability
    bot.trial_indices[corner_tuple] = 0

# Loop over corners
for corner in corners:
    corner_tuple = tuple(corner)  # Convert to tuple for lookup
    for run in range(10):
        print(f"Running trial at corner {corner}, run {run + 1}...")

        # Update trial index before starting the trial
        bot.trial_indices[corner_tuple] = run + 1

        # Construct trial ID
        trial_id = f"trial_{run + 1}_corner_{corner[0]}_{corner[1]}"

        # Reinitialize the same bot each time
        bot.initialization(
            mode=RobotMode.EXPLOIT,
            run_time_hours=1,
            start_loc=corner,  # Keep as list for initialization
            randomize_start_loc=False,
            stats_collector=stats_collector(output_dir=stats_folder)
        )

        # Set the trial ID in the driver
        bot.trial_id = trial_id

        # Run until the goal is reached and stats are saved
        bot.run()

        # Reset the simulation environment for the next trial
        bot.simulationReset()

print("All trials completed.")
