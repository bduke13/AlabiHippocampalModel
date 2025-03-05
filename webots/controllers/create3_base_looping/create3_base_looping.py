# NOTE - Only tested on learn Ojas. May have some unintended consequences for multiple exploit sessions
# May need to use bot.worldReload() instead but the for loop will no longer work as intended
import os

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode
from webots.controllers.create3_base.driver import Driver

bot = Driver()

for x in range(5):

    # 1. LEARN_OJAS
    # 2. LEARN_HEBB
    # 3. DMTP
    # 4. EXPLOIT
    # (optional) PLOTTING

    world_path = bot.getWorldPath()  # Get the full path to the world file
    world_name = os.path.splitext(os.path.basename(world_path))[
        0
    ]  # Extract just the world name

    world_name += f"_{x}"

    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        run_time_hours=0.05,
        randomize_start_loc=False,
        start_loc=[0, 0],
        goal_location=[1, 1],
        max_dist=6,
        show_bvc_activation=False,
        world_name=world_name,
        show_save_dialogue_and_pause=False,
    )

    bot.run()
    bot.simulationReset()

bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
