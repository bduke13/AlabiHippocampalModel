"""my_controller_iCreate controller."""

from driver import Driver

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode


bot = Driver()
# 1. LEARN_OJAS
# 2. LEARN_HEBB
# 3. DMTP
# 4. EXPLOIT
# (optional) PLOTTING

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    run_time_hours=5,
    randomize_start_loc=False,
    start_loc=[4, -4],
    goal_location=[-3, 3],
    max_dist=20,
    show_bvc_activation=False,
)

bot.run()
