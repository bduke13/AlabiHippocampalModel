"""my_controller_iCreate controller."""

from driver import Driver

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.webotsWheeledRobot import Create3Supervisor

my_supervisor = Create3Supervisor()
print(type(my_supervisor))

from core.robot.robot_mode import RobotMode


bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_loc=True,
    run_time_hours=2,
    start_loc=[4, -4],
)
bot.run()
