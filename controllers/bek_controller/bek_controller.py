"""my_controller_iCreate controller."""

from driver import Driver, RobotMode

bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_loc=True,
    run_time_hours=5,
    start_loc=[2, -2],
)
bot.run()
