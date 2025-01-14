"""my_controller_iCreate controller."""

from driver import Driver, RobotMode

bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_loc=False,
    run_time_hours=1.5,
    start_loc=[0, 0],
)
bot.run()
