"""my_controller_iCreate controller."""

from driver_vertical import Driver, RobotMode

bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    mode=RobotMode.PLOTTING,
    randomize_start_loc=True,
    run_time_hours=2,
    start_loc=[4, -4],
)
bot.run()
