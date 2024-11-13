"""my_controller_iCreate controller."""

from driver import Driver, RobotMode

bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    mode=RobotMode.RECORDING,
    randomize_start_loc=True,
    run_time_hours=0.5,
    start_loc=[4, -4],
)
bot.run()
