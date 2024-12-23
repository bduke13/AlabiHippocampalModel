"""my_controller_iCreate controller."""

from driver import Driver, RobotMode
from analysis.stats_collector import stats_collector

# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot = Driver()
stats_collector = stats_collector(output_dir="controllers\\bek_controller\\analysis\\stats")

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_loc=True,
    run_time_hours=2,
    dmtp_run_time_hours=0.5,
    start_loc=[2, -2],
    enable_multiscale=True,
    stats_collector=stats_collector
)
bot.run()
