"""my_controller_iCreate controller."""

import tensorflow as tf
from driver_vertical import Driver, RobotMode
from visualizations.overlayed_cells import plot_overlayed_cells

bot = Driver()
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING

sigma_distances = [x / 50 for x in range(1, 51, 1)]
sigma_angles = [y / 50 for y in range(1, 51, 1)]

for test_index in range(1):
    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        randomize_start_loc=True,
        run_time_hours=2,
        start_loc=[4, -4],
        # sigma_d=[sigma_distances[test_index]],
        # sigma_a=[sigma_angles[test_index]],
    )
    bot.run()
    plot_overlayed_cells(
        hmap_x_path="hmap_x.pkl",
        hmap_y_path="hmap_y.pkl",
        hmap_z_path="hmap_z.pkl",
        colors_path="visualizations/colors.json",
        gridsize=50,
        save_plot=False,
        suffix=f"_{str(test_index)}_50",
    )

    plot_overlayed_cells(
        hmap_x_path="hmap_x.pkl",
        hmap_y_path="hmap_y.pkl",
        hmap_z_path="hmap_z.pkl",
        colors_path="visualizations/colors.json",
        gridsize=100,
        save_plot=False,
        suffix=f"_{str(test_index)}_100",
    )
    bot.simulationReset()

bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
