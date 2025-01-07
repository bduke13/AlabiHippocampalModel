"""my_controller_iCreate controller."""

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    from robot_mode import RobotMode
    from driver_vertical import DriverVertical
    import random
    from visualizations.overlayed_cells import plot_overlayed_cells
    from visualizations.analysis_utils import load_hmaps
    from visualizations.html_pcn import generate_place_cells_report

    run_time_hours = 4
    number_tests = 1
    start_locations = [[4, 4], [-4, 4], [4, -4], [-4, -4]]
    models = ["3D_1L_1"]
    world_name = "outside_2"

    preferred_va = [0.0, 0.3]
    sigma_d = [0.5] * 2
    sigma_a = [0.3] * 2
    sigma_va = [0.01] * 2
    num_bvc_per_dir = 50

    model_type = models[0]

    bot = DriverVertical()
    # Create walls with angles for this trial
    # for index, model_type in enumerate(models):

    print(f"Running model {model_type} in world {world_name}")
    file_path = f"IJCNN/{model_type}/{world_name}/"
    start_location = random.choice(start_locations)

    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        randomize_start_loc=False,
        run_time_hours=run_time_hours,
        preferred_va=preferred_va,
        sigma_d=sigma_d,
        sigma_va=sigma_va,
        sigma_a=sigma_a,
        num_bvc_per_dir=num_bvc_per_dir,
        start_loc=start_location,
        file_prefix=file_path,
    )

    bot.run()

    # plot_overlayed_cells(
    #    hmap_x=bot.hmap_x,
    #    hmap_y=bot.hmap_y,
    #    hmap_z=bot.hmap_z,
    #    gridsize=100,
    #    save_plot=False,
    # )
    # generate_place_cells_report(
    #    bot.hmap_x, bot.hmap_y, bot.hmap_z, output_dir=file_path
    # )

    # bot.worldReload()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
