"""my_controller_iCreate controller."""

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    from robot_mode import RobotMode
    from driver_vertical import DriverVertical
    import random

    run_time_hours = 4
    number_tests = 1
    start_locations = [[4, 4], [-4, 4], [4, -4], [-4, -4]]
    models = ["3D_2L_250_1"]
    world_name = "upright_2"

    preferred_va = [0.0, 0.1]
    sigma_d = [0.75] * len(preferred_va)
    sigma_a = [0.1] * len(preferred_va)
    sigma_va = [0.01] * len(preferred_va)
    num_bvc_per_dir = 60

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

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
