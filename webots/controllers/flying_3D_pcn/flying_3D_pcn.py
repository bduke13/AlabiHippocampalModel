"""my_controller_iCreate controller."""

if __name__ == "__main__":
    from driver_3D import DriverFlying

    # Add root directory to python to be able to import
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
    sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

    from core.robot.robot_mode import RobotMode

    run_time_hours = 1

    preferred_va = [-1.2, -0.6, 0.0, 0.6, 1.2]
    sigma_d = [0.25] * len(preferred_va)
    sigma_a = [0.2] * len(preferred_va)
    sigma_va = [0.05] * len(preferred_va)
    num_bvc_per_dir = 25
    visual_bvc = False

    bot = DriverFlying()
    # Create walls with angles for this trial
    # for index, model_type in enumerate(models):

    file_path = f"3D_NAV/"

    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        randomize_start_loc=True,
        run_time_hours=run_time_hours,
        preferred_va=preferred_va,
        sigma_d=sigma_d,
        sigma_va=sigma_va,
        sigma_a=sigma_a,
        num_bvc_per_dir=num_bvc_per_dir,
        file_prefix=file_path,
        visual_bvc=visual_bvc,
    )

    bot.run()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
