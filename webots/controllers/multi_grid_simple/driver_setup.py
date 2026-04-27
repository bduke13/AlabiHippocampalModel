import pickle

import numpy as np
import torch

from layers.hdn import HeadDirectionLayer
from webots_control import randomize_robot_pose, set_robot_pose


def initialize_runtime_state(
    driver,
    *,
    run_time_hours: int,
    goal_location,
    max_dist: float,
    show_bvc_activation: bool,
) -> None:
    driver.show_bvc_activation = show_bvc_activation

    driver.num_place_cells = 500
    driver.num_bvc_per_dir = 50
    driver.sigma_r = 0.5
    driver.sigma_theta = 1
    driver.n_hd = 8
    driver.num_grid_modules = 8
    driver.num_grid_cells_per_module = 50
    driver.grid_spread_range = (1.2, 1.2)
    driver.grid_scale_multiplier = 1.0
    driver.grid_translation_scale = 1.0
    driver.grid_threshold = 0.7
    driver.timestep = 32 * 3
    driver.tau_w = 5

    driver.max_speed = 16
    driver.max_dist = max_dist
    driver.left_speed = driver.max_speed
    driver.right_speed = driver.max_speed
    driver.wheel_radius = 0.031
    driver.axle_length = 0.271756
    driver.run_time_minutes = run_time_hours * 60
    driver.num_steps = int(driver.run_time_minutes * 60 // (2 * driver.timestep / 1000))
    driver.goal_r = {"explore": 0.3, "exploit": 0.5}
    driver.goal_location = goal_location if goal_location is not None else [-3, 3]


def initialize_robot_pose(
    driver,
    *,
    randomize_start_loc: bool,
    start_loc,
    start_rotation=None,
) -> None:
    if randomize_start_loc:
        randomize_robot_pose(
            driver.robot,
            goal_location=driver.goal_location,
            rotation=start_rotation,
        )
        return

    if start_loc is None:
        raise ValueError("start_loc must be provided when randomize_start_loc is False.")
    set_robot_pose(driver.robot, start_loc, rotation=start_rotation)


def initialize_devices(driver) -> None:
    driver.compass = driver.getDevice("compass")
    driver.compass.enable(driver.timestep)
    driver.range_finder = driver.getDevice("range-finder")
    driver.range_finder.enable(driver.timestep)
    driver.left_bumper = driver.getDevice("bumper_left")
    driver.left_bumper.enable(driver.timestep)
    driver.right_bumper = driver.getDevice("bumper_right")
    driver.right_bumper.enable(driver.timestep)
    driver.collided = torch.zeros(2, dtype=torch.int32)
    driver.left_motor = driver.getDevice("left wheel motor")
    driver.right_motor = driver.getDevice("right wheel motor")
    driver.left_position_sensor = driver.getDevice("left wheel sensor")
    driver.left_position_sensor.enable(driver.timestep)
    driver.right_position_sensor = driver.getDevice("right wheel sensor")
    driver.right_position_sensor.enable(driver.timestep)

    driver.lidar_resolution = 720
    driver.boundaries = torch.zeros((driver.lidar_resolution, 1), device=driver.device)


def initialize_histories(driver) -> None:
    loaded_loc = None
    loaded_pcn = None
    loaded_bvc = None
    loaded_hdn = None
    loaded_gcn = None
    loaded_steps = 0

    load_hmaps_from_run_id = getattr(driver, "load_hmaps_from_run_id", None)
    if load_hmaps_from_run_id and getattr(driver, "hmap_load_dir", None):
        load_dir = driver.hmap_load_dir
        try:
            with open(load_dir / "hmap_loc.pkl", "rb") as input_file:
                loaded_loc = np.asarray(pickle.load(input_file))
            with open(load_dir / "hmap_pcn.pkl", "rb") as input_file:
                loaded_pcn = np.asarray(pickle.load(input_file))
            with open(load_dir / "hmap_bvc.pkl", "rb") as input_file:
                loaded_bvc = np.asarray(pickle.load(input_file))
            with open(load_dir / "hmap_hdn.pkl", "rb") as input_file:
                loaded_hdn_raw = pickle.load(input_file)
                if isinstance(loaded_hdn_raw, torch.Tensor):
                    loaded_hdn = loaded_hdn_raw.detach().cpu()
                else:
                    loaded_hdn = torch.as_tensor(loaded_hdn_raw, dtype=torch.float32, device="cpu")
            gcn_path = load_dir / "hmap_gcn.pkl"
            if gcn_path.exists():
                with open(gcn_path, "rb") as input_file:
                    loaded_gcn = np.asarray(pickle.load(input_file))

            loaded_steps = min(
                len(loaded_loc),
                len(loaded_pcn),
                len(loaded_bvc),
                len(loaded_hdn),
                len(loaded_gcn) if loaded_gcn is not None else len(loaded_hdn),
            )
            print(f"Loaded {loaded_steps} history steps from run {load_hmaps_from_run_id}")
        except FileNotFoundError:
            print(
                f"Could not load hmaps from prior run {load_hmaps_from_run_id} at {load_dir}; "
                "starting with empty histories instead."
            )
            loaded_loc = None
            loaded_pcn = None
            loaded_bvc = None
            loaded_hdn = None
            loaded_gcn = None
            loaded_steps = 0

    additional_steps = driver.num_steps
    total_steps = additional_steps + loaded_steps
    driver.num_steps = total_steps

    driver.hmap_loc = np.zeros((total_steps, 3))
    driver.hmap_pcn = torch.zeros(
        (total_steps, driver.pcn.num_pc),
        device=driver.device,
        dtype=torch.float32,
    )
    driver.hmap_bvc = torch.zeros(
        (total_steps, driver.pcn.bvc_layer.num_bvc),
        device=driver.device,
        dtype=torch.float32,
    )
    driver.hmap_hdn = torch.zeros(
        (total_steps, driver.n_hd),
        device="cpu",
        dtype=torch.float32,
    )
    driver.hmap_gcn = torch.zeros(
        (total_steps, driver.gcn.total_grid_cells),
        device=driver.device,
        dtype=torch.float32,
    )
    driver.directional_reward_estimates = torch.zeros(driver.n_hd, device=driver.device)
    driver.history_step_count = loaded_steps if loaded_steps > 0 else 1

    if loaded_steps > 0:
        if loaded_pcn.shape[1] != driver.pcn.num_pc:
            raise ValueError(
                f"Loaded PCN hmap width {loaded_pcn.shape[1]} does not match current num_pc {driver.pcn.num_pc}."
            )
        if loaded_bvc.shape[1] != driver.pcn.bvc_layer.num_bvc:
            raise ValueError(
                "Loaded BVC hmap width does not match current BVC count."
            )
        if loaded_hdn.shape[1] != driver.n_hd:
            raise ValueError(
                f"Loaded HDN hmap width {loaded_hdn.shape[1]} does not match current n_hd {driver.n_hd}."
            )
        if loaded_gcn is not None and loaded_gcn.shape[1] != driver.gcn.total_grid_cells:
            raise ValueError(
                "Loaded GCN hmap width does not match current grid-cell count."
            )

        driver.hmap_loc[:loaded_steps] = loaded_loc[:loaded_steps]
        driver.hmap_pcn[:loaded_steps] = torch.as_tensor(
            loaded_pcn[:loaded_steps],
            dtype=torch.float32,
            device=driver.device,
        )
        driver.hmap_bvc[:loaded_steps] = torch.as_tensor(
            loaded_bvc[:loaded_steps],
            dtype=torch.float32,
            device=driver.device,
        )
        driver.hmap_hdn[:loaded_steps] = loaded_hdn[:loaded_steps].to(dtype=torch.float32, device="cpu")
        if loaded_gcn is not None:
            driver.hmap_gcn[:loaded_steps] = torch.as_tensor(
                loaded_gcn[:loaded_steps],
                dtype=torch.float32,
                device=driver.device,
            )


def initialize_head_direction_layer(driver) -> None:
    driver.head_direction_layer = HeadDirectionLayer(
        num_cells=driver.n_hd,
        device=torch.device("cpu"),
    )
