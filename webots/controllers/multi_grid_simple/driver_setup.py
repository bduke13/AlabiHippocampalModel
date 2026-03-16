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


def initialize_robot_pose(driver, *, randomize_start_loc: bool, start_loc) -> None:
    if randomize_start_loc:
        randomize_robot_pose(driver.robot, goal_location=driver.goal_location)
        return

    if start_loc is None:
        raise ValueError("start_loc must be provided when randomize_start_loc is False.")
    set_robot_pose(driver.robot, start_loc)


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
    driver.hmap_loc = np.zeros((driver.num_steps, 3))
    driver.hmap_pcn = torch.zeros(
        (driver.num_steps, driver.pcn.num_pc),
        device=driver.device,
        dtype=torch.float32,
    )
    driver.hmap_bvc = torch.zeros(
        (driver.num_steps, driver.pcn.bvc_layer.num_bvc),
        device=driver.device,
        dtype=torch.float32,
    )
    driver.hmap_hdn = torch.zeros(
        (driver.num_steps, driver.n_hd),
        device="cpu",
        dtype=torch.float32,
    )
    driver.directional_reward_estimates = torch.zeros(driver.n_hd, device=driver.device)


def initialize_head_direction_layer(driver) -> None:
    driver.head_direction_layer = HeadDirectionLayer(
        num_cells=driver.n_hd,
        device=torch.device("cpu"),
    )
