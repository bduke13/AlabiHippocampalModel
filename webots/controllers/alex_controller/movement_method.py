"""
movement_method.py

Houses the movement logic: random_walk, exploit_reward_map, manual_control, etc.
"""

import numpy as np
import math

def random_walk(driver):
    collided = driver.check_collision()
    if collided:
        angle = np.random.uniform(-math.pi, math.pi)
        driver.turn(angle)
    else:
        driver.forward()

def exploit_reward_map(driver):
    if not driver.rcn or not driver.pcn:
        random_walk(driver)
        return

    n_hd = driver.pc_params.get("n_hd", 8)
    angles = np.linspace(0, 2 * math.pi, n_hd, endpoint=False)
    pot_rew = np.zeros(n_hd)

    for d in range(n_hd):
        future_act = driver.pcn.preplay(d, num_steps=1)
        driver.rcn.update_reward_cell_activations(future_act)
        pot_rew[d] = driver.rcn.reward_cell_activations.max().item()

    best_idx = np.argmax(pot_rew)
    best_angle = angles[best_idx]
    diff_angle = best_angle - math.radians(driver.current_heading_deg)
    driver.turn(diff_angle)
    driver.forward()

def manual_control(driver):
    """
    No keyboard in Supervisor, so let's fallback to random_walk or do nothing.
    We'll do random_walk for demonstration.
    """
    random_walk(driver)

MOVEMENT_METHODS = {
    "RANDOM": random_walk,
    "EXPLOIT": exploit_reward_map,
    "MANUAL": manual_control
}

def get_movement_function(method_name: str):
    return MOVEMENT_METHODS.get(method_name.upper(), random_walk)
