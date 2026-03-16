"""Non-runtime artifact and persistence helpers for Driver v2.

This module intentionally contains slow-path logic only:
- file I/O (save/clear)
- trial completion metadata
- goal-map export and stats finalization
"""

import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from core.robot.robot_mode import RobotMode

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_multi_goal_reward_maps(driver):
    """Create reward maps for each goal-scale combination."""
    multi_goal_dir = os.path.join(driver.network_dir, "multi_goal_rewards")
    os.makedirs(multi_goal_dir, exist_ok=True)

    print(f"[LEARN_LOCATIONS] Creating {len(driver.goals)} goals x {len(driver.scales)} scales reward maps")

    created_maps = 0
    for goal in driver.goals:
        for scale_idx, (pcn, rcn) in enumerate(zip(driver.pcns, driver.rcns)):
            pc_idx = driver.goal_place_cell_associations[goal["name"]][scale_idx]

            if pc_idx is None:
                print(f"[WARNING] No place cell associated with {goal['name']} for scale {scale_idx}")
                continue

            artificial_activations = torch.zeros_like(pcn.place_cell_activations)
            artificial_activations[pc_idx] = 1.0

            print(f"[LEARN_LOCATIONS] Creating reward map for {goal['name']} scale {scale_idx}: using PC {pc_idx}")

            goal_rcn = copy.deepcopy(rcn)
            goal_rcn.update_reward_cell_activations(artificial_activations, visit=True)

            if hasattr(goal_rcn, "replay_with_custom_activations"):
                goal_rcn.replay_with_custom_activations(pcn=pcn, custom_activations=artificial_activations)
            else:
                goal_rcn.replay(pcn=pcn)

            goal_rcn_path = os.path.join(multi_goal_dir, f"rcn_scale_{scale_idx}_goal_{goal['name']}.pkl")
            with open(goal_rcn_path, "wb") as f:
                pickle.dump(goal_rcn, f)

            created_maps += 1
            scale_name = driver.scales[scale_idx]["name"]
            print(f"[LEARN_LOCATIONS] Created: {scale_name}_goal_{goal['name']} (PC {pc_idx})")

    print(f"[LEARN_LOCATIONS] Successfully created {created_maps} reward maps")


def save_multi_goal_data(driver):
    """Save multi-goal association artifacts."""
    multi_goal_dir = os.path.join(driver.network_dir, "multi_goal_rewards")

    associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")
    association_data = {
        "goal_place_cell_associations": driver.goal_place_cell_associations,
        "goal_association_step": driver.goal_association_step,
        "goal_place_cell_activations": driver.goal_place_cell_activations,
        "goal_visit_counts": driver.goal_visit_counts,
        "goals": driver.goals,
        "scales": [{"scale_index": s["scale_index"], "name": s["name"]} for s in driver.scales],
        "total_steps": driver.step_count,
        "final_time": driver.getTime(),
    }
    with open(associations_path, "wb") as f:
        pickle.dump(association_data, f)

    print(f"[LEARN_LOCATIONS] Saved goal associations to {associations_path}")

    print("[LEARN_LOCATIONS] Final associations:")
    for goal_name, associations in driver.goal_place_cell_associations.items():
        goal_info = next(g for g in driver.goals if g["name"] == goal_name)
        visit_count = driver.goal_visit_counts[goal_name]
        print(f"  {goal_name} at {goal_info['location']}: {associations} (visits: {visit_count})")


def save_trial_completion_time(driver, trial_elapsed_time):
    """Save trial completion time to JSON for AUTO mode."""
    if driver.trial_base_dir is None:
        print("[WARNING] Cannot save trial completion time - not in AUTO mode")
        return

    completion_data = {
        "trial_number": driver.current_auto_trial,
        "world_name": driver.world_name,
        "completion_time_seconds": trial_elapsed_time,
        "coverage_percentage": driver.current_coverage_percentage * 100,
        "total_steps": driver.step_count,
        "trial_name": driver.auto_trial_name,
    }

    json_path = os.path.join(driver.trial_base_dir, "trial_completion_time.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(completion_data, f, indent=2)

    print(f"[AUTO_TRIAL] Saved trial completion time to {json_path}")
    print(f"[AUTO_TRIAL] Trial {driver.current_auto_trial}: {trial_elapsed_time:.1f}s, Coverage: {driver.current_coverage_percentage*100:.1f}%")


def handle_random_goal_exploitation(driver, goal):
    """Handle goal reached during EXPLOIT_LOCATIONS_RANDOM modes."""
    if driver.stats_collector:
        trial_time = driver.getTime() - driver.trial_start_time

        driver.stats_collector.update_stat("trial_id", driver.trial_id)
        driver.stats_collector.update_stat("start_location", driver.start_loc)
        driver.stats_collector.update_stat("goal_location", goal["location"])
        driver.stats_collector.update_stat("goal_name", goal["name"])
        driver.stats_collector.update_stat("total_distance_traveled", round(driver.total_distance_traveled, 2))
        driver.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
        driver.stats_collector.update_stat("goal_reached", True)

        if hasattr(driver, "optimal_path_distance") and driver.optimal_path_distance:
            path_ratio = driver.total_distance_traveled / driver.optimal_path_distance
            driver.stats_collector.update_stat("optimal_path_distance", round(driver.optimal_path_distance, 2))
            driver.stats_collector.update_stat("path_ratio", round(path_ratio, 2))
            print(
                f"[RANDOM_EXPLOIT] Goal '{goal['name']}' reached! "
                f"Distance: {driver.total_distance_traveled:.1f}m, "
                f"Time: {trial_time:.1f}s, "
                f"Ratio: {path_ratio:.2f}"
            )

        driver.stats_collector.update_stat("spawn_method", "random")
        driver.stats_collector.update_stat("path_failure_ratio", driver.path_failure_ratio)
        driver.stats_collector.update_stat("termination_reason", "goal_reached")
        driver.stats_collector.save_stats(driver.trial_id)

    driver.stop()
    driver.save(include_hmaps=True)
    driver.done = True


def handle_random_exploitation_timeout(driver):
    """Handle timeout during EXPLOIT_LOCATIONS_RANDOM modes."""
    if driver.stats_collector:
        trial_time = driver.getTime() - driver.trial_start_time

        driver.stats_collector.update_stat("trial_id", driver.trial_id)
        driver.stats_collector.update_stat("start_location", driver.start_loc)

        active_goal = next((g for g in driver.goals if g.get("active", False)), None)
        if active_goal:
            driver.stats_collector.update_stat("goal_location", active_goal["location"])
            driver.stats_collector.update_stat("goal_name", active_goal["name"])

        driver.stats_collector.update_stat("total_distance_traveled", round(driver.total_distance_traveled, 2))
        driver.stats_collector.update_stat("total_time_secs", round(trial_time, 2))
        driver.stats_collector.update_stat("goal_reached", False)

        if hasattr(driver, "optimal_path_distance") and driver.optimal_path_distance:
            path_ratio = driver.total_distance_traveled / driver.optimal_path_distance
            driver.stats_collector.update_stat("optimal_path_distance", round(driver.optimal_path_distance, 2))
            driver.stats_collector.update_stat("path_ratio", round(path_ratio, 2))
            termination_reason = "distance_limit" if path_ratio >= driver.path_failure_ratio else "time_limit"
        else:
            termination_reason = "time_limit"

        driver.stats_collector.update_stat("spawn_method", "random")
        driver.stats_collector.update_stat("path_failure_ratio", driver.path_failure_ratio)
        driver.stats_collector.update_stat("termination_reason", termination_reason)

        print(
            f"[RANDOM_EXPLOIT] Trial timeout - "
            f"Distance: {driver.total_distance_traveled:.1f}m, "
            f"Time: {trial_time:.1f}s, "
            f"Reason: {termination_reason}"
        )

        driver.stats_collector.save_stats(driver.trial_id)

    driver.stop()
    driver.save(include_hmaps=True)
    driver.done = True


def save_driver_outputs(
    driver,
    include_pcn: bool = False,
    include_rcn: bool = False,
    include_gcn: bool = False,
    include_hmaps: bool = False,
    save_trajectory: bool = False,
):
    """Persist driver outputs (networks, hmaps, trajectory)."""
    files_saved = []

    os.makedirs(driver.hmap_dir, exist_ok=True)
    os.makedirs(driver.network_dir, exist_ok=True)

    if include_pcn:
        for scale_def, pcn in zip(driver.scales, driver.pcns):
            scale_idx = scale_def["scale_index"]
            pcn_path = os.path.join(driver.network_dir, f"pcn_scale_{scale_idx}.pkl")
            with open(pcn_path, "wb") as f:
                pickle.dump(pcn, f)
            files_saved.append(pcn_path)

    if include_rcn:
        for scale_def, rcn in zip(driver.scales, driver.rcns):
            scale_idx = scale_def["scale_index"]
            rcn_path = os.path.join(driver.network_dir, f"rcn_scale_{scale_idx}.pkl")
            with open(rcn_path, "wb") as f:
                pickle.dump(rcn, f)
            files_saved.append(rcn_path)

    if include_gcn:
        for scale_def, gcn in zip(driver.scales, driver.gcns):
            if gcn is not None:
                scale_idx = scale_def["scale_index"]
                gcn_path = os.path.join(driver.network_dir, f"gcn_scale_{scale_idx}.pkl")
                with open(gcn_path, "wb") as f:
                    pickle.dump(gcn, f)
                files_saved.append(gcn_path)

    if include_hmaps:
        use_trial_prefix = driver.robot_mode == RobotMode.EXPLOIT_LOCATIONS_RANDOM
        prefix = f"{getattr(driver, 'trial_id', 'default')}_" if use_trial_prefix else ""

        hmap_loc_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_loc.pkl")
        with open(hmap_loc_path, "wb") as f:
            pickle.dump(driver.hmap_loc[: driver.step_count], f)
        files_saved.append(hmap_loc_path)

        hmap_hdn_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_hdn.pkl")
        with open(hmap_hdn_path, "wb") as f:
            pickle.dump(driver.hmap_hdn[: driver.step_count].cpu(), f)
        files_saved.append(hmap_hdn_path)

        for scale_def, pc_history in zip(driver.scales, driver.hmap_pcn_activities):
            scale_idx = scale_def["scale_index"]
            hmap_scale_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_pcn_scale_{scale_idx}.pkl")
            with open(hmap_scale_path, "wb") as f:
                pc_data = pc_history[: driver.step_count].cpu().numpy()
                pickle.dump(pc_data, f)
            files_saved.append(hmap_scale_path)

        for scale_def, gc_history in zip(driver.scales, driver.hmap_gcn_activities):
            if gc_history.numel() > 0:
                scale_idx = scale_def["scale_index"]
                hmap_scale_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_gcn_scale_{scale_idx}.pkl")
                with open(hmap_scale_path, "wb") as f:
                    gc_data = gc_history[: driver.step_count].cpu().numpy()
                    pickle.dump(gc_data, f)
                files_saved.append(hmap_scale_path)

        if hasattr(driver, "hmap_prox"):
            hmap_prox_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_prox.pkl")
            with open(hmap_prox_path, "wb") as f:
                prox_data = driver.hmap_prox[: driver.step_count].cpu().numpy()
                pickle.dump(prox_data, f)
            files_saved.append(hmap_prox_path)

        if hasattr(driver, "hmap_scale_priority") and use_trial_prefix:
            hmap_scale_priority_path = os.path.join(driver.hmap_dir, f"{prefix}hmap_scale_priority.pkl")
            with open(hmap_scale_priority_path, "wb") as f:
                scale_priority_data = driver.hmap_scale_priority[: driver.step_count].cpu().numpy()
                pickle.dump(scale_priority_data, f)
            files_saved.append(hmap_scale_priority_path)

    if save_trajectory:
        scale_name_list = [scale["name"] for scale in driver.scales]
        scale_order = ["small", "medium", "large", "xlarge"]
        scale_name_list = sorted(scale_name_list, key=lambda x: scale_order.index(x))
        scale_combination = "_".join(scale_name_list)

        base_stats_dir = os.path.join(PROJECT_ROOT, "analysis", "stats", driver.world_name, scale_combination)
        hmaps_path_dir = os.path.join(base_stats_dir, "hmaps")
        os.makedirs(hmaps_path_dir, exist_ok=True)

        trial_id = getattr(driver, "trial_id", "default")
        hmap_loc_file = os.path.join(hmaps_path_dir, f"{trial_id}_hmap_loc.pkl")
        hmap_scale_priority_file = os.path.join(hmaps_path_dir, f"{trial_id}_hmap_scale_priority.pkl")

        with open(hmap_loc_file, "wb") as f:
            pickle.dump(driver.hmap_loc[: driver.step_count], f)
            files_saved.append(hmap_loc_file)

        with open(hmap_scale_priority_file, "wb") as f:
            pickle.dump(driver.hmap_scale_priority[: driver.step_count].cpu().numpy(), f)
            files_saved.append(hmap_scale_priority_file)

        print(f"Saved path data for trial {trial_id} in {scale_combination}.")

    print(f"Files Saved: {files_saved}")
    print("Saving Done!")


def clear_driver_outputs(driver):
    """Remove network and hmap artifacts for a clean start."""
    if os.path.exists(driver.network_dir):
        for fname in os.listdir(driver.network_dir):
            if fname.startswith("pcn_scale_") or fname.startswith("rcn_scale_") or fname.startswith("gcn_scale_"):
                full_path = os.path.join(driver.network_dir, fname)
                try:
                    os.remove(full_path)
                    print(f"Removed: {full_path}")
                except FileNotFoundError:
                    pass

        for legacy_file in ["pcn.pkl", "rcn.pkl", "gcn.pkl"]:
            path = os.path.join(driver.network_dir, legacy_file)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed: {path}")
                except FileNotFoundError:
                    pass

    if os.path.exists(driver.hmap_dir):
        for fname in os.listdir(driver.hmap_dir):
            if "hmap_" in fname:
                full_path = os.path.join(driver.hmap_dir, fname)
                try:
                    os.remove(full_path)
                    print(f"Removed: {full_path}")
                except FileNotFoundError:
                    pass

    print("[DRIVER] Finished clearing old scale PCNs, RCNs, GCNs, and hmap files.")
