import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional


def initialize_run_artifacts(
    driver,
    *,
    controller_dir: Path,
    run_id: Optional[str],
    mode,
    run_time_hours: int,
    randomize_start_loc: bool,
    start_loc,
    start_rotation,
    load_networks_from_run_id,
    load_hmaps_from_run_id,
    runtime_profile_name,
    runtime_profile_overrides,
    runtime_parameters,
    enable_ojas,
    enable_stdp,
    goal_location,
    max_dist: float,
    show_bvc_activation: bool,
) -> None:
    runs_root = controller_dir / "runs"
    visualizations_root = controller_dir / "visualizations"

    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{driver.world_name}_{mode.name.lower()}"

    driver.controller_dir = controller_dir
    driver.runs_root = runs_root
    driver.visualizations_root = visualizations_root
    driver.run_id = run_id
    driver.run_dir = runs_root / run_id
    driver.hmap_dir = driver.run_dir / "hmaps"
    driver.network_dir = driver.run_dir / "networks"
    driver.network_load_dir = (
        runs_root / load_networks_from_run_id / "networks"
        if load_networks_from_run_id
        else driver.network_dir
    )
    driver.hmap_load_dir = (
        runs_root / load_hmaps_from_run_id / "hmaps"
        if load_hmaps_from_run_id
        else driver.hmap_dir
    )
    driver.visualization_dir = visualizations_root / run_id
    driver.config_path = driver.run_dir / "config.json"
    driver.metrics_path = driver.run_dir / "metrics.json"

    driver.hmap_dir.mkdir(parents=True, exist_ok=True)
    driver.network_dir.mkdir(parents=True, exist_ok=True)
    driver.visualization_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_id": driver.run_id,
        "world_name": driver.world_name,
        "mode": mode.name,
        "run_time_hours": run_time_hours,
        "randomize_start_loc": randomize_start_loc,
        "start_loc": start_loc,
        "start_rotation": start_rotation,
        "load_networks_from_run_id": load_networks_from_run_id,
        "load_hmaps_from_run_id": load_hmaps_from_run_id,
        "runtime_profile": runtime_profile_name,
        "runtime_profile_overrides": runtime_profile_overrides,
        "runtime_parameters": runtime_parameters,
        "enable_ojas": enable_ojas,
        "enable_stdp": enable_stdp,
        "goal_location": goal_location,
        "max_dist": max_dist,
        "show_bvc_activation": show_bvc_activation,
        "created_at": datetime.now().isoformat(),
    }
    with open(driver.config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, indent=2)


def write_metrics(driver, *, status: str, files_saved=None, extra=None) -> None:
    metrics = {
        "run_id": getattr(driver, "run_id", None),
        "world_name": getattr(driver, "world_name", None),
        "mode": getattr(driver, "robot_mode", None).name
        if getattr(driver, "robot_mode", None) is not None
        else None,
        "status": status,
        "completion_reason": getattr(driver, "trial_completion_reason", None),
        "files_saved": files_saved or [],
        "step_count": getattr(driver, "step_count", 0),
        "simulation_time_seconds": float(driver.getTime())
        if hasattr(driver, "getTime")
        else 0.0,
        "trial_elapsed_seconds": float(driver.elapsed_trial_time_seconds())
        if hasattr(driver, "elapsed_trial_time_seconds")
        else (
            float(driver.getTime())
            if hasattr(driver, "getTime")
            else 0.0
        ),
        "path_length": float(driver.compute_path_length())
        if hasattr(driver, "hmap_loc")
        else 0.0,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        metrics.update(extra)

    with open(driver.metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    try:
        from run_summary import update_run_summary
    except ImportError:
        from .run_summary import update_run_summary

    update_run_summary(driver.run_dir, visualization_dir=driver.visualization_dir)


def save_driver_state(
    driver,
    *,
    include_pcn: bool = False,
    include_rcn: bool = False,
    include_hmaps: bool = False,
) -> list[str]:
    files_saved: list[str] = []

    driver.hmap_dir.mkdir(parents=True, exist_ok=True)
    driver.network_dir.mkdir(parents=True, exist_ok=True)

    if include_pcn:
        pcn_path = driver.network_dir / "pcn.pkl"
        with open(pcn_path, "wb") as output:
            pickle.dump(driver.pcn, output)
        files_saved.append(str(pcn_path))

    if include_rcn:
        rcn_path = driver.network_dir / "rcn.pkl"
        with open(rcn_path, "wb") as output:
            pickle.dump(driver.rcn, output)
        files_saved.append(str(rcn_path))

    if include_hmaps:
        hmap_loc_path = driver.hmap_dir / "hmap_loc.pkl"
        with open(hmap_loc_path, "wb") as output:
            pickle.dump(driver.hmap_loc[: driver.step_count], output)
        files_saved.append(str(hmap_loc_path))

        hmap_pcn_path = driver.hmap_dir / "hmap_pcn.pkl"
        with open(hmap_pcn_path, "wb") as output:
            pcn_cpu = driver.hmap_pcn[: driver.step_count].cpu().numpy()
            pickle.dump(pcn_cpu, output)
        files_saved.append(str(hmap_pcn_path))

        hmap_hdn_path = driver.hmap_dir / "hmap_hdn.pkl"
        with open(hmap_hdn_path, "wb") as output:
            pickle.dump(driver.hmap_hdn[: driver.step_count], output)
        files_saved.append(str(hmap_hdn_path))

        hmap_bvc_path = driver.hmap_dir / "hmap_bvc.pkl"
        with open(hmap_bvc_path, "wb") as output:
            bvc_cpu = driver.hmap_bvc[: driver.step_count].cpu().numpy()
            pickle.dump(bvc_cpu, output)
        files_saved.append(str(hmap_bvc_path))

        hmap_gcn_path = driver.hmap_dir / "hmap_gcn.pkl"
        with open(hmap_gcn_path, "wb") as output:
            gcn_cpu = driver.hmap_gcn[: driver.step_count].cpu().numpy()
            pickle.dump(gcn_cpu, output)
        files_saved.append(str(hmap_gcn_path))

    return files_saved


def clear_driver_outputs(driver) -> None:
    files_to_remove = [
        driver.network_dir / "pcn.pkl",
        driver.network_dir / "rcn.pkl",
        driver.hmap_dir / "hmap_loc.pkl",
        driver.hmap_dir / "hmap_pcn.pkl",
        driver.hmap_dir / "hmap_bvc.pkl",
        driver.hmap_dir / "hmap_hdn.pkl",
        driver.hmap_dir / "hmap_gcn.pkl",
    ]

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found")
        except Exception as exc:
            print(f"Error removing {file_path}: {exc}")

    for directory in (driver.network_dir, driver.hmap_dir, driver.visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)
