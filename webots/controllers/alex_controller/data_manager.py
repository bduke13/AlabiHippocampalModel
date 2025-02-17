"""
data_manager.py

Saves pcn.pkl, rcn.pkl, hmap_*.pkl, etc. to webots/data/current_run
"""

import os
import pickle
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRENT_RUN_DIR = PROJECT_ROOT / "webots" / "data" / "current_run"

def save_run_data(
    driver,
    folder: Path = DEFAULT_CURRENT_RUN_DIR,
    include_pcn=True,
    include_rcn=True,
    include_hmaps=True,
    save_params=True,
    extra_info=None
):
    folder.mkdir(parents=True, exist_ok=True)
    files_saved = []

    # pcn.pkl
    if include_pcn and driver.pcn:
        pcn_path = folder / "pcn.pkl"
        with open(pcn_path, "wb") as f:
            pickle.dump(driver.pcn, f)
        files_saved.append(str(pcn_path))

    # rcn.pkl
    if include_rcn and driver.rcn:
        rcn_path = folder / "rcn.pkl"
        with open(rcn_path, "wb") as f:
            pickle.dump(driver.rcn, f)
        files_saved.append(str(rcn_path))

    if include_hmaps:
        # hmap_loc.pkl
        if hasattr(driver, "hmap_loc") and driver.hmap_loc is not None:
            loc_path = folder / "hmap_loc.pkl"
            with open(loc_path, "wb") as f:
                pickle.dump(driver.hmap_loc[: driver.step_count], f)
            files_saved.append(str(loc_path))

        # hmap_pcn.pkl
        if hasattr(driver, "hmap_pcn") and driver.hmap_pcn is not None:
            pcn_path_2 = folder / "hmap_pcn.pkl"
            with open(pcn_path_2, "wb") as f:
                pickle.dump(driver.hmap_pcn[: driver.step_count].cpu().numpy(), f)
            files_saved.append(str(pcn_path_2))

        # hmap_bvc.pkl
        if hasattr(driver, "hmap_bvc") and driver.hmap_bvc is not None:
            bvc_path = folder / "hmap_bvc.pkl"
            with open(bvc_path, "wb") as f:
                pickle.dump(driver.hmap_bvc[: driver.step_count].cpu().numpy(), f)
            files_saved.append(str(bvc_path))

        # hmap_hdn.pkl
        if hasattr(driver, "hmap_hdn") and driver.hmap_hdn is not None:
            hdn_path = folder / "hmap_hdn.pkl"
            with open(hdn_path, "wb") as f:
                pickle.dump(driver.hmap_hdn[: driver.step_count].cpu().numpy(), f)
            files_saved.append(str(hdn_path))

        # hmap_g.pkl
        if hasattr(driver, "hmap_g") and driver.hmap_g is not None:
            g_path = folder / "hmap_g.pkl"
            with open(g_path, "wb") as f:
                pickle.dump(driver.hmap_g[: driver.step_count], f)
            files_saved.append(str(g_path))

    # optional params
    if save_params:
        param_path = folder / "params.json"
        param_data = {
            "mode": driver.robot_mode.name if driver.robot_mode else "UNKNOWN",
            "movement_method": getattr(driver, "movement_method", "N/A"),
            "bvc_params": getattr(driver, "bvc_params", {}),
            "pc_params": getattr(driver, "pc_params", {}),
            "extra_info": extra_info if extra_info else {}
        }
        with open(param_path, "w") as f:
            json.dump(param_data, f, indent=2)
        files_saved.append(str(param_path))

    print(f"[DataManager] Saved files to {folder}:")
    for item in files_saved:
        print("   ", item)
    return files_saved

def clear_current_run(folder: Path = DEFAULT_CURRENT_RUN_DIR):
    if folder.is_dir():
        shutil.rmtree(folder)
        print(f"[DataManager] Cleared folder: {folder}")
    else:
        print(f"[DataManager] Nothing to clear; folder does not exist: {folder}")
