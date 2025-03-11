import os
import json
import csv
import shutil
from datetime import datetime

def generate_trial_folder(trial_base_dir: str, trial_name: str) -> str:
    """
    Generates a unique trial folder name in trial_base_dir.
    If trial_name is "none" or already exists, appends a counter.
    """
    os.makedirs(trial_base_dir, exist_ok=True)
    if trial_name.lower() == "none":
        base_name = "trial"
    else:
        base_name = trial_name
    trial_folder = os.path.join(trial_base_dir, base_name)
    counter = 1
    while os.path.exists(trial_folder):
        trial_folder = os.path.join(trial_base_dir, f"{base_name}_{counter}")
        counter += 1
    os.makedirs(trial_folder)
    return trial_folder

def save_trial_params(trial_folder: str, trial_params: dict):
    """Saves the trial parameters as a JSON file in the trial folder."""
    params_path = os.path.join(trial_folder, "trial_params.json")
    with open(params_path, "w") as fp:
        json.dump(trial_params, fp, indent=4)

def copy_pkl_data(pkl_base_dir: str, trial_folder: str):
    """
    Copies the pkl files (and their folder structure) from the standard save location into the trial folder.
    The pkl_base_dir is expected to have subfolders (e.g., hmaps, networks).
    In the trial folder, we recreate these subfolders.
    """
    for subfolder in ["hmaps", "networks"]:
        src_folder = os.path.join(pkl_base_dir, subfolder)
        if os.path.exists(src_folder):
            dst_folder = os.path.join(trial_folder, subfolder)
            shutil.copytree(src_folder, dst_folder)

def log_trial(trial_base_dir: str, trial_params: dict, trial_folder: str):
    """
    Appends a row to a master CSV file in trial_base_dir with summary info.
    """
    log_file = os.path.join(trial_base_dir, "trial_log.csv")
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as csvfile:
        fieldnames = list(trial_params.keys()) + ["trial_folder", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        trial_params["trial_folder"] = trial_folder
        trial_params["timestamp"] = datetime.now().isoformat()
        writer.writerow(trial_params)

def archive_trial_data(trial_params: dict, pkl_base_dir: str, controller_base_dir: str):
    """
    Archives the current trial data.
    - Creates a new trial folder under controller_base_dir.
    - Copies the pkl data from pkl_base_dir.
    - Saves a JSON file of the trial parameters.
    - Logs the trial in a master CSV.
    """
    trial_folder = generate_trial_folder(controller_base_dir, trial_params.get("trial_name", "trial"))
    save_trial_params(trial_folder, trial_params)
    copy_pkl_data(pkl_base_dir, trial_folder)
    log_trial(controller_base_dir, trial_params, trial_folder)
    print(f"Trial data archived to {trial_folder}")
