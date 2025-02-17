#!/usr/bin/env python
"""
File: webots/controllers/alex_controller/trial_manager.py

TrialManager organizes trial data by creating unique folders and logging a master summary.
Data are saved in two locations:
  - The global "current_run" folder (always updated with the latest run).
  - A trial-specific folder under "webots/data/trial_data" if trial saving is enabled.
The configuration is saved as <trial_name>_config.json in the trial folder.
"""

import os
import pickle
import json
import datetime
from typing import Optional
from pathlib import Path

# Compute project paths (this file is assumed to be in webots/controllers/alex_controller/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEBOTS_DIR = os.path.join(PROJECT_ROOT, "webots")  # lowercase as required
DATA_DIR = os.path.join(WEBOTS_DIR, "data")
TRIAL_DATA_DIR = os.path.join(DATA_DIR, "trial_data")
CURRENT_RUN_DIR = os.path.join(DATA_DIR, "current_run")

# Ensure base folders exist.
os.makedirs(TRIAL_DATA_DIR, exist_ok=True)
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)

class TrialManager:
    def __init__(self, save_trials: bool = True):
        self.save_trials = save_trials
        # If saving individual trials, use the trial_data directory; otherwise, use current_run.
        self.trial_base_dir = TRIAL_DATA_DIR if self.save_trials else CURRENT_RUN_DIR
        os.makedirs(self.trial_base_dir, exist_ok=True)
        self.master_log_path = os.path.join(TRIAL_DATA_DIR, "trials_summary.json")
        self.trials_summary = []
    
    def _get_next_trial_name(self):
        """Auto-generates a trial name (trial_1, trial_2, etc.) based on existing folders."""
        existing = [d for d in os.listdir(self.trial_base_dir) if d.startswith("trial_")]
        numbers = []
        for name in existing:
            try:
                num = int(name.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        next_num = max(numbers) + 1 if numbers else 1
        return f"trial_{next_num}"
    
    def start_trial(self, trial_config: dict, trial_name: Optional[str] = None):
        """
        Creates a folder for the trial and saves the configuration.
        If save_trials is True, the folder is created under webots/data/trial_data;
        otherwise, it uses the current_run folder.
        The configuration is saved as <trial_name>_config.json.
        """
        if trial_name is None:
            trial_name = self._get_next_trial_name() if self.save_trials else "current_run"
        self.current_trial_id = trial_name
        base_dir = self.trial_base_dir
        self.current_trial_dir = os.path.join(base_dir, trial_name)
        os.makedirs(self.current_trial_dir, exist_ok=True)
        config_filename = f"{trial_name}_config.json"
        self.save_metadata(config_filename, trial_config)
        print(f"[TrialManager] Created trial folder: {self.current_trial_dir}")
        return self.current_trial_dir
    
    def save_data(self, data, filename: str):
        """
        Saves a data object as a pickle file in the current trial folder.
        """
        file_path = os.path.join(self.current_trial_dir, filename)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[TrialManager] Saved {filename} in trial folder: {self.current_trial_dir}")
        return file_path

    def save_data_global(self, data, filename: str):
        """
        Saves a data object as a pickle file in the global current_run folder.
        """
        file_path = os.path.join(CURRENT_RUN_DIR, filename)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[TrialManager] Saved {filename} in global current_run folder: {CURRENT_RUN_DIR}")
        return file_path

    def save_metadata(self, filename: str, metadata: dict):
        """
        Saves metadata as a JSON file in the current trial folder.
        Uses default=str to convert non-serializable objects to strings.
        """
        file_path = os.path.join(self.current_trial_dir, filename)
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        print(f"[TrialManager] Saved metadata {filename} in trial folder: {self.current_trial_dir}")
        return file_path
    
    def log_trial_summary(self, summary: dict):
        """Appends a summary entry for this trial and updates the master log."""
        summary["trial_id"] = self.current_trial_id
        self.trials_summary.append(summary)
        with open(self.master_log_path, "w") as f:
            json.dump(self.trials_summary, f, indent=4, default=str)
        print(f"[TrialManager] Logged trial summary to {self.master_log_path}")
    
    def copy_global_to_trial(self, filename: str):
        """
        Copies a file from the global current_run folder into the current trial folder.
        """
        from shutil import copyfile
        src = os.path.join(CURRENT_RUN_DIR, filename)
        dst = os.path.join(self.current_trial_dir, filename)
        try:
            copyfile(src, dst)
            print(f"[TrialManager] Copied {filename} from current_run to trial folder.")
        except Exception as e:
            print(f"[TrialManager] Error copying {filename}: {e}")
