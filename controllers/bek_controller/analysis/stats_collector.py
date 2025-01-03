import json
import os
from datetime import datetime

class stats_collector:
    def __init__(self, output_dir):
        """
        Initializes the stats collection for a trial.

        Args:
            output_dir (str): Directory to save the JSON stats files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        self.stats = {
            "trial_id": None,
            "start_location": None,
            "goal_location": None,
            "total_distance_traveled": 0.0,
            "total_time_secs": 0.0,
            "success": False,
            "turn_count": 0,
            "collision_count": 0
        }

    def update_stat(self, key, value):
        """Update a specific stat."""
        if key in self.stats:
            self.stats[key] = value

    def save_stats(self, trial_id):
        """
        Save the trial stats to a JSON file.

        Args:
            trial_id (str): Unique identifier for the trial (e.g., 'trial_1').
        """
        filename = os.path.join(self.output_dir, f"{trial_id}.json")
        with open(filename, "w") as f:
            json.dump(self.stats, f, indent=4)
        print(f"Saved stats for {trial_id} to {filename}")