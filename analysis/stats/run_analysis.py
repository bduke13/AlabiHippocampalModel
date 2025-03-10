import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Patch  # For custom legend patches

# Use Times New Roman font for the plots
rcParams['font.family'] = 'Times New Roman'

###############################################################################
# 1) Set the root directory and define your worlds
###############################################################################

# Append 'analysis/stats' to that directory
STATS_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Change to that directory
os.chdir(STATS_ROOT_DIR)

# Define worlds
worlds = ["20x20_1obstacle"]  # Add more worlds if needed

# The corner we care about
corner_to_analyze = (8, -8)

# Metrics to compare
metrics = ['total_distance_traveled', 'total_time_secs', 'turn_count']

###############################################################################
# 2) Utility: Load JSON data for a specific corner
###############################################################################
def load_json_for_corner(json_dir, corner):
    """Loads all JSON files from `json_dir` for which the trial corner == `corner`
    and the JSON data has success==True. Returns a DataFrame."""
    data = []
    if not os.path.exists(json_dir):
        return pd.DataFrame()

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            # Expect something like "trial_1_corner_8_-8.json"
            try:
                parts = filename.split("_")
                x_coord = int(parts[-2])
                y_coord = int(parts[-1].split(".")[0])
                if (x_coord, y_coord) != corner:
                    continue
            except (IndexError, ValueError):
                print(f"Error parsing corner from filename: {filename}")
                continue

            full_path = os.path.join(json_dir, filename)
            with open(full_path, 'r') as f:
                file_data = json.load(f)
                if file_data.get("success") is True:
                    data.append(file_data)

    return pd.DataFrame(data)

###############################################################################
# 3) Gather stats from subfolders (strategies) in a given world
###############################################################################
def gather_stats_for_world(world_dir, corner):
    """Gathers stats from all subfolders (strategies) that contain a JSON folder."""
    subfolders = []
    world_dir = os.path.abspath(world_dir)  # Ensure absolute path

    if not os.path.exists(world_dir):
        print(f"Warning: World directory '{world_dir}' does not exist.")
        return []

    print(f"Checking world directory: {world_dir}")  # Debugging

    for item in os.listdir(world_dir):
        subfolder_path = os.path.join(world_dir, item)
        json_dir = os.path.join(subfolder_path, "JSON")

        if os.path.isdir(subfolder_path):
            print(f"Found strategy folder: {item}")  # Debugging

        if os.path.isdir(subfolder_path) and os.path.exists(json_dir):
            df = load_json_for_corner(json_dir, corner)
            if not df.empty:
                df['strategy'] = item
                print(f"Loaded data for strategy: '{item}'")  # Debugging
                subfolders.append(df)

    return subfolders

###############################################################################
# 4) Build a master DataFrame across all worlds
###############################################################################
all_data_records = []
for world_name in worlds:
    world_dir = os.path.join(STATS_ROOT_DIR, world_name)
    subfolder_dfs = gather_stats_for_world(world_dir, corner_to_analyze)

    if not subfolder_dfs:
        print(f"No data found for world: {world_name}")
        continue

    # Combine them
    combined_df = pd.concat(subfolder_dfs, ignore_index=True)
    combined_df['world'] = world_name
    all_data_records.append(combined_df)

if not all_data_records:
    print("No data found across any worlds.")
    raise SystemExit

full_df = pd.concat(all_data_records, ignore_index=True)
print("\nFull DataFrame loaded:")
print(full_df.head())

###############################################################################
# 5) Build Comparison DataFrame => world | strategy | metric_mean | metric_sem
###############################################################################
comparison_data = {"world": [], "strategy": []}
for metric in metrics:
    comparison_data[f"{metric}_mean"] = []
    comparison_data[f"{metric}_sem"] = []

grouped = full_df.groupby(["world", "strategy"], as_index=False)
for (w, strategy), group_df in grouped:
    comparison_data["world"].append(w)
    comparison_data["strategy"].append(strategy)
    for metric in metrics:
        comparison_data[f"{metric}_mean"].append(group_df[metric].mean())
        comparison_data[f"{metric}_sem"].append(group_df[metric].sem())

comparison_df = pd.DataFrame(comparison_data)
print("\nComparison DataFrame:\n", comparison_df)

###############################################################################
# 6) Plotting: Grouped bar charts => x-axis = each world, bars = strategies
###############################################################################
strategies = sorted(comparison_df["strategy"].unique().tolist())

COLOR_MAP = {
    "small": "#4a4a4a",
    "medium": "#d1d1d1",
    "large": "#87CEEB",
    "small_medium": "#f4a261",
    "small_medium_large": "#2a9d8f",
}
DEFAULT_COLOR = "lightgray"

ordered_worlds = list(worlds)  # So the order is consistent

for idx, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))

    # We'll place each world on the x-axis.
    x = np.arange(len(ordered_worlds))  # [0, 1, 2, ...] for each world
    # The bar width must be small enough to fit all strategies side by side
    width = 0.75 / len(strategies)
    offset_center = -0.5 * (len(strategies) - 1)

    max_val = comparison_df[f"{metric}_mean"].max() if not comparison_df.empty else 1
    placeholder_height = 0.05 * max_val
    used_labels = set()

    for i, strategy_name in enumerate(strategies):
        # Shift bars horizontally to group them in the same x-axis spot
        x_positions = x + (i + offset_center) * width

        # For each world, find the row with (world, strategy_name)
        bar_heights, bar_errors = [], []
        existing_rows = comparison_df[comparison_df["strategy"] == strategy_name]

        for w in ordered_worlds:
            row = existing_rows[existing_rows["world"] == w]
            if not row.empty:
                mean_val = float(row[f"{metric}_mean"].values[0])
                sem_val = float(row[f"{metric}_sem"].values[0])
                bar_heights.append(mean_val)
                bar_errors.append(sem_val)
            else:
                # No data => placeholder
                bar_heights.append(None)
                bar_errors.append(None)

        color = COLOR_MAP.get(strategy_name, DEFAULT_COLOR)
        label_str = strategy_name if strategy_name not in used_labels else None
        if label_str:
            used_labels.add(strategy_name)

        # Plot each bar
        for xi, height, err in zip(x_positions, bar_heights, bar_errors):
            if height is None:
                ax.bar(xi, placeholder_height, width=width, color=color, alpha=0.8,
                       hatch='////', edgecolor="black", linewidth=1)
            else:
                ax.bar(xi, height, width=width, yerr=err, color=color, alpha=0.8,
                       capsize=3, edgecolor="gray", linewidth=0.8, label=label_str)
            label_str = None  # Only label once

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_worlds, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    y_labels = {
        "total_distance_traveled": "Total Distance (m)",
        "total_time_secs":         "Time to Goal (s)",
        "turn_count":              "Turn Count"
    }
    ax.set_ylabel(y_labels.get(metric, metric), fontsize=14)
    ax.set_xlabel("World", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best", frameon=False, fontsize=10, ncol=1)

    plt.tight_layout()
    plt.show()
