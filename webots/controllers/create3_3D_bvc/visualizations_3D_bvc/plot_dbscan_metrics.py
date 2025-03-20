# %%
from visualizations.vis_utils import *
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import (
    get_model_hexbin_metrics,
    analyze_cosine_similarity_torch,
    plot_similarity_sums,
)
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Set pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONTROLLER_NAME = "create3_base"

# Parameters
root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")
trials = os.listdir(root_path)

experiments_base = [
    ["10x10_open", "Open", "2D - 1 Layer"],
    ["10x10_cross", "Cross", "2D - 1 Layer"],
]

experiments_obstacles = [
    ["3D_bvc_two_shapes_control", "Control", "3D_bvc"],
    ["3D_bvc_two_shapes_test", "Test", "3D_bvc"],
]

experiments_ceilings = [
    ["3D_bvc_ceilings_control", "Control", "3D_bvc"],
    ["3D_bvc_ceilings_test", "Test", "3D_bvc"],
]

experiments_scaling = [
    ["3D_bvc_cross_rotated_1_normal", "env1", "normal"],
    ["3D_bvc_cross_rotated_2_no_vert_scaling", "env2", "no_vert"],
    ["3D_bvc_cross_rotated_3_no_dist_scaling", "env3", "no_dist"],
    ["3D_bvc_cross_rotated_4_no_vert_or_dist_scaling", "env4", "no_vert_dist"],
]

experiments_rotated = [
    ["3D_bvc_cross_rotated_1_model0", "Upright", "2D - 1 Layer"],
    ["3D_bvc_cross_rotated_1_model1", "Upright", "3D - 1 Layer"],
    ["3D_bvc_cross_rotated_1_model2", "Upright", "3D - 2 Layer"],
    ["3D_bvc_cross_rotated_1_model3", "Upright", "3D - 3 Layer"],
    ["3D_bvc_cross_rotated_1_model4", "Upright", "3D - 4 Layer"],
    ["3D_bvc_cross_rotated_2_model0", "30° Tilt", "2D - 1 Layer"],
    ["3D_bvc_cross_rotated_2_model1", "30° Tilt", "3D - 1 Layer"],
    ["3D_bvc_cross_rotated_2_model2", "30° Tilt", "3D - 2 Layer"],
    ["3D_bvc_cross_rotated_2_model3", "30° Tilt", "3D - 3 Layer"],
    ["3D_bvc_cross_rotated_2_model4", "30° Tilt", "3D - 4 Layer"],
    ["3D_bvc_cross_rotated_3_model0", "45° Tilt", "2D - 1 Layer"],
    ["3D_bvc_cross_rotated_3_model1", "45° Tilt", "3D - 1 Layer"],
    ["3D_bvc_cross_rotated_3_model2", "45° Tilt", "3D - 2 Layer"],
    ["3D_bvc_cross_rotated_3_model3", "45° Tilt", "3D - 3 Layer"],
    ["3D_bvc_cross_rotated_3_model4", "45° Tilt", "3D - 4 Layer"],
    ["3D_bvc_cross_rotated_4_model0", "60° Tilt", "2D - 1 Layer"],
    ["3D_bvc_cross_rotated_4_model1", "60° Tilt", "3D - 1 Layer"],
    ["3D_bvc_cross_rotated_4_model2", "60° Tilt", "3D - 2 Layer"],
    ["3D_bvc_cross_rotated_4_model3", "60° Tilt", "3D - 3 Layer"],
    ["3D_bvc_cross_rotated_4_model4", "60° Tilt", "3D - 4 Layer"],
]

experiments = experiments_base
experiment_plot_name = "Baseline Experiment"

# Extract just the world names for filtering
worlds = [world for world, _, _ in experiments]

# Print experiment configuration
print("Experiment Configuration:")
for world, env_name, model_name in experiments:
    print(f"  {world} -> {env_name} ({model_name})")

world_trial_paths = {}
for world in worlds:
    world_trials = [os.path.join(root_path, x) for x in trials if world in x]
    world_trial_paths[world] = world_trials
    print(f"found {len(world_trials)} trials for {world}")
    print(world_trials)

# Flatten the list of trial paths
filtered_dirs = []
for world_trials in world_trial_paths.values():
    filtered_dirs.extend(world_trials)

print(f"Total trials to process: {len(filtered_dirs)}")

# %%
# Collect DBSCAN-based cluster metrics for each path
metrics_list = []
world_env_model_list = []

for path in filtered_dirs:
    try:
        print("Processing:", path)
        # Extract world, env, and model from path
        path_parts = path.split(os.sep)
        trial_name = path_parts[-1]

        # Find matching experiment
        matching_exp = None
        for world, env_name, model_name in experiments:
            if world in trial_name:
                matching_exp = (world, env_name, model_name)
                break

        if matching_exp:
            world, env_name, model_name = matching_exp
            world_env_model_list.append(
                {"world": world, "env": env_name, "model": model_name}
            )
        else:
            print(f"Warning: No matching experiment found for {trial_name}")
            world_env_model_list.append(
                {"world": "unknown", "env": "unknown", "model": "unknown"}
            )

        # Load the data
        hmap_loc, hmap_pcn = load_hmaps_from_dir(
            hmap_names=["hmap_loc", "hmap_pcn"], base_dir=path
        )
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

        # Get the metrics
        metrics = get_model_hexbin_metrics(hmap_x, hmap_y, hmap_pcn, verbose=False)
        metrics_list.append(metrics)
        print(metrics)
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        # Append placeholder to maintain alignment
        metrics_list.append(
            {
                "total_clusters": None,
                "non_zero_cells": None,
                "cells_with_multiple_clusters": None,
                "avg_clusters_per_non_zero_cell": None,
            }
        )
        world_env_model_list.append(
            {"world": "error", "env": "error", "model": "error"}
        )

# %%
# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics_list)
world_env_model_df = pd.DataFrame(world_env_model_list)

# Combine the dataframes
df = pd.concat([world_env_model_df, metrics_df], axis=1)

# Display DBSCAN metrics
print("\nModel Analysis Results (DBSCAN):")
print(df)

# Plot the metrics
bar_width = 0.4

# Set font family and sizes globally
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "legend.title_fontsize": 16,
    }
)

# Calculate normalized metrics
df["norm_non_zero_cells"] = df["non_zero_cells"] / 500  # Normalize by total cells
df["norm_cells_multiple_clusters"] = df["cells_with_multiple_clusters"] / 500

# 1. Plot proportion of non-zero cells
plt.figure(figsize=(10, 5))
ax1 = sns.barplot(
    data=df,
    x="env",
    y="norm_non_zero_cells",
    hue="model",
    width=bar_width,  # Set the width of the bars
)
ax1.set_title(f"Proportion of Cells where MI > 0: {experiment_plot_name}", pad=20)
ax1.set_xlabel("Environment")
ax1.set_ylabel("Proportion")
ax1.legend(title="Model", loc="lower left")
ax1.grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)
plt.tight_layout(pad=2.0)
plt.show()

# 2. Plot proportion of cells with multiple clusters
plt.figure(figsize=(10, 5))
ax2 = sns.barplot(
    data=df,
    x="env",
    y="norm_cells_multiple_clusters",
    hue="model",
    width=bar_width,  # Set the width of the bars
)
ax2.set_title(f"Proportion of Cells where MI > 1: {experiment_plot_name}", pad=20)
ax2.set_xlabel("Environment")
ax2.set_ylabel("Proportion")
ax2.legend(title="Model", loc="lower left")
ax2.grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)
plt.tight_layout(pad=2.0)
plt.show()

# 3. Plot average clusters per non-zero cell
plt.figure(figsize=(10, 5))
ax3 = sns.barplot(
    data=df,
    x="env",
    y="avg_clusters_per_non_zero_cell",
    hue="model",
    width=bar_width,  # Set the width of the bars
)
ax3.set_title(f"Average MI of Cells when MI > 0: {experiment_plot_name}", pad=20)
ax3.set_xlabel("Environment")
ax3.set_ylabel("Average Clusters")
ax3.legend(title="Model", loc="lower left")
ax3.grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)
plt.tight_layout(pad=2.0)
plt.show()

# Print the dataframe to the terminal
print("\nDataFrame used for plotting:")
print(
    df[
        [
            "world",
            "env",
            "model",
            "non_zero_cells",
            "norm_non_zero_cells",
            "cells_with_multiple_clusters",
            "norm_cells_multiple_clusters",
            "avg_clusters_per_non_zero_cell",
            "total_clusters",
        ]
    ].to_string()
)

# %%
# Optional: Save the metrics to a CSV file
# df.to_csv("dbscan_metrics_results.csv", index=False)

# %%
# Optional: Create a heatmap of the metrics by environment and model
# pivot_df = df.pivot_table(
#    index="model",
#    columns="env",
#    values=[
#        "norm_non_zero_cells",
#        "norm_cells_multiple_clusters",
#        "avg_clusters_per_non_zero_cell",
#    ],
# )

# plt.figure(figsize=(15, 10))
# sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".2f")
# plt.title("DBSCAN Metrics by Model and Environment")
# plt.tight_layout()
# plt.show()
