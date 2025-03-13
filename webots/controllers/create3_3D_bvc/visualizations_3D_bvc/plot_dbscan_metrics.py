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
    ["3D_bvc_ceilings_control", "env1", "3D_bvc"],
    ["3D_bvc_ceilings_test", "env2", "3D_bvc"],
]

experiments_scaling = [
    ["3D_bvc_cross_rotated_1_normal", "env1", "normal"],
    ["3D_bvc_cross_rotated_2_no_vert_scaling", "env2", "no_vert"],
    ["3D_bvc_cross_rotated_3_no_dist_scaling", "env3", "no_dist"],
    ["3D_bvc_cross_rotated_4_no_vert_or_dist_scaling", "env4", "no_vert_dist"],
]

experiments_rotated = [
    ["3D_bvc_cross_rotated_1_model0", "env1", "model0"],
    ["3D_bvc_cross_rotated_1_model1", "env1", "model1"],
    ["3D_bvc_cross_rotated_1_model2", "env1", "model2"],
    ["3D_bvc_cross_rotated_1_model3", "env1", "model3"],
    ["3D_bvc_cross_rotated_1_model4", "env1", "model4"],
    ["3D_bvc_cross_rotated_2_model0", "env2", "model0"],
    ["3D_bvc_cross_rotated_2_model1", "env2", "model1"],
    ["3D_bvc_cross_rotated_2_model2", "env2", "model2"],
    ["3D_bvc_cross_rotated_2_model3", "env2", "model3"],
    ["3D_bvc_cross_rotated_2_model4", "env2", "model4"],
    ["3D_bvc_cross_rotated_3_model0", "env3", "model0"],
    ["3D_bvc_cross_rotated_3_model1", "env3", "model1"],
    ["3D_bvc_cross_rotated_3_model2", "env3", "model2"],
    ["3D_bvc_cross_rotated_3_model3", "env3", "model3"],
    ["3D_bvc_cross_rotated_3_model4", "env3", "model4"],
    ["3D_bvc_cross_rotated_4_model0", "env4", "model0"],
    ["3D_bvc_cross_rotated_4_model1", "env4", "model1"],
    ["3D_bvc_cross_rotated_4_model2", "env4", "model2"],
    ["3D_bvc_cross_rotated_4_model3", "env4", "model3"],
    ["3D_bvc_cross_rotated_4_model4", "env4", "model4"],
]

experiments = experiments_base

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

# Set font family and sizes globally
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
    }
)

# Create subplots in a single row
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# 1. Plot proportion of non-zero cells
df["norm_non_zero_cells"] = df["non_zero_cells"] / 500  # Normalize by total cells
sns.barplot(
    data=df,
    x="env",
    y="norm_non_zero_cells",
    hue="model",
    ax=axes[0],
)
axes[0].set_title("Proportion of Non-Zero Cells")
axes[0].set_xlabel("Environment")
axes[0].set_ylabel("Proportion")
axes[0].legend(title="Model")
axes[0].grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)

# 2. Plot proportion of cells with multiple clusters
df["norm_cells_multiple_clusters"] = df["cells_with_multiple_clusters"] / 500
sns.barplot(
    data=df,
    x="env",
    y="norm_cells_multiple_clusters",
    hue="model",
    ax=axes[1],
)
axes[1].set_title("Proportion of Cells with Multiple Clusters")
axes[1].set_xlabel("Environment")
axes[1].set_ylabel("Proportion")
axes[1].legend(title="Model")
axes[1].grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)

# 3. Plot average clusters per non-zero cell
sns.barplot(
    data=df,
    x="env",
    y="avg_clusters_per_non_zero_cell",
    hue="model",
    ax=axes[2],
)
axes[2].set_title("Average Clusters per Non-Zero Cell")
axes[2].set_xlabel("Environment")
axes[2].set_ylabel("Average Clusters")
axes[2].legend(title="Model")
axes[2].grid(axis="y", linestyle="-", linewidth=0.5, alpha=1.0)

# Adjust layout
plt.tight_layout()
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
