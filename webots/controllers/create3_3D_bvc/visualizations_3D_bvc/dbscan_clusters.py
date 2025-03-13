# %%
from visualizations.vis_utils import *
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import (
    get_model_hexbin_metrics,
    analyze_cosine_similarity,
    plot_similarity_sums,
)
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Set pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONTROLLER_NAME = "create3_3D_bvc_looping"

# Parameters
root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")
trials = os.listdir(root_path)
# Define experiments as [world, env_name, model_name]

experiments_obstacles = [
    ["3D_bvc_two_shapes_control", "env1", "3D_bvc"],
    ["3D_bvc_two_shapes_test", "env2", "3D_bvc"],
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


experiments = experiments_rotated

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
# MODEL_NAMES = {
#    "2D_250": "2D Model",
#    "3D_2L_250_1": "3D-2Layer (0.1 rad)",
#    "3D_2L_250_2": "3D-2Layer (0.2 rad)",
#    "3D_3L_250": "3D-3Layer",
# }

# ENV_NAMES = {
#    "upright": "Upright",
#    "inside_shallow": "30° Tilt",
#    "inside_medium": "45° Tilt",
#    "inside_steep": "60° Tilt",
# }


# root_path = "IJCNN"
# directories = get_available_directories(root_path)

# First filter for directories containing '250'
# filtered_dirs = [d for d in directories if "250" in d]
# Then filter for desired path ends
# filtered_dirs = filter_directories(filtered_dirs, desired_path_ends)
# Remove any paths containing 'html_assets'
# filtered_dirs = [d for d in filtered_dirs if "html_assets" not in d]

# Extract path information
# path_info = []
# for directory in filtered_dirs:
#    parts = directory.rstrip("/").split("/")  # Remove trailing slash and split
#    last_two = parts[-2:]  # Get last two parts
#    path_info.append(
#        {"parent_dir": last_two[0], "end_dir": last_two[1], "full_path": directory}
#    )

# Create DataFrame
# df = pd.DataFrame(path_info)
# print(df)

print(filtered_dirs)
# %%
# Collect DBSCAN-based cluster metrics for each path
metrics_list = []
for path in filtered_dirs:
    try:
        print("Processing:", path)
        # hmap_loc, hmap_pcn = load_hmaps(
        # )
        hmap_loc, hmap_pcn = load_hmaps_from_dir(
            hmap_names=["hmap_loc", "hmap_pcn"], base_dir=path
        )
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

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

# %%
# Convert metrics to DataFrame, combine with df
metrics_df = pd.DataFrame(metrics_list)
df = pd.concat([df, metrics_df], axis=1)

# Display DBSCAN metrics
print("\nModel Analysis Results (DBSCAN):")
print(df)

# %%
# Filter for allowed models
allowed_models = ["2D_250", "3D_2L_250_1", "3D_2L_250_2", "3D_3L_250"]
df = df[df["parent_dir"].isin(allowed_models)]

# Desired order for environments
order = ["upright", "inside_shallow", "inside_medium", "inside_steep"]

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

# Create subplots in a single row with wider aspect ratio
fig, axes = plt.subplots(1, 3, figsize=(24, 4))  # Made wider to achieve 1:2 ratio

# Set the aspect ratio for each subplot to make the plotting area twice as wide as tall
for ax in axes:
    # Get the current position of the subplot
    pos = ax.get_position()
    # Calculate new position with 1:2 aspect ratio
    ax.set_position([pos.x0, pos.y0, pos.width, pos.width * 0.5])

#  proportion of non-zero cells
df["norm_non_zero_cells"] = df["non_zero_cells"] / 250
g1 = sns.barplot(
    data=df,
    x="end_dir",
    y="norm_non_zero_cells",
    hue="parent_dir",
    ax=axes[0],
    order=order,
)
axes[0].set_title("", fontsize=18)
axes[0].set_xticklabels([ENV_NAMES.get(env, env) for env in order], fontsize=18)
axes[0].set_xlabel("", fontsize=18)
axes[0].set_ylabel("Proportion", fontsize=18)

# Get handles and labels from the plot
handles, _ = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles=handles,
    labels=[MODEL_NAMES[m] for m in allowed_models],
    title="Model",
    loc="lower left",
)

#  proportion of cells with multiple clusters
df["norm_cells_multiple_clusters"] = df["cells_with_multiple_clusters"] / 250
g2 = sns.barplot(
    data=df,
    x="end_dir",
    y="norm_cells_multiple_clusters",
    hue="parent_dir",
    ax=axes[1],
    order=order,
)
axes[1].set_title("", fontsize=18)
axes[1].set_xticklabels([ENV_NAMES.get(env, env) for env in order], fontsize=18)
axes[1].set_xlabel("", fontsize=18)
axes[1].set_ylabel("Proportion", fontsize=18)

# Get handles and labels from the plot
handles, _ = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles=handles,
    labels=[MODEL_NAMES[m] for m in allowed_models],
    title="Model",
    loc="lower left",
)

#  average clusters per non-zero cell (already normalized, no need to divide)
g3 = sns.barplot(
    data=df,
    x="end_dir",
    y="avg_clusters_per_non_zero_cell",
    hue="parent_dir",
    ax=axes[2],
    order=order,
)
axes[2].set_title("", fontsize=18)
axes[2].set_xticklabels([ENV_NAMES.get(env, env) for env in order], fontsize=18)
axes[2].set_xlabel("", fontsize=18)
axes[2].set_ylabel("Average Clusters", fontsize=18)

# Get handles and labels from the plot
handles, _ = axes[2].get_legend_handles_labels()
axes[2].legend(
    handles=handles,
    labels=[MODEL_NAMES[m] for m in allowed_models],
    title="Model",
    loc="lower left",
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
