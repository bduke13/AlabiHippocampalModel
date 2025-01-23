# %%
from controllers.bek_controller.visualizations.analysis_utils import *
from controllers.bek_controller.visualizations.hexbins import (
    get_model_hexbin_metrics,
    analyze_cosine_similarity,
    plot_similarity_sums,
)
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

desired_path_ends = ["inside_shallow", "inside_medium", "inside_steep", "upright"]

MODEL_NAMES = {
    "2D_250": "2D Model",
    "3D_2L_250_1": "3D-2Layer (0.1 rad)",
    "3D_2L_250_2": "3D-2Layer (0.2 rad)",
    "3D_3L_250": "3D-3Layer",
}

ENV_NAMES = {
    "upright": "Upright",
    "inside_shallow": "Shallow Tilt 30°",
    "inside_medium": "Medium Tilt 45°",
    "inside_steep": "Steep Tilt 60°",
}


root_path = "controllers/bek_controller/IJCNN"
directories = get_available_directories(root_path)

# First filter for directories containing '250'
filtered_dirs = [d for d in directories if "250" in d]
# Then filter for desired path ends
filtered_dirs = filter_directories(filtered_dirs, desired_path_ends)
# Remove any paths containing 'html_assets'
filtered_dirs = [d for d in filtered_dirs if "html_assets" not in d]

# Extract path information
path_info = []
for directory in filtered_dirs:
    parts = directory.rstrip("/").split("/")  # Remove trailing slash and split
    last_two = parts[-2:]  # Get last two parts
    path_info.append(
        {"parent_dir": last_two[0], "end_dir": last_two[1], "full_path": directory}
    )

# Create DataFrame
df = pd.DataFrame(path_info)
print(df)

# %%
# Collect DBSCAN-based cluster metrics for each path
metrics_list = []
for path in df["full_path"]:
    try:
        print("Processing:", path)
        hmap_x, hmap_y, hmap_z = load_hmaps(path)
        metrics = get_model_hexbin_metrics(hmap_x, hmap_y, hmap_z, verbose=False)
        metrics_list.append(metrics)
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

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Metrics Comparison by Environment Type", fontsize=16)
axes = axes.flatten()

# Plot total clusters
sns.barplot(
    data=df,
    x="end_dir",
    y="total_clusters",
    hue="parent_dir",
    ax=axes[0],
    order=order,
)
axes[0].set_title("Total Clusters", fontsize=12)
axes[0].set_xticklabels([ENV_NAMES.get(env, env) for env in order])
axes[0].set_xlabel("", fontsize=10)
axes[0].set_ylabel("Clusters", fontsize=10)
axes[0].legend(
    title="Model", labels=[MODEL_NAMES[m] for m in allowed_models], loc="upper right"
)

# Plot proportion of non-zero cells
sns.barplot(
    data=df,
    x="end_dir",
    y="non_zero_cells",
    hue="parent_dir",
    ax=axes[1],
    order=order,
)
axes[1].set_title("Proportion of Non-Zero Cells", fontsize=12)
axes[1].set_xticklabels([ENV_NAMES.get(env, env) for env in order])
axes[1].set_xlabel("", fontsize=10)
axes[1].set_ylabel("Proportion", fontsize=10)

# Plot proportion of cells with multiple clusters
sns.barplot(
    data=df,
    x="end_dir",
    y="cells_with_multiple_clusters",
    hue="parent_dir",
    ax=axes[2],
    order=order,
)
axes[2].set_title("Proportion of Cells with Multiple Clusters", fontsize=12)
axes[2].set_xticklabels([ENV_NAMES.get(env, env) for env in order])
axes[2].set_xlabel("", fontsize=10)
axes[2].set_ylabel("Proportion", fontsize=10)

# Plot average clusters per non-zero cell
sns.barplot(
    data=df,
    x="end_dir",
    y="avg_clusters_per_non_zero_cell",
    hue="parent_dir",
    ax=axes[3],
    order=order,
)
axes[3].set_title("Average Clusters per Non-Zero Cell", fontsize=12)
axes[3].set_xticklabels([ENV_NAMES.get(env, env) for env in order])
axes[3].set_xlabel("", fontsize=10)
axes[3].set_ylabel("Average Clusters", fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
