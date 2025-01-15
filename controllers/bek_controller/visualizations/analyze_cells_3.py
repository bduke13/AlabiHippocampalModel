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
# Filter for the models right before plotting
allowed_models = ["2D_250", "3D_2L_250_1", "3D_3L_250"]
df = df[df["parent_dir"].isin(allowed_models)]
# Plot the DBSCAN metrics and cosine similarity on one graph
order = ["upright", "inside_shallow", "inside_medium", "inside_steep"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Metrics Comparison by Environment Type", fontsize=16)
axes = axes.flatten()

# Plot total clusters
sns.barplot(
    data=df, x="end_dir", y="total_clusters", hue="parent_dir", ax=axes[0], order=order
)
axes[0].set_title("Total Clusters")
axes[0].tick_params(axis="x", rotation=45)

# Plot non-zero cells
sns.barplot(
    data=df,
    x="end_dir",
    y="non_zero_cells",
    hue="parent_dir",
    ax=axes[1],
    order=order,
)
axes[1].set_title("Non-Zero Cells")
axes[1].tick_params(axis="x", rotation=45)

# Plot cells with more than 1 cluster
sns.barplot(
    data=df,
    x="end_dir",
    y="cells_with_multiple_clusters",
    hue="parent_dir",
    ax=axes[2],
    order=order,
)
axes[2].set_title("Cells with More Than 1 Cluster")
axes[2].tick_params(axis="x", rotation=45)

# Plot average clusters per non-zero cell
sns.barplot(
    data=df,
    x="end_dir",
    y="avg_clusters_per_non_zero_cell",
    hue="parent_dir",
    ax=axes[3],
    order=order,
)
axes[3].set_title("Average Clusters Per Non-Zero Cell")
axes[3].tick_params(axis="x", rotation=45)

# Plot cosine sum
sns.barplot(
    data=df,
    x="end_dir",
    y="cosine_sum",
    hue="parent_dir",
    ax=axes[4],
    order=order,
)
axes[4].set_title("Total Far-Distance Cosine Sum")
axes[4].tick_params(axis="x", rotation=45)

# Hide the 6th subplot
axes[5].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
