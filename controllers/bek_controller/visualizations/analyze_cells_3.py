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
                "avg_clusters_per_multi_cluster_cell": None,
            }
        )

# Convert metrics to DataFrame, combine with df
metrics_df = pd.DataFrame(metrics_list)
df = pd.concat([df, metrics_df], axis=1)

# Display DBSCAN metrics
print("\nModel Analysis Results (DBSCAN):")
print(df)

# %%
# Now compute COSINE SIMILARITIES for each path, save plot, and store sums
cosine_sums_list = []
for path in df["full_path"]:
    try:
        print("Computing cosine similarities for:", path)
        hmap_x, hmap_y, hmap_z = load_hmaps(path)

        # Run the far-distance cosine similarity analysis
        similarity_sums = analyze_cosine_similarity(
            hmap_x,
            hmap_y,
            hmap_z,
            gridsize=50,  # adjust if needed
            filter_bottom_ratio=0.1,  # adjust if needed
            distance_threshold=2.0,  # adjust if needed
        )

        # Save a scatter-plot of similarity sums in the same directory
        output_path = os.path.join(path, "cosine_sums_plot.png")
        fig, ax, total_sum = plot_similarity_sums(
            similarity_sums,
            title=f"Far-Cosine Similarity Sums ({path})",
            output_path=output_path,
            close_plot=True,
        )

        # Store the total sum for this path
        cosine_sums_list.append(total_sum)

    except Exception as e:
        print(f"Error computing cosine for {path}: {str(e)}")
        cosine_sums_list.append(None)

# Add the list of sums to the main DataFrame
df["cosine_sum"] = cosine_sums_list

# %%
# Display updated DataFrame with cosine sums
print("\nUpdated DataFrame with Cosine Sums:")
print(df)

# %%
# Plot the DBSCAN metrics by environment, as before
order = ["upright", "inside_shallow", "inside_medium", "inside_steep"]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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

# Plot average clusters per multi-cluster cell
sns.barplot(
    data=df,
    x="end_dir",
    y="avg_clusters_per_multi_cluster_cell",
    hue="parent_dir",
    ax=axes[4],
    order=order,
)
axes[4].set_title("Average Clusters Per Multi-Cluster Cell")
axes[4].tick_params(axis="x", rotation=45)

# Hide the 6th subplot
axes[5].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# Finally, create a NEW bar plot comparing the SUM of COSINE VALUES
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x="end_dir", y="cosine_sum", hue="parent_dir", order=order, ax=ax)
ax.set_title("Total Far-Distance Cosine Sum by Environment & Parent")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
