# %%
from controllers.bek_controller.visualizations.analysis_utils import *
from controllers.bek_controller.visualizations.hexbins import get_model_hexbin_metrics
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

# Extract last two parts of each path
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
# Collect metrics for each path
metrics_list = []
for path in df["full_path"]:
    try:
        print(path)
        hmap_x, hmap_y, hmap_z = load_hmaps(path)
        metrics = get_model_hexbin_metrics(hmap_x, hmap_y, hmap_z, verbose=True)
        metrics_list.append(metrics)
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        metrics_list.append(
            {
                "total_clusters": None,
                "non_zero_cells": None,
                "avg_clusters_per_cell": None,
            }
        )

# Add metrics to DataFrame
metrics_df = pd.DataFrame(metrics_list)
df = pd.concat([df, metrics_df], axis=1)
# Display results
print("\nModel Analysis Results:")
print(
    df[
        [
            "parent_dir",
            "end_dir",
            "total_clusters",
            "non_zero_cells",
            "avg_clusters_per_cell",
        ]
    ]
)

# %%
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Define the desired order for end_dir
order = ["upright", "inside_shallow", "inside_medium", "inside_steep"]

# Set up the figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Metrics Comparison by Environment Type")

# Plot total clusters
sns.barplot(
    data=df, x="end_dir", y="total_clusters", hue="parent_dir", ax=ax1, order=order
)
ax1.set_title("Total Clusters")
ax1.tick_params(axis="x", rotation=45)

# Plot non-zero cells
sns.barplot(
    data=df, x="end_dir", y="non_zero_cells", hue="parent_dir", ax=ax2, order=order
)
ax2.set_title("Non-zero Cells")
ax2.tick_params(axis="x", rotation=45)

# Plot average clusters per cell
sns.barplot(
    data=df,
    x="end_dir",
    y="avg_clusters_per_cell",
    hue="parent_dir",
    ax=ax3,
    order=order,
)
ax3.set_title("Average Clusters per Cell")
ax3.tick_params(axis="x", rotation=45)
ax3.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
