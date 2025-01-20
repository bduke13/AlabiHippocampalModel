# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from controllers.bek_controller.visualizations.hexbins import create_hexbin
from controllers.bek_controller.visualizations.analysis_utils import *

# Set pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Define paths and filters
root_path = "controllers/bek_controller/IJCNN"
desired_path_ends = ["inside_shallow", "inside_medium", "inside_steep", "upright"]


# Collect directories
directories = get_available_directories(root_path)
filtered_dirs = [d for d in directories if "250" in d]
filtered_dirs = filter_directories(filtered_dirs, desired_path_ends)
filtered_dirs = [d for d in filtered_dirs if "html_assets" not in d]

# Create DataFrame of paths
path_info = [
    {
        "parent_dir": directory.rstrip("/").split("/")[-2],
        "end_dir": directory.rstrip("/").split("/")[-1],
        "full_path": directory,
    }
    for directory in filtered_dirs
]
df = pd.DataFrame(path_info)

# %%
# Collect data points per hexbin for each world
bin_counts_per_world = {}

for path in df["full_path"]:
    print(f"Processing world: {path}")
    hmap_x, hmap_y, hmap_z = load_hmaps(path)
    all_counts = []

    for cell_index in range(hmap_z.shape[1]):
        _, _, _, _, counts = create_hexbin(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_z,
            normalize=True,
            analyze=True,
            get_counts=True,
            close_plot=True,
        )
        all_counts.extend(counts)

    bin_counts_per_world[path] = all_counts

# %%
bin_counts_per_world

# %%
# Plot distribution of bin counts for each world
plt.figure(figsize=(12, 8))
for world, counts in bin_counts_per_world.items():
    sns.histplot(
        counts, kde=True, label=world.split("/")[-1], bins=30, alpha=0.6, stat="density"
    )

plt.xlabel("Number of Data Points per Bin")
plt.ylabel("Density")
plt.title("Distribution of Data Points per Hexbin for Each World")
plt.legend(title="World")
plt.show()

# Analyze summary statistics
summary_stats = {
    world: {
        "mean": np.mean(counts),
        "median": np.median(counts),
        "max": np.max(counts),
        "min": np.min(counts),
        "std": np.std(counts),
    }
    for world, counts in bin_counts_per_world.items()
}
summary_df = pd.DataFrame(summary_stats).T

# Display summary statistics
print("\nSummary Statistics for Bin Counts:")
print(summary_df)

# Optionally save summary statistics to a CSV
output_csv = "bin_count_summary.csv"
summary_df.to_csv(output_csv)
print(f"Summary statistics saved to {output_csv}")
