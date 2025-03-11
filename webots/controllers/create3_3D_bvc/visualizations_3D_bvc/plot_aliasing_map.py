# %%
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import *

from visualizations.vis_utils import (
    load_hmaps,
    convert_xzy_hmaps,
)

# Load hmap data from hardcoded world name
hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

# Perform cosine similarity analysis
similarity_sums = analyze_cosine_similarity_torch(
    hmap_x,
    hmap_y,
    hmap_pcn,
    gridsize=50,
    filter_bottom_ratio=0.1,
    distance_threshold=3.0,
    device="cuda",
)

# Plot cosine similarity heatmap for a specific cell
fig, ax, total_sum = plot_similarity_sums(
    similarity_sums,
    title="My Cosine Similarity Plot",
    output_path=None,
    close_plot=False,
)

# Calculate the number of points (bins with similarity values)
num_points = len(similarity_sums)

# Calculate MSAI by dividing by the square of the number of points
msai = total_sum / (num_points**2) if num_points > 0 else 0

# Print both the raw sum and the normalized MSAI value
print(f"Raw Similarity Sum: {total_sum:.6f}")
print(f"Number of points: {num_points}")
print(f"Mean Spatial Aliasing Index (MSAI): {msai:.6f}")

fig.show()
