# %%
import numpy as np  # Make sure to import numpy
import matplotlib.pyplot as plt
import pickle  # Import pickle
from statsmodels.stats.weightstats import DescrStatsW


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Load hmap data
with open("hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))[10:]
with open("hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))[10:]
with open("hmap_z.pkl", "rb") as f:
    hmap_z = np.asarray(pickle.load(f))[10:]

# Calculate the place field centers (means)
num_cells = hmap_z.shape[-1]
means = np.empty([num_cells, 2])

for i in range(num_cells):
    try:
        x_mean = weighted_mean(hmap_x, weights=hmap_z[:, i])
        y_mean = weighted_mean(hmap_y, weights=hmap_z[:, i])
        means[i] = x_mean, y_mean
    except:
        means[i] = np.nan, np.nan  # Handle case with no firing

# Filter out cells with no activation (nan values)
valid_cells = ~np.isnan(means).any(axis=1)
valid_means = means[valid_cells]

# Plot the centers of the valid place fields
plt.figure(figsize=(8, 8))
plt.scatter(valid_means[:, 0], valid_means[:, 1], c="red", marker="x", s=50)
plt.title("Place Field Centers")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
