import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Function to plot place field centers with adjacencies
def plot_place_fields_with_adjacencies(
    valid_means, adjacencies, means, valid_cells, hmap_x, hmap_y, image_path=""
):
    if not os.path.exists(image_path):
        print(f"WARNING: {image_path} does not exist. Using blank background.")
        image_path = None

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    if image_path:
        # Load and plot the background image if provided
        img = plt.imread(image_path)
        ax.imshow(
            img,
            extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
            origin="upper",
        )

    # Plot the place field centers
    plt.scatter(valid_means[:, 0], valid_means[:, 1], c="red", marker="x", s=50)

    # Plot the adjacencies
    for i, j in adjacencies:
        if valid_cells[i] and valid_cells[j]:
            x_values = [means[i, 0], means[j, 0]]
            y_values = [means[i, 1], means[j, 1]]
            plt.plot(x_values, y_values, color="blue", linewidth=0.5)

    plt.title("Place Field Centers with Adjacencies")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


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

# Load the saved place cell network
with open("pcn.pkl", "rb") as f:
    pcn = pickle.load(f)

# Get adjacencies
adjacencies = pcn.get_adjacencies()

# Plot with the environment image as background
image_path = "environment_images/5x5_env_image.jpg"  # Set this to an empty string if no background is needed
plot_place_fields_with_adjacencies(
    valid_means, adjacencies, means, valid_cells, hmap_x, hmap_y, image_path
)
