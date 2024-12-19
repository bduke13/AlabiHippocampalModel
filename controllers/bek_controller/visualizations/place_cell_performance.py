import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.spatial.distance import cdist


def plot_distance_map(
    hmap_x,
    hmap_y,
    hmap_z,
    image_path="visualizations/environment_images/5x5_env_image.jpg",
    random_index=None,
    cmap="viridis",
):
    """
    Plots the Euclidean distance from a chosen activation vector to all others on the map.

    Args:
        hmap_x: (N,) array of x-coordinates for positions.
        hmap_y: (N,) array of y-coordinates for positions.
        hmap_z: (N, D) array of activation vectors (N positions, D dimensions).
        image_path: Path to the environment background image.
        random_index: Index of the reference point. If None, a random one is chosen.
        cmap: Colormap to use for distance visualization.
    """

    # Choose a reference index if not provided
    if random_index is None:
        random_index = np.random.randint(0, len(hmap_x))

    ref_activation = hmap_z[random_index]

    # Compute Euclidean distances from this reference point to all others
    # hmap_z: shape (N, D)
    # ref_activation: shape (D,)
    # We can use cdist to vectorize this
    distances = cdist(
        hmap_z, ref_activation[np.newaxis, :], metric="euclidean"
    ).flatten()

    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot background image if it exists
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        # We assume the extent matches the range of hmap_x and hmap_y
        ax.imshow(
            img,
            extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
            origin="upper",
        )
    else:
        print(f"WARNING: {image_path} does not exist. Using blank background.")

    # Scatter plot of points colored by distance
    sc = plt.scatter(
        hmap_x, hmap_y, c=distances, cmap=cmap, alpha=0.9, s=20, edgecolors="none"
    )

    # Add a colorbar to show the distance scale
    cbar = plt.colorbar(sc)
    cbar.set_label("Euclidean Distance")

    plt.title(f"Distances from point {random_index}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Load data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.array(pickle.load(f))

    # You can provide a specific index or leave it None for random
    plot_distance_map(hmap_x, hmap_y, hmap_z, random_index=None)
    print("Distance map plotted.")
