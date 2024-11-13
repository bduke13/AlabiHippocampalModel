# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.interpolate import interp1d


def plot_trajectory(
    hmap_x,
    hmap_y,
    image_path="visualizations/environment_images/5x5_env_image.jpg",
):
    """
    Plots the trajectory path over the environment image by interpolating between points.

    Args:
    - hmap_x: Array of x coordinates
    - hmap_y: Array of y coordinates
    - image_path: Path to the background environment image
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Load and plot background image if it exists
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        ax.imshow(
            img,
            extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
            origin="upper",
        )
    else:
        print(f"WARNING: {image_path} does not exist. Using blank background.")

    # Create more points for smoother path
    t = np.arange(len(hmap_x))
    t_interp = np.linspace(0, len(hmap_x) - 1, num=len(hmap_x) * 10)

    # Interpolate x and y coordinates
    f_x = interp1d(t, hmap_x, kind="cubic")
    f_y = interp1d(t, hmap_y, kind="cubic")

    x_smooth = f_x(t_interp)
    y_smooth = f_y(t_interp)

    # Plot the path
    plt.plot(x_smooth, y_smooth, "b-", linewidth=1, alpha=0.7, label="Path")

    # Plot points at each x,y coordinate
    plt.plot(hmap_x, hmap_y, "bo", markersize=3, alpha=0.7)

    plt.title("Agent Trajectory")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Load hmap data
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))

    # Plot the trajectory
    plot_trajectory(hmap_x, hmap_y)
    print("Trajectory plotting complete.")
