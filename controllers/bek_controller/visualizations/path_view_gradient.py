# %%
import numpy as np
from matplotlib.colors import Normalize
import pickle


def compute_colors(hmap_x, hmap_y, show_plot=False):
    """
    Computes the combined gradient colors for the trajectory.
    - X-axis: black to white (grayscale).
    - Y-axis: red to blue.

    Args:
    - hmap_x: Array of x coordinates.
    - hmap_y: Array of y coordinates.
    - show_plot: If True, displays the trajectory plot. Default is False.

    Returns:
    - colors: Array of RGB values associated with each point.
    """
    # Normalize X and Y values to [0, 1]
    x_norm = Normalize(vmin=min(hmap_x), vmax=max(hmap_x))(hmap_x)
    y_norm = Normalize(vmin=min(hmap_y), vmax=max(hmap_y))(hmap_y)

    # Create the X-axis grayscale gradient (black to white)
    gray_gradient = np.array([x_norm, x_norm, x_norm]).T  # Shape (n_points, 3)

    # Create the Y-axis red-to-blue gradient
    red_to_blue = np.zeros((len(hmap_y), 3))  # Shape (n_points, 3)
    red_to_blue[:, 0] = y_norm  # Red channel
    red_to_blue[:, 2] = 1 - y_norm  # Blue channel

    # Combine the two gradients (average the RGB values)
    colors = (gray_gradient + red_to_blue) / 2

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.scatter(hmap_x, hmap_y, c=colors, s=10, edgecolor="none")
        plt.title("Custom Gradient Colored Trajectory")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.show()

    return colors


if __name__ == "__main__":
    # Load hmap data
    with open("controllers/bek_controller/hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("controllers/bek_controller/hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))

    # Generate colors without plotting
    colors = compute_colors(hmap_x, hmap_y, show_plot=True)

    # Save the colors array to a file
    with open("controllers/bek_controller/point_colors.pkl", "wb") as f:
        pickle.dump(colors, f)

    print(
        f"Saved colors for {len(colors)} points to 'controllers/bek_controller/point_colors.pkl'."
    )
