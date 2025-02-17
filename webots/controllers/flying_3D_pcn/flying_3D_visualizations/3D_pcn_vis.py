# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import random


def plot_3d_trajectory_with_pcn(
    hmap_x: np.array,
    hmap_z: np.array,
    hmap_y: np.array,
    hmap_pcn: np.array,
    cell_indices_to_plot=None,
):
    """Create 3D plot of trajectory colored by most active place cell.

    Args:
        prefix: Path to data files
        cell_indices_to_plot: Optional list of cell indices to plot. If None, plots all cells.
    """
    # Find most active place cell at each timestep
    most_active_cells = np.argmax(hmap_pcn, axis=1)

    if cell_indices_to_plot is not None:
        # Create a mask for points where the most active cell is in our list
        mask = np.isin(most_active_cells, cell_indices_to_plot)

        # Filter the data points
        hmap_x = hmap_x[mask]
        hmap_y = hmap_y[mask]
        hmap_z = hmap_z[mask]
        most_active_cells = most_active_cells[mask]

        print(f"Plotting points for cells: {cell_indices_to_plot}")
        print(f"Number of points: {len(hmap_x)}")

    # Generate colors for each possible place cell index
    max_cell_id = np.max(most_active_cells) + 1
    cell_colors = generate_random_colors(max_cell_id)

    # Create color map for trajectory
    trajectory_colors = [cell_colors[cell_id] for cell_id in most_active_cells]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points colored by most active place cell
    ax.scatter(
        hmap_x,
        hmap_y,  # y is the horizontal coordinate
        hmap_z,  # z is the height
        c=trajectory_colors,
        s=10,  # size of points
    )

    # Set labels and title
    ax.set_xlabel("X Position", fontsize=14)
    ax.set_ylabel("Z Position", fontsize=14)
    ax.set_zlabel("Y Position", fontsize=14)
    ax.set_title("3D Trajectory Colored by Most Active Place Cell", fontsize=16)

    # Set axis limits
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0, 3])

    plt.show()


if __name__ == "__main__":
    from visualizations.vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
        generate_random_colors,
    )

    # Example usage
    data_path = "webots/controllers/flying_3D_pcn/"
    hmap_loc, hmap_pcn = load_hmaps()
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    plot_3d_trajectory_with_pcn(
        hmap_x=hmap_x,
        hmap_z=hmap_z,
        hmap_y=hmap_y,
        hmap_pcn=hmap_pcn,
        cell_indices_to_plot=range(1, 500),
    )
