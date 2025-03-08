# %%
from posix import replace
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def plot_single_cell_activation_3d(
    hmap_x: np.array,
    hmap_z: np.array,
    hmap_y: np.array,
    hmap_pcn: np.array,
    cell_index: int,
    min_activation: float = 0.01,
):
    """Create 3D plot of a single place cell's activation across the trajectory.

    Args:
        hmap_x: X coordinates of trajectory
        hmap_z: Z coordinates of trajectory
        hmap_y: Y coordinates of trajectory
        hmap_pcn: Place cell network activations
        cell_index: Index of the cell to visualize
        min_activation: Minimum activation threshold to filter out low activations
    """
    # Extract activations for the specified cell
    cell_activations = hmap_pcn[:, cell_index]

    # Apply threshold filter
    mask = cell_activations >= min_activation

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a colormap based on activation values
    norm = plt.Normalize(0, np.max(cell_activations))
    cmap = cm.viridis

    # Plot points with color intensity based on cell activation
    scatter = ax.scatter(
        hmap_x[mask],
        hmap_y[mask],  # y is the horizontal coordinate
        hmap_z[mask],  # z is the height
        c=cell_activations[mask],
        cmap=cmap,
        norm=norm,
        s=30,  # size of points
        alpha=0.2,  # transparency
    )

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(f"Cell {cell_index} Activation", fontsize=12)

    # Set labels and title
    ax.set_xlabel("X Position", fontsize=14)
    ax.set_ylabel("Z Position", fontsize=14)
    ax.set_zlabel("Y Position", fontsize=14)
    ax.set_title(f"Place Cell {cell_index} Activation in 3D Space", fontsize=16)

    # Set axis limits
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0, 5])

    plt.tight_layout()
    plt.show()


def plot_multiple_cells_activation_3d(
    hmap_x: np.array,
    hmap_z: np.array,
    hmap_y: np.array,
    hmap_pcn: np.array,
    cell_indices: list,
    subplot_layout=None,
    min_activation: float = 0.01,
):tau_denom
    """Create multiple 3D plots of place cell activations across the trajectory.

    Args:
        hmap_x: X coordinates of trajectory
        hmap_z: Z coordinates of trajectory
        hmap_y: Y coordinates of trajectory
        hmap_pcn: Place cell network activations
        cell_indices: List of cell indices to visualize
        subplot_layout: Optional tuple (rows, cols) for subplot layout
        min_activation: Minimum activation threshold to filter out low activations
    """
    num_cells = len(cell_indices)

    # Determine subplot layout if not provided
    if subplot_layout is None:
        cols = min(3, num_cells)
        rows = (num_cells + cols - 1) // cols
    else:
        rows, cols = subplot_layout

    # Create figure
    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    # Create a colormap
    cmap = cm.viridis

    # Global max for consistent color scaling
    global_max = np.max(hmap_pcn[:, cell_indices])
    norm = plt.Normalize(0, global_max)

    for i, cell_idx in enumerate(cell_indices):
        # Extract activations for the current cell
        cell_activations = hmap_pcn[:, cell_idx]

        # Apply threshold filter
        mask = cell_activations >= min_activation

        # Create subplot
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

        # Plot points with color intensity based on cell activation
        scatter = ax.scatter(
            hmap_x[mask],
            hmap_y[mask],
            hmap_z[mask],
            c=cell_activations[mask],
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.7,
        )

        # Set labels and title
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Z", fontsize=10)
        ax.set_zlabel("Y", fontsize=10)
        ax.set_title(f"Cell {cell_idx}", fontsize=12)

        # Set axis limits
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([0, 5])

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Cell Activation", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_overlayed_cells_3d(
    hmap_x: np.array,
    hmap_z: np.array,
    hmap_y: np.array,
    hmap_pcn: np.array,
    cell_indices: list,
    min_activation: float = 0.01,
):
    """Create a single 3D plot overlaying activations of multiple place cells with distinct colors.

    Args:
        hmap_x: X coordinates of trajectory
        hmap_z: Z coordinates of trajectory
        hmap_y: Y coordinates of trajectory
        hmap_pcn: Place cell network activations
        cell_indices: List of top cell indices to overlay
        min_activation: Minimum activation threshold to filter out low activations
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Generate distinct colors for the selected cells
    colors = generate_random_colors(len(cell_indices))

    for i, cell_index in enumerate(cell_indices):
        cell_activations = hmap_pcn[:, cell_index]

        # Apply threshold filter
        mask = cell_activations >= min_activation

        # Scatter plot for each cell with a unique color
        ax.scatter(
            hmap_x[mask],
            hmap_y[mask],
            hmap_z[mask],
            color=colors[i],
            label=f"Cell {cell_index}",
            s=20,
            alpha=0.7,
        )

    # Set labels and title
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Z Position", fontsize=12)
    ax.set_zlabel("Y Position", fontsize=12)
    ax.set_title("10 Random Place Cell Activations Overlaid", fontsize=14)

    # Set axis limits
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0, 5])

    # Add legend
    # ax.legend(loc="upper right", fontsize=10)

    plt.show()


if __name__ == "__main__":
    from visualizations.vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
        generate_random_colors,
        CONTROLLER_NAME,
        WORLD_NAME,
    )
    worlds = os.listdir('webots/controllers/flying_3D_pcn_looping/pkl/')

    for world in worlds:
        CONTROLLER_NAME = "flying_3D_pcn_looping"
        WORLD_NAME = world
        print(f'WORLD: {world}')



        # Load hmap data from hardcoded world name
        hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        for cell_index in range(0, hmap_pcn.shape[1], 5):
            plot_single_cell_activation_3d(
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                hmap_y=hmap_y,
                hmap_pcn=hmap_pcn,
                cell_index=cell_index,
            )

        plot_overlayed_cells_3d(
            hmap_x=hmap_x,
            hmap_z=hmap_z,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            cell_indices=np.random.choice(hmap_pcn.shape[1], size = 100, replace=False),
        )


    # %%
    # Get the total number of cells
    total_cells = hmap_pcn.shape[1]
    random_indices = np.random.choice(total_cells, size=10, replace=False)

    # Select 10 random cell indices
    if True:

        # Plot overlayed cells with random indices
        plot_overlayed_cells_3d(
            hmap_x=hmap_x,
            hmap_z=hmap_z,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            cell_indices=random_indices,
        )
        plot_overlayed_cells_3d(
            hmap_x=hmap_x,
            hmap_z=hmap_z,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            cell_indices=range(1000),
        )

    if True:
        index = 1
        plot_overlayed_cells_3d(
            hmap_x=hmap_x,
            hmap_z=hmap_z,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            cell_indices=[index],
        )
        plot_single_cell_activation_3d(
            hmap_x=hmap_x,
            hmap_z=hmap_z,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            cell_index=index,
        )

        print(total_cells)
        # Plot each of the randomly selected cells
        for cell_index in range(0, total_cells, 10):
            plot_single_cell_activation_3d(
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                hmap_y=hmap_y,
                hmap_pcn=hmap_pcn,
                cell_index=cell_index,
            )
