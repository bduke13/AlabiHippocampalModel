# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors


def plot_place_cell(
    cell_index,
    hmap_x,
    hmap_y,
    hmap_z,
    colors_rgb,
    output_dir="place_cell_images/",
    save_plot=True,
    show_plot=False,
    return_plot=False,
):
    """
    Plots a hexbin plot for a given place cell index and saves or shows it based on flags.

    Args:
    - cell_index: The index of the place cell to plot.
    - hmap_x: The x coordinates of the grid.
    - hmap_y: The y coordinates of the grid.
    - hmap_z: The activation data for the place cells (z-axis).
    - colors_rgb: List of RGB colors for plotting.
    - output_dir: Directory to save the plot (default is 'place_cell_images/').
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot on the screen (default is False).
    - return_plot: Boolean flag to return the figure object (default is False).
    """
    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Positions
    x = hmap_x
    y = hmap_y

    # Color for this cell
    color_rgb = colors_rgb[
        cell_index % len(colors_rgb)
    ]  # Use modulo to avoid index out of range

    # Create a hexbin plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
    hb = ax.hexbin(
        x,
        y,
        C=activations,
        gridsize=50,
        reduce_C_function=np.mean,
        cmap=None,
        edgecolors="none",
    )

    # Get aggregated activations per bin
    counts = hb.get_array()

    # Normalize counts for alpha values
    max_count = counts.max()
    if max_count > 0:
        counts_normalized = counts / max_count
    else:
        counts_normalized = counts

    # Create RGBA colors
    rgba_colors = np.zeros((len(counts), 4))
    rgba_colors[:, 0:3] = color_rgb  # Set RGB values
    rgba_colors[:, 3] = counts_normalized  # Set alpha based on activation

    # Set the facecolors of the hexbin collection
    hb.set_facecolors(rgba_colors)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Place Cell {cell_index} Activation")

    # Save the plot if save_plot flag is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"place_cell_{cell_index}.jpg")
        plt.savefig(file_path)
        print(f"Saved plot for place cell {cell_index} to {file_path}")

    # Show the plot if show_plot flag is True
    if show_plot:
        plt.show()

    # Return the figure if return_plot flag is True
    if return_plot:
        return fig

    # Close the figure to save memory
    plt.close(fig)


if __name__ == "__main__":
    # Create the directory if it doesn't exist
    output_dir = "place_cell_images/"
    os.makedirs(output_dir, exist_ok=True)

    # Load the colors list
    with open("visualizations/colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Load hmap data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    # Option to provide a list of specific place cells
    specific_cells = (
        # Replace with list of specific cell indices if needed, e.g., [0, 5, 10]
        [5, 0]
    )

    # If no specific cells are provided, select the top 100 cells based on total activation
    if specific_cells is None:
        total_activations = np.sum(hmap_z, axis=0)
        cell_indices = np.argsort(total_activations)[-100:]
    else:
        cell_indices = specific_cells

    # Loop through selected cells and save or show the plots
    for cell_index in cell_indices:
        plot_place_cell(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_z,
            colors_rgb,
            output_dir,
            save_plot=False,
            show_plot=True,
        )

    print(f"Processed plots for {len(cell_indices)} cell(s).")
