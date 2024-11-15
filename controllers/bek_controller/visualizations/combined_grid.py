import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from hexbins import plot_place_cell
import matplotlib.colors as mcolors


def create_grid(
    plot_objects,
    grid_size,
    output_dir="combined_plots",
    save_grid=True,
    show_grid=False,
    subplot_size=3,  # New argument to control individual subplot size
):
    """
    Creates a grid of plots from a list of Matplotlib Figure objects.

    Args:
    - plot_objects: A list of Matplotlib Figure objects.
    - grid_size: The size of the grid (e.g., 5 for a 5x5 grid).
    - output_dir: Directory to save the grid (default is 'combined_plots').
    - save_grid: Boolean flag to save the grid plot (default is True).
    - show_grid: Boolean flag to display the grid plot (default is False).
    - subplot_size: Size of each subplot (in inches).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_plots = len(plot_objects)
    cells_per_grid = grid_size**2

    for i in range(0, num_plots, cells_per_grid):
        # Set the figure size based on the number of subplots and the desired subplot size
        fig, axes = plt.subplots(
            grid_size,
            grid_size,
            figsize=(grid_size * subplot_size, grid_size * subplot_size),
        )

        for idx in range(cells_per_grid):
            row, col = divmod(idx, grid_size)
            if i + idx < num_plots:
                plot_obj = plot_objects[i + idx]

                # Ensure the figure has been drawn to create the renderer
                plot_obj.canvas.draw()

                # Render the figure object into the corresponding subplot
                axes[row, col].imshow(plot_obj.canvas.buffer_rgba())
                axes[row, col].axis("off")
            else:
                # Turn off axes for empty subplots
                axes[row, col].axis("off")

        if save_grid:
            grid_output_path = os.path.join(
                output_dir, f"grid_{i + 1}_to_{i + cells_per_grid}.jpg"
            )
            plt.savefig(grid_output_path)
            print(f"Saved grid of plots to {grid_output_path}")

        if show_grid:
            plt.show()

        # Close the figure to save memory after showing the grid
        plt.close(fig)


if __name__ == "__main__":
    # Assuming plots are generated using the plot_place_cell from hexbins
    output_dir = "combined_plots"

    # Load hmap data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    # Load the colors list
    with open("visualizations/colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    cell_indices = [x for x in range(hmap_z.shape[1])]

    # Collecting the plot objects
    plot_objects = []

    # Generate the plots and collect them
    for cell_index in cell_indices:
        plot = plot_place_cell(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_z,
            colors_rgb,
            save_plot=False,  # Don't save plots, just collect them
            show_plot=False,  # Don't show individual plots
            return_plot=True,  # Return the plot object to store
        )
        plot_objects.append(plot)
        plt.close(plot)

    # Now, create the grid and show all figures
    create_grid(plot_objects, grid_size=5, save_grid=False, show_grid=True)

    print("Grid plot creation complete.")
