# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors
import webbrowser


def plot_place_cells_group(
    cell_indices,
    hmap_x,
    hmap_y,
    hmap_z,
    colors_rgb,
    group_index,
    output_dir="visualizations/outputs/",
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
    # Create figure with 5 subplots side by side
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Place Cells Group {group_index}", fontsize=16)

    for idx, (ax, cell_index) in enumerate(zip(axes, cell_indices)):
        # Get activations for this cell
        activations = hmap_z[:, cell_index]

        # Color for this cell
        color_rgb = colors_rgb[
            cell_index % len(colors_rgb)
        ]  # Use modulo to avoid index out of range

        # Create a hexbin plot
        hb = ax.hexbin(
            hmap_x,
            hmap_y,
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
        rgba_colors[:, 0:3] = color_rgb
        rgba_colors[:, 3] = counts_normalized

        # Set the facecolors of the hexbin collection
        hb.set_facecolors(rgba_colors)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Cell {cell_index}")

    plt.tight_layout()

    # Save the plot if save_plot flag is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"place_cells_group_{group_index}.jpg")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot for place cells group {group_index} to {file_path}")

    # Show the plot if show_plot flag is True
    if show_plot:
        plt.show()

    # Return the figure if return_plot flag is True
    if return_plot:
        return fig

    # Close the figure to save memory
    plt.close(fig)


def generate_html_report(cell_indices, output_dir):
    """Generate an HTML report of all place cell visualizations."""
    html_path = os.path.join(output_dir, "place_cells_report.html")

    with open(html_path, "w") as f:
        f.write(
            """
        <html>
        <head>
            <title>Place Cell Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .group-viz { margin-bottom: 30px; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Place Cell Visualizations</h1>
        """
        )

        num_groups = (len(cell_indices) + 4) // 5  # Round up division by 5
        for i in range(num_groups):
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="place_cells_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path


if __name__ == "__main__":
    # Create the output directory
    output_dir = "visualizations/outputs/"
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

    # Select top 100 cells based on total activation
    total_activations = np.sum(hmap_z, axis=0)
    cell_indices = np.argsort(total_activations)[-200:]

    # Generate and save plots in groups of 5
    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i : i + 5]
        # Pad the group with None if it's not complete
        while len(group) < 5:
            group = np.append(group, group[-1])

        group_index = i // 5
        plot_place_cells_group(
            group,
            hmap_x,
            hmap_y,
            hmap_z,
            colors_rgb,
            group_index,
            output_dir=output_dir,
            save_plot=True,
            show_plot=False,
        )

    # Generate HTML report
    html_path = generate_html_report(cell_indices, output_dir)

    # Open in default browser
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"Processed plots for {len(cell_indices)} cell(s).")
    print(f"HTML report generated at: {html_path}")
