# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser


def plot_place_cells_group(
    cell_indices,
    hmap_x,
    hmap_y,
    hmap_pcn,
    colors_rgb,
    group_index,
    output_dir=None,
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
    - hmap_pcn: The activation data for the place cells (z-axis).
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
        activations = hmap_pcn[:, cell_index]

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

    # Save the plot if save_plot flag is True and output_dir is provided
    if save_plot and output_dir is not None:
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
    assets_path = "html_assets"  # Relative path for images in HTML

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
            <p><strong>Report Location:</strong> <code id="report-path"></code></p>
            <script>
                document.getElementById('report-path').textContent = window.location.pathname;
            </script>
        """
        )

        num_groups = (len(cell_indices) + 4) // 5  # Round up division by 5
        for i in range(num_groups):
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="html_assets/place_cells_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path


def generate_place_cells_report(
    hmap_x, hmap_y, hmap_pcn, output_dir="visualizations/outputs/", open_browser=True
):
    # Create output directory and assets subdirectory
    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)
    """
    Generate an HTML report of place cell visualizations.

    Args:
        hmap_x: The x coordinates of the grid
        hmap_y: The y coordinates of the grid
        hmap_pcn: The activation data for the place cells (z-axis)
        output_dir: Directory to save the visualizations (default: "visualizations/outputs/")
        open_browser: Whether to automatically open the report in browser (default: True)

    Returns:
        str: Path to the generated HTML report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate vibrant colors programmatically using HSV color space
    def generate_vibrant_colors(n):
        colors_rgb = []
        for i in range(n):
            # Use HSV color space for more vibrant colors
            hue = i / n
            saturation = 0.9  # High saturation for vibrancy
            value = 0.95  # High value for brightness
            # Convert HSV to RGB
            color = plt.cm.hsv(hue)[:3]  # Get RGB from HSV, exclude alpha
            # Adjust for vibrancy
            color = np.array(color) * saturation * value
            colors_rgb.append(color)
        return colors_rgb

    # Generate colors for all possible cells
    num_cells = hmap_pcn.shape[1]
    colors_rgb = generate_vibrant_colors(num_cells)

    # Try to read cells.csv if it exists, otherwise use all cells
    if os.path.exists("cells.csv"):
        cell_indices = np.loadtxt("cells.csv", dtype=int)
        print(f"Loaded {len(cell_indices)} cell indices from cells.csv")
    else:
        print("cells.csv not found, using all cells")
        cell_indices = np.arange(hmap_pcn.shape[1])

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
            hmap_pcn,
            colors_rgb,
            group_index,
            output_dir=assets_dir,
            save_plot=True,
            show_plot=False,
        )

    # Generate HTML report
    html_path = generate_html_report(cell_indices, output_dir)

    # Let the browser start opening the file (if requested)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"Processed plots for {len(cell_indices)} cell(s).")
    print(f"HTML report generated at: {html_path}")
    print(f"Images saved in: {assets_dir}")

    return html_path


# %%
if __name__ == "__main__":

    from vis_utils import load_hmaps

    # Example usage
    data_path = "webots/controllers/create3_base/"
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x = hmap_loc[:, 0]
    hmap_y = hmap_loc[:, 2]

    # %%
    generate_place_cells_report(
        hmap_x, hmap_y, hmap_pcn, output_dir=data_path + "html_report_outputs/"
    )
