import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def plot_cells_group(
    cell_indices,
    hmap_x,
    hmap_y,
    hmap_data,
    cell_type,
    colors_rgb,
    group_index,
    output_dir=None,
    save_plot=True,
    show_plot=False,
    return_plot=False,
):
    """
    Plots a hexbin plot for a group of cells (place or grid) and saves or shows it.

    Args:
        cell_indices: Indices of the cells to plot.
        hmap_x: X coordinates of the grid.
        hmap_y: Y coordinates of the grid.
        hmap_data: Activation data for the cells.
        cell_type: Type of cells ('place' or 'grid').
        colors_rgb: List of RGB colors for plotting.
        group_index: Index of the group for saving.
        output_dir: Directory to save the plot.
        save_plot: Boolean to save the plot.
        show_plot: Boolean to display the plot.
        return_plot: Boolean to return the figure object.
    """
    fig, axes = plt.subplots(1, len(cell_indices), figsize=(5 * len(cell_indices), 5))
    fig.suptitle(f"{cell_type.capitalize()} Cells Group {group_index}", fontsize=16)

    if len(cell_indices) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for idx, (ax, cell_index) in enumerate(zip(axes, cell_indices)):
        activations = hmap_data[:, cell_index]
        color_rgb = colors_rgb[cell_index % len(colors_rgb)]

        hb = ax.hexbin(
            hmap_x,
            hmap_y,
            C=activations,
            gridsize=100,
            reduce_C_function=np.mean,
            cmap=None,
            edgecolors="none",
        )

        counts = hb.get_array()
        max_count = counts.max() if counts.max() > 0 else 1
        counts_normalized = counts / max_count

        rgba_colors = np.zeros((len(counts), 4))
        rgba_colors[:, 0:3] = color_rgb
        rgba_colors[:, 3] = counts_normalized

        hb.set_facecolors(rgba_colors)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{cell_type.capitalize()} Cell {cell_index}")

    plt.tight_layout()

    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{cell_type}_cells_group_{group_index}.jpg")
        plt.savefig(file_path, bbox_inches="tight", dpi=500)
        print(f"Saved plot for {cell_type} cells group {group_index} to {file_path}")

    if show_plot:
        plt.show()

    if return_plot:
        return fig

    plt.close(fig)

def generate_html_report(cell_indices, cell_type, output_dir):
    """Generate an HTML report of cell visualizations."""
    html_path = os.path.join(output_dir, f"{cell_type}_cells_report.html")
    assets_path = "html_assets"

    with open(html_path, "w") as f:
        f.write(
            f"""
        <html>
        <head>
            <title>{cell_type.capitalize()} Cell Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .group-viz {{ margin-bottom: 30px; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>{cell_type.capitalize()} Cell Visualizations</h1>
            <p><strong>Report Location:</strong> <code id="report-path"></code></p>
            <script>
                document.getElementById('report-path').textContent = window.location.pathname;
            </script>
        """
        )

        num_groups = (len(cell_indices) + 4) // 5
        for i in range(num_groups):
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="{assets_path}/{cell_type}_cells_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path

def generate_cells_report(
    hmap_x, hmap_y, hmap_data, cell_type, num_modules=3, output_dir=None, open_browser=True
):
    """Generate a report for the specified cell type."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, f"{cell_type}_cells")
    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)

    num_cells = hmap_data.shape[1]
    colors_rgb = [plt.cm.hsv(i / num_cells)[:3] for i in range(num_cells)]

    if cell_type == "grid":
    # Select all neurons across modules
        num_neurons_per_module = num_cells // num_modules
        cell_indices = np.arange(num_cells)
    else:
        cell_indices = np.arange(num_cells)

    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i:i + 5]
        group_index = i // 5
        plot_cells_group(
            group,
            hmap_x,
            hmap_y,
            hmap_data,
            cell_type,
            colors_rgb,
            group_index,
            output_dir=assets_dir,
            save_plot=True,
            show_plot=False,
        )

    html_path = generate_html_report(cell_indices, cell_type, output_dir)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    print(f"HTML report generated at: {html_path}")

if __name__ == "__main__":
    # Load place cell data
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    generate_cells_report(hmap_x, hmap_y, hmap_pcn, "place")

    # Load grid cell data
    hmap_loc, hmap_grid = load_hmaps(hmap_names=["hmap_loc", "hmap_gcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    generate_cells_report(hmap_x, hmap_y, hmap_grid, "grid", num_modules=4)  # Adjust num_modules as needed