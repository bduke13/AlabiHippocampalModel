import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser
import torch
from multiprocessing import Pool
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

# Set device for PyTorch (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    nx=100,
    ny=100,
):
    """
    Plots a 2D histogram for a group of cells using GPU-accelerated binning and imshow.
    
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
        nx: Number of bins in x direction.
        ny: Number of bins in y direction.
    """
    fig, axes = plt.subplots(1, len(cell_indices), figsize=(5 * len(cell_indices), 5))
    fig.suptitle(f"{cell_type.capitalize()} Cells Group {group_index}", fontsize=16)
    if len(cell_indices) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Move coordinates to GPU
    hmap_x_t = torch.from_numpy(hmap_x).to(device)
    hmap_y_t = torch.from_numpy(hmap_y).to(device)

    # Define bin edges
    xmin, xmax = hmap_x.min(), hmap_x.max()
    ymin, ymax = hmap_y.min(), hmap_y.max()
    bin_width_x = (xmax - xmin) / nx
    bin_width_y = (ymax - ymin) / ny

    # Compute bin indices (same for all cells)
    ix = ((hmap_x_t - xmin) / bin_width_x).floor().long()
    iy = ((hmap_y_t - ymin) / bin_width_y).floor().long()
    ix = torch.clamp(ix, 0, nx - 1)
    iy = torch.clamp(iy, 0, ny - 1)
    bin_id = ix * ny + iy

    # Compute count per bin (same for all cells)
    count = torch.zeros(nx * ny, device=device)
    count.scatter_add_(0, bin_id, torch.ones_like(bin_id, dtype=torch.float))
    count_2d = count.view(nx, ny).cpu().numpy()

    for idx, (ax, cell_index) in enumerate(zip(axes, cell_indices)):
        activations = hmap_data[:, cell_index]
        activations_t = torch.from_numpy(activations).to(device)

        # Compute sum of activations per bin
        sum_C = torch.zeros(nx * ny, device=device)
        sum_C.scatter_add_(0, bin_id, activations_t)
        sum_C_2d = sum_C.view(nx, ny).cpu().numpy()

        # Compute mean activations per bin
        mean_C_2d = sum_C_2d / (count_2d + 1e-6)  # Avoid division by zero

        # Normalize mean activations for alpha channel
        max_mean = mean_C_2d.max() if mean_C_2d.max() > 0 else 1
        alpha = mean_C_2d / max_mean

        # Create RGBA image
        color_rgb = colors_rgb[cell_index % len(colors_rgb)]
        rgba = np.zeros((ny, nx, 4))  # imshow expects (height, width, 4)
        rgba[..., :3] = color_rgb
        rgba[..., 3] = alpha.T  # Transpose to match (ny, nx)

        # Plot using imshow
        ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax], origin='lower')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{cell_type.capitalize()} Cell {cell_index}")

    plt.tight_layout()
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{cell_type}_cellsgroup{group_index}.jpg")
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
                <img src="{assets_path}/{cell_type}_cellsgroup{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )
        f.write("</body></html>")
    return html_path

def worker(args):
    """Worker function for multiprocessing."""
    group_index, cell_type, output_dir, nx, ny = args
    # Load data in each process
    if cell_type == "place":
        hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
        hmap_data = hmap_pcn
    else:  # grid
        hmap_loc, hmap_grid = load_hmaps(hmap_names=["hmap_loc", "hmap_gcn"])
        hmap_data = hmap_grid
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    num_cells = hmap_data.shape[1]
    cell_indices = np.arange(num_cells)
    start = group_index * 5
    end = start + 5
    group = cell_indices[start:end]
    if len(group) == 0:
        return
    colors_rgb = [plt.cm.hsv(i / num_cells)[:3] for i in range(num_cells)]
    plot_cells_group(
        group,
        hmap_x,
        hmap_y,
        hmap_data,
        cell_type,
        colors_rgb,
        group_index,
        output_dir=output_dir,
        save_plot=True,
        show_plot=False,
        nx=nx,
        ny=ny,
    )

def generate_cells_report(
    cell_type, num_modules=3, output_dir=None, open_browser=True, num_processes=8, nx=100, ny=100
):
    """
    Generate a report for the specified cell type using multiprocessing and GPU acceleration.
    
    Args:
        cell_type: Type of cells ('place' or 'grid').
        num_modules: Number of modules for grid cells.
        output_dir: Directory to save the report.
        open_browser: Boolean to open the HTML report in a browser.
        num_processes: Number of parallel processes.
        nx: Number of bins in x direction.
        ny: Number of bins in y direction.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, f"{cell_type}_cells")
    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Load data once to determine num_cells
    if cell_type == "place":
        _, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
        num_cells = hmap_pcn.shape[1]
    else:
        _, hmap_grid = load_hmaps(hmap_names=["hmap_loc", "hmap_gcn"])
        num_cells = hmap_grid.shape[1]
    num_groups = (num_cells + 4) // 5  # Ceiling division

    # Parallelize plotting across groups
    with Pool(processes=num_processes) as pool:
        pool.map(worker, [(i, cell_type, assets_dir, nx, ny) for i in range(num_groups)])

    # Generate HTML report
    cell_indices = np.arange(num_cells)
    html_path = generate_html_report(cell_indices, cell_type, output_dir)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    print(f"HTML report generated at: {html_path}")

if __name__ == "__main__":
    # Place cells
    generate_cells_report("place", num_processes=8)
    # Grid cells
    generate_cells_report("grid", num_modules=4, num_processes=8)  # Adjust num_modules as needed