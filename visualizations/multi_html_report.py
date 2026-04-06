# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import from vis_utils
from visualizations.vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR,
)
from visualizations.place_cell_learning import load_place_cell_learning_scores


CELL_SORT_MODE = "learning_strength"


def _network_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"

def load_scale_hmaps(scale):
    """
    Load history map (hmap) data for a specific scale.
    
    Args:
        scale (int): Scale to load
        
    Returns:
        tuple: (hmap_loc, hmap_pcn)
    """
    import pickle
    import os
    import numpy as np
    
    # Define the path for hmaps
    hmap_directory = os.path.join(
        CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps"
    )
    
    # Load location data
    hmap_file = "hmap_loc.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    with open(file_path, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        # remove first element
        hmap_loc = hmap_loc[1:]
    print(f"Loaded hmap_loc from {file_path}")
    
    # Load PCN data for the specified scale
    hmap_file = f"hmap_pcn_scale_{scale}.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    try:
        with open(file_path, "rb") as f:
            hmap_pcn = np.array(pickle.load(f))
            # remove first element
            hmap_pcn = hmap_pcn[1:]
            print(f"Loaded hmap_pcn_scale_{scale} from {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Cannot process scale {scale}.")
        return None, None
    
    return hmap_loc, hmap_pcn


def plot_place_cells_group(
    cell_indices,
    hmap_x,
    hmap_y,
    hmap_pcn,
    colors_rgb,
    group_index,
    scale,
    output_dir=None,
    save_plot=True,
    show_plot=False,
    return_plot=False,
):
    """
    Plots a hexbin plot for a given place cell index and saves or shows it based on flags.

    Args:
    - cell_indices: List of place cell indices to plot.
    - hmap_x: The x coordinates of the grid.
    - hmap_y: The y coordinates of the grid.
    - hmap_pcn: The activation data for the place cells (z-axis).
    - colors_rgb: List of RGB colors for plotting.
    - group_index: The index of the group being plotted.
    - scale: The scale of place cells being plotted.
    - output_dir: Directory to save the plot.
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot on the screen (default is False).
    - return_plot: Boolean flag to return the figure object (default is False).
    """
    # Create figure with 5 subplots side by side
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Place Cells Group {group_index} - Scale {scale}", fontsize=16)

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
        file_path = os.path.join(output_dir, f"place_cells_scale_{scale}_group_{group_index}.jpg")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot for place cells scale {scale}, group {group_index} to {file_path}")

    # Show the plot if show_plot flag is True
    if show_plot:
        plt.show()

    # Return the figure if return_plot flag is True
    if return_plot:
        return fig

    # Close the figure to save memory
    plt.close(fig)


def generate_html_report(cell_indices, scale, output_dir):
    """Generate an HTML report of all place cell visualizations for a specific scale."""
    html_path = os.path.join(output_dir, f"place_cells_report_scale_{scale}.html")
    assets_path = "html_assets"  # Relative path for images in HTML

    with open(html_path, "w") as f:
        f.write(
            f"""
        <html>
        <head>
            <title>Place Cell Visualizations - Scale {scale}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .group-viz {{ margin-bottom: 30px; }}
                h1 {{ color: #333; }}
                .scale-links {{ margin-bottom: 20px; }}
                .scale-links a {{ margin-right: 15px; }}
            </style>
        </head>
        <body>
            <h1>Place Cell Visualizations - Scale {scale}</h1>
            <p><strong>Report Location:</strong> <code id="report-path"></code></p>
            
            <div class="scale-links">
                <strong>Other Scales:</strong>
                <a href="place_cells_report_scale_0.html">Scale 0</a>
                <a href="place_cells_report_scale_1.html">Scale 1</a>
                <a href="place_cells_report_scale_2.html">Scale 2</a>
            </div>
            
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
                <img src="html_assets/place_cells_scale_{scale}_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path


def generate_place_cells_report(
    scale,
    hmap_x,
    hmap_y,
    hmap_pcn,
    output_dir=None,
    open_browser=True,
    activation_threshold=None,
):
    """
    Generate an HTML report of place cell visualizations for a specific scale.

    Args:
        scale: The scale of place cells
        hmap_x: The x coordinates of the grid
        hmap_y: The y coordinates of the grid
        hmap_pcn: The activation data for the place cells (z-axis)
        output_dir: Directory to save the visualizations
        open_browser: Whether to automatically open the report in browser
        activation_threshold: Threshold for filtering out low-activity cells

    Returns:
        str: Path to the generated HTML report
    """
    # Use project root if output_dir not specified
    if output_dir is None:
        output_dir = os.path.join(
            OUTPUT_DIR, "place_cells"
        )
    
    # Create output directory and assets subdirectory
    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)
    
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

    total_activations = np.sum(np.abs(hmap_pcn), axis=0)
    candidate_indices = np.arange(hmap_pcn.shape[1], dtype=int)
    if activation_threshold is not None:
        active_mask = total_activations > activation_threshold
        candidate_indices = candidate_indices[active_mask]
        print(
            f"Scale {scale}: Filtered out {np.sum(~active_mask)} cells below display threshold {activation_threshold}"
        )
        print(f"Scale {scale}: Remaining cells: {len(candidate_indices)}")

    # Try to read cells.csv if it exists, otherwise use all cells
    cells_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cells_scale_{scale}.csv")
    if os.path.exists(cells_csv_path):
        cell_indices = np.loadtxt(cells_csv_path, dtype=int)
        if activation_threshold is not None:
            cell_indices = np.intersect1d(cell_indices, candidate_indices, assume_unique=False)
        print(f"Scale {scale}: Loaded {len(cell_indices)} cell indices from {cells_csv_path}")
    else:
        cell_indices = candidate_indices
        print(f"Scale {scale}: No cells.csv found, using ranked cells")

    ranking_note = "csv/fallback order"
    network_dir = _network_dir()
    unified_exists = (network_dir / "unified_pcn.pkl").exists()
    cell_scores, score_meta = load_place_cell_learning_scores(
        network_dir=network_dir,
        scale=scale,
        unified=unified_exists,
    )
    if (
        CELL_SORT_MODE == "learning_strength"
        and cell_scores is not None
        and len(cell_scores) == hmap_pcn.shape[1]
        and len(cell_indices) > 0
    ):
        score_slice = np.asarray(cell_scores, dtype=np.float32)[cell_indices]
        valid_mask = np.isfinite(score_slice) & (score_slice > 0)
        if np.any(valid_mask):
            cell_indices = cell_indices[valid_mask]
            score_slice = score_slice[valid_mask]
            sort_order = np.argsort(score_slice)[::-1]
            cell_indices = cell_indices[sort_order]
            ranking_note = "learning strength"
            if score_meta.get("used_fallback"):
                ranking_note += " [current-weight fallback]"
    elif len(cell_indices) > 0:
        sort_order = np.argsort(total_activations[cell_indices])[::-1]
        cell_indices = cell_indices[sort_order]
        ranking_note = "activity fallback"

    print(f"Scale {scale}: Sorted {len(cell_indices)} cells by {ranking_note}")

    # Generate and save plots in groups of 5
    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i : i + 5]
        # Pad the group with None if it's not complete
        group_length = len(group)
        if group_length < 5:
            # Duplicate the last element to fill the group
            group = np.append(group, np.repeat(group[-1], 5 - group_length))

        group_index = i // 5
        plot_place_cells_group(
            group,
            hmap_x,
            hmap_y,
            hmap_pcn,
            colors_rgb,
            group_index,
            scale,
            output_dir=assets_dir,
            save_plot=True,
            show_plot=False,
        )

    # Generate HTML report
    html_path = generate_html_report(cell_indices, scale, output_dir)

    # Let the browser start opening the file (if requested)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"Scale {scale}: Processed plots for {len(cell_indices)} cell(s)")
    if activation_threshold is not None:
        print(f"Scale {scale}: Display threshold: {activation_threshold}")
    print(f"Scale {scale}: HTML report: {html_path}")
    print(f"Scale {scale}: Images directory: {assets_dir}")

    return html_path


def generate_multi_scale_reports(
    scales=None,
    output_dir=None,
    open_browser=True,
    activation_threshold=None
):
    """
    Generate HTML reports for multiple scales of place cells.
    
    Args:
        scales: List of scales to generate reports for
        output_dir: Directory to save the reports
        open_browser: Whether to open the browser for each report
        activation_threshold: Threshold for filtering out inactive cells
        
    Returns:
        list: Paths to the generated HTML reports
    """
    if scales is None:
        scales = [0, 1, 2]
        
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "place_cells")
    
    # Create list to store report paths
    html_reports = []
    
    # Generate report for each scale
    for scale in scales:
        print(f"\n==== Processing Scale {scale} ====")
        
        # Load data for this scale
        hmap_loc, hmap_pcn = load_scale_hmaps(scale)
        
        # Skip this scale if data couldn't be loaded
        if hmap_loc is None or hmap_pcn is None:
            print(f"Skipping scale {scale} due to missing data")
            continue
            
        # Convert coordinates
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        
        # Generate report
        html_path = generate_place_cells_report(
            scale,
            hmap_x,
            hmap_y,
            hmap_pcn,
            output_dir=output_dir,
            open_browser=open_browser,
            activation_threshold=activation_threshold
        )
        
        # Store the report path
        html_reports.append(html_path)
        
    return html_reports


# %%
if __name__ == "__main__":
    print("Starting multi-scale place cell HTML report generation...")
    
    # Generate reports for scales 0, 1, and 2
    html_reports = generate_multi_scale_reports(
        scales=[0, 1, 2],
        activation_threshold=None,
        open_browser=True
    )
    
    print("\nAll reports generated successfully!")
    for i, report in enumerate(html_reports):
        print(f"Report {i+1}: {report}")
