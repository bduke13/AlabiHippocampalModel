"""
Simple place field visualization for all environments.
Directly loads and plots data without relying on complex imports.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration
CONTROLLER_NAME = "msg_controller"
ENVIRONMENTS = [
    "environment_1",
    "environment_2",
    "environment_3",
    "environment_4",
    "environment_5",
    "environment_6"
]
ARCHITECTURE = "auto"  # "auto", "unified", "separate"

# Absolute paths
DATA_BASE_DIR = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl"
OUTPUT_DIR = PROJECT_ROOT / "place_field_plots"


def generate_random_colors(n):
    """Generate n random vibrant colors."""
    np.random.seed(42)  # For reproducibility
    colors = []
    for _ in range(n):
        # Generate vibrant colors by ensuring at least one RGB component is high
        color = np.random.rand(3)
        max_idx = np.argmax(color)
        color[max_idx] = np.random.uniform(0.7, 1.0)
        colors.append(color)
    return colors


def load_hmap_loc(env_name):
    """Load location history for an environment."""
    hmap_dir = DATA_BASE_DIR / env_name / "hmaps"
    loc_file = hmap_dir / "hmap_loc.pkl"
    if not loc_file.exists():
        return None
    with open(loc_file, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
    if len(hmap_loc) > 1:
        hmap_loc = hmap_loc[1:]  # Remove first element (legacy behavior)
    return hmap_loc


def discover_available_scales(env_name):
    """Discover scale indices from hmap files on disk."""
    hmap_dir = DATA_BASE_DIR / env_name / "hmaps"
    if not hmap_dir.exists():
        return []
    scale_files = sorted(hmap_dir.glob("hmap_pcn_scale_*.pkl"))
    scales = []
    for fpath in scale_files:
        stem = fpath.stem  # hmap_pcn_scale_{i}
        try:
            scales.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return scales


def is_unified_run(env_name):
    """Check whether this environment uses unified architecture artifacts."""
    net_dir = DATA_BASE_DIR / env_name / "networks"
    return (net_dir / "unified_pcn.pkl").exists()


def load_hmap_data(env_name, scale):
    """
    Load heatmap data for a specific environment and scale.

    Returns:
        (hmap_loc, hmap_pcn) or (None, None) if files don't exist
    """
    hmap_dir = DATA_BASE_DIR / env_name / "hmaps"
    hmap_loc = load_hmap_loc(env_name)
    if hmap_loc is None:
        return None, None

    # Load PCN data
    pcn_file = hmap_dir / f"hmap_pcn_scale_{scale}.pkl"
    if not pcn_file.exists():
        return hmap_loc, None

    with open(pcn_file, "rb") as f:
        hmap_pcn = np.array(pickle.load(f))
    if len(hmap_pcn) > 1:
        hmap_pcn = hmap_pcn[1:]  # Remove first element

    return hmap_loc, hmap_pcn


def load_unified_aggregate_data(env_name, available_scales):
    """
    Load unified aggregate activations by concatenating all available scale matrices.
    This supports unified architecture where activations are stored per-scale in hmaps.
    """
    hmap_loc = load_hmap_loc(env_name)
    if hmap_loc is None or not available_scales:
        return None, None

    per_scale = []
    min_steps = len(hmap_loc)
    for scale in available_scales:
        _, hmap_pcn = load_hmap_data(env_name, scale)
        if hmap_pcn is None:
            continue
        per_scale.append(hmap_pcn)
        min_steps = min(min_steps, len(hmap_pcn))

    if not per_scale:
        return hmap_loc, None

    # Trim all arrays to common length then concatenate along cell dimension.
    hmap_loc = hmap_loc[:min_steps]
    per_scale = [arr[:min_steps] for arr in per_scale]
    hmap_pcn_unified = np.concatenate(per_scale, axis=1)
    return hmap_loc, hmap_pcn_unified


def plot_place_cells_overlay(env_name, scale, hmap_loc, hmap_pcn, num_cells=None, gridsize=200):
    """
    Create an overlay plot showing place cell firing fields.

    Args:
        env_name: Environment name
        scale: Scale index
        hmap_loc: Location data [N, 3] (x, y, z)
        hmap_pcn: Place cell activations [N, num_cells]
        num_cells: Number of cells to plot (None = all active cells)
        gridsize: Grid resolution
    """
    # Extract x and y coordinates for 2D navigation plane in the saved hmaps.
    hmap_x = hmap_loc[:, 0]
    hmap_y = hmap_loc[:, 1]

    # Find cells with non-zero activation
    total_activation = np.sum(hmap_pcn, axis=0)
    active_cells = np.where(total_activation > 0)[0]

    if len(active_cells) == 0:
        print(f"  WARNING: No active cells found for scale {scale}")
        return None

    # Use all active cells if num_cells is None, otherwise sample
    if num_cells is None:
        cell_indices = active_cells
        num_to_plot = len(active_cells)
    else:
        num_to_plot = min(num_cells, len(active_cells))
        cell_indices = np.random.choice(active_cells, size=num_to_plot, replace=False)

    print(f"  Plotting {num_to_plot} cells out of {len(active_cells)} active cells")

    # Define grid
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)

    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    # Initialize grids
    total_activations = np.zeros((gridsize, gridsize, num_to_plot))
    counts = np.zeros((gridsize, gridsize, num_to_plot))

    # Bin the data
    for i, (x, y) in enumerate(zip(hmap_x, hmap_y)):
        xi = np.digitize(x, xedges) - 1
        yi = np.digitize(y, yedges) - 1

        if 0 <= xi < gridsize and 0 <= yi < gridsize:
            for j, cell_idx in enumerate(cell_indices):
                activation = hmap_pcn[i, cell_idx]
                total_activations[xi, yi, j] += activation
                counts[xi, yi, j] += 1

    # Calculate mean activations
    mean_activations = np.divide(
        total_activations,
        counts,
        where=counts > 0,
        out=np.zeros_like(total_activations)
    )

    # Find cell with max activation per bin
    max_activation_per_bin = np.max(mean_activations, axis=2)
    cell_with_max = np.argmax(mean_activations, axis=2)

    # Normalize
    max_val = np.max(max_activation_per_bin)
    if max_val > 0:
        normalized = max_activation_per_bin / max_val
    else:
        normalized = max_activation_per_bin

    # Generate colors
    colors = generate_random_colors(num_to_plot)

    # Create RGB image
    image = np.zeros((gridsize, gridsize, 3))
    for i in range(gridsize):
        for j in range(gridsize):
            if normalized[i, j] > 0:
                cell_idx = cell_with_max[i, j]
                image[i, j, :] = normalized[i, j] * np.array(colors[cell_idx])

    # Transpose for correct orientation
    image = np.transpose(image, (1, 0, 2))

    return image, (xmin, xmax, ymin, ymax)


def visualize_environment(env_name):
    """Generate place field visualizations for one environment."""

    print(f"\n{'='*60}")
    print(f"Processing {env_name}")
    print(f"{'='*60}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Discover available scales dynamically from files.
    available_scales = discover_available_scales(env_name)

    if not available_scales:
        print(f"  ERROR: No data found for {env_name}")
        return

    unified_detected = is_unified_run(env_name)
    if ARCHITECTURE == "unified":
        show_unified = True
    elif ARCHITECTURE == "separate":
        show_unified = False
    else:
        show_unified = unified_detected

    print(f"  Found data for scales: {available_scales}")
    print(f"  Unified detected: {unified_detected}")
    print(f"  Visualization mode: {'unified+per-scale' if show_unified else 'per-scale'}")

    num_panels = len(available_scales) + (1 if show_unified else 0)
    fig, axes = plt.subplots(1, num_panels, figsize=(8 * num_panels, 8))
    if num_panels == 1:
        axes = [axes]

    fig.suptitle(f"{env_name} - Place Cell Fields", fontsize=16)

    panel_offset = 0
    if show_unified:
        print("\n  Unified aggregate:")
        hmap_loc_u, hmap_pcn_u = load_unified_aggregate_data(env_name, available_scales)
        result_u = None
        if hmap_loc_u is not None and hmap_pcn_u is not None:
            result_u = plot_place_cells_overlay(env_name, "unified", hmap_loc_u, hmap_pcn_u)

        ax_u = axes[0]
        if result_u is not None:
            image_u, extent_u = result_u
            ax_u.imshow(image_u, extent=extent_u, origin="lower")
            ax_u.set_xlabel("X Position (m)", fontsize=12)
            ax_u.set_ylabel("Y Position (m)", fontsize=12)
            ax_u.set_title("Unified (All Scales)", fontsize=14)
            ax_u.grid(True, alpha=0.3)
        else:
            ax_u.text(0.5, 0.5, "No unified data",
                      ha='center', va='center', transform=ax_u.transAxes)
            ax_u.set_title("Unified (no data)", fontsize=14)
        panel_offset = 1

    # Plot each scale
    for ax, scale in zip(axes[panel_offset:], available_scales):
        print(f"\n  Scale {scale}:")

        # Load data
        hmap_loc, hmap_pcn = load_hmap_data(env_name, scale)

        # Create overlay
        result = plot_place_cells_overlay(env_name, scale, hmap_loc, hmap_pcn)

        if result is not None:
            image, extent = result
            ax.imshow(image, extent=extent, origin="lower")
            ax.set_xlabel("X Position (m)", fontsize=12)
            ax.set_ylabel("Y Position (m)", fontsize=12)
            ax.set_title(f"Scale {scale}", fontsize=14)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No active cells",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Scale {scale} (no data)", fontsize=14)

    # Save figure
    output_file = OUTPUT_DIR / f"{env_name}_place_fields.png"
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')

    # Verify file was saved
    if output_file.exists():
        file_size = output_file.stat().st_size / 1024  # KB
        print(f"\n  [OK] Saved: {output_file}")
        print(f"       File size: {file_size:.1f} KB")
    else:
        print(f"\n  [ERROR] Failed to save {output_file}")

    plt.close()


def main():
    """Main execution."""
    print("="*60)
    print("Place Field Visualization")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Controller: {CONTROLLER_NAME}")
    print(f"Environments: {len(ENVIRONMENTS)}")
    print(f"Architecture mode: {ARCHITECTURE}")

    # Process each environment
    for env_name in ENVIRONMENTS:
        visualize_environment(env_name)

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print("="*60)

    # List all created files
    if OUTPUT_DIR.exists():
        png_files = list(OUTPUT_DIR.glob("*.png"))
        if png_files:
            print("\nCreated files:")
            for f in sorted(png_files):
                print(f"  - {f.name}")
        else:
            print("\nWARNING: No PNG files found in output directory!")
    else:
        print(f"\nERROR: Output directory does not exist: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
