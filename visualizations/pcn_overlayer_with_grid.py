import numpy as np
import os
import matplotlib.pyplot as plt
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def plot_overlayed_cells_with_grid(
    hmap_pcn,
    hmap_grid,
    hmap_x,
    hmap_y,
    gridsize=50,
    num_cells_to_sample=None,
    show_plot=True,
    save_path=None,
):
    """
    Plot side-by-side hexbin plots for grid and place cell activations.

    Args:
        hmap_pcn: Place cell activations.
        hmap_grid: Grid cell activations.
        hmap_x: X coordinates.
        hmap_y: Y coordinates.
        gridsize: Grid size for hexbin plot.
        num_cells_to_sample: Number of cells to sample (optional).
        show_plot: Boolean to display the plot.
        save_path: Path to save the plot (optional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot grid cells
    ax1.hexbin(
        hmap_x,
        hmap_y,
        C=np.sum(hmap_grid, axis=1),
        gridsize=gridsize,
        cmap="viridis",
        label="Grid Cells"
    )
    ax1.set_title("Grid Cells")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    # Plot place cells
    ax2.hexbin(
        hmap_x,
        hmap_y,
        C=np.sum(hmap_pcn, axis=1),
        gridsize=gridsize,
        cmap="plasma",
        label="Place Cells"
    )
    ax2.set_title("Place Cells")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved overlay plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    hmap_loc, hmap_pcn, hmap_grid = load_hmaps(["hmap_loc", "hmap_pcn", "hmap_gc"])
    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)
    plot_overlayed_cells_with_grid(
        hmap_pcn,
        hmap_grid,
        hmap_x,
        hmap_y,
        save_path=os.path.join(OUTPUT_DIR, "overlay_with_grid.png")
    )