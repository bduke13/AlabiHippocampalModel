# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from typing import Optional, List

# Hardcoded world name
WORLD_NAME = "20x20"

def load_pickle(file_path):
    """Load a pickle file from the given path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_hmaps_from_world(num_scales: int):
    """Load hmap data based on the hardcoded world name, supporting multiple scales."""
    base_path = f"webots/controllers/create3_base/pkl/{WORLD_NAME}/hmaps"
    
    # Load location map
    hmap_loc = load_pickle(os.path.join(base_path, "hmap_loc.pkl"))

    # Load multiple scales of PCN activations
    hmap_pcn_scales = []
    for i in range(num_scales):
        pcn_path = os.path.join(base_path, f"hmap_pcn_scale_{i}.pkl")
        if os.path.exists(pcn_path):
            hmap_pcn_scales.append(load_pickle(pcn_path))
        else:
            print(f"Warning: Missing PCN activation file for scale {i}")

    return hmap_loc, hmap_pcn_scales

def convert_xzy_hmaps(hmap_loc):
    """Convert hmap location array to separate x, y, z components."""
    hmap_x, hmap_z, hmap_y = hmap_loc[:, 0], hmap_loc[:, 1], hmap_loc[:, 2]
    return hmap_x, hmap_z, hmap_y

def generate_random_colors(num_colors):
    """Generate an array of random vibrant colors in RGB format."""
    return np.random.rand(num_colors, 3)

def plot_overlayed_cells(
    hmap_pcn_scales: List[np.ndarray],
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    gridsize: int = 100,
    num_cells_to_sample: Optional[int] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None,
    scale_idx: Optional[int] = None,
    cell_idx: Optional[int] = None,  # Plot a specific place cell
    plot_most_active: bool = False,  # New: Plot the most active place cell
):
    """
    Plots place cell activations with different scales.

    Args:
        hmap_pcn_scales: List of activation arrays, one per scale.
        hmap_x: X coordinates.
        hmap_y: Y coordinates.
        gridsize: Grid resolution.
        num_cells_to_sample: Number of cells to randomly sample (ignored if cell_idx is provided).
        show_plot: Whether to display the plot.
        save_path: Optional path to save the plot.
        scale_idx: Index of the scale to plot. If None, plots all scales.
        cell_idx: Index of a specific place cell to plot (if provided, overrides `num_cells_to_sample`).
        plot_most_active: If True, plots the place cell with the highest total activation.
    """

    if scale_idx is None:
        scales_to_plot = range(len(hmap_pcn_scales))  # Plot all scales
    else:
        scales_to_plot = [scale_idx]  # Plot a single scale

    for scale in scales_to_plot:
        hmap_pcn = hmap_pcn_scales[scale]

        # Calculate total activation per cell
        total_activation_per_cell = np.sum(hmap_pcn, axis=0)

        # Get indices of cells with non-zero activation
        nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

        # Determine which cell(s) to plot
        if plot_most_active:
            if len(nonzero_activation_indices) == 0:
                print(f"Warning: No active place cells in scale {scale}.")
                continue
            cell_idx = nonzero_activation_indices[np.argmax(total_activation_per_cell[nonzero_activation_indices])]
            print(f"Plotting most active place cell {cell_idx} for Scale {scale}")
            cell_indices = np.array([cell_idx])

        elif cell_idx is not None:
            # Ensure cell index is valid
            if cell_idx not in nonzero_activation_indices:
                print(f"Warning: Specified place cell {cell_idx} has no activation in scale {scale}.")
                continue
            cell_indices = np.array([cell_idx])  # Only plot this cell
            print(f"Plotting Place Cell {cell_idx} for Scale {scale}")

        else:
            # Default behavior: randomly select active cells
            if num_cells_to_sample is None:
                num_cells_to_plot = len(nonzero_activation_indices)
            else:
                num_cells_to_plot = min(num_cells_to_sample, len(nonzero_activation_indices))
            print(f"Plotting {num_cells_to_plot} cells for scale {scale}")
            cell_indices = np.random.choice(nonzero_activation_indices, size=num_cells_to_plot, replace=False)

        # Define grid boundaries
        xmin, xmax = np.min(hmap_x), np.max(hmap_x)
        ymin, ymax = np.min(hmap_y), np.max(hmap_y)

        # Create grid edges
        xedges = np.linspace(xmin, xmax, gridsize + 1)
        yedges = np.linspace(ymin, ymax, gridsize + 1)

        # Initialize activation storage
        total_activations_per_bin = np.zeros((gridsize, gridsize, len(cell_indices)))
        counts_per_bin = np.zeros((gridsize, gridsize, len(cell_indices)))

        # Process each selected place cell
        for idx, cell_index in enumerate(cell_indices):
            activations = hmap_pcn[:, cell_index]

            # Find where activation is nonzero
            mask = activations > 0
            if not np.any(mask):
                continue
            x, y, a = hmap_x[mask], hmap_y[mask], activations[mask]

            # Bin positions
            ix = np.digitize(x, xedges) - 1
            iy = np.digitize(y, yedges) - 1

            # Clip indices to valid range
            ix = np.clip(ix, 0, gridsize - 1)
            iy = np.clip(iy, 0, gridsize - 1)

            # Accumulate activations per bin
            for i, j, activation in zip(ix, iy, a):
                total_activations_per_bin[i, j, idx] += activation
                counts_per_bin[i, j, idx] += 1

        # Compute mean activation per bin
        mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
        nonzero_counts = counts_per_bin > 0
        mean_activation_per_bin[nonzero_counts] = (
            total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
        )

        # Find max activation per bin
        max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
        cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)

        # Normalize activation for visualization
        max_activation = np.max(max_mean_activation_per_bin)
        max_activation = max_activation if max_activation > 0 else 1  # Avoid division by zero
        normalized_activation = max_mean_activation_per_bin / max_activation

        # Generate colors
        colors_rgb = generate_random_colors(len(cell_indices))

        # Create color image
        image = np.zeros((gridsize, gridsize, 3))

        # Assign colors based on activations
        for i in range(gridsize):
            for j in range(gridsize):
                activation_level = normalized_activation[i, j]
                if activation_level > 0:
                    idx = cell_with_max_activation[i, j]
                    adjusted_color = activation_level * np.array(colors_rgb[idx])
                    image[i, j, :] = adjusted_color

        # Transpose for imshow
        image = np.transpose(image, (1, 0, 2))

        # Plot results
        plt.figure(figsize=(8, 8))
        extent = [xmin, xmax, ymin, ymax]
        plt.imshow(image, extent=extent, origin="lower")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(
            f"Overlay of {len(cell_indices)} Place Cell(s) (Scale {scale})"
            + (f" - Most Active: {cell_idx}" if plot_most_active else "")
        )

        # Save plot
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # Ensure save_dir is not an empty string
                os.makedirs(save_dir, exist_ok=True)
            suffix = f"_scale_{scale}_most_active" if plot_most_active else f"_scale_{scale}"
            plt.savefig(save_path.replace(".png", f"{suffix}.png"), dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path.replace('.png', f'{suffix}.png')}")

        if show_plot:
            plt.show()
        else:
            plt.close()


# Main execution
if __name__ == "__main__":
    num_scales = 3  # Adjust based on how many scales you have
    hmap_loc, hmap_pcn_scales = load_hmaps_from_world(num_scales)
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    plot_overlayed_cells(
    hmap_pcn_scales=hmap_pcn_scales,
    hmap_x=hmap_x,
    hmap_y=hmap_y,
    gridsize=100,
    num_cells_to_sample=10,
    show_plot=True,
    save_path="specific_cell.png",
    scale_idx=0,
    plot_most_active=False,  
    )

