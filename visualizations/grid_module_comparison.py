import numpy as np
import os
import matplotlib.pyplot as plt
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def plot_grid_modules(hmap_x, hmap_y, hmap_grid, num_modules=3, output_dir=OUTPUT_DIR):
    """Plot activations for different grid cell modules."""
    module_size = hmap_grid.shape[1] // num_modules  # Assuming equal-sized modules
    fig, axes = plt.subplots(1, num_modules, figsize=(5 * num_modules, 5))

    for i in range(num_modules):
        module_data = hmap_grid[:, i * module_size:(i + 1) * module_size]
        axes[i].hexbin(
            hmap_x,
            hmap_y,
            C=np.sum(module_data, axis=1),
            gridsize=50,
            cmap="viridis"
        )
        axes[i].set_title(f"Grid Module {i + 1}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_modules.png"))
    plt.show()

if __name__ == "__main__":
    hmap_loc, hmap_grid = load_hmaps(["hmap_loc", "hmap_grid"])
    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)
    plot_grid_modules(hmap_x, hmap_y, hmap_grid)