import os
import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


visualization_root = Path(__file__).resolve().parent
project_root = visualization_root.parent
sys.path.append(str(visualization_root))
sys.path.append(str(project_root))

from vis_utils import convert_xzy_hmaps, get_output_dir, resolve_data_context


def _load_pickle_array(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.array(pickle.load(f))
    return data[1:]


def _discover_gcn_scales(
    controller_name: Optional[str] = None,
    world_name: Optional[str] = None,
) -> List[int]:
    _, _, data_root = resolve_data_context(
        controller_name=controller_name,
        world_name=world_name,
        required_relpaths=["hmaps/hmap_loc.pkl"],
    )
    hmap_dir = data_root / "hmaps"
    scales = set()
    for path in hmap_dir.glob("hmap_gcn_scale_*.pkl"):
        match = re.match(r"hmap_gcn_scale_(\d+)\.pkl$", path.name)
        if match:
            scales.add(int(match.group(1)))
    return sorted(scales)


def load_grid_cell_data(
    scale: int,
    controller_name: Optional[str] = None,
    world_name: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load location and grid-cell history for one scale from the resolved dataset.
    """
    try:
        _, _, data_root = resolve_data_context(
            controller_name=controller_name,
            world_name=world_name,
            required_relpaths=[
                "hmaps/hmap_loc.pkl",
                f"hmaps/hmap_gcn_scale_{scale}.pkl",
            ],
        )
    except FileNotFoundError as exc:
        print(f"Error resolving data for scale {scale}: {exc}")
        return None, None

    hmap_dir = data_root / "hmaps"
    hmap_loc_path = hmap_dir / "hmap_loc.pkl"
    hmap_gcn_path = hmap_dir / f"hmap_gcn_scale_{scale}.pkl"

    try:
        hmap_loc = _load_pickle_array(hmap_loc_path)
        hmap_gcn = _load_pickle_array(hmap_gcn_path)
    except Exception as exc:
        print(f"Error loading grid-cell data for scale {scale}: {exc}")
        return None, None

    print(f"Loaded hmap_loc from {hmap_loc_path}")
    print(f"Loaded hmap_gcn_scale_{scale} from {hmap_gcn_path}")
    return hmap_loc, hmap_gcn


def plot_average_grid_activation(
    hmap_x,
    hmap_y,
    hmap_data,
    scale,
    ax=None,
    cmap="viridis",
):
    """
    Plot the average activation across all grid cells for one scale.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    average_activations = np.mean(hmap_data, axis=1)
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=average_activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap=cmap,
        edgecolors="none",
    )

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Average Activation")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scale {scale}: Average Activation Across All Grid Cells")
    return ax


def plot_single_grid_cell(
    hmap_x,
    hmap_y,
    hmap_data,
    cell_index,
    scale,
    ax=None,
    cmap="viridis",
):
    """
    Plot one selected grid cell for one scale.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    cell_activations = hmap_data[:, cell_index]
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=cell_activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap=cmap,
        edgecolors="none",
    )

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Activation")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scale {scale}: Grid Cell {cell_index}")
    return ax


def find_best_grid_cell(hmap_gcn):
    """
    Find a grid cell with strong activation patterns.
    """
    total_activations = np.sum(np.abs(hmap_gcn), axis=0)
    active_cells = np.where(total_activations > 0)[0]

    if len(active_cells) == 0:
        return 0

    top_cells = active_cells[np.argsort(-total_activations[active_cells])[:10]]
    return int(np.random.choice(top_cells))


def plot_multi_scale_grid_activations(
    scales: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    cmap: str = "viridis",
    controller_name: Optional[str] = None,
    world_name: Optional[str] = None,
):
    """
    Create a figure with grid-cell visualizations for the available scales.
    """
    if scales is None:
        scales = _discover_gcn_scales(controller_name=controller_name, world_name=world_name)
    if not scales:
        raise FileNotFoundError("No hmap_gcn_scale_*.pkl files were found for the resolved dataset.")

    fig = plt.figure(figsize=(6 * len(scales), 12))
    gs = gridspec.GridSpec(2, len(scales))

    for i, scale in enumerate(scales):
        print(f"\n==== Processing Scale {scale} ====")
        hmap_loc, hmap_gcn = load_grid_cell_data(
            scale=scale,
            controller_name=controller_name,
            world_name=world_name,
        )
        if hmap_loc is None or hmap_gcn is None:
            print(f"Skipping scale {scale} due to missing data")
            continue

        hmap_x, _hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        cell_index = find_best_grid_cell(hmap_gcn)

        ax_single = plt.subplot(gs[0, i])
        plot_single_grid_cell(
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_data=hmap_gcn,
            cell_index=cell_index,
            scale=scale,
            ax=ax_single,
            cmap=cmap,
        )

        ax_avg = plt.subplot(gs[1, i])
        plot_average_grid_activation(
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_data=hmap_gcn,
            scale=scale,
            ax=ax_avg,
            cmap=cmap,
        )

    plt.suptitle("Grid Cell Activations Across Scales", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    print("Starting multi-scale grid cell activation visualization...")
    np.random.seed(42)

    output_dir = os.path.join(get_output_dir(), "grid_cells")
    os.makedirs(output_dir, exist_ok=True)

    plot_multi_scale_grid_activations(
        save_path=os.path.join(output_dir, "grid_cell_activations_multi_scale.png"),
        show_plot=True,
        cmap="viridis",
    )

    print("Visualization complete!")
