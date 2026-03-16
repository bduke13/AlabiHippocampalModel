"""Minimal verification plotting entrypoint for a single multi_grid_simple run.

Usage:
    python plot_run.py <run_id>

Produces under visualizations/<run_id>/:
    trajectory.png  - 2D path of the agent (X vs Z plane)
    place_cells.png - overlayed place-field firing map

Artifact-contract assumptions:
    - runs/<run_id>/hmaps/hmap_loc.pkl:
        numpy array, shape (N, 3), Webots XYZ layout:
            col 0 = X
            col 1 = Y
            col 2 = Z
        Source: driver.update_hmaps() stores robot_position() output unchanged.
    - runs/<run_id>/hmaps/hmap_pcn.pkl:
        numpy array, shape (N, num_place_cells), float32.
        Already converted from torch via .cpu().numpy() in save_driver_state().
    - runs/<run_id>/config.json: optional, used for plot title only.
    - visualizations/<run_id>/ is created by initialize_run_artifacts(); this
        module also creates it if missing so plotting can run post-hoc.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from run_summary import update_run_summary
except ImportError:
    from .run_summary import update_run_summary

CONTROLLER_DIR = Path(__file__).resolve().parent
RUNS_DIR = CONTROLLER_DIR / "runs"
VIS_ROOT = CONTROLLER_DIR / "visualizations"


def _load_pkl(path: Path) -> np.ndarray:
    with open(path, "rb") as input_file:
        return np.array(pickle.load(input_file))


def load_run_artifacts(run_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (hmap_loc, hmap_pcn, config) for a saved run."""
    run_dir = RUNS_DIR / run_id
    hmap_loc_path = run_dir / "hmaps" / "hmap_loc.pkl"
    hmap_pcn_path = run_dir / "hmaps" / "hmap_pcn.pkl"
    config_path = run_dir / "config.json"

    for path in (hmap_loc_path, hmap_pcn_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                "Run the simulation with include_hmaps=True before plotting."
            )

    hmap_loc = _load_pkl(hmap_loc_path)
    hmap_pcn = _load_pkl(hmap_pcn_path)

    config: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as input_file:
            config = json.load(input_file)

    return hmap_loc, hmap_pcn, config


def plot_trajectory(
    hmap_loc: np.ndarray,
    *,
    run_id: str,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot the 2D agent trajectory in the X-Z plane."""
    x = hmap_loc[:, 0]
    z = hmap_loc[:, 2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, z, color="steelblue", linewidth=0.8, alpha=0.8, label="path")
    ax.plot(x[0], z[0], "go", markersize=7, label="start")
    ax.plot(x[-1], z[-1], "rs", markersize=7, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(title or f"Trajectory - {run_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory plot: {out_path}")


def _random_vibrant_colors(n: int, rng: np.random.Generator) -> np.ndarray:
    colors = np.zeros((n, 3))
    for index in range(n):
        while True:
            color = rng.random(3)
            color[rng.integers(3)] = rng.uniform(0.8, 1.0)
            if color.sum() > 1.2:
                colors[index] = color
                break
    return colors


def plot_place_cells(
    hmap_loc: np.ndarray,
    hmap_pcn: np.ndarray,
    *,
    run_id: str,
    out_path: Path,
    gridsize: int = 80,
    num_cells_to_sample: Optional[int] = 30,
    title: Optional[str] = None,
    seed: int = 42,
) -> None:
    """Render a simple overlayed place-field map."""
    rng = np.random.default_rng(seed)

    x = hmap_loc[:, 0]
    z = hmap_loc[:, 2]

    total_per_cell = hmap_pcn.sum(axis=0)
    active_indices = np.where(total_per_cell > 0)[0]

    if len(active_indices) == 0:
        print(f"[plot_place_cells] No active place cells found in run {run_id}; skipping.")
        return

    n_plot = len(active_indices) if num_cells_to_sample is None else min(num_cells_to_sample, len(active_indices))
    cell_indices = rng.choice(active_indices, size=n_plot, replace=False)
    print(f"[plot_place_cells] Rendering {n_plot}/{len(active_indices)} active cells.")

    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()

    if xmax == xmin:
        xmin -= 0.5
        xmax += 0.5
    if zmax == zmin:
        zmin -= 0.5
        zmax += 0.5

    xedges = np.linspace(xmin, xmax, gridsize + 1)
    zedges = np.linspace(zmin, zmax, gridsize + 1)

    total_act = np.zeros((gridsize, gridsize, n_plot))
    counts = np.zeros((gridsize, gridsize, n_plot))

    for render_index, cell_index in enumerate(cell_indices):
        activations = hmap_pcn[:, cell_index]
        mask = activations > 0
        if not mask.any():
            continue
        xi = np.clip(np.digitize(x[mask], xedges) - 1, 0, gridsize - 1)
        zi = np.clip(np.digitize(z[mask], zedges) - 1, 0, gridsize - 1)
        for bin_x, bin_z, activation in zip(xi, zi, activations[mask]):
            total_act[bin_x, bin_z, render_index] += activation
            counts[bin_x, bin_z, render_index] += 1

    mean_act = np.where(counts > 0, total_act / np.maximum(counts, 1), 0.0)
    max_per_bin = mean_act.max(axis=2)
    best_cell = mean_act.argmax(axis=2)

    global_max = max_per_bin.max()
    if global_max == 0:
        print("[plot_place_cells] All activations zero after binning; skipping.")
        return
    norm = max_per_bin / global_max

    colors = _random_vibrant_colors(n_plot, rng)
    image = np.zeros((gridsize, gridsize, 3))
    for bin_x in range(gridsize):
        for bin_z in range(gridsize):
            level = norm[bin_x, bin_z]
            if level > 0:
                image[bin_x, bin_z] = level * colors[best_cell[bin_x, bin_z]]

    image = np.transpose(image, (1, 0, 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=[xmin, xmax, zmin, zmax], origin="lower", aspect="equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title or f"Place cells ({n_plot} sampled) - {run_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved place-cell plot: {out_path}")


def run(
    run_id: str,
    *,
    gridsize: int = 80,
    num_cells_to_sample: Optional[int] = 30,
    seed: int = 42,
) -> Path:
    """Generate all current verification plots for a run and return the output directory."""
    hmap_loc, hmap_pcn, config = load_run_artifacts(run_id)

    vis_dir = VIS_ROOT / run_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    world = config.get("world_name", "")
    mode = config.get("mode", "")
    base_title = f"{run_id}" + (f"  [{world} / {mode}]" if world or mode else "")

    plot_trajectory(
        hmap_loc,
        run_id=run_id,
        out_path=vis_dir / "trajectory.png",
        title=f"Trajectory - {base_title}",
    )
    plot_place_cells(
        hmap_loc,
        hmap_pcn,
        run_id=run_id,
        out_path=vis_dir / "place_cells.png",
        gridsize=gridsize,
        num_cells_to_sample=num_cells_to_sample,
        title=f"Place cells - {base_title}",
        seed=seed,
    )

    update_run_summary(RUNS_DIR / run_id, visualization_dir=vis_dir)

    return vis_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce trajectory and place-cell plots for a completed multi_grid_simple run."
    )
    parser.add_argument("run_id", help="Run ID (subdirectory name under runs/)")
    parser.add_argument(
        "--cells",
        type=int,
        default=30,
        metavar="N",
        help="Max place cells to sample for the firing-map plot (default: 30). Pass 0 for all active cells.",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=80,
        metavar="G",
        help="Grid resolution for place-cell binning (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible cell colors (default: 42).",
    )
    args = parser.parse_args()

    num_cells = None if args.cells == 0 else args.cells
    vis_dir = run(
        args.run_id,
        gridsize=args.gridsize,
        num_cells_to_sample=num_cells,
        seed=args.seed,
    )
    print(f"\nDone. Outputs in: {vis_dir}")


if __name__ == "__main__":
    main()
