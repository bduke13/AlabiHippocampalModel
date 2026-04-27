"""Verification plotting entrypoint for a single multi_grid_simple run.

Usage:
    python plot_run.py <run_id>

Produces under visualizations/<run_id>/:
    trajectory.png          - 2D path of the agent (X vs Z plane)
    place_cells.png         - overlayed place-field firing map
    grid_cells.png          - runtime or fallback grid-cell hexbin plots
    aliasing_heatmap.png    - spatial aliasing index heatmap
    aliasing_stats.json     - aliasing summary and parameters

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
    - runs/<run_id>/hmaps/hmap_gcn.pkl: optional runtime grid-cell activations,
        shape (N, num_grid_cells), float32. Older runs may not have this file.
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
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except Exception:
    make_axes_locatable = None

try:
    from scipy.interpolate import griddata as _griddata
except Exception:
    _griddata = None

try:
    from scipy.stats import qmc as _qmc
except Exception:
    _qmc = None

try:
    from layers.gcn import GridCellLayer
except Exception:
    try:
        from .layers.gcn import GridCellLayer
    except Exception:
        GridCellLayer = None

try:
    from robot.webots_worlds import get_world_config
except Exception:
    try:
        from .robot.webots_worlds import get_world_config
    except Exception:
        get_world_config = None

try:
    from run_summary import update_run_summary
except ImportError:
    from .run_summary import update_run_summary

CONTROLLER_DIR = Path(__file__).resolve().parent
RUNS_DIR = CONTROLLER_DIR / "runs"
VIS_ROOT = CONTROLLER_DIR / "visualizations"

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _load_pkl(path: Path) -> np.ndarray:
    with open(path, "rb") as input_file:
        return np.array(pickle.load(input_file))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_run_artifacts(run_id: str) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Return (hmap_loc, hmap_pcn, hmap_gcn, config) for a saved run."""
    run_dir = RUNS_DIR / run_id
    hmap_loc_path = run_dir / "hmaps" / "hmap_loc.pkl"
    hmap_pcn_path = run_dir / "hmaps" / "hmap_pcn.pkl"
    hmap_gcn_path = run_dir / "hmaps" / "hmap_gcn.pkl"
    config_path = run_dir / "config.json"

    for path in (hmap_loc_path, hmap_pcn_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                "Run the simulation with include_hmaps=True before plotting."
            )

    hmap_loc = _load_pkl(hmap_loc_path)
    hmap_pcn = _load_pkl(hmap_pcn_path)
    hmap_gcn = _load_pkl(hmap_gcn_path) if hmap_gcn_path.exists() else None

    config: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as input_file:
            config = json.load(input_file)

    return hmap_loc, hmap_pcn, hmap_gcn, config


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

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, z, color="#2a6f97", linewidth=1.2, alpha=0.9, label="Path")
    ax.plot(x[0], z[0], "o", color="#2e8b57", markersize=7, label="Start")
    ax.plot(x[-1], z[-1], "s", color="#b22222", markersize=7, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    ax.set_title(title or f"Trajectory - {run_id}")
    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"Saved trajectory plot: {out_path}")


def _random_vibrant_colors(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rng.shuffle(hues)
    saturations = rng.uniform(0.82, 0.98, size=n)
    values = rng.uniform(0.92, 1.0, size=n)
    hsv = np.column_stack((hues, saturations, values))
    return mcolors.hsv_to_rgb(hsv)


def _overlay_world_obstacles(ax: plt.Axes, config: dict) -> None:
    if get_world_config is None:
        return
    world_name = config.get("world_name")
    if not world_name:
        return
    try:
        world = get_world_config(world_name)
    except Exception:
        return
    for obstacle in world.get("obstacles", []):
        if obstacle.get("type") != "rectangle":
            continue
        (x1, z1), (x2, z2) = obstacle["bounds"]
        xmin, xmax = sorted((float(x1), float(x2)))
        zmin, zmax = sorted((float(z1), float(z2)))
        rect = mpatches.Rectangle(
            (xmin, zmin),
            xmax - xmin,
            zmax - zmin,
            facecolor="#f2f2f2",
            edgecolor="#111111",
            linewidth=1.0,
            alpha=0.7,
            zorder=6,
        )
        ax.add_patch(rect)


def _grid_runtime_params_from_config(config: dict) -> dict:
    runtime_params = dict(config.get("runtime_parameters") or {})
    return {
        "world_name": config.get("world_name"),
        "num_modules": int(runtime_params.get("num_grid_modules", 8)),
        "cells_per_module": int(runtime_params.get("num_grid_cells_per_module", 50)),
        "spread_range": tuple(runtime_params.get("grid_spread_range", (1.2, 1.2))),
        "scale_multiplier": float(runtime_params.get("grid_scale_multiplier", 1.0)),
        "translation_scale": float(runtime_params.get("grid_translation_scale", 1.0)),
        "threshold": float(runtime_params.get("grid_threshold", 0.7)),
        "mask_resolution": int(runtime_params.get("grid_mask_resolution", 128)),
        "smooth_sigma": float(runtime_params.get("grid_smooth_sigma", 1.5)),
    }


def _build_grid_layer_from_config(config: dict) -> Optional[GridCellLayer]:
    if GridCellLayer is None:
        return None
    params = _grid_runtime_params_from_config(config)
    world_name = params.get("world_name")
    if not world_name:
        return None
    return GridCellLayer(
        num_modules=params["num_modules"],
        cells_per_module=params["cells_per_module"],
        spread_range=params["spread_range"],
        scale_multiplier=params["scale_multiplier"],
        translation_scale=params["translation_scale"],
        threshold=params["threshold"],
        threshold_type="soft",
        normalization="per-cell",
        world_name=world_name,
        mask_resolution=params["mask_resolution"],
        smooth_sigma=params["smooth_sigma"],
        device="cpu",
    )


def _apply_grid_mask_to_history(
    hmap_loc: np.ndarray,
    hmap_gcn: np.ndarray,
    config: dict,
) -> np.ndarray:
    gcn = _build_grid_layer_from_config(config)
    if gcn is None or not hasattr(gcn, "_mask") or gcn._mask is None:
        return hmap_gcn

    x = hmap_loc[:, 0].astype(np.float64, copy=False)
    z = hmap_loc[:, 2].astype(np.float64, copy=False)
    xi = np.rint((x + gcn.world_w / 2.0) * gcn._x_scale).astype(np.int64)
    yi = np.rint((z + gcn.world_h / 2.0) * gcn._y_scale).astype(np.int64)
    xi = np.clip(xi, 0, gcn.mask_resolution - 1)
    yi = np.clip(yi, 0, gcn.mask_resolution - 1)

    mask_lookup = gcn._mask.detach().cpu().numpy()[yi, xi, :]
    if mask_lookup.shape != hmap_gcn.shape:
        return hmap_gcn
    return hmap_gcn * mask_lookup


def _reference_grid_cell_parameters(
    *,
    num_modules: int,
    cells_per_module: int,
    spread_range: tuple[float, float],
    scale_multiplier: float,
    translation_scale: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    total_grid_cells = num_modules * cells_per_module

    rotations = np.repeat(
        np.linspace(0.0, 360.0, num_modules, endpoint=False),
        cells_per_module,
    )
    size_params = np.full(total_grid_cells, float(scale_multiplier), dtype=np.float64)
    spread_per_module = rng.uniform(spread_range[0], spread_range[1], size=num_modules)
    spread_params = np.repeat(spread_per_module, cells_per_module)

    x_trans = []
    z_trans = []
    base_size = 1.0
    K = np.array(
        [
            [base_size, 0.0],
            [base_size / 2.0, (np.sqrt(3.0) * base_size) / 2.0],
        ],
        dtype=np.float64,
    )
    K_inv = np.linalg.inv(K)

    for module_index in range(num_modules):
        theta_deg = float(rotations[module_index * cells_per_module])
        theta = np.deg2rad(theta_deg)
        if _qmc is not None:
            sobol = _qmc.Sobol(d=2, scramble=True, seed=seed + module_index)
            power = int(np.ceil(np.log2(cells_per_module)))
            u = sobol.random_base2(power)[:cells_per_module]
        else:
            u = rng.random((cells_per_module, 2))
        phi = 2.0 * np.pi * u
        t_rot = (K_inv @ phi.T).T
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        t_world = (rotation @ t_rot.T).T
        x_trans.append(t_world[:, 0])
        z_trans.append(t_world[:, 1])

    return {
        "rotations_rad": np.deg2rad(rotations),
        "size_params": size_params,
        "spread_params": spread_params,
        "x_trans": np.concatenate(x_trans) * float(translation_scale),
        "z_trans": np.concatenate(z_trans) * float(translation_scale),
    }


def _reference_grid_trajectory_activations(
    hmap_loc: np.ndarray,
    *,
    num_modules: int,
    cells_per_module: int,
    spread_range: tuple[float, float],
    scale_multiplier: float,
    translation_scale: float,
    threshold: float,
    seed: int,
) -> np.ndarray:
    x = hmap_loc[:, 0].astype(np.float64, copy=False)
    z = hmap_loc[:, 2].astype(np.float64, copy=False)

    params = _reference_grid_cell_parameters(
        num_modules=num_modules,
        cells_per_module=cells_per_module,
        spread_range=spread_range,
        scale_multiplier=scale_multiplier,
        translation_scale=translation_scale,
        seed=seed,
    )

    dx = x[:, None] - params["x_trans"][None, :]
    dz = z[:, None] - params["z_trans"][None, :]
    cos_theta = np.cos(params["rotations_rad"])[None, :]
    sin_theta = np.sin(params["rotations_rad"])[None, :]

    rx = cos_theta * dx + sin_theta * dz
    rz = -sin_theta * dx + cos_theta * dz

    freq = ((2.0 * np.pi) / params["size_params"])[None, :]
    z1 = np.cos(freq * rx)
    z2 = np.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * rz))
    z3 = np.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * rz))
    raw = (z1 + z2 + z3) / 3.0

    spread = params["spread_params"][None, :]
    acts = np.sign(raw) * np.power(np.abs(raw), 1.0 / spread)

    # Match the layer's effective per-cell normalization path for the default
    # initialization where cell_min/cell_max begin at [-1, 1].
    acts = np.clip((acts + 1.0) / 2.0, 0.0, 1.0)
    acts = np.clip((acts - threshold) / max(1.0 - threshold, 1e-8), 0.0, 1.0)
    return acts.astype(np.float32)


def _render_place_cell_overlay(
    *,
    x: np.ndarray,
    z: np.ndarray,
    hmap_pcn: np.ndarray,
    cell_indices: np.ndarray,
    gridsize: int,
    rng: np.random.Generator,
) -> Optional[tuple[np.ndarray, list[float]]]:
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

    n_plot = len(cell_indices)
    total_act = np.zeros((gridsize, gridsize, n_plot), dtype=np.float64)
    counts = np.zeros((gridsize, gridsize, n_plot), dtype=np.float64)

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

    mean_act = np.where(counts > 0, total_act / np.maximum(counts, 1.0), 0.0)
    max_per_bin = mean_act.max(axis=2)
    global_max = max_per_bin.max()
    if global_max <= 0:
        return None

    best_cell = mean_act.argmax(axis=2)
    norm = max_per_bin / global_max
    colors = _random_vibrant_colors(n_plot, rng)
    image = np.zeros((gridsize, gridsize, 3), dtype=np.float64)
    for bin_x in range(gridsize):
        for bin_z in range(gridsize):
            level = norm[bin_x, bin_z]
            if level > 0:
                image[bin_x, bin_z] = level * colors[best_cell[bin_x, bin_z]]

    return np.transpose(image, (1, 0, 2)), [xmin, xmax, zmin, zmax]


def _find_best_grid_cell(hmap_gcn: np.ndarray) -> int:
    total_activations = np.sum(np.abs(hmap_gcn), axis=0)
    active_cells = np.where(total_activations > 0)[0]
    if len(active_cells) == 0:
        return 0
    return int(active_cells[np.argmax(total_activations[active_cells])])


def _cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    normalized = matrix / safe_norms
    return normalized @ normalized.T


def _compute_chunked_aliasing_values(
    population_vectors: np.ndarray,
    centers: np.ndarray,
    *,
    distance_threshold: float,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aliasing values without materializing full pairwise matrices."""
    num_bins = population_vectors.shape[0]
    aliasing_values = np.zeros(num_bins, dtype=np.float64)
    comparison_counts = np.zeros(num_bins, dtype=np.int32)

    norms = np.linalg.norm(population_vectors, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    normalized_vectors = population_vectors / safe_norms

    centers = centers.astype(np.float64, copy=False)
    centers_squared = np.sum(centers**2, axis=1)

    for start in range(0, num_bins, batch_size):
        end = min(start + batch_size, num_bins)
        batch_vectors = normalized_vectors[start:end]
        similarities = batch_vectors @ normalized_vectors.T

        batch_centers = centers[start:end]
        batch_centers_squared = np.sum(batch_centers**2, axis=1, keepdims=True)
        distance_squared = (
            batch_centers_squared
            + centers_squared[None, :]
            - 2 * (batch_centers @ centers.T)
        )
        distances = np.sqrt(np.clip(distance_squared, 0.0, None))

        far_mask = distances >= distance_threshold
        row_indices = np.arange(start, end)
        far_mask[np.arange(end - start), row_indices] = False

        batch_counts = far_mask.sum(axis=1)
        comparison_counts[start:end] = batch_counts

        comparable = batch_counts > 0
        if np.any(comparable):
            batch_sums = (similarities * far_mask).sum(axis=1)
            aliasing_values[start:end][comparable] = (
                batch_sums[comparable] / batch_counts[comparable]
            )

    return aliasing_values, comparison_counts


def choose_aliasing_gridsize(
    hmap_loc: np.ndarray,
    *,
    target_visits_per_bin: int = 6,
    min_gridsize: int = 30,
    max_gridsize: int = 90,
) -> int:
    """Choose a reasonable square aliasing grid from the amount of data."""
    if hmap_loc.ndim != 2 or hmap_loc.shape[0] == 0:
        return min_gridsize
    target_visits_per_bin = max(target_visits_per_bin, 1)
    estimated_bin_count = max(hmap_loc.shape[0] / target_visits_per_bin, 1.0)
    gridsize = int(round(np.sqrt(estimated_bin_count)))
    return int(np.clip(gridsize, min_gridsize, max_gridsize))


def compute_aliasing_stats(
    hmap_loc: np.ndarray,
    hmap_pcn: np.ndarray,
    *,
    gridsize: Optional[int] = None,
    quantile: float = 0.1,
    min_visits: int = 3,
    distance_fraction: float = 0.25,
    distance_threshold: Optional[float] = None,
    min_distance_bins: int = 3,
) -> dict:
    """Compute a robust spatial aliasing heatmap and summary statistics.

    This follows the same core idea as the legacy MSAI plot: compare population
    vectors at spatial bins and average similarities only across bins that are
    sufficiently far apart. The implementation here adds explicit occupancy
    filtering and a less brittle distance default.
    """
    if hmap_loc.ndim != 2 or hmap_loc.shape[1] < 3:
        raise ValueError("hmap_loc must have shape (N, 3) or compatible XYZ columns.")
    if hmap_pcn.ndim != 2:
        raise ValueError("hmap_pcn must have shape (N, num_place_cells).")
    if hmap_loc.shape[0] != hmap_pcn.shape[0]:
        raise ValueError("hmap_loc and hmap_pcn must have the same number of timesteps.")
    if gridsize is None or gridsize <= 0:
        gridsize = choose_aliasing_gridsize(hmap_loc)
    if gridsize < 4:
        raise ValueError("gridsize must be at least 4.")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must be in [0, 1).")
    if min_visits < 1:
        raise ValueError("min_visits must be at least 1.")
    if distance_fraction <= 0:
        raise ValueError("distance_fraction must be positive.")
    if min_distance_bins < 1:
        raise ValueError("min_distance_bins must be at least 1.")

    x = hmap_loc[:, 0]
    z = hmap_loc[:, 2]

    xmin, xmax = float(x.min()), float(x.max())
    zmin, zmax = float(z.min()), float(z.max())
    if xmax == xmin:
        xmin -= 0.5
        xmax += 0.5
    if zmax == zmin:
        zmin -= 0.5
        zmax += 0.5

    xedges = np.linspace(xmin, xmax, gridsize + 1)
    zedges = np.linspace(zmin, zmax, gridsize + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    zcenters = 0.5 * (zedges[:-1] + zedges[1:])

    xi = np.clip(np.digitize(x, xedges) - 1, 0, gridsize - 1)
    zi = np.clip(np.digitize(z, zedges) - 1, 0, gridsize - 1)

    visits = np.zeros((gridsize, gridsize), dtype=np.int32)
    np.add.at(visits, (xi, zi), 1)

    num_cells = hmap_pcn.shape[1]
    total = np.zeros((gridsize, gridsize, num_cells), dtype=np.float64)
    for cell_index in range(num_cells):
        activations = hmap_pcn[:, cell_index]
        positive_mask = activations > 0
        if not positive_mask.any():
            continue
        np.add.at(
            total[..., cell_index],
            (xi[positive_mask], zi[positive_mask]),
            activations[positive_mask],
        )

    mean_act = np.divide(
        total,
        np.maximum(visits[..., None], 1),
        dtype=np.float64,
    )

    processed = np.zeros_like(mean_act)
    active_cells = 0
    for cell_index in range(num_cells):
        cell_map = mean_act[..., cell_index]
        valid_values = cell_map[visits >= min_visits]
        valid_values = valid_values[valid_values > 0]
        if valid_values.size == 0:
            continue
        threshold = float(np.quantile(valid_values, quantile))
        thresholded = np.where(cell_map >= threshold, cell_map, 0.0)
        cell_max = float(thresholded.max())
        if cell_max <= 0:
            continue
        processed[..., cell_index] = thresholded / cell_max
        active_cells += 1

    occupancy_mask = visits >= min_visits
    activity_mask = processed.sum(axis=2) > 0
    valid_mask = occupancy_mask & activity_mask

    if not np.any(valid_mask):
        return {
            "aliasing_map": np.zeros((gridsize, gridsize), dtype=np.float64),
            "occupancy_mask": occupancy_mask,
            "visits": visits,
            "msai": 0.0,
            "valid_bin_count": 0,
            "active_cell_count": active_cells,
            "distance_threshold": 0.0,
            "resolved_gridsize": int(gridsize),
            "xedges": xedges,
            "zedges": zedges,
            "xcenters": xcenters,
            "zcenters": zcenters,
        }

    valid_indices = np.argwhere(valid_mask)
    population_vectors = processed[valid_mask, :]
    centers_x = xcenters[valid_indices[:, 0]]
    centers_z = zcenters[valid_indices[:, 1]]
    centers = np.column_stack((centers_x, centers_z))

    x_step = float((xmax - xmin) / gridsize)
    z_step = float((zmax - zmin) / gridsize)
    bin_diagonal = float(np.hypot(x_step, z_step))
    span = float(max(xmax - xmin, zmax - zmin))
    if distance_threshold is None:
        distance_threshold = max(distance_fraction * span, min_distance_bins * bin_diagonal)

    aliasing_values, comparison_counts = _compute_chunked_aliasing_values(
        population_vectors,
        centers,
        distance_threshold=float(distance_threshold),
    )
    comparable = comparison_counts > 0

    aliasing_map = np.full((gridsize, gridsize), np.nan, dtype=np.float64)
    aliasing_map[occupancy_mask] = 0.0
    for value, (bin_x, bin_z) in zip(aliasing_values, valid_indices):
        aliasing_map[bin_x, bin_z] = value

    comparable_values = aliasing_values[comparable]
    return {
        "aliasing_map": aliasing_map,
        "occupancy_mask": occupancy_mask,
        "visits": visits,
        "msai": float(comparable_values.mean()) if comparable_values.size else 0.0,
        "valid_bin_count": int(valid_indices.shape[0]),
        "active_cell_count": int(active_cells),
        "distance_threshold": float(distance_threshold),
        "resolved_gridsize": int(gridsize),
        "xedges": xedges,
        "zedges": zedges,
        "xcenters": xcenters,
        "zcenters": zcenters,
    }


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
    """Render sampled and all-cell overlayed place-field maps."""
    rng = np.random.default_rng(seed)

    x = hmap_loc[:, 0]
    z = hmap_loc[:, 2]

    total_per_cell = hmap_pcn.sum(axis=0)
    active_indices = np.where(total_per_cell > 0)[0]

    if len(active_indices) == 0:
        print(f"[plot_place_cells] No active place cells found in run {run_id}; skipping.")
        return

    n_sampled = len(active_indices) if num_cells_to_sample is None else min(num_cells_to_sample, len(active_indices))
    sampled_indices = rng.choice(active_indices, size=n_sampled, replace=False)
    print(
        f"[plot_place_cells] Rendering sampled/all overlays: "
        f"{n_sampled}/{len(active_indices)} sampled, {len(active_indices)} total active."
    )

    sampled_render = _render_place_cell_overlay(
        x=x,
        z=z,
        hmap_pcn=hmap_pcn,
        cell_indices=sampled_indices,
        gridsize=gridsize,
        rng=rng,
    )
    all_render = _render_place_cell_overlay(
        x=x,
        z=z,
        hmap_pcn=hmap_pcn,
        cell_indices=active_indices,
        gridsize=gridsize,
        rng=rng,
    )

    if sampled_render is None and all_render is None:
        print("[plot_place_cells] All activations zero after binning; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, render, panel_title in (
        (axes[0], all_render, f"All {len(active_indices)} active cells"),
        (axes[1], sampled_render, f"{n_sampled} sampled active cells"),
    ):
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.grid(False)
        if render is None:
            ax.text(0.5, 0.5, "No activity", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel_title)
            continue
        image, extent = render
        ax.imshow(image, extent=extent, origin="lower", aspect="equal")
        ax.set_title(panel_title)
    fig.suptitle(title or f"Place cells - {run_id}", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save_figure(fig, out_path)
    print(f"Saved place-cell plot: {out_path}")


def _plot_grid_hexbin(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    z: np.ndarray,
    activations: np.ndarray,
    gridsize: int,
    title: str,
    colorbar_label: str,
) -> None:
    hb = ax.hexbin(
        x,
        z,
        C=activations,
        gridsize=gridsize,
        reduce_C_function=np.mean,
        cmap="viridis",
        edgecolors="none",
        mincnt=1,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_title(title)
    colorbar = plt.colorbar(hb, ax=ax, shrink=0.88)
    colorbar.set_label(colorbar_label)


def plot_grid_cells(
    hmap_loc: np.ndarray,
    *,
    run_id: str,
    hmap_gcn: Optional[np.ndarray],
    config: dict,
    out_path: Path,
    gridsize: int = 100,
    seed: int = 42,
    title: Optional[str] = None,
) -> None:
    """Render legacy-style grid-cell hexbins from runtime or fallback activations.

    Prefer saved runtime `hmap_gcn.pkl` when available. Older runs fall back to
    a deterministic post-hoc reconstruction using the local grid-cell layer
    parameterization so the plotting entrypoint remains backward compatible.
    """
    source_label = "Runtime"
    if hmap_gcn is None:
        params = _grid_runtime_params_from_config(config)
        source_label = "Post-hoc fallback"
        hmap_gcn = _reference_grid_trajectory_activations(
            hmap_loc,
            num_modules=params["num_modules"],
            cells_per_module=params["cells_per_module"],
            spread_range=params["spread_range"],
            scale_multiplier=params["scale_multiplier"],
            translation_scale=params["translation_scale"],
            threshold=params["threshold"],
            seed=seed,
        )

    if hmap_gcn.shape[0] != hmap_loc.shape[0]:
        aligned_steps = min(hmap_gcn.shape[0], hmap_loc.shape[0])
        print(
            f"[plot_grid_cells] Aligning location/history lengths for run {run_id}: "
            f"loc={hmap_loc.shape[0]}, gcn={hmap_gcn.shape[0]}, using {aligned_steps}."
        )
        hmap_loc = hmap_loc[:aligned_steps]
        hmap_gcn = hmap_gcn[:aligned_steps]

    if hmap_gcn.size == 0:
        print(f"[plot_grid_cells] No grid-cell activations available for run {run_id}; skipping.")
        return

    if source_label == "Runtime":
        masked_hmap_gcn = _apply_grid_mask_to_history(hmap_loc, hmap_gcn, config)
        if not np.array_equal(masked_hmap_gcn, hmap_gcn):
            print(f"[plot_grid_cells] Reapplied spatial mask for run {run_id} using saved world/profile config.")
        hmap_gcn = masked_hmap_gcn

    x = hmap_loc[:, 0]
    z = hmap_loc[:, 2]
    cell_index = _find_best_grid_cell(hmap_gcn)
    mean_activations = np.mean(hmap_gcn, axis=1)
    cell_activations = hmap_gcn[:, cell_index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    panel_title = title or f"Grid cells - {run_id}"
    _plot_grid_hexbin(
        axes[0],
        x=x,
        z=z,
        activations=cell_activations,
        gridsize=gridsize,
        title=f"{panel_title}\n{source_label} best cell {cell_index}",
        colorbar_label="Activation",
    )
    _plot_grid_hexbin(
        axes[1],
        x=x,
        z=z,
        activations=mean_activations,
        gridsize=gridsize,
        title=f"{panel_title}\n{source_label} average across all cells",
        colorbar_label="Average activation",
    )

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"Saved grid-cell plot: {out_path}")


def _plot_aliasing_contour(
    ax: plt.Axes,
    *,
    xcenters: np.ndarray,
    zcenters: np.ndarray,
    aliasing_map: np.ndarray,
    cmap: str,
) -> tuple[object, np.ndarray]:
    if _griddata is None:
        raise RuntimeError("SciPy griddata is required for contour-style aliasing plots.")

    XI, ZI = np.meshgrid(
        np.linspace(float(xcenters.min()), float(xcenters.max()), 180),
        np.linspace(float(zcenters.min()), float(zcenters.max()), 180),
        indexing="xy",
    )
    XX, ZZ = np.meshgrid(xcenters, zcenters, indexing="xy")
    points = np.column_stack([XX.ravel(), ZZ.ravel()])
    values = aliasing_map.T.ravel()
    finite_mask = np.isfinite(values)
    points = points[finite_mask]
    values = values[finite_mask]

    interpolated = _griddata(points, values, (XI, ZI), method="cubic", fill_value=0.0)
    levels = np.linspace(float(np.nanmin(interpolated)), float(np.nanmax(interpolated)), 25)
    if np.allclose(levels[0], levels[-1]):
        levels = np.linspace(0.0, max(float(np.nanmax(interpolated)), 1e-6), 25)
    contour = ax.contourf(XI, ZI, interpolated, levels=levels, cmap=cmap)
    ax.contour(XI, ZI, interpolated, levels=10, colors="black", alpha=0.22, linewidths=0.45)
    return contour, interpolated


def plot_aliasing_heatmap(
    hmap_loc: np.ndarray,
    hmap_pcn: np.ndarray,
    *,
    run_id: str,
    config: Optional[dict] = None,
    out_path: Path,
    stats_path: Optional[Path] = None,
    gridsize: Optional[int] = None,
    quantile: float = 0.1,
    min_visits: int = 3,
    distance_fraction: float = 0.25,
    distance_threshold: Optional[float] = None,
    min_distance_bins: int = 3,
    display_percentile: float = 98.0,
    title: Optional[str] = None,
) -> dict:
    """Render a spatial aliasing heatmap and return the computed stats."""
    stats = compute_aliasing_stats(
        hmap_loc,
        hmap_pcn,
        gridsize=gridsize,
        quantile=quantile,
        min_visits=min_visits,
        distance_fraction=distance_fraction,
        distance_threshold=distance_threshold,
        min_distance_bins=min_distance_bins,
    )

    finite_values = stats["aliasing_map"][np.isfinite(stats["aliasing_map"])]
    positive_values = finite_values[finite_values > 0]
    vmax = float(np.percentile(positive_values, display_percentile)) if positive_values.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    if _griddata is not None:
        artist, interpolated = _plot_aliasing_contour(
            ax,
            xcenters=stats["xcenters"],
            zcenters=stats["zcenters"],
            aliasing_map=stats["aliasing_map"],
            cmap="jet",
        )
        display_peak = float(np.nanmax(interpolated)) if np.size(interpolated) else 0.0
    else:
        aliasing_map = np.ma.masked_invalid(stats["aliasing_map"].T)
        artist = ax.pcolormesh(
            stats["xedges"],
            stats["zedges"],
            aliasing_map,
            shading="auto",
            cmap="jet",
            vmin=0.0,
            vmax=vmax,
        )
        display_peak = float(np.max(positive_values)) if positive_values.size else 0.0

    _overlay_world_obstacles(ax, config or {})
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_title(title or f"Spatial aliasing - {run_id}", pad=12)

    if make_axes_locatable is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colorbar = plt.colorbar(artist, cax=cax)
    else:
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.9)
    colorbar.set_label("SAI")

    fig.text(
        0.5,
        0.02,
        (
            f"MSAI = {stats['msai']:.4f}    "
            f"Valid bins = {stats['valid_bin_count']}    "
            f"Far-bin threshold = {stats['distance_threshold']:.2f} m"
        ),
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    _save_figure(fig, out_path)
    print(f"Saved aliasing plot: {out_path}")

    if stats_path is not None:
        stats_payload = {
            "run_id": run_id,
            "msai": stats["msai"],
            "valid_bin_count": stats["valid_bin_count"],
            "active_cell_count": stats["active_cell_count"],
            "distance_threshold": stats["distance_threshold"],
            "parameters": {
                "gridsize": gridsize,
                "resolved_gridsize": stats["resolved_gridsize"],
                "quantile": quantile,
                "min_visits": min_visits,
                "distance_fraction": distance_fraction,
                "distance_threshold": distance_threshold,
                "min_distance_bins": min_distance_bins,
                "display_percentile": display_percentile,
            },
            "display_vmax": vmax,
            "display_peak": display_peak,
        }
        _write_json(stats_path, stats_payload)
        print(f"Saved aliasing stats: {stats_path}")

    return stats


def run(
    run_id: str,
    *,
    gridsize: int = 80,
    place_gridsize: Optional[int] = None,
    num_cells_to_sample: Optional[int] = 30,
    seed: int = 42,
    aliasing_gridsize: Optional[int] = None,
    aliasing_quantile: float = 0.1,
    aliasing_min_visits: int = 3,
    aliasing_distance_fraction: float = 0.25,
    aliasing_distance_threshold: Optional[float] = None,
    aliasing_min_distance_bins: int = 3,
    aliasing_display_percentile: float = 98.0,
) -> Path:
    """Generate all current verification plots for a run and return the output directory."""
    hmap_loc, hmap_pcn, hmap_gcn, config = load_run_artifacts(run_id)

    vis_dir = VIS_ROOT / run_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    world = config.get("world_name", "")
    mode = config.get("mode", "")
    base_title = f"{run_id}" + (f"  [{world} / {mode}]" if world or mode else "")
    resolved_place_gridsize = place_gridsize if place_gridsize is not None else max(gridsize, 160)

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
        gridsize=resolved_place_gridsize,
        num_cells_to_sample=num_cells_to_sample,
        title=f"Place cells - {base_title}",
        seed=seed,
    )
    plot_grid_cells(
        hmap_loc,
        run_id=run_id,
        hmap_gcn=hmap_gcn,
        config=config,
        out_path=vis_dir / "grid_cells.png",
        title=f"Grid cells - {base_title}",
        seed=seed,
    )
    plot_aliasing_heatmap(
        hmap_loc,
        hmap_pcn,
        run_id=run_id,
        config=config,
        out_path=vis_dir / "aliasing_heatmap.png",
        stats_path=vis_dir / "aliasing_stats.json",
        gridsize=aliasing_gridsize,
        quantile=aliasing_quantile,
        min_visits=aliasing_min_visits,
        distance_fraction=aliasing_distance_fraction,
        distance_threshold=aliasing_distance_threshold,
        min_distance_bins=aliasing_min_distance_bins,
        display_percentile=aliasing_display_percentile,
        title=f"Spatial aliasing - {base_title}",
    )

    update_run_summary(RUNS_DIR / run_id, visualization_dir=vis_dir)

    return vis_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce verification plots for a completed multi_grid_simple run."
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
        help="Base grid resolution used by verification plots (default: 80).",
    )
    parser.add_argument(
        "--place-gridsize",
        type=int,
        default=0,
        metavar="G",
        help="Explicit place-cell overlay grid resolution. Use 0 for the auto high-resolution default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible cell colors (default: 42).",
    )
    parser.add_argument(
        "--aliasing-gridsize",
        type=int,
        default=0,
        metavar="G",
        help="Grid resolution for the aliasing heatmap. Use 0 for automatic sizing (default: 0).",
    )
    parser.add_argument(
        "--aliasing-quantile",
        type=float,
        default=0.1,
        help="Per-cell activation quantile used before aliasing comparison (default: 0.1).",
    )
    parser.add_argument(
        "--aliasing-min-visits",
        type=int,
        default=3,
        metavar="N",
        help="Minimum visits required for a spatial bin to contribute to aliasing (default: 3).",
    )
    parser.add_argument(
        "--aliasing-distance-fraction",
        type=float,
        default=0.25,
        help="Default far-bin distance as a fraction of arena span when no explicit threshold is given (default: 0.25).",
    )
    parser.add_argument(
        "--aliasing-distance-threshold",
        type=float,
        default=None,
        help="Explicit far-bin distance threshold in meters for aliasing comparisons.",
    )
    parser.add_argument(
        "--aliasing-min-distance-bins",
        type=int,
        default=3,
        metavar="N",
        help="Minimum far-bin threshold in units of bin diagonals (default: 3).",
    )
    parser.add_argument(
        "--aliasing-display-percentile",
        type=float,
        default=98.0,
        help="Upper percentile used for aliasing heatmap color scaling (default: 98.0).",
    )
    args = parser.parse_args()

    num_cells = None if args.cells == 0 else args.cells
    vis_dir = run(
        args.run_id,
        gridsize=args.gridsize,
        place_gridsize=None if args.place_gridsize == 0 else args.place_gridsize,
        num_cells_to_sample=num_cells,
        seed=args.seed,
        aliasing_gridsize=None if args.aliasing_gridsize == 0 else args.aliasing_gridsize,
        aliasing_quantile=args.aliasing_quantile,
        aliasing_min_visits=args.aliasing_min_visits,
        aliasing_distance_fraction=args.aliasing_distance_fraction,
        aliasing_distance_threshold=args.aliasing_distance_threshold,
        aliasing_min_distance_bins=args.aliasing_min_distance_bins,
        aliasing_display_percentile=args.aliasing_display_percentile,
    )
    print(f"\nDone. Outputs in: {vis_dir}")


if __name__ == "__main__":
    main()
