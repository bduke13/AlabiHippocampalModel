import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Flip this to False to visualize the original training hmaps instead.
USE_PLOTTING_OUTPUT = True
TRAINING_HMAP_FOLDER = "hmaps"
PLOTTING_HMAP_FOLDER = "hmaps_plotting"
FLIP_DISPLAY_Z_AXIS = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from vis_utils import (
    discover_available_scales,
    get_output_dir,
    resolve_data_context,
)


def _active_hmap_dir_name() -> str:
    return PLOTTING_HMAP_FOLDER if USE_PLOTTING_OUTPUT else TRAINING_HMAP_FOLDER


def _resolve_hmap_root() -> Path:
    hmap_dir = _active_hmap_dir_name()
    _, _, data_root = resolve_data_context(required_relpaths=[f"{hmap_dir}/hmap_loc.pkl"])
    return data_root / hmap_dir


def _resolve_data_root_with_active_hmaps(
    required_hmap_files: List[str],
    extra_required_relpaths: Optional[List[str]] = None,
) -> Path:
    hmap_dir = _active_hmap_dir_name()
    required_relpaths = [f"{hmap_dir}/{filename}" for filename in required_hmap_files]
    required_relpaths.extend(extra_required_relpaths or [])
    _, _, data_root = resolve_data_context(required_relpaths=required_relpaths)
    return data_root


def _strip_legacy_leading_row(arr: np.ndarray) -> np.ndarray:
    """
    Keep compatibility with existing visualization loaders that drop the first row.
    """
    if arr.ndim == 0 or arr.shape[0] <= 1:
        return arr
    return arr[1:]


def _discover_scales_from_unified_bundle() -> List[int]:
    try:
        data_root = _resolve_data_root_with_active_hmaps(
            ["hmap_loc.pkl"],
            extra_required_relpaths=["networks/pcn_unified.pkl"],
        )
    except FileNotFoundError:
        return []

    bundle_path = data_root / "networks" / "pcn_unified.pkl"
    if not bundle_path.exists():
        return []

    try:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception:
        return []

    if isinstance(bundle, dict) and bundle.get("format") == "unified_pcn_bundle_v1":
        scale_slices = bundle.get("scale_slices", [])
        return list(range(len(scale_slices)))

    if hasattr(bundle, "scale_boundaries"):
        boundaries = list(getattr(bundle, "scale_boundaries"))
        return list(range(max(0, len(boundaries) - 1)))

    return []


def _discover_scales_from_active_hmaps() -> List[int]:
    try:
        hmap_root = _resolve_hmap_root()
    except FileNotFoundError:
        return []

    scales = set()
    prefix = "hmap_pcn_scale_"
    suffix = ".pkl"
    for pkl_path in hmap_root.glob(f"{prefix}*{suffix}"):
        name = pkl_path.name
        try:
            scales.add(int(name[len(prefix) : -len(suffix)]))
        except ValueError:
            continue
    return sorted(scales)


def _load_unified_scale_slice(scale: int) -> Optional[np.ndarray]:
    try:
        data_root = _resolve_data_root_with_active_hmaps(
            ["hmap_pcn.pkl"],
            extra_required_relpaths=["networks/pcn_unified.pkl"],
        )
    except FileNotFoundError:
        return None

    hmap_path = data_root / _active_hmap_dir_name() / "hmap_pcn.pkl"
    bundle_path = data_root / "networks" / "pcn_unified.pkl"
    if not hmap_path.exists() or not bundle_path.exists():
        return None

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and bundle.get("format") == "unified_pcn_bundle_v1":
        scale_slices = bundle.get("scale_slices", [])
        if scale < 0 or scale >= len(scale_slices):
            return None
        start, end = scale_slices[scale]
    elif hasattr(bundle, "scale_boundaries"):
        boundaries = list(getattr(bundle, "scale_boundaries"))
        if scale < 0 or scale + 1 >= len(boundaries):
            return None
        start, end = boundaries[scale], boundaries[scale + 1]
    else:
        return None

    arr = _strip_legacy_leading_row(np.array(load_pickle(hmap_path)))

    if arr.ndim != 2 or end > arr.shape[1]:
        return None
    return arr[:, start:end]


def _load_unified_full_hmap() -> Optional[np.ndarray]:
    try:
        data_root = _resolve_data_root_with_active_hmaps(["hmap_pcn.pkl"])
    except FileNotFoundError:
        return None

    hmap_path = data_root / _active_hmap_dir_name() / "hmap_pcn.pkl"
    if not hmap_path.exists():
        return None

    arr = _strip_legacy_leading_row(np.array(load_pickle(hmap_path)))
    if arr.ndim != 2:
        return None
    return arr


def _resolve_scales(scales: Optional[List[int]] = None) -> List[int]:
    if scales is not None:
        return list(scales)

    unified = _discover_scales_from_unified_bundle()
    if unified:
        return unified

    active_hmap_scales = _discover_scales_from_active_hmaps()
    if active_hmap_scales:
        return active_hmap_scales

    discovered = discover_available_scales()
    if discovered:
        return discovered

    return [0, 1, 2]


def load_pickle(file_path):
    """Load a pickle file from the given path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_multi_scale_hmaps(scales=None):
    """
    Load history map (hmap) data for multiple scales.

    Supports both legacy per-scale outputs and unified controller runs using the
    saved training histories directly.
    """
    scales = _resolve_scales(scales)
    hmap_root = _resolve_hmap_root()

    hmap_loc = _strip_legacy_leading_row(np.array(load_pickle(hmap_root / "hmap_loc.pkl")))
    print(f"Loaded hmap_loc from {hmap_root / 'hmap_loc.pkl'}")

    hmap_pcn_dict = {}
    for scale in scales:
        unified_slice = _load_unified_scale_slice(scale)
        if unified_slice is not None:
            hmap_pcn_dict[scale] = unified_slice
            print(f"Loaded unified hmap_pcn slice for scale {scale}")
            continue

        per_scale_path = hmap_root / f"hmap_pcn_scale_{scale}.pkl"
        if per_scale_path.exists():
            hmap_pcn_dict[scale] = _strip_legacy_leading_row(np.array(load_pickle(per_scale_path)))
            print(f"Loaded hmap_pcn_scale_{scale} from {per_scale_path}")
            continue

        print(f"Warning: Missing PCN activation file for scale {scale}")

    unified_hmap_pcn = _load_unified_full_hmap()
    activity_source = "plotting_output" if USE_PLOTTING_OUTPUT else "training_history"
    return hmap_loc, hmap_pcn_dict, unified_hmap_pcn, f"{activity_source}:{hmap_root.name}"


def apply_time_window(
    hmap_loc: np.ndarray,
    hmap_pcn_dict: dict,
    unified_hmap_pcn: Optional[np.ndarray],
    time_window_fraction: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """
    Restrict all loaded histories to a fractional time window.

    Example: ``(0.8, 1.0)`` keeps the last 20% of training only.
    """
    if time_window_fraction is None:
        return hmap_loc, hmap_pcn_dict, unified_hmap_pcn

    start_frac, end_frac = time_window_fraction
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    end_frac = float(np.clip(end_frac, 0.0, 1.0))
    if end_frac <= start_frac:
        raise ValueError(
            f"Invalid time_window_fraction={time_window_fraction}; expected end > start."
        )

    total_steps = hmap_loc.shape[0]
    start_idx = min(total_steps - 1, int(np.floor(total_steps * start_frac)))
    end_idx = max(start_idx + 1, int(np.ceil(total_steps * end_frac)))
    end_idx = min(total_steps, end_idx)
    window = slice(start_idx, end_idx)

    print(
        f"Using history window {start_frac:.0%}-{end_frac:.0%} "
        f"(steps {start_idx}:{end_idx} of {total_steps})"
    )

    windowed_hmap_loc = hmap_loc[window]
    windowed_hmap_pcn_dict = {
        scale: (None if hmap is None else hmap[window])
        for scale, hmap in hmap_pcn_dict.items()
    }
    windowed_unified = None if unified_hmap_pcn is None else unified_hmap_pcn[window]
    return windowed_hmap_loc, windowed_hmap_pcn_dict, windowed_unified


def convert_xzy_hmaps(hmap_loc):
    """Convert hmap location array to separate x, y, z components."""
    hmap_x = hmap_loc[:, 0]
    hmap_z = hmap_loc[:, 1]
    hmap_y = -hmap_loc[:, 2] if FLIP_DISPLAY_Z_AXIS else hmap_loc[:, 2]
    return hmap_x, hmap_z, hmap_y


def generate_random_colors(num_colors):
    """Generate an array of random vibrant colors in RGB format."""
    return np.random.rand(num_colors, 3)


def build_top_activation_mask(
    activations: np.ndarray,
    top_activation_fraction: Optional[float],
) -> np.ndarray:
    """
    Keep only the strongest activation tail for a single cell.

    For example, ``top_activation_fraction=0.15`` keeps roughly the top 15% of
    positive activation samples for that cell.
    """
    positive_mask = activations > 0
    if not np.any(positive_mask):
        return positive_mask

    if top_activation_fraction is None or top_activation_fraction >= 1.0:
        return positive_mask

    top_activation_fraction = float(np.clip(top_activation_fraction, 0.0, 1.0))
    if top_activation_fraction <= 0.0:
        strongest_mask = np.zeros_like(activations, dtype=bool)
        strongest_mask[int(np.argmax(activations))] = True
        return strongest_mask

    cutoff = np.quantile(activations[positive_mask], 1.0 - top_activation_fraction)
    strongest_mask = positive_mask & (activations >= cutoff)
    if not np.any(strongest_mask):
        strongest_mask[int(np.argmax(activations))] = True
    return strongest_mask


def plot_scale_cells(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    scale: Union[int, str],
    gridsize: int = 200,
    num_cells_to_sample: Optional[int] = 10,
    top_activation_fraction: Optional[float] = 0.15,
    show_plot: bool = False,
    ax=None,
    title_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot place cells for a specific scale.
    """
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)
    nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

    if nonzero_activation_indices.size == 0:
        if ax is not None:
            ax.text(0.5, 0.5, f"No active cells for {title_label or scale}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return np.zeros((gridsize, gridsize, 3)), np.array([])

    if num_cells_to_sample is None:
        num_cells_to_plot = len(nonzero_activation_indices)
    else:
        num_cells_to_plot = min(num_cells_to_sample, len(nonzero_activation_indices))

    panel_label = title_label or f"Scale {scale}"
    print(f"{panel_label}: Plotting {num_cells_to_plot} cells out of {len(nonzero_activation_indices)} active cells")
    if top_activation_fraction is not None and top_activation_fraction < 1.0:
        print(f"{panel_label}: Restricting each cell to top {top_activation_fraction * 100:.0f}% of positive activations")

    cell_indices = np.random.choice(nonzero_activation_indices, size=num_cells_to_plot, replace=False)

    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))

    for idx, cell_index in enumerate(cell_indices):
        activations = hmap_pcn[:, cell_index]
        mask = build_top_activation_mask(activations, top_activation_fraction)
        if not np.any(mask):
            continue
        x, y, a = hmap_x[mask], hmap_y[mask], activations[mask]

        ix = np.clip(np.digitize(x, xedges) - 1, 0, gridsize - 1)
        iy = np.clip(np.digitize(y, yedges) - 1, 0, gridsize - 1)

        for i, j, activation in zip(ix, iy, a):
            total_activations_per_bin[i, j, idx] += activation
            counts_per_bin[i, j, idx] += 1

    mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
    nonzero_counts = counts_per_bin > 0
    mean_activation_per_bin[nonzero_counts] = (
        total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
    )

    max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
    cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)

    max_activation = np.max(max_mean_activation_per_bin)
    max_activation = max_activation if max_activation > 0 else 1
    normalized_activation = max_mean_activation_per_bin / max_activation

    colors_rgb = generate_random_colors(num_cells_to_plot)
    image = np.zeros((gridsize, gridsize, 3))

    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                image[i, j, :] = activation_level * np.array(colors_rgb[idx])

    image = np.transpose(image, (1, 0, 2))

    if ax is not None:
        extent = [xmin, xmax, ymin, ymax]
        ax.imshow(image, extent=extent, origin="lower")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{panel_label}: {num_cells_to_plot} Place Cells")

    return image, cell_indices


def plot_all_multi_scale_cells(
    scales: List[int] = None,
    gridsize: int = 200,
    num_cells_to_sample: Optional[int] = 10,
    top_activation_fraction: Optional[float] = 0.15,
    time_window_fraction: Optional[Tuple[float, float]] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None,
    plot_all_cells_too: bool = True,
    include_unified_view: bool = True,
):
    """
    Plot place cells for multiple scales.
    """
    scales = _resolve_scales(scales)

    hmap_loc, hmap_pcn_dict, unified_hmap_pcn, activity_source = load_multi_scale_hmaps(scales)
    if not include_unified_view:
        unified_hmap_pcn = None
    hmap_loc, hmap_pcn_dict, unified_hmap_pcn = apply_time_window(
        hmap_loc=hmap_loc,
        hmap_pcn_dict=hmap_pcn_dict,
        unified_hmap_pcn=unified_hmap_pcn,
        time_window_fraction=time_window_fraction,
    )
    print(f"Overlay source: {activity_source}")
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    view_specs: List[Tuple[Union[int, str], str, Optional[np.ndarray]]] = [
        (scale, f"Scale {scale}", hmap_pcn_dict.get(scale)) for scale in scales
    ]
    if unified_hmap_pcn is not None:
        view_specs.append(("unified", "Unified", unified_hmap_pcn))

    num_rows = 2 if plot_all_cells_too else 1
    num_cols = max(1, len(view_specs))
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols)

    for i, (view_key, title_label, hmap_pcn) in enumerate(view_specs):
        ax = plt.subplot(gs[0, i])
        if hmap_pcn is None:
            ax.text(0.5, 0.5, f"Missing data for {title_label}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        plot_scale_cells(
            hmap_pcn=hmap_pcn,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale=view_key,
            gridsize=gridsize,
            num_cells_to_sample=num_cells_to_sample,
            top_activation_fraction=top_activation_fraction,
            ax=ax,
            title_label=title_label,
        )

    if plot_all_cells_too:
        for i, (view_key, title_label, hmap_pcn) in enumerate(view_specs):
            ax = plt.subplot(gs[1, i])
            if hmap_pcn is None:
                ax.text(0.5, 0.5, f"Missing data for {title_label}", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            plot_scale_cells(
                hmap_pcn=hmap_pcn,
                hmap_x=hmap_x,
                hmap_y=hmap_y,
                scale=view_key,
                gridsize=gridsize,
                num_cells_to_sample=None,
                top_activation_fraction=top_activation_fraction,
                ax=ax,
                title_label=title_label,
            )

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def save_multi_scale_plots(
    base_filename: str = "multi_scale_pcn",
    scales: List[int] = None,
    gridsize: int = 200,
    num_cells_to_sample: int = 10,
    top_activation_fraction: Optional[float] = 0.15,
    time_window_fraction: Optional[Tuple[float, float]] = None,
    include_unified_view: bool = True,
):
    """
    Generate and save multi-scale place cell plots.
    """
    scales = _resolve_scales(scales)
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{base_filename}.png")

    plot_all_multi_scale_cells(
        scales=scales,
        gridsize=gridsize,
        num_cells_to_sample=num_cells_to_sample,
        top_activation_fraction=top_activation_fraction,
        time_window_fraction=time_window_fraction,
        show_plot=False,
        save_path=save_path,
        plot_all_cells_too=True,
        include_unified_view=include_unified_view,
    )

    print(f"Multi-scale plots saved to {save_path}")


if __name__ == "__main__":
    print("Starting multi-scale place cell visualization...")
    plot_all_multi_scale_cells(
        scales=None,
        gridsize=200,
        num_cells_to_sample=10,
        top_activation_fraction=1.00,
        time_window_fraction=None,
        show_plot=True,
        save_path=os.path.join(get_output_dir(), "multi_scale_pcn.png"),
        plot_all_cells_too=True,
        include_unified_view=True,
    )
    print("Visualization complete!")
