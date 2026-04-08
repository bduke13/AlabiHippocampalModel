"""
Place field visualization focused on honest formed-field structure.

This script now separates three different questions:
1. Where do formed fields actually cover space?
2. What do the strongest formed fields look like by field peak?
3. What do the strongest learned formed fields look like by |dw|?

The old single-panel top-10%-by-learning winner map made those three ideas look
like one thing, which could make fields appear random even when the formed-field
coverage was more coherent.
"""

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPT_DIR))

try:
    from visualizations.place_cell_learning import load_place_cell_learning_scores
except ModuleNotFoundError:
    def _to_numpy_array(value):
        """Convert tensors/parameters/arrays to a CPU numpy array."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=np.float32)


    def _score_place_cells_from_model(model_obj, scale=None, unified=False):
        """Fallback learning-strength score from current input/grid weights."""
        weight_names = ("w_in", "w_grid")
        scores = None
        for name in weight_names:
            weight = _to_numpy_array(getattr(model_obj, name, None))
            if weight is None or weight.ndim != 2:
                continue
            component = np.sum(np.abs(weight), axis=1, dtype=np.float64).astype(
                np.float32
            )
            scores = component if scores is None else (scores + component)

        if scores is None:
            return None

        if unified and scale is not None and hasattr(model_obj, "scale_boundaries"):
            boundaries = list(getattr(model_obj, "scale_boundaries", []))
            if len(boundaries) >= 2 and scale + 1 < len(boundaries):
                start = int(boundaries[scale])
                end = int(boundaries[scale + 1])
                scores = scores[start:end]
        return scores


    def load_place_cell_learning_scores(network_dir, scale=None, unified=False):
        """Fallback loader for per-cell learning-strength scores."""
        network_dir = Path(network_dir)
        meta = {"used_fallback": True, "error": None}

        candidate_paths = []
        if unified:
            candidate_paths.append(network_dir / "unified_pcn.pkl")
        elif scale is not None:
            candidate_paths.append(network_dir / f"pcn_scale_{scale}.pkl")
            candidate_paths.append(network_dir / "unified_pcn.pkl")
        else:
            candidate_paths.append(network_dir / "unified_pcn.pkl")

        model_obj = None
        model_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                with open(candidate, "rb") as f:
                    model_obj = pickle.load(f)
                model_path = candidate
                break

        if model_obj is None:
            meta["error"] = "No PCN weights found for learning-score fallback."
            return None, meta

        scores = _score_place_cells_from_model(
            model_obj,
            scale=scale,
            unified=bool(unified or model_path.name == "unified_pcn.pkl"),
        )
        if scores is None:
            meta["error"] = f"No supported weight matrices found in {model_path.name}."
            return None, meta
        return scores, meta

# Configuration
CONTROLLER_NAME = "msg_controller"
ENVIRONMENTS = [
    "environment_1",
    "environment_2",
    "environment_3",
    "environment_4",
    "environment_5",
    "environment_6",
]
MULTI_GOAL_WORLDS = {
    "20x20_multi_goal",
    "20x20_cross_multi_goal",
    "20x20_maze_multi_goal",
}
ARCHITECTURE = "auto"  # "auto", "unified", "separate"
CELL_RANKING_MODE = "learning_strength"
FALLBACK_PEAK_THRESHOLD = 0.10
FORMED_FIELD_PEAK_THRESHOLD = 0.10
FORMED_FIELD_ACTIVE_FRACTION = 0.50
FORMED_FIELD_MIN_BINS = 6
TOP_RANKED_FRACTION = 0.10
TOP_RANKED_MAX_CELLS = 64
TOP_RANKED_MIN_CELLS = 16
LEARNING_VIEW_RESTRICT_TO_FORMED = True
STRONG_FIELD_PERCENTILE = 50  # Only show fields with peak above this percentile of formed peaks
SCALE_WINNER_RELATIVE_MARGIN_THRESHOLD = 0.20

# Absolute paths
DATA_BASE_DIR = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl"
OUTPUT_DIR = PROJECT_ROOT / "place_field_plots"


def generate_random_colors(n):
    """Generate n random vibrant colors."""
    rng = np.random.default_rng(42)
    colors = rng.random((n, 3))
    if n > 0:
        max_idx = np.argmax(colors, axis=1)
        colors[np.arange(n), max_idx] = rng.uniform(0.7, 1.0, size=n)
    return colors


def discover_available_environments():
    """Discover environments with saved hmap data."""
    if not DATA_BASE_DIR.exists():
        return []
    envs = []
    for env_dir in sorted(DATA_BASE_DIR.iterdir()):
        if not env_dir.is_dir():
            continue
        if (env_dir / "hmaps" / "hmap_loc.pkl").exists():
            envs.append(env_dir.name)
    return envs


def resolve_environments():
    """Resolve which environments to process for this run."""
    discovered = discover_available_environments()
    preferred = [env for env in ENVIRONMENTS if env in discovered]
    if preferred:
        return preferred
    return discovered


def _extract_planar_xy(hmap_loc, env_name):
    """
    Return planar coordinates from current-controller hmap_loc.

    Current msg_driver writes canonical coordinates:
      [planar_x, planar_y, height]

    for both environment_1..6 and 20x20_*_multi_goal worlds.
    For unknown worlds, fall back to treating the smallest-range axis as height.
    """
    known_world = env_name in ENVIRONMENTS or env_name in MULTI_GOAL_WORLDS
    if known_world:
        return hmap_loc[:, 0], hmap_loc[:, 1]

    spans = np.ptp(hmap_loc, axis=0)
    height_axis = int(np.argmin(spans))
    planar_axes = [idx for idx in range(hmap_loc.shape[1]) if idx != height_axis]
    return hmap_loc[:, planar_axes[0]], hmap_loc[:, planar_axes[1]]


def load_hmap_loc(env_name):
    """Load location history for an environment."""
    loc_file = DATA_BASE_DIR / env_name / "hmaps" / "hmap_loc.pkl"
    if not loc_file.exists():
        return None
    with open(loc_file, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
    if len(hmap_loc) > 1:
        hmap_loc = hmap_loc[1:]
    return hmap_loc


def load_hmap_prox(env_name):
    """Load per-step proximity history for an environment."""
    prox_file = DATA_BASE_DIR / env_name / "hmaps" / "hmap_prox.pkl"
    if not prox_file.exists():
        return None
    with open(prox_file, "rb") as f:
        hmap_prox = np.array(pickle.load(f), dtype=np.float32)
    if len(hmap_prox) > 1:
        hmap_prox = hmap_prox[1:]
    return hmap_prox


def discover_available_scales(env_name):
    """Discover scale indices from hmap files on disk."""
    hmap_dir = DATA_BASE_DIR / env_name / "hmaps"
    if not hmap_dir.exists():
        return []
    scales = []
    for fpath in sorted(hmap_dir.glob("hmap_pcn_scale_*.pkl")):
        try:
            scales.append(int(fpath.stem.split("_")[-1]))
        except ValueError:
            continue
    return scales


def is_unified_run(env_name):
    """Check whether unified artifacts are available."""
    return (DATA_BASE_DIR / env_name / "networks" / "unified_pcn.pkl").exists()


def load_unified_expression_config(env_name):
    """Load the unified PCN and extract post-competition expression settings."""
    network_path = DATA_BASE_DIR / env_name / "networks" / "unified_pcn.pkl"
    if not network_path.exists():
        return None
    with open(network_path, "rb") as f:
        unified_pcn = pickle.load(f)
    d_opt = np.asarray(unified_pcn.d_opt.detach().cpu().numpy(), dtype=np.float32)
    sigma_tune = np.asarray(
        unified_pcn.sigma_tune_per_scale.detach().cpu().numpy(),
        dtype=np.float32,
    )
    configured_mode = str(
        getattr(
            unified_pcn,
            "learning_adaptation_mode",
            "gaussian_post_competition_expression",
        )
    ).strip().lower()
    effective_mode = str(
        getattr(
            unified_pcn,
            "last_learning_adaptation_mode",
            configured_mode,
        )
    ).strip().lower()
    if not effective_mode:
        effective_mode = configured_mode
    return {
        "learning_adaptation_mode": configured_mode,
        "effective_learning_adaptation_mode": effective_mode,
        "d_opt": d_opt,
        "sigma_tune": sigma_tune,
        "expression_power": float(
            getattr(unified_pcn, "post_competition_expression_power", 2.0)
        ),
        "large_scale_one_sided": bool(
            getattr(unified_pcn, "large_scale_one_sided", False)
        ),
        "large_scale_plateau": float(
            getattr(unified_pcn, "large_scale_plateau", 1.0)
        ),
        "large_scale_plateau_onset_sigma": float(
            getattr(unified_pcn, "large_scale_plateau_onset_sigma", 1.0)
        ),
        "large_scale_plateau_full_sigma": float(
            getattr(unified_pcn, "large_scale_plateau_full_sigma", 2.0)
        ),
        "num_scales": int(getattr(unified_pcn, "num_scales", len(d_opt))),
    }


def load_hmap_data(env_name, scale):
    """Load location and PC activation history for a scale."""
    hmap_loc = load_hmap_loc(env_name)
    if hmap_loc is None:
        return None, None

    pcn_file = DATA_BASE_DIR / env_name / "hmaps" / f"hmap_pcn_scale_{scale}.pkl"
    if not pcn_file.exists():
        return hmap_loc, None

    with open(pcn_file, "rb") as f:
        hmap_pcn = np.array(pickle.load(f))
    if len(hmap_pcn) > 1:
        hmap_pcn = hmap_pcn[1:]

    return hmap_loc, hmap_pcn


def load_unified_aggregate_data(env_name, available_scales):
    """Load a unified aggregate activation matrix by concatenating per-scale hmaps."""
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

    hmap_loc = hmap_loc[:min_steps]
    per_scale = [arr[:min_steps] for arr in per_scale]
    return hmap_loc, np.concatenate(per_scale, axis=1)


def load_place_cell_scores(env_name, scale=None, unified=False):
    """Load per-cell learning-strength scores from saved PCN weights."""
    network_dir = DATA_BASE_DIR / env_name / "networks"
    scores, meta = load_place_cell_learning_scores(
        network_dir=network_dir,
        scale=scale,
        unified=unified,
    )
    if scores is None and meta.get("error"):
        print(f"  Learning-score fallback: {meta['error']}")
    return scores, meta


def _prepare_spatial_bins(hmap_x, hmap_y, gridsize):
    """Precompute spatial bin assignments for streaming field analysis."""
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    xi = np.digitize(hmap_x, xedges) - 1
    yi = np.digitize(hmap_y, yedges) - 1
    valid = (xi >= 0) & (xi < gridsize) & (yi >= 0) & (yi < gridsize)
    flat_idx = (xi[valid] * gridsize + yi[valid]).astype(np.intp)
    counts_1d = np.bincount(flat_idx, minlength=gridsize * gridsize).astype(np.float32)

    return {
        "valid_mask": valid,
        "flat_idx": flat_idx,
        "counts_1d": counts_1d,
        "safe_counts": np.maximum(counts_1d, 1.0),
        "gridsize": gridsize,
        "extent": (xmin, xmax, ymin, ymax),
    }


def _stream_cell_mean_by_bin(hmap_pcn, bin_data, cell_index):
    """Compute one cell's mean activation per spatial bin without storing all cells."""
    values = np.bincount(
        bin_data["flat_idx"],
        weights=hmap_pcn[bin_data["valid_mask"], int(cell_index)].astype(np.float64),
        minlength=bin_data["counts_1d"].shape[0],
    ).astype(np.float32)
    values /= bin_data["safe_counts"]
    values[bin_data["counts_1d"] == 0] = 0.0
    return values


def _analyze_formed_fields(hmap_pcn, bin_data, candidate_cells):
    """Identify genuinely formed fields and summarize their spatial coverage."""
    num_bins = bin_data["counts_1d"].shape[0]
    coverage_count = np.zeros(num_bins, dtype=np.float32)
    coverage_strength = np.zeros(num_bins, dtype=np.float32)
    formed_mask = np.zeros(len(candidate_cells), dtype=bool)
    binned_peaks = np.zeros(len(candidate_cells), dtype=np.float32)

    for pos, cell_index in enumerate(candidate_cells):
        values = _stream_cell_mean_by_bin(hmap_pcn, bin_data, cell_index)
        peak = float(np.max(values))
        binned_peaks[pos] = peak
        if peak <= 0.0:
            continue

        active_mask = values >= (FORMED_FIELD_ACTIVE_FRACTION * peak)
        active_bin_count = int(np.count_nonzero(active_mask))
        if peak < FORMED_FIELD_PEAK_THRESHOLD or active_bin_count < FORMED_FIELD_MIN_BINS:
            continue

        formed_mask[pos] = True
        coverage_count[active_mask] += 1.0
        coverage_strength += values / peak

    formed_cells = candidate_cells[formed_mask]
    formed_peaks = binned_peaks[formed_mask]
    formed_count = int(formed_cells.shape[0])

    coverage_fraction = np.zeros_like(coverage_count)
    coverage_mean_strength = np.zeros_like(coverage_strength)
    if formed_count > 0:
        coverage_fraction = coverage_count / float(formed_count)
        coverage_mean_strength = coverage_strength / float(formed_count)

    gridsize = bin_data["gridsize"]
    return {
        "formed_cells": formed_cells,
        "formed_peaks": formed_peaks,
        "formed_count": formed_count,
        "coverage_fraction": coverage_fraction.reshape(gridsize, gridsize).T,
        "coverage_mean_strength": coverage_mean_strength.reshape(gridsize, gridsize).T,
    }


def _choose_top_cells(cell_indices, ranking_scores):
    """Select a representative subset for overlay views."""
    if len(cell_indices) == 0:
        return np.array([], dtype=np.int32)
    num_to_plot = int(np.ceil(len(cell_indices) * TOP_RANKED_FRACTION))
    num_to_plot = max(TOP_RANKED_MIN_CELLS, num_to_plot)
    num_to_plot = min(TOP_RANKED_MAX_CELLS, num_to_plot, len(cell_indices))
    order = np.argsort(ranking_scores)[::-1]
    return cell_indices[order[:num_to_plot]]


def _build_winner_overlay(hmap_pcn, bin_data, cell_indices, cell_alphas=None):
    """Create a winner-take-all overlay for a chosen subset of cells.

    Args:
        cell_alphas: Optional per-cell opacity array (0-1). When provided,
            each cell's contribution is scaled by its alpha, letting weak
            fields fade out while strong fields remain vivid.
    """
    if len(cell_indices) == 0:
        return None

    num_bins = bin_data["counts_1d"].shape[0]
    winner_strength = np.zeros(num_bins, dtype=np.float32)
    winner_idx = np.full(num_bins, -1, dtype=np.int32)

    for pos, cell_index in enumerate(cell_indices):
        values = _stream_cell_mean_by_bin(hmap_pcn, bin_data, cell_index)
        better = values > winner_strength
        winner_strength[better] = values[better]
        winner_idx[better] = pos

    peak = float(np.max(winner_strength))
    if peak <= 0.0:
        return None

    normalized = winner_strength / peak
    colors = generate_random_colors(len(cell_indices))
    image = np.zeros((num_bins, 3), dtype=np.float32)
    nonzero = winner_idx >= 0
    image[nonzero] = normalized[nonzero, np.newaxis] * colors[winner_idx[nonzero]]

    # Apply per-cell alpha scaling (dims weak fields, preserves strong ones)
    if cell_alphas is not None:
        per_bin_alpha = np.zeros(num_bins, dtype=np.float32)
        per_bin_alpha[nonzero] = cell_alphas[winner_idx[nonzero]]
        image[nonzero] *= per_bin_alpha[nonzero, np.newaxis]

    gridsize = bin_data["gridsize"]
    return image.reshape(gridsize, gridsize, 3).transpose(1, 0, 2)


def _mean_signal_by_bin(signal, bin_data):
    """Average a 1D per-step signal into spatial bins."""
    values = np.bincount(
        bin_data["flat_idx"],
        weights=np.asarray(signal)[bin_data["valid_mask"]].astype(np.float64),
        minlength=bin_data["counts_1d"].shape[0],
    ).astype(np.float32)
    values /= bin_data["safe_counts"]
    values[bin_data["counts_1d"] == 0] = 0.0
    return values


def _compute_scale_preference_matrix(proximity_series, config):
    """Reconstruct Gaussian scale preference per timestep from saved PCN config."""
    proximity = np.asarray(proximity_series, dtype=np.float32).reshape(-1, 1)
    d_opt = np.asarray(config["d_opt"], dtype=np.float32).reshape(1, -1)
    sigma = np.maximum(
        np.asarray(config["sigma_tune"], dtype=np.float32).reshape(1, -1),
        1e-6,
    )
    pref = np.exp(-((proximity - d_opt) ** 2) / (2.0 * (sigma ** 2))).astype(
        np.float32
    )
    if bool(config.get("large_scale_one_sided", False)) and pref.shape[1] > 0:
        last_idx = pref.shape[1] - 1
        plateau = float(config.get("large_scale_plateau", 1.0))
        onset_sigma = float(config.get("large_scale_plateau_onset_sigma", 1.0))
        full_sigma = float(config.get("large_scale_plateau_full_sigma", 2.0))
        d_last = float(d_opt[0, last_idx])
        sigma_last = float(sigma[0, last_idx])
        onset = d_last + (onset_sigma * sigma_last)
        full = d_last + (full_sigma * sigma_last)
        denom = max(full - onset, 1e-6)
        alpha = np.clip((proximity[:, 0] - onset) / denom, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        pref[:, last_idx] = (
            ((1.0 - alpha) * pref[:, last_idx]) + (alpha * plateau)
        ).astype(np.float32)
    return np.clip(pref, 0.0, 1.0).astype(np.float32)


def reconstruct_gaussian_post_competition_expression(proximity_series, config):
    """Rebuild the legacy Gaussian post-competition gains for diagnostics."""
    pref = _compute_scale_preference_matrix(proximity_series, config)
    power = max(1.0, float(config.get("expression_power", 2.0)))
    sharpened = np.power(np.maximum(pref, 1e-6), power, dtype=np.float32)
    gains = sharpened / np.maximum(
        np.max(sharpened, axis=1, keepdims=True), 1e-6
    )
    raw_gain = gains.copy()

    winner_idx = np.argmax(gains, axis=1).astype(np.int32)
    winner_onehot = np.zeros_like(gains, dtype=np.float32)
    winner_onehot[np.arange(gains.shape[0]), winner_idx] = 1.0
    sorted_gain = np.sort(gains, axis=1)
    if gains.shape[1] > 1:
        margin = sorted_gain[:, -1] - sorted_gain[:, -2]
    else:
        margin = sorted_gain[:, -1]
    relative_margin = margin / np.maximum(np.max(gains, axis=1), 1e-9)

    return {
        "scale_preference": pref,
        "raw_gain": raw_gain,
        "gain": gains,
        "winner_idx": winner_idx,
        "winner_onehot": winner_onehot,
        "winner_margin": margin.astype(np.float32),
        "winner_relative_margin": relative_margin.astype(np.float32),
    }


def build_gaussian_segmented_views(
    env_name,
    hmap_loc,
    hmap_prox,
    scale_hmaps,
    expression_config,
    gridsize=200,
):
    """Build winner-segmented and gain-weighted scale diagnostics."""
    if not scale_hmaps or hmap_loc is None or hmap_prox is None or expression_config is None:
        return None

    scales = sorted(scale_hmaps.keys())
    min_steps = min(
        [len(hmap_loc), len(hmap_prox)] + [arr.shape[0] for arr in scale_hmaps.values()]
    )
    if min_steps <= 0 or len(scales) != int(expression_config.get("num_scales", len(scales))):
        return None

    hmap_loc = hmap_loc[:min_steps]
    hmap_prox = hmap_prox[:min_steps]
    hmap_x, hmap_y = _extract_planar_xy(hmap_loc, env_name)
    bin_data = _prepare_spatial_bins(hmap_x, hmap_y, gridsize)
    expression = reconstruct_gaussian_post_competition_expression(
        hmap_prox,
        expression_config,
    )
    gains = expression["gain"][:min_steps]
    winner_idx_steps = expression["winner_idx"][:min_steps]
    winner_onehot = expression["winner_onehot"][:min_steps]

    gain_maps = {}
    winner_fraction_by_scale = {}
    segmented_hmaps = {}
    for scale_pos, scale in enumerate(scales):
        scale_gain = gains[:, scale_pos]
        gain_maps[scale] = _mean_signal_by_bin(scale_gain, bin_data).reshape(
            gridsize, gridsize
        ).T
        winner_fraction_by_scale[scale] = float(
            np.mean(winner_idx_steps == scale_pos)
        )
        winner_mask = winner_onehot[:, scale_pos].astype(np.float32)
        segmented_hmaps[scale] = scale_hmaps[scale][:min_steps] * winner_mask[:, None]

    winner_strength = np.max(gains, axis=1)
    winner_strength_map = _mean_signal_by_bin(winner_strength, bin_data).reshape(
        gridsize, gridsize
    ).T
    winner_rel_margin_map = _mean_signal_by_bin(
        expression["winner_relative_margin"][:min_steps],
        bin_data,
    ).reshape(gridsize, gridsize).T

    winner_idx_bins = np.full(bin_data["counts_1d"].shape[0], -1, dtype=np.int32)
    winner_counts = np.zeros((bin_data["counts_1d"].shape[0], len(scales)), dtype=np.float32)
    flat_idx = bin_data["flat_idx"]
    valid_mask = bin_data["valid_mask"]
    for scale_pos in range(len(scales)):
        binc = np.bincount(
            flat_idx,
            weights=winner_onehot[valid_mask, scale_pos].astype(np.float64),
            minlength=bin_data["counts_1d"].shape[0],
        ).astype(np.float32)
        winner_counts[:, scale_pos] = binc
    valid_bins = bin_data["counts_1d"] > 0
    winner_idx_bins[valid_bins] = np.argmax(winner_counts[valid_bins], axis=1).astype(
        np.int32
    )
    winner_rgb = np.zeros((winner_idx_bins.shape[0], 3), dtype=np.float32)
    scale_colors = np.array(
        [
            [0.86, 0.24, 0.18],
            [0.18, 0.57, 0.84],
            [0.18, 0.70, 0.32],
        ],
        dtype=np.float32,
    )
    valid_winners = winner_idx_bins >= 0
    winner_rgb[valid_winners] = scale_colors[winner_idx_bins[valid_winners]]
    peak_strength = float(np.max(winner_strength_map))
    if peak_strength > 0.0:
        flat_strength = winner_strength_map.T.reshape(-1)
        winner_rgb[valid_winners] *= (
            flat_strength[valid_winners] / peak_strength
        )[:, np.newaxis]

    return {
        "scales": scales,
        "mode_label": "Gaussian",
        "figure_title": "Gaussian Segmented Scale Routing",
        "winner_map_title": "Gaussian Winner Map",
        "winner_strength_title": "Winner Gain Strength",
        "per_scale_map_title_template": "Scale {scale} Mean Gain",
        "segmented_unified_label": "Unified (Gaussian-Segmented)",
        "segmented_scale_label_template": "Scale {scale} (Gaussian-Segmented)",
        "extent": bin_data["extent"],
        "gain_maps": gain_maps,
        "winner_strength": winner_strength_map,
        "winner_relative_margin": winner_rel_margin_map,
        "winner_rgb": winner_rgb.reshape(gridsize, gridsize, 3).transpose(1, 0, 2),
        "winner_fractions": winner_fraction_by_scale,
        "segmented_hmaps": segmented_hmaps,
        "trimmed_hmap_loc": hmap_loc,
    }


def build_relative_segmented_views(
    env_name,
    hmap_loc,
    scale_hmaps,
    gridsize=200,
):
    """Build segmented diagnostics from observed per-scale survivor evidence."""
    if not scale_hmaps or hmap_loc is None:
        return None

    scales = sorted(scale_hmaps.keys())
    min_steps = min([len(hmap_loc)] + [arr.shape[0] for arr in scale_hmaps.values()])
    if min_steps <= 0:
        return None

    hmap_loc = hmap_loc[:min_steps]
    hmap_x, hmap_y = _extract_planar_xy(hmap_loc, env_name)
    bin_data = _prepare_spatial_bins(hmap_x, hmap_y, gridsize)

    evidence_stack_steps = []
    for scale in scales:
        arr = scale_hmaps[scale][:min_steps]
        evidence_stack_steps.append(np.max(arr, axis=1).astype(np.float32))
    evidence_stack_steps = np.stack(evidence_stack_steps, axis=1)

    max_evidence = np.maximum(
        np.max(evidence_stack_steps, axis=1, keepdims=True),
        1e-6,
    )
    relative_evidence = evidence_stack_steps / max_evidence
    winner_idx_steps = np.argmax(evidence_stack_steps, axis=1).astype(np.int32)
    winner_onehot = np.zeros_like(relative_evidence, dtype=np.float32)
    winner_onehot[np.arange(relative_evidence.shape[0]), winner_idx_steps] = 1.0

    gain_maps = {}
    winner_fraction_by_scale = {}
    segmented_hmaps = {}
    for scale_pos, scale in enumerate(scales):
        gain_maps[scale] = _mean_signal_by_bin(
            relative_evidence[:, scale_pos],
            bin_data,
        ).reshape(gridsize, gridsize).T
        winner_fraction_by_scale[scale] = float(np.mean(winner_idx_steps == scale_pos))
        winner_mask = winner_onehot[:, scale_pos].astype(np.float32)
        segmented_hmaps[scale] = scale_hmaps[scale][:min_steps] * winner_mask[:, None]

    winner_strength = np.max(evidence_stack_steps, axis=1)
    winner_strength_map = _mean_signal_by_bin(winner_strength, bin_data).reshape(
        gridsize, gridsize
    ).T
    sorted_evidence = np.sort(evidence_stack_steps, axis=1)
    if evidence_stack_steps.shape[1] > 1:
        winner_margin = sorted_evidence[:, -1] - sorted_evidence[:, -2]
    else:
        winner_margin = sorted_evidence[:, -1]
    winner_relative_margin = winner_margin / np.maximum(winner_strength, 1e-9)
    winner_rel_margin_map = _mean_signal_by_bin(
        winner_relative_margin,
        bin_data,
    ).reshape(gridsize, gridsize).T

    winner_idx_bins = np.full(bin_data["counts_1d"].shape[0], -1, dtype=np.int32)
    winner_counts = np.zeros((bin_data["counts_1d"].shape[0], len(scales)), dtype=np.float32)
    flat_idx = bin_data["flat_idx"]
    valid_mask = bin_data["valid_mask"]
    for scale_pos in range(len(scales)):
        binc = np.bincount(
            flat_idx,
            weights=winner_onehot[valid_mask, scale_pos].astype(np.float64),
            minlength=bin_data["counts_1d"].shape[0],
        ).astype(np.float32)
        winner_counts[:, scale_pos] = binc
    valid_bins = bin_data["counts_1d"] > 0
    winner_idx_bins[valid_bins] = np.argmax(winner_counts[valid_bins], axis=1).astype(
        np.int32
    )
    winner_rgb = np.zeros((winner_idx_bins.shape[0], 3), dtype=np.float32)
    scale_colors = np.array(
        [
            [0.86, 0.24, 0.18],
            [0.18, 0.57, 0.84],
            [0.18, 0.70, 0.32],
        ],
        dtype=np.float32,
    )
    valid_winners = winner_idx_bins >= 0
    winner_rgb[valid_winners] = scale_colors[winner_idx_bins[valid_winners]]
    peak_strength = float(np.max(winner_strength_map))
    if peak_strength > 0.0:
        flat_strength = winner_strength_map.T.reshape(-1)
        winner_rgb[valid_winners] *= (
            flat_strength[valid_winners] / peak_strength
        )[:, np.newaxis]

    return {
        "scales": scales,
        "mode_label": "Relative Survivor",
        "figure_title": "Relative Survivor Segmented Scale Routing",
        "winner_map_title": "Relative Winner Map",
        "winner_strength_title": "Winner Evidence",
        "per_scale_map_title_template": "Scale {scale} Mean Relative Evidence",
        "segmented_unified_label": "Unified (Relative-Segmented)",
        "segmented_scale_label_template": "Scale {scale} (Relative-Segmented)",
        "extent": bin_data["extent"],
        "gain_maps": gain_maps,
        "winner_strength": winner_strength_map,
        "winner_relative_margin": winner_rel_margin_map,
        "winner_rgb": winner_rgb.reshape(gridsize, gridsize, 3).transpose(1, 0, 2),
        "winner_fractions": winner_fraction_by_scale,
        "segmented_hmaps": segmented_hmaps,
        "trimmed_hmap_loc": hmap_loc,
    }


def build_expression_segmented_views(
    env_name,
    hmap_loc,
    hmap_prox,
    scale_hmaps,
    expression_config,
    gridsize=200,
):
    """Build segmented diagnostics that match the saved PCN's learning expression mode."""
    if expression_config is None:
        return None
    mode = str(
        expression_config.get(
            "effective_learning_adaptation_mode",
            expression_config.get("learning_adaptation_mode", ""),
        )
    ).strip().lower()
    if mode == "relative_post_competition_expression":
        return build_relative_segmented_views(
            env_name,
            hmap_loc,
            scale_hmaps,
            gridsize=gridsize,
        )
    if hmap_prox is None:
        return None
    return build_gaussian_segmented_views(
        env_name,
        hmap_loc,
        hmap_prox,
        scale_hmaps,
        expression_config,
        gridsize=gridsize,
    )


def build_scale_winner_views(env_name, hmap_loc, scale_hmaps, gridsize=200):
    """
    Build per-bin scale winner diagnostics from post-competition scale evidence.

    For each timestep and scale, evidence is defined as the strongest cell in
    that scale (max activation across PCs). We then average that evidence into
    spatial bins and ask which scale wins each bin.
    """
    if not scale_hmaps:
        return None

    hmap_x, hmap_y = _extract_planar_xy(hmap_loc, env_name)
    min_steps = min([len(hmap_loc)] + [arr.shape[0] for arr in scale_hmaps.values()])
    if min_steps <= 0:
        return None

    hmap_x = hmap_x[:min_steps]
    hmap_y = hmap_y[:min_steps]
    bin_data = _prepare_spatial_bins(hmap_x, hmap_y, gridsize)

    scales = sorted(scale_hmaps.keys())
    evidence_maps = []
    evidence_stack_steps = []
    for scale in scales:
        arr = scale_hmaps[scale][:min_steps]
        evidence_t = np.max(arr, axis=1)
        evidence_stack_steps.append(evidence_t)
        mean_evidence = _mean_signal_by_bin(evidence_t, bin_data)
        evidence_maps.append(mean_evidence)

    evidence_stack_steps = np.stack(evidence_stack_steps, axis=1)
    winner_fracs = {}
    winner_idx_steps = np.argmax(evidence_stack_steps, axis=1)
    for idx, scale in enumerate(scales):
        winner_fracs[scale] = float(np.mean(winner_idx_steps == idx))

    evidence_maps = np.stack(evidence_maps, axis=0)
    winner_idx = np.argmax(evidence_maps, axis=0).astype(np.int32)
    winner_strength = np.max(evidence_maps, axis=0)
    sorted_evidence = np.sort(evidence_maps, axis=0)
    if evidence_maps.shape[0] > 1:
        winner_margin = sorted_evidence[-1] - sorted_evidence[-2]
    else:
        winner_margin = sorted_evidence[-1]
    winner_relative_margin = winner_margin / np.maximum(winner_strength, 1e-9)

    valid_bins = bin_data["counts_1d"] > 0
    winner_idx[~valid_bins] = -1
    winner_strength[~valid_bins] = 0.0
    winner_margin[~valid_bins] = 0.0
    winner_relative_margin[~valid_bins] = 0.0
    ambiguous_mask = (
        valid_bins
        & (winner_relative_margin < SCALE_WINNER_RELATIVE_MARGIN_THRESHOLD)
    )

    gridsize = bin_data["gridsize"]
    scale_colors = np.array(
        [
            [0.86, 0.24, 0.18],  # small
            [0.18, 0.57, 0.84],  # medium
            [0.18, 0.70, 0.32],  # large
        ],
        dtype=np.float32,
    )
    winner_rgb = np.zeros((winner_idx.shape[0], 3), dtype=np.float32)
    valid_winners = winner_idx >= 0
    winner_rgb[valid_winners] = scale_colors[winner_idx[valid_winners]]
    peak_strength = float(np.max(winner_strength))
    if peak_strength > 0.0:
        winner_rgb[valid_winners] *= (winner_strength[valid_winners] / peak_strength)[:, np.newaxis]
    winner_rgb[ambiguous_mask] = np.array([0.55, 0.55, 0.55], dtype=np.float32)

    return {
        "extent": bin_data["extent"],
        "scales": scales,
        "winner_rgb": winner_rgb.reshape(gridsize, gridsize, 3).transpose(1, 0, 2),
        "winner_strength": winner_strength.reshape(gridsize, gridsize).T,
        "winner_margin": winner_margin.reshape(gridsize, gridsize).T,
        "winner_relative_margin": winner_relative_margin.reshape(gridsize, gridsize).T,
        "ambiguous_mask": ambiguous_mask.reshape(gridsize, gridsize).T,
        "ambiguous_fraction": float(
            np.mean(ambiguous_mask[valid_bins]) if np.any(valid_bins) else 0.0
        ),
        "evidence_maps": {
            scale: evidence_maps[idx].reshape(gridsize, gridsize).T
            for idx, scale in enumerate(scales)
        },
        "winner_fractions": winner_fracs,
    }


def build_place_field_views(
    env_name,
    scale,
    hmap_loc,
    hmap_pcn,
    gridsize=200,
    cell_scores=None,
    score_meta=None,
):
    """Build honest place-field views for one source."""
    hmap_x, hmap_y = _extract_planar_xy(hmap_loc, env_name)

    peak_per_cell = np.max(hmap_pcn, axis=0)
    candidate_cells = np.where(peak_per_cell > FALLBACK_PEAK_THRESHOLD)[0].astype(np.int32)
    if len(candidate_cells) == 0:
        print(f"  WARNING: No active cells found for scale {scale}")
        return None

    bin_data = _prepare_spatial_bins(hmap_x, hmap_y, gridsize)
    formed = _analyze_formed_fields(hmap_pcn, bin_data, candidate_cells)
    formed_cells = formed["formed_cells"]
    if len(formed_cells) == 0:
        print(f"  WARNING: No formed fields found for scale {scale}")
        return None

    peak_subset = _choose_top_cells(formed_cells, formed["formed_peaks"])

    learning_subset = peak_subset
    learning_note = "top peak-formed fields"
    meta_suffix = ""
    if (
        CELL_RANKING_MODE == "learning_strength"
        and cell_scores is not None
        and len(cell_scores) == hmap_pcn.shape[1]
    ):
        learning_scores = np.asarray(cell_scores, dtype=np.float32)
        source_cells = formed_cells if LEARNING_VIEW_RESTRICT_TO_FORMED else candidate_cells
        learning_subset = _choose_top_cells(source_cells, learning_scores[source_cells])
        learning_note = (
            "top learned formed fields"
            if LEARNING_VIEW_RESTRICT_TO_FORMED
            else "top learned cells"
        )
        if score_meta:
            if score_meta.get("used_fallback"):
                meta_suffix = " [current-weight fallback]"
            else:
                meta_suffix = " [|dw_in| + |dw_grid|]"

    # --- Strong-fields-only subset (percentile filter on formed peaks) ---
    strong_threshold = float(np.percentile(formed["formed_peaks"], STRONG_FIELD_PERCENTILE))
    strong_mask = formed["formed_peaks"] >= strong_threshold
    strong_cells = formed_cells[strong_mask]
    strong_peaks = formed["formed_peaks"][strong_mask]
    strong_subset = _choose_top_cells(strong_cells, strong_peaks)

    # Alpha weights for peak overlay (normalized peak -> opacity)
    peak_alphas = None
    if len(peak_subset) > 0:
        peak_lookup = {
            int(cell): float(peak)
            for cell, peak in zip(formed_cells, formed["formed_peaks"])
        }
        peak_subset_peaks = np.array(
            [peak_lookup[int(cell)] for cell in peak_subset],
            dtype=np.float32,
        )
        max_peak = float(np.max(peak_subset_peaks))
        peak_alphas = (peak_subset_peaks / max(max_peak, 1e-6)).astype(np.float32)
        peak_alphas = np.clip(peak_alphas, 0.05, 1.0)

    print(
        f"  Formed fields: {len(formed_cells)} / {len(candidate_cells)} active cells; "
        f"peak subset={len(peak_subset)}, learning subset={len(learning_subset)} by {learning_note}{meta_suffix}; "
        f"strong fields (>p{STRONG_FIELD_PERCENTILE}): {len(strong_cells)} (threshold={strong_threshold:.3f})"
    )

    return {
        "extent": bin_data["extent"],
        "coverage_fraction": formed["coverage_fraction"],
        "coverage_mean_strength": formed["coverage_mean_strength"],
        "peak_overlay": _build_winner_overlay(hmap_pcn, bin_data, peak_subset, cell_alphas=peak_alphas),
        "learning_overlay": _build_winner_overlay(hmap_pcn, bin_data, learning_subset),
        "strong_overlay": _build_winner_overlay(hmap_pcn, bin_data, strong_subset),
        "formed_count": int(len(formed_cells)),
        "active_count": int(len(candidate_cells)),
        "peak_subset_count": int(len(peak_subset)),
        "learning_subset_count": int(len(learning_subset)),
        "strong_subset_count": int(len(strong_subset)),
        "strong_threshold": strong_threshold,
        "learning_note": learning_note,
    }


def visualize_environment(env_name):
    """Generate place field visualizations for one environment."""
    print(f"\n{'=' * 60}")
    print(f"Processing {env_name}")
    print(f"{'=' * 60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    available_scales = discover_available_scales(env_name)
    if not available_scales:
        print(f"  ERROR: No data found for {env_name}")
        return None

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

    row_labels = []
    row_payloads = []
    scale_hmaps_for_winner = {}
    score_by_scale = {}
    score_meta_by_scale = {}
    reference_hmap_loc = None

    if show_unified:
        print("\n  Unified aggregate:")
        hmap_loc_u, hmap_pcn_u = load_unified_aggregate_data(env_name, available_scales)
        score_u, score_meta_u = load_place_cell_scores(env_name, scale=None, unified=True)
        result_u = None
        if hmap_loc_u is not None and hmap_pcn_u is not None:
            result_u = build_place_field_views(
                env_name,
                "unified",
                hmap_loc_u,
                hmap_pcn_u,
                cell_scores=score_u,
                score_meta=score_meta_u,
            )
        row_labels.append("Unified (All Scales)")
        row_payloads.append(result_u)

    for scale in available_scales:
        print(f"\n  Scale {scale}:")
        hmap_loc, hmap_pcn = load_hmap_data(env_name, scale)
        if reference_hmap_loc is None and hmap_loc is not None:
            reference_hmap_loc = hmap_loc
        if hmap_pcn is not None:
            scale_hmaps_for_winner[scale] = hmap_pcn
        score_scale, score_meta_scale = load_place_cell_scores(
            env_name,
            scale=scale,
            unified=unified_detected,
        )
        score_by_scale[scale] = score_scale
        score_meta_by_scale[scale] = score_meta_scale
        result = None
        if hmap_loc is not None and hmap_pcn is not None:
            result = build_place_field_views(
                env_name,
                scale,
                hmap_loc,
                hmap_pcn,
                cell_scores=score_scale,
                score_meta=score_meta_scale,
            )
        row_labels.append(f"Scale {scale}")
        row_payloads.append(result)

    num_rows = len(row_labels)
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 6 * num_rows))
    if num_rows == 1:
        axes = np.asarray([axes])

    fig.suptitle(f"{env_name} - Formed Place Fields", fontsize=16)

    for row_idx, (label, result) in enumerate(zip(row_labels, row_payloads)):
        ax_cov, ax_peak, ax_learn, ax_strong = axes[row_idx]

        if result is None:
            for ax in (ax_cov, ax_peak, ax_learn, ax_strong):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            ax_cov.set_title(f"{label} - No formed fields", fontsize=13)
            continue

        extent = result["extent"]

        im_cov = ax_cov.imshow(
            result["coverage_fraction"],
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=max(float(np.max(result["coverage_fraction"])), 1e-6),
        )
        ax_cov.set_title(
            f"{label}\nCoverage Fraction ({result['formed_count']} formed / {result['active_count']} active)",
            fontsize=13,
        )
        ax_cov.set_xlabel("X Position (m)", fontsize=11)
        ax_cov.set_ylabel("Y Position (m)", fontsize=11)
        ax_cov.grid(True, alpha=0.2)
        fig.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)

        if result["peak_overlay"] is not None:
            ax_peak.imshow(result["peak_overlay"], extent=extent, origin="lower")
        else:
            ax_peak.text(0.5, 0.5, "No peak overlay", ha="center", va="center", transform=ax_peak.transAxes)
        ax_peak.set_title(
            f"{label}\nTop Peak Fields, Alpha-Weighted (n={result['peak_subset_count']})",
            fontsize=13,
        )
        ax_peak.set_xlabel("X Position (m)", fontsize=11)
        ax_peak.set_ylabel("Y Position (m)", fontsize=11)
        ax_peak.grid(True, alpha=0.2)

        if result["learning_overlay"] is not None:
            ax_learn.imshow(result["learning_overlay"], extent=extent, origin="lower")
        else:
            ax_learn.text(0.5, 0.5, "No learning overlay", ha="center", va="center", transform=ax_learn.transAxes)
        ax_learn.set_title(
            f"{label}\n{result['learning_note'].title()} (n={result['learning_subset_count']})",
            fontsize=13,
        )
        ax_learn.set_xlabel("X Position (m)", fontsize=11)
        ax_learn.set_ylabel("Y Position (m)", fontsize=11)
        ax_learn.grid(True, alpha=0.2)

        if result.get("strong_overlay") is not None:
            ax_strong.imshow(result["strong_overlay"], extent=extent, origin="lower")
        else:
            ax_strong.text(0.5, 0.5, "No strong fields", ha="center", va="center", transform=ax_strong.transAxes)
        ax_strong.set_title(
            f"{label}\nStrong Fields Only (n={result.get('strong_subset_count', 0)}, "
            f">p{STRONG_FIELD_PERCENTILE}, thr={result.get('strong_threshold', 0):.3f})",
            fontsize=13,
        )
        ax_strong.set_xlabel("X Position (m)", fontsize=11)
        ax_strong.set_ylabel("Y Position (m)", fontsize=11)
        ax_strong.grid(True, alpha=0.2)

    output_file = OUTPUT_DIR / f"{env_name}_place_fields.png"
    plt.tight_layout()
    try:
        plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    except PermissionError:
        alt_dir = DATA_BASE_DIR / env_name / "vis_outputs" / "place_fields"
        alt_dir.mkdir(parents=True, exist_ok=True)
        alt_output = alt_dir / f"{env_name}_place_fields_latest.png"
        plt.savefig(str(alt_output), dpi=150, bbox_inches="tight")
        output_file = alt_output

    if output_file.exists():
        file_size = output_file.stat().st_size / 1024
        print(f"\n  [OK] Saved: {output_file}")
        print(f"       File size: {file_size:.1f} KB")
    else:
        print(f"\n  [ERROR] Failed to save {output_file}")

    plt.close()

    winner_output = None
    segmented_output = None
    segmented_winner_output = None
    if scale_hmaps_for_winner and reference_hmap_loc is not None:
        winner_views = build_scale_winner_views(
            env_name,
            reference_hmap_loc,
            scale_hmaps_for_winner,
        )
        if winner_views is not None:
            fig_w, axes_w = plt.subplots(2, 3, figsize=(18, 10))
            fig_w.suptitle(f"{env_name} - Scale Winner Diagnostics", fontsize=16)
            extent = winner_views["extent"]

            ax = axes_w[0, 0]
            ax.imshow(winner_views["winner_rgb"], extent=extent, origin="lower")
            ax.set_title("Scale Winner Map", fontsize=13)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.grid(True, alpha=0.2)
            legend_lines = [
                f"Scale {scale}: {winner_views['winner_fractions'][scale] * 100:.1f}% step winners"
                for scale in winner_views["scales"]
            ]
            legend_lines.append(
                f"Ambiguous bins: {winner_views['ambiguous_fraction'] * 100:.1f}%"
            )
            ax.text(
                0.02,
                0.02,
                "\n".join(legend_lines),
                transform=ax.transAxes,
                fontsize=10,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"),
            )

            ax = axes_w[0, 1]
            im = ax.imshow(
                winner_views["winner_strength"],
                extent=extent,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=max(float(np.max(winner_views["winner_strength"])), 1e-6),
            )
            ax.set_title("Winning Evidence", fontsize=13)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.grid(True, alpha=0.2)
            fig_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes_w[0, 2]
            im = ax.imshow(
                winner_views["winner_relative_margin"],
                extent=extent,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=max(float(np.max(winner_views["winner_relative_margin"])), 1e-6),
            )
            ax.set_title("Winner Relative Margin", fontsize=13)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.grid(True, alpha=0.2)
            fig_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for col_idx, scale in enumerate(winner_views["scales"]):
                ax = axes_w[1, col_idx]
                evidence_map = winner_views["evidence_maps"][scale]
                im = ax.imshow(
                    evidence_map,
                    extent=extent,
                    origin="lower",
                    cmap="plasma",
                    vmin=0.0,
                    vmax=max(float(np.max(evidence_map)), 1e-6),
                )
                ax.set_title(f"Scale {scale} Mean Evidence", fontsize=13)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.2)
                fig_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            winner_output = OUTPUT_DIR / f"{env_name}_scale_winners.png"
            plt.tight_layout()
            try:
                plt.savefig(str(winner_output), dpi=150, bbox_inches="tight")
            except PermissionError:
                alt_dir = DATA_BASE_DIR / env_name / "vis_outputs" / "place_fields"
                alt_dir.mkdir(parents=True, exist_ok=True)
                winner_output = alt_dir / f"{env_name}_scale_winners_latest.png"
                plt.savefig(str(winner_output), dpi=150, bbox_inches="tight")
            plt.close(fig_w)
            if winner_output.exists():
                print(f"  [OK] Saved: {winner_output}")

        hmap_prox = load_hmap_prox(env_name)
        expression_config = load_unified_expression_config(env_name)
        if unified_detected and expression_config is not None:
            segmented_views = build_expression_segmented_views(
                env_name,
                reference_hmap_loc,
                hmap_prox,
                scale_hmaps_for_winner,
                expression_config,
            )
            if segmented_views is not None:
                seg_extent = segmented_views["extent"]

                fig_seg_w, axes_seg_w = plt.subplots(2, 3, figsize=(18, 10))
                fig_seg_w.suptitle(
                    f"{env_name} - {segmented_views['figure_title']}",
                    fontsize=16,
                )

                ax = axes_seg_w[0, 0]
                ax.imshow(segmented_views["winner_rgb"], extent=seg_extent, origin="lower")
                ax.set_title(segmented_views["winner_map_title"], fontsize=13)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.2)
                legend_lines = [
                    f"Scale {scale}: {segmented_views['winner_fractions'][scale] * 100:.1f}% routed winners"
                    for scale in segmented_views["scales"]
                ]
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(legend_lines),
                    transform=ax.transAxes,
                    fontsize=10,
                    color="white",
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"),
                )

                ax = axes_seg_w[0, 1]
                im = ax.imshow(
                    segmented_views["winner_strength"],
                    extent=seg_extent,
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=max(float(np.max(segmented_views["winner_strength"])), 1e-6),
                )
                ax.set_title(segmented_views["winner_strength_title"], fontsize=13)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.2)
                fig_seg_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax = axes_seg_w[0, 2]
                im = ax.imshow(
                    segmented_views["winner_relative_margin"],
                    extent=seg_extent,
                    origin="lower",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=max(
                        float(np.max(segmented_views["winner_relative_margin"])),
                        1e-6,
                    ),
                )
                ax.set_title("Winner Relative Margin", fontsize=13)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.2)
                fig_seg_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                for col_idx, scale in enumerate(segmented_views["scales"]):
                    ax = axes_seg_w[1, col_idx]
                    gain_map = segmented_views["gain_maps"][scale]
                    im = ax.imshow(
                        gain_map,
                        extent=seg_extent,
                        origin="lower",
                        cmap="plasma",
                        vmin=0.0,
                        vmax=max(float(np.max(gain_map)), 1e-6),
                    )
                    ax.set_title(
                        segmented_views["per_scale_map_title_template"].format(scale=scale),
                        fontsize=13,
                    )
                    ax.set_xlabel("X Position (m)")
                    ax.set_ylabel("Y Position (m)")
                    ax.grid(True, alpha=0.2)
                    fig_seg_w.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                segmented_winner_output = OUTPUT_DIR / f"{env_name}_segmented_scale_winners.png"
                plt.tight_layout()
                try:
                    plt.savefig(str(segmented_winner_output), dpi=150, bbox_inches="tight")
                except PermissionError:
                    alt_dir = DATA_BASE_DIR / env_name / "vis_outputs" / "place_fields"
                    alt_dir.mkdir(parents=True, exist_ok=True)
                    segmented_winner_output = (
                        alt_dir / f"{env_name}_segmented_scale_winners_latest.png"
                    )
                    plt.savefig(str(segmented_winner_output), dpi=150, bbox_inches="tight")
                plt.close(fig_seg_w)
                if segmented_winner_output.exists():
                    print(f"  [OK] Saved: {segmented_winner_output}")

                segmented_labels = []
                segmented_payloads = []
                segmented_loc = segmented_views["trimmed_hmap_loc"]

                segmented_unified = np.concatenate(
                    [
                        segmented_views["segmented_hmaps"][scale]
                        for scale in segmented_views["scales"]
                    ],
                    axis=1,
                )
                segmented_score_u, segmented_score_meta_u = load_place_cell_scores(
                    env_name,
                    scale=None,
                    unified=True,
                )
                segmented_unified_result = build_place_field_views(
                    env_name,
                    "segmented_unified",
                    segmented_loc,
                    segmented_unified,
                    cell_scores=segmented_score_u,
                    score_meta=segmented_score_meta_u,
                )
                segmented_labels.append(segmented_views["segmented_unified_label"])
                segmented_payloads.append(segmented_unified_result)

                for scale in segmented_views["scales"]:
                    segmented_result = build_place_field_views(
                        env_name,
                        f"segmented_{scale}",
                        segmented_loc,
                        segmented_views["segmented_hmaps"][scale],
                        cell_scores=score_by_scale.get(scale),
                        score_meta=score_meta_by_scale.get(scale),
                    )
                    segmented_labels.append(
                        segmented_views["segmented_scale_label_template"].format(
                            scale=scale
                        )
                    )
                    segmented_payloads.append(segmented_result)

                fig_seg_pf, axes_seg_pf = plt.subplots(
                    len(segmented_labels),
                    4,
                    figsize=(28, 6 * len(segmented_labels)),
                )
                if len(segmented_labels) == 1:
                    axes_seg_pf = np.asarray([axes_seg_pf])

                fig_seg_pf.suptitle(
                    f"{env_name} - {segmented_views['mode_label']} Segmented Place Fields",
                    fontsize=16,
                )

                for row_idx, (label, result) in enumerate(
                    zip(segmented_labels, segmented_payloads)
                ):
                    ax_cov, ax_peak, ax_learn, ax_strong = axes_seg_pf[row_idx]

                    if result is None:
                        for ax in (ax_cov, ax_peak, ax_learn, ax_strong):
                            ax.text(
                                0.5,
                                0.5,
                                "No segmented fields",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_xticks([])
                            ax.set_yticks([])
                        ax_cov.set_title(f"{label} - No formed fields", fontsize=13)
                        continue

                    extent = result["extent"]

                    im_cov = ax_cov.imshow(
                        result["coverage_fraction"],
                        extent=extent,
                        origin="lower",
                        cmap="magma",
                        vmin=0.0,
                        vmax=max(float(np.max(result["coverage_fraction"])), 1e-6),
                    )
                    ax_cov.set_title(
                        f"{label}\nCoverage Fraction ({result['formed_count']} formed / {result['active_count']} active)",
                        fontsize=13,
                    )
                    ax_cov.set_xlabel("X Position (m)", fontsize=11)
                    ax_cov.set_ylabel("Y Position (m)", fontsize=11)
                    ax_cov.grid(True, alpha=0.2)
                    fig_seg_pf.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)

                    if result["peak_overlay"] is not None:
                        ax_peak.imshow(result["peak_overlay"], extent=extent, origin="lower")
                    else:
                        ax_peak.text(
                            0.5,
                            0.5,
                            "No peak overlay",
                            ha="center",
                            va="center",
                            transform=ax_peak.transAxes,
                        )
                    ax_peak.set_title(
                        f"{label}\nTop Peak-Formed Fields (n={result['peak_subset_count']})",
                        fontsize=13,
                    )
                    ax_peak.set_xlabel("X Position (m)", fontsize=11)
                    ax_peak.set_ylabel("Y Position (m)", fontsize=11)
                    ax_peak.grid(True, alpha=0.2)

                    if result["learning_overlay"] is not None:
                        ax_learn.imshow(
                            result["learning_overlay"],
                            extent=extent,
                            origin="lower",
                        )
                    else:
                        ax_learn.text(
                            0.5,
                            0.5,
                            "No learning overlay",
                            ha="center",
                            va="center",
                            transform=ax_learn.transAxes,
                        )
                    ax_learn.set_title(
                        f"{label}\n{result['learning_note'].title()} (n={result['learning_subset_count']})",
                        fontsize=13,
                    )
                    ax_learn.set_xlabel("X Position (m)", fontsize=11)
                    ax_learn.set_ylabel("Y Position (m)", fontsize=11)
                    ax_learn.grid(True, alpha=0.2)

                segmented_output = OUTPUT_DIR / f"{env_name}_segmented_place_fields.png"
                plt.tight_layout()
                try:
                    plt.savefig(str(segmented_output), dpi=150, bbox_inches="tight")
                except PermissionError:
                    alt_dir = DATA_BASE_DIR / env_name / "vis_outputs" / "place_fields"
                    alt_dir.mkdir(parents=True, exist_ok=True)
                    segmented_output = (
                        alt_dir / f"{env_name}_segmented_place_fields_latest.png"
                    )
                    plt.savefig(str(segmented_output), dpi=150, bbox_inches="tight")
                plt.close(fig_seg_pf)
                if segmented_output.exists():
                    print(f"  [OK] Saved: {segmented_output}")

    created = []
    for candidate in (
        output_file,
        winner_output,
        segmented_winner_output,
        segmented_output,
    ):
        if candidate is not None and Path(candidate).exists():
            created.append(candidate)

    if created:
        return created
    return []


def main():
    """Main execution."""
    environments = resolve_environments()
    print("=" * 60)
    print("Place Field Visualization")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Controller: {CONTROLLER_NAME}")
    print(f"Environments: {len(environments)}")
    print(f"Architecture mode: {ARCHITECTURE}")
    print(f"Cell ranking mode: {CELL_RANKING_MODE}")

    if not environments:
        print("\nERROR: No environments with hmap data were found.")
        return

    created_files = []
    for env_name in environments:
        output_files = visualize_environment(env_name)
        if output_files:
            created_files.extend(output_files)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print("=" * 60)

    if created_files:
        print("\nCreated files:")
        for f in created_files:
            print(f"  - {Path(f).name}")
    else:
        print("\nWARNING: No output files were created.")


if __name__ == "__main__":
    main()
