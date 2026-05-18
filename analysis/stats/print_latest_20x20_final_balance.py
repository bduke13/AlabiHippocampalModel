from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PKL_ROOT = REPO_ROOT / "webots" / "controllers" / "multiscale_grid_controller" / "pkl"


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_latest_scale_diagnostics(root: Path, world_filter: str) -> Path:
    candidates = [
        path
        for path in root.rglob("scale_diagnostics.pkl")
        if world_filter.lower() in str(path).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No scale_diagnostics.pkl files found under {root} matching '{world_filter}'."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_scale_configs(diag_path: Path) -> list[dict[str, Any]]:
    network_path = diag_path.parent.parent / "networks" / "unified_pcn.pkl"
    if not network_path.exists():
        return []
    pcn = load_pickle(network_path)
    scale_configs = getattr(pcn, "scale_configs", None)
    return [dict(cfg) for cfg in scale_configs] if isinstance(scale_configs, list) else []


def load_hmap_locations(diag_path: Path) -> np.ndarray:
    loc_path = diag_path.parent / "hmap_loc.pkl"
    loc = load_pickle(loc_path)
    arr = np.asarray(loc, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad hmap_loc shape: {arr.shape}")
    return arr[:, :2]


def load_scale_activity_history(diag_path: Path, scale_idx: int) -> np.ndarray:
    path = diag_path.parent / f"hmap_pcn_scale_{scale_idx}.pkl"
    arr = np.asarray(load_pickle(path), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Bad hmap_pcn_scale_{scale_idx} shape: {arr.shape}")
    return arr


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _scale_name(scale_idx: int, scale_configs: list[dict[str, Any]]) -> str:
    scale_cfg = scale_configs[scale_idx] if scale_idx < len(scale_configs) else {}
    return str(scale_cfg.get("name", f"scale{scale_idx}"))


def _scale_alphas(scale_idx: int, scale_configs: list[dict[str, Any]]) -> tuple[float, float]:
    scale_cfg = scale_configs[scale_idx] if scale_idx < len(scale_configs) else {}
    alpha_pb = float(scale_cfg.get("alpha_pb", math.sqrt(0.5)))
    alpha_pg = float(scale_cfg.get("alpha_pg", math.sqrt(0.5)))
    return alpha_pb, alpha_pg


def _record_context_gates(record: dict[str, Any], scale_count: int) -> np.ndarray:
    values = record.get("bvc_context_gain", record.get("scale_preference", []))
    if not isinstance(values, (list, tuple, np.ndarray)):
        return np.zeros(scale_count, dtype=np.float32)
    gates = np.zeros(scale_count, dtype=np.float32)
    n = min(scale_count, len(values))
    if n > 0:
        gates[:n] = np.nan_to_num(np.asarray(values[:n], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(gates, 0.0, None)


def _align_xy_activity(xy: np.ndarray, activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(xy), len(activity))
    xy = xy[:n]
    activity = activity[:n]
    good = np.isfinite(xy).all(axis=1) & np.isfinite(activity).all(axis=1)
    return xy[good], activity[good]


def _binned_mean_activity(xy: np.ndarray, activity: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    xi = np.clip(np.searchsorted(xedges, xy[:, 0], side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(yedges, xy[:, 1], side="right") - 1, 0, bins - 1)
    flat = yi * bins + xi

    counts = np.bincount(flat, minlength=bins * bins).astype(np.float32)
    sums = np.zeros((bins * bins, activity.shape[1]), dtype=np.float32)
    np.add.at(sums, flat, activity.astype(np.float32, copy=False))
    means = np.divide(
        sums,
        np.maximum(counts[:, None], 1.0),
        out=np.zeros_like(sums),
        where=counts[:, None] > 0,
    )
    cell_area = float(((xmax - xmin) / max(1, bins)) * ((ymax - ymin) / max(1, bins)))
    return means.reshape(bins, bins, activity.shape[1]), counts.reshape(bins, bins), cell_area


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while queue:
                cur_row, cur_col = queue.popleft()
                component.append((cur_row, cur_col))
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if 0 <= nxt_row < height and 0 <= nxt_col < width:
                        if mask[nxt_row, nxt_col] and not visited[nxt_row, nxt_col]:
                            visited[nxt_row, nxt_col] = True
                            queue.append((nxt_row, nxt_col))
            components.append(component)
    return components


def compute_field_diagnostics(
    diag_path: Path,
    scale_idx: int,
    final_record: dict[str, Any],
    bins: int,
    min_peak: float,
    field_threshold_fraction: float,
    eps: float,
) -> dict[str, float]:
    xy = load_hmap_locations(diag_path)
    activity = load_scale_activity_history(diag_path, scale_idx)
    xy, activity = _align_xy_activity(xy, activity)
    if len(xy) == 0 or activity.size == 0:
        return {
            "active_pcs_per_location": 0.0,
            "unique_pcs_recruited": 0.0,
            "median_field_area": 0.0,
            "multi_lobed_count": 0.0,
            "peak_background_ratio_median": 0.0,
            "winner_margin_mean": 0.0,
            "winner_ratio_mean": 0.0,
        }

    top2 = np.partition(activity, kth=max(0, activity.shape[1] - 2), axis=1)[:, -2:]
    top1 = np.max(top2, axis=1)
    top2_vals = np.min(top2, axis=1)
    valid_winner = top1 > float(min_peak)

    field_maps, occupancy, cell_area = _binned_mean_activity(xy, activity, bins)
    occupied = occupancy > 0
    peaks = np.max(field_maps, axis=(0, 1))
    active_cells = np.where(peaks >= float(min_peak))[0]
    field_areas: list[float] = []
    peak_bg_ratios: list[float] = []
    multi_lobed_count = 0

    for cell_idx in active_cells:
        field_map = field_maps[:, :, cell_idx]
        peak = float(np.max(field_map))
        if peak <= float(eps):
            continue
        support = occupied & (field_map >= float(field_threshold_fraction) * peak)
        if not np.any(support):
            continue
        components = _connected_components(support)
        peak_row, peak_col = np.unravel_index(int(np.argmax(field_map)), field_map.shape)
        main_component = next(
            (component for component in components if (peak_row, peak_col) in component),
            components[0],
        )
        main_mask = np.zeros_like(support, dtype=bool)
        for row, col in main_component:
            main_mask[row, col] = True
        field_areas.append(float(np.sum(main_mask)) * cell_area)
        background_mask = occupied & (~main_mask)
        background_values = field_map[background_mask]
        background_mean = float(np.mean(background_values)) if background_values.size else 0.0
        peak_bg_ratios.append(peak / max(background_mean, float(eps)))
        if len(components) > 1:
            multi_lobed_count += 1

    unique_recruited = 0.0
    recruited_counts = final_record.get("unique_recruited_pc_counts", [])
    if scale_idx < len(recruited_counts):
        unique_recruited = _safe_float(recruited_counts[scale_idx])

    return {
        "active_pcs_per_location": float(np.mean(np.sum(activity > 0.0, axis=1))),
        "unique_pcs_recruited": unique_recruited,
        "median_field_area": float(np.median(np.asarray(field_areas, dtype=np.float32))) if field_areas else 0.0,
        "multi_lobed_count": float(multi_lobed_count),
        "peak_background_ratio_median": (
            float(np.median(np.asarray(peak_bg_ratios, dtype=np.float32))) if peak_bg_ratios else 0.0
        ),
        "winner_margin_mean": float(np.mean(top1[valid_winner] - top2_vals[valid_winner])) if np.any(valid_winner) else 0.0,
        "winner_ratio_mean": (
            float(np.mean(top1[valid_winner] / np.maximum(top2_vals[valid_winner], float(eps))))
            if np.any(valid_winner)
            else 0.0
        ),
    }


def summarize_final_balance(
    diag_path: Path,
    field_bins: int,
    min_peak: float,
    field_threshold_fraction: float,
    eps: float,
    context_gate_threshold: float,
) -> str:
    diagnostics = load_pickle(diag_path)
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError(f"No diagnostics found in {diag_path}.")

    final_record = diagnostics[-1]
    grid_per_scale = final_record.get("grid", {}).get("per_scale", [])
    comp_per_scale = final_record.get("competition", {}).get("per_scale", [])
    scale_configs = load_scale_configs(diag_path)

    scale_count = min(len(grid_per_scale), len(comp_per_scale))
    if scale_configs:
        scale_count = min(scale_count, len(scale_configs))
    if scale_count <= 0:
        raise ValueError(f"No per-scale diagnostics found in {diag_path}.")

    run_avg_stats = [
        {
            "bvc_exc": [],
            "grid_exc": [],
            "bvc_inh": [],
            "grid_inh": [],
            "recurrent_inh": [],
            "cross_inh": [],
            "activation_mean": [],
            "oja_bvc_update_norm": [],
            "oja_gc_update_norm": [],
        }
        for _ in range(scale_count)
    ]
    context_avg_stats = [
        {
            "bvc_exc": [],
            "grid_exc": [],
            "bvc_inh": [],
            "grid_inh": [],
            "recurrent_inh": [],
            "cross_inh": [],
            "activation_mean": [],
            "oja_bvc_update_norm": [],
            "oja_gc_update_norm": [],
            "gate": [],
        }
        for _ in range(scale_count)
    ]

    for record in diagnostics:
        record_grid_per_scale = record.get("grid", {}).get("per_scale", [])
        record_comp_per_scale = record.get("competition", {}).get("per_scale", [])
        record_activity_per_scale = record.get("scale_activity", [])
        record_oja_bvc_update_norm = record.get("oja_bvc_update_norm", [])
        record_oja_gc_update_norm = record.get("oja_gc_update_norm", [])
        record_gates = _record_context_gates(record, scale_count)
        gate_sum = float(np.sum(record_gates))
        dominant_scale = int(np.argmax(record_gates)) if gate_sum > 1e-9 else -1
        record_scale_count = min(scale_count, len(record_grid_per_scale), len(record_comp_per_scale))
        for scale_idx in range(record_scale_count):
            alpha_pb, alpha_pg = _scale_alphas(scale_idx, scale_configs)
            grid_block = record_grid_per_scale[scale_idx]
            comp_block = record_comp_per_scale[scale_idx]
            activity_block = record_activity_per_scale[scale_idx] if scale_idx < len(record_activity_per_scale) else {}

            raw_bvc = _safe_float(grid_block.get("raw_bvc_abs_mean", 0.0))
            raw_grid = _safe_float(grid_block.get("raw_grid_abs_mean", 0.0))
            bvc_mod = _safe_float(grid_block.get("bvc_excitation_modulation", 1.0), default=1.0)

            run_avg_stats[scale_idx]["bvc_exc"].append(alpha_pb * raw_bvc * bvc_mod)
            run_avg_stats[scale_idx]["grid_exc"].append(alpha_pg * raw_grid)
            run_avg_stats[scale_idx]["bvc_inh"].append(
                _safe_float(comp_block.get("bvc_inhibition", {}).get("abs_mean", 0.0))
            )
            run_avg_stats[scale_idx]["grid_inh"].append(
                _safe_float(comp_block.get("grid_inhibition", {}).get("abs_mean", 0.0))
            )
            run_avg_stats[scale_idx]["recurrent_inh"].append(
                _safe_float(comp_block.get("within_scale_recurrent_inhibition", {}).get("abs_mean", 0.0))
            )
            run_avg_stats[scale_idx]["cross_inh"].append(
                _safe_float(comp_block.get("cross_scale_inhibition", {}).get("abs_mean", 0.0))
            )
            run_avg_stats[scale_idx]["activation_mean"].append(_safe_float(activity_block.get("mean", 0.0)))
            run_avg_stats[scale_idx]["oja_bvc_update_norm"].append(
                _safe_float(
                    record_oja_bvc_update_norm[scale_idx] if scale_idx < len(record_oja_bvc_update_norm) else 0.0
                )
            )
            run_avg_stats[scale_idx]["oja_gc_update_norm"].append(
                _safe_float(
                    record_oja_gc_update_norm[scale_idx] if scale_idx < len(record_oja_gc_update_norm) else 0.0
                )
            )
            if dominant_scale == scale_idx and float(record_gates[scale_idx]) >= float(context_gate_threshold):
                context_avg_stats[scale_idx]["bvc_exc"].append(alpha_pb * raw_bvc * bvc_mod)
                context_avg_stats[scale_idx]["grid_exc"].append(alpha_pg * raw_grid)
                context_avg_stats[scale_idx]["bvc_inh"].append(
                    _safe_float(comp_block.get("bvc_inhibition", {}).get("abs_mean", 0.0))
                )
                context_avg_stats[scale_idx]["grid_inh"].append(
                    _safe_float(comp_block.get("grid_inhibition", {}).get("abs_mean", 0.0))
                )
                context_avg_stats[scale_idx]["recurrent_inh"].append(
                    _safe_float(comp_block.get("within_scale_recurrent_inhibition", {}).get("abs_mean", 0.0))
                )
                context_avg_stats[scale_idx]["cross_inh"].append(
                    _safe_float(comp_block.get("cross_scale_inhibition", {}).get("abs_mean", 0.0))
                )
                context_avg_stats[scale_idx]["activation_mean"].append(_safe_float(activity_block.get("mean", 0.0)))
                context_avg_stats[scale_idx]["oja_bvc_update_norm"].append(
                    _safe_float(
                        record_oja_bvc_update_norm[scale_idx] if scale_idx < len(record_oja_bvc_update_norm) else 0.0
                    )
                )
                context_avg_stats[scale_idx]["oja_gc_update_norm"].append(
                    _safe_float(
                        record_oja_gc_update_norm[scale_idx] if scale_idx < len(record_oja_gc_update_norm) else 0.0
                    )
                )
                context_avg_stats[scale_idx]["gate"].append(float(record_gates[scale_idx]))

    run_avg_parts: list[str] = []
    context_avg_parts: list[str] = []
    final_parts: list[str] = []
    field_parts: list[str] = []
    final_activity_per_scale = final_record.get("scale_activity", [])
    final_oja_bvc_update_norm = final_record.get("oja_bvc_update_norm", [])
    final_oja_gc_update_norm = final_record.get("oja_gc_update_norm", [])

    for scale_idx in range(scale_count):
        grid_block = grid_per_scale[scale_idx]
        comp_block = comp_per_scale[scale_idx]
        scale_name = _scale_name(scale_idx, scale_configs)
        alpha_pb, alpha_pg = _scale_alphas(scale_idx, scale_configs)

        raw_bvc = _safe_float(grid_block.get("raw_bvc_abs_mean", 0.0))
        raw_grid = _safe_float(grid_block.get("raw_grid_abs_mean", 0.0))
        bvc_mod = _safe_float(grid_block.get("bvc_excitation_modulation", 1.0), default=1.0)
        bvc_exc = alpha_pb * raw_bvc * bvc_mod
        grid_exc = alpha_pg * raw_grid
        bvc_inh = _safe_float(comp_block.get("bvc_inhibition", {}).get("abs_mean", 0.0))
        grid_inh = _safe_float(comp_block.get("grid_inhibition", {}).get("abs_mean", 0.0))
        recurrent_inh = _safe_float(comp_block.get("within_scale_recurrent_inhibition", {}).get("abs_mean", 0.0))
        cross_inh = _safe_float(comp_block.get("cross_scale_inhibition", {}).get("abs_mean", 0.0))
        final_activation_mean = _safe_float(
            (final_activity_per_scale[scale_idx] if scale_idx < len(final_activity_per_scale) else {}).get("mean", 0.0)
        )
        final_oja_bvc = _safe_float(final_oja_bvc_update_norm[scale_idx] if scale_idx < len(final_oja_bvc_update_norm) else 0.0)
        final_oja_gc = _safe_float(final_oja_gc_update_norm[scale_idx] if scale_idx < len(final_oja_gc_update_norm) else 0.0)

        run_avg_parts.append(
            f"{scale_name}[{scale_idx}]: "
            f"bvc_exc={sum(run_avg_stats[scale_idx]['bvc_exc']) / max(1, len(run_avg_stats[scale_idx]['bvc_exc'])):.4f}, "
            f"grid_exc={sum(run_avg_stats[scale_idx]['grid_exc']) / max(1, len(run_avg_stats[scale_idx]['grid_exc'])):.4f}, "
            f"bvc_inh={sum(run_avg_stats[scale_idx]['bvc_inh']) / max(1, len(run_avg_stats[scale_idx]['bvc_inh'])):.4f}, "
            f"grid_inh={sum(run_avg_stats[scale_idx]['grid_inh']) / max(1, len(run_avg_stats[scale_idx]['grid_inh'])):.4f}, "
            f"recurrent_inh={sum(run_avg_stats[scale_idx]['recurrent_inh']) / max(1, len(run_avg_stats[scale_idx]['recurrent_inh'])):.4f}, "
            f"cross_inh={sum(run_avg_stats[scale_idx]['cross_inh']) / max(1, len(run_avg_stats[scale_idx]['cross_inh'])):.4f}, "
            f"activation_mean={sum(run_avg_stats[scale_idx]['activation_mean']) / max(1, len(run_avg_stats[scale_idx]['activation_mean'])):.4f}, "
            f"oja_bvc_update_norm={sum(run_avg_stats[scale_idx]['oja_bvc_update_norm']) / max(1, len(run_avg_stats[scale_idx]['oja_bvc_update_norm'])):.6f}, "
            f"oja_gc_update_norm={sum(run_avg_stats[scale_idx]['oja_gc_update_norm']) / max(1, len(run_avg_stats[scale_idx]['oja_gc_update_norm'])):.6f}"
        )
        context_avg_parts.append(
            f"{scale_name}[{scale_idx}]: "
            f"samples={len(context_avg_stats[scale_idx]['gate'])}, "
            f"mean_gate={sum(context_avg_stats[scale_idx]['gate']) / max(1, len(context_avg_stats[scale_idx]['gate'])):.4f}, "
            f"bvc_exc={sum(context_avg_stats[scale_idx]['bvc_exc']) / max(1, len(context_avg_stats[scale_idx]['bvc_exc'])):.4f}, "
            f"grid_exc={sum(context_avg_stats[scale_idx]['grid_exc']) / max(1, len(context_avg_stats[scale_idx]['grid_exc'])):.4f}, "
            f"bvc_inh={sum(context_avg_stats[scale_idx]['bvc_inh']) / max(1, len(context_avg_stats[scale_idx]['bvc_inh'])):.4f}, "
            f"grid_inh={sum(context_avg_stats[scale_idx]['grid_inh']) / max(1, len(context_avg_stats[scale_idx]['grid_inh'])):.4f}, "
            f"recurrent_inh={sum(context_avg_stats[scale_idx]['recurrent_inh']) / max(1, len(context_avg_stats[scale_idx]['recurrent_inh'])):.4f}, "
            f"cross_inh={sum(context_avg_stats[scale_idx]['cross_inh']) / max(1, len(context_avg_stats[scale_idx]['cross_inh'])):.4f}, "
            f"activation_mean={sum(context_avg_stats[scale_idx]['activation_mean']) / max(1, len(context_avg_stats[scale_idx]['activation_mean'])):.4f}, "
            f"oja_bvc_update_norm={sum(context_avg_stats[scale_idx]['oja_bvc_update_norm']) / max(1, len(context_avg_stats[scale_idx]['oja_bvc_update_norm'])):.6f}, "
            f"oja_gc_update_norm={sum(context_avg_stats[scale_idx]['oja_gc_update_norm']) / max(1, len(context_avg_stats[scale_idx]['oja_gc_update_norm'])):.6f}"
        )
        final_parts.append(
            f"{scale_name}[{scale_idx}]: "
            f"bvc_exc={bvc_exc:.4f}, "
            f"grid_exc={grid_exc:.4f}, "
            f"bvc_inh={bvc_inh:.4f}, "
            f"grid_inh={grid_inh:.4f}, "
            f"recurrent_inh={recurrent_inh:.4f}, "
            f"cross_inh={cross_inh:.4f}, "
            f"activation_mean={final_activation_mean:.4f}, "
            f"oja_bvc_update_norm={final_oja_bvc:.6f}, "
            f"oja_gc_update_norm={final_oja_gc:.6f}"
        )
        field_diag = compute_field_diagnostics(
            diag_path=diag_path,
            scale_idx=scale_idx,
            final_record=final_record,
            bins=field_bins,
            min_peak=min_peak,
            field_threshold_fraction=field_threshold_fraction,
            eps=eps,
        )
        field_parts.append(
            f"{scale_name}[{scale_idx}]: "
            f"active_pcs_per_loc={field_diag['active_pcs_per_location']:.2f}, "
            f"unique_recruited={int(round(field_diag['unique_pcs_recruited']))}, "
            f"median_field_area={field_diag['median_field_area']:.4f}, "
            f"multi_lobed={int(round(field_diag['multi_lobed_count']))}, "
            f"peak_bg_ratio_med={field_diag['peak_background_ratio_median']:.4f}, "
            f"winner_margin_mean={field_diag['winner_margin_mean']:.4f}, "
            f"winner_ratio_mean={field_diag['winner_ratio_mean']:.4f}"
        )

    latest_run_dir = diag_path.parent.parent
    return (
        f"run={latest_run_dir} | "
        f"run_avg: {' | '.join(run_avg_parts)} | "
        f"context_avg: {' | '.join(context_avg_parts)} | "
        f"final: {' | '.join(final_parts)} | "
        f"field_diag: {' | '.join(field_parts)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print one-line final excitation/inhibition summary for the latest 20x20 manual run."
    )
    parser.add_argument(
        "--pkl-root",
        type=Path,
        default=DEFAULT_PKL_ROOT,
        help="Root directory containing manual run output folders.",
    )
    parser.add_argument(
        "--world-filter",
        type=str,
        default="20x20",
        help="Substring that must appear in the run path.",
    )
    parser.add_argument("--field-bins", type=int, default=45, help="Spatial bins per axis for field diagnostics.")
    parser.add_argument("--min-peak", type=float, default=0.02, help="Minimum place-field peak for cell-level metrics.")
    parser.add_argument(
        "--context-gate-threshold",
        type=float,
        default=0.10,
        help="Minimum dominant context gate for including a sample in context-aware averages.",
    )
    parser.add_argument(
        "--field-threshold-fraction",
        type=float,
        default=0.35,
        help="Fraction of a cell's peak used to define support for field area and lobes.",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="Small constant for ratio diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diag_path = find_latest_scale_diagnostics(args.pkl_root.expanduser().resolve(), args.world_filter)
    print(
        summarize_final_balance(
            diag_path,
            field_bins=int(args.field_bins),
            min_peak=float(args.min_peak),
            field_threshold_fraction=float(args.field_threshold_fraction),
            eps=float(args.eps),
            context_gate_threshold=float(args.context_gate_threshold),
        )
    )


if __name__ == "__main__":
    main()
