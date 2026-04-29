"""Generate place-field plots from saved PCN activity histories."""

import argparse
import json
import os
import pickle
import sys
from math import ceil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "webots" / "controllers" / "multiscale_grid_controller"
PKL_ROOT = CONTROLLER_DIR / "pkl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.robot.webots_worlds import list_available_worlds

    SUPPORTED_WORLDS = set(list_available_worlds())
except Exception:
    SUPPORTED_WORLDS = {
        "20x20",
        "20x20_1obstacle",
        "20x20_2obstacles",
        "20x20_goalBehindWall",
    }


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_world(requested: str | None) -> str:
    if requested:
        return requested
    if not PKL_ROOT.exists():
        raise FileNotFoundError(f"No pkl directory found at {PKL_ROOT}")
    candidates = []
    for world_dir in PKL_ROOT.iterdir():
        loc_path = world_dir / "hmaps" / "hmap_loc.pkl"
        if world_dir.name in SUPPORTED_WORLDS and loc_path.exists():
            candidates.append((loc_path.stat().st_mtime, world_dir.name))
    if not candidates:
        raise FileNotFoundError(f"No supported hmap data found under {PKL_ROOT}")
    return sorted(candidates, reverse=True)[0][1]


def world_dirs(world: str):
    base = PKL_ROOT / world
    return base / "hmaps", base / "vis_outputs" / "place_fields"


def load_locations(hmap_dir: Path) -> np.ndarray:
    loc = _as_numpy(_load_pickle(hmap_dir / "hmap_loc.pkl"))
    if loc.ndim != 2 or loc.shape[1] < 2:
        raise ValueError(f"Bad hmap_loc shape: {loc.shape}")
    return loc[:, :2]


def discover_scales(hmap_dir: Path) -> list[int]:
    scales = []
    for path in sorted(hmap_dir.glob("hmap_pcn_scale_*.pkl")):
        try:
            scales.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return scales


def load_scale_hmaps(hmap_dir: Path) -> dict[int, np.ndarray]:
    return {
        scale: _as_numpy(_load_pickle(hmap_dir / f"hmap_pcn_scale_{scale}.pkl"))
        for scale in discover_scales(hmap_dir)
    }


def align(xy: np.ndarray, values: np.ndarray):
    n = min(len(xy), len(values))
    xy = xy[:n]
    values = values[:n]
    good = np.isfinite(xy).all(axis=1) & np.isfinite(values).all(axis=1)
    return xy[good], values[good]


def draw_hex(ax, xy: np.ndarray, values: np.ndarray, title: str, bins: int, cmap: str = "viridis"):
    hb = ax.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=values,
        gridsize=bins,
        reduce_C_function=np.mean,
        mincnt=1,
        linewidths=0.0,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    return hb


def top_cells(activity: np.ndarray, top_k: int, min_peak: float):
    peaks = np.nanmax(activity, axis=0)
    candidates = np.where(peaks >= float(min_peak))[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), peaks
    order = candidates[np.argsort(peaks[candidates])[::-1]]
    return order[: int(top_k)], peaks


def field_quality(xy: np.ndarray, activity: np.ndarray, min_peak: float, bins: int):
    xy, activity = align(xy, activity)
    if len(xy) == 0 or activity.size == 0:
        return {"cell_count": 0, "active_cell_count": 0}
    peaks = np.nanmax(activity, axis=0)
    active = np.where(peaks >= float(min_peak))[0]
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    xi = np.clip(np.searchsorted(xedges, xy[:, 0], side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(yedges, xy[:, 1], side="right") - 1, 0, bins - 1)
    flat = yi * bins + xi
    qualities = []
    for cell in active:
        values = activity[:, cell]
        sums = np.zeros(bins * bins, dtype=np.float32)
        counts = np.bincount(flat, minlength=bins * bins).astype(np.float32)
        np.add.at(sums, flat, values.astype(np.float32, copy=False))
        means = np.divide(sums, np.maximum(counts, 1.0), out=np.zeros_like(sums), where=counts > 0)
        peak = float(np.max(means))
        if peak <= 1e-9:
            continue
        support = means >= 0.35 * peak
        peak_idx = int(np.argmax(means))
        peak_y, peak_x = divmod(peak_idx, bins)
        coords = np.column_stack(np.nonzero(support.reshape(bins, bins)))
        if len(coords) > 0:
            spread = float(np.mean(np.sqrt((coords[:, 0] - peak_y) ** 2 + (coords[:, 1] - peak_x) ** 2)))
        else:
            spread = 0.0
        sorted_means = np.sort(means[counts > 0])[::-1]
        second = float(sorted_means[1]) if len(sorted_means) > 1 else 0.0
        qualities.append(
            {
                "cell": int(cell),
                "peak": float(peaks[cell]),
                "support_bins": int(np.sum(support)),
                "compactness_proxy": float(1.0 / (1.0 + spread)),
                "peak_to_second_bin_ratio": float(peak / max(second, 1e-9)),
            }
        )
    return {
        "cell_count": int(activity.shape[1]),
        "active_cell_count": int(len(active)),
        "mean_peak": float(np.mean(peaks)) if peaks.size else 0.0,
        "max_peak": float(np.max(peaks)) if peaks.size else 0.0,
        "mean_compactness_proxy": float(np.mean([q["compactness_proxy"] for q in qualities])) if qualities else 0.0,
        "mean_support_bins": float(np.mean([q["support_bins"] for q in qualities])) if qualities else 0.0,
        "top_quality_cells": sorted(qualities, key=lambda q: q["peak"], reverse=True)[:20],
    }


def plot_scale_fields(world: str, scale: int, xy: np.ndarray, activity: np.ndarray, output_dir: Path, top_k: int, bins: int, min_peak: float):
    xy, activity = align(xy, activity)
    cells, peaks = top_cells(activity, top_k, min_peak)
    if len(cells) == 0:
        return None, {"cells": [], "peak_max": float(np.nanmax(peaks)) if peaks.size else 0.0}

    cols = min(4, len(cells))
    rows = ceil(len(cells) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows), squeeze=False)
    vmax = float(np.nanmax(activity[:, cells])) if len(cells) else 1.0
    for ax, cell in zip(axes.ravel(), cells):
        hb = draw_hex(
            ax,
            xy,
            activity[:, cell],
            f"cell {int(cell)} peak={peaks[cell]:.3f}",
            bins,
            cmap="magma",
        )
        hb.set_clim(0.0, vmax)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle(f"{world}: scale {scale} strongest place fields")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"place_fields_scale_{scale}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path, {
        "cells": [int(c) for c in cells],
        "peaks": [float(peaks[c]) for c in cells],
        "peak_max": float(np.nanmax(peaks)) if peaks.size else 0.0,
    }


def plot_coverage(world: str, xy: np.ndarray, scale_hmaps: dict[int, np.ndarray], output_dir: Path, bins: int):
    scales = sorted(scale_hmaps)
    fig, axes = plt.subplots(2, len(scales), figsize=(4.8 * len(scales), 8.0), squeeze=False)
    for col, scale in enumerate(scales):
        aligned_xy, activity = align(xy, scale_hmaps[scale])
        max_values = np.max(activity, axis=1)
        mean_values = np.mean(activity, axis=1)
        hb0 = draw_hex(axes[0, col], aligned_xy, max_values, f"scale {scale}: max PCN", bins)
        hb1 = draw_hex(axes[1, col], aligned_xy, mean_values, f"scale {scale}: mean PCN", bins)
        fig.colorbar(hb0, ax=axes[0, col], fraction=0.046, pad=0.04)
        fig.colorbar(hb1, ax=axes[1, col], fraction=0.046, pad=0.04)
    fig.suptitle(f"{world}: place-cell coverage")
    fig.tight_layout()
    out_path = output_dir / "place_field_coverage.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None, help="World name. Defaults to newest supported run.")
    parser.add_argument("--top-k", type=int, default=12, help="Top cells per scale.")
    parser.add_argument("--bins", type=int, default=70, help="Hexbin grid size.")
    parser.add_argument("--min-peak", type=float, default=0.02, help="Minimum cell peak activation.")
    parser.add_argument("--scale", type=int, default=None, help="Optional single scale to plot.")
    args = parser.parse_args()

    world = resolve_world(args.world)
    hmap_dir, output_dir = world_dirs(world)
    xy = load_locations(hmap_dir)
    scale_hmaps = load_scale_hmaps(hmap_dir)
    if args.scale is not None:
        scale_hmaps = {args.scale: scale_hmaps[args.scale]} if args.scale in scale_hmaps else {}
    if not scale_hmaps:
        raise FileNotFoundError(f"No matching hmap_pcn_scale_*.pkl files found in {hmap_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"world": world, "scales": {}}
    coverage = plot_coverage(world, xy, scale_hmaps, output_dir, args.bins)
    for scale, activity in sorted(scale_hmaps.items()):
        out_path, info = plot_scale_fields(
            world,
            scale,
            xy,
            activity,
            output_dir,
            args.top_k,
            args.bins,
            args.min_peak,
        )
        info["output"] = str(out_path) if out_path else None
        info["quality"] = field_quality(xy, activity, args.min_peak, args.bins)
        summary["scales"][str(scale)] = info
    summary["coverage_output"] = str(coverage)
    with open(output_dir / "place_field_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved place-field outputs to {output_dir}")


if __name__ == "__main__":
    main()
