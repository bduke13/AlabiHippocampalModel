"""Plot saved multiscale grid-cell activity diagnostics."""

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
    candidates = []
    for world_dir in PKL_ROOT.iterdir() if PKL_ROOT.exists() else []:
        loc_path = world_dir / "hmaps" / "hmap_loc.pkl"
        if world_dir.name in SUPPORTED_WORLDS and loc_path.exists():
            candidates.append((loc_path.stat().st_mtime, world_dir.name))
    if not candidates:
        raise FileNotFoundError(f"No supported hmap data found under {PKL_ROOT}")
    return sorted(candidates, reverse=True)[0][1]


def world_dirs(world: str):
    base = PKL_ROOT / world
    return base / "hmaps", base / "networks", base / "vis_outputs" / "grid_cells"


def load_locations(hmap_dir: Path) -> np.ndarray:
    loc = _as_numpy(_load_pickle(hmap_dir / "hmap_loc.pkl"))
    if loc.ndim != 2 or loc.shape[1] < 2:
        raise ValueError(f"Bad hmap_loc shape: {loc.shape}")
    return loc[:, :2]


def discover_scales(hmap_dir: Path) -> list[int]:
    scales = []
    for path in sorted(hmap_dir.glob("hmap_gcn_scale_*.pkl")):
        try:
            scales.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return scales


def load_gcn_hmaps(hmap_dir: Path) -> dict[int, np.ndarray]:
    return {
        scale: _as_numpy(_load_pickle(hmap_dir / f"hmap_gcn_scale_{scale}.pkl"))
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


def best_cell(activity: np.ndarray) -> int:
    return int(np.argmax(np.sum(np.abs(activity), axis=0)))


def plot_overview(world: str, xy: np.ndarray, gcn_hmaps: dict[int, np.ndarray], output_dir: Path, bins: int):
    scales = sorted(gcn_hmaps)
    fig, axes = plt.subplots(3, len(scales), figsize=(4.8 * len(scales), 11.5), squeeze=False)
    summary = {}
    for col, scale in enumerate(scales):
        aligned_xy, activity = align(xy, gcn_hmaps[scale])
        mean_values = np.mean(activity, axis=1)
        max_values = np.max(activity, axis=1)
        cell = best_cell(activity)
        hb0 = draw_hex(axes[0, col], aligned_xy, mean_values, f"scale {scale}: mean GCN", bins)
        hb1 = draw_hex(axes[1, col], aligned_xy, max_values, f"scale {scale}: max GCN", bins)
        hb2 = draw_hex(axes[2, col], aligned_xy, activity[:, cell], f"scale {scale}: cell {cell}", bins, cmap="magma")
        for hb, ax in ((hb0, axes[0, col]), (hb1, axes[1, col]), (hb2, axes[2, col])):
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        summary[str(scale)] = {
            "cells": int(activity.shape[1]),
            "samples": int(activity.shape[0]),
            "best_cell": cell,
            "mean_activation": float(np.mean(activity)),
            "max_activation": float(np.max(activity)),
        }
    fig.suptitle(f"{world}: multiscale grid-cell activity")
    fig.tight_layout()
    out_path = output_dir / "grid_cell_activations_multi_scale.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path, summary


def load_unified_pcn(network_dir: Path):
    path = network_dir / "unified_pcn.pkl"
    if not path.exists():
        return None
    return _load_pickle(path)


def module_metadata(unified_pcn, scale: int):
    if unified_pcn is None:
        return None
    configs = list(getattr(unified_pcn, "scale_configs", []))
    layers = list(getattr(unified_pcn, "grid_layers", []))
    for idx, cfg in enumerate(configs):
        if int(cfg.get("scale_index", idx)) != int(scale) or idx >= len(layers):
            continue
        layer = layers[idx]
        num_modules = int(getattr(layer, "num_modules", 0) or cfg.get("num_modules", 0) or 0)
        cells_per_module = int(getattr(layer, "cells_per_module", 0) or cfg.get("cells_per_module", 0) or 0)
        if num_modules <= 0 or cells_per_module <= 0:
            return None
        return num_modules, cells_per_module
    return None


def plot_module_maps(world: str, scale: int, xy: np.ndarray, activity: np.ndarray, metadata, output_dir: Path, bins: int):
    num_modules, cells_per_module = metadata
    cols = min(4, num_modules)
    rows = ceil(num_modules / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows), squeeze=False)
    aligned_xy, activity = align(xy, activity)
    for module_idx, ax in enumerate(axes.ravel()[:num_modules]):
        start = module_idx * cells_per_module
        end = min(start + cells_per_module, activity.shape[1])
        values = np.mean(activity[:, start:end], axis=1)
        hb = draw_hex(ax, aligned_xy, values, f"module {module_idx}", bins)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[num_modules:]:
        ax.axis("off")
    fig.suptitle(f"{world}: scale {scale} module means")
    fig.tight_layout()
    out_path = output_dir / f"grid_cell_modules_scale_{scale}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None, help="World name. Defaults to newest supported run.")
    parser.add_argument("--bins", type=int, default=70, help="Hexbin grid size.")
    parser.add_argument("--scale", type=int, default=None, help="Optional single scale to plot.")
    parser.add_argument("--skip-modules", action="store_true", help="Skip per-module mean maps.")
    args = parser.parse_args()

    world = resolve_world(args.world)
    hmap_dir, network_dir, output_dir = world_dirs(world)
    xy = load_locations(hmap_dir)
    gcn_hmaps = load_gcn_hmaps(hmap_dir)
    if args.scale is not None:
        gcn_hmaps = {args.scale: gcn_hmaps[args.scale]} if args.scale in gcn_hmaps else {}
    if not gcn_hmaps:
        raise FileNotFoundError(f"No matching hmap_gcn_scale_*.pkl files found in {hmap_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path, summary = plot_overview(world, xy, gcn_hmaps, output_dir, args.bins)
    summary_payload = {"world": world, "overview": str(overview_path), "scales": summary, "module_outputs": {}}
    if not args.skip_modules:
        unified_pcn = load_unified_pcn(network_dir)
        for scale, activity in sorted(gcn_hmaps.items()):
            meta = module_metadata(unified_pcn, scale)
            if meta is None:
                continue
            out_path = plot_module_maps(world, scale, xy, activity, meta, output_dir, args.bins)
            summary_payload["module_outputs"][str(scale)] = str(out_path)

    with open(output_dir / "grid_cell_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved grid-cell outputs to {output_dir}")


if __name__ == "__main__":
    main()
