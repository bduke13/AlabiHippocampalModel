"""MSAI heatmaps from saved multiscale place-cell histories."""

import argparse
import json
import os
import pickle
import sys
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


def _world_candidates():
    if not PKL_ROOT.exists():
        return []
    found = []
    for world_dir in PKL_ROOT.iterdir():
        loc_path = world_dir / "hmaps" / "hmap_loc.pkl"
        if world_dir.name in SUPPORTED_WORLDS and loc_path.exists():
            found.append((loc_path.stat().st_mtime, world_dir.name))
    return [name for _, name in sorted(found, reverse=True)]


def resolve_world(requested: str | None) -> str:
    if requested:
        return requested
    candidates = _world_candidates()
    if not candidates:
        raise FileNotFoundError(f"No supported hmap data found under {PKL_ROOT}")
    return candidates[0]


def world_dirs(world: str):
    base = PKL_ROOT / world
    return base / "hmaps", base / "vis_outputs" / "msai_heatmaps"


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


def load_pcn_hmaps(hmap_dir: Path) -> dict[int, np.ndarray]:
    hmaps = {}
    for scale in discover_scales(hmap_dir):
        hmaps[scale] = _as_numpy(_load_pickle(hmap_dir / f"hmap_pcn_scale_{scale}.pkl"))
    return hmaps


def align_xy_and_activity(xy: np.ndarray, activity: np.ndarray):
    n = min(len(xy), len(activity))
    xy = xy[:n]
    activity = activity[:n]
    good = np.isfinite(xy).all(axis=1) & np.isfinite(activity).all(axis=1)
    return xy[good], activity[good]


def binned_mean_vectors(xy: np.ndarray, activity: np.ndarray, bins: int):
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    xi = np.searchsorted(xedges, xy[:, 0], side="right") - 1
    yi = np.searchsorted(yedges, xy[:, 1], side="right") - 1
    xi = np.clip(xi, 0, bins - 1)
    yi = np.clip(yi, 0, bins - 1)
    flat = yi * bins + xi

    sums = np.zeros((bins * bins, activity.shape[1]), dtype=np.float32)
    counts = np.bincount(flat, minlength=bins * bins).astype(np.float32)
    np.add.at(sums, flat, activity.astype(np.float32, copy=False))
    occupied = counts > 0
    means = sums[occupied] / counts[occupied, None]

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    grid_x, grid_y = np.meshgrid(xcenters, ycenters)
    centers = np.column_stack([grid_x.ravel()[occupied], grid_y.ravel()[occupied]])
    return {
        "means": means,
        "centers": centers,
        "occupied": occupied,
        "extent": [float(xmin), float(xmax), float(ymin), float(ymax)],
        "bins": bins,
    }


def mean_distant_similarity(vectors: np.ndarray, coords: np.ndarray, min_separation: float):
    norms = np.linalg.norm(vectors, axis=1)
    active = norms > 1e-8
    scores = np.zeros(len(vectors), dtype=np.float32)
    if int(np.sum(active)) < 2:
        return scores

    unit = vectors[active] / norms[active, None]
    active_coords = coords[active]
    values = np.zeros(len(unit), dtype=np.float32)
    chunk = 512
    for start in range(0, len(unit), chunk):
        stop = min(start + chunk, len(unit))
        sim = unit[start:stop] @ unit.T
        delta = active_coords[start:stop, None, :] - active_coords[None, :, :]
        far = np.linalg.norm(delta, axis=2) >= float(min_separation)
        values[start:stop] = np.sum(sim * far, axis=1) / np.maximum(np.sum(far, axis=1), 1)

    scores[active] = np.clip(values, 0.0, 1.0)
    return scores


def compute_msai_map(xy: np.ndarray, activity: np.ndarray, bins: int, min_separation: float):
    xy, activity = align_xy_and_activity(xy, activity)
    if len(xy) == 0 or activity.size == 0:
        raise ValueError("No finite activity samples available")
    binned = binned_mean_vectors(xy, activity, bins)
    sai = mean_distant_similarity(binned["means"], binned["centers"], min_separation)
    grid = np.full(bins * bins, np.nan, dtype=np.float32)
    grid[binned["occupied"]] = sai
    return {
        "grid": grid.reshape(bins, bins),
        "msai": float(np.nanmean(sai)) if len(sai) else 0.0,
        "occupied_bins": int(np.sum(binned["occupied"])),
        "extent": binned["extent"],
    }


def plot_maps(world: str, maps: list[tuple[str, dict]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(maps), figsize=(5.0 * len(maps), 4.6), squeeze=False)
    for ax, (label, result) in zip(axes[0], maps):
        image = ax.imshow(
            result["grid"],
            origin="lower",
            extent=result["extent"],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"{label}\nMSAI={result['msai']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{world}: spatial aliasing index")
    fig.tight_layout()
    out_path = output_dir / "msai_heatmaps.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None, help="World name. Defaults to newest supported run.")
    parser.add_argument("--bins", type=int, default=45, help="Spatial bins per axis.")
    parser.add_argument("--min-separation", type=float, default=2.0, help="Ignore bins closer than this distance.")
    args = parser.parse_args()

    world = resolve_world(args.world)
    hmap_dir, output_dir = world_dirs(world)
    xy = load_locations(hmap_dir)
    scale_hmaps = load_pcn_hmaps(hmap_dir)
    if not scale_hmaps:
        raise FileNotFoundError(f"No hmap_pcn_scale_*.pkl files found in {hmap_dir}")

    maps = []
    summary = {"world": world, "bins": args.bins, "min_separation": args.min_separation, "maps": {}}
    for scale, activity in sorted(scale_hmaps.items()):
        result = compute_msai_map(xy, activity, args.bins, args.min_separation)
        label = f"scale {scale}"
        maps.append((label, result))
        summary["maps"][label] = {
            "msai": result["msai"],
            "occupied_bins": result["occupied_bins"],
            "cells": int(activity.shape[1]),
        }

    unified = np.concatenate([scale_hmaps[s] for s in sorted(scale_hmaps)], axis=1)
    result = compute_msai_map(xy, unified, args.bins, args.min_separation)
    maps.append(("unified", result))
    summary["maps"]["unified"] = {
        "msai": result["msai"],
        "occupied_bins": result["occupied_bins"],
        "cells": int(unified.shape[1]),
    }

    out_path = plot_maps(world, maps, output_dir)
    with open(output_dir / "msai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
