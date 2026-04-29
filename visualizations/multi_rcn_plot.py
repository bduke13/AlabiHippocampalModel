"""Plot saved unified reward readout over visited locations."""

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

GOALS_BY_WORLD = {
    "20x20": [-7.0, 7.0],
    "20x20_1obstacle": [-7.0, 7.0],
    "20x20_2obstacles": [-7.0, 7.0],
    "20x20_goalBehindWall": [-2.0, 2.0],
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
        rcn_path = world_dir / "networks" / "unified_rcn_goal.pkl"
        if world_dir.name in SUPPORTED_WORLDS and loc_path.exists() and rcn_path.exists():
            candidates.append((max(loc_path.stat().st_mtime, rcn_path.stat().st_mtime), world_dir.name))
    if not candidates:
        raise FileNotFoundError(f"No supported reward-map data found under {PKL_ROOT}")
    return sorted(candidates, reverse=True)[0][1]


def world_dirs(world: str):
    base = PKL_ROOT / world
    return base / "hmaps", base / "networks", base / "vis_outputs" / "reward_cells"


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


def load_pcn_hmaps(hmap_dir: Path):
    scale_hmaps = {
        scale: _as_numpy(_load_pickle(hmap_dir / f"hmap_pcn_scale_{scale}.pkl"))
        for scale in discover_scales(hmap_dir)
    }
    if not scale_hmaps:
        raise FileNotFoundError(f"No hmap_pcn_scale_*.pkl files found in {hmap_dir}")
    min_len = min(len(arr) for arr in scale_hmaps.values())
    scale_hmaps = {scale: arr[:min_len] for scale, arr in scale_hmaps.items()}
    unified = np.concatenate([scale_hmaps[s] for s in sorted(scale_hmaps)], axis=1)
    return scale_hmaps, unified


def align(xy: np.ndarray, activity: np.ndarray):
    n = min(len(xy), len(activity))
    xy = xy[:n]
    activity = activity[:n]
    good = np.isfinite(xy).all(axis=1) & np.isfinite(activity).all(axis=1)
    return xy[good], activity[good]


def load_goal(world: str, hmap_dir: Path):
    unified_path = hmap_dir / "unified_hmaps.pkl"
    if unified_path.exists():
        try:
            payload = _load_pickle(unified_path)
            if isinstance(payload, dict) and "goal_location" in payload:
                return [float(v) for v in payload["goal_location"][:2]]
        except Exception:
            pass
    return GOALS_BY_WORLD.get(world)


def get_weights(rcn) -> np.ndarray:
    weights = getattr(rcn, "w_in_effective", None)
    if weights is None:
        weights = getattr(rcn, "w_in", None)
    if weights is None:
        raise AttributeError("Saved RCN has no w_in_effective or w_in")
    return _as_numpy(weights).reshape(-1)


def scale_slices(rcn, scale_hmaps: dict[int, np.ndarray]):
    widths = [scale_hmaps[s].shape[1] for s in sorted(scale_hmaps)]
    boundaries = list(getattr(rcn, "scale_boundaries", []) or [])
    if len(boundaries) == len(widths) + 1 and int(boundaries[-1]) == int(sum(widths)):
        return {
            scale: (int(boundaries[i]), int(boundaries[i + 1]))
            for i, scale in enumerate(sorted(scale_hmaps))
        }
    slices = {}
    start = 0
    for scale, width in zip(sorted(scale_hmaps), widths):
        slices[scale] = (start, start + int(width))
        start += int(width)
    return slices


def compute_reward_maps(rcn, scale_hmaps: dict[int, np.ndarray], unified: np.ndarray):
    weights = get_weights(rcn)
    if len(weights) != unified.shape[1]:
        raise ValueError(f"RCN weight count {len(weights)} does not match PCN width {unified.shape[1]}")
    full = np.maximum(unified @ weights, 0.0)
    maps = {"unified": full}
    for scale, (start, end) in scale_slices(rcn, scale_hmaps).items():
        maps[f"scale {scale}"] = np.maximum(unified[:, start:end] @ weights[start:end], 0.0)
    return maps


def draw_hex(ax, xy: np.ndarray, values: np.ndarray, title: str, bins: int):
    hb = ax.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=values,
        gridsize=bins,
        reduce_C_function=np.mean,
        mincnt=1,
        linewidths=0.0,
        cmap="inferno",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    return hb


def plot_rewards(world: str, xy: np.ndarray, reward_maps: dict[str, np.ndarray], goal, output_dir: Path, bins: int):
    labels = [label for label in sorted(reward_maps) if label != "unified"] + ["unified"]
    fig, axes = plt.subplots(1, len(labels), figsize=(5.0 * len(labels), 4.8), squeeze=False)
    summary = {}
    for ax, label in zip(axes[0], labels):
        aligned_xy, values = align(xy, reward_maps[label].reshape(-1, 1))
        values = values.reshape(-1)
        hb = draw_hex(ax, aligned_xy, values, label, bins)
        if goal is not None:
            ax.scatter([goal[0]], [goal[1]], marker="*", s=170, c="cyan", edgecolors="black", linewidths=0.7)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        summary[label] = {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
        }
    fig.suptitle(f"{world}: unified reward readout")
    fig.tight_layout()
    out_path = output_dir / "reward_map.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path, summary


def recorded_reward_summary(hmap_dir: Path):
    path = hmap_dir / "hmap_reward.pkl"
    if not path.exists():
        return None
    values = _as_numpy(_load_pickle(path)).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return {
        "samples": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None, help="World name. Defaults to newest supported reward run.")
    parser.add_argument("--bins", type=int, default=70, help="Hexbin grid size.")
    args = parser.parse_args()

    world = resolve_world(args.world)
    hmap_dir, network_dir, output_dir = world_dirs(world)
    xy = load_locations(hmap_dir)
    scale_hmaps, unified = load_pcn_hmaps(hmap_dir)
    rcn = _load_pickle(network_dir / "unified_rcn_goal.pkl")
    reward_maps = compute_reward_maps(rcn, scale_hmaps, unified)
    goal = load_goal(world, hmap_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path, summary = plot_rewards(world, xy, reward_maps, goal, output_dir, args.bins)
    payload = {
        "world": world,
        "output": str(out_path),
        "goal": goal,
        "scales": sorted(int(s) for s in scale_hmaps),
        "reward": summary,
        "recorded_reward_trace": recorded_reward_summary(hmap_dir),
    }
    with open(output_dir / "reward_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
