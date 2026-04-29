"""Plot passive multiscale formation diagnostics saved by the controller."""

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "webots" / "controllers" / "multiscale_grid_controller"
PKL_ROOT = CONTROLLER_DIR / "pkl"
SUPPORTED_WORLDS = {"20x20", "20x20_1obstacle", "20x20_2obstacles", "20x20_goalBehindWall"}


def resolve_world(requested: str | None) -> str:
    if requested:
        return requested
    candidates = []
    for world_dir in PKL_ROOT.iterdir() if PKL_ROOT.exists() else []:
        path = world_dir / "hmaps" / "scale_diagnostics.pkl"
        if world_dir.name in SUPPORTED_WORLDS and path.exists():
            candidates.append((path.stat().st_mtime, world_dir.name))
    if not candidates:
        raise FileNotFoundError(f"No scale diagnostics found under {PKL_ROOT}")
    return sorted(candidates, reverse=True)[0][1]


def world_dirs(world: str):
    base = PKL_ROOT / world
    return base / "hmaps", base / "vis_outputs" / "scale_diagnostics"


def load_diagnostics(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else []


def stack_list(rows: list[dict], key: str) -> np.ndarray:
    values = [row.get(key, []) for row in rows]
    width = max((len(v) for v in values if isinstance(v, list)), default=0)
    arr = np.full((len(rows), width), np.nan, dtype=np.float32)
    for i, value in enumerate(values):
        if not isinstance(value, list):
            continue
        n = min(width, len(value))
        if n:
            arr[i, :n] = np.asarray(value[:n], dtype=np.float32)
    return arr


def numeric(rows: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, default)))
        except (TypeError, ValueError):
            out.append(float(default))
    return np.asarray(out, dtype=np.float32)


def plot(world: str, rows: list[dict], output_dir: Path):
    steps = numeric(rows, "step")
    x = numeric(rows, "x")
    y = numeric(rows, "y")
    prox = numeric(rows, "prox")
    dominant = numeric(rows, "dominant_scale", -1)
    masses = stack_list(rows, "scale_masses")
    pref = stack_list(rows, "scale_preference")
    gain = stack_list(rows, "bvc_context_gain")
    recurrent = stack_list(rows, "recurrent_inhibition_mean")
    n_scales = int(max(masses.shape[1], pref.shape[1], gain.shape[1], recurrent.shape[1]))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4))
    sc = axes[0, 0].scatter(x, y, c=dominant, s=7, cmap="tab10", vmin=-0.5, vmax=max(0.5, n_scales - 0.5))
    axes[0, 0].set_title("Dominant scale across trajectory")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for i in range(masses.shape[1]):
        axes[0, 1].plot(steps, masses[:, i], linewidth=1.1, label=f"scale {i}")
    axes[0, 1].set_title("Scale activity mass")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("sum |PCN|")
    axes[0, 1].legend(fontsize=8)

    for i in range(gain.shape[1]):
        axes[1, 0].scatter(prox, gain[:, i], s=6, alpha=0.45, label=f"scale {i}")
    axes[1, 0].set_title("BVC context gain vs proximity")
    axes[1, 0].set_xlabel("proximity")
    axes[1, 0].set_ylabel("gain")
    axes[1, 0].legend(fontsize=8)

    for i in range(recurrent.shape[1]):
        axes[1, 1].plot(steps, recurrent[:, i], linewidth=1.1, label=f"scale {i}")
    axes[1, 1].set_title("All-scale recurrent inhibition proxy")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("mean inhibition")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"{world}: multiscale formation diagnostics")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scale_diagnostics.png"
    fig.savefig(out_path, dpi=190)
    plt.close(fig)

    summary = {
        "world": world,
        "samples": int(len(rows)),
        "scales": {},
        "dominant_scale_counts": {
            str(int(scale)): int(np.sum(dominant == scale))
            for scale in sorted(set(int(v) for v in dominant if v >= 0))
        },
    }
    for i in range(n_scales):
        summary["scales"][str(i)] = {
            "mean_mass": float(np.nanmean(masses[:, i])) if i < masses.shape[1] else 0.0,
            "mean_preference": float(np.nanmean(pref[:, i])) if i < pref.shape[1] else 0.0,
            "mean_bvc_gain": float(np.nanmean(gain[:, i])) if i < gain.shape[1] else 0.0,
            "mean_recurrent_inhibition": float(np.nanmean(recurrent[:, i])) if i < recurrent.shape[1] else 0.0,
        }
    return out_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None, help="World name. Defaults to newest supported run.")
    args = parser.parse_args()

    world = resolve_world(args.world)
    hmap_dir, output_dir = world_dirs(world)
    rows = load_diagnostics(hmap_dir / "scale_diagnostics.pkl")
    if not rows:
        raise FileNotFoundError(f"No diagnostics rows found in {hmap_dir / 'scale_diagnostics.pkl'}")
    out_path, summary = plot(world, rows, output_dir)
    with open(output_dir / "scale_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
