"""
Reward-map visualisation for single-goal, unified-multiscale architecture.

Each saved unified_rcn_goal_<name>.pkl is one row in the figure.
Columns: per-scale reward slice  +  full unified reward.
Goal position (★) and checkpoint positions (◆) are overlaid on every panel.
"""
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, rcParams

from vis_utils import (
    CONTROLLER_NAME,
    CONTROLLER_PATH_PREFIX,
    OUTPUT_DIR,
    WORLD_NAME,
    load_hmaps,
    load_layer_pkl,
    get_env_goal_pos,
    get_env_checkpoint_positions,
)


# ── path helpers ──────────────────────────────────────────────────────────────

def _network_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"


def _hmap_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "hmaps"


def _multi_goal_dir() -> Path:
    return _network_dir() / "multi_goal_rewards"


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_hmap_loc() -> np.ndarray:
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    return np.array(hmap_loc)[:, :2]          # (T, 2)  x, y floor plane


def _discover_scales() -> List[int]:
    found = []
    for fpath in sorted(_hmap_dir().glob("hmap_pcn_scale_*.pkl")):
        try:
            found.append(int(fpath.stem.split("_")[-1]))
        except ValueError:
            continue
    return found


def _load_scale_hmap(scale_idx: int) -> np.ndarray:
    data = load_hmaps([f"hmap_pcn_scale_{scale_idx}"])
    if isinstance(data, list):
        data = data[0]
    return np.array(data)


def _discover_goal_rcn_files() -> List[Tuple[str, Path]]:
    mg = _multi_goal_dir()
    if not mg.exists():
        return []
    return [
        (p.stem.replace("unified_rcn_goal_", ""), p)
        for p in sorted(mg.glob("unified_rcn_goal_*.pkl"))
    ]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_unified_layers():
    unified_rcn = load_layer_pkl("unified_rcn")
    unified_pcn = None
    try:
        unified_pcn = load_layer_pkl("unified_pcn")
    except Exception:
        pass
    return unified_rcn, unified_pcn


def _build_scale_slices(
    unified_pcn,
    available_scales: List[int],
    hmap_by_scale: Dict[int, np.ndarray],
) -> List[Tuple[int, int, int]]:
    if (
        unified_pcn is not None
        and hasattr(unified_pcn, "scale_boundaries")
        and hasattr(unified_pcn, "scale_configs")
        and len(unified_pcn.scale_boundaries) >= 2
    ):
        slices = []
        for i, cfg in enumerate(unified_pcn.scale_configs):
            scale_idx = int(cfg.get("scale_index", i))
            if scale_idx in hmap_by_scale:
                slices.append((scale_idx,
                                int(unified_pcn.scale_boundaries[i]),
                                int(unified_pcn.scale_boundaries[i + 1])))
        if slices:
            return slices

    # Fallback: infer from column counts
    slices, start = [], 0
    for s in sorted(available_scales):
        width = int(hmap_by_scale[s].shape[1])
        slices.append((s, start, start + width))
        start += width
    return slices


# ── reward computation ────────────────────────────────────────────────────────

def _compute_reward(
    weights_1d: np.ndarray,
    hmap_pcn: np.ndarray,
    denom_override: Optional[float] = None,
) -> torch.Tensor:
    w = torch.tensor(weights_1d, dtype=torch.float32).view(1, -1)
    p = torch.tensor(hmap_pcn.T, dtype=torch.float32)
    if w.shape[1] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: weights={w.shape[1]} vs pcn={p.shape[0]}")
    if denom_override is None:
        denom = torch.clamp(torch.sum(torch.abs(w), dim=1, keepdim=True), min=1e-12)
    else:
        denom = torch.tensor([[max(float(denom_override), 1e-12)]], dtype=torch.float32)
    return torch.squeeze(torch.tensordot(w, p, dims=1) / denom)


def _weights_1d(weights_obj) -> np.ndarray:
    if isinstance(weights_obj, torch.Tensor):
        return weights_obj.detach().cpu().numpy().reshape(-1)
    return np.asarray(weights_obj).reshape(-1)


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_reward_hex(
    ax,
    hmap_xy: np.ndarray,
    reward: torch.Tensor,
    title: str,
    cmap_name: str = "plasma",
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    goal_pos: Optional[Tuple[float, float]] = None,
    checkpoint_positions: Optional[List[Tuple[float, float]]] = None,
):
    hmap_x, hmap_y = hmap_xy[:, 0], hmap_xy[:, 1]
    r_np = reward.detach().cpu().numpy()

    cntr = ax.hexbin(
        hmap_x, hmap_y, r_np,
        gridsize=100,
        cmap=cm.get_cmap(cmap_name),
        alpha=0.85,
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay checkpoint positions
    if checkpoint_positions:
        for (cx, cy) in checkpoint_positions:
            ax.plot(cx, cy, marker="D", color="white",
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)

    # Overlay goal position
    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], marker="*", color="lime",
                markersize=14, markeredgecolor="black",
                markeredgewidth=0.8, zorder=6)

    ax.set_xlim(float(np.min(hmap_x)), float(np.max(hmap_x)))
    ax.set_ylim(float(np.min(hmap_y)), float(np.max(hmap_y)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=10)
    if show_colorbar:
        plt.colorbar(cntr, ax=ax)


# ── main entry point ──────────────────────────────────────────────────────────

def plot_reward_maps(
    scales: Optional[List[int]] = None,
    cmap_name: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plot reward maps for all saved goal RCNs.

    Layout: rows = goals, cols = [scale_0, scale_1, ..., unified].
    ★  marks the goal position.
    ◆  marks checkpoint positions.
    """
    if scales is None:
        scales = _discover_scales() or [0, 1, 2]

    goal_files = _discover_goal_rcn_files()
    if not goal_files:
        print("[WARNING] No unified goal RCN files found in", _multi_goal_dir())
        return

    hmap_xy = _load_hmap_loc()
    _, unified_pcn = _load_unified_layers()
    hmap_by_scale: Dict[int, np.ndarray] = {s: _load_scale_hmap(s) for s in scales}
    scale_slices = _build_scale_slices(unified_pcn, scales, hmap_by_scale)

    # Build unified hmap
    min_steps = min(len(hmap_xy), *(arr.shape[0] for arr in hmap_by_scale.values()))
    hmap_xy = hmap_xy[:min_steps]
    hmap_unified = np.concatenate(
        [hmap_by_scale[s][:min_steps] for s, _, _ in scale_slices], axis=1
    )

    # Spatial overlays from vis_utils
    goal_pos = get_env_goal_pos(WORLD_NAME)
    checkpoint_pos = get_env_checkpoint_positions(WORLD_NAME)

    n_goals = len(goal_files)
    n_scale_cols = len(scale_slices)
    ncols = n_scale_cols + 1          # per-scale + unified

    rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        n_goals, ncols,
        figsize=(4.5 * ncols, 4.0 * n_goals),
        squeeze=False,
    )
    env_label = WORLD_NAME.replace("_", " ").title()
    n_cp = len(checkpoint_pos)
    fig.suptitle(
        f"Reward Maps — {env_label}  "
        f"({'no checkpoints' if n_cp == 0 else f'{n_cp} checkpoint(s)'})",
        fontsize=13, y=1.01,
    )

    for row, (goal_name, goal_path) in enumerate(goal_files):
        goal_rcn = _load_pickle(goal_path)
        goal_weights = _weights_1d(goal_rcn.w_in_effective)
        unified_denom = float(np.sum(np.abs(goal_weights)))

        # Per-scale columns
        for col, (scale_idx, s_start, s_end) in enumerate(scale_slices):
            ax = axes[row][col]
            scale_weights = goal_weights[s_start:s_end]
            reward_s = _compute_reward(
                scale_weights,
                hmap_by_scale[scale_idx][:min_steps],
                denom_override=unified_denom,
            )
            _plot_reward_hex(
                ax, hmap_xy, reward_s,
                title=f"{goal_name} | Scale {scale_idx}",
                cmap_name=cmap_name,
                goal_pos=goal_pos,
                checkpoint_positions=checkpoint_pos,
            )

        # Unified column (last)
        ax_u = axes[row][ncols - 1]
        reward_u = _compute_reward(goal_weights, hmap_unified)
        _plot_reward_hex(
            ax_u, hmap_xy, reward_u,
            title=f"{goal_name} | Unified",
            cmap_name=cmap_name,
            goal_pos=goal_pos,
            checkpoint_positions=checkpoint_pos,
        )

    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {out}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# Keep old name as alias so any existing code importing plot_all_scales still works.
plot_all_scales = plot_reward_maps


if __name__ == "__main__":
    print(f"Plotting reward maps for {WORLD_NAME}...")
    output_file = Path(OUTPUT_DIR) / "rcn_plots" / "reward_maps.png"
    plot_reward_maps(
        scales=None,
        cmap_name="plasma",
        save_path=str(output_file),
        show_plot=True,
    )
    print("Done.")
