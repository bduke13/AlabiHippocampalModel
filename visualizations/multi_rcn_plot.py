import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
)


def _network_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"


def _hmap_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "hmaps"


def _load_hmap_loc() -> np.ndarray:
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    hmap_loc = np.array(hmap_loc)
    # Use planar axes for this project: x and y are columns 0 and 1.
    return hmap_loc[:, :2]


def _discover_scales() -> List[int]:
    hmaps = _hmap_dir()
    if not hmaps.exists():
        return []
    found = []
    for fpath in sorted(hmaps.glob("hmap_pcn_scale_*.pkl")):
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


def _compute_reward(weights_1d: np.ndarray, hmap_pcn: np.ndarray) -> torch.Tensor:
    w = torch.tensor(weights_1d, dtype=torch.float32).view(1, -1)
    p = torch.tensor(hmap_pcn.T, dtype=torch.float32)

    if w.shape[1] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: weights={w.shape[1]} vs pcn={p.shape[0]}")

    # Match RewardCell forward equation:
    # r_t = (w · p_t) / sum(|w|), not /(sum p_t)
    safe_denom = torch.clamp(torch.sum(torch.abs(w), dim=1, keepdim=True), min=1e-12)
    reward = torch.tensordot(w, p, dims=1) / safe_denom
    return torch.squeeze(reward)


def _weights_to_numpy_1d(weights_obj) -> np.ndarray:
    """Convert torch/cpu/cuda or numpy-like weights into a flat CPU numpy array."""
    if isinstance(weights_obj, torch.Tensor):
        return weights_obj.detach().cpu().numpy().reshape(-1)
    return np.asarray(weights_obj).reshape(-1)


def _plot_reward_hex(
    ax,
    hmap_xy: np.ndarray,
    reward: torch.Tensor,
    title: str,
    cmap_name: str = "plasma",
    show_colorbar: bool = True,
):
    hmap_x = hmap_xy[:, 0]
    hmap_y = hmap_xy[:, 1]
    cmap = cm.get_cmap(cmap_name)

    cntr = ax.hexbin(
        hmap_x,
        hmap_y,
        reward.detach().cpu().numpy(),
        gridsize=100,
        cmap=cmap,
        alpha=0.75,
    )
    ax.set_xlim(float(np.min(hmap_x)), float(np.max(hmap_x)))
    ax.set_ylim(float(np.min(hmap_y)), float(np.max(hmap_y)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    if show_colorbar:
        plt.colorbar(cntr, ax=ax)


def _is_unified_available() -> bool:
    net = _network_dir()
    return (net / "unified_rcn.pkl").exists()


def _multi_goal_dir() -> Path:
    return _network_dir() / "multi_goal_rewards"


def _discover_unified_goal_rcn_files() -> List[Tuple[str, Path]]:
    mg = _multi_goal_dir()
    if not mg.exists():
        return []
    found: List[Tuple[str, Path]] = []
    for p in sorted(mg.glob("unified_rcn_goal_*.pkl")):
        stem = p.stem
        goal_name = stem.replace("unified_rcn_goal_", "")
        found.append((goal_name, p))
    return found


def _load_pickle_path(path: Path):
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


def _build_unified_scale_slices(
    unified_pcn,
    available_scales: List[int],
    hmap_by_scale: Dict[int, np.ndarray],
) -> List[Tuple[int, int, int]]:
    # Preferred path: use saved unified_pcn boundaries + scale_configs.
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
                start = int(unified_pcn.scale_boundaries[i])
                end = int(unified_pcn.scale_boundaries[i + 1])
                slices.append((scale_idx, start, end))
        if slices:
            return slices

    # Fallback: infer order by ascending scale index and column counts.
    slices = []
    start = 0
    for scale_idx in sorted(available_scales):
        width = int(hmap_by_scale[scale_idx].shape[1])
        end = start + width
        slices.append((scale_idx, start, end))
        start = end
    return slices


def plot_all_scales(
    scales: Optional[List[int]] = None,
    cmap_name: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    architecture: str = "auto",  # auto | unified | separate
):
    if scales is None:
        scales = _discover_scales() or [0, 1, 2]

    hmap_xy = _load_hmap_loc()
    rcParams.update({"font.size": 12})

    unified_mode = _is_unified_available() if architecture == "auto" else (architecture == "unified")

    if unified_mode:
        unified_rcn, unified_pcn = _load_unified_layers()
        hmap_by_scale: Dict[int, np.ndarray] = {s: _load_scale_hmap(s) for s in scales}
        scale_slices = _build_unified_scale_slices(unified_pcn, scales, hmap_by_scale)

        # Build unified activation history in the same slice order.
        per_scale_arrays = [hmap_by_scale[s] for s, _, _ in scale_slices]
        min_steps = min([len(hmap_xy)] + [arr.shape[0] for arr in per_scale_arrays]) if per_scale_arrays else len(hmap_xy)
        hmap_xy = hmap_xy[:min_steps]
        per_scale_arrays = [arr[:min_steps] for arr in per_scale_arrays]
        hmap_unified = np.concatenate(per_scale_arrays, axis=1)

        unified_weights = _weights_to_numpy_1d(unified_rcn.w_in_effective)
        goal_files = _discover_unified_goal_rcn_files()

        include_scale_panels = True
        n_goal = len(goal_files)
        n_scale_panels = 1 + len(scale_slices) if include_scale_panels else 0
        ncols = max(n_goal if n_goal > 0 else 1, n_scale_panels if n_scale_panels > 0 else 1)
        nrows = 2 if include_scale_panels else 1

        fig = plt.figure(figsize=(5.5 * ncols, 5 * nrows))
        fig.suptitle("Unified Reward Maps: Goals + Scale Breakdown", fontsize=16)
        gs = gridspec.GridSpec(nrows, ncols)

        # Row 1: per-goal unified reward maps (main view)
        if n_goal > 0:
            for i, (goal_name, goal_path) in enumerate(goal_files):
                ax = plt.subplot(gs[0, i])
                goal_rcn = _load_pickle_path(goal_path)
                goal_weights = _weights_to_numpy_1d(goal_rcn.w_in_effective)
                reward_goal = _compute_reward(goal_weights, hmap_unified)
                _plot_reward_hex(
                    ax,
                    hmap_xy,
                    reward_goal,
                    title=f"Goal: {goal_name}",
                    cmap_name=cmap_name,
                    show_colorbar=True,
                )
        else:
            ax = plt.subplot(gs[0, 0])
            reward_u = _compute_reward(unified_weights, hmap_unified)
            _plot_reward_hex(ax, hmap_xy, reward_u, title="Unified (No goal files found)", cmap_name=cmap_name, show_colorbar=True)

        # Row 2: optional scale breakdown for first goal (or base unified as fallback)
        if include_scale_panels:
            scale_weights = unified_weights
            scale_title_prefix = "Base Unified"
            if n_goal > 0:
                first_goal_name, first_goal_path = goal_files[0]
                first_goal_rcn = _load_pickle_path(first_goal_path)
                scale_weights = _weights_to_numpy_1d(first_goal_rcn.w_in_effective)
                scale_title_prefix = f"Goal {first_goal_name}"

            ax0 = plt.subplot(gs[1, 0])
            reward_u = _compute_reward(scale_weights, hmap_unified)
            _plot_reward_hex(
                ax0,
                hmap_xy,
                reward_u,
                title=f"{scale_title_prefix}: Unified",
                cmap_name=cmap_name,
                show_colorbar=True,
            )

            for i, (scale_idx, start, end) in enumerate(scale_slices, start=1):
                ax = plt.subplot(gs[1, i])
                hmap_scale = hmap_by_scale[scale_idx][:min_steps]
                reward_s = _compute_reward(scale_weights[start:end], hmap_scale)
                _plot_reward_hex(
                    ax,
                    hmap_xy,
                    reward_s,
                    title=f"{scale_title_prefix}: Scale {scale_idx}",
                    cmap_name=cmap_name,
                    show_colorbar=True,
                )

    else:
        fig = plt.figure(figsize=(6 * len(scales), 5))
        fig.suptitle("Reward Cell Network Activation Across Scales", fontsize=16)
        gs = gridspec.GridSpec(1, len(scales))

        for i, scale_idx in enumerate(scales):
            ax = plt.subplot(gs[0, i])
            rcn = load_layer_pkl(f"rcn_scale_{scale_idx}")
            hmap_scale = _load_scale_hmap(scale_idx)
            min_steps = min(len(hmap_xy), hmap_scale.shape[0])
            reward_s = _compute_reward(_weights_to_numpy_1d(rcn.w_in_effective), hmap_scale[:min_steps])
            _plot_reward_hex(
                ax,
                hmap_xy[:min_steps],
                reward_s,
                title=f"Scale {scale_idx}",
                cmap_name=cmap_name,
                show_colorbar=True,
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


if __name__ == "__main__":
    print("Starting RCN visualization...")
    output_file = Path(OUTPUT_DIR) / "rcn_plots" / "rcn_all_scales.png"
    plot_all_scales(
        scales=None,
        cmap_name="plasma",
        save_path=str(output_file),
        show_plot=True,
        architecture="auto",
    )
    print("RCN visualization complete.")
