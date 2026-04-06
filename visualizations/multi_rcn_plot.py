"""
Reward-map visualisation for single-goal, unified-multiscale architecture.

Each saved unified_rcn_goal_<name>.pkl is one row in the figure.
Columns: per-scale reward slice  +  full unified reward.
Goal position (★) and checkpoint positions (◆) are overlaid on every panel.
"""
import math
import pickle
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap

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
    denom_override: Optional[Union[float, np.ndarray]] = None,
    normalization_mode: str = "input_l1",
) -> torch.Tensor:
    w = torch.tensor(weights_1d, dtype=torch.float32).view(1, -1)
    p = torch.tensor(hmap_pcn.T, dtype=torch.float32)
    if w.shape[1] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: weights={w.shape[1]} vs pcn={p.shape[0]}")

    reward_numer = torch.squeeze(torch.tensordot(w, p, dims=1))
    mode = str(normalization_mode).strip().lower()
    if mode in {"input_l1", "input", "l1", "v11"}:
        mode = "input_l1"
    else:
        mode = "weight_mass"

    if denom_override is None:
        if mode == "input_l1":
            denom = torch.clamp(torch.sum(torch.abs(p), dim=0), min=1e-4)
        else:
            denom = torch.clamp(torch.sum(torch.abs(w), dim=1).squeeze(), min=1e-12)
    else:
        denom = torch.as_tensor(denom_override, dtype=torch.float32)
        if denom.ndim == 0:
            floor = 1e-4 if mode == "input_l1" else 1e-12
            denom = torch.clamp(denom, min=floor)
        else:
            denom = denom.reshape(-1)
            if denom.shape[0] != reward_numer.shape[0]:
                raise ValueError(
                    f"Denominator length mismatch: denom={denom.shape[0]} vs reward={reward_numer.shape[0]}"
                )
            floor = 1e-4 if mode == "input_l1" else 1e-12
            denom = torch.clamp(denom, min=floor)

    return reward_numer / denom


def _weights_1d(weights_obj) -> np.ndarray:
    if isinstance(weights_obj, torch.Tensor):
        return weights_obj.detach().cpu().numpy().reshape(-1)
    return np.asarray(weights_obj).reshape(-1)


def _resolved_reward_normalization_mode(rcn_obj) -> str:
    mode = str(getattr(rcn_obj, "reward_normalization_mode", "input_l1")).strip().lower()
    if mode in {"input_l1", "input", "l1", "v11"}:
        return "input_l1"
    return "weight_mass"


def _extract_route_selection_summary(rcn_obj) -> Optional[str]:
    debug_log = str(getattr(rcn_obj, "goal_map_debug_log", "") or "")
    marker = "route_select("
    start = debug_log.find(marker)
    if start < 0:
        return None

    depth = 0
    end = None
    for idx in range(start, len(debug_log)):
        ch = debug_log[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        return None

    return debug_log[start:end]


def _extract_segment_debug(rcn_obj) -> Optional[Dict[str, object]]:
    debug = getattr(rcn_obj, "goal_map_segment_debug", None)
    return debug if isinstance(debug, dict) else None


def _segment_weight_maps(
    segment_debug: Optional[Dict[str, object]],
) -> List[Tuple[str, np.ndarray]]:
    if not segment_debug:
        return []

    source_names = [str(name) for name in segment_debug.get("source_names", [])]
    total_maps = segment_debug.get("total_maps_normalized_pre_smooth")
    if not isinstance(total_maps, dict) or not total_maps:
        total_maps = segment_debug.get("total_maps_pre_smooth")
    if not isinstance(total_maps, dict):
        return []

    items: List[Tuple[str, np.ndarray]] = []
    for source_name in source_names:
        weights_obj = total_maps.get(source_name)
        if weights_obj is None:
            continue
        items.append((source_name, _weights_1d(weights_obj)))
    return items


def _compute_segment_rewards(
    segment_debug: Optional[Dict[str, object]],
    hmap_unified: np.ndarray,
    normalization_mode: str,
    denom_override: Optional[Union[float, np.ndarray]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    rewards: List[Tuple[str, torch.Tensor]] = []
    for source_name, weights_1d in _segment_weight_maps(segment_debug):
        rewards.append(
            (
                source_name,
                _compute_reward(
                    weights_1d,
                    hmap_unified,
                    denom_override=denom_override,
                    normalization_mode=normalization_mode,
                ),
            )
        )
    return rewards


def _compute_presmooth_unified_reward(
    segment_debug: Optional[Dict[str, object]],
    hmap_unified: np.ndarray,
    normalization_mode: str,
    denom_override: Optional[Union[float, np.ndarray]] = None,
) -> Optional[torch.Tensor]:
    if not segment_debug:
        return None

    composed = segment_debug.get("composed_total_pre_smooth")
    if composed is None:
        return None

    return _compute_reward(
        _weights_1d(composed),
        hmap_unified,
        denom_override=denom_override,
        normalization_mode=normalization_mode,
    )


def _compute_segment_winner(
    segment_rewards: List[Tuple[str, torch.Tensor]],
) -> Optional[np.ndarray]:
    if not segment_rewards:
        return None

    reward_stack = np.stack(
        [reward.detach().cpu().numpy().reshape(-1) for _, reward in segment_rewards],
        axis=0,
    )
    winner_idx = np.argmax(reward_stack, axis=0).astype(np.int32)
    winner_strength = np.max(reward_stack, axis=0)
    winner_idx[winner_strength <= 1e-12] = -1
    return winner_idx


# ── plotting ─────────────────────────────────────────────────────────────────

def _draw_notes_panel(ax, text: Optional[str], title: str = "Notes") -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)
    if not text:
        ax.text(
            0.02,
            0.98,
            "No route-selection summary saved.",
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
            va="top",
            ha="left",
        )
        return

    ax.text(
        0.02,
        0.98,
        textwrap.fill(text, width=42),
        transform=ax.transAxes,
        fontsize=8,
        family="monospace",
        color="black",
        va="top",
        ha="left",
        wrap=True,
    )


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


def _plot_owner_panel(
    ax,
    hmap_xy: np.ndarray,
    winner_idx: np.ndarray,
    source_names: List[str],
    goal_pos: Optional[Tuple[float, float]] = None,
    checkpoint_positions: Optional[List[Tuple[float, float]]] = None,
):
    hmap_x, hmap_y = hmap_xy[:, 0], hmap_xy[:, 1]
    n_sources = max(1, len(source_names))
    base_colors = list(plt.get_cmap("tab10").colors)
    colors = [base_colors[i % len(base_colors)] for i in range(n_sources)] + [(0.75, 0.75, 0.75)]
    cmap = ListedColormap(colors)

    color_idx = winner_idx.copy()
    color_idx[color_idx < 0] = n_sources
    ax.scatter(
        hmap_x,
        hmap_y,
        c=color_idx,
        s=6,
        cmap=cmap,
        alpha=0.75,
        linewidths=0.0,
    )

    if checkpoint_positions:
        for (cx, cy) in checkpoint_positions:
            ax.plot(cx, cy, marker="D", color="white",
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)

    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], marker="*", color="lime",
                markersize=14, markeredgecolor="black",
                markeredgewidth=0.8, zorder=6)

    legend_lines = [f"{idx}: {name}" for idx, name in enumerate(source_names)]
    legend_lines.append(f"{n_sources}: none")
    ax.text(
        0.02,
        0.02,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        color="black",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    ax.set_xlim(float(np.min(hmap_x)), float(np.max(hmap_x)))
    ax.set_ylim(float(np.min(hmap_y)), float(np.max(hmap_y)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Dominant Source", fontsize=10)


# ── main entry point ──────────────────────────────────────────────────────────

def _plot_reward_maps_legacy(
    scales: Optional[List[int]] = None,
    cmap_name: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    share_color_scale: bool = True,
):
    """
    Plot reward maps for all saved goal RCNs.

    Layout: rows = goals, cols = [scale_0, scale_1, ..., unified, notes].
    ★  marks the goal position.
    ◆  marks checkpoint positions.
    """
    if scales is None:
        scales = _discover_scales() or [0, 1, 2]

    goal_files = _discover_goal_rcn_files()
    if not goal_files:
        print("[WARNING] No unified goal RCN files found in", _multi_goal_dir())
        return
    goal_entries = [
        (goal_name, goal_path, _load_pickle(goal_path))
        for goal_name, goal_path in goal_files
    ]

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

    n_goals = len(goal_entries)
    n_scale_cols = len(scale_slices)
    max_segment_cols = max(
        len(_extract_segment_debug(goal_rcn).get("source_names", []))
        if _extract_segment_debug(goal_rcn)
        else 0
        for _, _, goal_rcn in goal_entries
    )
    has_owner_col = max_segment_cols > 0
    ncols = n_scale_cols + 2 + max_segment_cols + (1 if has_owner_col else 0)

    rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        n_goals, ncols,
        figsize=(
            4.2 * (n_scale_cols + 1 + max_segment_cols + (1 if has_owner_col else 0))
            + 3.8,
            4.0 * n_goals,
        ),
        squeeze=False,
    )
    env_label = WORLD_NAME.replace("_", " ").title()
    n_cp = len(checkpoint_pos)
    fig.suptitle(
        f"Reward Maps — {env_label}  "
        f"({'no checkpoints' if n_cp == 0 else f'{n_cp} checkpoint(s)'})",
        fontsize=13, y=1.01,
    )

    for row, (goal_name, goal_path, goal_rcn) in enumerate(goal_entries):
        goal_weights = _weights_1d(goal_rcn.w_in_effective)
        reward_mode = _resolved_reward_normalization_mode(goal_rcn)
        route_summary = _extract_route_selection_summary(goal_rcn)
        segment_debug = _extract_segment_debug(goal_rcn)
        if reward_mode == "input_l1":
            unified_denom = np.sum(np.abs(hmap_unified), axis=1)
        else:
            unified_denom = float(np.sum(np.abs(goal_weights)))

        row_rewards: List[torch.Tensor] = []
        for scale_idx, s_start, s_end in scale_slices:
            scale_weights = goal_weights[s_start:s_end]
            row_rewards.append(
                _compute_reward(
                    scale_weights,
                    hmap_by_scale[scale_idx][:min_steps],
                    denom_override=unified_denom,
                    normalization_mode=reward_mode,
                )
            )
        row_rewards.append(
            _compute_reward(
                goal_weights,
                hmap_unified,
                normalization_mode=reward_mode,
            )
        )
        segment_rewards = _compute_segment_rewards(
            segment_debug,
            hmap_unified,
            reward_mode,
        )
        row_rewards.extend([reward for _, reward in segment_rewards])

        row_vmin = 0.0 if share_color_scale else None
        row_vmax = None
        if share_color_scale:
            row_values = np.concatenate(
                [reward.detach().cpu().numpy().reshape(-1) for reward in row_rewards]
            )
            row_vmax = float(np.percentile(row_values, 99.5))
            row_vmax = max(row_vmax, 1e-12)

        # Per-scale columns
        for col, (scale_idx, s_start, s_end) in enumerate(scale_slices):
            ax = axes[row][col]
            reward_s = row_rewards[col]
            _plot_reward_hex(
                ax, hmap_xy, reward_s,
                title=f"{goal_name} | Scale {scale_idx}",
                cmap_name=cmap_name,
                vmin=row_vmin,
                vmax=row_vmax,
                goal_pos=goal_pos,
                checkpoint_positions=checkpoint_pos,
            )

        # Unified column
        ax_u = axes[row][n_scale_cols]
        reward_u = row_rewards[-1]
        _plot_reward_hex(
            ax_u, hmap_xy, reward_u,
            title=f"{goal_name} | Unified",
            cmap_name=cmap_name,
            vmin=row_vmin,
            vmax=row_vmax,
            goal_pos=goal_pos,
            checkpoint_positions=checkpoint_pos,
        )

        segment_start_col = n_scale_cols + 1
        for seg_col in range(max_segment_cols):
            ax_seg = axes[row][segment_start_col + seg_col]
            if seg_col >= len(segment_rewards):
                ax_seg.axis("off")
                continue
            source_name, segment_reward = segment_rewards[seg_col]
            _plot_reward_hex(
                ax_seg, hmap_xy, segment_reward,
                title=f"{goal_name} | {source_name} PreSmooth",
                cmap_name=cmap_name,
                vmin=row_vmin,
                vmax=row_vmax,
                goal_pos=goal_pos,
                checkpoint_positions=checkpoint_pos,
            )

        if has_owner_col:
            owner_col = segment_start_col + max_segment_cols
            ax_owner = axes[row][owner_col]
            winner_idx = _compute_segment_winner(segment_rewards)
            if winner_idx is None:
                ax_owner.axis("off")
            else:
                _plot_owner_panel(
                    ax_owner,
                    hmap_xy,
                    winner_idx,
                    [source_name for source_name, _ in segment_rewards],
                    goal_pos=goal_pos,
                    checkpoint_positions=checkpoint_pos,
                )

        notes_ax = axes[row][ncols - 1]
        _draw_notes_panel(notes_ax, route_summary)
        if route_summary:
            print(f"[RCN DEBUG] {goal_name}: {route_summary}")

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


def plot_reward_maps(
    scales: Optional[List[int]] = None,
    cmap_name: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    share_color_scale: bool = True,
    max_panels_per_row: int = 4,
):
    """
    Plot reward maps for all saved goal RCNs.

    Panels are wrapped across multiple rows per goal so segmented diagnostics
    do not force one very wide figure.
    """
    if scales is None:
        scales = _discover_scales() or [0, 1, 2]

    goal_files = _discover_goal_rcn_files()
    if not goal_files:
        print("[WARNING] No unified goal RCN files found in", _multi_goal_dir())
        return
    goal_entries = [
        (goal_name, goal_path, _load_pickle(goal_path))
        for goal_name, goal_path in goal_files
    ]

    hmap_xy = _load_hmap_loc()
    _, unified_pcn = _load_unified_layers()
    hmap_by_scale: Dict[int, np.ndarray] = {s: _load_scale_hmap(s) for s in scales}
    scale_slices = _build_scale_slices(unified_pcn, scales, hmap_by_scale)

    min_steps = min(len(hmap_xy), *(arr.shape[0] for arr in hmap_by_scale.values()))
    hmap_xy = hmap_xy[:min_steps]
    hmap_unified = np.concatenate(
        [hmap_by_scale[s][:min_steps] for s, _, _ in scale_slices], axis=1
    )

    goal_pos = get_env_goal_pos(WORLD_NAME)
    checkpoint_pos = get_env_checkpoint_positions(WORLD_NAME)

    rcParams.update({"font.size": 10})
    max_cols = max(1, int(max_panels_per_row))
    per_goal_payloads: List[Dict[str, object]] = []
    total_panel_rows = 0

    for goal_name, goal_path, goal_rcn in goal_entries:
        goal_weights = _weights_1d(goal_rcn.w_in_effective)
        reward_mode = _resolved_reward_normalization_mode(goal_rcn)
        route_summary = _extract_route_selection_summary(goal_rcn)
        segment_debug = _extract_segment_debug(goal_rcn)
        if reward_mode == "input_l1":
            unified_denom = np.sum(np.abs(hmap_unified), axis=1)
            segment_denom = None
        else:
            unified_denom = float(np.sum(np.abs(goal_weights)))
            segment_denom = unified_denom

        row_rewards: List[torch.Tensor] = []
        panels: List[Tuple[str, object]] = []
        for scale_idx, s_start, s_end in scale_slices:
            scale_reward = _compute_reward(
                goal_weights[s_start:s_end],
                hmap_by_scale[scale_idx][:min_steps],
                denom_override=unified_denom,
                normalization_mode=reward_mode,
            )
            row_rewards.append(scale_reward)
            panels.append(("reward", (f"{goal_name} | Scale {scale_idx}", scale_reward)))

        presmooth_unified = _compute_presmooth_unified_reward(
            segment_debug,
            hmap_unified,
            reward_mode,
            denom_override=segment_denom,
        )
        if presmooth_unified is not None:
            row_rewards.append(presmooth_unified)
            panels.append(
                ("reward", (f"{goal_name} | Unified PreSmooth", presmooth_unified))
            )

        unified_reward = _compute_reward(
            goal_weights,
            hmap_unified,
            normalization_mode=reward_mode,
        )
        row_rewards.append(unified_reward)
        panels.append(("reward", (f"{goal_name} | Unified Final", unified_reward)))

        segment_rewards = _compute_segment_rewards(
            segment_debug,
            hmap_unified,
            reward_mode,
            denom_override=segment_denom,
        )
        row_rewards.extend([reward for _, reward in segment_rewards])
        for source_name, segment_reward in segment_rewards:
            panels.append(
                (
                    "reward",
                    (
                        f"{goal_name} | {source_name} Contribution\n(PreSmooth)",
                        segment_reward,
                    ),
                )
            )

        winner_idx = _compute_segment_winner(segment_rewards)
        if winner_idx is not None:
            panels.append(
                (
                    "owner",
                    (
                        winner_idx,
                        [source_name for source_name, _ in segment_rewards],
                    ),
                )
            )

        notes_lines = [
            "Unified PreSmooth = composed map before final smoothing.",
            "Source Contribution = one source on the same reward scale as Unified.",
            "Unified Final = saved w_in_effective after final smoothing.",
        ]
        if route_summary:
            notes_lines.append(route_summary)
        panels.append(("notes", "\n".join(notes_lines)))

        row_vmin = 0.0 if share_color_scale else None
        row_vmax = None
        if share_color_scale and row_rewards:
            row_values = np.concatenate(
                [reward.detach().cpu().numpy().reshape(-1) for reward in row_rewards]
            )
            row_vmax = float(np.percentile(row_values, 99.5))
            row_vmax = max(row_vmax, 1e-12)

        panel_rows = max(1, int(math.ceil(len(panels) / max_cols)))
        total_panel_rows += panel_rows
        per_goal_payloads.append(
            {
                "goal_name": goal_name,
                "route_summary": route_summary,
                "panels": panels,
                "panel_rows": panel_rows,
                "row_vmin": row_vmin,
                "row_vmax": row_vmax,
            }
        )

    fig, axes = plt.subplots(
        total_panel_rows,
        max_cols,
        figsize=(4.8 * max_cols, 4.1 * total_panel_rows),
        squeeze=False,
    )
    env_label = WORLD_NAME.replace("_", " ").title()
    n_cp = len(checkpoint_pos)
    fig.suptitle(
        f"Reward Maps - {env_label}  "
        f"({'no checkpoints' if n_cp == 0 else f'{n_cp} checkpoint(s)'})",
        fontsize=13,
        y=1.01,
    )

    row_offset = 0
    for payload in per_goal_payloads:
        goal_name = str(payload["goal_name"])
        route_summary = payload["route_summary"]
        panels = list(payload["panels"])
        panel_rows = int(payload["panel_rows"])
        row_vmin = payload["row_vmin"]
        row_vmax = payload["row_vmax"]

        for row_in_goal in range(panel_rows):
            for col in range(max_cols):
                ax = axes[row_offset + row_in_goal][col]
                panel_idx = row_in_goal * max_cols + col
                if panel_idx >= len(panels):
                    ax.axis("off")
                    continue

                panel_kind, panel_payload = panels[panel_idx]
                if panel_kind == "reward":
                    title, reward = panel_payload
                    _plot_reward_hex(
                        ax,
                        hmap_xy,
                        reward,
                        title=title,
                        cmap_name=cmap_name,
                        vmin=row_vmin,
                        vmax=row_vmax,
                        goal_pos=goal_pos,
                        checkpoint_positions=checkpoint_pos,
                    )
                elif panel_kind == "owner":
                    winner_idx, source_names = panel_payload
                    _plot_owner_panel(
                        ax,
                        hmap_xy,
                        winner_idx,
                        source_names,
                        goal_pos=goal_pos,
                        checkpoint_positions=checkpoint_pos,
                    )
                    ax.set_title(f"{goal_name} | Dominant Source\n(PreSmooth)", fontsize=10)
                elif panel_kind == "notes":
                    _draw_notes_panel(ax, str(panel_payload), title=f"{goal_name} | Notes")
                else:
                    ax.axis("off")

        if route_summary:
            print(f"[RCN DEBUG] {goal_name}: {route_summary}")
        row_offset += panel_rows

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
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
