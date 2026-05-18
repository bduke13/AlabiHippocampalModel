# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
from matplotlib import cm, rcParams
from pathlib import Path
from typing import Dict, List, Optional

# Make the visualization package and project root importable when run as a script.
visualization_root = Path(__file__).resolve().parent
project_root = visualization_root.parent
sys.path.append(str(visualization_root))
sys.path.append(str(project_root))

from vis_utils import (
    convert_xzy_hmaps,
    discover_available_scales,
    get_output_dir,
    load_hmaps,
    load_layer_pkl,
    resolve_data_context,
)

FLIP_DISPLAY_Z_AXIS = True
PLOT_ALL_GOALS = True
GOAL_ORDER = ["yellow", "blue", "green", "red"]


def _rcn_has_signal(rcn) -> bool:
    weights = torch.as_tensor(
        getattr(rcn, "w_in_effective", getattr(rcn, "w_in", [])),
        dtype=torch.float32,
    ).detach().cpu()
    return bool(torch.any(torch.abs(weights) > 1e-9).item())


def _get_rcn_weights(rcn) -> torch.Tensor:
    return torch.as_tensor(
        getattr(rcn, "w_in_effective", getattr(rcn, "w_in", [])),
        dtype=torch.float32,
    )


def _discover_goal_specific_rcn_paths() -> Dict[str, Path]:
    _, _, data_root = resolve_data_context(required_relpaths=["hmaps/hmap_loc.pkl"])
    multi_goal_dir = data_root / "networks" / "multi_goal_rewards"
    goal_paths: Dict[str, Path] = {}
    if not multi_goal_dir.exists():
        return goal_paths

    for path in sorted(multi_goal_dir.glob("unified_rcn_goal_*.pkl")):
        goal_name = path.stem.replace("unified_rcn_goal_", "", 1)
        goal_paths[goal_name] = path
    return goal_paths


def _ordered_goal_names(goal_paths: Dict[str, Path]) -> List[str]:
    order = {goal_name: idx for idx, goal_name in enumerate(GOAL_ORDER)}
    return sorted(goal_paths, key=lambda goal_name: (order.get(goal_name, len(order)), goal_name))


def _display_coordinates(hmap_loc):
    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)
    if FLIP_DISPLAY_Z_AXIS:
        hmap_y = -hmap_y
    return hmap_x, hmap_y


def load_unified_rcn_data(goal_name: Optional[str] = None):
    """
    Load the unified RCN and the unified place-cell history.

    Returns:
        tuple: (rcn, hmap_loc, hmap_pcn, source_label) or (None, None, None, None) if loading fails
    """
    try:
        source_label = "default"
        if goal_name is not None:
            goal_paths = _discover_goal_specific_rcn_paths()
            if goal_name not in goal_paths:
                raise FileNotFoundError(
                    f"Goal-specific unified RCN '{goal_name}' not found. "
                    f"Available goals: {sorted(goal_paths)}"
                )
            with open(goal_paths[goal_name], "rb") as f:
                rcn = pickle.load(f)
            source_label = goal_name
        else:
            rcn = load_layer_pkl(layer_name="unified_rcn_goal")
            if not _rcn_has_signal(rcn):
                goal_paths = _discover_goal_specific_rcn_paths()
                if goal_paths:
                    fallback_goal = sorted(goal_paths)[0]
                    with open(goal_paths[fallback_goal], "rb") as f:
                        rcn = pickle.load(f)
                    source_label = fallback_goal
                    print(
                        f"[WARN] unified_rcn_goal.pkl is empty; "
                        f"falling back to goal-specific map '{fallback_goal}'."
                    )

        hmap_loc, hmap_pcn = load_hmaps(["hmap_loc", "hmap_pcn"])
        print(f"Successfully loaded unified RCN data ({source_label})")
        return rcn, hmap_loc, hmap_pcn, source_label
    except Exception as exc:
        print(f"Error loading unified RCN data: {exc}")
        return None, None, None, None


def _resolve_scale_entries(rcn, scales: Optional[List[int]] = None) -> List[Dict]:
    """Map requested scale indices to slices within the unified reward/PC vectors."""
    boundaries = list(getattr(rcn, "scale_boundaries", []))
    if len(boundaries) < 2:
        raise ValueError("Unified RCN does not expose valid scale boundaries")

    scale_configs = list(getattr(rcn, "scale_configs", []) or [])
    entries = []
    for block_idx in range(len(boundaries) - 1):
        cfg = scale_configs[block_idx] if block_idx < len(scale_configs) else {}
        scale_idx = int(cfg.get("scale_index", block_idx))
        scale_name = str(cfg.get("name", f"scale_{scale_idx}"))
        start, end = int(boundaries[block_idx]), int(boundaries[block_idx + 1])
        entries.append(
            {
                "block_idx": block_idx,
                "scale_idx": scale_idx,
                "scale_name": scale_name,
                "start": start,
                "end": end,
            }
        )

    if scales is None:
        return entries

    requested = set(int(scale) for scale in scales)
    filtered = [entry for entry in entries if entry["scale_idx"] in requested]
    if not filtered:
        raise ValueError(f"Requested scales {sorted(requested)} are not present in the unified RCN")
    return filtered


def compute_reward_function(weight_block, hmap_pcn_block, denom_source=None):
    """
    Compute reward values from a weight block and matching PC activations.

    Args:
        weight_block: Reward weights for one scale block or the full unified map.
        hmap_pcn_block: Matching place-cell activity history, shape (num_steps, num_pc_block).
        denom_source: Optional activation history used for normalization. If omitted,
            normalization uses hmap_pcn_block itself.

    Returns:
        torch.Tensor: Reward values for each timestep.
    """
    w_in_float32 = torch.as_tensor(weight_block, dtype=torch.float32).clone().detach().cpu().view(1, -1)
    hmap_pcn_float32 = torch.as_tensor(hmap_pcn_block, dtype=torch.float32).cpu()
    denom_float32 = (
        torch.as_tensor(denom_source, dtype=torch.float32).cpu()
        if denom_source is not None
        else hmap_pcn_float32
    )

    hmap_pcn_t = hmap_pcn_float32.T
    denom_t = denom_float32.T

    if w_in_float32.shape[1] != hmap_pcn_t.shape[0]:
        raise ValueError(
            f"Dimension mismatch: reward weights expect {w_in_float32.shape[1]} place cells, "
            f"but the provided history block has {hmap_pcn_t.shape[0]}."
        )

    sum_activations = torch.sum(denom_t, dim=0)
    safe_denom = torch.where(sum_activations > 0, sum_activations, torch.ones_like(sum_activations))
    reward_function = torch.tensordot(w_in_float32, hmap_pcn_t, dims=1) / safe_denom
    return torch.squeeze(reward_function)


def plot_rcn_activation_single(
    weight_block,
    hmap_pcn_block,
    hmap_x,
    hmap_y,
    scale_label,
    ax=None,
    cmap_name="plasma",
    show_colorbar=True,
    denom_source=None,
):
    """
    Plot a unified-RCN reward contribution map for one scale slice.
    """
    if ax is None:
        _, ax = plt.subplots(dpi=150)

    reward_function = compute_reward_function(
        weight_block=weight_block,
        hmap_pcn_block=hmap_pcn_block,
        denom_source=denom_source,
    )

    if reward_function.shape != hmap_x.shape:
        raise ValueError(
            f"Shape mismatch: reward_function has shape {reward_function.shape}, "
            f"but expected {hmap_x.shape}."
        )

    x_min, x_max = np.min(hmap_x), np.max(hmap_x)
    y_min, y_max = np.min(hmap_y), np.max(hmap_y)
    cmap = cm.get_cmap(cmap_name)

    cntr = ax.hexbin(
        hmap_x,
        hmap_y,
        reward_function.detach().cpu().numpy(),
        gridsize=100,
        cmap=cmap,
        alpha=0.6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(str(scale_label))

    if show_colorbar:
        plt.colorbar(cntr, ax=ax)

    return ax, cntr


def plot_all_scales(
    scales=None,
    goal_name: Optional[str] = None,
    cmap_name="plasma",
    save_path=None,
    show_plot=True,
    include_unified_total: bool = True,
):
    """
    Plot the unified reward map and per-scale contributions side by side.
    """
    rcn, hmap_loc, unified_hmap_pcn, source_label = load_unified_rcn_data(goal_name=goal_name)
    if rcn is None:
        return

    scale_entries = _resolve_scale_entries(rcn, scales=scales)
    hmap_x, hmap_y = _display_coordinates(hmap_loc)
    full_weights = _get_rcn_weights(rcn)
    num_panels = len(scale_entries) + (1 if include_unified_total else 0)

    fig, axes = plt.subplots(
        1,
        max(1, num_panels),
        figsize=(5 * max(1, num_panels), 5),
        dpi=150,
        squeeze=False,
    )
    fig.suptitle(
        f"Unified Reward Map Overview"
        f"{'' if source_label in (None, 'default') else f' - {source_label}'}",
        fontsize=16,
    )
    rcParams.update({"font.size": 12})

    panel_axes = list(axes[0])
    if include_unified_total:
        unified_ax = panel_axes.pop(0)
        plot_rcn_activation_single(
            weight_block=full_weights,
            hmap_pcn_block=unified_hmap_pcn,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale_label="unified total",
            ax=unified_ax,
            cmap_name=cmap_name,
            show_colorbar=True,
            denom_source=unified_hmap_pcn,
        )
        if source_label in (None, "default"):
            unified_ax.set_title("Unified RCN Reward Map")
        else:
            unified_ax.set_title(f"Unified RCN Reward Map - {source_label}")

    for ax, entry in zip(panel_axes, scale_entries):
        start, end = entry["start"], entry["end"]
        weight_block = full_weights[:, start:end]
        hmap_pcn_block = unified_hmap_pcn[:, start:end]
        scale_label = f"{entry['scale_name']} ({entry['scale_idx']})"
        plot_rcn_activation_single(
            weight_block=weight_block,
            hmap_pcn_block=hmap_pcn_block,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale_label=scale_label,
            ax=ax,
            cmap_name=cmap_name,
            show_colorbar=True,
            denom_source=None,
        )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_all_goals(
    scales=None,
    cmap_name="plasma",
    save_path=None,
    show_plot=True,
    include_unified_total: bool = True,
):
    """
    Plot goal-specific unified reward maps for every saved goal.
    """
    goal_paths = _discover_goal_specific_rcn_paths()
    goal_names = _ordered_goal_names(goal_paths)
    if not goal_names:
        print("[WARN] No goal-specific RCN maps found; plotting the default unified RCN.")
        plot_all_scales(
            scales=scales,
            goal_name=None,
            cmap_name=cmap_name,
            save_path=save_path,
            show_plot=show_plot,
            include_unified_total=include_unified_total,
        )
        return

    rows = []
    max_panels = 0
    for goal_name in goal_names:
        rcn, hmap_loc, unified_hmap_pcn, _ = load_unified_rcn_data(goal_name=goal_name)
        if rcn is None:
            continue

        scale_entries = _resolve_scale_entries(rcn, scales=scales)
        full_weights = _get_rcn_weights(rcn)
        hmap_x, hmap_y = _display_coordinates(hmap_loc)
        rows.append(
            {
                "goal_name": goal_name,
                "scale_entries": scale_entries,
                "full_weights": full_weights,
                "hmap_pcn": unified_hmap_pcn,
                "hmap_x": hmap_x,
                "hmap_y": hmap_y,
            }
        )
        max_panels = max(max_panels, len(scale_entries) + (1 if include_unified_total else 0))

    if not rows:
        print("[WARN] No valid goal-specific RCN maps could be loaded.")
        return

    fig, axes = plt.subplots(
        len(rows),
        max(1, max_panels),
        figsize=(5 * max(1, max_panels), 4.5 * len(rows)),
        dpi=150,
        squeeze=False,
    )
    fig.suptitle("Unified Reward Map Overview - All Goals", fontsize=16)
    rcParams.update({"font.size": 12})

    for row_idx, row in enumerate(rows):
        panel_axes = list(axes[row_idx])
        goal_name = row["goal_name"]
        col_idx = 0

        if include_unified_total:
            ax = panel_axes[col_idx]
            plot_rcn_activation_single(
                weight_block=row["full_weights"],
                hmap_pcn_block=row["hmap_pcn"],
                hmap_x=row["hmap_x"],
                hmap_y=row["hmap_y"],
                scale_label="unified total",
                ax=ax,
                cmap_name=cmap_name,
                show_colorbar=True,
                denom_source=row["hmap_pcn"],
            )
            ax.set_title(f"{goal_name} - unified total")
            col_idx += 1

        for entry in row["scale_entries"]:
            ax = panel_axes[col_idx]
            start, end = entry["start"], entry["end"]
            plot_rcn_activation_single(
                weight_block=row["full_weights"][:, start:end],
                hmap_pcn_block=row["hmap_pcn"][:, start:end],
                hmap_x=row["hmap_x"],
                hmap_y=row["hmap_y"],
                scale_label=f"{entry['scale_name']} ({entry['scale_idx']})",
                ax=ax,
                cmap_name=cmap_name,
                show_colorbar=True,
                denom_source=None,
            )
            ax.set_title(f"{goal_name} - {entry['scale_name']} ({entry['scale_idx']})")
            col_idx += 1

        for ax in panel_axes[col_idx:]:
            ax.set_axis_off()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_rcn_plots(
    scales=None,
    goal_name: Optional[str] = None,
    cmap_name="plasma",
):
    """
    Generate and save per-scale unified-RCN contribution plots.
    """
    output_dir = os.path.join(get_output_dir(), "rcn_plots")
    os.makedirs(output_dir, exist_ok=True)

    if PLOT_ALL_GOALS and goal_name is None:
        goal_paths = _discover_goal_specific_rcn_paths()
        if goal_paths:
            plot_all_goals(
                scales=scales or discover_available_scales() or None,
                cmap_name=cmap_name,
                save_path=os.path.join(output_dir, "rcn_all_goals_all_scales.png"),
                show_plot=False,
                include_unified_total=True,
            )
            return

    rcn, hmap_loc, unified_hmap_pcn, source_label = load_unified_rcn_data(goal_name=goal_name)
    if rcn is None:
        return

    scale_entries = _resolve_scale_entries(rcn, scales=scales or discover_available_scales() or None)
    hmap_x, hmap_y = _display_coordinates(hmap_loc)

    plot_all_scales(
        scales=[entry["scale_idx"] for entry in scale_entries],
        goal_name=goal_name,
        cmap_name=cmap_name,
        save_path=os.path.join(output_dir, "rcn_all_scales.png"),
        show_plot=False,
        include_unified_total=True,
    )

    full_weights = _get_rcn_weights(rcn)
    fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
    plot_rcn_activation_single(
        weight_block=full_weights,
        hmap_pcn_block=unified_hmap_pcn,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        scale_label="unified total",
        ax=ax,
        cmap_name=cmap_name,
        show_colorbar=True,
        denom_source=unified_hmap_pcn,
    )
    if source_label in (None, "default"):
        ax.set_title("Unified RCN Reward Map", fontsize=14)
        unified_save_path = os.path.join(output_dir, "rcn_unified_total.png")
    else:
        ax.set_title(f"Unified RCN Reward Map - {source_label}", fontsize=14)
        unified_save_path = os.path.join(output_dir, f"rcn_unified_total_{source_label}.png")
    plt.savefig(unified_save_path, dpi=300, bbox_inches="tight")
    print(f"Saved unified reward map to {unified_save_path}")
    plt.close(fig)

    for entry in scale_entries:
        start, end = entry["start"], entry["end"]
        weight_block = full_weights[:, start:end]
        hmap_pcn_block = unified_hmap_pcn[:, start:end]

        fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
        plot_rcn_activation_single(
            weight_block=weight_block,
            hmap_pcn_block=hmap_pcn_block,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale_label=f"{entry['scale_name']} ({entry['scale_idx']})",
            ax=ax,
            cmap_name=cmap_name,
            show_colorbar=True,
            denom_source=None,
        )
        if source_label in (None, "default"):
            ax.set_title(f"Unified RCN Contribution - {entry['scale_name']}", fontsize=14)
            save_path = os.path.join(output_dir, f"rcn_scale_{entry['scale_idx']}.png")
        else:
            ax.set_title(
                f"Unified RCN Contribution - {entry['scale_name']} - {source_label}",
                fontsize=14,
            )
            save_path = os.path.join(output_dir, f"rcn_scale_{entry['scale_idx']}_{source_label}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved scale contribution plot to {save_path}")
        plt.close(fig)


# %%
if __name__ == "__main__":
    print("Starting unified multi-scale RCN visualization...")

    scales = discover_available_scales() or None
    goal_name = None
    print(f"Discovered scales: {scales}")
    print(f"Using output directory: {get_output_dir()}")

    if PLOT_ALL_GOALS and goal_name is None:
        plot_all_goals(
            scales=scales,
            cmap_name="plasma",
            save_path=os.path.join(get_output_dir(), "rcn_all_goals_all_scales.png"),
            show_plot=False,
            include_unified_total=True,
        )
    else:
        plot_all_scales(
            scales=scales,
            goal_name=goal_name,
            cmap_name="plasma",
            save_path=os.path.join(get_output_dir(), "rcn_all_scales.png"),
            show_plot=False,
            include_unified_total=True,
        )

    print("Unified RCN visualization complete!")
