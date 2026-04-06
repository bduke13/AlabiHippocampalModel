"""
Reward-map visualization for masked checkpoint-room contributions.

Each panel shows the saved segmented unified RCN contribution for one
checkpoint source (`cp0`, `cp1`, ...) after the room / directional replay mask
has been applied. It also plots the goal source and the full unified goal map
with the exploit-time room mask applied at the goal position. The spatial
plotting style matches `multi_rcn_plot.py`.
"""
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.robot.webots_worlds import get_world_obstacles
from multi_rcn_plot import (
    _build_scale_slices,
    _compute_reward,
    _discover_goal_rcn_files,
    _discover_scales,
    _extract_segment_debug,
    _load_hmap_loc,
    _load_pickle,
    _load_scale_hmap,
    _load_unified_layers,
    _plot_reward_hex,
    _resolved_reward_normalization_mode,
    _weights_1d,
)
from vis_utils import (
    CONTROLLER_NAME,
    CONTROLLER_PATH_PREFIX,
    OUTPUT_DIR,
    WORLD_NAME,
    get_env_checkpoint_positions,
    get_env_goal_pos,
)


def _network_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"


def _multi_goal_dir() -> Path:
    return _network_dir() / "multi_goal_rewards"


def _load_goal_associations() -> dict:
    assoc_path = _multi_goal_dir() / "goal_associations.pkl"
    with open(assoc_path, "rb") as f:
        return pickle.load(f)


def _checkpoint_title(source_name: str) -> str:
    source_name = str(source_name)
    if source_name.startswith("cp") and source_name[2:].isdigit():
        return f"checkpoint_{source_name[2:]}"
    return source_name


def _compute_unified_pc_centers(
    hmap_xy: np.ndarray,
    hmap_by_scale: Dict[int, np.ndarray],
    unified_pcn,
    scale_slices: List[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pc_total = int(unified_pcn.num_pc_total)
    centers = torch.zeros((num_pc_total, 2), dtype=torch.float32)
    visited = torch.zeros(num_pc_total, dtype=torch.bool)
    hmap_x = torch.as_tensor(hmap_xy[:, 0], dtype=torch.float32)
    hmap_y = torch.as_tensor(hmap_xy[:, 1], dtype=torch.float32)

    for scale_idx, s_start, s_end in scale_slices:
        acts = torch.as_tensor(hmap_by_scale[scale_idx], dtype=torch.float32)
        acts_sum = acts.sum(dim=0)
        valid = acts_sum > 0.05
        if not bool(torch.any(valid).item()):
            continue
        denom = acts_sum.clamp(min=1e-12)
        centers[s_start:s_end, 0] = torch.matmul(acts.T, hmap_x) / denom
        centers[s_start:s_end, 1] = torch.matmul(acts.T, hmap_y) / denom
        visited[s_start:s_end] = valid
    return centers, visited


def _extract_selected_checkpoint_order_from_goal_map_log(debug_log: str) -> List[int]:
    text = str(debug_log or "")
    marker = "selected=["
    start = text.find(marker)
    if start < 0:
        return []
    end = text.find("]", start + len(marker))
    if end < 0:
        return []
    raw = text[start + len(marker) : end].strip()
    if not raw:
        return []

    parsed: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token.startswith("cp"):
            token = token[2:]
        try:
            parsed.append(int(token))
        except ValueError:
            continue
    return parsed


def _goal_location_from_assoc(assoc_data: dict, goal_name: str) -> Optional[Tuple[float, float]]:
    for goal in assoc_data.get("goals", []):
        if str(goal.get("name")) == str(goal_name):
            loc = goal.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
    goal_pos = get_env_goal_pos(WORLD_NAME)
    if goal_pos is None:
        return None
    return float(goal_pos[0]), float(goal_pos[1])


def _infer_checkpoint_support_region(
    checkpoint_x: float,
    checkpoint_y: float,
):
    obstacles = get_world_obstacles(WORLD_NAME)
    if not obstacles:
        return None

    tol = 0.25
    candidates = []
    rects = []
    for obstacle in obstacles:
        if obstacle.get("type") != "rectangle":
            continue
        bounds = obstacle.get("bounds")
        if not bounds:
            continue
        xmin = min(bounds[0][0], bounds[1][0])
        xmax = max(bounds[0][0], bounds[1][0])
        ymin = min(bounds[0][1], bounds[1][1])
        ymax = max(bounds[0][1], bounds[1][1])
        width = max(1e-6, xmax - xmin)
        height = max(1e-6, ymax - ymin)
        orientation = "vertical" if height >= width else "horizontal"
        rects.append((orientation, xmin, xmax, ymin, ymax))

    for idx_a in range(len(rects)):
        orient_a, ax0, ax1, ay0, ay1 = rects[idx_a]
        for idx_b in range(idx_a + 1, len(rects)):
            orient_b, bx0, bx1, by0, by1 = rects[idx_b]
            if orient_a != orient_b:
                continue
            if orient_a == "vertical":
                overlap_min = max(ax0, bx0)
                overlap_max = min(ax1, bx1)
                if overlap_max < overlap_min - 1e-6:
                    continue
                if ay1 <= by0:
                    gap_min, gap_max = ay1, by0
                elif by1 <= ay0:
                    gap_min, gap_max = by1, ay0
                else:
                    continue
                if not (gap_min - tol <= checkpoint_y <= gap_max + tol):
                    continue
                if not (overlap_min - tol <= checkpoint_x <= overlap_max + tol):
                    continue
                axis_center = 0.5 * (overlap_min + overlap_max)
                half_width = max(0.15, 0.5 * max(0.0, overlap_max - overlap_min) + 0.05)
                score = abs(checkpoint_x - axis_center) + 0.25 * abs(
                    checkpoint_y - (0.5 * (gap_min + gap_max))
                )
                candidates.append(
                    (
                        score,
                        {
                            "kind": "segment",
                            "orientation": "vertical",
                            "p0": (axis_center, gap_min),
                            "p1": (axis_center, gap_max),
                            "half_width": half_width,
                        },
                    )
                )
            else:
                overlap_min = max(ay0, by0)
                overlap_max = min(ay1, by1)
                if overlap_max < overlap_min - 1e-6:
                    continue
                if ax1 <= bx0:
                    gap_min, gap_max = ax1, bx0
                elif bx1 <= ax0:
                    gap_min, gap_max = bx1, ax0
                else:
                    continue
                if not (gap_min - tol <= checkpoint_x <= gap_max + tol):
                    continue
                if not (overlap_min - tol <= checkpoint_y <= overlap_max + tol):
                    continue
                axis_center = 0.5 * (overlap_min + overlap_max)
                half_width = max(0.15, 0.5 * max(0.0, overlap_max - overlap_min) + 0.05)
                score = abs(checkpoint_y - axis_center) + 0.25 * abs(
                    checkpoint_x - (0.5 * (gap_min + gap_max))
                )
                candidates.append(
                    (
                        score,
                        {
                            "kind": "segment",
                            "orientation": "horizontal",
                            "p0": (gap_min, axis_center),
                            "p1": (gap_max, axis_center),
                            "half_width": half_width,
                        },
                    )
                )

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _support_region_frame(support_region):
    if support_region is None or support_region.get("kind") != "segment":
        return None
    x0, y0 = support_region["p0"]
    x1, y1 = support_region["p1"]
    vx = float(x1) - float(x0)
    vy = float(y1) - float(y0)
    norm = math.hypot(vx, vy)
    if norm <= 1e-9:
        return None
    tx = vx / norm
    ty = vy / norm
    nx = -ty
    ny = tx
    return {
        "center": (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1))),
        "tangent": (tx, ty),
        "normal": (nx, ny),
        "half_width": float(max(0.0, support_region.get("half_width", 0.0))),
    }


def _checkpoint_support_side_from_point(
    checkpoint_positions: List[Tuple[float, float]],
    checkpoint_idx: int,
    point_x: float,
    point_y: float,
    fallback_side: int = 0,
):
    if checkpoint_idx < 0 or checkpoint_idx >= len(checkpoint_positions):
        return int(fallback_side), None, None
    cx, cy = checkpoint_positions[checkpoint_idx]
    support_region = _infer_checkpoint_support_region(float(cx), float(cy))
    frame = _support_region_frame(support_region)
    if frame is None:
        return int(fallback_side), support_region, None
    frame_cx, frame_cy = frame["center"]
    nx, ny = frame["normal"]
    signed = ((float(point_x) - frame_cx) * nx) + ((float(point_y) - frame_cy) * ny)
    side_eps = max(0.05, float(frame.get("half_width", 0.0)) + 0.03)
    if signed > side_eps:
        return 1, support_region, frame
    if signed < -side_eps:
        return -1, support_region, frame
    return int(fallback_side), support_region, frame


def _build_goal_position_room_mask(
    goal_name: str,
    goal_rcn,
    assoc_data: dict,
    unified_pcn,
    centers: torch.Tensor,
    visited: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    checkpoint_positions = [
        (float(pos[0]), float(pos[1]))
        for pos in assoc_data.get("checkpoints", [])
        if isinstance(pos, (list, tuple)) and len(pos) >= 2
    ]
    num_pc = int(unified_pcn.num_pc_total)
    full_mask = torch.ones(num_pc, dtype=torch.float32)
    if not checkpoint_positions:
        return full_mask, "roommask=no_checkpoints"

    goal_loc = _goal_location_from_assoc(assoc_data, goal_name)
    if goal_loc is None:
        return full_mask, "roommask=no_goal_loc"
    goal_x, goal_y = goal_loc

    debug_log = str(getattr(goal_rcn, "goal_map_debug_log", "") or "")
    chain_near_to_far = _extract_selected_checkpoint_order_from_goal_map_log(debug_log)
    if not chain_near_to_far:
        indexed = [
            (i, math.hypot(float(cx) - goal_x, float(cy) - goal_y))
            for i, (cx, cy) in enumerate(checkpoint_positions)
        ]
        indexed.sort(key=lambda item: item[1])
        chain_near_to_far = [i for i, _ in indexed]
    if not chain_near_to_far:
        return full_mask, "roommask=no_chain"
    chain = list(reversed(chain_near_to_far))
    masked_logs: List[str] = []

    for chain_pos, ck_idx in enumerate(chain):
        if ck_idx < 0 or ck_idx >= len(checkpoint_positions):
            continue
        cx, cy = checkpoint_positions[ck_idx]
        support_region = _infer_checkpoint_support_region(float(cx), float(cy))
        frame = _support_region_frame(support_region)
        if frame is None:
            continue

        if chain_pos + 1 < len(chain):
            downstream_idx = int(chain[chain_pos + 1])
            if downstream_idx < 0 or downstream_idx >= len(checkpoint_positions):
                downstream_target_xy = (goal_x, goal_y)
                target_label = "goal"
            else:
                downstream_target_xy = checkpoint_positions[downstream_idx]
                target_label = f"cp{downstream_idx}"
        else:
            downstream_target_xy = (goal_x, goal_y)
            target_label = "goal"

        frame_cx, frame_cy = frame["center"]
        nx, ny = frame["normal"]
        doorway_eps = max(0.15, float(frame.get("half_width", 0.0)) + 0.12)
        downstream_signed = (
            (float(downstream_target_xy[0]) - frame_cx) * nx
            + (float(downstream_target_xy[1]) - frame_cy) * ny
        )
        if abs(float(downstream_signed)) <= max(0.05, 0.5 * doorway_eps):
            continue

        downstream_side = 1.0 if downstream_signed > 0.0 else -1.0
        agent_side, _, _ = _checkpoint_support_side_from_point(
            checkpoint_positions,
            ck_idx,
            goal_x,
            goal_y,
            fallback_side=0,
        )
        if float(agent_side) != downstream_side:
            continue

        signed = (((centers[:, 0] - frame_cx) * nx) + ((centers[:, 1] - frame_cy) * ny))
        behind_mask = (signed * downstream_side <= doorway_eps) & visited
        if not bool(torch.any(behind_mask).item()):
            continue

        full_mask[behind_mask] = 0.0
        masked_logs.append(
            f"cp{ck_idx}(n={int(torch.count_nonzero(behind_mask).item())},"
            f"target={target_label},dir={'+' if downstream_side > 0.0 else '-'},src=geom)"
        )

    if not masked_logs:
        return full_mask, "roommask=all_clear"
    return full_mask, f"roommask=[{','.join(masked_logs)}]"


def _compute_plot_panels(
    goal_rcn,
    goal_name: str,
    assoc_data: dict,
    unified_pcn,
    hmap_xy: np.ndarray,
    hmap_by_scale: Dict[int, np.ndarray],
    scale_slices: List[Tuple[int, int, int]],
    hmap_unified: np.ndarray,
) -> List[Tuple[str, torch.Tensor, Dict[str, object]]]:
    segment_debug = _extract_segment_debug(goal_rcn)
    if not segment_debug:
        print(
            "[WARNING] No goal_map_segment_debug in artifact; "
            "showing only full goal panels."
        )
    reward_mode = _resolved_reward_normalization_mode(goal_rcn)
    goal_weights = _weights_1d(goal_rcn.w_in_effective)
    if reward_mode == "input_l1":
        source_denom_override = None
    else:
        source_denom_override = float(np.sum(np.abs(goal_weights)))
    total_maps = segment_debug.get("total_maps_pre_smooth", {}) if segment_debug else {}
    metadata_map = segment_debug.get("metadata", {}) if segment_debug else {}
    centers, visited = _compute_unified_pc_centers(
        hmap_xy,
        hmap_by_scale,
        unified_pcn,
        scale_slices,
    )

    panels: List[Tuple[str, torch.Tensor, Dict[str, object]]] = []

    goal_final_reward = _compute_reward(
        goal_weights,
        hmap_unified,
        normalization_mode=reward_mode,
    )
    panels.append(("goal_final", goal_final_reward, {"kind": "goal_final"}))

    if isinstance(total_maps, dict):
        goal_weights_obj = total_maps.get("goal")
        if goal_weights_obj is not None:
            goal_segment_reward = _compute_reward(
                _weights_1d(goal_weights_obj),
                hmap_unified,
                denom_override=source_denom_override,
                normalization_mode=reward_mode,
            )
            panels.append(
                (
                    "goal_source",
                    goal_segment_reward,
                    metadata_map.get("goal", {}) if isinstance(metadata_map.get("goal", {}), dict) else {},
                )
            )

    goal_room_mask, roommask_log = _build_goal_position_room_mask(
        goal_name=goal_name,
        goal_rcn=goal_rcn,
        assoc_data=assoc_data,
        unified_pcn=unified_pcn,
        centers=centers,
        visited=visited,
    )
    masked_goal_weights = goal_weights * goal_room_mask.detach().cpu().numpy().reshape(-1)
    goal_masked_reward = _compute_reward(
        masked_goal_weights,
        hmap_unified,
        denom_override=source_denom_override,
        normalization_mode=reward_mode,
    )
    panels.append(
        (
            "goal_masked_at_goal",
            goal_masked_reward,
            {
                "kind": "goal_masked_at_goal",
                "roommask_log": roommask_log,
                "masked_count": int(torch.count_nonzero(goal_room_mask <= 1e-8).item()),
            },
        )
    )
    if isinstance(total_maps, dict):
        goal_weights_obj = total_maps.get("goal")
        if goal_weights_obj is not None:
            masked_goal_source_weights = (
                _weights_1d(goal_weights_obj)
                * goal_room_mask.detach().cpu().numpy().reshape(-1)
            )
            goal_source_masked_reward = _compute_reward(
                masked_goal_source_weights,
                hmap_unified,
                denom_override=source_denom_override,
                normalization_mode=reward_mode,
            )
            panels.append(
                (
                    "goal_source_masked_at_goal",
                    goal_source_masked_reward,
                    {
                        "kind": "goal_masked_at_goal",
                        "roommask_log": roommask_log,
                        "masked_count": int(torch.count_nonzero(goal_room_mask <= 1e-8).item()),
                    },
                )
            )

    if isinstance(total_maps, dict) and segment_debug:
        for source_name in segment_debug.get("source_names", []):
            source_name = str(source_name)
            if not source_name.startswith("cp"):
                continue
            weights_obj = total_maps.get(source_name)
            if weights_obj is None:
                continue
            reward = _compute_reward(
                _weights_1d(weights_obj),
                hmap_unified,
                denom_override=source_denom_override,
                normalization_mode=reward_mode,
            )
            metadata = metadata_map.get(source_name, {})
            panels.append((source_name, reward, metadata if isinstance(metadata, dict) else {}))

    return panels


def _panel_title(panel_name: str) -> str:
    panel_name = str(panel_name)
    if panel_name == "goal_final":
        return "goal_final"
    if panel_name == "goal_source":
        return "goal_source"
    if panel_name == "goal_masked_at_goal":
        return "goal_final_masked_at_goal"
    if panel_name == "goal_source_masked_at_goal":
        return "goal_source_masked_at_goal"
    return _checkpoint_title(panel_name)


def _panel_overlay_text(metadata: Dict[str, object]) -> Optional[str]:
    kind = str(metadata.get("kind", "")).strip().lower()
    if kind == "goal_masked_at_goal":
        masked_count = metadata.get("masked_count", "?")
        roommask_log = str(metadata.get("roommask_log", "roommask=?"))
        return f"masked={masked_count}\n{roommask_log}"

    support_count = metadata.get("support_count", None)
    replay_count = metadata.get("replay_count", None)
    source_role = str(metadata.get("source_role", "")).strip()
    sink_type = str(metadata.get("sink_type", "")).strip()
    seed_source = str(metadata.get("seed_source", "")).strip()
    if support_count is not None or replay_count is not None or source_role:
        support_txt = "?" if support_count is None else int(support_count)
        replay_txt = "?" if replay_count is None else int(replay_count)
        prefix = ""
        if source_role:
            prefix = source_role
            if sink_type:
                prefix = f"{prefix}:{sink_type}"
        suffix = f"\nseed={seed_source}" if seed_source else ""
        if prefix:
            return f"{prefix}\nsupport={support_txt} replay={replay_txt}{suffix}"
        return f"support={support_txt} replay={replay_txt}{suffix}"

    return None


def _finalize_figure(fig, show_plot: bool) -> None:
    backend_name = str(plt.get_backend())
    if show_plot and "agg" not in backend_name.lower():
        plt.show()
    else:
        plt.close(fig)


def plot_masked_checkpoint_room_maps(
    scales: Optional[List[int]] = None,
    cmap_name: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    share_color_scale: bool = True,
    max_panels_per_row: int = 4,
):
    if scales is None:
        scales = _discover_scales() or [0, 1, 2]

    goal_files = _discover_goal_rcn_files()
    if not goal_files:
        print("[WARNING] No unified goal RCN files found.")
        return

    goal_entries = [
        (goal_name, goal_path, _load_pickle(goal_path))
        for goal_name, goal_path in goal_files
    ]

    hmap_xy = _load_hmap_loc()
    _, unified_pcn = _load_unified_layers()
    hmap_by_scale: Dict[int, np.ndarray] = {s: _load_scale_hmap(s) for s in scales}
    scale_slices = _build_scale_slices(unified_pcn, scales, hmap_by_scale)
    assoc_data = _load_goal_associations()

    min_steps = min(len(hmap_xy), *(arr.shape[0] for arr in hmap_by_scale.values()))
    hmap_xy = hmap_xy[:min_steps]
    hmap_by_scale = {s: arr[:min_steps] for s, arr in hmap_by_scale.items()}
    hmap_unified = np.concatenate(
        [hmap_by_scale[s][:min_steps] for s, _, _ in scale_slices], axis=1
    )

    goal_pos = get_env_goal_pos(WORLD_NAME)
    checkpoint_pos = get_env_checkpoint_positions(WORLD_NAME)
    max_cols = max(1, int(max_panels_per_row))

    payloads: List[Dict[str, object]] = []
    total_panel_rows = 0

    for goal_name, _, goal_rcn in goal_entries:
        panels = _compute_plot_panels(
            goal_rcn=goal_rcn,
            goal_name=str(goal_name),
            assoc_data=assoc_data,
            unified_pcn=unified_pcn,
            hmap_xy=hmap_xy,
            hmap_by_scale=hmap_by_scale,
            scale_slices=scale_slices,
            hmap_unified=hmap_unified,
        )
        if not panels:
            continue

        row_vmin = 0.0 if share_color_scale else None
        row_vmax = None
        if share_color_scale:
            row_values = np.concatenate(
                [reward.detach().cpu().numpy().reshape(-1) for _, reward, _ in panels]
            )
            row_vmax = float(np.percentile(row_values, 99.5))
            row_vmax = max(row_vmax, 1e-12)

        panel_rows = max(1, int(math.ceil(len(panels) / max_cols)))
        total_panel_rows += panel_rows
        payloads.append(
            {
                "goal_name": str(goal_name),
                "panels": panels,
                "panel_rows": panel_rows,
                "row_vmin": row_vmin,
                "row_vmax": row_vmax,
            }
        )

    if not payloads:
        print(
            "[WARNING] No goal/checkpoint contributions found in goal_map_segment_debug."
        )
        return

    fig, axes = plt.subplots(
        total_panel_rows,
        max_cols,
        figsize=(4.8 * max_cols, 4.1 * total_panel_rows),
        squeeze=False,
    )
    env_label = WORLD_NAME.replace("_", " ").title()
    fig.suptitle(
        f"Goal + Masked Checkpoint Room RCN Maps - {env_label}",
        fontsize=13,
        y=1.01,
    )

    row_offset = 0
    for payload in payloads:
        goal_name = str(payload["goal_name"])
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

                source_name, reward, metadata = panels[panel_idx]
                panel_title = _panel_title(source_name)
                _plot_reward_hex(
                    ax,
                    hmap_xy,
                    reward,
                    title=panel_title,
                    cmap_name=cmap_name,
                    vmin=row_vmin,
                    vmax=row_vmax,
                    goal_pos=goal_pos,
                    checkpoint_positions=checkpoint_pos,
                )
                if row_in_goal == 0 and col == 0:
                    ax.text(
                        0.0,
                        1.10,
                        f"Goal: {goal_name}",
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                    )
                overlay_text = _panel_overlay_text(metadata)
                if overlay_text:
                    ax.text(
                        0.02,
                        0.02,
                        overlay_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        family="monospace",
                        color="black",
                        va="bottom",
                        ha="left",
                        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "0.7"},
                    )

        row_offset += panel_rows

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {out}")
    _finalize_figure(fig, show_plot=show_plot)


if __name__ == "__main__":
    print(f"Plotting masked checkpoint-room maps for {WORLD_NAME}...")
    output_file = Path(OUTPUT_DIR) / "rcn_plots" / "masked_checkpoint_room_maps.png"
    plot_masked_checkpoint_room_maps(
        scales=None,
        cmap_name="plasma",
        save_path=str(output_file),
        show_plot=False,
    )
    print("Done.")
