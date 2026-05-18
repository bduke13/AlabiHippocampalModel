import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.robot.webots_worlds import expanded_obstacle_corners, get_world_config


WORLDS = [
    ("1 Obstacle", "20x20_1obstacle"),
    ("2 Obstacles", "20x20_2obstacles"),
    ("Cross", "20x20_cross_multi_goal"),
]

OUTPUT_PATH = Path(__file__).resolve().parent / "parser_geometry_debug.png"

GOAL_COLORS = {
    "red": "#e00000",
    "green": "#00cc18",
    "blue": "#0017d9",
    "yellow": "#d9d400",
}


def _draw_world(ax, label: str, world_name: str) -> None:
    config = get_world_config(world_name)
    width, height = config["size"]
    half_w, half_h = width / 2.0, height / 2.0

    ax.set_title(f"{label}\n{world_name}")
    ax.set_xlim(-half_w - 0.75, half_w + 0.75)
    ax.set_ylim(-half_h - 0.75, half_h + 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)

    arena = patches.Rectangle(
        (-half_w, -half_h),
        width,
        height,
        facecolor="#f2f2f2",
        edgecolor="#222222",
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(arena)

    true_label_used = False
    bounds_label_used = False
    for obstacle in config.get("obstacles", []):
        if obstacle.get("type") != "rectangle":
            continue

        corners = np.asarray(expanded_obstacle_corners(obstacle), dtype=np.float64)
        polygon = patches.Polygon(
            corners,
            closed=True,
            facecolor="#8e9ab1",
            edgecolor="#4f5a70",
            linewidth=1.2,
            alpha=0.85,
            label="true parsed footprint" if not true_label_used else None,
            zorder=3,
        )
        true_label_used = True
        ax.add_patch(polygon)

        (min_x, min_z), (max_x, max_z) = obstacle["bounds"]
        bounds = patches.Rectangle(
            (min_x, min_z),
            max_x - min_x,
            max_z - min_z,
            facecolor="none",
            edgecolor="#cc3333",
            linewidth=1.0,
            linestyle="--",
            label="retained AABB bounds" if not bounds_label_used else None,
            zorder=2,
        )
        bounds_label_used = True
        ax.add_patch(bounds)

        center = obstacle.get("center")
        if center is not None:
            ax.text(
                center[0],
                center[1],
                obstacle.get("name", ""),
                fontsize=8,
                ha="center",
                va="center",
                color="#111111",
                zorder=4,
            )

    for goal in config.get("goals", []):
        color = GOAL_COLORS.get(goal.get("name"), "#555555")
        goal_patch = patches.Circle(
            goal["location"],
            goal.get("radius", 1.0),
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.6,
            alpha=0.95,
            zorder=5,
        )
        ax.add_patch(goal_patch)

    if true_label_used or bounds_label_used:
        ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(18, 6), constrained_layout=True)
    for ax, (label, world_name) in zip(axes, WORLDS):
        _draw_world(ax, label, world_name)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved parser geometry debug plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
