"""Paper-facing navigation comparison plots from saved Old run artifacts."""

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

try:
    from exploit_metrics import astar_shortest_path, draw_world, load_walls
except Exception:
    astar_shortest_path = None
    draw_world = None
    load_walls = None


EXPLOIT_MODE_MARKERS = ("EXPLOIT", "PLOTTING")


@dataclass
class RunData:
    label: str
    hmap_dir: Path
    world: str | None
    rows: list[dict]
    summary: dict
    goal_xy: tuple[float, float] | None
    trials: list[dict]
    distance_trials: list[tuple[np.ndarray, np.ndarray]]
    path_lengths: list[float]
    path_ratios: list[float]
    success_rate: float


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _xy_from_rows(rows: Iterable[dict]) -> np.ndarray:
    xy = []
    for row in rows:
        xy.append([_as_float(row.get("x"), np.nan), _as_float(row.get("y"), np.nan)])
    arr = np.asarray(xy, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    good = np.isfinite(arr).all(axis=1)
    return arr[good]


def _path_length(xy: np.ndarray, max_step_m: float = 2.0) -> float:
    if len(xy) < 2:
        return 0.0
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    delta = delta[np.isfinite(delta) & (delta < max_step_m)]
    return float(np.sum(delta))


def _straight_distance(start_xy, goal_xy) -> float:
    if not start_xy or not goal_xy:
        return 0.0
    return float(math.hypot(float(start_xy[0]) - float(goal_xy[0]), float(start_xy[1]) - float(goal_xy[1])))


def _group_rows_by_trial(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_as_int(row.get("trial"), 0), []).append(row)
    trials = []
    for _, trial_rows in sorted(grouped.items()):
        trials.append(sorted(trial_rows, key=lambda row: (_as_int(row.get("trial_step"), 0), _as_int(row.get("step"), 0))))
    return trials


def _filter_navigation_rows(rows: list[dict]) -> list[dict]:
    marked = [
        row for row in rows
        if any(marker in str(row.get("mode", "")).upper() for marker in EXPLOIT_MODE_MARKERS)
    ]
    return marked if marked else rows


def _load_rows(hmap_dir: Path) -> list[dict]:
    for name in ("exploit_trajectory.csv", "trajectory.csv"):
        path = hmap_dir / name
        if path.exists():
            return _filter_navigation_rows(_read_csv(path))

    loc_path = hmap_dir / "hmap_loc.pkl"
    if not loc_path.exists():
        return []
    loc = np.asarray(_load_pickle(loc_path), dtype=np.float32)
    if loc.ndim != 2 or loc.shape[1] < 2:
        return []
    return [
        {"step": i, "trial": 0, "trial_step": i, "x": float(x), "y": float(y)}
        for i, (x, y) in enumerate(loc[:, :2])
    ]


def _load_summary(hmap_dir: Path) -> dict:
    path = hmap_dir / "exploit_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    hmaps_path = hmap_dir / "unified_hmaps.pkl"
    if hmaps_path.exists():
        try:
            hmaps = _load_pickle(hmaps_path)
        except Exception:
            return {}
        summary = dict(hmaps.get("exploit_summary", {}) or {})
        if "goal_xy" not in summary and hmaps.get("goal_location") is not None:
            summary["goal_xy"] = hmaps["goal_location"]
        return summary
    return {}


def _goal_from_artifacts(hmap_dir: Path, summary: dict, explicit_goal: tuple[float, float] | None):
    if explicit_goal is not None:
        return explicit_goal
    goal = summary.get("goal_xy")
    if goal and len(goal) >= 2:
        return float(goal[0]), float(goal[1])
    hmaps_path = hmap_dir / "unified_hmaps.pkl"
    if hmaps_path.exists():
        try:
            hmaps = _load_pickle(hmaps_path)
            goal = hmaps.get("goal_location")
            if goal and len(goal) >= 2:
                return float(goal[0]), float(goal[1])
        except Exception:
            pass
    return None


def _infer_world(hmap_dir: Path, summary: dict) -> str | None:
    world = summary.get("world")
    if world:
        return str(world)
    parts = list(hmap_dir.parts)
    if "pkl" in parts:
        idx = parts.index("pkl")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    parent = hmap_dir.parent.name
    return parent if parent in SUPPORTED_WORLDS else None


def _resolve_hmap_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    candidates = [path, path / "hmaps"]
    if path.name != "hmaps":
        candidates.append(path.parent / "hmaps")
    for candidate in candidates:
        if (
            (candidate / "trajectory.csv").exists()
            or (candidate / "exploit_trajectory.csv").exists()
            or (candidate / "hmap_loc.pkl").exists()
            or (candidate / "exploit_summary.json").exists()
        ):
            return candidate
    raise FileNotFoundError(f"Could not find Old hmap artifacts under {path}")


def _discover_run_dirs(world: str | None) -> list[Path]:
    if world:
        return [PKL_ROOT / world / "hmaps"]
    candidates = []
    if PKL_ROOT.exists():
        for world_dir in PKL_ROOT.iterdir():
            hmap_dir = world_dir / "hmaps"
            if world_dir.name in SUPPORTED_WORLDS and hmap_dir.exists():
                newest = max(
                    (p.stat().st_mtime for p in hmap_dir.glob("*") if p.is_file()),
                    default=0.0,
                )
                candidates.append((newest, hmap_dir))
    if not candidates:
        raise FileNotFoundError(f"No Old hmap artifacts found under {PKL_ROOT}")
    return [hmap_dir for _, hmap_dir in sorted(candidates, reverse=True)[:1]]


def _distance_trials(rows: list[dict], goal_xy: tuple[float, float] | None):
    if goal_xy is None:
        return []
    output = []
    for trial_rows in _group_rows_by_trial(rows):
        xy = _xy_from_rows(trial_rows)
        if len(xy) == 0:
            continue
        steps = np.asarray(
            [_as_int(row.get("trial_step"), idx) for idx, row in enumerate(trial_rows[: len(xy)])],
            dtype=np.float32,
        )
        distance = np.linalg.norm(xy - np.asarray(goal_xy, dtype=np.float32), axis=1)
        output.append((steps[: len(distance)], distance))
    return output


def _trial_xy_from_rows(rows: list[dict]) -> list[np.ndarray]:
    return [_xy_from_rows(group) for group in _group_rows_by_trial(rows)]


def _compute_path_ratios(
    run: RunData,
    ratio_baseline: str,
    astar_resolution: float,
    robot_margin: float,
) -> list[float]:
    if ratio_baseline == "summary":
        ratios = [
            _as_float(t.get("path_ratio_to_astar", t.get("path_ratio_to_straight")), np.nan)
            for t in run.trials
        ]
        return [r for r in ratios if np.isfinite(r) and r > 0.0]

    walls = []
    if ratio_baseline == "astar" and load_walls is not None and run.world:
        try:
            walls = load_walls(run.world)
        except Exception:
            walls = []

    ratios = []
    trial_xys = _trial_xy_from_rows(run.rows)
    for idx, trial in enumerate(run.trials or [{} for _ in trial_xys]):
        actual = _as_float(trial.get("path_length_m"), np.nan)
        if not np.isfinite(actual) or actual <= 0.0:
            xy = trial_xys[idx] if idx < len(trial_xys) else np.empty((0, 2), dtype=np.float32)
            actual = _path_length(xy)
        start = trial.get("start_xy")
        goal = trial.get("goal_xy", run.goal_xy)
        if not start and idx < len(trial_xys) and len(trial_xys[idx]):
            start = trial_xys[idx][0].tolist()
        baseline = 0.0
        if ratio_baseline == "astar" and astar_shortest_path is not None and start and goal:
            baseline = astar_shortest_path(start, goal, walls, resolution=astar_resolution, margin=robot_margin) or 0.0
        if baseline <= 1e-6:
            baseline = _straight_distance(start, goal)
        if actual > 0.0 and baseline > 1e-6:
            ratios.append(float(actual / baseline))
    return ratios


def load_run(
    path: Path,
    label: str | None,
    explicit_goal: tuple[float, float] | None,
    ratio_baseline: str,
    astar_resolution: float,
    robot_margin: float,
) -> RunData:
    hmap_dir = _resolve_hmap_dir(path)
    rows = _load_rows(hmap_dir)
    summary = _load_summary(hmap_dir)
    world = _infer_world(hmap_dir, summary)
    goal_xy = _goal_from_artifacts(hmap_dir, summary, explicit_goal)
    trials = list(summary.get("trials", []) or [])
    if not trials:
        for idx, xy in enumerate(_trial_xy_from_rows(rows)):
            if len(xy) == 0:
                continue
            trials.append(
                {
                    "trial": idx,
                    "start_xy": xy[0].tolist(),
                    "goal_xy": list(goal_xy) if goal_xy else None,
                    "path_length_m": _path_length(xy),
                    "success": False,
                }
            )
    path_lengths = [
        _as_float(trial.get("path_length_m"), np.nan)
        for trial in trials
        if np.isfinite(_as_float(trial.get("path_length_m"), np.nan))
    ]
    if not path_lengths:
        path_lengths = [_path_length(xy) for xy in _trial_xy_from_rows(rows) if len(xy) > 1]
    success_rate = _as_float(summary.get("success_rate"), np.nan)
    if not np.isfinite(success_rate):
        success_rate = float(np.mean([bool(t.get("success", False)) for t in trials])) if trials else 0.0
    run = RunData(
        label=label or world or hmap_dir.parent.name,
        hmap_dir=hmap_dir,
        world=world,
        rows=rows,
        summary=summary,
        goal_xy=goal_xy,
        trials=trials,
        distance_trials=_distance_trials(rows, goal_xy),
        path_lengths=path_lengths,
        path_ratios=[],
        success_rate=success_rate,
    )
    run.path_ratios = _compute_path_ratios(run, ratio_baseline, astar_resolution, robot_margin)
    return run


def _mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _sem_or_zero(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def _plot_bar(ax, labels: list[str], means: list[float], errors: list[float], colors, ylabel: str, title: str):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, color=colors[: len(labels)], edgecolor="black", linewidth=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_comparison(runs: list[RunData], output_path: Path, ratio_baseline: str):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    labels = [run.label for run in runs]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(runs))]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    ax_traj, ax_dist, ax_length, ax_success, ax_ratio, ax_table = axes.ravel()

    worlds = {run.world for run in runs if run.world}
    if len(worlds) == 1 and draw_world is not None and load_walls is not None:
        world = next(iter(worlds))
        try:
            draw_world(ax_traj, load_walls(world))
        except Exception:
            ax_traj.set_aspect("equal", adjustable="box")
    else:
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")

    for idx, run in enumerate(runs):
        trial_xys = _trial_xy_from_rows(run.rows)
        for trial_idx, xy in enumerate(trial_xys):
            if len(xy) < 2:
                continue
            ax_traj.plot(
                xy[:, 0],
                xy[:, 1],
                color=colors[idx],
                linewidth=1.4,
                alpha=0.78 if trial_idx == 0 else 0.32,
                label=run.label if trial_idx == 0 else None,
            )
            if trial_idx == 0:
                ax_traj.scatter(xy[0, 0], xy[0, 1], color=colors[idx], edgecolors="black", s=32, zorder=5)
        if run.goal_xy:
            ax_traj.scatter(
                [run.goal_xy[0]],
                [run.goal_xy[1]],
                marker="*",
                s=190,
                color="#ffd23f",
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
    ax_traj.set_title("Trajectory overlay")
    ax_traj.legend(loc="best")

    for idx, run in enumerate(runs):
        for trial_idx, (steps, distance) in enumerate(run.distance_trials):
            ax_dist.plot(
                steps,
                distance,
                color=colors[idx],
                alpha=0.84 if trial_idx == 0 else 0.28,
                linewidth=1.5 if trial_idx == 0 else 0.9,
                label=run.label if trial_idx == 0 else None,
            )
    ax_dist.set_xlabel("trial step")
    ax_dist.set_ylabel("distance to goal (m)")
    ax_dist.set_title("Goal-distance curves")
    ax_dist.grid(alpha=0.25)
    ax_dist.legend(loc="best")

    length_means = [_mean_or_nan(run.path_lengths) for run in runs]
    length_errors = [_sem_or_zero(run.path_lengths) for run in runs]
    _plot_bar(ax_length, labels, length_means, length_errors, colors, "meters", "Path length")

    success = [run.success_rate for run in runs]
    _plot_bar(ax_success, labels, success, [0.0] * len(runs), colors, "success rate", "Navigation success")
    ax_success.set_ylim(0.0, 1.05)

    ratio_means = [_mean_or_nan(run.path_ratios) for run in runs]
    ratio_errors = [_sem_or_zero(run.path_ratios) for run in runs]
    _plot_bar(ax_ratio, labels, ratio_means, ratio_errors, colors, "actual / baseline", f"Path ratio ({ratio_baseline})")
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_table.axis("off")
    table_rows = []
    for run in runs:
        table_rows.append(
            [
                run.label,
                str(len(run.trials)),
                f"{run.success_rate:.2f}",
                f"{_mean_or_nan(run.path_lengths):.2f}",
                f"{_mean_or_nan(run.path_ratios):.2f}",
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["run", "trials", "success", "path m", "ratio"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax_table.set_title("Summary")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(runs: list[RunData], output_path: Path, ratio_baseline: str):
    payload = {
        "ratio_baseline": ratio_baseline,
        "runs": [
            {
                "label": run.label,
                "world": run.world,
                "hmap_dir": str(run.hmap_dir),
                "trial_count": len(run.trials),
                "success_rate": run.success_rate,
                "mean_path_length_m": _mean_or_nan(run.path_lengths),
                "sem_path_length_m": _sem_or_zero(run.path_lengths),
                "mean_path_ratio": _mean_or_nan(run.path_ratios),
                "sem_path_ratio": _sem_or_zero(run.path_ratios),
            }
            for run in runs
        ],
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_goal(values: list[str] | None):
    if not values:
        return None
    if len(values) != 2:
        raise ValueError("--goal expects exactly two values: X Y")
    return float(values[0]), float(values[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run directories, hmaps directories, or files inside a run.")
    parser.add_argument("--world", default=None, help="Use pkl/<world>/hmaps when no runs are provided.")
    parser.add_argument("--labels", nargs="*", default=None, help="Labels matching the run order.")
    parser.add_argument("--goal", nargs="*", default=None, help="Override goal as X Y.")
    parser.add_argument(
        "--ratio-baseline",
        choices=("astar", "straight", "summary"),
        default="astar",
        help="Baseline for path ratio. astar falls back to straight-line distance when needed.",
    )
    parser.add_argument("--astar-resolution", type=float, default=0.25)
    parser.add_argument("--robot-margin", type=float, default=0.25)
    parser.add_argument("--output", default=None, help="Output image path. Defaults under vis_outputs/navigation_comparison.")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs] if args.runs else _discover_run_dirs(args.world)
    labels = args.labels or []
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--labels must have the same count as the resolved runs")
    explicit_goal = parse_goal(args.goal)
    runs = [
        load_run(
            path,
            labels[idx] if idx < len(labels) else None,
            explicit_goal,
            args.ratio_baseline,
            args.astar_resolution,
            args.robot_margin,
        )
        for idx, path in enumerate(run_dirs)
    ]

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        if output_path.suffix == "":
            output_path = output_path / "navigation_comparison.png"
    elif len(runs) == 1 and runs[0].world:
        output_path = PKL_ROOT / runs[0].world / "vis_outputs" / "navigation_comparison" / "navigation_comparison.png"
    else:
        output_path = PROJECT_ROOT / "visualizations" / "navigation_comparison.png"

    plot_comparison(runs, output_path, args.ratio_baseline)
    write_summary(runs, output_path, args.ratio_baseline)
    print(f"Saved {output_path}")
    print(f"Saved {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
