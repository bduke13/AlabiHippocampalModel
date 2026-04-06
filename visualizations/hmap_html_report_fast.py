# hmap_html_report_fast.py
# Unified-compatible fast HTML report for place-cell hmaps.

import os
import sys
import pickle
import webbrowser
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from visualizations.place_cell_learning import load_place_cell_learning_scores

from visualizations.vis_utils import (
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR,
)

CELL_SORT_MODE = "learning_strength"


def _hmap_dir() -> str:
    return os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")


def _network_dir() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"


def discover_available_scales() -> List[int]:
    hdir = Path(_hmap_dir())
    if not hdir.exists():
        return []
    scales = []
    for p in sorted(hdir.glob("hmap_pcn_scale_*.pkl")):
        try:
            scales.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return scales


def load_scale_hmaps(scale: int):
    hdir = _hmap_dir()
    try:
        with open(os.path.join(hdir, "hmap_loc.pkl"), "rb") as f:
            hmap_loc = np.array(pickle.load(f))[1:]
    except FileNotFoundError:
        print(f"[Scale {scale}] Missing hmap_loc.pkl in {hdir}")
        return None, None

    pcn_path = os.path.join(hdir, f"hmap_pcn_scale_{scale}.pkl")
    try:
        with open(pcn_path, "rb") as f:
            hmap_pcn = np.array(pickle.load(f))[1:]
    except FileNotFoundError:
        print(f"[Scale {scale}] Missing {pcn_path}")
        return None, None

    return hmap_loc, hmap_pcn


def load_unified_hmaps(scales: List[int]):
    if not scales:
        return None, None

    per_scale = []
    hmap_loc_ref = None
    min_steps = None

    for scale in scales:
        hmap_loc, hmap_pcn = load_scale_hmaps(scale)
        if hmap_loc is None or hmap_pcn is None:
            continue
        if hmap_loc_ref is None:
            hmap_loc_ref = hmap_loc
        per_scale.append(hmap_pcn)
        n = min(hmap_loc.shape[0], hmap_pcn.shape[0])
        min_steps = n if min_steps is None else min(min_steps, n)

    if hmap_loc_ref is None or not per_scale or min_steps is None:
        return None, None

    hmap_loc_ref = hmap_loc_ref[:min_steps]
    per_scale = [arr[:min_steps] for arr in per_scale]
    hmap_pcn_unified = np.concatenate(per_scale, axis=1)
    return hmap_loc_ref, hmap_pcn_unified


def _load_learning_scores_for_label(label_token: str):
    network_dir = _network_dir()
    unified_exists = (network_dir / "unified_pcn.pkl").exists()

    if label_token == "unified":
        return load_place_cell_learning_scores(network_dir, scale=None, unified=True)

    if label_token.startswith("scale_"):
        scale = int(label_token.split("_")[-1])
        return load_place_cell_learning_scores(network_dir, scale=scale, unified=unified_exists)

    return None, {"error": f"unknown_label_token:{label_token}"}


def _generate_vibrant_colors(n: int):
    if n <= 0:
        return np.zeros((0, 3))
    hues = np.linspace(0, 1, num=n, endpoint=False)
    colors = np.array([plt.cm.hsv(h)[:3] for h in hues], dtype=float)
    colors = np.clip(colors * 0.9 * 0.95, 0.0, 1.0)
    return colors


def _plot_group_worker(task):
    (
        output_dir_assets,
        label_token,
        group_index,
        group_cell_indices,
        hmap_x,
        hmap_y,
        hmap_pcn,
        colors_rgb,
        gridsize,
        dpi,
    ) = task

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Place Cells Group {group_index} - {label_token}", fontsize=14)

    for ax, cell_index in zip(axes, group_cell_indices):
        activations = hmap_pcn[:, int(cell_index)]
        base_color = colors_rgb[int(cell_index) % len(colors_rgb)]

        hb = ax.hexbin(
            hmap_x,
            hmap_y,
            C=activations,
            gridsize=gridsize,
            reduce_C_function=np.mean,
            cmap=None,
            edgecolors="none",
        )

        counts = hb.get_array()
        if counts.size:
            max_count = counts.max()
            norm = counts / max_count if max_count > 0 else counts
            rgba = np.empty((counts.shape[0], 4), dtype=float)
            rgba[:, :3] = base_color
            rgba[:, 3] = norm
            hb.set_facecolors(rgba)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Cell {int(cell_index)}", fontsize=11)

    os.makedirs(output_dir_assets, exist_ok=True)
    out_path = os.path.join(output_dir_assets, f"place_cells_{label_token}_group_{group_index}.jpg")
    plt.savefig(out_path, dpi=dpi, pil_kwargs={"quality": 90})
    plt.close(fig)
    return out_path


def _generate_html_report(
    cell_indices,
    label: str,
    label_token: str,
    output_dir: str,
    report_name: str,
    nav_links: Optional[List[Tuple[str, str]]] = None,
):
    html_path = os.path.join(output_dir, report_name)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            f"""<html>
<head>
<title>Place Cell Visualizations - {label}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .group-viz {{ margin-bottom: 30px; }}
  h1 {{ color: #333; }}
  .scale-links {{ margin-bottom: 20px; }}
  .scale-links a {{ margin-right: 15px; }}
</style>
</head>
<body>
  <h1>Place Cell Visualizations - {label}</h1>
  <p><strong>Report Location:</strong> <code id="report-path"></code></p>
  <div class="scale-links">
    <strong>Available Reports:</strong>
"""
        )

        if nav_links:
            for link_label, href in nav_links:
                f.write(f'    <a href="{href}">{link_label}</a>\n')

        f.write("""
  </div>
  <script>document.getElementById('report-path').textContent = window.location.pathname;</script>
""")

        num_groups = (len(cell_indices) + 4) // 5
        for i in range(num_groups):
            f.write(
                f"""
  <div class="group-viz">
    <h3>Group {i}</h3>
    <img src="html_assets/place_cells_{label_token}_group_{i}.jpg" style="max-width: 100%;">
  </div>
"""
            )

        f.write("</body></html>")

    return html_path


def generate_place_cells_report(
    label,
    label_token,
    hmap_x,
    hmap_y,
    hmap_pcn,
    output_dir=None,
    open_browser=True,
    activation_threshold=None,
    gridsize=50,
    dpi=300,
    n_workers=None,
    report_name: Optional[str] = None,
    nav_links: Optional[List[Tuple[str, str]]] = None,
):
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "place_cells")

    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)

    totals = np.sum(np.abs(hmap_pcn), axis=0)
    cell_indices = np.arange(hmap_pcn.shape[1], dtype=int)

    if activation_threshold is not None:
        active_mask = totals > activation_threshold
        cell_indices = cell_indices[active_mask]
        totals = totals[active_mask]
        print(
            f"[{label}] Filtered {np.sum(~active_mask)} cells below display threshold {activation_threshold}. "
            f"Remaining: {len(cell_indices)}"
        )

    cell_scores, score_meta = _load_learning_scores_for_label(label_token)
    ranking_note = "activity fallback"
    if (
        CELL_SORT_MODE == "learning_strength"
        and cell_scores is not None
        and len(cell_scores) == hmap_pcn.shape[1]
    ):
        score_slice = np.asarray(cell_scores, dtype=np.float32)[cell_indices]
        valid_mask = np.isfinite(score_slice) & (score_slice > 0)
        if np.any(valid_mask):
            cell_indices = cell_indices[valid_mask]
            score_slice = score_slice[valid_mask]
            sort_order = np.argsort(score_slice)[::-1]
            cell_indices = cell_indices[sort_order]
            ranking_note = "learning strength"
            if score_meta.get("used_fallback"):
                ranking_note += " [current-weight fallback]"
        else:
            sort_order = np.argsort(totals)[::-1]
            cell_indices = cell_indices[sort_order]
    else:
        sort_order = np.argsort(totals)[::-1]
        cell_indices = cell_indices[sort_order]

    print(f"[{label}] Showing {len(cell_indices)} cells sorted by {ranking_note}")

    colors_rgb = _generate_vibrant_colors(hmap_pcn.shape[1])

    tasks = []
    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i : i + 5]
        if len(group) < 5:
            group = np.append(group, np.repeat(group[-1], 5 - len(group))).astype(int)
        group_index = i // 5
        tasks.append(
            (
                assets_dir,
                label_token,
                group_index,
                group,
                hmap_x,
                hmap_y,
                hmap_pcn,
                colors_rgb,
                gridsize,
                dpi,
            )
        )

    if n_workers is None or n_workers <= 0:
        n_workers = max(1, min(cpu_count(), len(tasks)))

    if n_workers > 1 and len(tasks) > 1:
        print(f"[{label}] Rendering {len(tasks)} groups with {n_workers} workers...")
        with Pool(processes=n_workers) as pool:
            for _ in pool.imap_unordered(_plot_group_worker, tasks, chunksize=1):
                pass
    else:
        print(f"[{label}] Rendering {len(tasks)} groups sequentially...")
        for t in tasks:
            _plot_group_worker(t)

    if report_name is None:
        report_name = f"place_cells_report_{label_token}.html"
    html_path = _generate_html_report(cell_indices, label, label_token, output_dir, report_name, nav_links=nav_links)

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"[{label}] HTML: {html_path}")
    print(f"[{label}] Images: {assets_dir}")
    return html_path


def generate_multi_scale_reports(
    scales=None,
    output_dir=None,
    open_browser=True,
    activation_threshold=None,
    gridsize=50,
    dpi=300,
    n_workers=None,
):
    if scales is None:
        scales = discover_available_scales() or [0, 1, 2]
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "place_cells")

    report_specs: List[Tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for scale in scales:
        print(f"\n==== Processing Scale {scale} ====")
        hmap_loc, hmap_pcn = load_scale_hmaps(scale)
        if hmap_loc is None or hmap_pcn is None:
            print(f"[Scale {scale}] Skipped (missing data)")
            continue
        hmap_x, hmap_y = hmap_loc[:, 0], hmap_loc[:, 1]
        report_specs.append((f"Scale {scale}", f"scale_{scale}", f"place_cells_report_scale_{scale}.html", hmap_x, hmap_y, hmap_pcn))

    if len(report_specs) >= 2:
        hmap_loc_u, hmap_pcn_u = load_unified_hmaps(scales)
        if hmap_loc_u is not None and hmap_pcn_u is not None:
            hmap_x_u, hmap_y_u = hmap_loc_u[:, 0], hmap_loc_u[:, 1]
            report_specs.insert(0, ("Unified (All Scales)", "unified", "place_cells_report_unified.html", hmap_x_u, hmap_y_u, hmap_pcn_u))

    nav_links = [(label, report_file) for (label, _token, report_file, *_rest) in report_specs]

    html_reports = []
    for label, label_token, report_file, hmap_x, hmap_y, hmap_pcn in report_specs:
        print(f"\n==== Rendering Report: {label} ====")
        html_path = generate_place_cells_report(
            label=label,
            label_token=label_token,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            output_dir=output_dir,
            open_browser=open_browser,
            activation_threshold=activation_threshold,
            gridsize=gridsize,
            dpi=dpi,
            n_workers=n_workers,
            report_name=report_file,
            nav_links=nav_links,
        )
        html_reports.append(html_path)

    return html_reports


if __name__ == "__main__":
    print("Starting unified-compatible place cell HTML report generation (FAST)...")

    reports = generate_multi_scale_reports(
        scales=None,
        activation_threshold=None,
        open_browser=True,
        gridsize=50,
        dpi=300,
        n_workers=None,
    )

    print("\nAll reports generated!")
    for i, p in enumerate(reports, 1):
        print(f"Report {i}: {p}")
