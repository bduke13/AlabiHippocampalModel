# multi_html_report_fast.py
# Drop-in faster version of multi_html_report.py:
# - Parallel rendering of group figures
# - Non-interactive Agg backend
# - Same outputs, paths, and HTML layout

import os
import sys
import pickle
import webbrowser
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Force non-interactive backend for speed in workers
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # just in case
import matplotlib.pyplot as plt

# Project root & imports (same as original)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR,
)

# ---------------------------
# I/O helpers (unchanged I/O)
# ---------------------------
def _hmap_dir():
    return os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")

def load_scale_hmaps(scale: int):
    """
    Load hmap_loc and hmap_pcn for a scale.
    Returns (hmap_loc, hmap_pcn) or (None, None) if missing.
    """
    hdir = _hmap_dir()
    try:
        with open(os.path.join(hdir, "hmap_loc.pkl"), "rb") as f:
            hmap_loc = np.array(pickle.load(f))[1:]  # drop first element
    except FileNotFoundError:
        print(f"[Scale {scale}] Missing hmap_loc.pkl in {hdir}")
        return None, None

    pcn_path = os.path.join(hdir, f"hmap_pcn_scale_{scale}.pkl")
    try:
        with open(pcn_path, "rb") as f:
            hmap_pcn = np.array(pickle.load(f))[1:]  # drop first element
    except FileNotFoundError:
        print(f"[Scale {scale}] Missing {pcn_path}")
        return None, None

    return hmap_loc, hmap_pcn


def _generate_vibrant_colors(n: int):
    # Precompute vibrant HSV → RGB (no alpha) once
    # Using matplotlib cm.hsv yields bright distinct colors.
    if n <= 0:
        return np.zeros((0, 3))
    hues = np.linspace(0, 1, num=n, endpoint=False)
    colors = np.array([plt.cm.hsv(h)[:3] for h in hues], dtype=float)
    # Slightly boost vibrancy
    colors = np.clip(colors * 0.9 * 0.95, 0.0, 1.0)
    return colors


# ---------------------------
# Parallel worker
# ---------------------------
def _plot_group_worker(task):
    """
    Worker that renders ONE group (up to 5 cells) image for a given scale.

    Args (packed in task tuple):
        output_dir_assets: str - directory to save images (html_assets)
        scale: int
        group_index: int
        group_cell_indices: np.ndarray[int] size 5
        hmap_x: np.ndarray shape (T,)
        hmap_y: np.ndarray shape (T,)
        hmap_pcn: np.ndarray shape (T, Ncells)  (possibly threshold-filtered)
        colors_rgb: np.ndarray shape (Ncells, 3)
        gridsize: int
        dpi: int
    """
    (
        output_dir_assets,
        scale,
        group_index,
        group_cell_indices,
        hmap_x,
        hmap_y,
        hmap_pcn,
        colors_rgb,
        gridsize,
        dpi,
    ) = task

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # same layout
    fig.suptitle(f"Place Cells Group {group_index} - Scale {scale}", fontsize=14)

    # Small optimization: fix axis labels only once per axes; no spines tweaks for speed
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

    # Avoid bbox_inches='tight' (slow). Keep high DPI for similar quality.
    os.makedirs(output_dir_assets, exist_ok=True)
    out_path = os.path.join(
        output_dir_assets, f"place_cells_scale_{scale}_group_{group_index}.jpg"
    )
    # Use PIL kwargs for a faster/lighter JPEG without visible loss
    plt.savefig(out_path, dpi=dpi, pil_kwargs={"quality": 90})
    plt.close(fig)
    return out_path


# ---------------------------
# Report generation (HTML)
# ---------------------------
def _generate_html_report(cell_indices, scale, output_dir):
    html_path = os.path.join(output_dir, f"place_cells_report_scale_{scale}.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            f"""<html>
<head>
<title>Place Cell Visualizations - Scale {scale}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .group-viz {{ margin-bottom: 30px; }}
  h1 {{ color: #333; }}
  .scale-links {{ margin-bottom: 20px; }}
  .scale-links a {{ margin-right: 15px; }}
</style>
</head>
<body>
  <h1>Place Cell Visualizations - Scale {scale}</h1>
  <p><strong>Report Location:</strong> <code id="report-path"></code></p>
  <div class="scale-links">
    <strong>Other Scales:</strong>
    <a href="place_cells_report_scale_0.html">Scale 0</a>
    <a href="place_cells_report_scale_1.html">Scale 1</a>
    <a href="place_cells_report_scale_2.html">Scale 2</a>
  </div>
  <script>document.getElementById('report-path').textContent = window.location.pathname;</script>
"""
        )

        num_groups = (len(cell_indices) + 4) // 5
        for i in range(num_groups):
            f.write(
                f"""
  <div class="group-viz">
    <h3>Group {i}</h3>
    <img src="html_assets/place_cells_scale_{scale}_group_{i}.jpg" style="max-width: 100%;">
  </div>
"""
            )

        f.write("</body></html>")

    return html_path


# ---------------------------
# Public API (same behavior)
# ---------------------------
def generate_place_cells_report(
    scale,
    hmap_x,
    hmap_y,
    hmap_pcn,
    output_dir=None,
    open_browser=True,
    activation_threshold=None,
    gridsize=50,
    dpi=300,
    n_workers=None,
):
    """
    Generate an HTML report for a given scale (FAST).

    Params are compatible with original; added:
      - gridsize (default 50, same visual density)
      - dpi (default 300 to match quality)
      - n_workers: int or None (defaults to cpu_count())

    Returns: path to the HTML report.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "place_cells")

    assets_dir = os.path.join(output_dir, "html_assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Filter inactive cells if requested (same semantics)
    if activation_threshold is not None:
        totals = np.sum(np.abs(hmap_pcn), axis=0)
        active_mask = totals > activation_threshold
        hmap_pcn = hmap_pcn[:, active_mask]
        print(
            f"[Scale {scale}] Filtered {np.sum(~active_mask)} cells below threshold {activation_threshold}. "
            f"Remaining: {hmap_pcn.shape[1]}"
        )

    # cell indices (optional CSV per scale)
    cells_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cells_scale_{scale}.csv")
    if os.path.exists(cells_csv_path):
        cell_indices = np.loadtxt(cells_csv_path, dtype=int)
        # Guard against indices > filtered count
        cell_indices = cell_indices[cell_indices < hmap_pcn.shape[1]]
        print(f"[Scale {scale}] Loaded {len(cell_indices)} cell indices from {cells_csv_path}")
    else:
        cell_indices = np.arange(hmap_pcn.shape[1], dtype=int)
        print(f"[Scale {scale}] No cells CSV found; using all {len(cell_indices)} cells")

    # Precompute colors once
    colors_rgb = _generate_vibrant_colors(hmap_pcn.shape[1])

    # Build group tasks (exactly 5 cells per group; pad with last)
    tasks = []
    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i : i + 5]
        if len(group) < 5:
            group = np.append(group, np.repeat(group[-1], 5 - len(group))).astype(int)
        group_index = i // 5
        tasks.append(
            (
                assets_dir,
                scale,
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

    # Parallel rendering
    if n_workers is None or n_workers <= 0:
        n_workers = max(1, min(cpu_count(), len(tasks)))

    if n_workers > 1 and len(tasks) > 1:
        print(f"[Scale {scale}] Rendering {len(tasks)} groups with {n_workers} workers...")
        with Pool(processes=n_workers) as pool:
            for _ in pool.imap_unordered(_plot_group_worker, tasks, chunksize=1):
                pass
    else:
        print(f"[Scale {scale}] Rendering {len(tasks)} groups sequentially...")
        for t in tasks:
            _plot_group_worker(t)

    # HTML
    html_path = _generate_html_report(cell_indices, scale, output_dir)

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"[Scale {scale}] HTML: {html_path}")
    print(f"[Scale {scale}] Images: {assets_dir}")
    return html_path


def generate_multi_scale_reports(
    scales=None,
    output_dir=None,
    open_browser=True,
    activation_threshold=0.1,
    gridsize=50,
    dpi=300,
    n_workers=None,
):
    """
    Multi-scale wrapper (FAST).
    Mirrors original signature with added performance knobs.
    """
    if scales is None:
        scales = [0, 1, 2]
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "place_cells")

    html_reports = []

    for scale in scales:
        print(f"\n==== Processing Scale {scale} ====")
        hmap_loc, hmap_pcn = load_scale_hmaps(scale)
        if hmap_loc is None or hmap_pcn is None:
            print(f"[Scale {scale}] Skipped (missing data)")
            continue

        # Same coordinate conversion as original
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

        html_path = generate_place_cells_report(
            scale=scale,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            output_dir=output_dir,
            open_browser=open_browser,
            activation_threshold=activation_threshold,
            gridsize=gridsize,
            dpi=dpi,
            n_workers=n_workers,
        )
        html_reports.append(html_path)

    return html_reports


if __name__ == "__main__":
    print("Starting multi-scale place cell HTML report generation (FAST)...")

    reports = generate_multi_scale_reports(
        scales=[0, 1, 2],
        activation_threshold=0.2,  # adjust as needed
        open_browser=True,
        gridsize=50,               # same visual density as original
        dpi=300,                   # same quality; drop to ~200 for extra speed
        n_workers=None,            # default: use min(cpu_count, num_groups)
    )

    print("\nAll reports generated!")
    for i, p in enumerate(reports, 1):
        print(f"Report {i}: {p}")
