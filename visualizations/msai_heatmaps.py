import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR,
)

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

SCALE_NAMES = {0: 'Small', 1: 'Medium', 2: 'Large'}


def _hmaps_root() -> str:
    return os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")


def discover_available_scales() -> List[int]:
    root = Path(_hmaps_root())
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("hmap_pcn_scale_*.pkl")):
        try:
            out.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return out


def load_hmap_loc() -> Optional[np.ndarray]:
    fp = os.path.join(_hmaps_root(), "hmap_loc.pkl")
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None


def load_hmap_pcn(scale: int) -> Optional[np.ndarray]:
    fp = os.path.join(_hmaps_root(), f"hmap_pcn_scale_{scale}.pkl")
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None


def load_hmap_pcn_unified(scales: List[int]) -> Optional[np.ndarray]:
    if not scales:
        return None
    per_scale = []
    min_steps = None
    for s in scales:
        arr = load_hmap_pcn(s)
        if arr is None:
            continue
        per_scale.append(arr)
        min_steps = arr.shape[0] if min_steps is None else min(min_steps, arr.shape[0])

    if not per_scale or min_steps is None:
        return None

    per_scale = [a[:min_steps] for a in per_scale]
    return np.concatenate(per_scale, axis=1)


def _bin_stats_for_all_cells(
    x: np.ndarray, y: np.ndarray, pcn: np.ndarray, gridsize: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert x.shape[0] == y.shape[0] == pcn.shape[0], "Time dimension mismatch"

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    T, C = pcn.shape
    gx = gridsize
    gy = gridsize

    total = np.zeros((gx, gy, C), dtype=np.float64)
    count = np.zeros((gx, gy, C), dtype=np.float64)

    ix = np.digitize(x, xedges) - 1
    iy = np.digitize(y, yedges) - 1
    ix = np.clip(ix, 0, gx - 1)
    iy = np.clip(iy, 0, gy - 1)

    pos_mask = pcn > 0.0
    for c in range(C):
        mask = pos_mask[:, c]
        if not np.any(mask):
            continue
        ix_c = ix[mask]
        iy_c = iy[mask]
        a_c = pcn[mask, c]
        np.add.at(total[..., c], (ix_c, iy_c), a_c)
        np.add.at(count[..., c], (ix_c, iy_c), 1.0)

    mean_act = np.zeros_like(total)
    nz = count > 0
    mean_act[nz] = total[nz] / count[nz]

    valid_mask = (count.sum(axis=2) > 0)
    return mean_act, xcenters, ycenters, valid_mask


def _per_cell_threshold_and_normalize(mean_act: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    gx, gy, C = mean_act.shape
    proc = np.zeros_like(mean_act)
    for c in range(C):
        mc = mean_act[..., c]
        vals = mc[mc > 0]
        if vals.size == 0:
            continue
        thr = np.quantile(vals, quantile)
        mc2 = np.where(mc >= thr, mc, 0.0)
        mmax = mc2.max()
        if mmax > 0:
            mc2 = mc2 / mmax
        proc[..., c] = mc2
    return proc


def _cosine_similarity(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    An = A / norms
    return An @ An.T


def compute_sai_map_and_msai(
    x: np.ndarray,
    y: np.ndarray,
    pcn: np.ndarray,
    gridsize: int = 100,
    dist_threshold: Optional[float] = None,
    quantile: float = 0.10,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean_act, xcenters, ycenters, valid_mask = _bin_stats_for_all_cells(x, y, pcn, gridsize=gridsize)
    proc = _per_cell_threshold_and_normalize(mean_act, quantile=quantile)

    gx, gy, C = proc.shape
    XX, YY = np.meshgrid(xcenters, ycenters, indexing='ij')
    bin_xy = np.stack([XX, YY], axis=-1)

    idxs = np.argwhere(valid_mask)
    if idxs.shape[0] == 0:
        return np.zeros((gx, gy)), 0.0, xcenters, ycenters

    A = proc[valid_mask, :]
    centers = bin_xy[valid_mask, :]

    if dist_threshold is None:
        span = max(xcenters.max() - xcenters.min(), ycenters.max() - ycenters.min())
        dist_threshold = 0.3 * span

    S = _cosine_similarity(A)

    ci2 = np.sum(centers**2, axis=1, keepdims=True)
    d2 = ci2 + ci2.T - 2 * (centers @ centers.T)
    D = np.sqrt(np.clip(d2, 0.0, None))

    far_mask = (D > dist_threshold)
    np.fill_diagonal(far_mask, False)

    denom = np.maximum(far_mask.sum(axis=1), 1)
    sai_vals = (S * far_mask).sum(axis=1) / denom

    sai_grid = np.zeros((gx, gy), dtype=np.float64)
    for i, (ix, iy) in enumerate(idxs):
        sai_grid[ix, iy] = sai_vals[i]

    msai = sai_vals.mean() if sai_vals.size > 0 else 0.0
    return sai_grid, float(msai), xcenters, ycenters


def _plot_sai_panel(ax, xcenters, ycenters, sai_grid, cmap='jet', add_colorbar=False):
    XI, YI = np.meshgrid(
        np.linspace(xcenters.min(), xcenters.max(), 150),
        np.linspace(ycenters.min(), ycenters.max(), 150),
        indexing='xy',
    )

    XX, YY = np.meshgrid(xcenters, ycenters, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = sai_grid.T.ravel()
    ZI = griddata(pts, vals, (XI, YI), method='cubic', fill_value=0.0)

    cs = ax.contourf(XI, YI, ZI, levels=np.linspace(ZI.min(), ZI.max(), 25), cmap=cmap)
    ax.contour(XI, YI, ZI, levels=10, colors='black', alpha=0.25, linewidths=0.5)

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax)
        cbar.set_label('SAI', fontsize=14)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=16)
    ax.set_ylabel('Y Position (m)', fontsize=16)
    ax.tick_params(labelsize=14)


def generate_msai_heatmaps(
    scales: Optional[List[int]] = None,
    include_unified: bool = True,
    gridsize: int = 240,
    quantile: float = 0.10,
    dist_threshold: Optional[float] = None,
    cmap: str = 'jet',
    output_dir: Optional[str] = None,
    fname: str = 'msai_all_scales.png',
) -> Tuple[str, Dict[str, float]]:
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "paper_figures", "msai")
    os.makedirs(output_dir, exist_ok=True)

    if scales is None:
        scales = discover_available_scales() or [0, 1, 2]

    hmap_loc = load_hmap_loc()
    if hmap_loc is None:
        raise FileNotFoundError("hmap_loc.pkl not found.")

    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)

    panels: List[Tuple[str, np.ndarray]] = []
    for scale in scales:
        pcn = load_hmap_pcn(scale)
        if pcn is not None:
            n = min(hmap_x.shape[0], pcn.shape[0])
            panels.append((f"Scale {scale}", pcn[:n]))

    if include_unified and len(scales) >= 2:
        pcn_u = load_hmap_pcn_unified(scales)
        if pcn_u is not None:
            n = min(hmap_x.shape[0], pcn_u.shape[0])
            panels.insert(0, ("Unified", pcn_u[:n]))

    if not panels:
        raise FileNotFoundError("No hmap_pcn_scale_*.pkl files were found.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    msai_per_panel: Dict[str, float] = {}

    for col, (label, pcn) in enumerate(panels):
        ax = axes[0, col]
        n_steps = min(hmap_x.shape[0], pcn.shape[0])
        sai_grid, msai, xcenters, ycenters = compute_sai_map_and_msai(
            hmap_x[:n_steps], hmap_y[:n_steps], pcn[:n_steps], gridsize=gridsize,
            dist_threshold=dist_threshold, quantile=quantile
        )
        msai_per_panel[label] = msai

        _plot_sai_panel(ax, xcenters, ycenters, sai_grid, cmap=cmap, add_colorbar=(col == n - 1))
        if label.startswith("Scale"):
            try:
                s_idx = int(label.split()[-1])
                title = SCALE_NAMES.get(s_idx, label)
            except ValueError:
                title = label
        else:
            title = label
        ax.set_title(title, fontsize=18)

    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return out_path, msai_per_panel


if __name__ == "__main__":
    path, msai = generate_msai_heatmaps(
        scales=None,
        include_unified=True,
        gridsize=120,
        quantile=0.10,
        dist_threshold=None,
        cmap='jet',
        output_dir=None,
        fname='msai_all_scales.png',
    )
    print("MSAI:", msai)
