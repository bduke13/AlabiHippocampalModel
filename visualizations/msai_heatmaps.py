
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

# project imports
from vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR
)

# Publication-like defaults (match paper_visuals_selected.py)
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

# ------------------------------
# IO helpers (mirrors paper_visuals_selected.py behavior)
# ------------------------------
def _hmaps_root() -> str:
    return os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")

def load_hmap_loc() -> Optional[np.ndarray]:
    fp = os.path.join(_hmaps_root(), "hmap_loc.pkl")
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]  # drop first to match prior conventions
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None

def load_hmap_pcn(scale: int) -> Optional[np.ndarray]:
    fn = f"hmap_pcn_scale_{scale}.pkl"
    fp = os.path.join(_hmaps_root(), fn)
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]  # match prior behavior
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None

# ------------------------------
# Core SAI / MSAI computation
# ------------------------------
def _bin_stats_for_all_cells(
    x: np.ndarray, y: np.ndarray, pcn: np.ndarray, gridsize: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For all cells, compute mean activation per (i,j) grid bin.
    Returns:
      mean_act: (gx, gy, C) mean activation per bin per cell
      xcenters: (gx,) bin centers in x
      ycenters: (gy,) bin centers in y
      valid_mask: (gx, gy) boolean bins that were visited by any cell
    """
    assert x.shape[0] == y.shape[0] == pcn.shape[0], "Time dimension mismatch"

    # grid edges
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    T, C = pcn.shape
    gx = gridsize
    gy = gridsize

    # accumulators
    total = np.zeros((gx, gy, C), dtype=np.float64)
    count = np.zeros((gx, gy, C), dtype=np.float64)

    # For efficiency, pre-bin all timesteps once
    ix = np.digitize(x, xedges) - 1
    iy = np.digitize(y, yedges) - 1
    ix = np.clip(ix, 0, gx - 1)
    iy = np.clip(iy, 0, gy - 1)

    # Accumulate per cell
    # We treat nonpositive activations as zero (no contribution)
    pos_mask = pcn > 0.0
    for c in range(C):
        mask = pos_mask[:, c]
        if not np.any(mask):
            continue
        # gather indices and activations
        ix_c = ix[mask]
        iy_c = iy[mask]
        a_c = pcn[mask, c]
        np.add.at(total[..., c], (ix_c, iy_c), a_c)
        np.add.at(count[..., c], (ix_c, iy_c), 1.0)

    mean_act = np.zeros_like(total)
    nz = count > 0
    mean_act[nz] = total[nz] / count[nz]

    # any bin visited by any cell
    valid_mask = (count.sum(axis=2) > 0)

    return mean_act, xcenters, ycenters, valid_mask

def _per_cell_threshold_and_normalize(mean_act: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """
    For each cell (last axis), threshold at its per-cell quantile and normalize to [0,1].
    Inputs:
      mean_act: (gx, gy, C)
    Returns:
      proc_act: (gx, gy, C)
    """
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
    """
    Row-wise cosine similarity for A (N x C) -> (N x N)
    """
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    An = A / norms
    return An @ An.T

def compute_sai_map_and_msai(
    x: np.ndarray,
    y: np.ndarray,
    pcn: np.ndarray,
    gridsize: int = 100,
    dist_threshold: Optional[float] = None,
    quantile: float = 0.10
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute SAI per bin (as a grid) and MSAI scalar for a single scale.

    Returns:
      sai_grid: (gx, gy) SAI values per grid bin (0 where invalid)
      msai: scalar mean over valid bins
      xcenters, ycenters: 1D arrays of bin centers
    """
    mean_act, xcenters, ycenters, valid_mask = _bin_stats_for_all_cells(x, y, pcn, gridsize=gridsize)
    proc = _per_cell_threshold_and_normalize(mean_act, quantile=quantile)

    # Flatten valid bins to a list for pairwise similarities
    gx, gy, C = proc.shape
    XX, YY = np.meshgrid(xcenters, ycenters, indexing='ij')
    bin_xy = np.stack([XX, YY], axis=-1)  # (gx, gy, 2)

    # Indices of valid bins
    idxs = np.argwhere(valid_mask)  # (Nb, 2)
    if idxs.shape[0] == 0:
        return np.zeros((gx, gy)), 0.0, xcenters, ycenters

    # Build population activation vectors per valid bin
    A = proc[valid_mask, :]  # (Nb, C)
    centers = bin_xy[valid_mask, :]  # (Nb, 2)

    # Distance threshold (if None, choose 0.3 * max span)
    if dist_threshold is None:
        span = max(xcenters.max() - xcenters.min(), ycenters.max() - ycenters.min())
        dist_threshold = 0.3 * span

    # Pairwise cosine similarity
    S = _cosine_similarity(A)  # (Nb x Nb)

    # Compute distance mask for "far" bins
    ci2 = np.sum(centers**2, axis=1, keepdims=True)
    d2 = ci2 + ci2.T - 2 * (centers @ centers.T)
    D = np.sqrt(np.clip(d2, 0.0, None))

    far_mask = (D > dist_threshold)
    np.fill_diagonal(far_mask, False)

    denom = np.maximum(far_mask.sum(axis=1), 1)
    sai_vals = (S * far_mask).sum(axis=1) / denom

    # Scatter back to grid
    sai_grid = np.zeros((gx, gy), dtype=np.float64)
    for (i, (ix, iy)) in enumerate(idxs):
        sai_grid[ix, iy] = sai_vals[i]

    # MSAI = mean over valid bins
    msai = sai_vals.mean() if sai_vals.size > 0 else 0.0
    return sai_grid, float(msai), xcenters, ycenters

# ------------------------------
# Plotting combined figure (1 x N) like paper_visuals_selected.py
# ------------------------------
def _plot_sai_panel(ax, xcenters, ycenters, sai_grid, cmap='jet', add_colorbar=False):
    # Interpolate to a smooth field for publication-style contourf (matches visual tone)
    XI, YI = np.meshgrid(
        np.linspace(xcenters.min(), xcenters.max(), 150),
        np.linspace(ycenters.min(), ycenters.max(), 150),
        indexing='xy'
    )
    # griddata wants scattered points, so build them
    gx, gy = sai_grid.shape
    XX, YY = np.meshgrid(xcenters, ycenters, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    vals = sai_grid.T.ravel()  # transpose to align indexing ('xy') for display
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
    return cs

def generate_msai_heatmaps(
    scales: List[int] = [0, 1, 2],
    gridsize: int = 240,
    quantile: float = 0.10,
    dist_threshold: Optional[float] = None,
    cmap: str = 'jet',
    output_dir: Optional[str] = None,
    fname: str = 'msai_all_scales.png'
) -> Tuple[str, Dict[int, float]]:
    """
    Computes SAI maps and MSAI for each scale and saves a combined 1xN heatmap figure.
    Returns (path, {scale: MSAI})
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "paper_figures", "msai")
    os.makedirs(output_dir, exist_ok=True)

    # load location data
    hmap_loc = load_hmap_loc()
    if hmap_loc is None:
        raise FileNotFoundError("hmap_loc.pkl not found.")

    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)

    n = len(scales)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)

    msai_per_scale: Dict[int, float] = {}

    for col, scale in enumerate(scales):
        pcn = load_hmap_pcn(scale)
        ax = axes[0, col]
        if pcn is None:
            ax.text(0.5, 0.5, f'No PCN for scale {scale}', ha='center', va='center', transform=ax.transAxes)
            continue

        sai_grid, msai, xcenters, ycenters = compute_sai_map_and_msai(
            hmap_x, hmap_y, pcn, gridsize=gridsize, dist_threshold=dist_threshold, quantile=quantile
        )
        msai_per_scale[scale] = msai

        _plot_sai_panel(ax, xcenters, ycenters, sai_grid, cmap=cmap, add_colorbar=(col == n-1))
        ax.set_title(f"{SCALE_NAMES.get(scale, str(scale))}", fontsize=18)

    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return out_path, msai_per_scale

# ------------------------------
# Script entry
# ------------------------------
if __name__ == "__main__":
    path, msai = generate_msai_heatmaps(
        scales=[0, 1, 2],
        gridsize=120,
        quantile=0.10,        # 10% per-cell threshold
        dist_threshold=None,  # auto: 0.3 * max span
        cmap='jet',
        output_dir=None,
        fname='msai_all_scales.png'
    )
    print("MSAI per scale:", msai)
