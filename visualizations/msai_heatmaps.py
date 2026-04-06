import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from vis_utils import (
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
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
ENVIRONMENTS = [f"environment_{i}" for i in range(1, 7)]
MULTI_GOAL_WORLDS = {
    "20x20_multi_goal",
    "20x20_cross_multi_goal",
    "20x20_maze_multi_goal",
}
_CACHED_HMAPS_ROOT: Optional[str] = None


def _pkl_root() -> Path:
    return Path(CONTROLLER_PATH_PREFIX) / CONTROLLER_NAME / "pkl"


def _hmaps_root(env_name: str) -> str:
    return str(_pkl_root() / env_name / "hmaps")


def _resolve_hmaps_root(env_name: Optional[str] = None, allow_fallback: bool = True) -> Optional[str]:
    """
    Resolve an existing hmaps root.
    Priority:
    1) configured world path from vis_utils
    2) newest available */hmaps directory under controller pkl root containing hmap_loc.pkl
    """
    # Strict environment lookup mode (used by batch processing).
    if env_name is not None:
        strict = Path(_hmaps_root(env_name))
        if strict.exists() and (strict / "hmap_loc.pkl").exists():
            return str(strict)
        return None

    global _CACHED_HMAPS_ROOT
    if _CACHED_HMAPS_ROOT is not None:
        return _CACHED_HMAPS_ROOT

    preferred = Path(_hmaps_root(WORLD_NAME))
    if preferred.exists() and (preferred / "hmap_loc.pkl").exists():
        _CACHED_HMAPS_ROOT = str(preferred)
        return _CACHED_HMAPS_ROOT
    if not allow_fallback:
        return None

    pkl_root = _pkl_root()
    candidates = []
    if pkl_root.exists():
        for env_dir in pkl_root.iterdir():
            if not env_dir.is_dir():
                continue
            hmaps_dir = env_dir / "hmaps"
            hmap_loc = hmaps_dir / "hmap_loc.pkl"
            if hmaps_dir.exists() and hmap_loc.exists():
                candidates.append(hmaps_dir)

    if not candidates:
        return None

    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    print(
        f"[INFO] Using discovered hmaps folder: {chosen} "
        f"(configured WORLD_NAME='{WORLD_NAME}' not found)"
    )
    _CACHED_HMAPS_ROOT = str(chosen)
    return _CACHED_HMAPS_ROOT


def _default_output_dir(hmaps_root: str) -> str:
    """
    Default output next to the resolved environment data:
      .../pkl/<env>/hmaps -> .../pkl/<env>/vis_outputs/paper_figures/msai
    """
    root = Path(hmaps_root)
    env_dir = root.parent
    return str(env_dir / "vis_outputs" / "paper_figures" / "msai")


def discover_available_scales(hmaps_root: str) -> List[int]:
    root = Path(hmaps_root)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("hmap_pcn_scale_*.pkl")):
        try:
            out.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return out


def load_hmap_loc(hmaps_root: str) -> Optional[np.ndarray]:
    fp = os.path.join(hmaps_root, "hmap_loc.pkl")
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None


def _extract_planar_xy(hmap_loc: np.ndarray, env_name: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return planar coordinates from current-controller hmap_loc.

    Current msg_driver writes hmap_loc in canonical controller coordinates:
      [planar_x, planar_y, height]

    That is true for both:
    - environment_1 .. environment_6
    - 20x20_*_multi_goal worlds

    For unknown worlds, fall back to a simple heuristic:
    - treat the axis with the smallest range as height
    - use the remaining two axes as the planar coordinates
    """
    known_world = (
        env_name in ENVIRONMENTS
        or env_name in MULTI_GOAL_WORLDS
    )
    if known_world:
        return hmap_loc[:, 0], hmap_loc[:, 1]

    spans = np.ptp(hmap_loc, axis=0)
    height_axis = int(np.argmin(spans))
    planar_axes = [idx for idx in range(hmap_loc.shape[1]) if idx != height_axis]
    return hmap_loc[:, planar_axes[0]], hmap_loc[:, planar_axes[1]]


def load_hmap_pcn(hmaps_root: str, scale: int) -> Optional[np.ndarray]:
    fp = os.path.join(hmaps_root, f"hmap_pcn_scale_{scale}.pkl")
    try:
        with open(fp, "rb") as f:
            arr = np.array(pickle.load(f))
        return arr[1:]
    except FileNotFoundError:
        print(f"[ERROR] {fp} not found.")
        return None


def load_hmap_pcn_unified(hmaps_root: str, scales: List[int]) -> Optional[np.ndarray]:
    if not scales:
        return None
    per_scale = []
    min_steps = None
    for s in scales:
        arr = load_hmap_pcn(hmaps_root, s)
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
    ZI = np.nan_to_num(ZI, nan=0.0, posinf=0.0, neginf=0.0)

    zmin = float(np.min(ZI))
    zmax = float(np.max(ZI))

    # Handle degenerate fields (all equal): contourf requires strictly increasing levels.
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        zmin, zmax = 0.0, 1.0
        ZI = np.zeros_like(ZI)

    if zmax <= zmin + 1e-12:
        eps = 1e-6
        levels = np.linspace(zmin, zmin + eps, 25)
    else:
        levels = np.linspace(zmin, zmax, 25)

    cs = ax.contourf(XI, YI, ZI, levels=levels, cmap=cmap)
    if zmax > zmin + 1e-12:
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
    env_name: Optional[str] = None,
    scales: Optional[List[int]] = None,
    include_unified: bool = True,
    gridsize: int = 240,
    quantile: float = 0.10,
    dist_threshold: Optional[float] = None,
    cmap: str = 'jet',
    output_dir: Optional[str] = None,
    fname: str = 'msai_all_scales.png',
) -> Tuple[str, Dict[str, float]]:
    hmaps_root = _resolve_hmaps_root(env_name=env_name, allow_fallback=(env_name is None))
    if hmaps_root is None:
        raise FileNotFoundError(
            f"No hmap data found for environment '{env_name}'."
            if env_name
            else "No hmap data found under controller pkl directory."
        )

    if output_dir is None:
        output_dir = _default_output_dir(hmaps_root)
    os.makedirs(output_dir, exist_ok=True)

    if scales is None:
        scales = discover_available_scales(hmaps_root) or [0, 1, 2]

    hmap_loc = load_hmap_loc(hmaps_root)
    if hmap_loc is None:
        raise FileNotFoundError(f"hmap_loc.pkl not found in {hmaps_root}")

    hmap_x, hmap_y = _extract_planar_xy(hmap_loc, env_name)

    panels: List[Tuple[str, np.ndarray]] = []
    for scale in scales:
        pcn = load_hmap_pcn(hmaps_root, scale)
        if pcn is not None:
            n = min(hmap_x.shape[0], pcn.shape[0])
            panels.append((f"Scale {scale}", pcn[:n]))

    if include_unified and len(scales) >= 2:
        pcn_u = load_hmap_pcn_unified(hmaps_root, scales)
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


def generate_msai_heatmaps_all_environments(
    environments: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Generate MSAI plots/metrics for each environment with available data.
    Missing environments are skipped (no crash).
    """
    if environments is None:
        environments = ENVIRONMENTS

    results: Dict[str, Dict[str, float]] = {}
    for env_name in environments:
        hmaps_root = _resolve_hmaps_root(env_name=env_name, allow_fallback=False)
        if hmaps_root is None:
            print(f"[SKIP] {env_name}: no hmap_loc.pkl found")
            continue

        try:
            fname = f"msai_all_scales_{env_name}.png"
            out_path, msai = generate_msai_heatmaps(
                env_name=env_name,
                fname=fname,
                **kwargs,
            )
            results[env_name] = msai
            print(f"[OK] {env_name}: {out_path}")
        except FileNotFoundError as e:
            print(f"[SKIP] {env_name}: {e}")

    if not results:
        print("[WARN] No environments with usable MSAI inputs were found.")
    return results


if __name__ == "__main__":
    resolved_root = _resolve_hmaps_root(env_name=None, allow_fallback=True)
    if resolved_root is None:
        raise FileNotFoundError("No hmap data found under controller pkl directory.")

    resolved_world = Path(resolved_root).parent.name
    path, msai = generate_msai_heatmaps(
        env_name=resolved_world,
        scales=None,
        include_unified=True,
        gridsize=120,
        quantile=0.10,
        dist_threshold=None,
        cmap='jet',
        output_dir=None,
        fname=f"msai_all_scales_{resolved_world}.png",
    )
    print(f"MSAI for {resolved_world}:", msai)
    print(f"Saved: {path}")
