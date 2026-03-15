"""
Place Field Gate Simulator
==========================
Visualises the Gaussian proximity gate that controls which scale of place cells
forms at each location in the environment.

Panels
------
Row 0 : per-scale gate value g_s(proximity)   [0,1]  – prediction of where
        place fields of each scale will form.
Row 1 : proximity map (input to gate) + dominant-scale map.
"""

import copy
import json
import threading
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Environment definitions  (X-Z floor plane, Y is up in Webots)
# bounds format: [[x1, z1], [x2, z2]]
# ─────────────────────────────────────────────────────────────────────────────
WORLD_CONFIGS: Dict[str, Dict] = {
    "environment_1": {
        "size": [20.0, 20.0],
        "obstacles": [],
    },
    "environment_2": {
        "size": [20.0, 20.0],
        "obstacles": [
            {"type": "rectangle", "name": "wall1", "bounds": [[-0.1, -4.0],  [ 0.1, 10.0]]},
            {"type": "rectangle", "name": "wall2", "bounds": [[-0.1, -9.8],  [ 0.1, -5.8]]},
        ],
    },
    "environment_3": {
        "size": [20.0, 20.0],
        "obstacles": [
            {"type": "rectangle", "name": "wall1", "bounds": [[-0.1, -4.0],  [ 0.1, 10.0]]},
            {"type": "rectangle", "name": "wall2", "bounds": [[-0.1, -9.8],  [ 0.1, -5.8]]},
            {"type": "rectangle", "name": "wall3", "bounds": [[ 0.0,  4.6],  [ 8.0,  4.8]]},
        ],
    },
    "environment_4": {
        "size": [20.0, 20.0],
        "obstacles": [
            {"type": "rectangle", "name": "wall1",  "bounds": [[-0.1, -4.0],  [ 0.1, 10.0]]},
            {"type": "rectangle", "name": "wall2",  "bounds": [[-0.1, -9.8],  [ 0.1, -5.8]]},
            {"type": "rectangle", "name": "wall3",  "bounds": [[ 0.0,  4.6],  [ 8.0,  4.8]]},
            {"type": "rectangle", "name": "wall4",  "bounds": [[-10.0, 4.6],  [-2.0,  4.8]]},
            {"type": "rectangle", "name": "wall5",  "bounds": [[-6.1, -9.8],  [-5.9,  3.2]]},
            {"type": "rectangle", "name": "wall6",  "bounds": [[-5.9, -5.9],  [-1.9, -5.7]]},
            {"type": "rectangle", "name": "wall7",  "bounds": [[ 2.0, -4.0],  [ 8.0, -3.8]]},
        ],
    },
    "environment_5": {
        "size": [20.0, 20.0],
        "obstacles": [
            {"type": "rectangle", "name": "wall1",  "bounds": [[-0.1, -4.0],  [ 0.1, 10.0]]},
            {"type": "rectangle", "name": "wall2",  "bounds": [[-0.1, -9.8],  [ 0.1, -5.8]]},
            {"type": "rectangle", "name": "wall3",  "bounds": [[ 0.0,  4.6],  [ 8.0,  4.8]]},
            {"type": "rectangle", "name": "wall4",  "bounds": [[-10.0, 4.6],  [-2.0,  4.8]]},
            {"type": "rectangle", "name": "wall5",  "bounds": [[-6.1, -9.8],  [-5.9,  3.2]]},
            {"type": "rectangle", "name": "wall6",  "bounds": [[-5.9, -5.9],  [-1.9, -5.7]]},
            {"type": "rectangle", "name": "wall7",  "bounds": [[ 2.0, -4.0],  [ 8.0, -3.8]]},
            {"type": "rectangle", "name": "wall8",  "bounds": [[ 4.9, -3.9],  [ 5.1,  4.3]]},
            {"type": "rectangle", "name": "wall9",  "bounds": [[ 4.9, -8.95], [ 5.1, -5.45]]},
            {"type": "rectangle", "name": "wall10", "bounds": [[ 5.0, -5.7],  [ 9.8, -5.5]]},
            {"type": "rectangle", "name": "wall11", "bounds": [[-4.85,-4.0],  [-1.15,-3.8]]},
        ],
    },
    "environment_6": {
        "size": [20.0, 20.0],
        "obstacles": [
            {"type": "rectangle", "name": "wall1",    "bounds": [[-0.1,  -4.0],   [ 0.1,  10.0]]},
            {"type": "rectangle", "name": "wall2",    "bounds": [[-0.1,  -9.8],   [ 0.1,  -5.8]]},
            {"type": "rectangle", "name": "wall3",    "bounds": [[ 0.0,   4.6],   [ 8.0,   4.8]]},
            {"type": "rectangle", "name": "wall4",    "bounds": [[-10.0,  4.6],   [-2.0,   4.8]]},
            {"type": "rectangle", "name": "wall5",    "bounds": [[-6.1,  -9.8],   [-5.9,   3.2]]},
            {"type": "rectangle", "name": "wall6",    "bounds": [[-5.9,  -5.9],   [-1.9,  -5.7]]},
            {"type": "rectangle", "name": "wall7",    "bounds": [[ 2.0,  -4.0],   [ 8.0,  -3.8]]},
            {"type": "rectangle", "name": "wall8",    "bounds": [[ 4.9,  -3.9],   [ 5.1,   4.3]]},
            {"type": "rectangle", "name": "wall9",    "bounds": [[ 4.9,  -8.95],  [ 5.1,  -5.45]]},
            {"type": "rectangle", "name": "wall10",   "bounds": [[ 5.0,  -5.7],   [ 9.8,  -5.5]]},
            {"type": "rectangle", "name": "wall11",   "bounds": [[-4.85, -4.0],   [-1.15, -3.8]]},
            {"type": "rectangle", "name": "wall3_1",  "bounds": [[ 2.0,   7.3],   [10.0,   7.5]]},
            {"type": "rectangle", "name": "wall3_2",  "bounds": [[-10.0,  7.3],   [-2.0,   7.5]]},
            {"type": "rectangle", "name": "wall13",   "bounds": [[ 0.0,  -0.2],   [ 5.0,   0.0]]},
            {"type": "rectangle", "name": "wall13_1", "bounds": [[ 6.6,  -0.2],   [ 8.6,   0.0]]},
            {"type": "rectangle", "name": "wall11_2", "bounds": [[ 1.4,  -6.2],   [ 3.6,  -6.0]]},
            {"type": "rectangle", "name": "wall9_1",  "bounds": [[ 2.4,  -9.75],  [ 2.6,  -6.25]]},
            {"type": "rectangle", "name": "wall8_2",  "bounds": [[ 4.9,   5.9],   [ 5.1,   8.7]]},
            {"type": "rectangle", "name": "wall8_3",  "bounds": [[-5.1,   5.9],   [-4.9,   8.7]]},
            {"type": "rectangle", "name": "wall8_4",  "bounds": [[-8.7,  -0.6],   [-7.3,  -0.4]]},
            {"type": "rectangle", "name": "wall8_1",  "bounds": [[ 7.5,  -0.05],  [ 7.7,   4.65]]},
            {"type": "rectangle", "name": "wall8_sub","bounds": [[ 4.9,  -5.45],  [ 5.1,  -2.95]]},
            {"type": "rectangle", "name": "wall12",   "bounds": [[-3.1,  -3.9],   [-2.9,   4.3]]},
            {"type": "rectangle", "name": "wall12_1", "bounds": [[-7.9,  -9.85],  [-7.7,  -0.55]]},
        ],
    },
}

GOAL_POSITIONS = {
    "red":    ( 9.0,  9.0),
    "green":  (-9.0,  9.0),
    "blue":   ( 9.0, -9.0),
    "yellow": (-9.0, -9.0),
}
GOAL_COLORS = {
    "red": "red", "green": "limegreen",
    "blue": "dodgerblue", "yellow": "gold",
}

# ─────────────────────────────────────────────────────────────────────────────
# Default scale configurations (matches msg_controller.py SCALES_DEFS_GRID)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SCALES = [
    {"name": "Small",  "d_opt": 0.7, "sigma_r": 0.5, "sigma_tune_k": 1.00,
     "large_one_sided": False, "plateau": 1.0, "gamma_cross": 0.40, "color": "#e74c3c"},
    {"name": "Medium", "d_opt": 2.5, "sigma_r": 1.0, "sigma_tune_k": 0.75,
     "large_one_sided": False, "plateau": 1.0, "gamma_cross": 0.43, "color": "#2ecc71"},
    {"name": "Large",  "d_opt": 5.0, "sigma_r": 1.5, "sigma_tune_k": 0.70,
     "large_one_sided": True,  "plateau": 1.0, "gamma_cross": 0.43, "color": "#3498db"},
]

# Gate modes
# "normal"               : standard g_s(proximity) Gaussian gate
# "no_gate_no_inhibition": g_s = 1 everywhere; cross-scale inhibition also removed
# "no_gate_with_inhibition": g_s = 1 on excitation side; cross-scale inhibition
#                            (1 - g_s)*gamma_cross still subtracts from effective activation
GATE_MODES = ["normal", "no_gate_no_inhibition", "no_gate_with_inhibition"]

# ─────────────────────────────────────────────────────────────────────────────
# LiDAR ray casting (vectorised over positions, loop over segments)
# ─────────────────────────────────────────────────────────────────────────────
def _build_segments(wc: Dict) -> Tuple[List, List]:
    """
    Return (x_segs, z_segs).
      x_segs: list of (x_coord, z_lo, z_hi)  — vertical walls
      z_segs: list of (z_coord, x_lo, x_hi)  — horizontal walls
    """
    W, H = wc["size"]
    hw, hh = W / 2.0, H / 2.0

    x_segs = [(-hw, -hh, hh), (hw, -hh, hh)]          # world left / right
    z_segs = [(-hh, -hw, hw), (hh, -hw, hw)]           # world bottom / top

    for obs in wc.get("obstacles", []):
        if obs.get("type") == "rectangle":
            (x1, z1), (x2, z2) = obs["bounds"]
            xmin, xmax = min(x1, x2), max(x1, x2)
            zmin, zmax = min(z1, z2), max(z1, z2)
            x_segs += [(xmin, zmin, zmax), (xmax, zmin, zmax)]
            z_segs += [(zmin, xmin, xmax), (zmax, xmin, xmax)]

    return x_segs, z_segs


def cast_lidar(positions: np.ndarray,
               x_segs: List, z_segs: List,
               n_rays: int = 720,
               max_dist: float = 20.0) -> np.ndarray:
    """
    Cast n_rays LiDAR beams from each position.

    positions : (N, 2)  in (x, z)
    returns   : (N, n_rays) distances
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    dx = np.cos(angles)   # (R,)
    dz = np.sin(angles)   # (R,)

    px = positions[:, 0]  # (N,)
    pz = positions[:, 1]  # (N,)

    dist = np.full((len(positions), n_rays), max_dist, dtype=np.float32)

    # Vertical walls  x = c_x,  z ∈ [lo_z, hi_z]
    for c_x, lo_z, hi_z in x_segs:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c_x - px[:, None]) / dx[None, :]          # (N, R)
        hit_z = pz[:, None] + t * dz[None, :]
        valid = (t > 1e-6) & (hit_z >= lo_z) & (hit_z <= hi_z) & (t < dist)
        dist = np.where(valid, t, dist)

    # Horizontal walls  z = c_z,  x ∈ [lo_x, hi_x]
    for c_z, lo_x, hi_x in z_segs:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c_z - pz[:, None]) / dz[None, :]          # (N, R)
        hit_x = px[:, None] + t * dx[None, :]
        valid = (t > 1e-6) & (hit_x >= lo_x) & (hit_x <= hi_x) & (t < dist)
        dist = np.where(valid, t, dist)

    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Proximity estimators (vectorised over N positions)
# ─────────────────────────────────────────────────────────────────────────────
def _prox_min(lidar: np.ndarray) -> np.ndarray:
    return np.min(lidar, axis=1)


def _prox_mean(lidar: np.ndarray) -> np.ndarray:
    return np.mean(lidar, axis=1)


def _prox_local_minima(lidar: np.ndarray) -> np.ndarray:
    """5-ray circular median filter → bilateral argmin constraint."""
    sm = np.median(np.stack([
        np.roll(lidar, -2, axis=1),
        np.roll(lidar, -1, axis=1),
        lidar,
        np.roll(lidar,  1, axis=1),
        np.roll(lidar,  2, axis=1),
    ], axis=0), axis=0)

    N, n = sm.shape
    idx1 = np.argmin(sm, axis=1)
    d1   = sm[np.arange(N), idx1]

    opp  = (idx1 + n // 2) % n
    hw   = n // 4
    offs = np.arange(-hw, hw + 1)
    oi   = (opp[:, None] + offs[None, :]) % n           # (N, W)
    d2   = np.min(sm[np.arange(N)[:, None], oi], axis=1)

    return (d1 + d2) / 2.0


def _prox_raw_local_minima(lidar: np.ndarray) -> np.ndarray:
    """Strict local minima on raw scan (no median filter)."""
    f  = lidar
    n  = f.shape[1]
    N  = len(f)

    prev = np.roll(f, 1,  axis=1)
    nxt  = np.roll(f, -1, axis=1)
    is_lm = (f < prev) & (f < nxt)

    masked  = np.where(is_lm, f, np.inf)
    has_lm  = np.any(is_lm, axis=1)
    idx1    = np.where(has_lm, np.argmin(masked, axis=1), np.argmin(f, axis=1))
    d1      = f[np.arange(N), idx1]

    opp  = (idx1 + n // 2) % n
    hw   = n // 4
    offs = np.arange(-hw, hw + 1)
    oi   = (opp[:, None] + offs[None, :]) % n
    wv   = f[np.arange(N)[:, None], oi]
    olm  = is_lm[np.arange(N)[:, None], oi]
    lmv  = np.where(olm, wv, np.inf)
    d2   = np.where(np.any(olm, axis=1), np.min(lmv, axis=1), np.min(wv, axis=1))

    return (d1 + d2) / 2.0


PROX_FUNCS = {
    "min":               _prox_min,
    "mean":              _prox_mean,
    "local_minima":      _prox_local_minima,
    "raw_local_minima":  _prox_raw_local_minima,
}


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian scale gate
# ─────────────────────────────────────────────────────────────────────────────
def compute_gate(prox: np.ndarray, d_opt: float, sigma_tune: float,
                 one_sided: bool, plateau: float) -> np.ndarray:
    g = np.exp(-((prox - d_opt) ** 2) / (2.0 * max(sigma_tune, 1e-6) ** 2))
    if one_sided:
        g = np.where(prox >= d_opt, plateau, g)
    return np.clip(g, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Free-space mask
# ─────────────────────────────────────────────────────────────────────────────
def build_free_mask(xs: np.ndarray, zs: np.ndarray,
                    obstacles: List, clearance: float = 0.12) -> np.ndarray:
    X, Z = np.meshgrid(xs, zs)
    free = np.ones(X.shape, dtype=bool)
    for obs in obstacles:
        if obs.get("type") == "rectangle":
            (x1, z1), (x2, z2) = obs["bounds"]
            xmn = min(x1, x2) - clearance;  xmx = max(x1, x2) + clearance
            zmn = min(z1, z2) - clearance;  zmx = max(z1, z2) + clearance
            free &= ~((X >= xmn) & (X <= xmx) & (Z >= zmn) & (Z <= zmx))
    return free


# ─────────────────────────────────────────────────────────────────────────────
# Simulator GUI
# ─────────────────────────────────────────────────────────────────────────────
class PlaceFieldGateSimulator:

    N_COLS = 3   # = max(len(DEFAULT_SCALES), 2)
    N_ROWS = 3   # row 0: per-scale gates  row 1: prox + dominant  row 2: pairwise overlaps

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Place Field Gate Simulator")
        self.root.geometry("1500x860")

        self.world_name   = "environment_6"
        self.resolution   = 64
        self.n_rays       = 720
        self.prox_mode    = "local_minima"
        self.gate_mode    = "normal"
        self.max_dist     = 20.0
        self.scale_params = copy.deepcopy(DEFAULT_SCALES)

        # cache keys
        self._lidar_cache: Optional[np.ndarray] = None
        self._lidar_key:   Optional[tuple]       = None
        self._xs = self._zs = None
        self._colorbars: list = []

        self._build_gui()
        self._trigger_update()

    # ── GUI construction ──────────────────────────────────────────────────────
    def _build_gui(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── left scrollable panel ──
        left_frame = ttk.Frame(paned, width=290)
        paned.add(left_frame, weight=0)
        lc = tk.Canvas(left_frame, width=280, highlightthickness=0)
        sb = ttk.Scrollbar(left_frame, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(lc)
        lc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<MouseWheel>", lambda e: lc.yview_scroll(-1*(e.delta//120), "units"))

        def row_label(text, bold=False, color=None, r=[0]):
            kw = {"row": r[0], "column": 0, "columnspan": 2,
                  "sticky": tk.W, "padx": 6, "pady": (6, 1)}
            font = ("", 10, "bold") if bold else ("", 9)
            lbl = ttk.Label(inner, text=text, font=font)
            if color:
                lbl.configure(foreground=color)
            lbl.grid(**kw)
            r[0] += 1

        def row_combo(label, var, values, r=[0]):
            ttk.Label(inner, text=label).grid(
                row=r[0], column=0, sticky=tk.W, padx=6, pady=1)
            cb = ttk.Combobox(inner, textvariable=var, values=values,
                              state="readonly", width=14)
            cb.grid(row=r[0], column=1, sticky=tk.W, pady=1)
            r[0] += 1
            return cb

        def row_entry(label, var, r=[0]):
            ttk.Label(inner, text=label).grid(
                row=r[0], column=0, sticky=tk.W, padx=6, pady=1)
            e = ttk.Entry(inner, textvariable=var, width=9)
            e.grid(row=r[0], column=1, sticky=tk.W, pady=1)
            r[0] += 1
            return e

        def sep(r=[0]):
            ttk.Separator(inner, orient=tk.HORIZONTAL).grid(
                row=r[0], column=0, columnspan=2, sticky=tk.EW,
                padx=4, pady=5)
            r[0] += 1

        # shared row counter via mutable default — reset per section
        _r = [0]
        def rl(t, bold=False, color=None): row_label(t, bold, color, _r)
        def rc(l, v, vals):               return row_combo(l, v, vals, _r)
        def re(l, v):                     return row_entry(l, v, _r)
        def sp():                         sep(_r)

        rl("Environment", bold=True)
        self.world_var = tk.StringVar(value=self.world_name)
        cb = ttk.Combobox(inner, textvariable=self.world_var,
                          values=list(WORLD_CONFIGS.keys()),
                          state="readonly", width=20)
        cb.grid(row=_r[0], column=0, columnspan=2,
                sticky=tk.EW, padx=6, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._invalidate_lidar())
        _r[0] += 1
        sp()

        rl("Computation", bold=True)
        self.res_var  = tk.StringVar(value=str(self.resolution))
        self.rays_var = tk.StringVar(value=str(self.n_rays))
        self.prox_var = tk.StringVar(value=self.prox_mode)
        rc("Resolution:", self.res_var,
           ["32", "48", "64", "80", "100", "128"])
        rc("LiDAR rays:", self.rays_var, ["180", "360", "720"])
        rc("Proximity mode:", self.prox_var,
           ["min", "mean", "local_minima", "raw_local_minima"])
        self.gate_mode_var = tk.StringVar(value=self.gate_mode)
        rc("Gate mode:", self.gate_mode_var, GATE_MODES)
        sp()

        rl("Scale Parameters", bold=True)
        self._scale_vars = []
        for sc in self.scale_params:
            rl(f"── {sc['name']} ──", color=sc["color"])
            d_v  = tk.StringVar(value=str(sc["d_opt"]))
            stk  = tk.StringVar(value=str(sc["sigma_tune_k"]))
            sr_v = tk.StringVar(value=str(sc["sigma_r"]))
            gc_v = tk.StringVar(value=str(sc["gamma_cross"]))
            os_v = tk.BooleanVar(value=sc["large_one_sided"])
            re("  d_opt (m):",    d_v)
            re("  σ_tune_k:",     stk)
            re("  σ_r (m):",      sr_v)
            re("  γ_cross:",      gc_v)
            ttk.Checkbutton(inner, text="  one-sided gate",
                            variable=os_v).grid(
                row=_r[0], column=0, columnspan=2,
                sticky=tk.W, padx=6, pady=1)
            _r[0] += 1
            self._scale_vars.append((d_v, stk, sr_v, gc_v, os_v))
        sp()

        self.update_btn = ttk.Button(
            inner, text="Update Simulation",
            command=self._trigger_update)
        self.update_btn.grid(row=_r[0], column=0, columnspan=2,
                             padx=6, pady=8, sticky=tk.EW)
        _r[0] += 1

        # ── Presets ──
        rl("Presets", bold=True)
        self.preset_name_var = tk.StringVar(value="")
        ttk.Entry(inner, textvariable=self.preset_name_var,
                  width=18).grid(row=_r[0], column=0, columnspan=2,
                                 sticky=tk.EW, padx=6, pady=2)
        _r[0] += 1
        ttk.Button(inner, text="Save Preset",
                   command=self._save_preset).grid(
            row=_r[0], column=0, columnspan=2,
            sticky=tk.EW, padx=6, pady=2)
        _r[0] += 1
        sp()
        self.preset_load_var = tk.StringVar(value="")
        self.preset_cb = ttk.Combobox(inner, textvariable=self.preset_load_var,
                                      state="readonly", width=18)
        self.preset_cb.grid(row=_r[0], column=0, columnspan=2,
                            sticky=tk.EW, padx=6, pady=2)
        _r[0] += 1
        ttk.Button(inner, text="Load Preset",
                   command=self._load_preset).grid(
            row=_r[0], column=0, columnspan=2,
            sticky=tk.EW, padx=6, pady=2)
        _r[0] += 1
        ttk.Button(inner, text="Refresh List",
                   command=self._refresh_presets).grid(
            row=_r[0], column=0, columnspan=2,
            sticky=tk.EW, padx=6, pady=2)
        _r[0] += 1
        self._refresh_presets()

        self.status_var = tk.StringVar(value="Initialising…")
        ttk.Label(inner, textvariable=self.status_var,
                  foreground="gray", wraplength=240,
                  justify=tk.LEFT).grid(
            row=_r[0], column=0, columnspan=2, padx=6, pady=4)
        _r[0] += 1

        # ── right matplotlib panel ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.fig, self.axes = plt.subplots(
            self.N_ROWS, self.N_COLS,
            figsize=(14, 10),
            constrained_layout=True,
        )
        if self.axes.ndim == 1:
            self.axes = self.axes.reshape(self.N_ROWS, self.N_COLS)

        self.fig.patch.set_facecolor("#111")
        for ax in self.axes.flat:
            ax.set_facecolor("#111")
            ax.tick_params(colors="#ccc", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    # ── presets ───────────────────────────────────────────────────────────────
    @property
    def _presets_dir(self) -> Path:
        d = Path(__file__).parent / "gate_presets"
        d.mkdir(exist_ok=True)
        return d

    def _save_preset(self):
        if not self._read_params():
            return
        name = self.preset_name_var.get().strip()
        if not name:
            messagebox.showerror("Save Preset", "Enter a preset name first.")
            return
        data = {
            "world_name":  self.world_name,
            "resolution":  self.resolution,
            "n_rays":      self.n_rays,
            "prox_mode":   self.prox_mode,
            "gate_mode":   self.gate_mode,
            "scales":      copy.deepcopy(self.scale_params),
        }
        path = self._presets_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status_var.set(f"Saved: {name}.json")
        self._refresh_presets()

    def _load_preset(self):
        name = self.preset_load_var.get().strip()
        if not name:
            return
        path = self._presets_dir / f"{name}.json"
        if not path.exists():
            messagebox.showerror("Load Preset", f"File not found: {path.name}")
            return
        with open(path) as f:
            data = json.load(f)
        # Apply to GUI vars
        self.world_var.set(data.get("world_name", self.world_name))
        self.world_name = data["world_name"]
        self.res_var.set(str(data.get("resolution", self.resolution)))
        self.rays_var.set(str(data.get("n_rays", self.n_rays)))
        self.prox_var.set(data.get("prox_mode", self.prox_mode))
        self.gate_mode_var.set(data.get("gate_mode", self.gate_mode))
        for i, sc in enumerate(data.get("scales", [])):
            if i >= len(self._scale_vars):
                break
            d_v, stk, sr_v, gc_v, os_v = self._scale_vars[i]
            d_v.set(str(sc.get("d_opt",        self.scale_params[i]["d_opt"])))
            stk.set(str(sc.get("sigma_tune_k", self.scale_params[i]["sigma_tune_k"])))
            sr_v.set(str(sc.get("sigma_r",     self.scale_params[i]["sigma_r"])))
            gc_v.set(str(sc.get("gamma_cross", self.scale_params[i]["gamma_cross"])))
            os_v.set(bool(sc.get("large_one_sided", self.scale_params[i]["large_one_sided"])))
        self._lidar_cache = None   # may need re-cast if env changed
        self.status_var.set(f"Loaded: {name}.json")
        self._trigger_update()

    def _refresh_presets(self):
        names = sorted(p.stem for p in self._presets_dir.glob("*.json"))
        self.preset_cb["values"] = names
        if names and not self.preset_load_var.get():
            self.preset_load_var.set(names[0])

    # ── helpers ───────────────────────────────────────────────────────────────
    def _invalidate_lidar(self):
        self.world_name = self.world_var.get()
        self._lidar_cache = None
        self._trigger_update()

    def _read_params(self) -> bool:
        try:
            self.world_name = self.world_var.get()
            self.resolution = int(self.res_var.get())
            self.n_rays     = int(self.rays_var.get())
            self.prox_mode  = self.prox_var.get()
            self.gate_mode  = self.gate_mode_var.get()
            for i, (d_v, stk, sr_v, gc_v, os_v) in enumerate(self._scale_vars):
                self.scale_params[i]["d_opt"]          = float(d_v.get())
                self.scale_params[i]["sigma_tune_k"]   = float(stk.get())
                self.scale_params[i]["sigma_r"]        = float(sr_v.get())
                self.scale_params[i]["gamma_cross"]    = float(gc_v.get())
                self.scale_params[i]["large_one_sided"]= os_v.get()
            return True
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return False

    def _trigger_update(self):
        if not self._read_params():
            return
        self.update_btn.configure(state="disabled")
        self.status_var.set("Working…")
        self.root.update_idletasks()
        threading.Thread(target=self._run, daemon=True).start()

    # ── computation ───────────────────────────────────────────────────────────
    def _run(self):
        try:
            self._compute()
            self.root.after(0, self._plot)
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda: self.status_var.set(f"Error: {msg}"))
            self.root.after(0, lambda: self.update_btn.configure(state="normal"))

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _compute(self):
        wc  = WORLD_CONFIGS[self.world_name]
        W, H = wc["size"]
        res  = self.resolution

        xs = np.linspace(-W / 2 + 0.1, W / 2 - 0.1, res, dtype=np.float32)
        zs = np.linspace(-H / 2 + 0.1, H / 2 - 0.1, res, dtype=np.float32)
        self._xs, self._zs = xs, zs

        Xg, Zg = np.meshgrid(xs, zs)                        # (res, res)
        pos     = np.stack([Xg.ravel(), Zg.ravel()], axis=1) # (N, 2)

        free = build_free_mask(xs, zs, wc["obstacles"])      # (res, res)
        self._free = free

        # ── LiDAR (cached unless env/res/rays changed) ──
        key = (self.world_name, res, self.n_rays)
        if self._lidar_key != key or self._lidar_cache is None:
            self._set_status("Casting LiDAR rays…")
            x_segs, z_segs = _build_segments(wc)
            lidar = cast_lidar(pos, x_segs, z_segs,
                               n_rays=self.n_rays, max_dist=self.max_dist)
            lidar[~free.ravel(), :] = self.max_dist
            self._lidar_cache = lidar
            self._lidar_key   = key
        else:
            lidar = self._lidar_cache

        # ── Proximity ──
        self._set_status("Computing proximity…")
        fn   = PROX_FUNCS.get(self.prox_mode, _prox_min)
        prox = fn(lidar)                                      # (N,)
        pm   = prox.reshape(res, res).astype(np.float32)
        pm[~free] = np.nan
        self._prox_map = pm

        # ── Gate per scale ──
        self._gate_maps = []
        for sc in self.scale_params:
            sigma_tune  = sc["sigma_tune_k"] * sc["sigma_r"]
            gamma_cross = float(sc.get("gamma_cross", 0.4))
            # Always compute the proximity-based Gaussian (needed for inhibition term)
            g = compute_gate(prox, sc["d_opt"], sigma_tune,
                             sc["large_one_sided"], sc["plateau"])

            if self.gate_mode == "no_gate_no_inhibition":
                # Excitation gate = 1, inhibition term removed → uniform full activation
                effective = np.ones_like(g)
            elif self.gate_mode == "no_gate_with_inhibition":
                # Excitation gate = 1, but cross-scale inhibition (1 - g)*gamma_cross
                # still subtracts from effective activation.
                # effective = 1 - gamma_cross * (1 - g_s)
                effective = np.clip(1.0 - gamma_cross * (1.0 - g), 0.0, 1.0)
            else:
                # normal: show the gate directly
                effective = g

            gm = effective.reshape(res, res).astype(np.float32)
            gm[~free] = np.nan
            self._gate_maps.append(gm)

        # ── Dominant scale ──
        stack = np.stack(self._gate_maps, axis=0)             # (S, H, W)
        dom   = np.full((res, res), -1.0, dtype=np.float32)
        dom[free] = np.argmax(
            np.nan_to_num(stack, nan=0.0)[:, free], axis=0
        ).astype(np.float32)
        self._dom_map = dom

        # ── Pairwise overlap maps  (product of gate values) ──
        # For 3 scales: pairs (0,1), (0,2), (1,2) → fills exactly N_COLS columns.
        S = len(self._gate_maps)
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
        self._overlap_maps = []   # list of (i, j, overlap_map, mean_overlap)
        for i, j in pairs:
            ov = self._gate_maps[i] * self._gate_maps[j]   # element-wise product
            free_vals = ov[free]
            mean_ov   = float(np.nanmean(free_vals)) if free_vals.size else 0.0
            max_ov    = float(np.nanmax(free_vals))  if free_vals.size else 0.0
            self._overlap_maps.append((i, j, ov, mean_ov, max_ov))

    # ── plotting ──────────────────────────────────────────────────────────────
    def _plot(self):
        for cb in self._colorbars:
            cb.remove()
        self._colorbars.clear()
        for ax in self.axes.flat:
            ax.cla()
            ax.set_facecolor("#111")

        xs, zs = self._xs, self._zs
        ext    = [float(xs[0]), float(xs[-1]), float(zs[0]), float(zs[-1])]
        wc     = WORLD_CONFIGS[self.world_name]

        # Row 0 — per-scale gate maps
        for i, (gm, sc) in enumerate(zip(self._gate_maps, self.scale_params)):
            ax  = self.axes[0, i]
            im  = ax.imshow(gm, extent=ext, origin="lower",
                            cmap="viridis", vmin=0.0, vmax=1.0,
                            interpolation="bilinear", aspect="equal")
            cb  = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(colors="#ccc", labelsize=6)
            self._colorbars.append(cb)
            st  = sc["sigma_tune_k"] * sc["sigma_r"]
            ax.set_title(
                f"{sc['name']}   d_opt={sc['d_opt']}m   σ_tune={st:.2f}m"
                + ("   [one-sided]" if sc["large_one_sided"] else ""),
                color=sc["color"], fontsize=8, pad=3)
            ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
            ax.set_ylabel("Z (m)", color="#aaa", fontsize=7)
            ax.tick_params(colors="#ccc", labelsize=7)
            self._draw_env(ax, wc)

        # Row 1 col 0 — proximity map
        ax_p = self.axes[1, 0]
        im   = ax_p.imshow(self._prox_map, extent=ext, origin="lower",
                           cmap="plasma", interpolation="bilinear",
                           aspect="equal")
        cb   = self.fig.colorbar(im, ax=ax_p, fraction=0.046, pad=0.03,
                                 label="proximity (m)")
        cb.ax.tick_params(colors="#ccc", labelsize=6)
        self._colorbars.append(cb)
        ax_p.set_title(f"Proximity  [{self.prox_mode}]",
                       color="#ddd", fontsize=8, pad=3)
        ax_p.set_xlabel("X (m)", color="#aaa", fontsize=7)
        ax_p.set_ylabel("Z (m)", color="#aaa", fontsize=7)
        ax_p.tick_params(colors="#ccc", labelsize=7)
        self._draw_env(ax_p, wc)

        # Row 1 col 1 — dominant scale
        ax_d   = self.axes[1, 1]
        colors = ["#222"] + [sc["color"] for sc in self.scale_params]
        cmap_d = mcolors.ListedColormap(colors)
        dom_sh = self._dom_map + 1          # -1→0(obstacle), 0→1, 1→2 …
        im     = ax_d.imshow(dom_sh, extent=ext, origin="lower",
                             cmap=cmap_d,
                             vmin=0, vmax=len(self.scale_params),
                             interpolation="nearest", aspect="equal")
        ax_d.set_title("Dominant Scale", color="#ddd", fontsize=8, pad=3)
        ax_d.set_xlabel("X (m)", color="#aaa", fontsize=7)
        ax_d.set_ylabel("Z (m)", color="#aaa", fontsize=7)
        ax_d.tick_params(colors="#ccc", labelsize=7)
        legend_h = [patches.Patch(color=sc["color"], label=sc["name"])
                    for sc in self.scale_params]
        ax_d.legend(handles=legend_h, loc="upper right",
                    fontsize=7, framealpha=0.5)
        self._draw_env(ax_d, wc)

        # Row 1 remaining cols — hide
        for i in range(2, self.N_COLS):
            self.axes[1, i].set_visible(False)

        # Row 2 — pairwise overlap maps  (g_i × g_j)
        for col, (i, j, ov, mean_ov, max_ov) in enumerate(self._overlap_maps):
            if col >= self.N_COLS:
                break
            ax   = self.axes[2, col]
            ni   = self.scale_params[i]["name"]
            nj   = self.scale_params[j]["name"]
            im   = ax.imshow(ov, extent=ext, origin="lower",
                             cmap="hot", vmin=0.0, vmax=1.0,
                             interpolation="bilinear", aspect="equal")
            cb   = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(colors="#ccc", labelsize=6)
            self._colorbars.append(cb)
            ax.set_title(
                f"Overlap  "
                f"[{ni}]×[{nj}]   "
                f"μ={mean_ov:.3f}   max={max_ov:.3f}",
                color="#ddd", fontsize=8, pad=3)
            ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
            ax.set_ylabel("Z (m)", color="#aaa", fontsize=7)
            ax.tick_params(colors="#ccc", labelsize=7)
            self._draw_env(ax, wc)

        # Hide any leftover overlap columns
        for col in range(len(self._overlap_maps), self.N_COLS):
            self.axes[2, col].set_visible(False)

        self.fig.suptitle(
            f"{self.world_name}  |  res={self.resolution}  "
            f"rays={self.n_rays}  prox={self.prox_mode}",
            color="white", fontsize=9)

        self.canvas.draw()
        self.status_var.set("Done.")
        self.update_btn.configure(state="normal")

    def _draw_env(self, ax: plt.Axes, wc: Dict):
        for obs in wc.get("obstacles", []):
            if obs.get("type") == "rectangle":
                (x1, z1), (x2, z2) = obs["bounds"]
                xmn, xmx = min(x1, x2), max(x1, x2)
                zmn, zmx = min(z1, z2), max(z1, z2)
                ax.add_patch(patches.Rectangle(
                    (xmn, zmn), xmx - xmn, zmx - zmn,
                    linewidth=0.8, edgecolor="white",
                    facecolor="#555", alpha=0.85, zorder=3))
        for name, (gx, gz) in GOAL_POSITIONS.items():
            ax.plot(gx, gz, "o", color=GOAL_COLORS[name],
                    markersize=7, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6)


if __name__ == "__main__":
    root = tk.Tk()
    PlaceFieldGateSimulator(root)
    root.mainloop()
