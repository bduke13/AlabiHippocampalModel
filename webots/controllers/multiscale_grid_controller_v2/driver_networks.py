"""Network setup helpers for Driver v2.

This module contains initialization-time logic for loading/creating:
- PCNs
- RCNs
- GCNs

Keeping these out of the runtime driver loop helps keep the driver focused on
simulation-time behavior.
"""

import math
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch

from core.layers.grid_cell_layer_v2 import GridCellLayer
from core.layers.multiscale_bvc import BoundaryVectorCellLayer
from core.layers.multiscale_pcn import PlaceCellLayer
from core.layers.multiscale_pcn_with_gcn_v2 import MultiscalePlaceCellWithGrid
from core.layers.reward_cell_layer_test import C_LAMBDA, RewardCellLayerTest

STEPS_PER_LAMBDA = 8


def _steps_for_scale(scale_def: Dict[str, Any]) -> int:
    sigma_pc_s = scale_def.get("sigma_pc_s", scale_def.get("sigma_r", 1.0))
    return int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA))


def load_pcns(driver, enable_ojas: Optional[bool], enable_stdp: Optional[bool]):
    driver.pcns = []
    for i, scale_def in enumerate(driver.scales):
        scale_idx = scale_def["scale_index"]
        fname = f"pcn_scale_{scale_idx}.pkl"
        path = os.path.join(driver.network_dir, fname)

        gcn = driver.gcns[i]
        num_grid_cells = scale_def.get("num_grid_cells", 0) if gcn else 0

        pcn = load_or_init_pcn_for_scale(
            driver,
            path,
            scale_def,
            num_grid_cells,
            enable_ojas,
            enable_stdp,
        )
        driver.pcns.append(pcn)


def load_or_init_pcn_for_scale(driver, path, scale_def, num_grid_cells, enable_ojas, enable_stdp):
    try:
        with open(path, "rb") as f:
            pcn = pickle.load(f)
        print(f"[DRIVER] Loaded existing PCN from {path}")

        pcn_class_name = pcn.__class__.__name__

        if pcn_class_name == "MultiscalePlaceCellWithGrid":
            if not isinstance(pcn, MultiscalePlaceCellWithGrid):
                # Migrate legacy pickles to the v2 class so runtime uses optimized methods.
                pcn.__class__ = MultiscalePlaceCellWithGrid
                print("[DRIVER] Migrated loaded PCN instance to MultiscalePlaceCellWithGrid v2 runtime class")

            if pcn.n_hd != driver.n_hd:
                print(f"[DRIVER] WARNING: Loaded PCN has n_hd={pcn.n_hd} but driver expects n_hd={driver.n_hd}")
                print("[DRIVER] Reinitializing hd_cell_trace tensor to match current n_hd")
                pcn.n_hd = driver.n_hd
                pcn.hd_cell_trace = torch.zeros((driver.n_hd, 1, 1), dtype=torch.float64, device=driver.device)
                if hasattr(pcn, "w_rec_hd"):
                    pcn.w_rec_hd = torch.zeros((driver.n_hd, pcn.num_pc, pcn.num_pc), dtype=pcn.dtype, device=driver.device)

            if enable_ojas is not None:
                pcn.enable_ojas = enable_ojas
            if enable_stdp is not None:
                pcn.enable_stdp = enable_stdp

            pcn.gamma_pp = scale_def.get("gamma_pp", 0.5)
            pcn.gamma_pb = scale_def.get("gamma_pb", 0.3)
            pcn.grid_influence = scale_def.get("grid_influence", 0.5)
            pcn.gamma_pg = scale_def.get("gamma_pg", 0.3)

            pcn.enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
            pcn.correlation_window = scale_def.get("correlation_window", 100)
            pcn.correlation_update_freq = scale_def.get("correlation_update_freq", 10)
            pcn.correlation_scaling = scale_def.get("correlation_scaling", 2.0)
            pcn.min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
            pcn.correlation_threshold = scale_def.get("correlation_threshold", 0.01)
            pcn.adjacency_learning_mode = scale_def.get("adjacency_learning_mode", "dense")
            pcn.adjacency_topk = scale_def.get("adjacency_topk")
            pcn.adjacency_activity_floor = scale_def.get("adjacency_activity_floor", 0.0)

            if scale_def.get("alpha_pb") is not None:
                pcn.alpha_pb = scale_def.get("alpha_pb")
            if scale_def.get("alpha_pg") is not None:
                pcn.alpha_pg = scale_def.get("alpha_pg")

            if hasattr(pcn, "_correlation_weights_dirty"):
                pcn._correlation_weights_dirty = True
            if hasattr(pcn, "_ensure_learning_caches"):
                pcn._ensure_learning_caches()
            pcn._ensure_runtime_caches()

            print(
                f"[DRIVER] Updated MultiscalePlaceCellWithGrid PCN - grid_influence: {pcn.grid_influence}, "
                f"gamma_pp: {pcn.gamma_pp}, gamma_pb: {pcn.gamma_pb}, alpha_pb: {pcn.alpha_pb:.4f}, alpha_pg: {pcn.alpha_pg:.4f}"
            )
        else:
            if enable_ojas is not None:
                pcn.enable_ojas = enable_ojas
            if enable_stdp is not None:
                pcn.enable_stdp = enable_stdp
            pcn.gamma_pp = scale_def.get("gamma_pp", 0.5)
            pcn.gamma_pb = scale_def.get("gamma_pb", 0.3)

        print(f"[DRIVER] Updated PCN for {path} - enable_ojas: {pcn.enable_ojas}, enable_stdp: {pcn.enable_stdp}")

    except (FileNotFoundError, pickle.UnpicklingError):
        print(f"[DRIVER] Initializing new PCN for {path}")

        num_bvc_per_dir = scale_def.get("num_bvc_per_dir", 50)
        bvc = BoundaryVectorCellLayer(
            max_dist=driver.max_dist,
            n_res=720,
            n_hd=driver.n_hd,
            sigma_theta=scale_def.get("sigma_theta"),
            sigma_r=scale_def.get("sigma_r"),
            num_bvc_per_dir=num_bvc_per_dir,
            device=driver.device,
        )
        print(f"[DRIVER] Created BVC layer with {num_bvc_per_dir} BVCs per direction (total: {num_bvc_per_dir * driver.n_hd} BVCs)")

        if num_grid_cells > 0:
            w_in_init_ratio = scale_def.get("w_in_init_ratio", 0.25)
            w_grid_init_ratio = scale_def.get("w_grid_init_ratio", 0.25)

            enable_correlation_weighting = scale_def.get("enable_correlation_weighting", True)
            correlation_window = scale_def.get("correlation_window", 100)
            correlation_update_freq = scale_def.get("correlation_update_freq", 10)
            correlation_scaling = scale_def.get("correlation_scaling", 2.0)
            min_correlation_weight = scale_def.get("min_correlation_weight", 0.1)
            correlation_threshold = scale_def.get("correlation_threshold", 0.01)

            alpha_pb = scale_def.get("alpha_pb", None)
            alpha_pg = scale_def.get("alpha_pg", None)

            pcn = MultiscalePlaceCellWithGrid(
                bvc_layer=bvc,
                num_pc=scale_def["num_pc"],
                num_grid_cells=num_grid_cells,
                timestep=driver.timestep,
                n_hd=driver.n_hd,
                enable_ojas=enable_ojas if enable_ojas is not None else False,
                enable_stdp=enable_stdp if enable_stdp is not None else False,
                w_in_init_ratio=w_in_init_ratio,
                w_grid_init_ratio=w_grid_init_ratio,
                w_grid_init_strategy=scale_def.get("w_grid_init_strategy", "balanced_modules"),
                gc_num_modules=scale_def.get("num_modules"),
                gc_cells_per_module=scale_def.get("cells_per_module"),
                grid_influence=scale_def.get("grid_influence", 0.5),
                gamma_pp=scale_def.get("gamma_pp", 0.5),
                gamma_pb=scale_def.get("gamma_pb", 0.25),
                gamma_pg=scale_def.get("gamma_pg", 0.3),
                alpha_pb=alpha_pb,
                alpha_pg=alpha_pg,
                enable_correlation_weighting=enable_correlation_weighting,
                correlation_window=correlation_window,
                correlation_update_freq=correlation_update_freq,
                correlation_scaling=correlation_scaling,
                min_correlation_weight=min_correlation_weight,
                correlation_threshold=correlation_threshold,
                adjacency_learning_mode=scale_def.get("adjacency_learning_mode", "dense"),
                adjacency_topk=scale_def.get("adjacency_topk"),
                adjacency_activity_floor=scale_def.get("adjacency_activity_floor", 0.0),
                device=driver.device,
            )
            alpha_pb_val = alpha_pb if alpha_pb is not None else np.sqrt(0.5)
            alpha_pg_val = alpha_pg if alpha_pg is not None else np.sqrt(0.5)
            print(
                f"[DRIVER] Created MultiscalePlaceCellWithGrid with {num_grid_cells} grid cells, "
                f"grid_influence={scale_def.get('grid_influence', 0.5)}, w_in_ratio={w_in_init_ratio}, "
                f"w_grid_ratio={w_grid_init_ratio}, alpha_pb={alpha_pb_val:.4f}, alpha_pg={alpha_pg_val:.4f}"
            )
        else:
            w_in_init_ratio = scale_def.get("w_in_init_ratio", 0.25)
            pcn = PlaceCellLayer(
                bvc_layer=bvc,
                num_pc=scale_def["num_pc"],
                timestep=driver.timestep,
                n_hd=driver.n_hd,
                enable_ojas=enable_ojas,
                enable_stdp=enable_stdp,
                w_in_init_ratio=w_in_init_ratio,
                gamma_pp=scale_def.get("gamma_pp", 0.5),
                gamma_pb=scale_def.get("gamma_pb", 0.3),
                device=driver.device,
            )
            print(f"[DRIVER] Created standard PlaceCellLayer without grid cells, w_in_ratio={w_in_init_ratio}")

    return pcn


def load_rcns(driver):
    driver.rcns = []
    for idx, scale_def in enumerate(driver.scales):
        scale_idx = scale_def["scale_index"]
        fname = f"rcn_scale_{scale_idx}.pkl"
        path = os.path.join(driver.network_dir, fname)
        learning_rate = driver.rcn_learning_rates[idx]
        rcn = load_or_init_rcn_for_scale(driver, path, scale_def, learning_rate)
        driver.rcns.append(rcn)


def load_or_init_rcn_for_scale(driver, path, scale_def, learning_rate):
    try:
        with open(path, "rb") as f:
            rcn = pickle.load(f)
        print(f"[DRIVER] Loaded existing RCN from {path}")
    except Exception:
        print(f"[DRIVER] Initializing new RCN for {path} with learning rate {learning_rate}")
        sigma_pc_s = scale_def.get("sigma_pc_s", scale_def.get("sigma_r"))
        rcn = RewardCellLayerTest(
            num_place_cells=scale_def["num_pc"],
            num_replay=3,
            learning_rate=learning_rate,
            replay_timesteps=scale_def.get("replay_timesteps", 20),
            replay_decay_factor=scale_def.get("replay_decay_factor", 6),
            custom_replay_timesteps=scale_def.get("custom_replay_timesteps", _steps_for_scale(scale_def)),
            initial_value_multiplier=scale_def.get("initial_value_multiplier", 1.0),
            sigma_pc_s=sigma_pc_s,
            device=driver.device,
        )
    return rcn


def load_goal_specific_rcns(driver, goal_name):
    """Load goal-specific RCNs for EXPLOIT_LOCATIONS_RANDOM mode."""
    multi_goal_dir = os.path.join(driver.network_dir, "multi_goal_rewards")

    if not os.path.exists(multi_goal_dir):
        print(f"[WARNING] Multi-goal rewards directory not found: {multi_goal_dir}")
        print("[WARNING] Make sure to run LEARN_LOCATIONS_COVERAGE first!")
        return

    print(f"[DRIVER] Loading goal-specific RCNs for goal: {goal_name}")
    driver.rcns = []

    for scale_def in driver.scales:
        scale_idx = scale_def["scale_index"]
        goal_rcn_path = os.path.join(multi_goal_dir, f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl")

        try:
            with open(goal_rcn_path, "rb") as f:
                rcn = pickle.load(f)
            print(f"[DRIVER] Loaded goal-specific RCN: {goal_rcn_path}")
            driver.rcns.append(rcn)
        except FileNotFoundError:
            print(f"[ERROR] Goal-specific RCN not found: {goal_rcn_path}")
            print("[ERROR] Make sure LEARN_LOCATIONS_COVERAGE has been run for this goal!")
            raise

    driver.rcn = driver.rcns[0] if driver.rcns else None
    print(f"[DRIVER] Loaded {len(driver.rcns)} goal-specific RCNs for '{goal_name}'")


def init_grid_cell_networks(driver):
    """Initialize grid cell networks for each scale based on scale parameters."""
    driver.gcns = []

    for scale_def in driver.scales:
        scale_idx = scale_def["scale_index"]
        fname = f"gcn_scale_{scale_idx}.pkl"
        path = os.path.join(driver.network_dir, fname)

        try:
            with open(path, "rb") as f:
                gcn = pickle.load(f)
            print(f"[DRIVER] Loaded existing GCN from {path}")
            if gcn is not None and gcn.__class__.__name__ == "GridCellLayer" and not isinstance(gcn, GridCellLayer):
                gcn.__class__ = GridCellLayer
                print("[DRIVER] Migrated loaded GCN instance to GridCellLayer v2 runtime class")
            if gcn is not None:
                if not hasattr(gcn, "activation_cache_size"):
                    gcn.activation_cache_size = 0
                if not hasattr(gcn, "activation_cache_quantization"):
                    gcn.activation_cache_quantization = None
                if not hasattr(gcn, "_activation_cache"):
                    from collections import OrderedDict
                    gcn._activation_cache = OrderedDict()
                if not hasattr(gcn, "freq_params"):
                    gcn.freq_params = (2.0 * np.pi) / gcn.size_params
                if not hasattr(gcn, "inv_spread_params"):
                    gcn.inv_spread_params = 1.0 / gcn.spread_params
                if not hasattr(gcn, "_sqrt3_over_2"):
                    gcn._sqrt3_over_2 = float(np.sqrt(3.0) / 2.0)
                if gcn.normalization == "per-cell":
                    gcn.min_max_updated = True
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[DRIVER] Initializing new GCN for scale {scale_idx}")
            num_grid_cells = scale_def.get("num_grid_cells", 0)
            num_modules = scale_def.get("num_modules", 8)
            cells_per_module = scale_def.get("cells_per_module", max(1, num_grid_cells // max(1, num_modules)))
            spread_range = scale_def.get("spread_range", (1.2, 1.2))
            scale_multiplier = scale_def.get("scale_multiplier", 1.0)
            translation_scale = scale_def.get("translation_scale", 1.0)
            mask_resolution = scale_def.get("mask_resolution", 128)
            smooth_sigma = scale_def.get("smooth_sigma", 1.5)
            activation_cache_size = scale_def.get("activation_cache_size", 0)
            activation_cache_quantization = scale_def.get("activation_cache_quantization")

            if num_grid_cells == 0:
                driver.gcns.append(None)
                continue

            gcn = GridCellLayer(
                num_modules=num_modules,
                cells_per_module=cells_per_module,
                spread_range=spread_range,
                scale_multiplier=scale_multiplier,
                translation_scale=translation_scale,
                threshold=0.7,
                threshold_type="soft",
                normalization="per-cell",
                world_name=driver.world_name,
                mask_resolution=mask_resolution,
                smooth_sigma=smooth_sigma,
                activation_cache_size=activation_cache_size,
                activation_cache_quantization=activation_cache_quantization,
                device=driver.device.type,
                dtype=driver.dtype,
            )

        driver.gcns.append(gcn)
