"""
Check if place cells learned properly through Oja's rule.
"""

import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROLLER_NAME = "multiscale_controller"

def check_pcn_learning(env_name, scale):
    """Check if place cell network learned properly."""

    # Load the network
    network_file = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl" / env_name / "networks" / f"pcn_scale_{scale}.pkl"

    if not network_file.exists():
        print(f"  Network file not found: {network_file}")
        return False

    with open(network_file, "rb") as f:
        pcn = pickle.load(f)

    print(f"\n{env_name} - Scale {scale}:")
    print(f"  Network type: {type(pcn)}")

    # Check if it has w_in (input weights from BVC to place cells)
    if hasattr(pcn, 'w_in'):
        import torch
        weights = pcn.w_in
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        print(f"  Input weights (w_in) shape: {weights.shape}")
        print(f"  Weight statistics:")
        print(f"    Min: {np.min(weights):.6f}")
        print(f"    Max: {np.max(weights):.6f}")
        print(f"    Mean: {np.mean(weights):.6f}")
        print(f"    Std: {np.std(weights):.6f}")

        # Check how many cells have non-zero weights
        cell_norms = np.linalg.norm(weights, axis=0)
        nonzero_cells = np.sum(cell_norms > 1e-6)
        print(f"  Cells with learned weights: {nonzero_cells}/{len(cell_norms)}")

        # Check if weights are all the same (no learning)
        if np.std(weights) < 1e-10:
            print("  WARNING: All weights are nearly identical - no learning occurred!")
            return False

        # Show top 5 cells by weight norm
        top_indices = np.argsort(cell_norms)[-5:][::-1]
        print(f"  Top 5 cells by weight norm:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. Cell {idx}: norm = {cell_norms[idx]:.6f}")

        # Check weight diversity (should be diverse if learning worked)
        weight_std_per_cell = np.std(weights, axis=0)
        print(f"  Weight diversity per cell (std):")
        print(f"    Min: {np.min(weight_std_per_cell):.6f}")
        print(f"    Max: {np.max(weight_std_per_cell):.6f}")
        print(f"    Mean: {np.mean(weight_std_per_cell):.6f}")

        return True
    else:
        print(f"  ERROR: Network has no 'w_in' attribute")
        print(f"  Available attributes: {dir(pcn)}")
        return False


def check_hmap_data(env_name, scale):
    """Check the heatmap data."""

    hmap_pcn_file = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl" / env_name / "hmaps" / f"hmap_pcn_scale_{scale}.pkl"
    hmap_loc_file = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl" / env_name / "hmaps" / "hmap_loc.pkl"

    if not hmap_pcn_file.exists():
        print(f"  Heatmap file not found: {hmap_pcn_file}")
        return False

    with open(hmap_pcn_file, "rb") as f:
        hmap_pcn = np.array(pickle.load(f))
        hmap_pcn = hmap_pcn[1:]  # Remove first element

    with open(hmap_loc_file, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        hmap_loc = hmap_loc[1:]  # Remove first element

    print(f"\n{env_name} - Scale {scale} Heatmap:")
    print(f"  Shape: {hmap_pcn.shape} (timesteps, num_cells)")
    print(f"  Data statistics:")
    print(f"    Min: {np.min(hmap_pcn):.6f}")
    print(f"    Max: {np.max(hmap_pcn):.6f}")
    print(f"    Mean: {np.mean(hmap_pcn):.6f}")
    print(f"    Std: {np.std(hmap_pcn):.6f}")

    # Check activation per cell
    total_activation = np.sum(hmap_pcn, axis=0)
    active_cells = np.sum(total_activation > 0)
    print(f"  Active cells: {active_cells}/{hmap_pcn.shape[1]}")

    # Check if data is all zeros
    if np.max(hmap_pcn) < 1e-10:
        print("  WARNING: All activations are nearly zero!")
        return False

    # Find a cell with high activation and check its spatial pattern
    top_cells = np.argsort(total_activation)[-5:][::-1]
    print(f"\n  Top 5 most active cells:")
    for i, cell_idx in enumerate(top_cells):
        total_act = total_activation[cell_idx]
        max_act = np.max(hmap_pcn[:, cell_idx])
        num_active_steps = np.sum(hmap_pcn[:, cell_idx] > 0)
        print(f"    {i+1}. Cell {cell_idx}: total={total_act:.3f}, max={max_act:.3f}, active_steps={num_active_steps}")

        # Check spatial localization
        if num_active_steps > 0:
            active_indices = np.where(hmap_pcn[:, cell_idx] > 0.01)[0]
            if len(active_indices) > 0:
                active_locs = hmap_loc[active_indices]
                x_range = np.max(active_locs[:, 0]) - np.min(active_locs[:, 0])
                z_range = np.max(active_locs[:, 2]) - np.min(active_locs[:, 2])
                print(f"       Spatial extent: x={x_range:.2f}m, z={z_range:.2f}m")

    return True


def main():
    print("="*70)
    print("Checking Place Cell Learning")
    print("="*70)

    # Check one environment in detail
    env_name = "environment_1"

    print("\n" + "="*70)
    print("NETWORK WEIGHTS (from LEARN_OJAS)")
    print("="*70)

    for scale in [0, 1, 2]:
        check_pcn_learning(env_name, scale)

    print("\n" + "="*70)
    print("HEATMAP DATA (from PLOTTING)")
    print("="*70)

    for scale in [0, 1, 2]:
        check_hmap_data(env_name, scale)


if __name__ == "__main__":
    main()
