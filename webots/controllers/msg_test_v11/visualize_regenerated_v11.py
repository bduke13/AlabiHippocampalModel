"""
Standalone script to visualize regenerated reward maps.

Edit the configuration below to specify which regenerated directory to visualize.
"""

import sys
import os
import pickle
import torch
from pathlib import Path

##############################################################################
# CONFIGURATION - Edit this to match your regenerated directory
##############################################################################

# Name of the regenerated directory to visualize
# Leave as None to auto-detect the most recent one
REGENERATED_DIR_NAME = None  # e.g., "regenerated_20251111_214412"

# Path to network (should match what was used in regenerate_reward_maps.py)
NETWORK_PATH = "pkl/20x20_maze_multi_goal"

##############################################################################

# Add project root and visualizations to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "visualizations"))

# Import visualization utilities
import vis_utils

# Import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from matplotlib import pyplot as plt
    from matplotlib import gridspec, cm
except ImportError as e:
    print(f"Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def visualize_regenerated_directory(regenerated_dirname, network_path="pkl/20x20_maze_multi_goal"):
    """
    Visualize reward maps from a regenerated directory.

    Args:
        regenerated_dirname: Name of the regenerated directory (e.g., "regenerated_20251111_214412")
        network_path: Path to the network directory (relative to msg_test_v10)
    """
    script_dir = Path(__file__).resolve().parent

    # Build paths
    network_base = script_dir / network_path / "networks"
    regenerated_dir = network_base / "multi_goal_rewards" / regenerated_dirname
    original_multi_goal_dir = network_base / "multi_goal_rewards"

    if not regenerated_dir.exists():
        print(f"Error: Regenerated directory not found: {regenerated_dir}")
        return

    print(f"[VIS] Regenerated directory: {regenerated_dir}")
    print(f"[VIS] Original multi_goal_rewards: {original_multi_goal_dir}")

    # Extract world name from network_path
    world_name = network_path.split('/')[-1]

    # Set paths for vis_utils to load hmaps from correct location
    vis_utils.set_custom_paths(controller_name="msg_test_v11", world_name=world_name)

    # Load goal associations from original location
    associations_path = original_multi_goal_dir / "goal_associations.pkl"
    if not associations_path.exists():
        print(f"Error: goal_associations.pkl not found at {associations_path}")
        print(f"Make sure you've run LEARN_LOCATIONS_COVERAGE mode first!")
        return

    with open(associations_path, "rb") as f:
        associations = pickle.load(f)

    goals_data = associations['goals']
    scales_data = associations['scales']

    # Extract goal names and sort in standard order (red, green, blue, yellow)
    # This matches the order used in multi_goal_rcn_plot.py
    STANDARD_GOAL_ORDER = ["blue", "green", "red", "yellow"]
    available_goals = [g['name'] for g in goals_data]

    # Sort goals according to standard order, keeping any non-standard goals at the end
    goals = sorted(available_goals, key=lambda x: STANDARD_GOAL_ORDER.index(x) if x in STANDARD_GOAL_ORDER else len(STANDARD_GOAL_ORDER))
    scales = sorted([s['scale_index'] for s in scales_data])

    print(f"[VIS] Found {len(goals)} goals × {len(scales)} scales")
    print(f"[VIS] Goals: {goals}")
    print(f"[VIS] Scales: {scales}")

    # Load shared location data
    print(f"[VIS] Loading hmap_loc...")
    hmap_loc = vis_utils.load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    hmap_x, _, hmap_y = vis_utils.convert_xzy_hmaps(hmap_loc)
    print(f"[VIS] Loaded hmap with {len(hmap_x)} points")

    # Create visualization output directory
    vis_output_dir = regenerated_dir / "visualizations"
    vis_output_dir.mkdir(exist_ok=True)

    # Generate main grid plot
    print(f"\n[VIS] Generating grid plot...")
    fig = plt.figure(figsize=(3.5 * len(scales), 3.5 * len(goals)))
    gs = gridspec.GridSpec(len(goals), len(scales),
                          hspace=0.3, wspace=0.1,
                          left=0.08, right=0.98,
                          bottom=0.08, top=0.92)

    successful_plots = 0

    for goal_idx, goal_name in enumerate(goals):
        for scale_idx_pos, scale in enumerate(scales):
            try:
                # Load regenerated RCN directly from regenerated directory
                rcn_filename = f"rcn_scale_{scale}_goal_{goal_name}.pkl"
                rcn_path = regenerated_dir / rcn_filename

                if not rcn_path.exists():
                    print(f"[WARNING] RCN not found: {rcn_path}")
                    continue

                print(f"[VIS] Loading {rcn_filename}...")
                with open(rcn_path, "rb") as f:
                    rcn = pickle.load(f)

                # Load corresponding hmap_pcn from original hmaps location
                print(f"[VIS] Loading hmap_pcn_scale_{scale}...")
                _, hmap_pcn = vis_utils.load_multi_goal_hmaps(goal_name=goal_name, scale_idx=scale)
                print(f"[VIS] Loaded hmap_pcn with shape {hmap_pcn.shape}")

                # Create subplot
                ax = plt.subplot(gs[goal_idx, scale_idx_pos])
                ax.set_aspect('equal')

                # Compute reward function
                w_in_float32 = torch.tensor(rcn.w_in_effective, dtype=torch.float32).clone().detach().cpu()
                hmap_pcn_float32 = torch.tensor(hmap_pcn.T, dtype=torch.float32)

                sum_activations = torch.sum(hmap_pcn_float32, dim=0)
                safe_denom = torch.where(sum_activations > 0, sum_activations, torch.ones_like(sum_activations))
                reward_function = torch.tensordot(w_in_float32, hmap_pcn_float32, dims=1) / safe_denom
                reward_function = torch.squeeze(reward_function)

                # Plot
                cmap = cm.get_cmap("plasma")
                ax.hexbin(hmap_x, hmap_y, reward_function.detach().cpu().numpy(),
                         gridsize=100, cmap=cmap, alpha=0.7)

                # Add goal marker - look up by name since order may have changed
                goal_info = next((g for g in goals_data if g['name'] == goal_name), None)
                if goal_info:
                    goal_x, goal_y = goal_info['location']
                    ax.scatter(goal_x, goal_y, c='white', s=200, marker='*',
                              edgecolors='black', linewidth=2, zorder=10)

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"{goal_name.title()} - Scale {scale}")

                successful_plots += 1
                print(f"[VIS] ✓ Plotted {goal_name} scale {scale}")

            except Exception as e:
                print(f"[ERROR] Failed to plot {goal_name} scale {scale}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n[VIS] Successfully plotted {successful_plots}/{len(goals) * len(scales)} combinations")

    # Save figure
    main_plot_path = vis_output_dir / "all_goals_all_scales.png"
    print(f"[VIS] Saving to {main_plot_path}...")
    plt.savefig(str(main_plot_path), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Visualization saved!")
    print(f"{'='*60}")
    print(f"Location: {main_plot_path}")
    print(f"Open this file to view your regenerated reward maps.")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    multi_goal_dir = script_dir / NETWORK_PATH.replace('/', os.sep) / "networks" / "multi_goal_rewards"

    # Auto-detect or use specified directory
    if REGENERATED_DIR_NAME is None:
        # Find the most recent regenerated directory
        if not multi_goal_dir.exists():
            print(f"Error: multi_goal_rewards directory not found at {multi_goal_dir}")
            print("Run regenerate_reward_maps.py first!")
            sys.exit(1)

        regenerated_dirs = sorted([
            d for d in multi_goal_dir.iterdir()
            if d.is_dir() and d.name.startswith("regenerated_")
        ], key=lambda x: x.name, reverse=True)

        if not regenerated_dirs:
            print("No regenerated directories found.")
            print("Run regenerate_reward_maps.py first!")
            sys.exit(1)

        regenerated_dirname = regenerated_dirs[0].name
        print(f"[INFO] Auto-detected most recent: {regenerated_dirname}")

    else:
        regenerated_dirname = REGENERATED_DIR_NAME
        print(f"[INFO] Using specified directory: {regenerated_dirname}")

    # Run visualization
    visualize_regenerated_directory(regenerated_dirname, network_path=NETWORK_PATH)
