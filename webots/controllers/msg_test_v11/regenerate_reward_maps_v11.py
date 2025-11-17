"""
Regenerate Multi-Goal Reward Maps with Custom Parameters

This script allows you to regenerate reward maps for trained LEARN_LOCATIONS models
using different replay parameters without needing to retrain the entire network.

Simply edit the configuration parameters below and run:
    python regenerate_reward_maps.py

The script will:
1. Load the saved goal associations and trained networks
2. Apply the new replay parameters you specify below
3. Regenerate all reward maps
4. Save them to a new directory with timestamped suffix
"""

import sys
import os
import math
import pickle
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # From msg_test_v10 to project root
sys.path.append(str(PROJECT_ROOT))

from core.layers.reward_cell_layer_v11 import RewardCellLayerTest

# Add visualizations to path for plotting
VISUALIZATIONS_PATH = PROJECT_ROOT / "visualizations"
sys.path.append(str(VISUALIZATIONS_PATH))

##############################################################################
# EDITABLE CONFIGURATION - MODIFY THESE PARAMETERS
##############################################################################

# Path to your trained model (relative to msg_test_v10 folder)
# Options:
#   "pkl/20x20_maze_multi_goal"           # Standard mode
#   "pkl_GC_COR/e_puck_world_msg_1"       # AUTO mode trial
NETWORK_PATH = "pkl/20x20_maze_multi_goal"

# Scale-specific replay parameters
# Each scale can have different parameters for optimal reward map shape
# Configuration implements hierarchical multi-scale reward propagation using sigma_pc_s:
# - Small scale: Small sigma_pc_s → short lambda_s, high A_s → strongest near goal
# - Medium scale: Medium sigma_pc_s → medium lambda_s, medium A_s → strongest at medium distances
# - Large scale: Large sigma_pc_s → long lambda_s, low A_s → strongest far from goal
SCALE_SPECIFIC_CONFIG = {
    0: {  # small scale - precise, local navigation
        "sigma_pc_s": 1.0,                     # Place field size (drives lambda_s and A_s)
        "replay_timesteps": 0,
    },
    1: {  # medium scale - balanced
        "sigma_pc_s": 1.5,                     # Place field size (drives lambda_s and A_s)
        "replay_timesteps": 0,
    },
    2: {  # large scale - broad, global navigation
        "sigma_pc_s": 3.0,                     # Place field size (drives lambda_s and A_s)
        "replay_timesteps": 0,
    },
}

# Global fallback config (used if scale not in SCALE_SPECIFIC_CONFIG above)
GLOBAL_REPLAY_CONFIG = {
    "sigma_pc_s": 1.0,  # Default place field size
    "replay_timesteps": 20,
}

# Replay step budget is proportional to lambda_s to normalize spread per time constant
STEPS_PER_LAMBDA = 8  # Adjust to push farther (higher) or be more local (lower)

# Output directory suffix (auto-timestamped)
OUTPUT_SUFFIX = f"regenerated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Visualization settings
AUTO_VISUALIZE = True  # Set to False to skip automatic visualization
SHOW_PLOTS = False  # Set to True to display plots interactively (will block script)

##############################################################################
# Main Script - No need to modify below this line
##############################################################################

def load_networks_and_associations(network_dir):
    """Load PCNs, RCNs, and goal associations from saved trial."""

    # Load goal associations
    multi_goal_dir = os.path.join(network_dir, "multi_goal_rewards")
    associations_path = os.path.join(multi_goal_dir, "goal_associations.pkl")

    if not os.path.exists(associations_path):
        raise FileNotFoundError(
            f"Goal associations not found at {associations_path}\n"
            f"Make sure you've run LEARN_LOCATIONS_COVERAGE mode first!"
        )

    with open(associations_path, "rb") as f:
        association_data = pickle.load(f)

    print(f"[INFO] Loaded goal associations from {associations_path}")
    print(f"[INFO] Goals: {[g['name'] for g in association_data['goals']]}")
    print(f"[INFO] Scales: {[s['name'] for s in association_data['scales']]}")

    # Load PCNs and RCNs for each scale
    pcns = []
    rcns = []

    for scale in association_data['scales']:
        scale_idx = scale['scale_index']

        # Load PCN
        pcn_path = os.path.join(network_dir, f"pcn_scale_{scale_idx}.pkl")
        if not os.path.exists(pcn_path):
            raise FileNotFoundError(f"PCN not found: {pcn_path}")

        with open(pcn_path, "rb") as f:
            pcn = pickle.load(f)
        pcns.append(pcn)
        print(f"[INFO] Loaded PCN for scale {scale_idx}: {pcn_path}")

        # Load RCN
        rcn_path = os.path.join(network_dir, f"rcn_scale_{scale_idx}.pkl")
        if not os.path.exists(rcn_path):
            raise FileNotFoundError(f"RCN not found: {rcn_path}")

        with open(rcn_path, "rb") as f:
            rcn = pickle.load(f)
        rcns.append(rcn)
        print(f"[INFO] Loaded RCN for scale {scale_idx}: {rcn_path}")

    return association_data, pcns, rcns


def apply_replay_config_to_rcn(rcn, scale_idx):
    """Apply the configured replay parameters to an RCN."""

    # Use scale-specific config if available, otherwise use global
    if scale_idx in SCALE_SPECIFIC_CONFIG:
        config = SCALE_SPECIFIC_CONFIG[scale_idx]
        print(f"[INFO] Using scale-specific config for scale {scale_idx}")
    else:
        config = GLOBAL_REPLAY_CONFIG
        print(f"[INFO] Using global config for scale {scale_idx}")

    # Update replay parameters
    rcn.replay_timesteps = config.get("replay_timesteps", 20)
    rcn.replay_decay_factor = config.get("replay_decay_factor", 6)
    rcn.custom_replay_timesteps = config.get("custom_replay_timesteps", 40)

    return rcn


def regenerate_reward_maps(association_data, pcns, rcns, output_dir):
    """Regenerate reward maps with new parameters from scratch (zero weights)."""

    os.makedirs(output_dir, exist_ok=True)

    goals = association_data['goals']
    goal_associations = association_data['goal_place_cell_associations']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from core.layers.reward_cell_layer_v11 import C_LAMBDA, C_REWARD_PER_CELL

    print(f"\n[REGENERATE] Creating {len(goals)} goals × {len(pcns)} scales reward maps FROM SCRATCH")
    print(f"[REGENERATE] Output directory: {output_dir}")
    print(f"\n[REGENERATE] Scale-Specific Configuration:")
    for scale_idx, config in SCALE_SPECIFIC_CONFIG.items():
        sigma_pc_s = config.get('sigma_pc_s', 1.0)
        # Import constants from reward_cell_layer_v11 to show calculated values
        from core.layers.reward_cell_layer_v11 import C_LAMBDA, C_REWARD_PER_CELL

        # Get num_place_cells from the corresponding PCN (for info only)
        if scale_idx < len(pcns):
            num_pc = pcns[scale_idx].place_cell_activations.shape[0]
        else:
            num_pc = 500  # fallback

        lambda_s = C_LAMBDA * sigma_pc_s
        C_REWARD = C_REWARD_PER_CELL  # fixed per-scale budget
        A_s = C_REWARD / lambda_s
        custom_steps = config.get("custom_replay_timesteps", int(math.ceil(lambda_s * STEPS_PER_LAMBDA)))
        print(f"  Scale {scale_idx}: num_pc={num_pc}, sigma_pc_s={sigma_pc_s:.2f}, lambda_s={lambda_s:.2f}, "
              f"C_REWARD={C_REWARD:.1f}, A_s={A_s:.2f}, timesteps={custom_steps}")

    created_maps = 0

    for goal in goals:
        goal_name = goal['name']

        for scale_idx, (pcn, rcn) in enumerate(zip(pcns, rcns)):
            pc_idx = goal_associations[goal_name][scale_idx]

            if pc_idx is None:
                print(f"[WARNING] No place cell associated with {goal_name} for scale {scale_idx}")
                continue

            # Create artificial activation pattern
            artificial_activations = torch.zeros_like(pcn.place_cell_activations)
            artificial_activations[pc_idx] = 1.0

            print(f"\n[REGENERATE] Creating reward map for {goal_name} scale {scale_idx}:")
            print(f"  Using PC {pc_idx}")

            # Get configuration for this scale
            if scale_idx in SCALE_SPECIFIC_CONFIG:
                config = SCALE_SPECIFIC_CONFIG[scale_idx]
            else:
                config = GLOBAL_REPLAY_CONFIG

            # Create FRESH RCN with zero weights and new replay parameters
            num_place_cells = pcn.place_cell_activations.shape[0]
            sigma_pc_s = config.get("sigma_pc_s", 1.0)

            custom_steps = config.get("custom_replay_timesteps", int(math.ceil(C_LAMBDA * sigma_pc_s * STEPS_PER_LAMBDA)))

            goal_rcn = RewardCellLayerTest(
                num_place_cells=num_place_cells,
                num_replay=3,
                learning_rate=rcn.learning_rate,  # Keep original learning rate
                replay_timesteps=config.get("replay_timesteps", 20),
                custom_replay_timesteps=custom_steps,
                sigma_pc_s=sigma_pc_s,  # Use scale-dependent propagation
                device=device,
            )

            print(f"  Replay config: timesteps={goal_rcn.custom_replay_timesteps}, "
                  f"sigma_pc_s={sigma_pc_s:.2f}, lambda_s={goal_rcn.lambda_s:.2f}, A_s={goal_rcn.A_s:.4f}")
            print(f"  Starting with ZERO weights")

            # Initialize with goal activation (this adds initial weights)
            goal_rcn.update_reward_cell_activations(artificial_activations, visit=True)

            # Replay with custom activations to build reward map
            if hasattr(goal_rcn, 'replay_with_custom_activations'):
                goal_rcn.replay_with_custom_activations(
                    pcn=pcn,
                    custom_activations=artificial_activations
                )
                # Show final weight sum
                final_weight_sum = goal_rcn.w_in_effective.sum().item()
                print(f"  Final w_in_effective sum: {final_weight_sum:.6f}")
            else:
                print(f"[WARNING] RCN doesn't have replay_with_custom_activations, using standard replay")
                goal_rcn.replay(pcn=pcn)
                final_weight_sum = goal_rcn.w_in_effective.sum().item()
                print(f"  Final w_in_effective sum: {final_weight_sum:.6f}")

            # Save regenerated RCN
            rcn_filename = f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
            rcn_path = os.path.join(output_dir, rcn_filename)

            with open(rcn_path, "wb") as f:
                pickle.dump(goal_rcn, f)

            created_maps += 1
            print(f"  [SAVED] {rcn_filename}")

    print(f"\n[SUCCESS] Regenerated {created_maps} reward maps")
    print(f"[SUCCESS] Saved to: {output_dir}")

    # Save configuration used for this regeneration
    config_path = os.path.join(output_dir, "regeneration_config.txt")
    with open(config_path, "w") as f:
        f.write("Reward Map Regeneration Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network Path: {NETWORK_PATH}\n\n")

        if SCALE_SPECIFIC_CONFIG:
            f.write("Scale-Specific Configuration:\n")
            from core.layers.reward_cell_layer_v11 import C_LAMBDA, C_REWARD_PER_CELL
            for scale_idx, config in SCALE_SPECIFIC_CONFIG.items():
                f.write(f"  Scale {scale_idx}:\n")
                sigma_pc_s = config.get('sigma_pc_s', 1.0)

                # Get num_place_cells from the corresponding PCN if available
                if scale_idx < len(pcns):
                    num_pc = pcns[scale_idx].place_cell_activations.shape[0]
                else:
                    num_pc = 500  # fallback

                lambda_s = C_LAMBDA * sigma_pc_s
                C_REWARD = C_REWARD_PER_CELL  # fixed per-scale budget
                A_s = C_REWARD / lambda_s
                custom_steps = config.get("custom_replay_timesteps", int(math.ceil(lambda_s * STEPS_PER_LAMBDA)))
                f.write(f"    num_place_cells: {num_pc}\n")
                f.write(f"    sigma_pc_s: {sigma_pc_s}\n")
                f.write(f"    lambda_s (calculated): {lambda_s:.4f}\n")
                f.write(f"    C_REWARD (calculated): {C_REWARD:.1f}\n")
                f.write(f"    A_s (calculated): {A_s:.4f}\n")
                f.write(f"    custom_replay_timesteps: {custom_steps}\n")
                for key, value in config.items():
                    if key not in ('sigma_pc_s', 'custom_replay_timesteps'):
                        f.write(f"    {key}: {value}\n")

        f.write("\nGlobal Fallback Configuration:\n")
        for key, value in GLOBAL_REPLAY_CONFIG.items():
            f.write(f"  {key}: {value}\n")

    print(f"[INFO] Configuration saved to: {config_path}")

    return created_maps


def visualize_regenerated_maps(regenerated_dir, original_multi_goal_dir):
    """
    Generate visualizations for the regenerated reward maps.

    Args:
        regenerated_dir: Path to the directory containing regenerated RCN files
        original_multi_goal_dir: Path to original multi_goal_rewards directory (for goal_associations.pkl)
    """
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    try:
        # Check if matplotlib is available
        try:
            import matplotlib
            from matplotlib import pyplot as plt
            from matplotlib import gridspec, cm
        except ImportError as e:
            print(f"\n[WARNING] Visualization requires matplotlib: {e}")
            print(f"[INFO] Install matplotlib with: pip install matplotlib")
            print(f"[INFO] Skipping visualization - regenerated maps are still available")
            return None

        # Import visualization tools (import from vis_utils to avoid torch dependency in multi_goal_rcn_plot)
        import sys
        sys.path.insert(0, str(VISUALIZATIONS_PATH))

        # Import directly from vis_utils to avoid module-level torch imports
        import vis_utils
        load_multi_goal_hmaps = vis_utils.load_multi_goal_hmaps
        load_hmaps = vis_utils.load_hmaps
        convert_xzy_hmaps = vis_utils.convert_xzy_hmaps
        set_custom_paths = vis_utils.set_custom_paths

        # Set custom paths to point to our regenerated directory
        world_name = NETWORK_PATH.split('/')[-1]

        # The visualization expects data in: webots/controllers/CONTROLLER/pkl/WORLD/networks/multi_goal_rewards/
        # We need to temporarily point it to our regenerated directory

        print(f"[VIS] Loading regenerated maps from: {regenerated_dir}")
        print(f"[VIS] World: {world_name}")

        # Create visualization output directory
        vis_output_dir = os.path.join(regenerated_dir, "visualizations")
        os.makedirs(vis_output_dir, exist_ok=True)

        # Load shared location data (from original hmaps directory)
        set_custom_paths(controller_name="msg_test_v10", world_name=world_name)
        hmap_loc = load_hmaps(["hmap_loc"])
        if isinstance(hmap_loc, list):
            hmap_loc = hmap_loc[0]
        hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)

        # Get available combinations from regenerated directory
        import pickle

        # Load goal associations from original location
        associations_path = os.path.join(original_multi_goal_dir, "goal_associations.pkl")
        if not os.path.exists(associations_path):
            print(f"[WARNING] Could not find goal_associations.pkl at {associations_path}")
            return None

        with open(associations_path, "rb") as f:
            associations = pickle.load(f)

        goals_data = associations['goals']
        scales_data = associations['scales']

        goals = [g['name'] for g in goals_data]
        scales = [s['scale_index'] for s in scales_data]

        print(f"[VIS] Found {len(goals)} goals × {len(scales)} scales")

        # Generate main grid plot
        fig = plt.figure(figsize=(3.5 * len(scales), 3.5 * len(goals)))
        gs = gridspec.GridSpec(len(goals), len(scales),
                              hspace=0.3, wspace=0.1,
                              left=0.08, right=0.98,
                              bottom=0.08, top=0.92)

        successful_plots = 0

        for goal_idx, goal_name in enumerate(goals):
            for scale_idx, scale in enumerate(scales):
                try:
                    # Load regenerated RCN directly
                    rcn_filename = f"rcn_scale_{scale}_goal_{goal_name}.pkl"
                    rcn_path = os.path.join(regenerated_dir, rcn_filename)

                    if not os.path.exists(rcn_path):
                        print(f"[WARNING] RCN not found: {rcn_path}")
                        continue

                    with open(rcn_path, "rb") as f:
                        rcn = pickle.load(f)

                    # Load corresponding hmap_pcn from original location
                    _, hmap_pcn = load_multi_goal_hmaps(goal_name=goal_name, scale_idx=scale)

                    # Create subplot
                    ax = plt.subplot(gs[goal_idx, scale_idx])
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

                    # Add goal marker
                    goal_info = goals_data[goal_idx]
                    goal_x, goal_y = goal_info['location']
                    ax.scatter(goal_x, goal_y, c='white', s=200, marker='*',
                              edgecolors='black', linewidth=2, zorder=10)

                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_title(f"{goal_name.title()} - Scale {scale}")

                    successful_plots += 1

                except Exception as e:
                    print(f"[WARNING] Failed to plot {goal_name} scale {scale}: {e}")

        print(f"[VIS] Successfully plotted {successful_plots}/{len(goals) * len(scales)} combinations")

        # Save figure
        main_plot_path = os.path.join(vis_output_dir, "all_goals_all_scales.png")
        plt.savefig(main_plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        print(f"[SUCCESS] Visualization saved to: {main_plot_path}")
        return vis_output_dir

    except Exception as e:
        print(f"\n[WARNING] Visualization failed: {e}")
        print(f"You can manually run visualizations later using multi_goal_rcn_plot.py")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Get script directory
    script_dir = Path(__file__).resolve().parent

    # Build full path to network directory
    network_dir = script_dir / NETWORK_PATH / "networks"

    # Verify directory exists
    if not os.path.exists(network_dir):
        print(f"Error: Network directory not found: {network_dir}")
        print(f"\nMake sure NETWORK_PATH is set correctly at the top of this script.")
        print(f"Current NETWORK_PATH: {NETWORK_PATH}")
        print(f"\nHave you run LEARN_LOCATIONS_COVERAGE mode?")
        return

    print(f"[INFO] Loading from: {network_dir}")

    try:
        # Load data
        association_data, pcns, rcns = load_networks_and_associations(network_dir)

        # Create output directory with suffix
        output_base = network_dir / "multi_goal_rewards"
        output_dir = output_base / OUTPUT_SUFFIX

        # Regenerate maps
        num_created = regenerate_reward_maps(association_data, pcns, rcns, str(output_dir))

        print(f"\n{'='*60}")
        print(f"REGENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Created {num_created} reward maps")

        # Generate visualizations if enabled
        if AUTO_VISUALIZE:
            original_multi_goal_dir = output_base / "multi_goal_rewards"
            visualize_regenerated_maps(
                regenerated_dir=str(output_dir),
                original_multi_goal_dir=str(original_multi_goal_dir)
            )

        print(f"\nRegenerated maps saved to: {output_dir}")
        print(f"\nTo visualize: python visualize_regenerated.py")
        print(f"To use in simulation: swap this directory with multi_goal_rewards/")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
