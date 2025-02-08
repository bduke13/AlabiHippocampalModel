# %%
import sys
import types
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from matplotlib import cm, rcParams
import os


def plot_rcn_activation(context=0):
    """
    Visualizes the reward cell activations across the environment.

    Parameters:
      context (int): Context index for the reward cell weights.
    """
    print(os.listdir())
    # Load the RCN layer (reward cell network)
    rcn_path = os.path.join("rcn.pkl")
    with open(rcn_path, "rb") as f:
        rc_net = pickle.load(f)

    # Load the hmap_z
    hmap_pcn_path = os.path.join("hmap_pcn.pkl")
    with open(hmap_pcn_path, "rb") as f:
        hmap_z = pickle.load(f)

    # Load hmap_x and hmap_y locations
    hmap_loc_path = os.path.join("hmap_loc.pkl")
    with open(hmap_loc_path, "rb") as f:
        hmap_loc = pickle.load(f)

    # Use effective weights and cast to float32
    w_in_float32 = (
        torch.tensor(rc_net.w_in_effective, dtype=torch.float32).detach().cpu()
    )
    hmap_z_float32 = torch.tensor(hmap_z.T, dtype=torch.float32)

    # Normalize reward function by the sum of activations
    sum_activations = torch.sum(torch.tensor(hmap_z, dtype=torch.float32), dim=-1)
    safe_denom = torch.where(
        sum_activations > 0, sum_activations, torch.ones_like(sum_activations)
    )

    reward_function = (
        torch.tensordot(w_in_float32[context], hmap_z_float32, dims=1) / safe_denom
    )
    reward_function = torch.squeeze(reward_function)

    hmap_x = hmap_loc[:, 0]
    hmap_y = hmap_loc[:, 2]

    if reward_function.shape != hmap_x.shape:
        raise ValueError(
            f"Shape mismatch: reward_function has shape {reward_function.shape}, "
            f"but expected {hmap_x.shape}."
        )

    # Plotting the reward function
    fig, ax = plt.subplots(dpi=150)
    ax.set_aspect("equal")

    cntr = ax.hexbin(
        hmap_x,
        hmap_y,
        reward_function.detach().cpu().numpy(),
        gridsize=100,
        cmap=cmap,
        alpha=0.6,
    )

    # Adjust the view
    ax.set_ylim(5, -5)
    ax.set_xlim(5, -5)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Draw the goal location as a green circle
    goal = plt.Circle(goalLocation, goal_r, color="green", alpha=0.5, fill=True)
    ax.add_patch(goal)

    # Add a colorbar and title, then show the plot
    fig.colorbar(cntr)
    plt.title(f"Reward Map Visualization for context {context}")
    plt.show()


if __name__ == "__main__":
    # This should be the absolute path to controllers/bek_controller
    bek_controller_dir = os.path.join(os.getcwd(), "controllers", "create3_base")

    # Change into that directory
    os.chdir(bek_controller_dir)

    # Now we expect rcn.pkl to be in this directory
    print("Current directory is:", os.getcwd())
    print("Contents:", os.listdir())  # Check if 'rcn.pkl' is here

    rcn_path = "rcn.pkl"  # or just use the string directly
    with open(rcn_path, "rb") as f:
        rc_net = pickle.load(f)

    # Set up the colormap and default settings
    cmap = cm.get_cmap("plasma")
    rcParams.update({"font.size": 12})
    goal_r = 0.3  # Define goal radius
    goalLocation = [-1, 1]  # Define goal location (can be updated)

    plot_rcn_activation(context=0)
