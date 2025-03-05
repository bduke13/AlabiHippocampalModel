# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm, rcParams
from typing import List


def plot_rcn_activation(
    rcn,
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    goal_r: float,
    goal_location: List[float],
    context: int = 0,
):
    """
    Visualizes the reward cell activations across the environment.

    Parameters:
      context (int): Context index for the reward cell weights.
    """

    # Use effective weights and cast to float32
    w_in_float32 = torch.tensor(rcn.w_in_effective, dtype=torch.float32).detach().cpu()
    hmap_pcn_float32 = torch.tensor(hmap_pcn.T, dtype=torch.float32)

    # Normalize reward function by the sum of activations
    sum_activations = torch.sum(torch.tensor(hmap_pcn, dtype=torch.float32), dim=-1)
    safe_denom = torch.where(
        sum_activations > 0, sum_activations, torch.ones_like(sum_activations)
    )

    reward_function = (
        torch.tensordot(w_in_float32[context], hmap_pcn_float32, dims=1) / safe_denom
    )
    reward_function = torch.squeeze(reward_function)

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
    goal = plt.Circle(goal_location, goal_r, color="green", alpha=0.5, fill=True)
    ax.add_patch(goal)

    # Add a colorbar and title, then show the plot
    fig.colorbar(cntr)
    plt.title(f"Reward Map Visualization for context {context}")
    plt.show()


if __name__ == "__main__":
    from vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
        load_layer_pkl,
    )

    # Load hmap data
    hmap_loc, hmap_pcn = load_hmaps()
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    rcn = load_layer_pkl(layer_name="rcn")

    # Set up the colormap and default settings
    cmap = cm.get_cmap("plasma")
    rcParams.update({"font.size": 12})
    goal_r = 0.3  # Define goal radius
    goal_location = [-3.0, 3.0]  # Define goal location (can be updated)

    plot_rcn_activation(
        rcn=rcn,
        hmap_pcn=hmap_pcn,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        goal_r=goal_r,
        goal_location=goal_location,
    )
