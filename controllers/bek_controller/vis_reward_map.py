# %%
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm, rcParams

# Set up the colormap and default settings
cmap = cm.get_cmap("plasma")
rcParams.update({"font.size": 12})
goal_r = 0.3  # Define goal radius
goalLocation = [-1, 1]  # Define goal location (can be updated)


def plot_rcn_activation(context=0):
    """
    Visualizes the reward cell activations across the environment.

    Parameters:
    context (int): Context index for the reward cell weights.
    """
    # Load the RCN layer (reward cell network)
    with open("rcn.pkl", "rb") as f:
        rc_net = pickle.load(f)

    # Use the effective weights instead of w_in
    w_in_float32 = tf.cast(
        rc_net.w_in_effective, tf.float32
    )  # Cast rc_net effective weights to float32

    # Ensure that hmap_z is loaded and cast to float32
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = pickle.load(f)
    hmap_z_float32 = tf.cast(
        hmap_z.T, tf.float32
    )  # Transpose and cast hmap_z to float32

    # Compute the total place cell activations to use in normalizing the reward activations
    sum_activations = tf.reduce_sum(hmap_z, axis=-1)
    safe_denominator = tf.where(
        sum_activations > 0, sum_activations, 1
    )  # Avoid division by zero
    safe_denominator = tf.cast(safe_denominator, tf.float32)

    # Compute the reward function using the effective weights and the place cell activations
    reward_function = (
        tf.tensordot(w_in_float32[context], hmap_z_float32, axes=1) / safe_denominator
    )

    # Flatten reward function to ensure it's a 1D array
    reward_function = tf.squeeze(reward_function)

    # Load hmap_x and hmap_y
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = pickle.load(f)
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = pickle.load(f)

    # Check if reward_function has the same shape as hmap_x and hmap_y
    if reward_function.shape != hmap_x.shape:
        raise ValueError(
            f"Shape mismatch: reward_function has shape {reward_function.shape}, but expected {hmap_x.shape}."
        )

    # Plot the reward function over the environment
    fig, ax = plt.subplots(dpi=150)
    ax.set_aspect("equal")

    # Create a hexbin plot for the reward activations
    cntr = ax.hexbin(
        hmap_x, hmap_y, reward_function.numpy(), gridsize=100, cmap=cmap, alpha=0.6
    )

    # Rotate the plot by 180 degrees by reversing the x and y limits
    ax.set_ylim(5, -5)
    ax.set_xlim(5, -5)
    ax.set_axis_off()

    # Increase the size of the map relative to the overall plot
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Draw the goal location as a green circle
    goal = plt.Circle(goalLocation, goal_r, color="green", alpha=0.5, fill=True)
    ax.add_patch(goal)

    # Add a colorbar for the reward values
    fig.colorbar(cntr)
    plt.title(f"Reward Map Visualization for context {context}")
    plt.show()


# Call the function with a specific context
plot_rcn_activation(context=0)
