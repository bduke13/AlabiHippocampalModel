# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button
import torch
from torch.utils.data import DataLoader
import time


def grid_cell_pattern(
    x, y, size=1.0, rotation=0.0, spread=1.0, x_trans=0.0, y_trans=0.0, use_gpu=True
):
    """
    Generate a grid-cell firing field using cosine combinations with PyTorch.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Convert inputs to PyTorch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

    # Apply translation
    x = x - x_trans
    y = y - y_trans

    # Rotate coordinates
    theta = torch.tensor(np.radians(rotation), device=device)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t

    # Create the grid pattern
    z = (
        torch.cos(size * x_rot)
        + torch.cos(0.5 * size * x_rot + size * y_rot)
        + torch.cos(0.5 * size * x_rot - size * y_rot)
    )

    # Normalize and apply spread transformation
    z = z / 3.0
    z = torch.sign(z) * torch.pow(torch.abs(z), 1 / spread)

    return z


# Create figure and adjust subplots
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.35)  # Make room for sliders

from matplotlib.widgets import RangeSlider

# Create slider axes
slider_color = "lightgoldenrodyellow"
num_cells_ax = plt.axes([0.2, 0.28, 0.5, 0.02])
axis_size_ax = plt.axes([0.2, 0.25, 0.5, 0.02])
density_ax = plt.axes([0.2, 0.22, 0.5, 0.02])
size_range_ax = plt.axes([0.2, 0.19, 0.5, 0.02])
rotation_range_ax = plt.axes([0.2, 0.16, 0.5, 0.02])
spread_range_ax = plt.axes([0.2, 0.13, 0.5, 0.02])
x_trans_range_ax = plt.axes([0.2, 0.10, 0.5, 0.02])
y_trans_range_ax = plt.axes([0.2, 0.07, 0.5, 0.02])

# Create sliders
num_cells_slider = Slider(
    num_cells_ax, "Number of Cells", 100, 5000, valinit=2000, valstep=100
)
axis_size_slider = Slider(axis_size_ax, "Axis Size", 1, 100, valinit=3)
density_slider = Slider(density_ax, "Density", 50, 400, valinit=200, valstep=50)

# Create range sliders for min/max values
size_range_slider = RangeSlider(
    size_range_ax, "Size Range", 0.1, 5.0, valinit=(0.5, 2.0)
)
rotation_range_slider = RangeSlider(
    rotation_range_ax, "Rotation Range", 0, 90, valinit=(0, 25)
)
spread_range_slider = RangeSlider(
    spread_range_ax, "Spread Range", 0.1, 2.0, valinit=(0.5, 1.0)
)
x_trans_range_slider = RangeSlider(
    x_trans_range_ax, "X Translation Range", -5.0, 5.0, valinit=(-2.0, 2.0)
)
y_trans_range_slider = RangeSlider(
    y_trans_range_ax, "Y Translation Range", -5.0, 5.0, valinit=(-2.0, 2.0)
)

# Create buttons
button_color = "lightblue"
reset_ax = plt.axes([0.8, 0.15, 0.1, 0.04])
run_ax = plt.axes([0.8, 0.21, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset", color=slider_color)
run_button = Button(run_ax, "Run", color=button_color)

# Initialize parameters
params = {
    "num_cells": 2000,
    "axis_size": 3 * np.pi,
    "density": 200,
    "size_range": (0.5, 2),
    "rotation_range": (0, 25),
    "spread_range": (0.5, 1.0),
    "x_trans_range": (-2.0, 2.0),
    "y_trans_range": (-2.0, 2.0),
}


def compute_patterns(params, use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create the spatial grid
    x = np.linspace(
        -params["axis_size"] * np.pi, params["axis_size"] * np.pi, params["density"]
    )
    y = np.linspace(
        -params["axis_size"] * np.pi, params["axis_size"] * np.pi, params["density"]
    )
    X, Y = np.meshgrid(x, y)

    # Convert to PyTorch tensors and move to GPU if available
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    patterns = []
    batch_size = 100  # Process patterns in batches

    for i in range(0, params["num_cells"], batch_size):
        batch_patterns = []
        current_batch_size = min(batch_size, params["num_cells"] - i)

        # Generate parameters for the batch
        size_params = (
            torch.FloatTensor(current_batch_size)
            .uniform_(*params["size_range"])
            .to(device)
        )
        rotation_params = (
            torch.FloatTensor(current_batch_size)
            .uniform_(*params["rotation_range"])
            .to(device)
        )
        spread_params = (
            torch.FloatTensor(current_batch_size)
            .uniform_(*params["spread_range"])
            .to(device)
        )
        x_trans_params = (
            torch.FloatTensor(current_batch_size)
            .uniform_(*params["x_trans_range"])
            .to(device)
        )
        y_trans_params = (
            torch.FloatTensor(current_batch_size)
            .uniform_(*params["y_trans_range"])
            .to(device)
        )

        # Generate patterns for the batch
        for j in range(current_batch_size):
            Z = grid_cell_pattern(
                X,
                Y,
                size=size_params[j].item(),
                rotation=rotation_params[j].item(),
                spread=spread_params[j].item(),
                x_trans=x_trans_params[j].item(),
                y_trans=y_trans_params[j].item(),
                use_gpu=use_gpu,
            )
            batch_patterns.append(Z.flatten())

        # Stack and move to CPU
        batch_tensor = torch.stack(batch_patterns).cpu().numpy()
        patterns.append(batch_tensor)

    # Combine all batches
    data_matrix = np.concatenate(patterns, axis=0)
    return data_matrix


def update(val=None):
    start_time = time.time()

    # Update parameters from sliders
    params["num_cells"] = int(num_cells_slider.val)
    params["axis_size"] = axis_size_slider.val
    params["density"] = int(density_slider.val)
    params["size_range"] = tuple(size_range_slider.val)
    params["rotation_range"] = tuple(rotation_range_slider.val)
    params["spread_range"] = tuple(spread_range_slider.val)
    params["x_trans_range"] = tuple(x_trans_range_slider.val)
    params["y_trans_range"] = tuple(y_trans_range_slider.val)

    # Compute patterns using GPU if available
    data_matrix = compute_patterns(params, use_gpu=True)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_matrix)

    # Update plot
    ax.clear()
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        c=np.linspace(0, 1, params["num_cells"]),
        cmap="viridis",
        s=100,
    )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(
        f"3D PCA Reduction of Grid Cell Patterns\nComputation time: {time.time() - start_time:.2f}s"
    )
    ax.view_init(elev=20, azim=45)

    fig.canvas.draw_idle()


def reset(event):
    num_cells_slider.reset()
    axis_size_slider.reset()
    density_slider.reset()
    size_range_slider.reset()
    rotation_range_slider.reset()
    spread_range_slider.reset()
    x_trans_range_slider.reset()
    y_trans_range_slider.reset()
    update()


def on_run_clicked(event):
    update()


# Register callbacks
reset_button.on_clicked(reset)
run_button.on_clicked(on_run_clicked)

# Initial plot
update()

plt.show()
