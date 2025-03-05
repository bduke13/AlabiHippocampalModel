import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.widgets import TextBox, Button
import torch
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
fig = plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.35)  # Make room for text boxes and buttons

# Create subplots: PCA and average pattern
ax_pca = fig.add_subplot(121, projection="3d")
ax_overlay = fig.add_subplot(122)

# Create a dedicated axes for the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

# Define y-positions for text box rows (top-down)
row_ys = [0.32 - i * 0.04 for i in range(8)]

# Create text box axes and widgets
# Single-value parameters
num_cells_ax = plt.axes([0.2, row_ys[0], 0.3, 0.03])
num_cells_text = TextBox(num_cells_ax, "Number of Cells", initial="2000")

axis_size_ax = plt.axes([0.2, row_ys[1], 0.3, 0.03])
axis_size_text = TextBox(axis_size_ax, "Axis Size", initial="3")

density_ax = plt.axes([0.2, row_ys[2], 0.3, 0.03])
density_text = TextBox(density_ax, "Density", initial="200")

# Range parameters (min and max)
size_min_ax = plt.axes([0.2, row_ys[3], 0.15, 0.03])
size_min_text = TextBox(size_min_ax, "Size Min", initial="0.5")
size_max_ax = plt.axes([0.45, row_ys[3], 0.15, 0.03])
size_max_text = TextBox(size_max_ax, "Size Max", initial="2.0")

rotation_min_ax = plt.axes([0.2, row_ys[4], 0.15, 0.03])
rotation_min_text = TextBox(rotation_min_ax, "Rotation Min", initial="0")
rotation_max_ax = plt.axes([0.45, row_ys[4], 0.15, 0.03])
rotation_max_text = TextBox(rotation_max_ax, "Rotation Max", initial="25")

spread_min_ax = plt.axes([0.2, row_ys[5], 0.15, 0.03])
spread_min_text = TextBox(spread_min_ax, "Spread Min", initial="0.5")
spread_max_ax = plt.axes([0.45, row_ys[5], 0.15, 0.03])
spread_max_text = TextBox(spread_max_ax, "Spread Max", initial="1.0")

x_trans_min_ax = plt.axes([0.2, row_ys[6], 0.15, 0.03])
x_trans_min_text = TextBox(x_trans_min_ax, "X Trans Min", initial="-2.0")
x_trans_max_ax = plt.axes([0.45, row_ys[6], 0.15, 0.03])
x_trans_max_text = TextBox(x_trans_max_ax, "X Trans Max", initial="2.0")

y_trans_min_ax = plt.axes([0.2, row_ys[7], 0.15, 0.03])
y_trans_min_text = TextBox(y_trans_min_ax, "Y Trans Min", initial="-2.0")
y_trans_max_ax = plt.axes([0.45, row_ys[7], 0.15, 0.03])
y_trans_max_text = TextBox(y_trans_max_ax, "Y Trans Max", initial="2.0")

# Create buttons
reset_ax = plt.axes([0.7, 0.25, 0.1, 0.04])
run_ax = plt.axes([0.7, 0.30, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset", color="lightgoldenrodyellow")
run_button = Button(run_ax, "Run", color="lightblue")

# Parameters dictionary (will be updated from text boxes)
params = {}


def compute_patterns(params, use_gpu=True):
    np.random.seed(42)
    torch.manual_seed(42)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create the spatial grid
    x = np.linspace(-params["axis_size"], params["axis_size"], params["density"])
    y = np.linspace(-params["axis_size"], params["axis_size"], params["density"])
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

    # Read values from text boxes with error handling
    try:
        num_cells = int(num_cells_text.text)
        axis_size = float(axis_size_text.text) * np.pi  # Multiplier for pi
        density = int(density_text.text)
        size_min = float(size_min_text.text)
        size_max = float(size_max_text.text)
        rotation_min = float(rotation_min_text.text)
        rotation_max = float(rotation_max_text.text)
        spread_min = float(spread_min_text.text)
        spread_max = float(spread_max_text.text)
        x_trans_min = float(x_trans_min_text.text)
        x_trans_max = float(x_trans_max_text.text)
        y_trans_min = float(y_trans_min_text.text)
        y_trans_max = float(y_trans_max_text.text)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Update params dictionary
    params["num_cells"] = num_cells
    params["axis_size"] = axis_size  # Already multiplied by pi
    params["density"] = density
    params["size_range"] = (size_min, size_max)
    params["rotation_range"] = (rotation_min, rotation_max)
    params["spread_range"] = (spread_min, spread_max)
    params["x_trans_range"] = (x_trans_min, x_trans_max)
    params["y_trans_range"] = (y_trans_min, y_trans_max)

    # Compute patterns
    data_matrix = compute_patterns(params, use_gpu=True)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_matrix)

    # Reshape data matrix for averaging
    grid_size = int(np.sqrt(data_matrix.shape[1]))
    patterns_2d = data_matrix.reshape(params["num_cells"], grid_size, grid_size)

    # Create overlay by averaging patterns
    overlay = np.mean(patterns_2d, axis=0)

    # Update PCA plot
    ax_pca.clear()
    ax_pca.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        c=np.linspace(0, 1, params["num_cells"]),
        cmap="viridis",
        s=50,
    )
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    ax_pca.set_zlabel("PC 3")
    ax_pca.set_title("3D PCA Reduction")
    ax_pca.view_init(elev=20, azim=45)

    # Update overlay plot
    ax_overlay.clear()
    im = ax_overlay.imshow(
        overlay,
        cmap="viridis",
        extent=[
            -params["axis_size"],
            params["axis_size"],
            -params["axis_size"],
            params["axis_size"],
        ],
    )
    ax_overlay.set_title("Average Grid Cell Pattern")
    ax_overlay.set_xlabel("X")
    ax_overlay.set_ylabel("Y")

    # Update colorbar in the dedicated axes
    fig.colorbar(im, cax=cbar_ax)

    # Display computation time
    fig.suptitle(f"Computation time: {time.time() - start_time:.2f}s", y=0.95)

    fig.canvas.draw_idle()


def reset(event):
    # Reset text boxes to initial values
    num_cells_text.set_val("2000")
    axis_size_text.set_val("3")
    density_text.set_val("200")
    size_min_text.set_val("0.5")
    size_max_text.set_val("2.0")
    rotation_min_text.set_val("0")
    rotation_max_text.set_val("25")
    spread_min_text.set_val("0.5")
    spread_max_text.set_val("1.0")
    x_trans_min_text.set_val("-2.0")
    x_trans_max_text.set_val("2.0")
    y_trans_min_text.set_val("-2.0")
    y_trans_max_text.set_val("2.0")
    update()


# Register button callbacks
reset_button.on_clicked(reset)
run_button.on_clicked(update)

# Initial plot
update()

plt.show()