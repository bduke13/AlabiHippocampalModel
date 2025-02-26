# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, CheckButtons


def f(x, y, size=1.0, rotation=0.0, spread=1.0, x_trans=0.0, y_trans=0.0):
    """
    f(x, y) = cos(size*x_rot)
              + cos(0.5*size*x_rot + size*y_rot)
              + cos(0.5*size*x_rot - size*y_rot)
    Added size parameter to control overall scaling
    Added rotation parameter to rotate the pattern
    Added spread parameter to control peak separation
    Added x_trans and y_trans parameters for translation
    """
    # Apply translation
    x = x - x_trans
    y = y - y_trans
    # Convert rotation to radians and compute trig functions
    theta = np.radians(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate coordinates
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t

    # Calculate base pattern
    z = (
        np.cos(size * x_rot)
        + np.cos(0.5 * size * x_rot + size * y_rot)
        + np.cos(0.5 * size * x_rot - size * y_rot)
    )

    # Normalize to [-1, 1] range
    z = z / 3.0  # Since we're adding 3 cosines

    # Apply spread transformation
    z = np.sign(z) * np.power(np.abs(z), 1 / spread)

    return z


# Create a figure with adjusted size to accommodate sliders and labels
fig = plt.figure(figsize=(12, 12))

# Create main axes for the surface plot
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.35)  # Make more room for sliders and labels


# Initial plot setup function
def setup_initial_plot():
    # Get initial values
    initial_range = 4 * np.pi
    initial_density = 50  # Changed to match new default density

    # Create grid
    x = np.linspace(-initial_range, initial_range, initial_density)
    y = np.linspace(-initial_range, initial_range, initial_density)
    X, Y = np.meshgrid(x, y)

    # Compute initial surface
    Z = f(X, Y, size=1.0)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    # Set initial view limits
    ax.set_xlim(-initial_range, initial_range)
    ax.set_ylim(-initial_range, initial_range)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_box_aspect([1, 1, 0.5])

    return surf


# Create initial surface plot
surf = setup_initial_plot()

# Add group labels
fig.text(0.15, 0.3, "Grid Cell Properties:", fontsize=10, fontweight="bold")
fig.text(0.15, 0.12, "Plot Properties:", fontsize=10, fontweight="bold")

# Grid Cell Properties group
size_slider_ax = plt.axes([0.2, 0.29, 0.5, 0.03])
rotation_slider_ax = plt.axes([0.2, 0.25, 0.5, 0.03])
spread_slider_ax = plt.axes([0.2, 0.21, 0.5, 0.03])
x_trans_slider_ax = plt.axes([0.2, 0.17, 0.5, 0.03])
y_trans_slider_ax = plt.axes([0.2, 0.13, 0.5, 0.03])
threshold_slider_ax = plt.axes([0.2, 0.09, 0.5, 0.03])

# Plot Properties group
range_slider_ax = plt.axes([0.2, 0.08, 0.5, 0.03])
density_slider_ax = plt.axes([0.2, 0.04, 0.5, 0.03])

# Reset button and checkboxes
reset_ax = plt.axes([0.8, 0.04, 0.1, 0.03])
gridlines_checkbox_ax = plt.axes([0.8, 0.12, 0.1, 0.03])
threshold_checkbox_ax = plt.axes([0.8, 0.08, 0.1, 0.03])

size_slider = Slider(
    ax=size_slider_ax,
    label="Size",
    valmin=0.1,
    valmax=2.0,
    valinit=1.0,
)

rotation_slider = Slider(
    ax=rotation_slider_ax,
    label="Rotation",
    valmin=-180.0,
    valmax=180.0,
    valinit=0.0,
)

spread_slider = Slider(
    ax=spread_slider_ax,
    label="Spread",
    valmin=0.25,
    valmax=2.0,
    valinit=1.0,
)

x_trans_slider = Slider(
    ax=x_trans_slider_ax,
    label="X Translation",
    valmin=-5.0,
    valmax=5.0,
    valinit=0.0,
)

y_trans_slider = Slider(
    ax=y_trans_slider_ax,
    label="Y Translation",
    valmin=-5.0,
    valmax=5.0,
    valinit=0.0,
)

range_slider = Slider(
    ax=range_slider_ax,
    label="View Range",
    valmin=1.0,
    valmax=10.0,
    valinit=4.0,  # Starting at 4 since that was our original multiplier
)

density_slider = Slider(
    ax=density_slider_ax,
    label="Point Density",
    valmin=25,
    valmax=75,
    valinit=50,
    valstep=5,  # Step by 5 points for smoother changes
)

# Add checkboxes
gridlines_checkbox = CheckButtons(gridlines_checkbox_ax, ["Gridlines"], [False])
threshold_checkbox = CheckButtons(threshold_checkbox_ax, ["Threshold"], [False])

# Add threshold slider
threshold_slider = Slider(
    ax=threshold_slider_ax,
    label="Threshold",
    valmin=-1.0,
    valmax=1.0,
    valinit=0.0,
    valstep=0.1,
)


# Update function for the slider
def update(val):
    global surf  # Ensure surf is updated

    # Remove previous surface plot
    for coll in ax.collections:
        coll.remove()

    # Update grid resolution
    current_range = range_slider.val * np.pi
    current_density = int(density_slider.val)  # Ensure it's an integer

    # Generate higher-resolution grid
    x = np.linspace(-current_range, current_range, current_density)
    y = np.linspace(-current_range, current_range, current_density)
    X_new, Y_new = np.meshgrid(x, y)

    # Compute function values
    Z = f(
        X_new,
        Y_new,
        size=size_slider.val,
        rotation=rotation_slider.val,
        spread=spread_slider.val,
        x_trans=x_trans_slider.val,
        y_trans=y_trans_slider.val,
    )

    # Apply threshold if enabled
    if threshold_checkbox.get_status()[0]:
        Z_masked = np.ma.masked_where(Z < threshold_slider.val, Z)
    else:
        Z_masked = Z

    # Update the surface plot with gridlines based on checkbox state
    show_gridlines = gridlines_checkbox.get_status()[0]
    surf = ax.plot_surface(
        X_new,
        Y_new,
        Z_masked,
        cmap="viridis",
        edgecolor="k" if show_gridlines else None,
        rstride=1,
        cstride=1,
    )
    # Ensure the axes reflect the updated density
    ax.set_xlim(-current_range, current_range)
    ax.set_ylim(-current_range, current_range)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_box_aspect([1, 1, 0.5])

    # Update labels and title
    ax.set_xlabel("x", fontsize=12, labelpad=10)
    ax.set_ylabel("y", fontsize=12, labelpad=10)
    ax.set_zlabel("f(x, y)", fontsize=12, labelpad=10)
    ax.set_title("Grid Cell Pattern", fontsize=14, pad=20)

    fig.canvas.draw_idle()


# Create reset button
reset_button = Button(reset_ax, "Reset", hovercolor="0.975")


# Reset function
def reset(event):
    size_slider.reset()
    rotation_slider.reset()
    spread_slider.reset()
    x_trans_slider.reset()
    y_trans_slider.reset()
    range_slider.reset()
    density_slider.reset()
    threshold_slider.reset()


# Function to handle checkbox changes
def on_checkbox_clicked(label):
    update(None)  # Trigger update to refresh the plot


# Register the update function with all controls
size_slider.on_changed(update)
rotation_slider.on_changed(update)
spread_slider.on_changed(update)
x_trans_slider.on_changed(update)
y_trans_slider.on_changed(update)
range_slider.on_changed(update)
density_slider.on_changed(update)
threshold_slider.on_changed(update)
reset_button.on_clicked(reset)
gridlines_checkbox.on_clicked(on_checkbox_clicked)
threshold_checkbox.on_clicked(on_checkbox_clicked)

# Set initial labels and title
ax.set_xlabel("x", fontsize=12, labelpad=10)
ax.set_ylabel("y", fontsize=12, labelpad=10)
ax.set_zlabel("f(x, y)", fontsize=12, labelpad=10)
ax.set_title("Grid Cell Pattern", fontsize=14, pad=20)

# Set tick label sizes
ax.tick_params(axis="both", which="major", labelsize=10)

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
