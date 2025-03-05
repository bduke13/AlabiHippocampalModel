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
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.55)  # Increase bottom margin for extra room

# Initial plot setup function
def setup_initial_plot():
    initial_range = 4 * np.pi
    initial_density = 50

    x = np.linspace(-initial_range, initial_range, initial_density)
    y = np.linspace(-initial_range, initial_range, initial_density)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y, size=1.0)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlim(-initial_range, initial_range)
    ax.set_ylim(-initial_range, initial_range)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_box_aspect([1, 1, 0.5])

    return surf


# Create initial surface plot
surf = setup_initial_plot()

# Add group labels
fig.text(0.05, 0.56, "Grid Cell Properties:", fontsize=10, fontweight="bold")
fig.text(0.05, 0.22, "Plot Properties:", fontsize=10, fontweight="bold")

# Grid Cell Properties group sliders
size_slider_ax = plt.axes([0.2, 0.50, 0.55, 0.03])
rotation_slider_ax = plt.axes([0.2, 0.45, 0.55, 0.03])
spread_slider_ax = plt.axes([0.2, 0.40, 0.55, 0.03])
x_trans_slider_ax = plt.axes([0.2, 0.35, 0.55, 0.03])
y_trans_slider_ax = plt.axes([0.2, 0.30, 0.55, 0.03])
threshold_slider_ax = plt.axes([0.2, 0.25, 0.55, 0.03])

# Plot Properties group sliders
range_slider_ax = plt.axes([0.2, 0.20, 0.55, 0.03])
density_slider_ax = plt.axes([0.2, 0.15, 0.55, 0.03])

# Reset button and checkboxes (placed in a separate column)
reset_ax = plt.axes([0.8, 0.50, 0.1, 0.04])
gridlines_checkbox_ax = plt.axes([0.8, 0.45, 0.1, 0.04])
threshold_checkbox_ax = plt.axes([0.8, 0.40, 0.1, 0.04])

# Create sliders
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

threshold_slider = Slider(
    ax=threshold_slider_ax,
    label="Threshold",
    valmin=-1.0,
    valmax=1.0,
    valinit=0.0,
    valstep=0.1,
)

range_slider = Slider(
    ax=range_slider_ax,
    label="View Range",
    valmin=1.0,
    valmax=10.0,
    valinit=4.0,
)

density_slider = Slider(
    ax=density_slider_ax,
    label="Point Density",
    valmin=25,
    valmax=75,
    valinit=50,
    valstep=5,
)

# Create checkboxes
gridlines_checkbox = CheckButtons(gridlines_checkbox_ax, ["Gridlines"], [False])
threshold_checkbox = CheckButtons(threshold_checkbox_ax, ["Threshold"], [False])

# Create reset button
reset_button = Button(reset_ax, "Reset", hovercolor="0.975")


# Update function for the slider
def update(val):
    global surf  # Ensure surf is updated

    # Remove previous surface plot
    for coll in ax.collections:
        coll.remove()

    current_range = range_slider.val * np.pi
    current_density = int(density_slider.val)

    x = np.linspace(-current_range, current_range, current_density)
    y = np.linspace(-current_range, current_range, current_density)
    X_new, Y_new = np.meshgrid(x, y)

    Z = f(
        X_new,
        Y_new,
        size=size_slider.val,
        rotation=rotation_slider.val,
        spread=spread_slider.val,
        x_trans=x_trans_slider.val,
        y_trans=y_trans_slider.val,
    )

    if threshold_checkbox.get_status()[0]:
        Z_masked = np.ma.masked_where(Z < threshold_slider.val, Z)
    else:
        Z_masked = Z

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

    ax.set_xlim(-current_range, current_range)
    ax.set_ylim(-current_range, current_range)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_box_aspect([1, 1, 0.5])

    ax.set_xlabel("x", fontsize=12, labelpad=10)
    ax.set_ylabel("y", fontsize=12, labelpad=10)
    ax.set_zlabel("f(x, y)", fontsize=12, labelpad=10)
    ax.set_title("Grid Cell Pattern", fontsize=14, pad=20)

    fig.canvas.draw_idle()


# Reset function
def reset(event):
    size_slider.reset()
    rotation_slider.reset()
    spread_slider.reset()
    x_trans_slider.reset()
    y_trans_slider.reset()
    threshold_slider.reset()
    range_slider.reset()
    density_slider.reset()


# Function to handle checkbox changes
def on_checkbox_clicked(label):
    update(None)  # Trigger update to refresh the plot


# Register update callbacks
size_slider.on_changed(update)
rotation_slider.on_changed(update)
spread_slider.on_changed(update)
x_trans_slider.on_changed(update)
y_trans_slider.on_changed(update)
threshold_slider.on_changed(update)
range_slider.on_changed(update)
density_slider.on_changed(update)
reset_button.on_clicked(reset)
gridlines_checkbox.on_clicked(on_checkbox_clicked)
threshold_checkbox.on_clicked(on_checkbox_clicked)

# Set initial labels and title
ax.set_xlabel("x", fontsize=12, labelpad=10)
ax.set_ylabel("y", fontsize=12, labelpad=10)
ax.set_zlabel("f(x, y)", fontsize=12, labelpad=10)
ax.set_title("Grid Cell Pattern", fontsize=14, pad=20)
ax.tick_params(axis="both", which="major", labelsize=10)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
