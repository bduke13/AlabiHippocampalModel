# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 3D BVC field")
    parser.add_argument(
        "--show_2d_plots",
        action="store_true",
        help="Show 2D histogram and heatmap plots",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80,
        help="Percentile threshold for point visualization (0-100)",
    )
    parser.add_argument(
        "--min_activation",
        type=float,
        default=None,
        help="Minimum absolute activation value to display (overrides percentile threshold)",
    )
    parser.add_argument(
        "--point_density",
        type=float,
        default=1.0,
        help="Density of points to display (0.1-1.0)",
    )
    return parser.parse_args()


show_2d_plots = False
min_activation = 0.000001
# Define grid ranges and density
x_range = (-2.5, 2.5)
y_range = (-2.5, 2.5)
z_range = (-2.5, 2.5)
density = 5  # Adjust this for resolution

# Define BVC parameters
preferred_distance = 2.0  # d_i
preferred_horiz_angle = np.pi / 4  # φ_i (in radians)
preferred_vert_angle = np.pi / 4  # ψ_i (in radians)

sigma_r = 2.5  # Tuning width for distance
sigma_theta_deg = 50  # Tuning width for horizontal direction
sigma_phi_deg = 50  # Tuning width for vertical direction

sigma_theta = np.radians(sigma_theta_deg)
sigma_phi = np.radians(sigma_phi_deg)

print(f"sigma_theta = {sigma_theta} rad, sigma_phi = {sigma_phi} rad")

# Generate grid points
x_values = np.linspace(*x_range, int((x_range[1] - x_range[0]) * density))
y_values = np.linspace(*y_range, int((y_range[1] - y_range[0]) * density))
z_values = np.linspace(*z_range, int((z_range[1] - z_range[0]) * density))

X, Y, Z = np.meshgrid(x_values, y_values, z_values)
X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()

# Convert to spherical coordinates
r_j = np.sqrt(X_flat**2 + Y_flat**2 + Z_flat**2)  # Distance
theta_j = np.arctan2(Y_flat, X_flat)  # Horizontal angle
phi_j = np.arcsin(Z_flat / (r_j + 1e-6))  # Vertical angle (avoid div by zero)

# Compute the BVC activation based on Equation (2)
activation = (
    np.exp(-((r_j - preferred_distance) ** 2) / (2 * sigma_r**2))
    / np.sqrt(2 * np.pi * sigma_r**2)
    * np.exp(-((theta_j - preferred_horiz_angle) ** 2) / (2 * sigma_theta**2))
    / np.sqrt(2 * np.pi * sigma_theta**2)
    * np.exp(-((phi_j - preferred_vert_angle) ** 2) / (2 * sigma_phi**2))
    / np.sqrt(2 * np.pi * sigma_phi**2)
)

# Store the raw activation values before normalization
raw_activation = activation.copy()

# Normalize activation between [0, 0.5]
activation = activation / activation.max() * 0.5

# Calculate the preferred location in Cartesian coordinates
preferred_x = (
    preferred_distance * np.cos(preferred_horiz_angle) * np.cos(preferred_vert_angle)
)
preferred_y = (
    preferred_distance * np.sin(preferred_horiz_angle) * np.cos(preferred_vert_angle)
)
preferred_z = preferred_distance * np.sin(preferred_vert_angle)

# Compute the preferred axis vector (from origin to preferred location)
v = np.array([preferred_x, preferred_y, preferred_z])
v_norm = np.linalg.norm(v)
v_unit = v / v_norm if v_norm != 0 else np.array([1, 0, 0])

# Choose an arbitrary vector that is not parallel to v_unit
arbitrary = (
    np.array([0, 0, 1])
    if abs(np.dot(v_unit, np.array([0, 0, 1]))) < 0.9
    else np.array([1, 0, 0])
)

# Compute two orthonormal vectors (u1 and u2) perpendicular to v_unit
u1 = np.cross(v_unit, arbitrary)
u1 = u1 / np.linalg.norm(u1)
u2 = np.cross(v_unit, u1)
u2 = u2 / np.linalg.norm(u2)

# --- Apply diagonal masking to "chunk out" half of the distribution ---
# For each grid point, check if it's on one side of the x=y plane
# Points where x > y are removed (keeping only the points where x ≤ y)
mask = X_flat > Y_flat
activation[mask] = 0
# --- End masking ---

# Reshape the activation values to match the 3D grid
activation_3d = activation.reshape(X.shape)

# Create an interactive 3D plot
fig = go.Figure()

# Filter points based on activation threshold to reduce visual clutter
# Only show points with significant activation
if min_activation is not None:
    # Use absolute minimum activation value if provided
    activation_threshold = min_activation
    print(f"Using minimum activation threshold: {min_activation}")
    mask = activation > activation_threshold
else:
    # Use percentile threshold if no minimum activation is provided
    threshold_percentile = 80
    activation_threshold = np.percentile(
        activation[activation > 0], threshold_percentile
    )
    print(
        f"Using {threshold_percentile}th percentile threshold: {activation_threshold:.6f}"
    )

# Get coordinates and values of points above threshold
x_points = X_flat[mask]
y_points = Y_flat[mask]
z_points = Z_flat[mask]
activation_points = activation[mask]

# Print information about the filtered points
print(f"Number of points displayed: {len(x_points)} (out of {len(X_flat)} total)")
if len(activation_points) > 0:
    print(
        f"Activation range of displayed points: {activation_points.min():.6f} to {activation_points.max():.6f}"
    )

# Normalize point sizes based on activation values
min_size = 1
max_size = 30
point_sizes = min_size + (max_size - min_size) * (
    activation_points / activation_points.max()
)

# Add scatter points colored by activation value
fig.add_trace(
    go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode="markers",
        marker=dict(
            size=point_sizes,
            color=activation_points,
            colorscale="Plasma",
            opacity=0.8,
            colorbar=dict(title="Activation"),
            showscale=True,
        ),
        name="BVC Activation Points",
    )
)

# Optional: Add a very low opacity volume to show the general shape
# Comment this out if you only want the points
# fig.add_trace(
#     go.Volume(
#         x=X.flatten(),
#         y=Y.flatten(),
#         z=Z.flatten(),
#         value=activation.flatten(),
#         isomin=activation_threshold * 0.5,
#         isomax=activation.max(),
#         opacity=0.05,
#         surface_count=10,
#         colorscale="Plasma",
#         showscale=False,
#         name="Volume Outline",
#     )
# )

# Add a large red sphere at the origin (0,0,0)
fig.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        marker=dict(size=15, color="red", symbol="circle"),
        name="Origin",
    )
)

# Add a large green sphere at the preferred location
fig.add_trace(
    go.Scatter3d(
        x=[preferred_x],
        y=[preferred_y],
        z=[preferred_z],
        mode="markers",
        marker=dict(size=15, color="green", symbol="circle"),
        name="Preferred Location",
    )
)

# Add a black dashed line from the origin to the preferred location
fig.add_trace(
    go.Scatter3d(
        x=[0, preferred_x],
        y=[0, preferred_y],
        z=[0, preferred_z],
        mode="lines",
        line=dict(color="black", width=6, dash="dash"),
        name="Preferred Distance Vector",
    )
)

# Add coordinate axis lines
# X-axis (red)
fig.add_trace(
    go.Scatter3d(
        x=[0, x_range[1]],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line=dict(color="red", width=3),
        name="X-axis",
    )
)
# Y-axis (green)
fig.add_trace(
    go.Scatter3d(
        x=[0, 0],
        y=[0, y_range[1]],
        z=[0, 0],
        mode="lines",
        line=dict(color="green", width=3),
        name="Y-axis",
    )
)
# Z-axis (blue)
fig.add_trace(
    go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, z_range[1]],
        mode="lines",
        line=dict(color="blue", width=3),
        name="Z-axis",
    )
)

# Set layout and fix axis ranges with a side angle camera view
fig.update_layout(
    title="3D Boundary Vector Cell (BVC) Firing Field - Point Cloud Visualization",
    scene=dict(
        xaxis=dict(title="X", range=[x_range[0], x_range[1]]),
        yaxis=dict(title="Y", range=[y_range[0], y_range[1]]),
        zaxis=dict(title="Z", range=[z_range[0], z_range[1]]),
        camera=dict(eye=dict(x=1.5, y=0.1, z=0.5), center=dict(x=0, y=0, z=0)),
        aspectmode="cube",
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        traceorder="normal",
        font=dict(family="sans-serif", size=12, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    ),
)

# Display the 3D plot
fig.show()

# Create and display 2D heatmap only if show_2d_plots is True
if show_2d_plots:
    z_mid_idx = (
        len(z_values) // 2
    )  # Use the middle index for the z=0 plane (or nearest)
    activation_2d = activation.reshape(X.shape)[:, :, z_mid_idx]

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=activation_2d,
            x=x_values,
            y=y_values,
            colorscale="Plasma",
            colorbar=dict(title="Activation"),
        )
    )

    # Add coordinate axes to 2D plot
    # X-axis (red)
    heatmap_fig.add_trace(
        go.Scatter(
            x=[0, x_range[1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="red", width=2),
            name="X-axis",
        )
    )
    # Y-axis (green)
    heatmap_fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, y_range[1]],
            mode="lines",
            line=dict(color="green", width=2),
            name="Y-axis",
        )
    )

    # Add a marker for the origin
    heatmap_fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                size=12, color="red", symbol="circle", line=dict(color="black", width=1)
            ),
            name="Origin",
        )
    )

    # Add a marker for the preferred location projection on the z=0 plane
    if abs(preferred_z) < max(z_range):
        heatmap_fig.add_trace(
            go.Scatter(
                x=[preferred_x],
                y=[preferred_y],
                mode="markers",
                marker=dict(
                    size=12,
                    color="green",
                    symbol="circle",
                    line=dict(color="black", width=1),
                ),
                name="Preferred Location (z-projection)",
            )
        )
        heatmap_fig.add_trace(
            go.Scatter(
                x=[0, preferred_x],
                y=[0, preferred_y],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Preferred Vector (z-projection)",
            )
        )

    heatmap_fig.update_layout(
        title="2D Heatmap of BVC Activations (z=0 plane, quadrant removed)",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )

    heatmap_fig.show()

# Add a main function to make the script runnable
if __name__ == "__main__":
    print(
        f"3D BVC visualization complete. 2D plots {'shown' if show_2d_plots else 'hidden'}."
    )
    print(f"Point density: {point_density}")
    print("Usage options:")
    print("  --show_2d_plots         Show 2D heatmap plots")
    print("  --threshold VALUE       Set percentile threshold for points (0-100)")
    print(
        "  --min_activation VALUE  Set minimum absolute activation value (overrides percentile)"
    )
    print("  --point_density VALUE   Set density of points to display (0.1-1.0)")
