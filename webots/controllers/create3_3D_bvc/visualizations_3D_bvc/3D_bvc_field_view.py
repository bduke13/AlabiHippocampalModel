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
    return parser.parse_args()


# Check if running as script or in interactive mode
try:
    args = parse_args()
    show_2d_plots = args.show_2d_plots
except SystemExit:  # Catch the SystemExit when running in notebook
    show_2d_plots = True  # Default when running in notebook/interactive mode

# Define grid ranges and density
x_range = (0, 20)
y_range = (0, 20)
z_range = (0, 20)
density = 5  # Adjust this for resolution

# Define BVC parameters
preferred_distance = 2.0  # d_i
preferred_horiz_angle = np.pi / 4  # φ_i (in radians)
preferred_vert_angle = np.pi / 6  # ψ_i (in radians)

sigma_r = 0.5  # Tuning width for distance
sigma_theta = 0.02  # Tuning width for horizontal direction
sigma_phi = 0.02  # Tuning width for vertical direction

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

# --- Apply quadrant masking to "chunk out" one quarter of the distribution ---
# For each grid point, compute its dot products with u1 and u2.
# Points with positive dot products with both (i.e. in one quadrant) are removed.
dot1 = X_flat * u1[0] + Y_flat * u1[1] + Z_flat * u1[2]
dot2 = X_flat * u2[0] + Y_flat * u2[1] + Z_flat * u2[2]
mask = (dot1 > 0) & (dot2 > 0)
activation[mask] = 0
# --- End masking ---

# Reshape the activation values to match the 3D grid
activation_3d = activation.reshape(X.shape)

# Create an interactive 3D plot
fig = go.Figure()

# Determine visualization approach based on activation distribution
high_threshold = np.percentile(activation, 90)
mid_threshold = np.percentile(activation, 70)

# Add isosurfaces for better 3D visualization (multiple levels)
for level_idx, level in enumerate(np.linspace(mid_threshold, high_threshold, 4)):
    opacity = 0.2 + 0.1 * level_idx  # Increase opacity for higher activation levels
    fig.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=activation.flatten(),
            isomin=level,
            isomax=level + 0.02,
            opacity=opacity,
            surface_count=2,
            colorscale="Plasma",
            showscale=level_idx == 0,
            colorbar=dict(title="Activation") if level_idx == 0 else None,
            name=f"Activation Level {level:.3f}",
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        )
    )

# Add a volume rendering for smoother visualization of the full 3D structure
fig.add_trace(
    go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=activation.flatten(),
        isomin=mid_threshold * 0.8,
        isomax=high_threshold,
        opacity=0.1,
        surface_count=20,
        colorscale="Plasma",
        showscale=False,
        name="Volume Rendering",
    )
)

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

# Set layout and fix axis ranges with an improved camera angle
fig.update_layout(
    title="3D Boundary Vector Cell (BVC) Firing Field (with quadrant removed)",
    scene=dict(
        xaxis=dict(title="X", range=[x_range[0], x_range[1]]),
        yaxis=dict(title="Y", range=[y_range[0], y_range[1]]),
        zaxis=dict(title="Z", range=[z_range[0], z_range[1]]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0)),
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
    print("To show 2D plots, run with: python 3D_bvc_field_view.py --show_2d_plots")
