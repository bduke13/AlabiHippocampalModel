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
    show_2d_plots = False  # Default when running in notebook/interactive mode

# Define grid ranges and density
x_range = (-5, 5)
y_range = (-5, 5)
z_range = (-5, 5)
density = 10  # Adjust this for resolution

# Define BVC parameters
preferred_distance = 4.0  # d_i
preferred_horiz_angle = np.pi / 4  # φ_i (in radians)
preferred_vert_angle = np.pi / 6  # ψ_i (in radians)

sigma_r = 1.0  # Tuning width for distance
sigma_theta = 0.5  # Tuning width for horizontal direction
sigma_phi = 0.5  # Tuning width for vertical direction

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

# Create and display histogram only if show_2d_plots is True
if show_2d_plots:
    # Create a histogram of the activation distribution
    hist_fig = go.Figure(
        data=[
            go.Histogram(
                x=raw_activation,
                nbinsx=50,
                marker_color="rgba(100, 0, 200, 0.7)",
                opacity=0.8,
                name="Activation Distribution",
            )
        ]
    )

    hist_fig.update_layout(
        title="Distribution of BVC Activations",
        xaxis_title="Activation Value",
        yaxis_title="Frequency",
        bargap=0.05,
        template="plotly_white",
    )

    # Add a vertical line for the mean activation
    hist_fig.add_vline(
        x=np.mean(raw_activation),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(raw_activation):.4f}",
        annotation_position="top right",
    )

    # Add a vertical line for the median activation
    hist_fig.add_vline(
        x=np.median(raw_activation),
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {np.median(raw_activation):.4f}",
        annotation_position="top left",
    )

    # Display the histogram
    hist_fig.show()

# Calculate the preferred location in Cartesian coordinates
preferred_x = (
    preferred_distance * np.cos(preferred_horiz_angle) * np.cos(preferred_vert_angle)
)
preferred_y = (
    preferred_distance * np.sin(preferred_horiz_angle) * np.cos(preferred_vert_angle)
)
preferred_z = preferred_distance * np.sin(preferred_vert_angle)

# Create a 3D visualization with isosurfaces for better visibility
# Reshape the activation values to match the 3D grid
activation_3d = activation.reshape(X.shape)

# Create an interactive 3D plot
fig = go.Figure()

# Determine visualization approach based on activation distribution
# Find the 90th percentile for high activation threshold
high_threshold = np.percentile(activation, 90)
# Find a lower threshold that captures the structure but excludes noise
mid_threshold = np.percentile(activation, 70)

# Add isosurface for better 3D visualization
# Create multiple isosurfaces at different activation levels
# Use more levels for better visualization of the 3D structure
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
            surface_count=2,  # More detailed surfaces
            colorscale="Plasma",
            showscale=level_idx == 0,  # Only show colorbar for the first isosurface
            colorbar=dict(title="Activation") if level_idx == 0 else None,
            name=f"Activation Level {level:.3f}",
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        )
    )

# Add a volume rendering for better visualization of the full 3D structure
fig.add_trace(
    go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=activation.flatten(),
        isomin=mid_threshold * 0.8,  # Slightly lower threshold for volume
        isomax=high_threshold,
        opacity=0.1,  # Very transparent
        surface_count=20,  # More surfaces for smoother rendering
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
        marker=dict(
            size=15,  # Increased size
            color="red",
            symbol="circle",
        ),
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
        marker=dict(
            size=15,  # Increased size
            color="green",
            symbol="circle",
        ),
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
        line=dict(
            color="black",
            width=6,  # Increased line width
            dash="dash",
        ),
        name="Preferred Distance Vector",
    )
)

# Set layout and fix axis ranges with improved camera angle
fig.update_layout(
    title="3D Boundary Vector Cell (BVC) Firing Field",
    scene=dict(
        xaxis=dict(title="X", range=[x_range[0], x_range[1]]),
        yaxis=dict(title="Y", range=[y_range[0], y_range[1]]),
        zaxis=dict(title="Z", range=[z_range[0], z_range[1]]),
        # Set a good initial camera angle to see the structure
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0),
        ),
        aspectmode="cube",  # Force equal aspect ratio
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


# Display the plot
fig.show()

# Create and display 2D heatmap only if show_2d_plots is True
if show_2d_plots:
    # Create a 2D heatmap of activations at z=0 plane
    z_mid_idx = len(z_values) // 2  # Get the middle index for z=0 (or closest to 0)
    z_slice = Z[:, :, z_mid_idx]
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

    # Add a marker for the preferred location projection on z=0 plane
    if abs(preferred_z) < max(
        z_range
    ):  # Only if the preferred location is within z range
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

        # Add a dashed line from origin to preferred location
        heatmap_fig.add_trace(
            go.Scatter(
                x=[0, preferred_x],
                y=[0, preferred_y],
                mode="lines",
                line=dict(
                    color="black",
                    width=2,
                    dash="dash",
                ),
                name="Preferred Vector (z-projection)",
            )
        )

    heatmap_fig.update_layout(
        title="2D Heatmap of BVC Activations (z=0 plane)",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )

    # Display the heatmap
    heatmap_fig.show()

# Add a main function to make the script runnable
if __name__ == "__main__":
    print(
        f"3D BVC visualization complete. 2D plots {'shown' if show_2d_plots else 'hidden'}."
    )
    print("To show 2D plots, run with: python 3D_bvc_field_view.py --show_2d_plots")
