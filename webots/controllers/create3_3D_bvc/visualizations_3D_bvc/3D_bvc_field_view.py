# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core.layers.boundary_vector_cell_layer_3D import BoundaryVectorCellLayer3D


def compute_3d_bvc_response(
    bvc_distance,
    bvc_horiz_angle,
    bvc_vert_angle,
    point_x,
    point_y,
    point_z,
    sigma_r,
    sigma_theta,
    sigma_phi,
):
    """Compute BVC response for a boundary point at (x,y,z) relative to agent at origin."""
    # Convert Cartesian to Spherical coordinates
    r = torch.sqrt(point_x**2 + point_y**2 + point_z**2)
    theta = torch.atan2(point_y, point_x)  # horizontal angle (azimuth)
    phi = torch.asin(point_z / r)  # vertical angle (elevation)

    # Wrap horizontal angle difference to [-π, π]
    theta_diff = torch.atan2(
        torch.sin(theta - bvc_horiz_angle), torch.cos(theta - bvc_horiz_angle)
    )

    # Vertical angle difference (no wrapping needed as it's in [-π/2, π/2])
    phi_diff = phi - bvc_vert_angle

    # Distance component
    distance_component = torch.exp(-((r - bvc_distance) ** 2) / (2 * sigma_r**2))
    distance_component /= torch.sqrt(2 * torch.pi * sigma_r**2)

    # Horizontal angle component
    horiz_component = torch.exp(-(theta_diff**2) / (2 * sigma_theta**2))
    horiz_component /= torch.sqrt(2 * torch.pi * sigma_theta**2)

    # Vertical angle component
    vert_component = torch.exp(-(phi_diff**2) / (2 * sigma_phi**2))
    vert_component /= torch.sqrt(2 * torch.pi * sigma_phi**2)

    return distance_component * horiz_component * vert_component


def main():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize BVC layer with a single preferred vertical angle
    bvc_layer = BoundaryVectorCellLayer3D(
        max_dist=12.0,
        n_hd=8,
        phi_vert_preferred=[0.0],  # 0.3 radians ≈ 17 degrees elevation
        sigma_rs=[1.0],
        sigma_thetas=[0.3],  # broader horizontal tuning
        sigma_phis=[0.3],  # broader vertical tuning
        scaling_factors=[1.0],
        num_bvc_per_dir=50,
        device=device,
    )

    # Select a random BVC to visualize
    bvc_idx = np.random.randint(0, bvc_layer.num_bvc)
    selected_distance = bvc_layer.d_i[bvc_idx].item()
    selected_horiz_angle = bvc_layer.theta_i[bvc_idx].item()
    selected_vert_angle = bvc_layer.psi_i[bvc_idx].item()

    print(f"\nSelected BVC parameters:")
    print(f"Distance: {selected_distance:.2f}m")
    print(f"Horizontal angle: {np.rad2deg(selected_horiz_angle):.1f}°")
    print(f"Vertical angle: {np.rad2deg(selected_vert_angle):.1f}°")

    # Create 3D grid of points
    resolution = 300
    max_range = 15
    x = torch.linspace(-max_range, max_range, resolution, device=device)
    y = torch.linspace(-max_range, max_range, resolution, device=device)
    z = torch.linspace(-max_range, max_range, resolution, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Compute BVC response for each point
    response = compute_3d_bvc_response(
        selected_distance,
        selected_horiz_angle,
        selected_vert_angle,
        X,
        Y,
        Z,
        bvc_layer.sigma_r[bvc_idx],
        bvc_layer.sigma_theta[bvc_idx],
        bvc_layer.sigma_phi[bvc_idx],
    )

    # Normalize response to [0, 1]
    response = response / response.max()

    # Create two subplots: full 3D view and cross-section
    fig = plt.figure(figsize=(15, 7))

    # 1. Full 3D view (with cutout)
    ax1 = fig.add_subplot(121, projection="3d")

    # Create mask for cutout (show only points where y > 0)
    mask = Y > 0

    # Plot points where response > threshold and Z <= 0
    threshold = 0.01
    strong_response = response > threshold
    horizontal_mask = Z <= 0  # Only keep points at or below horizontal plane
    points_to_plot = strong_response & horizontal_mask

    # Convert to numpy and apply mask
    X_plot = X[points_to_plot].cpu().numpy()
    Y_plot = Y[points_to_plot].cpu().numpy()
    Z_plot = Z[points_to_plot].cpu().numpy()
    colors = response[points_to_plot].cpu().numpy()

    scatter = ax1.scatter(X_plot, Y_plot, Z_plot, c=colors, cmap="viridis", alpha=0.3)

    # Set equal aspect ratio for better 3D visualization
    ax1.set_box_aspect([1, 1, 1])
    plt.colorbar(scatter, ax=ax1, label="Normalized Response")

    # Plot the agent position
    ax1.scatter([0], [0], [0], color="red", s=100, marker="o", label="Agent")

    # Plot the preferred location
    x_pref = (
        selected_distance * np.cos(selected_vert_angle) * np.cos(selected_horiz_angle)
    )
    y_pref = (
        selected_distance * np.cos(selected_vert_angle) * np.sin(selected_horiz_angle)
    )
    z_pref = selected_distance * np.sin(selected_vert_angle)
    ax1.scatter(
        [x_pref],
        [y_pref],
        [z_pref],
        color="red",
        s=200,
        marker="*",
        label="Preferred Location",
    )

    # Draw line from origin to preferred location
    ax1.plot([0, x_pref], [0, y_pref], [0, z_pref], "r--", alpha=0.5)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D BVC Response Field\n(Z ≤ 0)")

    # 2. Horizontal cross-section at preferred vertical height
    ax2 = fig.add_subplot(122)

    # Find closest Z plane to preferred height
    z_idx = torch.argmin(torch.abs(z - z_pref))
    cross_section = response[:, :, z_idx].cpu().numpy()

    # Plot cross-section
    im = ax2.imshow(
        cross_section.T,
        origin="lower",
        extent=[-max_range, max_range, -max_range, max_range],
        cmap="viridis",
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(
        f"Horizontal Cross-section at preferred elevation\nZ = {z_pref:.2f}m (elevation angle: {np.rad2deg(selected_vert_angle):.1f}°)"
    )

    # Add colorbar
    plt.colorbar(im, ax=ax2, label="Normalized Response")

    # Plot agent and preferred location on cross-section
    ax2.plot(0, 0, "ro", markersize=10, label="Agent")
    ax2.plot(x_pref, y_pref, "r*", markersize=15, label="Preferred Location")
    ax2.plot([0, x_pref], [0, y_pref], "r--", alpha=0.5)

    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
