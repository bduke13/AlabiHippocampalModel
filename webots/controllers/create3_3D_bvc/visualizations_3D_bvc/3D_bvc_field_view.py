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
    """Compute BVC response for a boundary point at (x,y,z) relative to the agent at origin."""
    # Convert Cartesian to Spherical coordinates
    r = torch.sqrt(point_x**2 + point_y**2 + point_z**2)
    theta = torch.atan2(point_y, point_x)  # horizontal angle
    phi = torch.asin(point_z / r)  # vertical angle (assuming r>0)

    # Horizontal angle difference wrapped to [-pi, pi]
    theta_diff = torch.atan2(
        torch.sin(theta - bvc_horiz_angle), torch.cos(theta - bvc_horiz_angle)
    )
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    phi_vert_preferred = [0.0]
    sigma_rs = [5] * len(phi_vert_preferred)
    sigma_thetas = [1] * len(phi_vert_preferred)
    sigma_phis = [2] * len(phi_vert_preferred)
    scaling_factors = [1] * len(phi_vert_preferred)

    # Initialize your BVC layer
    bvc_layer = BoundaryVectorCellLayer3D(
        device=device,
        max_dist=10,
        n_hd=8,
        sigma_rs=sigma_rs,
        sigma_thetas=sigma_thetas,
        sigma_phis=sigma_phis,
        scaling_factors=scaling_factors,
    )

    # Visualization options
    cutoff_above_horizontal = True  # Toggle to show only points below horizontal plane

    # Pick a BVC (only one in this example)
    bvc_idx = 0
    selected_distance = bvc_layer.d_i[bvc_idx].item()
    selected_horiz_angle = bvc_layer.theta_i[bvc_idx].item()
    selected_vert_angle = bvc_layer.psi_i[bvc_idx].item()

    print(f"\nSelected BVC parameters:")
    print(f"  Distance:         {selected_distance:.2f} m")
    print(f"  Horizontal angle: {np.rad2deg(selected_horiz_angle):.1f}°")
    print(f"  Vertical angle:   {np.rad2deg(selected_vert_angle):.1f}°")

    # Create a smaller 3D grid of points
    resolution = 60
    max_range = 15
    x = torch.linspace(-max_range, max_range, resolution, device=device)
    y = torch.linspace(-max_range, max_range, resolution, device=device)
    z = torch.linspace(-max_range, max_range, resolution, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Compute BVC response over the grid
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

    # Normalize to [0, 1]
    response = response / response.max()

    # Masks
    threshold = 0.02
    strong_response = response > threshold
    front_mask = Y > 0  # optional if you want to cut out half

    if cutoff_above_horizontal:
        horizontal_mask = Z <= 0
        points_to_plot = strong_response & horizontal_mask & front_mask
    else:
        points_to_plot = strong_response & front_mask
    print(f"Points above threshold: {points_to_plot.sum().item()}")

    # -- Optionally sub-sample to speed up scatter plot --
    idxs = torch.nonzero(points_to_plot).squeeze()
    # Shuffle indices randomly
    rand_perm = torch.randperm(len(idxs), device=device)
    # Keep at most 40k points
    max_points = 40000
    idxs = idxs[rand_perm[:max_points]]

    # Extract coordinates
    X_plot = X[idxs[:, 0], idxs[:, 1], idxs[:, 2]].cpu().numpy()
    Y_plot = Y[idxs[:, 0], idxs[:, 1], idxs[:, 2]].cpu().numpy()
    Z_plot = Z[idxs[:, 0], idxs[:, 1], idxs[:, 2]].cpu().numpy()
    colors = response[idxs[:, 0], idxs[:, 1], idxs[:, 2]].cpu().numpy()

    # Set up the figure
    fig = plt.figure(figsize=(15, 7))

    # 1) 3D Scatter
    ax1 = fig.add_subplot(121, projection="3d")

    sc = ax1.scatter(X_plot, Y_plot, Z_plot, c=colors, cmap="viridis", alpha=0.3, s=5)
    ax1.set_box_aspect((1, 1, 1))
    plt.colorbar(sc, ax=ax1, label="Normalized Response")

    # Plot the agent origin
    ax1.scatter([0], [0], [0], color="red", s=100, marker="o", label="Agent")

    # Preferred location
    x_pref = (
        selected_distance * np.cos(selected_vert_angle) * np.cos(selected_horiz_angle)
    )
    y_pref = (
        selected_distance * np.cos(selected_vert_angle) * np.sin(selected_horiz_angle)
    )
    z_pref = selected_distance * np.sin(selected_vert_angle)

    ax1.scatter(
        [x_pref], [y_pref], [z_pref], color="red", s=200, marker="*", label="Preferred"
    )
    ax1.plot([0, x_pref], [0, y_pref], [0, z_pref], "r--", alpha=0.5)

    title = "3D BVC Response Field"
    if cutoff_above_horizontal:
        title += " (Below Horizontal Plane)"
    ax1.set_title(title)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    # 2) Horizontal cross-section at or near the preferred vertical angle
    ax2 = fig.add_subplot(122)
    # Find nearest z plane to the actual z_pref
    z_idx = torch.argmin(torch.abs(z - z_pref))
    cross_section = response[:, :, z_idx].cpu().numpy().T

    # Show cross-section
    im = ax2.imshow(
        cross_section,
        origin="lower",
        extent=(-max_range, max_range, -max_range, max_range),
        cmap="viridis",
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax2, label="Normalized Response")

    # Mark agent and preferred location on cross-section
    ax2.plot(0, 0, "ro", markersize=8, label="Agent")
    ax2.plot(x_pref, y_pref, "r*", markersize=12, label="Preferred")
    ax2.plot([0, x_pref], [0, y_pref], "r--", alpha=0.5)

    ax2.set_title(
        f"Cross-section at Z = {z[z_idx].item():.2f} m\n"
        f"(Preferred Z ~ {z_pref:.2f} m)"
    )
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
