# %%
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer


def compute_bvc_response_at_point(
    bvc_distance, bvc_angle, point_x, point_y, sigma_r, sigma_theta
):
    """Compute BVC response for a boundary point at (x,y) relative to agent at origin."""
    # Convert point to polar coordinates
    r = torch.sqrt(point_x**2 + point_y**2)
    theta = torch.atan2(point_y, point_x)

    # Wrap angle difference to [-π, π]
    angle_diff = torch.remainder(theta - bvc_angle + torch.pi, 2 * torch.pi) - torch.pi

    # Compute distance and angular components using Gaussian functions
    distance_component = torch.exp(
        -((r - bvc_distance) ** 2) / (2 * sigma_r**2)
    ) / torch.sqrt(2 * torch.pi * sigma_r**2)
    angular_component = torch.exp(
        -(angle_diff**2) / (2 * sigma_theta**2)
    ) / torch.sqrt(2 * torch.pi * sigma_theta**2)

    return distance_component * angular_component


def main():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize BVC layer
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        n_res=720,
        n_hd=8,
        sigma_theta=10,
        sigma_r=0.5,
        num_bvc_per_dir=50,
        device=device,
    )

    # Randomly select a BVC
    bvc_idx = np.random.randint(0, bvc_layer.num_bvc)
    selected_distance = bvc_layer.d_i[0, bvc_idx].item()
    selected_angle = bvc_layer.phi_i[0, bvc_idx].item()

    print(
        f"Selected BVC - Distance: {selected_distance:.2f}m, Angle: {selected_angle:.2f}rad ({np.rad2deg(selected_angle):.1f}°)"
    )

    # Create grid of points
    resolution = 100
    x = torch.linspace(-15, 15, resolution, device=device)
    y = torch.linspace(-15, 15, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Compute BVC response for each point
    response = compute_bvc_response_at_point(
        selected_distance,
        selected_angle,
        X,
        Y,
        bvc_layer.sigma_r,
        bvc_layer.sigma_theta,
    )

    # Normalize response to [0, 1]
    response = response / response.max()

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Plot the response field
    plt.pcolormesh(X.cpu(), Y.cpu(), response.cpu(), shading="auto", cmap="viridis")
    plt.colorbar(label="BVC Response")

    # Plot the agent (origin)
    plt.plot(0, 0, "ro", markersize=10, label="Agent")

    # Plot the BVC's preferred location
    x_pref = selected_distance * np.cos(selected_angle)
    y_pref = selected_distance * np.sin(selected_angle)
    plt.plot(x_pref, y_pref, "r*", markersize=15, label="BVC Preferred Location")

    # Plot a line from origin to preferred location
    plt.plot([0, x_pref], [0, y_pref], "r--", alpha=0.5)

    plt.title(
        f"BVC Response Field\nPreferred Distance: {selected_distance:.1f}m, Angle: {np.rad2deg(selected_angle):.1f}°"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
