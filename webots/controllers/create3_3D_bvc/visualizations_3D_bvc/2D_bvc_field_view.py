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


def plot_bvc_response(sigma_theta, sigma_r, bvc_idx=3, device="cpu"):
    """Plot BVC response field for a given sigma_theta and sigma_r."""
    # Initialize BVC layer with the specified parameters
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=2,
        n_res=720,
        n_hd=8,
        sigma_theta=sigma_theta,
        sigma_r=sigma_r,
        num_bvc_per_dir=2,
        device=device,
    )

    # Select the BVC
    selected_distance = bvc_layer.d_i[0, bvc_idx].item()
    selected_angle = bvc_layer.phi_i[0, bvc_idx].item()

    # Create grid of points
    resolution = 200
    x = torch.linspace(-5, 5, resolution, device=device)
    y = torch.linspace(-5, 5, resolution, device=device)
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
        f"BVC Response Field\nσ_θ: {sigma_theta}°, σ_r: {sigma_r}m\n"
        f"Preferred Distance: {selected_distance:.1f}m, Angle: {np.rad2deg(selected_angle):.1f}°"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    return plt.gcf()


def main():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define parameter combinations
    sigma_thetas = [1, 10, 50]
    sigma_rs = [0.25, 1, 2.5]

    # Create a grid of subplots
    fig, axes = plt.subplots(len(sigma_rs), len(sigma_thetas), figsize=(16, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # BVC index to use consistently across all plots
    bvc_idx = 3

    # Initialize a sample BVC layer to get the selected BVC's properties
    sample_bvc_layer = BoundaryVectorCellLayer(
        max_dist=2,
        n_res=720,
        n_hd=8,
        sigma_theta=1,
        sigma_r=0.25,
        num_bvc_per_dir=2,
        device=device,
    )
    selected_distance = sample_bvc_layer.d_i[0, bvc_idx].item()
    selected_angle = sample_bvc_layer.phi_i[0, bvc_idx].item()

    print(f"num_bvcs: {sample_bvc_layer.num_bvc}")
    print(
        f"Selected BVC - Distance: {selected_distance:.2f}m, Angle: {selected_angle:.2f}rad ({np.rad2deg(selected_angle):.1f}°)"
    )

    # Loop through all parameter combinations
    for i, sigma_r in enumerate(sigma_rs):
        for j, sigma_theta in enumerate(sigma_thetas):
            print(f"Plotting for σ_θ: {sigma_theta}°, σ_r: {sigma_r}m")

            # Create grid of points
            resolution = 200
            x = torch.linspace(-5, 5, resolution, device=device)
            y = torch.linspace(-5, 5, resolution, device=device)
            X, Y = torch.meshgrid(x, y, indexing="xy")

            # Initialize BVC layer with current parameters
            bvc_layer = BoundaryVectorCellLayer(
                max_dist=2,
                n_res=720,
                n_hd=8,
                sigma_theta=sigma_theta,
                sigma_r=sigma_r,
                num_bvc_per_dir=2,
                device=device,
            )

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

            # Plot on the corresponding subplot
            ax = axes[i, j]
            im = ax.pcolormesh(
                X.cpu(), Y.cpu(), response.cpu(), shading="auto", cmap="viridis"
            )

            # Plot the agent (origin)
            ax.plot(0, 0, "ro", markersize=6, label="Agent")

            # Plot the BVC's preferred location
            x_pref = selected_distance * np.cos(selected_angle)
            y_pref = selected_distance * np.sin(selected_angle)
            ax.plot(x_pref, y_pref, "r*", markersize=10, label="BVC Preferred")

            # Plot a line from origin to preferred location
            ax.plot([0, x_pref], [0, y_pref], "r--", alpha=0.5)

            ax.set_title(f"σ_θ: {sigma_theta}°, σ_r: {sigma_r}m")
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.grid(True)

            # Add colorbar
            plt.colorbar(im, ax=ax, label="Response")

    # Add a common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2)

    # Add overall title
    fig.suptitle(
        f"BVC Response Fields for Different Parameter Combinations\n"
        f"Preferred Distance: {selected_distance:.1f}m, Angle: {np.rad2deg(selected_angle):.1f}°",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Also create individual plots for each combination
    print("\nGenerating individual plots for each parameter combination...")
    for sigma_theta in sigma_thetas:
        for sigma_r in sigma_rs:
            fig = plot_bvc_response(sigma_theta, sigma_r, bvc_idx, device)
            plt.savefig(
                f"bvc_response_theta{sigma_theta}_r{sigma_r}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
