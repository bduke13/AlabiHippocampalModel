# %%
import torch
import numpy as np
import matplotlib.pyplot as plt


class BoundaryVectorCellLayer:
    """Implementation of the BVC model based on the given equation from the paper.
    Each BVC is tuned to a preferred distance and angle, with activations modeled
    as the sum of the product of two Gaussian receptive fields (distance and angle)
    for each lidar point compared to the BVC's tuned parameters.

    The original equation can be found in Ade's Disseration describing this layer in Equation 3.1 on page 43.
    """

    def __init__(
        self,
        n_res: int,  # LiDAR resolution (e.g., 720)
        n_hd: int,  # Number of head directions
        sigma_theta: float,  # Standard deviation for angular tuning (degrees)
        sigma_r: float,  # Standard deviation for distance tuning
        max_dist: float,  # Maximum response distance
        num_bvc_per_dir: int = 50,  # Number of BVCs per head direction
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        Args:
            max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
            n_res: Size of input vector to the BVC layer (e.g., 720 for RPLidar).
            n_hd: Number of head direction cells, each representing a preferred angular direction for the BVCs.
            sigma_theta: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
            sigma_r: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
            dtype: PyTorch data type (e.g., torch.float32).
            device: The computational device to use (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.dtype = dtype
        self.n_res = n_res

        # Convert sigma_theta from degrees to radians
        self.sigma_theta = torch.tensor(
            np.deg2rad(sigma_theta), dtype=dtype, device=device
        )
        self.sigma_r = torch.tensor(sigma_r, dtype=dtype, device=device)

        # Create LiDAR angles: 0 to 2π in n_res steps
        self.lidar_angles = torch.linspace(
            0, 2 * torch.pi, steps=n_res, dtype=dtype, device=device
        )

        # Create preferred distances per BVC
        tuned_dist = torch.linspace(
            0, max_dist, steps=num_bvc_per_dir, dtype=dtype, device=device
        )
        N_dist = len(tuned_dist)

        # Generate preferred angles (head directions)
        # Using n_hd+1 steps and removing the last point to avoid duplicate at 2π
        preferred_angles = torch.linspace(
            0, 2 * torch.pi, steps=n_hd + 1, dtype=dtype, device=device
        )[:-1]

        # Tile distances and angles for matrix computation
        self.d_i = tuned_dist.repeat(n_hd).unsqueeze(0)  # (1, N_dist * n_hd)
        self.phi_i = preferred_angles.repeat_interleave(N_dist).unsqueeze(
            0
        )  # (1, N_dist * n_hd)

        self.num_bvc = self.d_i.numel()

        # Compute Angular Gaussian Matrix (n_res, n_res)
        lidar_angles_expanded = self.lidar_angles.unsqueeze(0)  # (1, n_res)
        phi_i_expanded = self.phi_i.T  # (n_res, 1)
        angular_diff = torch.remainder(
            torch.abs(lidar_angles_expanded - phi_i_expanded), 2 * torch.pi
        )
        angular_diff = torch.minimum(
            angular_diff, 2 * torch.pi - angular_diff
        )  # Ensure shortest difference

        self.angular_gaussian_matrix = torch.exp(
            -(angular_diff**2) / (2 * self.sigma_theta**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_theta**2)

        self.bvc_activations = None

    def get_bvc_activation(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute the activation of the BVC neurons given LiDAR distance readings."""
        # Expand distance tensor for broadcasting
        distances_expanded = distances.unsqueeze(0)  # (1, n_res)
        d_i_expanded = self.d_i.T  # (n_res, 1)

        # Compute Distance Gaussian Matrix (n_res, n_res)
        distance_gaussian_matrix = torch.exp(
            -((distances_expanded - d_i_expanded) ** 2) / (2 * self.sigma_r**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_r**2)

        # Element-wise multiplication and sum along angle axis
        bvc_pre_sum_activations = (
            self.angular_gaussian_matrix * distance_gaussian_matrix
        )
        bvc_activations = torch.sum(bvc_pre_sum_activations, dim=1)

        # # **Normalize activations to [0, 0.5]**
        # max_activation = torch.max(bvc_activations)
        # if max_activation > 0:
        #     bvc_activations /= 2.0 * max_activation
        bvc_activations = bvc_activations / self.num_bvc

        self.bvc_activations = bvc_activations
        return bvc_activations

    def plot_activation(self, distances: np.ndarray) -> None:
        """Visualizes the BVC activations in a polar plot and overlays the boundary (LiDAR readings)."""
        distances_tensor = torch.tensor(distances, dtype=self.dtype, device=self.device)
        activations = self.get_bvc_activation(distances_tensor).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        # Scatter plot for BVC activations (Color and Size reflect activation strength)
        ax.scatter(
            self.phi_i.cpu().numpy(),
            self.d_i.cpu().numpy(),
            c=activations,
            cmap="viridis",
            s=activations * 150,
            alpha=0.75,
        )

        # Overlay boundary (LiDAR readings)
        angles_np = self.lidar_angles.detach().cpu().numpy()
        ax.plot(
            angles_np, distances, "r-", linewidth=2, label="Boundary (LiDAR readings)"
        )

        ax.set_title("BVC Activation Heatmap with Boundary")
        ax.legend()
        plt.show()

    def plot_activation_histogram(self, distances: np.ndarray, bins: int = 30) -> None:
        """Plots a histogram of BVC activations to visualize their distribution.

        Args:
            distances: LiDAR distance readings as a numpy array.
            bins: Number of histogram bins to use (default: 30).
        """
        # Get activations
        distances_tensor = torch.tensor(distances, dtype=self.dtype, device=self.device)
        activations = self.get_bvc_activation(distances_tensor).detach().cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        counts, edges, bars = ax.hist(
            activations, bins=bins, alpha=0.7, color="steelblue", edgecolor="black"
        )

        # Add vertical line for mean
        mean_activation = np.mean(activations)
        ax.axvline(
            mean_activation,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_activation:.4f}",
        )

        # Add vertical line for median
        median_activation = np.median(activations)
        ax.axvline(
            median_activation,
            color="green",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median_activation:.4f}",
        )

        # Add statistics as text
        stats_text = (
            f"Min: {np.min(activations):.4f}\n"
            f"Max: {np.max(activations):.4f}\n"
            f"Mean: {mean_activation:.4f}\n"
            f"Median: {median_activation:.4f}\n"
            f"Std Dev: {np.std(activations):.4f}\n"
            f"Active cells: {np.sum(activations > 0.01)}/{len(activations)}"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set labels and title
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Number of Cells")
        ax.set_title("Histogram of BVC Activations")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_kernel_heatmaps(self, distances: np.ndarray) -> None:
        """Visualizes the angular kernel, distance kernel, and pre-sum activations as heatmaps.

        Args:
            distances: LiDAR distance readings as a numpy array.
        """
        # Convert distances to tensor
        distances_tensor = torch.tensor(distances, dtype=self.dtype, device=self.device)

        # Compute distance Gaussian matrix
        distances_expanded = distances_tensor.unsqueeze(0)  # (1, n_res)
        d_i_expanded = self.d_i.T  # (n_res, 1)
        distance_gaussian_matrix = torch.exp(
            -((distances_expanded - d_i_expanded) ** 2) / (2 * self.sigma_r**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_r**2)

        # Compute pre-sum activations
        bvc_pre_sum_activations = (
            self.angular_gaussian_matrix * distance_gaussian_matrix
        )

        # Convert to numpy for plotting
        angular_matrix_np = self.angular_gaussian_matrix.detach().cpu().numpy()
        distance_matrix_np = distance_gaussian_matrix.detach().cpu().numpy()
        pre_sum_activations_np = bvc_pre_sum_activations.detach().cpu().numpy()

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # Plot angular Gaussian matrix
        im0 = axes[0].imshow(
            angular_matrix_np, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        axes[0].set_title("Angular Gaussian Matrix (BVCs × LiDAR angles)")
        axes[0].set_xlabel("LiDAR Angle Index (0 to 2π)")
        axes[0].set_ylabel("BVC Index")
        fig.colorbar(im0, ax=axes[0], label="Angular Tuning Value")

        # Plot distance Gaussian matrix
        im1 = axes[1].imshow(
            distance_matrix_np, aspect="auto", cmap="plasma", interpolation="nearest"
        )
        axes[1].set_title("Distance Gaussian Matrix (BVCs × LiDAR distances)")
        axes[1].set_xlabel("LiDAR Distance Index")
        axes[1].set_ylabel("BVC Index")
        fig.colorbar(im1, ax=axes[1], label="Distance Tuning Value")

        # Plot pre-sum activations
        im2 = axes[2].imshow(
            pre_sum_activations_np,
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
        )
        axes[2].set_title("Pre-Sum Activations (Angular × Distance)")
        axes[2].set_xlabel("LiDAR Reading Index")
        axes[2].set_ylabel("BVC Index")
        fig.colorbar(im2, ax=axes[2], label="Activation Value")

        # Add annotations for matrix shapes
        axes[0].text(
            0.02,
            0.98,
            f"Shape: {angular_matrix_np.shape}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        axes[1].text(
            0.02,
            0.98,
            f"Shape: {distance_matrix_np.shape}",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        axes[2].text(
            0.02,
            0.98,
            f"Shape: {pre_sum_activations_np.shape}",
            transform=axes[2].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Add explanation of what we're seeing
        explanation = (
            "These heatmaps show how BVC activations are computed:\n"
            "1. Angular Matrix: How each BVC responds to different angles\n"
            "2. Distance Matrix: How each BVC responds to different distances\n"
            "3. Pre-Sum Activations: Element-wise product of angular and distance matrices\n"
            "Final BVC activations are computed by summing across each row of the pre-sum matrix"
        )
        fig.text(
            0.5,
            0.01,
            explanation,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()


# Example Usage
if __name__ == "__main__":
    n_points = 720  # LiDAR resolution
    max_r = 10
    min_r = 5
    n_star_peaks = 7

    # Create star-shaped LiDAR distance pattern
    distances = np.ones(n_points) * min_r
    star_interval = n_points // (n_star_peaks * 2)
    for i in range(0, n_star_peaks * 2, 2):
        start_idx = i * star_interval
        distances[start_idx : start_idx + star_interval] = max_r

    # Initialize BVC layer
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        n_res=n_points,
        n_hd=8,
        sigma_theta=90,  # Angular tuning width
        sigma_r=1,  # Distance tuning width
        num_bvc_per_dir=50,
    )

    # Plot activation with boundary
    bvc_layer.plot_activation(distances)

    # Plot histogram of activations
    bvc_layer.plot_activation_histogram(distances)

    # Plot kernel heatmaps
    bvc_layer.plot_kernel_heatmaps(distances)