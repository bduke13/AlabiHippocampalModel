# %%
import torch
import numpy as np
import matplotlib.pyplot as plt


class BoundaryVectorCellLayer3D:
    """
    A 3D Boundary Vector Cell (BVC) layer implemented in PyTorch. This class
    models neurons tuned to distance, horizontal angle, and vertical angle
    in a 3D scan. Each neuron's response is computed as the product of:
      - A Gaussian over distance,
      - A wrapped Gaussian over horizontal angle (azimuth),
      - A direct Gaussian over vertical angle (elevation).

    Then, all points in the 3D scan are summed to get each neuron's activation.
    """

    def __init__(
        self,
        max_dist: float,
        n_hd: int,
        preferred_vertical_angles: list = [0],
        sigma_d_list: list = [0.3],
        sigma_ang_list: list = [0.025],
        sigma_vert_list: list = [0.025],
        scaling_factors: list = [1.0],
        num_bvc_per_dir: int = 50,
        num_rows: int = 90,
        num_cols: int = 180,
        top_cutoff_percentage: float = 0.0,
        bottom_cutoff_percentage: float = 0.5,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            max_dist: The maximum radial distance that the BVCs are tuned to.
            n_hd: Number of horizontal directions (azimuth) to distribute BVCs across.
            preferred_vertical_angles: A list of preferred vertical angles (in radians) for groups of BVCs.
            sigma_d_list: A list of distance-tuning sigmas, one per group of BVCs.
            sigma_ang_list: A list of horizontal-angle-tuning sigmas, one per group of BVCs.
            sigma_vert_list: A list of vertical-angle-tuning sigmas, one per group of BVCs.
            scaling_factors: Optional scaling factors for each group (unused in this basic version).
            num_bvc_per_dir: Number of distance tunings per horizontal direction.
            num_rows: Number of vertical “rows” in the input 3D scan.
            num_cols: Number of horizontal “columns” in the input 3D scan.
            top_cutoff_percentage: Fraction of the top rows to ignore.
            bottom_cutoff_percentage: Fraction of the bottom rows to ignore.
            dtype: PyTorch tensor data type.
            device: PyTorch device (e.g., "cpu" or "cuda").
        """

        self.device = device
        self.dtype = dtype
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Prepare distance and horizontal angles
        d_i = np.linspace(0, max_dist, num=num_bvc_per_dir)  # e.g. 50 distances
        N_dist = len(d_i)
        phi_horiz = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        d_i = np.tile(d_i, n_hd)  # shape: (n_hd * num_bvc_per_dir,)

        # We will create one set of BVCs for each preferred_vertical_angle
        # so total BVC count = (n_hd * num_bvc_per_dir) * len(preferred_vertical_angles).
        self.num_bvc = num_bvc_per_dir * n_hd * len(preferred_vertical_angles)

        # Prepare lists to hold repeated parameters for each group
        d_i_all = []
        phi_i_all = []  # horizontal angles
        phi_i_vert_all = []  # vertical angles
        sigma_d_all = []
        sigma_ang_all = []
        sigma_vert_all = []
        scaling_factors_all = []

        # Populate parameter arrays for each vertical angle group
        for idx, vert_angle in enumerate(preferred_vertical_angles):
            num_neurons = len(phi_horiz)
            d_i_all.extend(d_i)
            phi_i_all.extend(phi_horiz)
            phi_i_vert_all.extend([vert_angle] * num_neurons)
            sigma_d_all.extend([sigma_d_list[idx]] * num_neurons)
            sigma_ang_all.extend([sigma_ang_list[idx]] * num_neurons)
            sigma_vert_all.extend([sigma_vert_list[idx]] * num_neurons)
            scaling_factors_all.extend([scaling_factors[idx]] * num_neurons)

        # Convert everything to tensors
        self.d_i = torch.tensor(d_i_all, dtype=dtype, device=device)  # (M,)
        self.phi_i = torch.tensor(phi_i_all, dtype=dtype, device=device)  # (M,)
        self.phi_i_vert = torch.tensor(phi_i_vert_all, dtype=dtype, device=device)
        self.sigma_d = torch.tensor(sigma_d_all, dtype=dtype, device=device)
        self.sigma_ang = torch.tensor(sigma_ang_all, dtype=dtype, device=device)
        self.sigma_vert = torch.tensor(sigma_vert_all, dtype=dtype, device=device)
        self.scaling_factors = torch.tensor(
            scaling_factors_all, dtype=dtype, device=device
        )

        # Precompute denominators and constants
        self.two_sigma_d_squared = 2.0 * (self.sigma_d**2)
        self.two_sigma_ang_squared = 2.0 * (self.sigma_ang**2)
        self.two_sigma_vert_squared = 2.0 * (self.sigma_vert**2)
        self.sqrt_2pi_sigma_d = torch.sqrt(2.0 * np.pi * (self.sigma_d**2))
        self.sqrt_2pi_sigma_ang = torch.sqrt(2.0 * np.pi * (self.sigma_ang**2))
        self.sqrt_2pi_sigma_vert = torch.sqrt(2.0 * np.pi * (self.sigma_vert**2))

        # Prepare row/column cutoffs
        self.top_idx = int(num_rows * top_cutoff_percentage * num_cols)
        self.bottom_idx = int(num_rows * bottom_cutoff_percentage * num_cols)

        # Create a grid of vertical & horizontal angles
        # vertical angles (lat) go from +pi/2 down to -pi/2
        lat_angles = torch.linspace(
            np.pi / 2.0, -np.pi / 2.0, steps=num_rows, dtype=dtype, device=device
        )
        lon_angles = torch.linspace(
            0.0, 2.0 * np.pi, steps=num_cols, dtype=dtype, device=device
        )

        lon_mesh, lat_mesh = torch.meshgrid(lon_angles, lat_angles, indexing="xy")
        # Flatten lat/lon
        lat_flat = lat_mesh.flatten()  # shape: (num_rows * num_cols,)
        lon_flat = lon_mesh.flatten()  # shape: (num_rows * num_cols,)

        # Slice based on top/bottom cutoff
        lat_slice = lat_flat[self.top_idx : self.bottom_idx]  # shape: (N_points,)
        lon_slice = lon_flat[self.top_idx : self.bottom_idx]  # shape: (N_points,)

        # For precomputation, we want shape (N_points, 1) - (1, M) for broadcasting
        lat_slice_2d = lat_slice.unsqueeze(1)  # (N_points, 1)
        lon_slice_2d = lon_slice.unsqueeze(1)  # (N_points, 1)
        phi_i_2d = self.phi_i.unsqueeze(0)  # (1, M)
        phi_i_vert_2d = self.phi_i_vert.unsqueeze(0)  # (1, M)

        # Horizontal angle difference with wrapping
        horiz_diff = torch.atan2(
            torch.sin(lon_slice_2d - phi_i_2d),
            torch.cos(lon_slice_2d - phi_i_2d),
        )  # (N_points, M)
        horiz_gauss = (
            torch.exp(-((horiz_diff**2) / self.two_sigma_ang_squared))
            / self.sqrt_2pi_sigma_ang
        )

        # Vertical angle difference (no wrapping needed for [-pi/2, +pi/2])
        vert_diff = lat_slice_2d - phi_i_vert_2d
        vert_gauss = (
            torch.exp(-((vert_diff**2) / self.two_sigma_vert_squared))
            / self.sqrt_2pi_sigma_vert
        )

        # Combine horizontally and vertically into a single precomputed matrix
        # point_gaussian_precomputed[p, m] = horizontal_gaussian * vertical_gaussian
        self.point_gaussian_precomputed = (
            horiz_gauss * vert_gauss
        )  # shape (N_points, M)

        # Store for later usage
        self.lat_flat = lat_flat
        self.lon_flat = lon_flat
        self.N_points = lat_slice.shape[0]  # # of valid points for the slice

    def get_bvc_activation(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Given a 3D scan (distance array) of shape (num_rows, num_cols), compute
        the activation of each BVC neuron.
        """
        # Flatten scan data
        dist_flat = distances.view(-1)  # shape: (num_rows * num_cols,)
        # Slice according to top and bottom cutoffs
        dist_slice = dist_flat[self.top_idx : self.bottom_idx]  # shape: (N_points,)

        # Expand to shape (N_points, 1) for broadcasting
        dist_slice_2d = dist_slice.unsqueeze(1)  # (N_points, 1)
        d_i_2d = self.d_i.unsqueeze(0)  # (1, M)

        # Distance gaussian: shape (N_points, M)
        delta_d = dist_slice_2d - d_i_2d
        dist_gauss = (
            torch.exp(-(delta_d**2) / self.two_sigma_d_squared) / self.sqrt_2pi_sigma_d
        )

        # Multiply by precomputed angle Gaussians
        # shape (N_points, M)
        combined = dist_gauss * self.point_gaussian_precomputed

        # Sum across all points to get the final activation for each neuron
        bvc_activations = torch.sum(combined, dim=0)  # shape: (M,)

        # Normalize the final activation to [0, 0.5]
        max_val = torch.max(bvc_activations)
        if max_val > 0:
            bvc_activations = bvc_activations / (2.0 * max_val)

        return bvc_activations

    def plot_activation_distribution(
        self, distances: np.ndarray, return_plot: bool = False
    ):
        """
        Plot a histogram of BVC activation values and a sorted line plot, to
        see how strongly the BVCs respond for the given 3D scan.
        """
        # Convert scan data to torch
        distances_t = torch.tensor(distances, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            activations = self.get_bvc_activation(distances_t).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Histogram
        ax1.hist(activations, bins=50, color="skyblue", edgecolor="black")
        ax1.set_title("Distribution of BVC Activations")
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Count")

        # Sorted plot
        sorted_activations = np.sort(activations)
        ax2.plot(sorted_activations, "b-", linewidth=2)
        ax2.set_title("Sorted BVC Activations")
        ax2.set_xlabel("Neuron Index (sorted by activation)")
        ax2.set_ylabel("Activation Value")

        plt.tight_layout()
        if return_plot:
            return fig
        else:
            plt.show()

    def plot_activation(self, distances: torch.Tensor, return_plot: bool = False):
        """
        Create a 3D radial plot of the scan data and overlay BVCs using their
        spherical coordinates (distance, horizontal angle, vertical angle).

        The scan data is drawn as points in 3D space, color-coded by distance,
        and the BVC centers are drawn as scatter points sized by activation.
        """
        # 1) Compute BVC activations
        with torch.no_grad():
            activations = self.get_bvc_activation(distances).cpu().numpy()

        # Avoid zero to keep some minimal marker size
        activations = np.maximum(activations, 1e-3)
        activations_norm = activations / np.max(activations)

        # 2) Prepare the raw scan's 3D points
        dist_flat = distances.view(-1)
        dist_slice = dist_flat[self.top_idx : self.bottom_idx].cpu().numpy()
        lat_slice = self.lat_flat[self.top_idx : self.bottom_idx].cpu().numpy()
        lon_slice = self.lon_flat[self.top_idx : self.bottom_idx].cpu().numpy()

        # Filter out invalid distances
        valid_mask = ~(np.isnan(dist_slice) | np.isinf(dist_slice) | (dist_slice <= 0))
        dist_slice = dist_slice[valid_mask]
        lat_slice = lat_slice[valid_mask]
        lon_slice = lon_slice[valid_mask]

        if dist_slice.size == 0:
            print("Warning: No valid data points to plot!")
            return None

        # Convert raw scan from spherical to Cartesian
        #   r = dist, θ=lon (azimuth), φ=lat (elevation)
        x_points = dist_slice * np.cos(lat_slice) * np.cos(lon_slice)
        y_points = dist_slice * np.cos(lat_slice) * np.sin(lon_slice)
        z_points = dist_slice * np.sin(lat_slice)

        # 3) Make a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter the scan points colored by distance
        scatter = ax.scatter(
            x_points, y_points, z_points, c=dist_slice, cmap="viridis", s=10, alpha=0.6
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Distance (m)")

        # Add the scanner position at origin
        ax.scatter(0, 0, 0, color="red", s=100, marker="o", label="Scanner")

        # Optionally, draw a faint wireframe sphere
        r = dist_slice.max() * 0.5
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = r * np.outer(np.cos(u), np.sin(v))
        y_sphere = r * np.outer(np.sin(u), np.sin(v))
        z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(
            x_sphere, y_sphere, z_sphere, color="gray", alpha=0.2, linewidth=0.5
        )

        # 4) Plot BVC centers, sized by activation
        d_i_np = self.d_i.cpu().numpy()
        phi_i_np = self.phi_i.cpu().numpy()
        phi_i_vert_np = self.phi_i_vert.cpu().numpy()

        # Convert each BVC's spherical coords to Cartesian
        x_bvc = d_i_np * np.cos(phi_i_vert_np) * np.cos(phi_i_np)
        y_bvc = d_i_np * np.cos(phi_i_vert_np) * np.sin(phi_i_np)
        z_bvc = d_i_np * np.sin(phi_i_vert_np)

        # Size them by activation
        sizes = activations_norm * 100.0  # scale factor for marker size
        # For color, let's just use one color or you can map them
        ax.scatter(
            x_bvc,
            y_bvc,
            z_bvc,
            s=sizes,
            c="r",
            alpha=0.4,
            edgecolor="black",
            label="BVC Centers",
        )

        # Set labels and adjust aspect ratio
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Scan + BVC Activation")

        # Make the plot roughly cubic
        max_range = (
            np.array(
                [
                    x_points.max() - x_points.min(),
                    y_points.max() - y_points.min(),
                    z_points.max() - z_points.min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (x_points.max() + x_points.min()) / 2.0
        mid_y = (y_points.max() + y_points.min()) / 2.0
        mid_z = (z_points.max() + z_points.min()) / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()

        if return_plot:
            return fig
        else:
            plt.show()

    def generate_cell_names(self, output_file="cell_names.txt"):
        """
        Write out each BVC's name and preferred parameters (horizontal angle, vertical angle, distance)
        to a file for inspection/debugging.
        """
        phi_i_np = self.phi_i.cpu().numpy()
        phi_i_vert_np = self.phi_i_vert.cpu().numpy()
        d_i_np = self.d_i.cpu().numpy()

        with open(output_file, "w") as f:
            for i in range(self.num_bvc):
                horiz_angle_deg = (np.degrees(phi_i_np[i])) % 360
                vert_angle_deg = np.degrees(phi_i_vert_np[i])
                distance = d_i_np[i]
                name = (
                    f"BVC_{i}_Horiz{horiz_angle_deg:.1f}_"
                    f"Vert{vert_angle_deg:.1f}_Dist{distance:.1f}"
                )
                f.write(name + "\n")

        print(f"Cell names saved to {output_file}")


# ---------------------------
# Example usage / test
# ---------------------------
if __name__ == "__main__":
    # Suppose we have a 90x180 3D scan (vertical_boundaries)
    # For demonstration, we'll construct synthetic data:
    num_rows, num_cols = 90, 180
    radius_min, radius_max = 1.0, 10.0
    vertical_boundaries = np.random.uniform(
        radius_min, radius_max, (num_rows, num_cols)
    )

    preferred_vertical_angles = [0, 0.3, 0.6]  # in radians
    sigma_d_list = [0.2, 0.2, 0.2]
    sigma_ang_list = [0.025, 0.05, 0.05]
    sigma_vert_list = [0.025, 0.1, 0.1]
    scaling_factors = [1.0, 0.5, 0.1]

    # Create BVC layer
    bvc_layer_3d = BoundaryVectorCellLayer3D(
        max_dist=12.0,
        n_hd=8,
        preferred_vertical_angles=preferred_vertical_angles,
        sigma_d_list=sigma_d_list,
        sigma_ang_list=sigma_ang_list,
        sigma_vert_list=sigma_vert_list,
        scaling_factors=scaling_factors,
        num_bvc_per_dir=50,
        num_rows=num_rows,
        num_cols=num_cols,
        top_cutoff_percentage=0.0,
        bottom_cutoff_percentage=0.5,
        device="cpu",
    )

    # Plot activation distribution
    bvc_layer_3d.plot_activation_distribution(vertical_boundaries)

    # Plot the 3D radial scan + BVC centers
    bvc_layer_3d.plot_activation(
        torch.tensor(vertical_boundaries, dtype=torch.float32), return_plot=False
    )
