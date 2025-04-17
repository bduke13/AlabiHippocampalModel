# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch._prims_common import dtype_or_default


class HeadDirectionLayer:
    def __init__(
        self,
        num_cells: int,
        theta_0: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes a layer of head direction cells using PyTorch.

        Creates a layer of cells that represent the agent's heading in a discretized way.
        Each cell has a preferred direction, with activations based on the proximity
        of the agent's current heading to the cell's preferred direction.

        Args:
            num_cells: Number of head direction cells, evenly spaced across 360 degrees.
            theta_0: Optional anchor angle in radians to offset all preferred directions.
        """
        self.num_cells = num_cells
        self.theta_0 = theta_0
        self.dtype = dtype
        self.device = device

        # Create equally spaced angles [0, 2π) for HD cells
        # shape: (num_cells,)
        theta_i = torch.linspace(
            start=0,
            end=2 * torch.pi,
            steps=self.num_cells + 1,
            dtype=dtype,
            device=device,
        )[:-1]

        # Stack cos and sin into shape (num_cells, 2)
        # Each row = [cos(θ_i + θ0), sin(θ_i + θ0)]
        self.tuning_kernel = torch.stack(
            [
                torch.cos(theta_i + self.theta_0),
                torch.sin(theta_i + self.theta_0),
            ],
            dim=1,
        ).to(dtype=dtype, device=device)

        # Will hold the most recent activation vector of shape (num_cells,)
        self.hd_activations = None

    def get_hd_activation(self, v_in: np.ndarray) -> torch.Tensor:
        """Computes the activation of head direction cells given a 2D heading vector.

        The activation is calculated via dot product between each cell's preferred
        direction and the current heading vector.

        Args:
            v_in: A 2-element torch.Tensor representing the current heading direction (e.g., [cos θ, sin θ]).

        Returns:
            A 1D torch.Tensor of shape (num_cells,) with activation values.
        """
        if isinstance(v_in, torch.Tensor):
            v_in = v_in.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            v_in = torch.tensor(data=v_in, dtype=self.dtype, device=self.device)

        if v_in.shape != (2,):
            raise ValueError(
                "v_in should be a 2-element torch.Tensor, e.g. [cosθ, sinθ]."
            )

        activation = torch.matmul(self.tuning_kernel, v_in)
        self.hd_activations = activation
        return activation

    def plot_activation(self, plot_type: str = "bar", return_plot: bool = False):
        """Plots the activation levels of head direction cells.

        Converts self.hd_activations to a NumPy array for plotting.

        Args:
            plot_type: Type of plot to create ('bar' or 'radial').
            return_plot: If True, returns the plot object instead of displaying it.

        Returns:
            The matplotlib Figure if return_plot is True, otherwise None.

        Raises:
            ValueError: If activation state is not set or plot_type is invalid.
        """
        if self.hd_activations is None:
            raise ValueError(
                "Activation state is not set. Please call 'get_hd_activation' first."
            )

        # Check if there's any nonzero activation
        if not torch.any(self.hd_activations):
            raise ValueError(
                "Current activation is zero or empty. Did you provide a valid heading vector?"
            )

        # Convert to CPU NumPy for plotting
        activations_np = self.hd_activations.detach().cpu().numpy()

        # Create the labels (e.g., 0°, 45°, 90°, etc.) for each head direction cell
        categories = [
            f"{int(round(np.rad2deg(angle)))}°"
            for angle in np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
        ]

        if plot_type == "bar":
            fig, ax = plt.subplots()
            ax.bar(categories, activations_np)
            ax.set_xlabel("Head Direction Cells (Degrees)")
            ax.set_ylabel("Activation")
            ax.set_title("Head Direction Layer Activation")
            plt.xticks(rotation=45, ha="right")

        elif plot_type == "radial":
            # For radial, we need angles from 0 to 2π
            angles = np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.bar(
                angles, activations_np, width=2 * np.pi / self.num_cells, align="center"
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(categories)
            ax.set_title("Head Direction Layer Activation")

        else:
            raise ValueError("Invalid plot_type. Choose 'bar' or 'radial'.")

        plt.tight_layout()
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    # 1. Create a layer with 8 direction cells
    hd_layer = HeadDirectionLayer(num_cells=8, theta_0=0.0)

    # 2. Suppose current heading vector is [cos(45°), sin(45°)] = [√2/2, √2/2]
    heading_vec = torch.tensor(
        [np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=torch.float32
    )

    # 3. Compute the activations
    activations = hd_layer.get_hd_activation(heading_vec)
    print("HD Activations:", activations)

    # 4. Plot them in radial form
    hd_layer.plot_activation(plot_type="bar")
