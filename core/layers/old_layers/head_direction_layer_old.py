import numpy as np
import matplotlib.pyplot as plt


class HeadDirectionLayer:
    def __init__(self, num_cells: int) -> None:
        """Initializes a layer of head direction cells.

        Creates a layer of cells that represent the agent's heading in a discretized way.
        Each cell has a preferred direction, with activations based on the proximity
        of the agent's current heading to the cell's preferred direction.

        Args:
            num_cells: Number of head direction cells, evenly spaced across 360 degrees.
        """
        self.num_cells = num_cells
        self.hd_activations = None  # Will store the latest activation

    def get_hd_activation(self, theta_0: float, v_in: np.ndarray):
        """Computes the activation of head direction cells based on current heading.

        Args:
            theta_0: Heading angle offset (in radians).
            v_in: 1D array of length 2, representing the current heading (e.g. [cos θ, sin θ]).

        Returns:
            A 1D NumPy array of shape (num_cells,) containing the activation levels of
            each head-direction cell.
        """
        # This creates exactly 'num_cells' angles evenly spaced from 0 to 2π (exclusive).
        theta_i = np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)

        # Create the tuning kernel: shape (2, num_cells)
        # First row: cos(θ_i + θ_0), Second row: sin(θ_i + θ_0)
        D = np.stack([np.cos(theta_i + theta_0), np.sin(theta_i + theta_0)], axis=0)
        # D.shape == (2, num_cells)

        # v_in should be shape (2,). Dot product => shape (num_cells,)
        # np.dot((2,), (2, num_cells)) => ERROR. We need np.dot((2,), (2, num_cells)) to broadcast properly.
        # Actually we want v_in dot D => shape (num_cells,). We can do:
        #  activation = v_in.dot(D), but we must transpose D or rearrange arguments:
        #   - np.dot(v_in, D) => shape (num_cells,) only if D is shape (2, num_cells)
        activation = np.dot(v_in, D)

        self.hd_activations = activation
        return activation

    def plot_activation(self, plot_type: str = "bar", return_plot: bool = False):
        """Plots the activation levels of head direction cells.

        Args:
            plot_type: Type of plot to create ('bar' or 'radial').
            return_plot: If True, returns the figure object instead of displaying it.

        Returns:
            A matplotlib Figure if return_plot is True, otherwise None.

        Raises:
            ValueError: If activation state is not set or plot_type is invalid.
        """
        if self.hd_activations is None or not np.any(self.hd_activations):
            raise ValueError(
                "Activation state is not set. Please call 'get_hd_activation' first."
            )

        # Create labels for each head direction cell (e.g. "0°, 45°, 90°...", etc.)
        categories = [
            f"{int(round(np.rad2deg(angle)))}°"
            for angle in np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
        ]

        if plot_type == "bar":
            fig, ax = plt.subplots()
            ax.bar(categories, self.hd_activations)
            ax.set_xlabel("Head Direction Cells (Degrees)")
            ax.set_ylabel("Activation")
            ax.set_title("Head Direction Layer Activation")
            plt.xticks(rotation=45, ha="right")

        elif plot_type == "radial":
            angles = np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
            r = self.hd_activations
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.bar(
                angles, r, width=2 * np.pi / self.num_cells, bottom=0.0, align="center"
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(categories)
            plt.title("Head Direction Layer Activation")

        else:
            raise ValueError("Invalid plot_type. Choose 'bar' or 'radial'.")

        plt.tight_layout()
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # 1. Create a layer with 8 direction cells
    hd_layer = HeadDirectionLayer(num_cells=8)

    # 2. Suppose our offset (theta_0) is 0, and heading vector is [cos(45°), sin(45°)]
    theta_0 = 0.0
    heading_vec = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])

    # 3. Compute the activations
    activations = hd_layer.get_hd_activation(theta_0, heading_vec)
    print("HD Activations:", activations)

    # 4. Plot them in radial form
    hd_layer.plot_activation(plot_type="bar")
