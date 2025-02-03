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

    def get_hd_activation(self, theta_0: float, v_in: np.ndarray):
        """Computes the activation of head direction cells based on current heading.

        Calculates how much the agent's current heading aligns with each cell's
        preferred direction using dot product computation.

        Args:
            theta_0: Heading angle of the anchor cue in radians (default is 0).
            v_in: Vector representing the current heading direction of the agent.

        Returns:
            Activation levels of head direction cells as numpy array. Each value
            indicates alignment between current heading and cell's preferred direction.
        """
        # Create equally spaced angles for HD cells
        theta_i = np.arange(0, 2 * np.pi, 2 * np.pi / self.num_cells)

        # Generate tuning kernel for directions
        D = np.stack([np.cos(theta_i + theta_0), np.sin(theta_i + theta_0)], axis=0)

        # Compute dot product between current heading and preferred directions
        activation = np.dot(v_in, D)

        self.state = activation

        # Shape: (self.num_cells,)
        return activation

    def plot_activation(self, plot_type: str = "bar", return_plot: bool = False):
        """Plots the activation levels of head direction cells.

        Args:
            plot_type: Type of plot to create ('bar' or 'radial').
            return_plot: If True, returns the plot object instead of displaying.

        Returns:
            plt.Figure if return_plot is True, otherwise None.

        Raises:
            ValueError: If activation state is not set or plot_type is invalid.
        """
        if self.state is None or not np.any(self.state):
            raise ValueError(
                "Activation state is not set. Please call 'get_hd_activation' first to compute activations."
            )

        # Create the labels for each head direction cell based on the evenly spaced angles
        categories = [
            f"{int(round(np.rad2deg(angle)))}°"
            for angle in np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
        ]

        if plot_type == "bar":
            fig, ax = plt.subplots()
            ax.bar(categories, self.state)
            ax.set_xlabel("Head Direction Cells (Degrees)")
            ax.set_ylabel("Activation Magnitude")
            ax.set_title("Head Direction Layer Activation")
            plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        elif plot_type == "radial":
            angles = np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
            r = self.state
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

        # Return the figure object if requested, otherwise show the plot
        if return_plot:
            return fig
        else:
            plt.show()
