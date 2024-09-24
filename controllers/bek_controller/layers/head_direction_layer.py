import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class HeadDirectionLayer:
    def __init__(self, num_cells: int) -> None:
        """
        Initializes a layer of head direction cells that represent the agent's heading in a discretized way.
        Each head direction cell has a preferred direction in space, and the activation of each cell is
        determined based on how close the agent's current heading is to the cell's preferred direction.

        The layer uses 'num_cells' head direction cells, evenly spaced across 360 degrees, and the activation
        of each cell is calculated using a dot product between the agent's current heading direction and the 
        cell's preferred direction.

        Parameters:
        - num_cells (int): The number of head direction cells in the layer, each representing a unique direction.
        """
        self.num_cells = num_cells

    def get_hd_activation(self, theta_0: float, v_in: np.ndarray):
        """
        Computes the activation of the head direction cells based on the agent's current heading direction. 

        Each head direction cell has a preferred direction, and this method calculates how much the agent's 
        current heading (represented by 'v_in') aligns with each of the head direction cells' preferred directions.

        The preferred directions of the head direction cells are evenly spaced across 360 degrees, and the 
        activation is calculated using the dot product of the current heading vector ('v_in') and the preferred 
        directions of the head direction cells.

        Parameters:
        - theta_0 (float): Heading angle of the anchor cue. Default is 0 degrees.
        - v_in (np.ndarray): A vector representing the current heading direction of the agent.

        Returns:
        - np.ndarray: The activation levels of the head direction cells, where each activation indicates how closely 
          the agent's current heading aligns with the preferred direction of each head direction cell.
        """
        theta_i = np.arange(0, 2 * np.pi, 2 * np.pi / self.num_cells)  # Create equally spaced angles for the head direction cells
        D = np.stack([np.cos(theta_i + theta_0), np.sin(theta_i + theta_0)], axis=0)  # Generate tuning kernel for directions
        
        # Compute the dot product of the input vector (current heading) with the preferred directions of the cells
        activation = np.dot(v_in, D)

        self.state = activation

        # Shape: (self.num_cells,)
        return activation


    def plot_activation(self, plot_type: str = "bar", return_plot: bool = False):
        """
        Plot the activation levels of each head direction cell.

        Parameters:
        - plot_type (str): The type of plot ('bar' for bar chart or 'radial' for radial plot).
        - return_plot (bool): If True, returns the plot object instead of showing it.

        Returns:
        - None or plt.Figure: If return_plot=True, returns the figure object. Otherwise, it shows the plot.
        """
        if self.state is None or not np.any(self.state):
            raise ValueError(
                "Activation state is not set. Please call 'get_hd_activation' first to compute activations."
            )

        # Create the labels for each head direction cell based on the evenly spaced angles
        categories = [
            f"{int(round(np.rad2deg(angle)))}°" for angle in np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
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
