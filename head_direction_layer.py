import numpy as np
import matplotlib.pyplot as plt


class HeadDirectionLayer:
    def __init__(
        self, num_cells: int, theta_0: float = 0.0, unit: str = "degree"
    ) -> None:
        """
        Initialize the HeadDirectionLayer, which models a discretized representation
        of heading that can be used by a learning model.

        This model creates a specified number of head direction cells, each corresponding
        to a unique, evenly spaced direction in 360 degrees. The activation of each cell
        is determined by the dot product between the agent's current heading and the
        cell's preferred (tuned) direction.

        A cell will have an activation of 1 when the agent's heading exactly matches its
        preferred direction, with neighboring cells showing decreasing activations
        down to 0 as the heading deviates.

        Parameters:
        - num_cells (int): The number of head direction cells in the layer.
        - theta_0 (float): The reference direction (in degrees or radians) for the anchor
          cue, defaulting to 0.0 (North).
        - unit (str): The unit of the input direction, either 'degree' or 'radian'.
          Default is 'degree'.
        """
        self.num_cells = num_cells  # Number of head direction cells
        self.unit = unit  # Unit of input direction ('degree' or 'radian')
        self.theta_0 = self._convert_to_radians(
            theta_0
        )  # Convert the base direction to radians if necessary
        self.tuned_directions = (
            self._generate_tuned_directions()
        )  # Generate and store the preferred directions
        self.state = np.zeros(
            num_cells
        )  # Initialize the activation state of the head direction cells to zeros

    def _convert_to_radians(self, value: float) -> float:
        """
        Convert the input direction to radians if the unit is in degrees.

        Parameters:
        - value (float): The input direction angle.

        Returns:
        - float: The direction in radians.
        """
        return np.deg2rad(value) if self.unit == "degree" else value

    def _generate_tuned_directions(self) -> np.ndarray:
        """
        Generate the tuning directions for the head direction cells.

        The preferred directions are uniformly distributed in 2D space.

        Returns:
        - np.ndarray: A 2xN array of cosine and sine values representing the preferred directions.
        """
        theta_i = np.linspace(
            0, 2 * np.pi, self.num_cells, endpoint=False
        )  # Create equally spaced angles
        D = np.empty((2, self.num_cells))  # Initialize the kernel array
        D[0] = np.cos(
            theta_i + self.theta_0
        )  # Cosine component of the preferred directions
        D[1] = np.sin(
            theta_i + self.theta_0
        )  # Sine component of the preferred directions
        return D

    def head_direction(self, theta_i: float) -> np.ndarray:
        """
        Compute the activation of the head direction cells based on the input direction.

        Parameters:
        - theta_i (float): The heading angle of the agent (in degrees or radians) in reference to theta_0.

        Returns:
        - np.ndarray: The activation levels of the head direction cells.
        """
        theta_i_rad = self._convert_to_radians(
            theta_i
        )  # Convert the heading angle to radians if necessary
        v_in = np.array([np.cos(theta_i_rad), np.sin(theta_i_rad)])  # Heading vector

        # Compute the dot product of the heading vector with the preferred directions
        activation = np.dot(v_in, self.tuned_directions)

        # Apply a non-linearity (ReLU) to only keep positive activations
        self.state = np.maximum(0, activation)
        return self.state

    def plot_activation(self, plot_type: str = "bar") -> None:
        """
        Plot the activation levels of each head direction cell.

        Parameters:
        - plot_type (str): The type of plot ('bar' for bar chart or 'radial' for radial plot).

        Returns:
        - None
        """
        if self.state is None:
            raise ValueError(
                "Activation state is not set. Please call 'head_direction' first."
            )

        # Label each direction based on the unit
        categories = [
            (
                f"{int(round(np.rad2deg(angle)))}°"
                if self.unit == "degree"
                else f"{round(angle, 2)} rad"
            )
            for angle in np.linspace(0, 2 * np.pi, self.num_cells, endpoint=False)
        ]

        if plot_type == "bar":
            plt.bar(categories, self.state)
            plt.xlabel(f"Tuned Directions ({self.unit})")
            plt.ylabel("Activation Magnitude")
            plt.title("Head Direction Layer Activation")
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
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the HeadDirectionLayer with 16 head direction cells, input in degrees, and theta_0 set to North (0 degrees)
    hd_layer = HeadDirectionLayer(num_cells=8, theta_0=0.0, unit="degree")

    # Compute the activation with a heading of 45 degrees
    activations = hd_layer.head_direction(theta_i=45.0)

    # Plot the activation as a bar chart
    hd_layer.plot_activation(plot_type="bar")

    # Plot the activation as a radial chart
    hd_layer.plot_activation(plot_type="radial")

    # Output the activation values
    print("Activations:", activations)
