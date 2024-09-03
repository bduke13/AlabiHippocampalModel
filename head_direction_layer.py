import numpy as np
import matplotlib.pyplot as plt


class HeadDirectionLayer:
    def __init__(self, size, unit="degree"):
        """
        Initialize the HeadDirectionLayer with the number of head direction cells
        and the unit of input direction (degree or radian).
        """
        self.size = size  # Number of head direction cells
        self.unit = unit  # Unit of input direction ('degree' or 'radian')
        self.state = None  # Store the activation state of the head direction cells
        self.tuned_directions = None  # Store the tuned directions for labeling

    def _convert_to_radians(self, value):
        """
        Convert the input direction to radians if necessary.
        """
        return np.deg2rad(value) if self.unit == "degree" else value

    def _generate_tuning_kernel(self):
        """
        Generate unit vectors for preferred directions uniformly distributed in 2D space.
        Also store the tuned directions in degrees or radians for labeling.
        """
        angles = np.linspace(0, 2 * np.pi, self.size, endpoint=False)
        self.tuned_directions = np.rad2deg(angles) if self.unit == "degree" else angles
        return np.array([np.cos(angles), np.sin(angles)])

    def head_direction(self, direction):
        """
        Compute the activation of head direction cells based on the input direction.
        """
        direction_rad = self._convert_to_radians(
            direction
        )  # Convert direction to radians
        v_in = np.array(
            [np.cos(direction_rad), np.sin(direction_rad)]
        )  # Heading vector

        tuning_vectors = self._generate_tuning_kernel()  # Preferred directions of cells

        self.state = np.dot(
            tuning_vectors.T, v_in
        )  # Compute dot product for activation
        self.state[self.state < 0] = 0  # Set negative activations to zero

        return self.state

    def plot_activation(self, plot_type="bar"):
        """
        Plot the activation levels of each head direction cell, using the tuned directions as labels.
        Allows the user to choose between a bar chart or a radial plot.
        """
        if self.state is None:
            raise ValueError(
                "Activation state is not set. Please call 'head_direction' first."
            )

        categories = [
            (
                f"{int(round(angle))}°"
                if self.unit == "degree"
                else f"{round(angle, 2)} rad"
            )
            for angle in self.tuned_directions
        ]

        if plot_type == "bar":
            plt.bar(categories, self.state)
            plt.xlabel(f"Tuned Directions ({self.unit})")
            plt.ylabel("Activation Magnitude")
            plt.title("Head Direction Layer Activation")
            plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        elif plot_type == "radial":
            angles = np.linspace(0, 2 * np.pi, self.size, endpoint=False)
            r = self.state
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.bar(angles, r, width=2 * np.pi / self.size, bottom=0.0, align="center")
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
    # Initialize the HeadDirectionLayer with 8 head direction cells and input in degrees
    hd_layer = HeadDirectionLayer(size=16, unit="degree")

    # Compute the activation with a heading of 45 degrees
    activations = hd_layer.head_direction(direction=45)

    # Plot the activation as a bar chart
    hd_layer.plot_activation(plot_type="bar")

    # Plot the activation as a radial chart
    hd_layer.plot_activation(plot_type="radial")

    # Output the activation values
    print("Activations:", activations)
