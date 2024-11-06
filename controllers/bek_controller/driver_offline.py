import numpy as np
import tensorflow as tf
import pickle

from layers.place_cell_layer_vision import PlaceCellLayer
from layers.head_direction_layer import HeadDirectionLayer
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

# Set random seed for reproducibility
tf.random.set_seed(5)
np.random.seed(5)


class OfflineDriver:
    """
    The OfflineDriver class loads recorded sensor data and trains a Place Cell Network (PCN)
    using the same learning mechanisms as in the LEARN_OJAS mode to form place fields.
    It also saves the hmaps (hmap_x, hmap_y, hmap_z, hmap_h) for analysis.
    """

    def __init__(self, data_dir="", num_place_cells=200, n_hd=8, timestep=96):
        """
        Initializes the OfflineDriver with the specified parameters.

        Parameters:
            data_dir (str): Directory where the recorded data files are located.
            num_place_cells (int): Number of place cells in the network.
            n_hd (int): Number of head direction cells.
            timestep (int): Timestep used in the simulation (in milliseconds).
        """
        self.data_dir = data_dir
        self.num_place_cells = num_place_cells
        self.n_hd = n_hd
        self.timestep = timestep

        # Load recorded data
        self.load_recorded_data()

        # Initialize the Place Cell Network and Head Direction Layer
        self.initialize_networks()

        # Initialize hmaps
        self.initialize_hmaps()

    def load_recorded_data(self):
        """
        Loads the recorded sensor data from the specified directory.
        """
        self.positions = np.load(self.data_dir + "recorded_positions.npy")
        self.headings = np.load(self.data_dir + "recorded_headings.npy")
        self.lidar_data = np.load(self.data_dir + "recorded_lidar.npy")

        print(f"Loaded recorded data:")
        print(f"Positions shape: {self.positions.shape}")
        print(f"Headings shape: {self.headings.shape}")
        print(f"LiDAR data shape: {self.lidar_data.shape}")

        self.num_steps = self.positions.shape[0]

    def initialize_networks(self):
        """
        Initializes the Place Cell Network and Head Direction Layer.
        """
        bvcLayer = BoundaryVectorCellLayer(
            max_dist=10,
            input_dim=720,
            n_hd=self.n_hd,
            sigma_ang=90,
            sigma_d=0.5,
        )

        self.pcn = PlaceCellLayer(
            bvc_layer=bvcLayer,
            num_pc=self.num_place_cells,
            timestep=self.timestep,
            n_hd=self.n_hd,
        )
        self.pcn.reset_activations()
        print("Initialized new Place Cell Network.")

        self.head_direction_layer = HeadDirectionLayer(num_cells=self.n_hd)
        print("Initialized Head Direction Layer.")

    def initialize_hmaps(self):
        """
        Initializes the history maps for storing positions, place cell activations, etc.
        """
        self.hmap_x = np.zeros(self.num_steps)
        self.hmap_y = np.zeros(self.num_steps)
        self.hmap_z = np.zeros((self.num_steps, self.num_place_cells))
        self.hmap_h = np.zeros((self.num_steps, self.n_hd))
        # If you have goal estimates or other data, you can initialize hmap_g as well
        # self.hmap_g = np.zeros(self.num_steps)

    def run(self):
        print("Starting offline training...")
        for step in range(self.num_steps):
            # Get the data for the current timestep
            boundaries = self.lidar_data[step]
            position = self.positions[step]
            heading_deg = self.headings[step]
            heading_rad = np.deg2rad(heading_deg)

            # Adjust LiDAR data based on heading
            boundaries = np.roll(boundaries, 2 * int(heading_deg))

            # Prepare input data
            angles = np.linspace(0, 2 * np.pi, len(boundaries), endpoint=False)

            # Compute head direction activations
            theta_0 = 0  # Anchor direction (can be adjusted if needed)
            v_in = np.array([np.cos(heading_rad), np.sin(heading_rad)])
            hd_activations = self.head_direction_layer.get_hd_activation(
                theta_0=theta_0, v_in=v_in
            )

            # Update place cell activations with learning
            self.pcn.get_place_cell_activations(
                input_data=[boundaries, angles],
                hd_activations=hd_activations,
                mode="learning",  # Use 'learning' mode for Oja's learning
                collided=False,
            )

            # Store data in hmaps
            self.hmap_x[step] = position[0]
            self.hmap_y[step] = position[1]
            self.hmap_z[step] = self.pcn.place_cell_activations.numpy()
            self.hmap_h[step] = hd_activations

            # Print progress every 1000 steps
            if step % 1000 == 0:
                print(f"Step {step}/{self.num_steps}")

        print("Offline training completed.")

    def save_pcn(self, filename="pcn_offline.pkl"):
        """
        Saves the trained Place Cell Network to a file.

        Parameters:
            filename (str): Name of the file to save the PCN.
        """
        with open(filename, "wb") as output:
            pickle.dump(self.pcn, output)
        print(f"Saved trained Place Cell Network to {filename}")

    def save_hmaps(self, include_hmaps=True):
        """
        Saves the history maps to files.

        Parameters:
            include_hmaps (bool): If True, saves the history maps.
        """
        if include_hmaps:
            with open("hmap_x.pkl", "wb") as output:
                pickle.dump(self.hmap_x, output)
            with open("hmap_y.pkl", "wb") as output:
                pickle.dump(self.hmap_y, output)
            with open("hmap_z.pkl", "wb") as output:
                pickle.dump(self.hmap_z, output)
            with open("hmap_h.pkl", "wb") as output:
                pickle.dump(self.hmap_h, output)
            # If you have hmap_g, save it as well
            # with open('hmap_g.pkl', 'wb') as output:
            #     pickle.dump(self.hmap_g, output)
            print("Saved hmaps.")

    # Usage example


if __name__ == "__main__":
    # Create an instance of the OfflineDriver
    offline_driver = OfflineDriver(
        data_dir="",  # Set to the directory where your recorded data is located
        num_place_cells=200,
        n_hd=8,
        timestep=96,  # Adjust if different in your simulation
    )

    # Run the offline training
    offline_driver.run()

    # Save the trained Place Cell Network
    offline_driver.save_pcn(filename="pcn_offline.pkl")

    # Save the hmaps
    offline_driver.save_hmaps()
