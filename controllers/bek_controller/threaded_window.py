# plot_window.py
import matplotlib
import tensorflow as tf

matplotlib.use("TkAgg")  # Use the TkAgg backend to avoid Qt issues

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Tuple, List


class BasePlotWindow(ABC):
    """Abstract base class for plot windows"""

    @abstractmethod
    def setup_plot(self) -> Tuple[plt.Figure, List[Any]]:
        """Setup the plot and return the figure and list of artists to animate"""
        pass

    @abstractmethod
    def update_data(self, new_data: Any) -> None:
        """Update the internal data storage with new data"""
        pass

    @abstractmethod
    def update_plot(self) -> Tuple[Any, ...]:
        """Update the plot and return the artists that were modified"""
        pass


class GenericPlotProcess:
    """Generic plotting process manager that can work with any plot window type"""

    def __init__(self, plot_window: BasePlotWindow, update_interval: int = 100):
        self.data_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=self._plotting_process,
            args=(self.data_queue, plot_window, update_interval),
        )
        self.process.start()

    @staticmethod
    def _plotting_process(
        data_queue: multiprocessing.Queue,
        plot_window: BasePlotWindow,
        update_interval: int,
    ):
        """Internal method that handles the plotting process"""
        fig, artists = plot_window.setup_plot()

        def animate(frame):
            while not data_queue.empty():
                new_data = data_queue.get()
                plot_window.update_data(new_data)
            return plot_window.update_plot()

        ani = FuncAnimation(
            fig,
            animate,
            interval=update_interval,
            blit=True,
            cache_frame_data=False,  # Disable frame caching to prevent warning
        )
        plt.show()

    def add_data(self, data: Any) -> None:
        """Add any type of data to the plot queue"""
        self.data_queue.put(data)

    def close(self) -> None:
        """Cleanup the plotting process"""
        self.process.terminate()
        self.process.join()


class HexbinPlotWindow(BasePlotWindow):
    def __init__(
        self,
        hexbin_data_path: str,  # Path to directory containing hexbin*.npz files
        xlim: Tuple[float, float] = (-5, 5),
        ylim: Tuple[float, float] = (-5, 5),
    ):
        self.hexbin_data_path = hexbin_data_path
        self.stored_patterns = {}
        self.current_activations = None
        self.xlim = xlim
        self.ylim = ylim

        # Load all hexbin data
        self._load_hexbin_data()

    def setup_plot(self) -> Tuple[plt.Figure, List[Any]]:
        """Setup the hexbin plot and return the figure and list of artists to animate"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_title("Place Cell Activations")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True, linestyle="--", alpha=0.7)

        # Create initial hexbin using coordinates from first pattern
        first_cell = list(self.stored_patterns.keys())[0]
        first_pattern = self.stored_patterns[first_cell]
        coords = first_pattern["coordinates"]

        self.hb = self.ax.hexbin(
            coords[:, 0],  # x coordinates
            coords[:, 1],  # y coordinates
            C=np.zeros_like(coords[:, 0]),  # Initial zero activations
            gridsize=30,  # Fixed gridsize for consistency
            extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
            cmap="viridis",  # Better for visualization
            vmin=0.0,
            vmax=1.0,
            reduce_C_function=np.mean,
        )

        plt.colorbar(self.hb, ax=self.ax, label="Activation")
        return self.fig, [self.hb]

    def _load_hexbin_data(self):
        """Load all hexbin data from npz files"""
        import os
        import numpy as np

        print(f"Loading hexbin data from {self.hexbin_data_path}")

        # Sort filenames to ensure consistent cell indexing
        filenames = sorted(
            [
                f
                for f in os.listdir(self.hexbin_data_path)
                if f.startswith("hexbin_cell_") and f.endswith(".npz")
            ],
            key=lambda x: int(x.replace("hexbin_cell_", "").replace(".npz", "")),
        )

        for filename in filenames:
            # Extract cell index from filename
            cell_idx = int(filename.replace("hexbin_cell_", "").replace(".npz", ""))

            # Load the data
            data = np.load(os.path.join(self.hexbin_data_path, filename))

            # Store in the format needed for plotting
            self.stored_patterns[cell_idx] = {
                "coordinates": data["coordinates"],
                "counts": data["activations"],
                "gridsize": int(data["gridsize"][0]),
            }

        print(f"Loaded data for {len(self.stored_patterns)} cells")

        # Verify the cell indices match the expected range
        expected_indices = set(range(len(self.stored_patterns)))
        actual_indices = set(self.stored_patterns.keys())
        if expected_indices != actual_indices:
            print("Warning: Missing or unexpected cell indices!")
            print(f"Missing indices: {expected_indices - actual_indices}")
            print(f"Unexpected indices: {actual_indices - expected_indices}")

    def update_data(self, new_data: dict) -> None:
        """
        Update with new place cell activations
        new_data should be: {cell_index: activation_value}
        """
        self.current_activations = new_data

    def update_plot(self) -> Tuple[Any, ...]:
        if self.current_activations is None:
            return (self.hb,)

        # Convert tensor to numpy if needed
        if tf.is_tensor(self.current_activations):
            activations = self.current_activations.numpy()
        else:
            activations = self.current_activations

        # Print debug info about activations
        print(f"\nActivation vector shape: {activations.shape}")
        print(f"Number of stored patterns: {len(self.stored_patterns)}")
        print(f"Non-zero activations at indices: {np.nonzero(activations)[0]}")

        # Get coordinates from first pattern (all patterns share same coordinates)
        first_cell = list(self.stored_patterns.keys())[0]
        coords = self.stored_patterns[first_cell]["coordinates"]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # Initialize combined activation
        combined_activation = np.zeros_like(x_coords)

        # Track which patterns are being used
        used_patterns = []

        # Iterate through all cells and accumulate their contributions
        for cell_idx in range(len(activations)):
            activation_value = float(activations[cell_idx])
            if activation_value > 0:  # Only process active cells
                if cell_idx in self.stored_patterns:
                    pattern = self.stored_patterns[cell_idx]["counts"]
                    combined_activation += pattern * activation_value
                    used_patterns.append(cell_idx)
                else:
                    print(f"Warning: No stored pattern for active cell {cell_idx}")

        if used_patterns:
            print(f"Used patterns for cells: {used_patterns}")
            print(
                f"Combined activation range: {combined_activation.min():.3f} to {combined_activation.max():.3f}"
            )

        # Create new hexbin with updated values
        self.hb.remove()  # Remove old hexbin
        self.hb = self.ax.hexbin(
            x_coords,
            y_coords,
            C=combined_activation,
            gridsize=30,  # Adjust this value to change resolution
            extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
            cmap="viridis",
            vmin=0.0,
            vmax=(
                np.max(combined_activation) if np.max(combined_activation) > 0 else 1.0
            ),
            reduce_C_function=np.mean,  # Use mean for smoother visualization
        )

        return (self.hb,)


class TrackingPlotWindow(BasePlotWindow):
    def __init__(
        self,
        history_buffer_len: int = 500,
        xlim: Tuple[float, float] = (-2.5, 2.5),
        ylim: Tuple[float, float] = (-2.5, 2.5),
    ):
        self.history_buffer_len = history_buffer_len
        self.data_x = []
        self.data_y = []
        self.xlim = xlim
        self.ylim = ylim

    def setup_plot(self) -> Tuple[plt.Figure, List[Any]]:
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.xlim)  # X from 0 to 800
        self.ax.set_ylim(self.ylim)  # Y from 600 to 0 (inverted)
        self.ax.set_title("Agent Position")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True)

        (self.line,) = self.ax.plot([], [], "b-")
        (self.point,) = self.ax.plot([], [], "ro", markersize=8)

        return self.fig, [self.line, self.point]

    def update_data(self, new_data: Tuple[float, float]) -> None:
        x, y = new_data
        self.data_x.append(x)
        self.data_y.append(y)

        if len(self.data_x) > self.history_buffer_len:
            self.data_x = self.data_x[-self.history_buffer_len :]
            self.data_y = self.data_y[-self.history_buffer_len :]

    def update_plot(self) -> Tuple[Any, ...]:
        self.line.set_data(self.data_x, self.data_y)
        if self.data_x:
            self.point.set_data([self.data_x[-1]], [self.data_y[-1]])
        return self.line, self.point
