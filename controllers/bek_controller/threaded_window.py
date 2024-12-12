# plot_window.py
import matplotlib

matplotlib.use("TkAgg")  # Use the TkAgg backend to avoid Qt issues

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
