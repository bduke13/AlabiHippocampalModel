from controller import Supervisor
import numpy as np

class VelodyneController(Supervisor):
    """Controller for processing Velodyne Puck LiDAR data."""

    def __init__(self):
        super().__init__()

        # Set the timestep
        self.timestep = int(self.getBasicTimeStep())

        # Initialize Velodyne LiDAR
        self.lidar = self.getDevice("velodyne")
        if self.lidar is None:
            raise RuntimeError("Velodyne LiDAR not found. Ensure the name matches Webots configuration.")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()  # Enable point cloud data

        # Set agent position
        self.agent = self.getFromDef("agent")
        if self.agent:
            self.agent.getField("translation").setSFVec3f([2, 0.3, -2])
        else:
            print("Agent DEF not found. Ensure the DEF name matches 'agent'.")

    def save_point_cloud(self):
        """Save the Velodyne point cloud data and extract the horizontal layer."""
        # Retrieve the point cloud data (list of points [x, y, z])
        point_cloud = self.lidar.getPointCloud()
        if not point_cloud:
            print("No point cloud data available.")
            return

        # Convert the point cloud to a NumPy array
        point_cloud_array = np.array([[p.x, p.y, p.z] for p in point_cloud])

        print(f"Full point cloud: {point_cloud_array}")

        # Save the full point cloud and horizon layer
        np.save("velodyne_point_cloud.npy", point_cloud_array)

        print(f"Full point cloud saved: {point_cloud_array.shape[0]} points")

    def run(self):
        """Run the controller until the time limit is reached and save LiDAR data."""
        max_time = 10.0  # Time limit in seconds

        while self.step(self.timestep) != -1:
            # Check if the simulation has reached the time limit
            if self.getTime() >= max_time:
                self.save_point_cloud()

                # Pause the simulation
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
                break


if __name__ == "__main__":
    controller = VelodyneController()
    controller.run()