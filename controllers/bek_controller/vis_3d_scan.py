# %%
import numpy as np
import matplotlib.pyplot as plt

# Load the data
vertical_boundaries = np.load("first_vertical_scan.npy")

# Reshape the data from (259200,) to (360, 720)
reshaped_data = vertical_boundaries.reshape(360, 720)

# Create the plot
plt.figure(figsize=(12, 6))
plt.imshow(reshaped_data, cmap="viridis")
plt.colorbar(label="Value")
plt.title("Vertical Boundaries Data")
plt.xlabel("Longitude (720 points)")
plt.ylabel("Latitude (360 points)")
plt.show()

# %%
# Get the middle scan line (180th row)
middle_scan = reshaped_data[180, :]
center_point = middle_scan[720 // 2]

# Add intermediate visualization of the 180th scan line as a horizontal bar
plt.figure(figsize=(12, 2))
plt.imshow(middle_scan.reshape(1, -1), cmap="viridis", aspect="auto")
plt.colorbar(label="Value")
plt.title("180th Scan Line (Horizontal Bar)")
plt.xlabel("Longitude (720 points)")
plt.axvline(x=360, color="r", linestyle="--", label="Center (360th point)")
plt.legend()
plt.yticks([])  # Remove y-axis ticks since it's just one line
plt.show()

# %%
# Create angles for longitude (720 points around circle)
lon_angles = np.linspace(0, 2 * np.pi, 720)
# Create angles for latitude (360 points from -90 to +90 degrees)
lat_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)

# Initialize arrays for 3D coordinates
x_coords = []
y_coords = []
z_coords = []
values = []

# Calculate 3D coordinates for each point
for lat_idx, lat in enumerate(lat_angles):
    r = reshaped_data[lat_idx, :]

    # Calculate 3D coordinates
    x = r * np.cos(lat) * np.cos(lon_angles)
    y = r * np.cos(lat) * np.sin(lon_angles)
    z = r * np.sin(lat)

    x_coords.extend(x)
    y_coords.extend(y)
    z_coords.extend(z)
    values.extend(r)

# Convert to numpy arrays
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = -1 * np.array(z_coords)
values = np.array(values)

# Create 3D plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")

# Create scatter plot with smaller point size and increased transparency
scatter = ax.scatter(
    x_coords, y_coords, z_coords, c=values, cmap="viridis", alpha=0.1, s=1
)
plt.colorbar(scatter, label="Value")

# Customize the plot
ax.set_title("Full 3D Scan Projection (0-360°)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add grid
ax.grid(True)

# Add a red line for the 180th row (horizontal reference)
ref_r = reshaped_data[180, :]
ref_x = ref_r * np.cos(lon_angles)
ref_y = ref_r * np.sin(lon_angles)
ref_z = np.zeros_like(ref_x)
ax.plot(ref_x, ref_y, ref_z, "r-", linewidth=2, label="180° Reference")

ax.legend()
plt.show()
