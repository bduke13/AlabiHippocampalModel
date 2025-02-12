# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def create_icosahedron(radius=1):
    """
    Create vertices of an icosahedron and compute its faces using the convex hull.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Define the 12 vertices of an icosahedron
    vertices = np.array(
        [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ]
    )

    # Normalize vertices so that they lie on the sphere of the given radius.
    vertices *= radius / np.sqrt(1 + phi**2)

    # Use ConvexHull to get the faces (each face is a triangle)
    hull = ConvexHull(vertices)
    faces = hull.simplices  # each row gives indices for one face

    return vertices, faces


def plot_transparent_icosahedron(
    radius=1, alpha=0.3, edge_color="blue", face_color="lightblue"
):
    """
    Plot a transparent icosahedron with the center point and axis lines from the center
    to the center of each face.
    """
    vertices, faces = create_icosahedron(radius)

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Build list of triangles for Poly3DCollection
    triangles = [vertices[face] for face in faces]

    # Plot the triangular faces
    collection = Poly3DCollection(
        triangles, alpha=alpha, edgecolor=edge_color, facecolor=face_color
    )
    ax.add_collection3d(collection)

    # Compute and plot axis lines from the center (0,0,0) to the center of each face
    for triangle in triangles:
        face_center = np.mean(triangle, axis=0)
        # Draw a dashed line from the origin to the face center
        ax.plot(
            [0, face_center[0]],
            [0, face_center[1]],
            [0, face_center[2]],
            color="red",
            linestyle="--",
            linewidth=1,
        )

    # Mark the center point (origin)
    ax.scatter([0], [0], [0], color="black", s=100, label="Center")

    # Set equal scaling. For newer versions of matplotlib, use set_box_aspect.
    ax.set_box_aspect((1, 1, 1))

    # Set plot limits so the whole icosahedron is visible.
    max_range = radius * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Transparent Icosahedron with Face Axis Lines")

    # Optional: add a legend for the center marker
    ax.legend()

    return fig, ax


def main():
    fig, ax = plot_transparent_icosahedron(
        radius=2, alpha=0.3, edge_color="darkblue", face_color="lightblue"
    )
    plt.show()


if __name__ == "__main__":
    main()
