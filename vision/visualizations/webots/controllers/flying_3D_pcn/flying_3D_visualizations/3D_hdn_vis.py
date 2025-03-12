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
    # (The icosahedron has edge length sqrt(2+phi), so we scale accordingly.)
    vertices *= radius / np.sqrt(1 + phi**2)

    # Use ConvexHull to get the faces (each face is a triangle)
    hull = ConvexHull(vertices)
    faces = hull.simplices  # each row gives indices for one face

    return vertices, faces


def plot_transparent_icosahedron(
    radius=1,
    alpha=0.3,
    edge_color="blue",
    face_color="lightblue",
    target_vector=np.array([1, 1, 1]),
):
    """
    Plot a transparent icosahedron with the center point, axis lines from the center
    to each face, and an overlay of cosine similarity lines for a specified target vector.
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

    # --- 1. Normalize the target vector ---
    target_norm = target_vector / np.linalg.norm(target_vector)

    # Plot each face's axis line + "similarity" line
    for triangle in triangles:
        face_center = np.mean(triangle, axis=0)

        # --- 2. Red dotted line: from origin to face center ---
        ax.plot(
            [0, face_center[0]],
            [0, face_center[1]],
            [0, face_center[2]],
            color="red",
            linestyle="--",
            linewidth=1,
        )

        # --- 3. Compute cosine similarity and plot green line ---
        #     (a) Normalize face center
        face_center_norm = face_center / np.linalg.norm(face_center)
        #     (b) cos_sim = dot product of normalized vectors
        cos_sim = np.dot(face_center_norm, target_norm)

        #     (c) scale the face center by cos_sim
        sim_endpoint = cos_sim * face_center

        ax.plot(
            [0, sim_endpoint[0]],
            [0, sim_endpoint[1]],
            [0, sim_endpoint[2]],
            color="green",
            linewidth=2,
        )

    # Mark the center point (origin)
    ax.scatter([0], [0], [0], color="black", s=100, label="Center")

    # --- 4. Optionally plot the target vector itself from the origin ---
    # E.g. if you want a line from (0,0,0) to (1,1,1):
    # (You can comment this out if you don't want to display it.)
    ax.plot(
        [0, target_vector[0]],
        [0, target_vector[1]],
        [0, target_vector[2]],
        color="purple",
        linewidth=2,
        label="Target Vector",
    )

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
    ax.set_title("Transparent Icosahedron with Face Axis Lines + Cosine Similarity")

    # Optional: add a legend
    ax.legend()

    return fig, ax


def main():
    # Example usage: radius=2, slightly transparent, and a target vector of (1,1,1)
    fig, ax = plot_transparent_icosahedron(
        radius=2,
        alpha=0.3,
        edge_color="darkblue",
        face_color="lightblue",
        target_vector=np.array([1, 1, 1]),
    )
    plt.show()


if __name__ == "__main__":
    main()
