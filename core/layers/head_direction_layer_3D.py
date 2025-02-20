# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def create_icosahedron_torch(
    radius=1.0, dtype=torch.float32, device: torch.device = torch.device("cpu")
):
    """
    Creates an icosahedron (12 vertices, 20 triangular faces) using numpy + scipy,
    then converts the result to torch tensors.

    Returns:
        vertices_torch: (12, 3) torch tensor
        faces_torch: (20, 3) torch tensor (indices of each triangular face)
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Define the 12 vertices of an icosahedron in numpy
    vertices_np = np.array(
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
        ],
        dtype=np.float32,
    )

    # Normalize so that the icosahedron has circumscribed sphere = radius
    #  icosahedron side length = sqrt(2 + phi), we scale for radius=1
    scale_factor = radius / np.sqrt(1 + phi**2)
    vertices_np *= scale_factor

    # Use ConvexHull to get the faces
    hull = ConvexHull(vertices_np)
    faces_np = hull.simplices  # shape (20, 3) indices into vertices

    # Convert to torch
    vertices_torch = torch.tensor(vertices_np, dtype=dtype, device=device)
    faces_torch = torch.tensor(faces_np, dtype=torch.long, device=device)

    return vertices_torch, faces_torch


class HeadDirectionLayer3D:
    """
    A 3D Head Direction layer based on the faces of an icosahedron.
    Each face center is a 'preferred direction' in 3D space.
    """

    def __init__(self, dtype=torch.float32, device="cpu"):
        """
        Creates a 3D head direction layer with 20 face directions from an icosahedron.

        Args:
            dtype: torch dtype (e.g. torch.float32)
            device: device string (e.g. 'cpu' or 'cuda')
        """
        """
        Creates a 3D head direction layer with 20 face directions from an icosahedron.

        Args:
            dtype: torch dtype (e.g. torch.float32)
            device: device string (e.g. 'cpu' or 'cuda')
        """
        self.dtype = dtype
        self.device = device

        # Create an icosahedron (radius=1)
        self.vertices, self.faces = create_icosahedron_torch(
            radius=1.0, dtype=self.dtype, device=self.device
        )

        # Compute the center of each face => shape (20, 3)
        face_centers = []
        for face in self.faces:
            tri_verts = self.vertices[face]  # shape (3, 3)
            center = tri_verts.mean(dim=0)  # shape (3,)
            face_centers.append(center)

        self.face_centers = torch.stack(face_centers, dim=0)  # (20, 3)

        # Normalize each face center => "preferred direction" in 3D
        self.tuning_kernel = self.face_centers / self.face_centers.norm(
            dim=1, keepdim=True
        )  # (20, 3)

        self.hd_activations = None  # Will store the most recent activation (20,)

    def get_hd_activation(self, reference_velocity: torch.Tensor) -> torch.Tensor:
        """
        Computes the head-direction activation for a 3D reference_velocity.

        Activation = dot product (cosine similarity) between normalized reference_velocity
        and each face's normalized center.

        Args:
            reference_velocity: a 3D torch Tensor of shape (3,).

        Returns:
            1D torch Tensor of shape (20,) with activation for each face.
        """
        if reference_velocity.shape != (3,):
            raise ValueError("reference_velocity must be shape (3,).")

        # Ensure input is on correct device
        reference_velocity = reference_velocity.to(device=self.device, dtype=self.dtype)

        # Normalize the input velocity
        ref_norm = reference_velocity / reference_velocity.norm()

        # Dot product => (20,) = (20, 3) x (3,)
        activation = torch.matmul(self.tuning_kernel, ref_norm)
        self.hd_activations = activation
        return activation

    def plot_activations_3d(self, reference_velocity=None, return_fig=False):
        """
        Creates a 3D matplotlib plot of:
          - The icosahedron (transparent)
          - Red dashed lines from the origin to each face center
          - Green lines scaled by the activation value from the origin
          - Optionally, a purple line for the reference_velocity

        Args:
            reference_velocity: optional (3,) vector to draw in the plot
            return_fig: if True, return the figure (else call plt.show()).

        Returns:
            fig or None
        """
        if self.hd_activations is None:
            raise ValueError(
                "No activations found. Please call get_hd_activation() first."
            )

        # Move data to cpu + numpy for matplotlib
        vertices_np = self.vertices.detach().cpu().numpy()
        faces_np = self.faces.detach().cpu().numpy()
        face_centers_np = self.face_centers.detach().cpu().numpy()
        activations_np = self.hd_activations.detach().cpu().numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("3D Head Direction Layer Activations (Icosahedron)")

        # Build the triangles for Poly3DCollection
        triangles = [vertices_np[f] for f in faces_np]

        # Create a transparent icosahedron
        icos_coll = Poly3DCollection(
            triangles, alpha=0.2, edgecolor="darkblue", facecolor="lightblue"
        )
        ax.add_collection3d(icos_coll)

        # For each face, we draw two lines:
        #   1) Red dashed line from origin -> face_center
        #   2) Green line from origin -> activation * face_center
        # (activation is the cosine similarity, so it might be negative.)
        for fc, act in zip(face_centers_np, activations_np):
            # Red dashed line
            ax.plot(
                [0, fc[0]],
                [0, fc[1]],
                [0, fc[2]],
                color="red",
                linestyle="--",
                linewidth=1,
            )
            # Green line scaled by activation
            scaled = act * fc  # scale face center by cos_sim
            ax.plot(
                [0, scaled[0]],
                [0, scaled[1]],
                [0, scaled[2]],
                color="green",
                linewidth=4,
            )

        # Optionally plot the reference velocity in purple
        if reference_velocity is not None:
            rv = reference_velocity.detach().cpu().numpy()
            ax.plot(
                [0, rv[0]],
                [0, rv[1]],
                [0, rv[2]],
                color="purple",
                linewidth=2,
                label="Reference Vector",
            )

        # Mark the origin (center)
        ax.scatter([0], [0], [0], color="black", s=50, label="Origin")

        # Make axes appear the same scale
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set plot limits
        max_range = 1.5  # a bit bigger than radius=1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.legend()
        if return_fig:
            return fig
        else:
            plt.show()


# --------------
# Example usage
# --------------
if __name__ == "__main__":
    # 1. Create a 3D head direction layer
    hd3d = HeadDirectionLayer3D(dtype=torch.float32, device="cpu")

    # 2. Suppose we have a current 3D heading vector: e.g. (1, 1, 1)
    v_in = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    # 3. Get the HD activations (cosine similarities w.r.t. each face center)
    activations_3d = hd3d.get_hd_activation(v_in)
    print("3D HD Activations (size 20):", activations_3d)

    # 4. Plot them
    #    We pass reference_velocity=v_in so the purple vector is drawn
    hd3d.plot_activations_3d(reference_velocity=v_in)
