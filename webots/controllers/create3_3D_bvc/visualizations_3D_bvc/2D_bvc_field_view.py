# %%
# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

bvc = BoundaryVectorCellLayer(
    n_res=720, n_hd=8, sigma_theta=0.02, sigma_r=0.5, max_dist=10
)

# plotting code hee
