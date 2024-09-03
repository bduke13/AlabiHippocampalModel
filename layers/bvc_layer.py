import numpy as np
import tensorflow as tf
tf.random.set_seed(5)

class bvcLayer():
    def __init__(self, max_dist=12, input_dim=720, n_hd=8, sigma_ang=90, sigma_d=.5):
        '''
        Initializes the boundary vector cell (BVC) layer.

        Parameters:
        max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
        input_dim: Size of input vector to the BVC layer (720 for RPLidar).
        n_hd: Number of head direction cells.
        sigma_ang: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
        sigma_d: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
        '''
        
        # Preferred distances for each BVC; determines how sensitive each BVC is to specific distances.
        # Shape: (1, num_distances), where num_distances = n_hd * (max_dist / (sigma_d/2))
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d/2), n_hd)[np.newaxis, :]  
        
        # Total number of BVC tuning points (number of preferred distances) = 384 ---> 8 head directions * 48 distances per head direction.
        self.num_distances = self.d_i.size  
        
        # Indices for input vector, aligning BVCs with specific head directions.
        # Shape: (1, num_distances)
        self.input_indices = np.repeat(np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int), max_dist/(sigma_d/2))[np.newaxis, :]
        
        # Preferred angles for each BVC (in radians).
        # Shape: (1, num_distances)
        self.phi_i = np.linspace(0, 2*np.pi, input_dim)[self.input_indices]  
        
        # Angular standard deviation for BVC tuning (converted to radians).
        self.sigma_ang  = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)  
        
        # Placeholder for BVC output.
        self.bvc_out = None  

        # Distance standard deviation for BVC tuning.
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)  

    def compute_bvc_activation(self, distances, angles):
        # NOTE: This bypasses the __call__ method in Ade's original code
        # How to use (in driver.py): self.pcn.compute_place_cell_activation([self.boundaries, np.linspace(0, 2*np.pi, 720, False)], self.hdv, self.context, self.mode, np.any(self.collided))

        PI = tf.constant(np.pi) 

        # Gaussian function for distance tuning
        distance_gaussian = tf.exp(-(distances[self.input_indices] - self.d_i)**2 / (2 * self.sigma_d**2)) / tf.sqrt(2 * PI * self.sigma_d**2)
        
        # Gaussian function for angular tuning
        angular_gaussian = tf.exp(-((angles[self.input_indices] - self.phi_i)**2) / (2 * self.sigma_ang**2)) / tf.sqrt(2 * PI * self.sigma_ang**2)
        
        # Return the product of distance and angular Gaussian functions
        return tf.reduce_sum((distance_gaussian * angular_gaussian), 0)
