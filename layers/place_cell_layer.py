import numpy as np
import tensorflow as tf
tf.random.set_seed(5)
from numpy.random import default_rng
from layers.boundary_vector_cell_layer import BoundaryVectorCellLayer

class PlaceCellLayer(): # Called continuously during explore loop in driver.py
    def __init__(self, num_pc=1000, input_dim=720, timestep=32*3, max_dist=12, n_hd=8):
        '''
        Initializes the Place Cell Layer.

        Parameters:
        num_pc: Number of place cells in the layer.
        input_dim: Dimension of the input vector to the layer (e.g., 720 for RPLidar).
        timestep: Time step for simulation or learning updates.
        max_dist: Maximum distance that the boundary vector cells (BVCs) respond to.
        n_hd: Number of head direction cells.
        '''
        rng = default_rng()

        # Number of place cells
        self.num_pc = num_pc
        
        # Initialize the Boundary Vector Cell (BVC) layer
        self.bvcLayer = BoundaryVectorCellLayer(max_dist=max_dist, input_dim=input_dim, n_hd=n_hd, sigma_ang=90, sigma_d=0.2)
        
        # Number of BVCs (Boundary Vector Cells)
        self.num_bvc = self.bvcLayer.num_distances # NOTE: This is a bit misleading, but it works for now
        
        # Recurrent weight matrix for place-to-place cell connections, considering head direction
        # Original: self.w_rec_c
        # Shape: (n_hd, num_pc, num_pc)
        self.w_rec_place_to_place = tf.Variable(np.zeros((n_hd, num_pc, num_pc)), dtype=tf.float32)
        
        # Input weight matrix connecting place cells to BVCs
        # Shape: (n_hd, num_pc, num_pc)
        self.w_in = tf.Variable(rng.binomial(1, .2, (num_pc, self.num_bvc)), dtype=tf.float32)
        
        # Recurrent weight matrix for head direction and place cell interactions
        # Original: self.w_rec
        # Shape: (n_hd, num_pc, num_pc)
        self.w_rec_hd_place = tf.zeros(shape=(n_hd, num_pc, num_pc), dtype=tf.float32)
        
        # Initial activation values for place cells
        # Original: self.v
        # Shape: (num_pc,)
        self.place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)
        
        # Time constant for updating place cell activations
        self.tau = timestep / 1000
        
        # Initial activations for boundary vector cells (BVCs)
        # Original: self.bvc_v
        # Shape: (num_bvc,)
        self.bvc_activations = tf.zeros(self.num_bvc, dtype=tf.float32)
        
        # Learning rate for the place cell update rule
        self.alpha = 0.5
        
        # Previous place cell activations
        # Original: self.v_prev
        # Shape: (num_pc,)
        self.prev_place_cell_activations = tf.zeros(num_pc, dtype=tf.float32)
        
        # Initial weights for the input connections from BVCs to place cells
        self.initial_w_in = tf.Variable(self.w_in)
        
        # Temporary variable for the current activation update step
        # Original: self.z
        # Shape: (num_pc,)
        self.activation_update = tf.zeros_like(self.place_cell_activations, dtype=tf.float32)
        
        # Placeholder for recurrent connections
        # Original: self.rec
        self.recurrent_input = 0
        
        # Head direction modulation (if applicable, otherwise remains None)
        self.head_direction_modulation = None
        
        # Boundary cell activation values (if any boundary cells are used)
        # Original: self.bc_v
        self.boundary_cell_activations = tf.zeros((n_hd, num_pc))
        
        # Trace of place cell activations for eligibility tracking
        # Original: self.trace
        self.place_cell_trace = tf.zeros_like(self.place_cell_activations)
        
        # Trace of head direction cells for eligibility tracking
        # Original: self.hd_trace
        self.hd_cell_trace = tf.zeros((n_hd, 1, 1), tf.float64)

    def get_place_cell_activations(self, input_data, hd_activations, mode="learn", collided=False):
        '''
        Computes the activation of place cells based on the input from boundary vector cells (BVCs) and head direction vectors (HDV).

        Parameters:
        input_data: Input to the BVC layer. 
                    - input_data[0]: Array of distances (e.g., from RPLidar).
                    - input_data[1]: Array of angles corresponding to those distances.
        hd_activations: Head direction activations.
        mode: Operation mode, typically "learn" or "test".
        collided: Boolean indicating if the agent has collided with an obstacle.
        '''

        # Store the previous place cell activations
        self.prev_place_cell_activations = tf.identity(self.place_cell_activations)

        # Compute BVC activations based on the input distances and angles
        self.bvc_activations = self.bvcLayer.get_bvc_activation(input_data[0], input_data[1])
        
        # Compute the input to place cells by taking the dot product of the input weights and BVC activations
        place_cell_input = tf.tensordot(self.w_in, self.bvc_activations, 1) - 0.3 * tf.reduce_sum(self.bvc_activations)
        
        # Update the activation variable `activation_update` with the new input, applying a scaling factor and considering previous activations
        self.activation_update += 0.1 * (place_cell_input - self.activation_update - self.alpha * tf.reduce_sum(tf.cast(self.place_cell_activations, tf.float32)))
        
        # Apply ReLU followed by tanh to compute the new place cell activations
        self.place_cell_activations = tf.tanh(tf.nn.relu(self.activation_update))

        # Update the eligibility trace and weights if in "dmtp" mode and no collision
        if np.any(self.place_cell_activations) and mode == "dmtp" and not collided:
            if self.place_cell_trace is None:
                self.place_cell_trace = tf.zeros_like(self.place_cell_activations)

            # Update the eligibility trace for place cells and head direction cells
            self.place_cell_trace += self.tau / 3 * (self.place_cell_activations - self.place_cell_trace)
            self.hd_cell_trace += self.tau / 3 * (np.nan_to_num(hd_activations)[:, np.newaxis, np.newaxis] - self.hd_cell_trace)
            
            # Update recurrent weights for place cell interactions modulated by head direction
            self.w_rec_hd_place += tf.cast(np.nan_to_num(hd_activations)[:, np.newaxis, np.newaxis], tf.float32) * (
                tf.tensordot(self.place_cell_activations[:, np.newaxis], self.place_cell_trace[np.newaxis, :], 1) -
                tf.tensordot(self.place_cell_trace[:, np.newaxis], self.place_cell_activations[np.newaxis, :], 1)
            )

        # If the mode is not "learning", update the input weights based on the current activations and BVC activations
        if np.any(self.place_cell_activations) and not (mode == 'learning'):
            weight_update = self.tau * (
                self.place_cell_activations[:, np.newaxis] * (self.bvc_activations[np.newaxis, :] - 
                1 / np.sqrt(0.5) * self.place_cell_activations[:, np.newaxis] * self.w_in)
            )
            self.w_in.assign_add(weight_update)

    def reset_activations(self):
        self.place_cell_activations *= 0
        self.activation_update *= 0
        self.place_cell_trace = None

    def exploit(self, direction, num_steps=1):
        '''
        Exploits the current state to generate place cell activations based on the recurrent weights.

        Parameters:
        direction: The direction in which to exploit the recurrent weights. 
        num_steps: Number of exploitation steps to perform.
        '''

        # Copy the current place cell activations
        # Shape: (num_place_cells,)
        place_cell_activations = tf.identity(self.place_cell_activations)
        
        # Check for collision (optional, currently commented out)
        # if tf.tensordot(self.boundary_cell_weights, self.place_cell_activations, 1)[direction] >= 0.5:
        #     return tf.zeros_like(place_cell_activations)

        # Iterate to update the place cell activations
        for step in range(num_steps):
            previous_activations = tf.identity(place_cell_activations)
            
            # Compute new activations based on recurrent weights and previous activations
            # Shape: (num_place_cells,)
            place_cell_activations = tf.tanh(
                tf.nn.relu(
                    tf.tensordot(tf.cast(self.w_rec_hd_place[direction], tf.float32), previous_activations, 1) 
                    - previous_activations
                )
            )

        # Visualization
        # fig = plot.figure(1)
        # ax = fig.add_subplot(3, 3, plot_location[direction])
        # current_estimate = np.dot(hmap_z, place_cell_activations)
        # try:
        #     ax.tricontourf(hmap_x, hmap_y, current_estimate, cmap=cmap)
        #     ax.set_aspect('equal')
        #     ax.set_ylim(5, -5)
        #     ax.set_title("Norm {0:.2f}".format(tf.linalg.norm(place_cell_activations, 1)))
        # except:
        #     pass
        
        # Return the updated place cell activations
        return place_cell_activations
    
    def __getitem__(self, index):
        '''
        Retrieves the place cell activation at the specified index.
        Example: 
            activation = place_cell_layer[5] = activation of the place cell at index 5

        Parameters:
        index: The index of the place cell activation to retrieve.

        Returns:
        The activation value of the place cell at the specified index.
        '''
        return self.place_cell_activations[index]