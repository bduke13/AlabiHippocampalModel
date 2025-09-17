import torch
import math
import copy
DEBUG = False

class RewardCellLayerV4:
    def __init__(
        self,
        num_place_cells=200,
        num_replay=20,
        learning_rate=0.1,
        replay_timesteps=1,
        replay_decay_constant=6.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Reward Cell Layer V3 with fixed normalizations.

        Args:
            num_place_cells: Dimension of the input vector (number of place cells).
            num_replay: Number of replay iterations.
            learning_rate: Learning rate for weight updates.
            replay_timesteps: Number of timesteps for replay (was hardcoded as 40).
            replay_decay_constant: Decay constant for exponential decay (was hardcoded as 6).
            device: Device to run computations on ("cpu" or "cuda").
        """
        

        if DEBUG:
            print(f"[RCN_V3] Initialized with:")
            print(f"  - Learning rate: {learning_rate}")
            print(f"  - Replay timesteps: {replay_timesteps}")
            print(f"  - Replay decay constant: {replay_decay_constant}")
            print(f"  - Num place cells: {num_place_cells}")

        self.device = device
        self.num_replay = num_replay
        self.learning_rate = learning_rate
        
        # New parameterized replay settings
        self.replay_timesteps = replay_timesteps
        self.replay_decay_constant = replay_decay_constant

        # Initialize reward cell activation as a tensor of shape (1, 1)
        self.reward_cell_activations = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Initialize weights with small random values (stddev=0.01)
        """
        self.w_in = (
            torch.randn((1, num_place_cells), dtype=torch.float32, device=self.device)
            * 0.0001
        )
        self.w_in_effective = self.w_in.clone()
        """
        self.w_in = torch.zeros(
            (1, num_place_cells), dtype=torch.float32, device=self.device
        )
        self.w_in_effective = self.w_in.clone()

    def update_reward_cell_activations(self, input_data, visit=False):
        """
        Compute the activations of the reward cell based on input data.
        V3: Removed L1 normalization that created scale bias.

        Args:
            input_data: A torch.Tensor (shape: [num_place_cells]) representing the input.
            visit: If True, update the weights based on the input.
        """
        # Ensure input_data is on the correct device
        input_data = input_data.to(self.device)

        # V3 CHANGE: Remove L1 normalization that diluted signal with more place cells
        # OLD: activations = torch.matmul(self.w_in_effective, input_data) / safe_denominator
        # NEW: Direct computation without normalization
        activations = torch.matmul(self.w_in_effective, input_data)

        # Clamp activations to reasonable range and reshape to (1, 1)
        self.reward_cell_activations = torch.clamp(activations, -1e6, 1e6).view(1, 1)

        if visit:
            if DEBUG:
                self.debug_weights("Before reward learning")
                print(f"[DEBUG] Place cell device: {input_data.device}")
                print(f"[DEBUG] RCN w_in_effective device: {self.w_in_effective.device}")
                print(f"[DEBUG] RCN self.device attribute: {self.device}")
                print(f"[DEBUG] Learning rate: {self.learning_rate}")
                print(f"[DEBUG] Input data sum: {input_data.sum().item()}")
                print(f"[DEBUG] Input data shape: {input_data.shape}")
                
                # Test if tensor operation would work
                try:
                    test_result = self.w_in_effective + self.learning_rate * input_data
                    print(f"[DEBUG] Tensor operation successful")
                except Exception as e:
                    print(f"[DEBUG] Tensor operation FAILED: {e}")
                print(f"[RCN_V3] REWARD LEARNING TRIGGERED!")
                print(f"  - Learning rate: {self.learning_rate}")
                print(f"  - Input sum: {input_data.sum().item():.4f}")
                print(f"  - Input max: {input_data.max().item():.4f}")
                print(f"  - Activation before: {self.reward_cell_activations.item():.4f}")

            # Update weights directly based on the input (broadcast input_data over the row)
            updated_weights = self.w_in_effective + self.learning_rate * input_data
            self.w_in_effective = updated_weights

            if DEBUG:
                self.debug_weights("After reward learning")
                print(f"  - Weight update magnitude: {(self.learning_rate * input_data).norm().item():.4f}")
                print(f"  - Weight magnitude after update: {self.w_in_effective.norm().item():.4f}")

    def replay(self, pcn):
        """
        Replay the place cell activations and update reward cell weights.
        V3: Removed problematic L2 and infinity normalizations.

        Args:
            pcn: A place cell network instance expected to have attributes:
                 - place_cell_activations: torch.Tensor of shape (num_pc,)
                 - w_rec_tripartite: torch.Tensor of shape (n_hd, num_pc, num_pc)
        """
        if DEBUG:
            print(f"[RCN_V3] REPLAY STARTING:")
            print(f"  - Timesteps: {self.replay_timesteps}")
            print(f"  - Decay constant: {self.replay_decay_constant}")

        # Use a deep copy to prevent external modifications
        pcn_copy = copy.deepcopy(pcn)

        # Move tensors to the correct device
        place_cell_activations = pcn_copy.place_cell_activations.to(
            self.device
        )  # shape: (num_pc,)
        w_rec_tripartite = pcn_copy.w_rec_tripartite.to(
            self.device
        )  # shape: (n_hd, num_pc, num_pc)

        # Initialize the weight update tensor with the same shape as w_in: (1, num_place_cells)
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        if DEBUG:
            print(f"  - Initial place cell activation sum: {place_cell_activations.sum().item():.4f}")
            print(f"  - Initial place cell activation max: {place_cell_activations.max().item():.4f}")

        # Perform replay for the specified number of time steps (parameterized)
        for time_step in range(self.replay_timesteps):
            # Use parameterized decay constant
            exponential_decay_factor = math.exp(-time_step / self.replay_decay_constant)

            # V3 CHANGE: Remove L2 normalization that created scale bias
            # OLD: normalized_place_cell_activations = place_cell_activations / norm_val
            # NEW: Use raw activations with stability check
            
            # Basic stability check - replace NaN values with zeros
            stable_place_cell_activations = torch.where(
                torch.isnan(place_cell_activations),
                torch.zeros_like(place_cell_activations),
                place_cell_activations,
            )

            # V3 CHANGE: Accumulate without normalization 
            # This preserves the biological nature of replay accumulation
            weight_update = (
                weight_update
                + exponential_decay_factor * stable_place_cell_activations
            )

            # Update place cell activations using recurrent connections
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[0]  # shape: (num_pc, num_pc)

            # Use torch.matmul to compute activations
            dot_val = torch.matmul(recurrent_weights_max.float(), place_cell_activations)
            updated_place_cell_activations = torch.relu(dot_val + place_cell_activations)
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # V3 CHANGE: Replace infinity norm with simple scaling for stability
        # This maintains signal strength better across different network sizes
        
        # Check for reasonable magnitude and scale if needed
        weight_magnitude = torch.norm(weight_update, p=2)  # Use L2 norm for magnitude check
        
        if weight_magnitude > 1e-6:
            # Scale by a reasonable factor rather than normalizing to 1
            # This preserves relative signal strength while preventing explosion
            max_reasonable_magnitude = 10.0  # Adjustable parameter
            if weight_magnitude > max_reasonable_magnitude:
                scaling_factor = max_reasonable_magnitude / weight_magnitude
                weight_update = weight_update * scaling_factor
                if DEBUG:
                    print(f"  - Scaled weight update by factor: {scaling_factor:.4f}")
        
        # Update the weights without harsh normalization
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

        if DEBUG:
            print(f"[RCN_V3] REPLAY COMPLETED:")
            print(f"  - Final weight update magnitude: {weight_magnitude.item():.6f}")
            print(f"  - Weight magnitude after replay: {self.w_in_effective.norm().item():.4f}")
            print(f"  - Final place cell sum: {place_cell_activations.sum().item():.4f}")

    def td_update(self, input_data, next_reward):
        """
        Perform a temporal difference (TD) update for reward learning.

        Args:
            input_data: A torch.Tensor representing the input vector (shape: [num_place_cells]).
            next_reward: A float representing the reward at the next timestep.
        """
        input_data = input_data.to(self.device)

        # Compute prediction using the full weight vector (a single reward cell)
        prediction = torch.matmul(self.w_in_effective, input_data)  # shape: (1,)
        delta = next_reward - prediction.squeeze()

        if DEBUG:
            print(f"[RCN_V3] TD UPDATE:")
            print(f"  - Prediction: {prediction.item():.4f}")
            print(f"  - Actual reward: {next_reward:.4f}")
            print(f"  - TD error: {delta.item():.4f}")

        # Update weights based on the TD learning rule.
        updated_weights = self.w_in_effective + self.learning_rate * delta * input_data
        self.w_in_effective = updated_weights
        

    def get_replay_info(self):
        """
        Get current replay parameters for debugging/logging.
        S
        Returns:
            dict: Dictionary containing replay parameters.
        """
        return {
            "replay_timesteps": self.replay_timesteps,
            "replay_decay_constant": self.replay_decay_constant,
            "learning_rate": self.learning_rate,
            "debug_enabled": DEBUG
        }
    
    # Add this method to RewardCellLayerV3 class
    def debug_weights(self, context=""):
        """Debug helper to track weight changes"""
        weight_sum = torch.sum(torch.abs(self.w_in_effective)).item()
        weight_max = torch.max(torch.abs(self.w_in_effective)).item()
        print(f"[RCN_DEBUG] {context}: weight_sum={weight_sum:.6f}, weight_max={weight_max:.6f}")
        
        # Check if weights look corrupted (random vs spatial pattern)
        if weight_sum > 0:
            std = torch.std(self.w_in_effective).item()
            mean = torch.mean(self.w_in_effective).item()
            print(f"[RCN_DEBUG] {context}: mean={mean:.6f}, std={std:.6f}")

    def replay_with_custom_activations(self, pcn, custom_activations):
        """
        Replay the place cell activations and update reward cell weights using custom activations.
        This method is identical to replay() but uses provided activations instead of PCN's current state.
        Used specifically for multi-goal reward map creation.

        Args:
            pcn: A place cell network instance expected to have attributes:
                - w_rec_tripartite: torch.Tensor of shape (n_hd, num_pc, num_pc)
            custom_activations: torch.Tensor of shape (num_pc,) - the activations to use for replay
        """
        if DEBUG:
            print(f"[RCN_V3] REPLAY WITH CUSTOM ACTIVATIONS STARTING:")
            print(f"  - Timesteps: {self.replay_timesteps}")
            print(f"  - Decay constant: {self.replay_decay_constant}")
            print(f"  - Custom activations sum: {custom_activations.sum().item():.4f}")
            print(f"  - Custom activations max: {custom_activations.max().item():.4f}")

        # Use a deep copy to prevent external modifications
        pcn_copy = copy.deepcopy(pcn)

        # Move tensors to the correct device - use CUSTOM activations instead of PCN's current state
        place_cell_activations = custom_activations.to(self.device)  # ← KEY DIFFERENCE: Use custom activations
        w_rec_tripartite = pcn_copy.w_rec_tripartite.to(self.device)  # shape: (n_hd, num_pc, num_pc)

        # Initialize the weight update tensor with the same shape as w_in: (1, num_place_cells)
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        if DEBUG:
            print(f"  - Starting place cell activation sum: {place_cell_activations.sum().item():.4f}")
            print(f"  - Starting place cell activation max: {place_cell_activations.max().item():.4f}")

        # Perform replay for the specified number of time steps (parameterized)
        for time_step in range(self.replay_timesteps):
            # Use parameterized decay constant
            exponential_decay_factor = math.exp(-time_step / self.replay_decay_constant)

            # V3 CHANGE: Remove L2 normalization that created scale bias
            # Basic stability check - replace NaN values with zeros
            stable_place_cell_activations = torch.where(
                torch.isnan(place_cell_activations),
                torch.zeros_like(place_cell_activations),
                place_cell_activations,
            )

            # V3 CHANGE: Accumulate without normalization 
            # This preserves the biological nature of replay accumulation
            weight_update = (
                weight_update
                + exponential_decay_factor * stable_place_cell_activations
            )

            # Update place cell activations using recurrent connections
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[0]  # shape: (num_pc, num_pc)

            # Use torch.matmul to compute activations
            dot_val = torch.matmul(recurrent_weights_max.float(), place_cell_activations)
            updated_place_cell_activations = torch.relu(dot_val + place_cell_activations)
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # V3 CHANGE: Replace infinity norm with simple scaling for stability
        # This maintains signal strength better across different network sizes
        
        # Check for reasonable magnitude and scale if needed
        weight_magnitude = torch.norm(weight_update, p=2)  # Use L2 norm for magnitude check
        
        if weight_magnitude > 1e-6:
            # Scale by a reasonable factor rather than normalizing to 1
            # This preserves relative signal strength while preventing explosion
            max_reasonable_magnitude = 10.0  # Adjustable parameter
            if weight_magnitude > max_reasonable_magnitude:
                scaling_factor = max_reasonable_magnitude / weight_magnitude
                weight_update = weight_update * scaling_factor
                if DEBUG:
                    print(f"  - Scaled weight update by factor: {scaling_factor:.4f}")
        
        # Update the weights without harsh normalization
        self.w_in = self.w_in + weight_update
        self.w_in_effective = self.w_in.clone()

        if DEBUG:
            print(f"[RCN_V3] REPLAY WITH CUSTOM ACTIVATIONS COMPLETED:")
            print(f"  - Final weight update magnitude: {weight_magnitude.item():.6f}")
            print(f"  - Weight magnitude after replay: {self.w_in_effective.norm().item():.4f}")
            print(f"  - Final place cell sum: {place_cell_activations.sum().item():.4f}")