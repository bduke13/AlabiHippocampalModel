import torch
import math


class RewardCellLayer:
    def __init__(
        self, num_reward_cells=1, input_dim=200, num_replay=3, context=0, device="cpu"
    ):
        """
        Initialize the Reward Cell Layer.

        Args:
            num_reward_cells: Number of reward cells in the model.
            input_dim: Dimension of the input vector to the layer.
            num_replay: Number of replay iterations.
            context: The index of the reward cell to use.
            device: Device to run the computations on ("cpu" or "cuda").
        """
        self.device = device
        self.context = context
        self.num_reward_cells = num_reward_cells

        # Create reward cell activations on the specified device.
        self.reward_cell_activations = torch.zeros(
            (num_reward_cells, 1), dtype=torch.float32, device=self.device
        )

        self.num_replay = num_replay

        # Initialize weights with small random values on the device.
        self.w_in = (
            torch.randn(
                (num_reward_cells, input_dim), dtype=torch.float32, device=self.device
            )
            * 0.01
        )
        self.w_in_effective = self.w_in.clone()

    def update_reward_cell_activations(self, input_data, visit=False):
        """
        Compute the activations of reward cells based on input data.

        Args:
            input_data: A torch.Tensor (shape: [input_dim]) representing the input.
            visit: If True, update the weights of the selected reward cell.
        """
        # Ensure input_data is on the correct device.
        input_data = input_data.to(self.device)
        input_norm = torch.norm(input_data, p=1)
        safe_denominator = torch.max(
            input_norm, torch.tensor(1e-4, dtype=input_data.dtype, device=self.device)
        )

        # Compute activations and clip extreme values.
        activations = torch.matmul(self.w_in_effective, input_data) / safe_denominator
        self.reward_cell_activations = torch.clamp(activations, -1e6, 1e6).unsqueeze(1)

        if visit:
            learning_rate = 0.01  # Adjust as needed.
            updated_weights = (
                self.w_in_effective[self.context] + learning_rate * input_data
            )
            self.w_in_effective[self.context] = updated_weights

    def replay(self, pcn):
        """
        Replay the place cell activations and update reward cell weights.

        Args:
            pcn: A place cell network instance expected to have attributes:
                 - place_cell_activations: torch.Tensor
                 - w_rec_tripartite: torch.Tensor
        """
        # Copy only the required tensors instead of the entire PCN object
        place_cell_activations = pcn.place_cell_activations.clone().to(self.device)
        w_rec_tripartite = pcn.w_rec_tripartite.clone().to(self.device)

        # Initialize the weight update tensor on the same device.
        weight_update = torch.zeros_like(self.w_in, device=self.device)

        for time_step in range(10):
            exponential_decay_factor = math.exp(-time_step / 6)

            # Normalize the place cell activations (ensuring they are on self.device).
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(
                        1e-12, dtype=place_cell_activations.dtype, device=self.device
                    ),
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations, device=self.device),
                normalized_place_cell_activations,
            )

            # Update the weight_update for the current context.
            weight_update[self.context] += (
                exponential_decay_factor * normalized_place_cell_activations
            )

            # Update the place cell activations
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[0]
            dot_val = torch.matmul(
                recurrent_weights_max.float(), place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # Normalize the weight update using the infinity norm.
        norm_inf = torch.norm(weight_update, p=float("inf"))
        if norm_inf > 0:
            normalized_weight_update = weight_update / norm_inf
        else:
            normalized_weight_update = weight_update

        # Update weights and synchronize effective weights.
        self.w_in = self.w_in + normalized_weight_update
        self.w_in_effective = self.w_in.clone()

    def td_update(self, input_data, next_reward):
        """
        Perform a temporal difference update.

        Args:
            input_data: A torch.Tensor representing the input vector.
            next_reward: A float representing the reward at the next timestep.
        """
        input_data = input_data.to(self.device)
        prediction = torch.dot(self.w_in_effective[self.context], input_data)
        delta = next_reward - prediction
        learning_rate = 0.1  # Adjust as needed.
        updated_weights = (
            self.w_in_effective[self.context] + learning_rate * delta * input_data
        )
        self.w_in_effective[self.context] = updated_weights
