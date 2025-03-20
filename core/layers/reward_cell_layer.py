import torch
import math
import copy


class RewardCellLayer:
    def __init__(
        self,
        num_place_cells=200,
        num_replay=3,
        learning_rate=0.1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Reward Cell Layer.

        Args:
            num_place_cells: Dimension of the input vector (number of place cells).
            num_replay: Number of replay iterations.
            learning_rate: Learning rate for weight updates.
            device: Device to run computations on ("cpu" or "cuda").
        """
        self.device = device
        self.num_replay = num_replay
        self.learning_rate = learning_rate

        # Initialize reward cell activation as a tensor of shape (1, 1)
        self.reward_cell_activations = torch.zeros(
            (1, 1), dtype=torch.float32, device=self.device
        )

        # Initialize weights with small random values (stddev=0.01)
        self.w_in = (
            torch.randn((1, num_place_cells), dtype=torch.float32, device=self.device)
            * 0.0001
        )
        self.w_in_effective = self.w_in.clone()

    def update_reward_cell_activations(self, input_data, visit=False):
        """
        Compute the activations of the reward cell based on input data.

        Args:
            input_data: A torch.Tensor (shape: [num_place_cells]) representing the input.
            visit: If True, update the weights based on the input.
        """
        # Ensure input_data is on the correct device
        input_data = input_data.to(self.device)

        # Compute L1 norm and a safe denominator to avoid division by zero
        input_norm = torch.norm(input_data, p=1)
        safe_denominator = torch.max(
            input_norm, torch.tensor(1e-4, dtype=torch.float32, device=self.device)
        )

        # Compute activations via matrix multiplication and normalization
        activations = torch.matmul(self.w_in_effective, input_data) / safe_denominator

        # Clamp activations to be between 0 and 1e6 and reshape to (1, 1)
        self.reward_cell_activations = torch.clamp(activations, 0, 1e6).view(1, 1)

        if visit:
            # Update weights directly based on the input (broadcast input_data over the row)
            updated_weights = self.w_in_effective + self.learning_rate * input_data
            self.w_in_effective = updated_weights

    def replay(self, pcn):
        """
        Replay the place cell activations and update reward cell weights.

        Args:
            pcn: A place cell network instance expected to have attributes:
                 - place_cell_activations: torch.Tensor of shape (num_pc,)
                 - w_rec_tripartite: torch.Tensor of shape (n_hd, num_pc, num_pc)
        """
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

        # Perform replay for a fixed number of time steps (20, as in the original TF version)
        for time_step in range(20):  # ⬅ FIX: Use 20 iterations instead of 10
            exponential_decay_factor = math.exp(-time_step / 6)

            # Normalize place cell activations (L2 norm with stability)
            norm_val = torch.sqrt(
                torch.max(
                    torch.sum(place_cell_activations**2),
                    torch.tensor(
                        1e-12, dtype=place_cell_activations.dtype, device=self.device
                    ),
                )
            )
            normalized_place_cell_activations = place_cell_activations / norm_val

            # Replace any NaN values with zeros
            normalized_place_cell_activations = torch.where(
                torch.isnan(normalized_place_cell_activations),
                torch.zeros_like(normalized_place_cell_activations),
                normalized_place_cell_activations,
            )

            # Cumulatively update the weight update
            weight_update = (
                weight_update
                + exponential_decay_factor * normalized_place_cell_activations
            )

            # Update place cell activations:
            recurrent_weights_max = torch.max(w_rec_tripartite, dim=0)[
                0
            ]  # shape: (num_pc, num_pc)

            # Use torch.matmul to compute activations
            dot_val = torch.matmul(
                recurrent_weights_max.float(), place_cell_activations
            )
            updated_place_cell_activations = torch.relu(
                dot_val + place_cell_activations
            )
            place_cell_activations = torch.tanh(updated_place_cell_activations)

        # Normalize the weight update using the infinity norm **(Avoid extreme scaling issues)**
        norm_inf = torch.norm(weight_update, p=float("inf"))
        if norm_inf > 1e-6:
            normalized_weight_update = weight_update / norm_inf
        else:
            normalized_weight_update = weight_update

        # Update the weights and synchronize the effective weights.
        self.w_in = self.w_in + normalized_weight_update
        self.w_in_effective = self.w_in.clone()

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

        # Update weights based on the TD learning rule.
        updated_weights = self.w_in_effective + self.learning_rate * delta * input_data
        self.w_in_effective = updated_weights
