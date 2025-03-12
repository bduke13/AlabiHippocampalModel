# fusion_layer.py
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, encoder_weights_path, input_dim=1680, hidden_dim=400):
        """
        Construct the fusion layer:
        :param encoder_weights_path: Path to the pretrained encoder weights (e.g., "encoder_weights.pth")
        :param input_dim: Dimension of the fused input (visual 1280 + BVC 400 = 1680)
        :param hidden_dim: Dimension of the encoder output (hidden layer), e.g., 400
        """
        super(FusionLayer, self).__init__()
        # Build the encoder: a single fully-connected layer mapping from 1680 to hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Load the pretrained encoder parameters
        self.encoder.load_state_dict(torch.load('encoder_weights.pth', map_location=torch.device('cpu')))
        # Set the encoder to evaluation mode to disable dropout, batch norm, etc.
        self.encoder.eval()

    def forward(self, bvc_output, visual_output):
        """
        Forward propagation:
        :param bvc_output: Real-time output from the BVC layer, shape (400,)
        :param visual_output: Real-time output from the visual pretrained model, shape (1280,)
        :return: The fused feature vector, shape (hidden_dim,), e.g., (400,)
        """
        # Concatenation: concatenate BVC output (400-d) and visual output (1280-d) sequentially to form a 1680-d vector
        fused_input = torch.cat([bvc_output, visual_output], dim=0)  # shape: (1680,)

        # Normalization: perform min-max normalization on the fused vector to scale it to the range [0, 1]
        min_val = fused_input.min()
        max_val = fused_input.max()
        if max_val - min_val < 1e-6:
            normalized = fused_input - min_val  # Avoid division by zero
        else:
            normalized = (fused_input - min_val) / (max_val - min_val)

        # Add a batch dimension (batch=1), shape becomes (1, 1680)
        normalized = normalized.unsqueeze(0)

        # Obtain the fused representation through the pretrained encoder
        encoded = self.encoder(normalized)  # shape: (1, hidden_dim)
        # Remove the batch dimension and return the result with shape (hidden_dim,)
        return encoded.squeeze(0)

# Test code: when running this file directly, execute a simple test
if __name__ == "__main__":
    # Create a FusionLayer instance and load parameters from "encoder_weights.pth"
    fusion_layer = FusionLayer("encoder_weights.pth")
    # Simulate BVC output (400-d) and visual output (1280-d) using random vectors
    bvc_output = torch.rand(400)
    visual_output = torch.rand(1280)
    # Obtain the fused feature representation
    fused_features = fusion_layer(bvc_output, visual_output)
    print("Fused features shape:", fused_features.shape)
    # Expected output example: Fused features shape: torch.Size([400])