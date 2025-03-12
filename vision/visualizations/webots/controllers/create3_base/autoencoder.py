#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# Define the autoencoder model: input layer -> hidden layer -> output layer
class Autoencoder(nn.Module):
    def __init__(self, input_dim=1680, hidden_dim=400, output_dim=1680):
        super(Autoencoder, self).__init__()
        # Encoder part: input -> hidden layer
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Decoder part: hidden layer -> output
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Get the hidden layer representation
        encoded = self.encoder(x)
        # Reconstruct the output
        decoded = self.decoder(encoded)
        return encoded, decoded


def main():
    # ------------------------------
    # 1. Load training data
    # ------------------------------
    # Assume normalized_total_output.csv is in the current directory, no header, with 1680 numbers per row
    data = pd.read_csv("normalized_total_output.csv", header=None).values.astype(np.float32)
    # The shape of data should be (100, 1680)
    print("Loaded training data, shape:", data.shape)

    # Convert to torch.Tensor
    data_tensor = torch.from_numpy(data)

    # Build dataset and DataLoader
    dataset = TensorDataset(data_tensor)
    batch_size = 16  # Adjust according to your needs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------
    # 2. Define the model, loss function, and optimizer
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=1680, hidden_dim=400, output_dim=1680).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------
    # 3. Train the model
    # ------------------------------
    num_epochs = 200  # Adjust according to your requirements
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)  # inputs shape: (batch_size, 1680)
            optimizer.zero_grad()
            encoded, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    # ------------------------------
    # 4. Save the encoder parameters
    # ------------------------------
    # Save only the weights and biases of the encoder part
    torch.save(model.encoder.state_dict(), "encoder_weights.pth")
    print("Encoder parameters saved to encoder_weights.pth")


if __name__ == "__main__":
    main()