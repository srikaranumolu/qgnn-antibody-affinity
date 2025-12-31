import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BindingAffinityGNN(nn.Module):
    """
    Graph Neural Network for predicting antibody-antigen binding affinity
    """

    def __init__(self, input_dim=6, hidden_dim=128, dropout=0.3):
        """
        Args:
            input_dim: Number of node features (6 for one-hot element encoding)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(BindingAffinityGNN, self).__init__()

        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object with:
                - data.x: Node features [num_nodes, input_dim]
                - data.edge_index: Edge connections [2, num_edges]
                - data.batch: Batch assignment [num_nodes]

        Returns:
            predictions: [batch_size, 1] tensor of predicted pKd values
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph convolution 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph convolution 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling: aggregate all node features
        # Output: [batch_size, hidden_dim]
        x = global_mean_pool(x, batch)

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("Testing BindingAffinityGNN model...\n")

    # Create model
    model = BindingAffinityGNN(input_dim=6, hidden_dim=128)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Create dummy data
    from torch_geometric.data import Data, Batch

    # Dummy graph 1: 10 nodes
    x1 = torch.randn(10, 6)
    edge_index1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)

    # Dummy graph 2: 15 nodes
    x2 = torch.randn(15, 6)
    edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)

    # Create batch
    batch = Batch.from_data_list([data1, data2])

    # Forward pass
    output = model(batch)
    print(f"\nInput: Batch of 2 graphs (10 and 15 nodes)")
    print(f"Output shape: {output.shape}")  # Should be [2, 1]
    print(f"Predictions: {output.squeeze().tolist()}")

    print("\n✓ Model works!")