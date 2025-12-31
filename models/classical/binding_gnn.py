"""
Classical Graph Neural Network for antibody-antigen binding affinity prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BindingAffinityGNN(nn.Module):
    """
    Graph Neural Network for predicting antibody-antigen binding affinity (pKd)
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
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric Batch object with:
                - data.x: Node features [num_nodes, input_dim]
                - data.edge_index: Edge connections [2, num_edges]
                - data.batch: Batch assignment [num_nodes]

        Returns:
            predictions: [batch_size, 1] tensor of predicted pKd values
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GCN Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GCN Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling: [num_nodes, hidden_dim] -> [batch_size, hidden_dim]
        x = global_mean_pool(x, batch)

        # FC Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == "__main__":
    print("Testing BindingAffinityGNN...\n")

    model = BindingAffinityGNN(input_dim=6, hidden_dim=128, dropout=0.3)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Create dummy batch
    from torch_geometric.data import Data, Batch

    data1 = Data(x=torch.randn(100, 6), edge_index=torch.randint(0, 100, (2, 200)))
    data2 = Data(x=torch.randn(150, 6), edge_index=torch.randint(0, 150, (2, 300)))
    batch = Batch.from_data_list([data1, data2])

    output = model(batch)
    print(f"\nInput: Batch of 2 graphs")
    print(f"Output shape: {output.shape}")  # [2, 1]
    print(f"Predictions: {output.squeeze().tolist()}")
    print("\n✓ Model works!")