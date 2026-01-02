"""
Graph Attention Network for Antibody-Antigen Binding Affinity Prediction
==========================================================================

Uses Graph Attention Networks (GAT) with multi-head attention and
attention-based pooling to focus on binding site interactions.

Architecture:
- 3 GAT layers with 8 attention heads each
- Batch normalization after each layer
- Attention-based global pooling (learns which atoms matter)
- 3 fully connected layers for regression
- Dropout for regularization

Key improvements over GCN:
1. Attention mechanism focuses on important atom interactions
2. Multi-head attention captures diverse binding patterns
3. Attention pooling preserves binding site signal
4. Deeper expressiveness for long-range interactions

Author: RSEF Quantum Antibody Binding Project
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, GlobalAttention


class BindingAffinityGAT(nn.Module):
    """
    Graph Attention Network for predicting antibody-antigen binding affinity (pKd)

    Uses attention mechanisms to identify and focus on critical binding interactions
    """

    def __init__(self, input_dim=49, hidden_dim=128, num_heads=8, dropout=0.2):
        """
        Args:
            input_dim: Number of node features (49 for enhanced features)
            hidden_dim: Hidden layer dimension (per head)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(BindingAffinityGAT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout

        # Graph Attention layers
        # Layer 1: input_dim -> hidden_dim * num_heads
        self.gat1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 2: hidden_dim * num_heads -> hidden_dim * num_heads
        self.gat2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 3: hidden_dim * num_heads -> hidden_dim (single head for pooling)
        self.gat3 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False,  # Average attention heads for final layer
            add_self_loops=True
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Attention-based pooling
        # Learns to weight atoms by their importance for binding
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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

        # GAT Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # ELU works better than ReLU for GAT
        x = self.dropout(x)

        # GAT Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # GAT Layer 3
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Attention-based global pooling
        # This learns to focus on binding site atoms
        x = self.pool(x, batch)  # [batch_size, hidden_dim]

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, data):
        """
        Extract attention weights for visualization
        Returns attention weights from each layer
        """
        x, edge_index = data.x, data.edge_index

        attention_weights = []

        # Layer 1
        x, (edge_index_1, alpha_1) = self.gat1(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_1, alpha_1))
        x = self.bn1(x)
        x = F.elu(x)

        # Layer 2
        x, (edge_index_2, alpha_2) = self.gat2(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_2, alpha_2))
        x = self.bn2(x)
        x = F.elu(x)

        # Layer 3
        x, (edge_index_3, alpha_3) = self.gat3(x, edge_index, return_attention_weights=True)
        attention_weights.append((edge_index_3, alpha_3))

        return attention_weights


# Test the model
if __name__ == "__main__":
    print("Testing BindingAffinityGAT model...\n")

    # Create model
    model = BindingAffinityGAT(input_dim=49, hidden_dim=128, num_heads=8, dropout=0.2)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Create dummy data
    from torch_geometric.data import Data, Batch

    # Dummy graph 1: 100 nodes, 49D features
    x1 = torch.randn(100, 49)
    edge_index1 = torch.randint(0, 100, (2, 200))
    data1 = Data(x=x1, edge_index=edge_index1)

    # Dummy graph 2: 150 nodes, 49D features
    x2 = torch.randn(150, 49)
    edge_index2 = torch.randint(0, 150, (2, 300))
    data2 = Data(x=x2, edge_index=edge_index2)

    # Create batch
    batch = Batch.from_data_list([data1, data2])

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"\nInput: Batch of 2 graphs (100 and 150 nodes)")
    print(f"Output shape: {output.shape}")  # Should be [2, 1]
    print(f"Predictions: {output.squeeze().tolist()}")

    # Test attention extraction
    print("\nTesting attention weight extraction...")
    attention_weights = model.get_attention_weights(data1)
    print(f"Number of GAT layers: {len(attention_weights)}")
    for i, (edge_idx, alpha) in enumerate(attention_weights):
        print(f"  Layer {i+1}: {alpha.shape} attention weights")

    print("\n✓ Model works correctly!")
    print("\nKey features:")
    print("  ✓ Multi-head attention (8 heads)")
    print("  ✓ Attention-based pooling")
    print("  ✓ Batch normalization")
    print("  ✓ ELU activations")
    print("  ✓ Dropout regularization")
    print("\nThis architecture can identify and focus on binding site interactions!")