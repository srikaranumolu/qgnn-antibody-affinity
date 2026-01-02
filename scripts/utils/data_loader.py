"""
PyTorch DataLoader for antibody-antigen graphs
"""
import torch
from torch_geometric.data import Dataset
import pandas as pd
import os


class AntibodyAffinityDataset(Dataset):
    """
    Dataset for loading antibody-antigen graphs with pKd labels
    """

    def __init__(self, csv_file):
        """
        Args:
            csv_file: Path to CSV with columns ['PDB_ID', 'graph_path', 'pKd', ...]
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)

    def len(self):
        return len(self.data)

    def get(self, idx):
        """
        Load graph and return with pKd label
        """
        row = self.data.iloc[idx]

        # Load preprocessed graph
        graph = torch.load(row['graph_path'], weights_only=False)

        # ADD THE LABEL HERE (this was missing!)
        graph.y = torch.tensor([row['pKd']], dtype=torch.float)

        # Add PDB ID for reference
        graph.pdb_id = row['PDB_ID']

        return graph


# Test it
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    print("Testing dataset...")

    # Update path to your actual splits location
    dataset = AntibodyAffinityDataset('../../data/splits/train.csv')
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Check one sample
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  PDB ID: {sample.pdb_id}")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.num_edges}")
    print(f"  Node features: {sample.x.shape}")
    print(f"  pKd label: {sample.y.item():.2f}")

    # Test batch loading
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print(f"\nBatch test (batch_size=8):")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Total nodes: {batch.num_nodes}")
    print(f"  Total edges: {batch.num_edges}")
    print(f"  Labels shape: {batch.y.shape}")

    print("\n✓ DataLoader working correctly!")