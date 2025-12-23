"""PyTorch Dataset placeholder (does not import torch to keep requirements optional)."""

class SimpleDataset:
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'id': self.ids[idx], 'features': []}

