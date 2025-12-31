"""
Train classical GNN for binding affinity prediction
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy.stats import pearsonr
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AntibodyAffinityDataset
sys.path.append('../../models/classical')
from antibody_gnn import BindingAffinityGNN

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch).squeeze()
        loss = criterion(predictions, batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch).squeeze()
            loss = criterion(predictions, batch.y.squeeze())

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pearson_r, _ = pearsonr(all_preds, all_labels)
    rmse = np.sqrt(np.mean((all_preds - all_labels)**2))
    mae = np.mean(np.abs(all_preds - all_labels))

    return {
        'loss': total_loss / len(loader),
        'pearson_r': pearson_r,
        'rmse': rmse,
        'mae': mae,
        'predictions': all_preds,
        'labels': all_labels
    }

print("="*70)
print("TRAINING CLASSICAL GNN")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create datasets
print("\n[1/5] Loading datasets...")
train_dataset = AntibodyAffinityDataset('../data/splits/train.csv')
val_dataset = AntibodyAffinityDataset('../data/splits/val.csv')

print(f"   Train: {len(train_dataset)}")
print(f"   Val: {len(val_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
print("\n[2/5] Creating model...")
model = BindingAffinityGNN(input_dim=6, hidden_dim=128, dropout=0.3).to(device)
print(f"   Parameters: {model.count_parameters():,}")

# Training setup
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training loop
print("\n[3/5] Training...")
best_val_r = -1
patience_counter = 0
patience = 20
epochs = 200

for epoch in range(epochs):
    start = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    scheduler.step(val_metrics['loss'])

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val R: {val_metrics['pearson_r']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | "
              f"Time: {time.time()-start:.1f}s")

    # Save best
    if val_metrics['pearson_r'] > best_val_r:
        best_val_r = val_metrics['pearson_r']
        os.makedirs('../results', exist_ok=True)
        torch.save(model.state_dict(), '../results/best_classical_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\n[4/5] Training complete!")
print(f"   Best Val Pearson R: {best_val_r:.4f}")

print("\n✓ Model saved to results/best_classical_model.pt")