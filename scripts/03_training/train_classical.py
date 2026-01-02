"""
Training Script for GAT-based Binding Affinity Prediction
==========================================================

Trains Graph Attention Network with enhanced 49D features and
attention-based pooling for antibody-antigen binding prediction.

Features:
- Proper train/val/test splits
- Early stopping on validation RMSE
- Comprehensive metrics (Pearson R, RMSE, MAE)
- Model checkpointing
- Progress tracking

Author: RSEF Quantum Antibody Binding Project
Date: January 2026
"""

import os
import sys
import math
import json
import yaml
import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.classical.binding_gat import BindingAffinityGAT
from scripts.utils.data_loader import AntibodyAffinityDataset


def rmse(y_pred, y_true):
    """Root Mean Squared Error"""
    return math.sqrt(np.mean((y_pred - y_true) ** 2))


def mae(y_pred, y_true):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset"""
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)

        # Handle both tensor and scalar labels
        batch_preds = out.cpu().numpy()
        batch_labels = batch.y.squeeze(-1).cpu().numpy()

        # Convert to flat arrays
        batch_preds = np.atleast_1d(batch_preds)
        batch_labels = np.atleast_1d(batch_labels)

        preds.extend(batch_preds.tolist())
        trues.extend(batch_labels.tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    r, p = pearsonr(preds, trues) if len(preds) > 1 else (float("nan"), float("nan"))

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": float(rmse(preds, trues)),
        "mae": float(mae(preds, trues)),
        "predictions": preds.tolist(),
        "true_values": trues.tolist()
    }


def train():
    print("=" * 70)
    print("TRAINING GAT FOR BINDING AFFINITY PREDICTION")
    print("=" * 70)

    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "gat_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract hyperparameters
    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_heads = config["model"]["num_heads"]
    dropout = config["model"]["dropout"]

    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    max_epochs = config["training"]["max_epochs"]
    patience = config["training"]["patience"]
    weight_decay = config["training"]["weight_decay"]

    train_csv = os.path.join(PROJECT_ROOT, config["paths"]["train_csv"])
    val_csv = os.path.join(PROJECT_ROOT, config["paths"]["val_csv"])
    test_csv = os.path.join(PROJECT_ROOT, config["paths"]["test_csv"])
    save_dir = os.path.join(PROJECT_ROOT, config["paths"]["save_dir"])
    metrics_dir = os.path.join(PROJECT_ROOT, config["paths"]["metrics_dir"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/7] Device: {device}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    print(f"\n[2/7] Loading datasets...")
    train_ds = AntibodyAffinityDataset(train_csv)
    val_ds = AntibodyAffinityDataset(val_csv)
    test_ds = AntibodyAffinityDataset(test_csv)

    print(f"      Train: {len(train_ds)} samples")
    print(f"      Val:   {len(val_ds)} samples")
    print(f"      Test:  {len(test_ds)} samples")

    # Check feature dimensions
    sample = train_ds[0]
    actual_feat_dim = sample.x.shape[1]
    print(f"      Feature dim: {actual_feat_dim}D")

    if actual_feat_dim != input_dim:
        print(f"\n⚠ WARNING: Config expects {input_dim}D but graphs have {actual_feat_dim}D!")
        print(f"      Using actual dimension: {actual_feat_dim}D")
        input_dim = actual_feat_dim

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create model
    print(f"\n[3/7] Creating GAT model...")
    model = BindingAffinityGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    print(f"      Architecture: GAT with {num_heads} attention heads")
    print(f"      Parameters: {model.count_parameters():,}")
    print(f"      Hidden dim: {hidden_dim} per head")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Baseline metrics
    print(f"\n[4/7] Computing baseline...")
    train_labels = np.array([train_ds[i].y.item() for i in range(len(train_ds))])
    baseline_rmse = train_labels.std()
    print(f"      Baseline RMSE (label std): {baseline_rmse:.4f}")
    print(f"      Target: Beat {baseline_rmse:.4f} to show learning")

    # Training loop
    print(f"\n[5/7] Training...")
    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    training_history = {
        'train_loss': [],
        'val_rmse': [],
        'val_r': []
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch).squeeze(-1)
            true = batch.y.squeeze(-1)

            # Ensure both are 1D tensors
            pred = pred.view(-1)
            true = true.view(-1)

            loss = criterion(pred, true)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step(val_metrics['rmse'])

        # Record history
        training_history['train_loss'].append(avg_loss)
        training_history['val_rmse'].append(val_metrics['rmse'])
        training_history['val_r'].append(val_metrics['pearson_r'])

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch {epoch:03d}/{max_epochs} | "
                  f"Train MSE: {avg_loss:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"Val R: {val_metrics['pearson_r']:.4f}")

        # Early stopping
        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n      Early stopping at epoch {epoch}")
            print(f"      Best Val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break

    # Restore best model
    print(f"\n[6/7] Loading best model (Epoch {best_epoch}, Val RMSE: {best_val_rmse:.4f})...")
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final evaluation on all splits
    print(f"\n[7/7] Final evaluation...")

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nTrain Set:")
    print(f"  Pearson R: {train_metrics['pearson_r']:.4f}")
    print(f"  RMSE:      {train_metrics['rmse']:.4f}")
    print(f"  MAE:       {train_metrics['mae']:.4f}")

    print(f"\nValidation Set:")
    print(f"  Pearson R: {val_metrics['pearson_r']:.4f}")
    print(f"  RMSE:      {val_metrics['rmse']:.4f}")
    print(f"  MAE:       {val_metrics['mae']:.4f}")

    print(f"\nTest Set:")
    print(f"  Pearson R: {test_metrics['pearson_r']:.4f}")
    print(f"  RMSE:      {test_metrics['rmse']:.4f}")
    print(f"  MAE:       {test_metrics['mae']:.4f}")

    # Performance assessment
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)

    if test_metrics['rmse'] < baseline_rmse:
        improvement = ((baseline_rmse - test_metrics['rmse']) / baseline_rmse) * 100
        print(f"✓ Model beats baseline by {improvement:.1f}%")
    else:
        print(f"✗ Model does not beat baseline ({test_metrics['rmse']:.4f} vs {baseline_rmse:.4f})")

    if test_metrics['pearson_r'] >= 0.50:
        print(f"✓ Strong correlation (R = {test_metrics['pearson_r']:.4f})")
    elif test_metrics['pearson_r'] >= 0.35:
        print(f"◐ Moderate correlation (R = {test_metrics['pearson_r']:.4f})")
    else:
        print(f"✗ Weak correlation (R = {test_metrics['pearson_r']:.4f})")

    # Save everything
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "gat_model.pt")
    metrics_path = os.path.join(metrics_dir, "gat_metrics.json")
    history_path = os.path.join(metrics_dir, "gat_training_history.json")

    # Save model
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'best_epoch': best_epoch,
        'best_val_rmse': best_val_rmse
    }, model_path)

    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'baseline_rmse': baseline_rmse,
        'best_epoch': best_epoch
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save training history
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  Model:      {model_path}")
    print(f"  Metrics:    {metrics_path}")
    print(f"  History:    {history_path}")
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train()