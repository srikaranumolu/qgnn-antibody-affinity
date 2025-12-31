"""
Train classical GNN for antibody-antigen binding affinity prediction
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

from models.classical.binding_gnn import BindingAffinityGNN
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
        preds.extend(out.cpu().numpy().tolist())
        trues.extend(batch.y.squeeze(-1).cpu().numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    r, p = pearsonr(preds, trues) if len(preds) > 1 else (float("nan"), float("nan"))

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": float(rmse(preds, trues)),
        "mae": float(mae(preds, trues)),
    }


def train():
    print("=" * 70)
    print("TRAINING CLASSICAL GNN")
    print("=" * 70)

    config_path = os.path.join(PROJECT_ROOT, "configs", "classical_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract hyperparameters
    input_dim = config["model"]["input_dim"]
    hidden_dim = config["model"]["hidden_dim"]
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
    print(f"\n[1/6] Device: {device}")

    # Load datasets
    print(f"\n[2/6] Loading datasets...")
    train_ds = AntibodyAffinityDataset(train_csv)
    val_ds = AntibodyAffinityDataset(val_csv)
    test_ds = AntibodyAffinityDataset(test_csv)

    print(f"      Train: {len(train_ds)} samples")
    print(f"      Val:   {len(val_ds)} samples")
    print(f"      Test:  {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create model
    print(f"\n[3/6] Creating model...")
    model = BindingAffinityGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    print(f"      Parameters: {model.count_parameters():,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # Training loop
    print(f"\n[4/6] Training...")
    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch).squeeze(-1)
            true = batch.y.squeeze(-1)
            loss = criterion(pred, true)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:03d}/{max_epochs} | "
                  f"Train MSE: {avg_loss:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"Val R: {val_metrics['pearson_r']:.4f}")

        # Early stopping
        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n      Early stopping at epoch {epoch}")
            break

    # Restore best model
    print(f"\n[5/6] Loading best model (Val RMSE: {best_val_rmse:.4f})...")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    print(f"\n[6/6] Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 70)
    print("TEST METRICS")
    print("=" * 70)
    print(f"  Pearson R: {test_metrics['pearson_r']:.4f}")
    print(f"  RMSE:      {test_metrics['rmse']:.4f}")
    print(f"  MAE:       {test_metrics['mae']:.4f}")

    # Save model and metrics
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "classical_gnn.pt")
    metrics_path = os.path.join(metrics_dir, "classical_gnn_test.json")

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  Model:   {model_path}")
    print(f"  Metrics: {metrics_path}")
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train()