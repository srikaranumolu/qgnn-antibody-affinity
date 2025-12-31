import os
import math
import json
import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
import sys



PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from models.classical.binding_gnn import BindingAffinityGNN
from scripts.preprocessing.prepare_labeled_graphs import GraphAffinityDataset



def rmse(y_pred, y_true):
    return math.sqrt(np.mean((y_pred - y_true) ** 2))


def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)  # [batch]
        preds.extend(out.detach().cpu().numpy().tolist())
        trues.extend(batch.y.squeeze(-1).detach().cpu().numpy().tolist())

    preds = np.array(preds, dtype=float)
    trues = np.array(trues, dtype=float)

    r, p = pearsonr(preds, trues) if len(preds) > 1 else (float("nan"), float("nan"))
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": float(rmse(preds, trues)),
        "mae": float(mae(preds, trues)),
    }


def train():
    # Paths (match your project structure)
    processed_dir = "data/processed"
    train_csv = "data/splits/train.csv"
    val_csv = "data/splits/val.csv"
    test_csv = "data/splits/test.csv"

    # Hyperparameters
    input_dim = 6
    hidden_dim = 128
    dropout = 0.3
    batch_size = 16
    lr = 1e-3
    max_epochs = 200
    patience = 20  # early stopping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets + loaders
    train_ds = GraphAffinityDataset(processed_dir, train_csv)
    val_ds = GraphAffinityDataset(processed_dir, val_csv)
    test_ds = GraphAffinityDataset(processed_dir, test_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = BindingAffinityGNN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    print(model)
    print(f"Trainable params: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop with early stopping
    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)          # [batch]
            true = batch.y.squeeze(-1)              # [batch]
            loss = criterion(pred, true)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | Train MSE: {avg_loss:.4f} "
            f"| Val RMSE: {val_metrics['rmse']:.4f} | Val R: {val_metrics['pearson_r']:.4f}"
        )

        # Early stopping on val RMSE
        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics
    test_metrics = evaluate(model, test_loader, device)
    print("\nTEST METRICS:")
    print(json.dumps(test_metrics, indent=2))

    # Save model + metrics
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    torch.save(model.state_dict(), "results/models/classical_gnn.pt")
    with open("results/metrics/classical_gnn_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved:")
    print("  results/models/classical_gnn.pt")
    print("  results/metrics/classical_gnn_test.json")


if __name__ == "__main__":
    train()
