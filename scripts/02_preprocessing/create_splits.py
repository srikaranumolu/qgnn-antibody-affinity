"""
Create train / validation / test splits for antibody–antigen GNN dataset
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Paths (robust to where the script is run from)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

CLEANED_DATA = os.path.join(DATA_DIR, "saaint_cleaned.csv")

# ---------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------
print("=" * 70)
print("CREATING DATASET SPLITS")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv(CLEANED_DATA)
print(f"\n[1/4] Loaded dataset: {len(df)} entries")

# ---------------------------------------------------------------------
# Check which graph files exist
# ---------------------------------------------------------------------
available_graphs = {
    f.replace("_graph.pt", "").replace("_model_0", "")
    for f in os.listdir(PROCESSED_DIR)
    if f.endswith(".pt")
}

print(f"\n[2/4] Found {len(available_graphs)} graph files")

# Filter dataset to only entries with graphs
df_with_graphs = df[df["PDB_ID"].str.lower().isin(available_graphs)].copy()

print(f"      Matched: {len(df_with_graphs)} entries have graphs")
print(f"      Missing: {len(df) - len(df_with_graphs)} entries without graphs")

# ---------------------------------------------------------------------
# Add graph paths
# ---------------------------------------------------------------------
df_with_graphs["graph_path"] = df_with_graphs["PDB_ID"].apply(
    lambda x: os.path.join(PROCESSED_DIR, f"{x.lower()}_model_0_graph.pt")
)

# Verify graph paths exist
missing = [
    row["PDB_ID"]
    for _, row in df_with_graphs.iterrows()
    if not os.path.exists(row["graph_path"])
]

if missing:
    print(f"\n      WARNING: {len(missing)} graph files not found!")
    print(f"      First few missing: {missing[:5]}")

# ---------------------------------------------------------------------
# Sort for deterministic splits
# ---------------------------------------------------------------------
df_with_graphs = df_with_graphs.sort_values("PDB_ID").reset_index(drop=True)

# ---------------------------------------------------------------------
# Create splits (70 / 15 / 15)
# ---------------------------------------------------------------------
print(f"\n[3/4] Creating splits (70/15/15)...")

train, temp = train_test_split(
    df_with_graphs,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print(f"      Train: {len(train)} ({len(train)/len(df_with_graphs)*100:.1f}%)")
print(f"      Val:   {len(val)} ({len(val)/len(df_with_graphs)*100:.1f}%)")
print(f"      Test:  {len(test)} ({len(test)/len(df_with_graphs)*100:.1f}%)")

# ---------------------------------------------------------------------
# Save splits
# ---------------------------------------------------------------------
os.makedirs(SPLITS_DIR, exist_ok=True)

train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

print(f"\n[4/4] ✓ Saved splits to data/splits/")

# ---------------------------------------------------------------------
# Split statistics
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("SPLIT STATISTICS")
print("=" * 70)

for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    print(f"\n{name}:")
    print(f"  Samples: {len(split_df)}")
    print(f"  pKd range: [{split_df['pKd'].min():.2f}, {split_df['pKd'].max():.2f}]")
    print(f"  pKd mean: {split_df['pKd'].mean():.2f} ± {split_df['pKd'].std():.2f}")

print("\n✓ READY FOR TRAINING!")
