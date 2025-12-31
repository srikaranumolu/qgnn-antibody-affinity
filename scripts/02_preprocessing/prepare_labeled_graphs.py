import os
import torch
import pandas as pd

CSV_PATH = "../../data/saaint_cleaned.csv"
GRAPHS_DIR = "../../data/processed/"

df = pd.read_csv(CSV_PATH, usecols=["PDB_ID", "pKd"])
df["PDB_ID"] = df["PDB_ID"].str.lower()

files = os.listdir(GRAPHS_DIR)

labeled = 0
missing = set()

for _, row in df.iterrows():
    pdb_id = row["PDB_ID"]
    pkd = row["pKd"]

    matching_files = [
        f for f in files
        if f.startswith(f"{pdb_id}_")
        and f.endswith("_graph.pt")
    ]

    if not matching_files:
        missing.add(pdb_id)
        continue

    for fname in matching_files:
        path = os.path.join(GRAPHS_DIR, fname)
        graph = torch.load(path, weights_only=False)
        graph.y = torch.tensor([pkd], dtype=torch.float)
        torch.save(graph, path)
        labeled += 1

print(f"✓ Labeled {labeled} graphs")

if missing:
    print(f"⚠ Missing graphs for {len(missing)} PDB IDs")
    print("First 10 missing:", list(missing)[:10])
