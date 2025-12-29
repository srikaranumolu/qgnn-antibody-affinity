import os
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PDB_ID_FILE = os.path.join(BASE_DIR, "data/splits/pdb_ids.txt")
ALL_PDB_DIR = os.path.join(BASE_DIR, "data/raw/saaint_all_pdbs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/raw/saaint_selected_pdbs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load PDB IDs
with open(PDB_ID_FILE) as f:
    pdb_ids = {line.strip().lower() for line in f if line.strip()}

print(f"Loaded {len(pdb_ids)} PDB IDs")

copied = 0

for filename in os.listdir(ALL_PDB_DIR):
    if not filename.endswith(".pdb"):
        continue

    # PDB ID is first 4 characters (e.g., 9x45_model_0.pdb)
    pdb_id = filename[:4].lower()

    if pdb_id in pdb_ids:
        src = os.path.join(ALL_PDB_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, filename)
        shutil.copyfile(src, dst)
        copied += 1

print(f"Copied {copied} PDB files to data/raw/saaint_selected_pdbs/")
