import os

# CHANGE THESE PATHS
processed_dir = r"../../data/processed"
pdb_ids = r"..\..\data\pdb_ids.txt"

# Read expected filenames
with open(pdb_ids, "r") as f:
    expected_files = {line.strip()+ "_model_0_graph.pt" for line in f if line.strip()}

# Get actual filenames in folder
actual_files = set(os.listdir(processed_dir))

# Find differences
missing_files = expected_files - actual_files
extra_files = actual_files - expected_files

print(f"Expected files: {len(expected_files)}")
print(f"Actual files: {len(actual_files)}")

print("\nMissing files:")
for file in sorted(missing_files):
    print(file)

print("\nExtra files:")
for file in sorted(extra_files):
    print(file)
