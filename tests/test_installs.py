print("Testing installations...\n")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except:
    print("✗ PyTorch FAILED")

try:
    import torch_geometric
    print("✓ PyTorch Geometric")
except:
    print("✗ PyTorch Geometric FAILED")

try:
    import pennylane
    print(f"✓ PennyLane {pennylane.__version__}")
except:
    print("✗ PennyLane FAILED")

try:
    from rdkit import Chem
    print("✓ RDKit")
except:
    print("✗ RDKit FAILED")

try:
    from Bio import PDB
    print("✓ BioPython")
except:
    print("✗ BioPython FAILED")

try:
    import pandas, numpy, matplotlib, seaborn, scipy
    print("✓ Data science libraries")
except:
    print("✗ Data libraries FAILED")

print("\n✅ If you see ✓ for everything, you're ready!")