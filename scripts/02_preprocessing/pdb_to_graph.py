"""
Convert PDB file to PyTorch Geometric graph with RICH MOLECULAR FEATURES
Updated version with 41-dimensional node features for binding affinity prediction
"""

import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
import os


# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================

# Element types (6D one-hot)
ELEMENT_TYPES = ['C', 'N', 'O', 'S', 'P', 'Other']

# Common atom types in proteins (10D one-hot)
ATOM_TYPES = ['CA', 'C', 'N', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'Other']

# Amino acid types (20D one-hot)
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Aromatic residues
AROMATIC_RESIDUES = {'PHE', 'TRP', 'TYR', 'HIS'}

# Backbone atoms
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}

# Hydrogen bond donors (atoms that can donate H)
HB_DONORS = {'N', 'O', 'OG', 'OG1', 'OH', 'NE', 'NH1', 'NH2', 'NZ', 'ND1', 'ND2', 'NE2'}

# Hydrogen bond acceptors (atoms that can accept H)
HB_ACCEPTORS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'N'}

# Charged residues
CHARGED_POSITIVE = {'ARG', 'LYS', 'HIS'}
CHARGED_NEGATIVE = {'ASP', 'GLU'}


def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D points"""
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def get_element_features(element):
    """Convert element to 6D one-hot encoding"""
    idx = ELEMENT_TYPES.index(element) if element in ELEMENT_TYPES else ELEMENT_TYPES.index('Other')
    one_hot = [0] * len(ELEMENT_TYPES)
    one_hot[idx] = 1
    return one_hot


def get_atom_type_features(atom_name):
    """Convert atom name to 10D one-hot encoding"""
    idx = ATOM_TYPES.index(atom_name) if atom_name in ATOM_TYPES else ATOM_TYPES.index('Other')
    one_hot = [0] * len(ATOM_TYPES)
    one_hot[idx] = 1
    return one_hot


def get_residue_features(residue_name):
    """Convert residue to 20D one-hot encoding"""
    idx = AMINO_ACIDS.index(residue_name) if residue_name in AMINO_ACIDS else -1
    one_hot = [0] * len(AMINO_ACIDS)
    if idx != -1:
        one_hot[idx] = 1
    return one_hot


def get_additional_features(atom_name, residue_name):
    """
    Get additional binary/continuous features (5D total)
    Returns: [is_aromatic, is_backbone, is_hb_donor, is_hb_acceptor, charge]
    """
    is_aromatic = 1.0 if residue_name in AROMATIC_RESIDUES else 0.0
    is_backbone = 1.0 if atom_name in BACKBONE_ATOMS else 0.0
    is_hb_donor = 1.0 if atom_name in HB_DONORS else 0.0
    is_hb_acceptor = 1.0 if atom_name in HB_ACCEPTORS else 0.0

    # Charge indicator: +1 for positive, -1 for negative, 0 for neutral
    if residue_name in CHARGED_POSITIVE:
        charge = 1.0
    elif residue_name in CHARGED_NEGATIVE:
        charge = -1.0
    else:
        charge = 0.0

    return [is_aromatic, is_backbone, is_hb_donor, is_hb_acceptor, charge]


def pdb_to_graph(pdb_file, distance_cutoff=5.0):
    """
    Convert PDB file to graph with rich molecular features

    Node features (41D total):
    - Element type: 6D one-hot (C, N, O, S, P, Other)
    - Atom type: 10D one-hot (CA, C, N, O, CB, etc.)
    - Residue type: 20D one-hot (20 amino acids)
    - Is aromatic: 1D binary
    - Is backbone: 1D binary
    - Is HB donor: 1D binary
    - Is HB acceptor: 1D binary
    - Charge: 1D continuous (-1, 0, +1)

    Args:
        pdb_file: Path to PDB file
        distance_cutoff: Connect atoms within this distance (Angstroms)

    Returns:
        PyTorch Geometric Data object with rich features
    """
    print(f"\nProcessing: {pdb_file}")

    # Load PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    # Extract all atoms with residue information
    atoms = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        atoms.append({
            'element': atom.element,
            'atom_name': atom.name,
            'residue_name': residue.resname,
            'coord': atom.get_coord()
        })

    print(f"  Loaded {len(atoms)} atoms")

    # Create rich node features (41D)
    node_features = []
    positions = []

    for atom in atoms:
        element = atom['element']
        atom_name = atom['atom_name']
        residue_name = atom['residue_name']

        # Build 41D feature vector
        features = []

        # Element features (6D)
        features.extend(get_element_features(element))

        # Atom type features (10D)
        features.extend(get_atom_type_features(atom_name))

        # Residue features (20D)
        features.extend(get_residue_features(residue_name))

        # Additional features (5D)
        features.extend(get_additional_features(atom_name, residue_name))

        node_features.append(features)
        positions.append(atom['coord'])

    # Convert to tensors efficiently (numpy first, then torch)
    x = torch.from_numpy(np.array(node_features, dtype=np.float32))
    pos = torch.from_numpy(np.array(positions, dtype=np.float32))

    print(f"  Node features shape: {x.shape} (41D per node)")
    print(f"  Feature breakdown: 6D element + 10D atom + 20D residue + 5D properties")

    # Create edges - connect atoms within distance_cutoff (OPTIMIZED)
    print(f"  Creating edges (cutoff={distance_cutoff}Å)...")

    # Convert positions to numpy array for vectorized operations
    pos_array = np.array(positions, dtype=np.float32)
    num_atoms = len(positions)

    edge_list = []

    # Use vectorized distance calculations in chunks to save memory
    chunk_size = 1000
    for i in range(0, num_atoms, chunk_size):
        end_i = min(i + chunk_size, num_atoms)

        if i % (chunk_size * 5) == 0:
            print(f"    Progress: {i}/{num_atoms} atoms ({100 * i // num_atoms}%)")

        # Calculate distances for chunk
        for idx in range(i, end_i):
            # Vectorized distance calculation to all subsequent atoms
            distances = np.linalg.norm(pos_array[idx+1:] - pos_array[idx], axis=1)

            # Find atoms within cutoff
            neighbors = np.where(distances < distance_cutoff)[0] + (idx + 1)

            # Add edges (both directions for undirected graph)
            for j in neighbors:
                edge_list.append([idx, j])
                edge_list.append([j, idx])

    print(f"  ✓ Created {len(edge_list)} edges")

    # Convert edges to tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, pos=pos)

    # Sanity check
    print(f"  Feature stats:")
    print(f"    Mean: {x.mean(dim=0)[:10].tolist()}")  # First 10 dims
    print(f"    Unique feature vectors: {torch.unique(x, dim=0).shape[0]}")

    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    pdb_ids = os.path.join(PROJECT_ROOT, "data", "pdb_ids.txt")

    with open(pdb_ids, 'r') as pdbs:
        ids = [
            os.path.join(
                PROJECT_ROOT,
                "data",
                "raw",
                "saaint_selected_pdbs",
                line.strip() + "_model_0.pdb"
            )
            for line in pdbs
            if line.strip()
        ]

    # ========================================================================
    # CONFIGURE RANGE HERE - Split work between you and your partner!
    # ========================================================================
    # Example splits for 705 files:
    # Partner 1: start=0,   end=353  (first half)
    # Partner 2: start=353, end=705  (second half)
    # ========================================================================

    start = 0      # ← CHANGE THIS
    end = 5 # ← CHANGE THIS (or use len(ids) for all remaining)

    # Safety checks
    end = min(end, len(ids))
    start = max(0, min(start, end))

    pdb_files = ids[start:end]

    last_file_basename = os.path.basename(pdb_files[-1]) if pdb_files else 'N/A'
    print(f"Processing {len(pdb_files)} files from {start} to {end}")
    print(f"Last file: {last_file_basename}")

    print("=" * 70)
    print("CONVERTING PDB FILES TO GRAPHS WITH RICH FEATURES")
    print("=" * 70)

    results = []
    graph_objects = []

    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    figures_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    for pdb_file in pdb_files:
        try:
            # Convert to graph with rich features
            graph = pdb_to_graph(pdb_file, distance_cutoff=5.0)

            pdb_id = os.path.basename(pdb_file).replace('.pdb', '')

            # Save graph
            save_path = os.path.join(processed_dir, f'{pdb_id}_graph.pt')
            torch.save(graph, save_path)
            print(f"  💾 Saved to: {save_path}")

            result = {
                'pdb_id': pdb_id,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'feature_dim': graph.x.shape[1],
                'status': '✓ SUCCESS'
            }
            results.append(result)
            graph_objects.append((pdb_id, graph))

            print(f"  ✓ Graph saved: {graph.num_nodes} nodes, {graph.num_edges} edges, {graph.x.shape[1]}D features\n")

        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            results.append({
                'pdb_id': os.path.basename(pdb_file).replace('.pdb', ''),
                'status': f'✗ FAILED: {e}'
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'PDB ID':<12} {'Nodes':<8} {'Edges':<10} {'Features':<10} {'Status':<20}")
    print("-" * 70)

    for r in results:
        nodes = r.get('num_nodes', 'N/A')
        edges = r.get('num_edges', 'N/A')
        feat_dim = r.get('feature_dim', 'N/A')
        print(f"{r['pdb_id']:<12} {nodes:<8} {edges:<10} {feat_dim:<10} {r['status']:<20}")

    print(f"\n✓ Conversion complete! Graphs with 41D features saved to {processed_dir}")

    # Simple visualization (optional)
    if graph_objects:
        print("\nFirst graph feature check:")
        pdb_id, graph = graph_objects[0]
        print(f"  {pdb_id}: {graph.x.shape} features")
        print(f"  Sample features (first node): {graph.x[0][:10].tolist()}")