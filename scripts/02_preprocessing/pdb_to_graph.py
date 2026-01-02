"""
Advanced PDB to Graph Converter with Interface-Aware Features
==============================================================

Converts protein structures to graphs with rich 49-dimensional node features
optimized for antibody-antigen binding affinity prediction.

Feature Breakdown (49D total):
- Element type: 6D one-hot
- Atom type: 10D one-hot
- Residue type: 20D one-hot
- Aromatic: 1D binary
- Backbone: 1D binary
- H-bond donor: 1D binary
- H-bond acceptor: 1D binary
- Residue charge: 1D continuous
- Hydrophobicity: 1D continuous (Kyte-Doolittle scale)
- B-factor: 1D continuous (flexibility/uncertainty)
- Chain ID: 1D continuous (0=H/L chains, 1=antigen)
- Distance to interface: 1D continuous
- Is at interface: 1D binary
- Local density: 1D continuous
- SASA: 1D continuous (solvent accessible surface area)
- Electrostatic: 1D continuous (charge * SASA)

Author: RSEF Quantum Antibody Binding Project
Date: January 2026
"""

import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


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

# Hydrogen bond donors/acceptors
HB_DONORS = {'N', 'O', 'OG', 'OG1', 'OH', 'NE', 'NH1', 'NH2', 'NZ', 'ND1', 'ND2', 'NE2'}
HB_ACCEPTORS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'N'}

# Charged residues
CHARGED_POSITIVE = {'ARG', 'LYS', 'HIS'}
CHARGED_NEGATIVE = {'ASP', 'GLU'}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}


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
    Get additional binary/continuous features (5D)
    Returns: [is_aromatic, is_backbone, is_hb_donor, is_hb_acceptor, charge]
    """
    is_aromatic = 1.0 if residue_name in AROMATIC_RESIDUES else 0.0
    is_backbone = 1.0 if atom_name in BACKBONE_ATOMS else 0.0
    is_hb_donor = 1.0 if atom_name in HB_DONORS else 0.0
    is_hb_acceptor = 1.0 if atom_name in HB_ACCEPTORS else 0.0

    # Charge indicator
    if residue_name in CHARGED_POSITIVE:
        charge = 1.0
    elif residue_name in CHARGED_NEGATIVE:
        charge = -1.0
    else:
        charge = 0.0

    return [is_aromatic, is_backbone, is_hb_donor, is_hb_acceptor, charge]


def get_hydrophobicity(residue_name):
    """Get Kyte-Doolittle hydrophobicity score (normalized to [-1, 1])"""
    raw_score = HYDROPHOBICITY.get(residue_name, 0.0)
    # Normalize: range is approximately [-4.5, 4.5]
    normalized = raw_score / 4.5
    return normalized


def calculate_sasa_simple(pos_array, atom_idx, probe_radius=1.4):
    """
    Simple solvent accessible surface area approximation
    Based on number of nearby atoms (inverse relationship)
    """
    coord = pos_array[atom_idx]
    distances = np.linalg.norm(pos_array - coord, axis=1)

    # Count atoms within probe radius + typical atom radius (3.5Å)
    nearby_count = np.sum((distances > 0) & (distances < 3.5 + probe_radius))

    # Normalize: 0 (buried) to 1 (exposed)
    # Typical max neighbors ~ 20-30
    sasa_score = max(0, 1.0 - nearby_count / 25.0)
    return sasa_score


def detect_interface(pos_array, chain_types, interface_cutoff=10.0):
    """
    Detect binding interface between antibody and antigen
    Returns: centroid of interface region
    """
    # Separate antibody and antigen atoms
    ab_mask = chain_types == 0
    ag_mask = chain_types == 1

    if not ab_mask.any() or not ag_mask.any():
        # If can't separate chains, return center of mass
        return pos_array.mean(axis=0)

    ab_coords = pos_array[ab_mask]
    ag_coords = pos_array[ag_mask]

    # Find interface atoms (antibody atoms close to antigen)
    interface_atoms = []
    for ab_coord in ab_coords:
        distances = np.linalg.norm(ag_coords - ab_coord, axis=1)
        if distances.min() < interface_cutoff:
            interface_atoms.append(ab_coord)

    if len(interface_atoms) == 0:
        # Fallback to midpoint between centroids
        ab_center = ab_coords.mean(axis=0)
        ag_center = ag_coords.mean(axis=0)
        return (ab_center + ag_center) / 2

    # Interface centroid
    interface_centroid = np.mean(interface_atoms, axis=0)
    return interface_centroid


def identify_chain_types(structure):
    """
    Identify which chains are antibody vs antigen
    Heuristic: Chains H/L are usually antibody, others are antigen
    Returns: dict mapping chain_id -> 0 (antibody) or 1 (antigen)
    """
    chain_map = {}
    chains = list(structure.get_chains())

    for chain in chains:
        chain_id = chain.id
        # Common antibody chain IDs
        if chain_id in {'H', 'L', 'A', 'B'}:
            chain_map[chain_id] = 0  # Antibody
        else:
            chain_map[chain_id] = 1  # Antigen

    # If only 1-2 chains, assume first is antibody, rest antigen
    if len(chain_map) <= 2:
        for i, chain_id in enumerate(chain_map.keys()):
            chain_map[chain_id] = 0 if i == 0 else 1

    return chain_map


def pdb_to_graph(pdb_file, distance_cutoff=5.0, interface_cutoff=10.0):
    """
    Convert PDB file to graph with advanced 49D node features

    Args:
        pdb_file: Path to PDB file
        distance_cutoff: Distance for edge creation (Angstroms)
        interface_cutoff: Distance for interface detection (Angstroms)

    Returns:
        PyTorch Geometric Data object with 49D features per node
    """
    print(f"\nProcessing: {os.path.basename(pdb_file)}")

    # Load PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    # Identify chain types
    chain_map = identify_chain_types(structure)

    # Extract all atoms with metadata
    atoms = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()

        atoms.append({
            'element': atom.element,
            'atom_name': atom.name,
            'residue_name': residue.resname,
            'coord': atom.get_coord(),
            'bfactor': atom.get_bfactor(),
            'chain_id': chain.id,
            'chain_type': chain_map.get(chain.id, 1)  # Default to antigen if unknown
        })

    print(f"  Loaded {len(atoms)} atoms from {len(chain_map)} chains")

    # Create position array for vectorized operations
    pos_array = np.array([a['coord'] for a in atoms], dtype=np.float32)
    chain_types = np.array([a['chain_type'] for a in atoms], dtype=np.float32)

    # Detect interface
    interface_centroid = detect_interface(pos_array, chain_types, interface_cutoff)
    print(f"  Interface centroid: [{interface_centroid[0]:.1f}, {interface_centroid[1]:.1f}, {interface_centroid[2]:.1f}]")

    # Build 49D feature vectors
    node_features = []

    for idx, atom in enumerate(atoms):
        element = atom['element']
        atom_name = atom['atom_name']
        residue_name = atom['residue_name']
        coord = atom['coord']
        bfactor = atom['bfactor']
        chain_type = atom['chain_type']

        # Build feature vector
        features = []

        # Original 41D features
        features.extend(get_element_features(element))              # 6D
        features.extend(get_atom_type_features(atom_name))          # 10D
        features.extend(get_residue_features(residue_name))         # 20D
        features.extend(get_additional_features(atom_name, residue_name))  # 5D

        # NEW: Advanced features (8D)
        # 1. Hydrophobicity (normalized)
        hydrophobicity = get_hydrophobicity(residue_name)
        features.append(hydrophobicity)

        # 2. B-factor (normalized to [0, 1], typical range 10-100)
        bfactor_norm = min(bfactor / 100.0, 1.0)
        features.append(bfactor_norm)

        # 3. Chain type (0 = antibody, 1 = antigen)
        features.append(float(chain_type))

        # 4. Distance to interface (normalized)
        dist_to_interface = np.linalg.norm(coord - interface_centroid)
        dist_norm = dist_to_interface / 50.0  # Normalize by typical protein size
        features.append(dist_norm)

        # 5. Is at interface (binary)
        is_at_interface = 1.0 if dist_to_interface < interface_cutoff else 0.0
        features.append(is_at_interface)

        # 6. Local density (normalized)
        distances = np.linalg.norm(pos_array - coord, axis=1)
        local_density = np.sum((distances > 0) & (distances < 5.0))  # Count within 5Å
        density_norm = min(local_density / 30.0, 1.0)  # Normalize
        features.append(density_norm)

        # 7. SASA approximation
        sasa = calculate_sasa_simple(pos_array, idx)
        features.append(sasa)

        # 8. Electrostatic potential (enhanced from simple charge)
        # Consider both residue charge and local environment
        base_charge = features[40]  # The charge we computed earlier
        # Modulate by SASA (exposed charges matter more)
        electrostatic = base_charge * (0.5 + 0.5 * sasa)
        features.append(electrostatic)

        node_features.append(features)

    # Convert to tensors efficiently
    x = torch.from_numpy(np.array(node_features, dtype=np.float32))
    pos = torch.from_numpy(pos_array)

    print(f"  Node features: {x.shape} (49D per node)")

    # Create edges using optimized vectorized approach
    print(f"  Creating edges (cutoff={distance_cutoff}Å)...")

    edge_list = []
    num_atoms = len(pos_array)

    # Vectorized distance calculations in chunks
    chunk_size = 1000
    for i in range(0, num_atoms, chunk_size):
        end_i = min(i + chunk_size, num_atoms)

        if i % (chunk_size * 5) == 0:
            print(f"    Progress: {i}/{num_atoms} atoms ({100 * i // num_atoms}%)")

        for idx in range(i, end_i):
            # Vectorized distance calculation
            distances = np.linalg.norm(pos_array[idx+1:] - pos_array[idx], axis=1)
            neighbors = np.where(distances < distance_cutoff)[0] + (idx + 1)

            # Add edges (undirected)
            for j in neighbors:
                edge_list.append([idx, j])
                edge_list.append([j, idx])

    print(f"  ✓ Created {len(edge_list)} edges")

    # Convert edges to tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, pos=pos)

    # Feature statistics
    print(f"  Feature diversity: {torch.unique(x, dim=0).shape[0]} unique vectors")
    print(f"  Non-zero features/node: {(x != 0).sum(dim=1).float().mean():.1f}")
    print(f"  Interface atoms: {(x[:, 44] == 1).sum().item()}")  # Column 44 = is_at_interface

    return data


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    pdb_ids_file = os.path.join(PROJECT_ROOT, "data", "pdb_ids.txt")

    with open(pdb_ids_file, 'r') as f:
        ids = [
            os.path.join(
                PROJECT_ROOT,
                "data",
                "raw",
                "saaint_selected_pdbs",
                line.strip() + "_model_0.pdb"
            )
            for line in f
            if line.strip()
        ]

    # ========================================================================
    # CONFIGURE RANGE - Split work between partners
    # ========================================================================
    start = 0
    end = len(ids)  # Process all by default

    end = min(end, len(ids))
    start = max(0, min(start, end))

    pdb_files = ids[start:end]

    print("=" * 70)
    print("CONVERTING PDB FILES TO GRAPHS (49D FEATURES)")
    print("=" * 70)
    print(f"Processing {len(pdb_files)} files from index {start} to {end}")

    results = []
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    for pdb_file in pdb_files:
        try:
            # Convert to graph
            graph = pdb_to_graph(pdb_file, distance_cutoff=5.0, interface_cutoff=10.0)

            pdb_id = os.path.basename(pdb_file).replace('.pdb', '')

            # Save graph
            save_path = os.path.join(processed_dir, f'{pdb_id}_graph.pt')
            torch.save(graph, save_path)

            results.append({
                'pdb_id': pdb_id,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'feature_dim': graph.x.shape[1],
                'interface_atoms': (graph.x[:, 44] == 1).sum().item(),
                'status': '✓'
            })

            print(f"  ✓ Saved: {pdb_id} ({graph.num_nodes} nodes, {graph.num_edges} edges)\n")

        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            results.append({
                'pdb_id': os.path.basename(pdb_file).replace('.pdb', ''),
                'status': f'✗ {str(e)[:50]}'
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'PDB ID':<12} {'Nodes':<8} {'Edges':<10} {'Interface':<10} {'Status':<10}")
    print("-" * 70)

    for r in results:
        nodes = r.get('num_nodes', 'N/A')
        edges = r.get('num_edges', 'N/A')
        interface = r.get('interface_atoms', 'N/A')
        status = r['status']
        print(f"{r['pdb_id']:<12} {nodes:<8} {edges:<10} {interface:<10} {status:<10}")

    success_count = sum(1 for r in results if r['status'] == '✓')
    print(f"\n✓ Successfully processed {success_count}/{len(results)} structures")
    print(f"✓ Graphs saved to: {processed_dir}")