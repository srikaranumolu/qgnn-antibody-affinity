"""
Convert PDB file to PyTorch Geometric graph
"""

import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
import os


def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D points"""
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def pdb_to_graph(pdb_file, distance_cutoff=5.0):
    """
    Convert PDB file to graph

    Args:
        pdb_file: Path to PDB file
        distance_cutoff: Connect atoms within this distance (Angstroms)

    Returns:
        PyTorch Geometric Data object
    """
    print(f"\nProcessing: {pdb_file}")

    # Load PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    # Extract all atoms
    atoms = []
    for atom in structure.get_atoms():
        atoms.append({
            'element': atom.element,
            'coord': atom.get_coord()
        })

    print(f"  Loaded {len(atoms)} atoms")

    # Define element types for one-hot encoding
    element_types = ['C', 'N', 'O', 'S', 'P', 'Other']
    element_to_idx = {e: i for i, e in enumerate(element_types)}

    # Create node features
    node_features = []
    positions = []

    for atom in atoms:
        element = atom['element']

        # Map to element type
        if element not in element_to_idx:
            element = 'Other'
        idx = element_to_idx[element]

        # One-hot encode: [0,0,1,0,0,0] for O (oxygen)
        one_hot = [0] * len(element_types)
        one_hot[idx] = 1

        node_features.append(one_hot)
        positions.append(atom['coord'])

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    print(f"  Node features shape: {x.shape}")

    # Create edges - connect atoms within distance_cutoff
    print(f"  Creating edges (cutoff={distance_cutoff}Å)...")

    edge_list = []
    num_atoms = len(positions)

    # Progress tracking
    progress_interval = max(num_atoms // 10, 1)

    for i in range(num_atoms):
        if i % progress_interval == 0:
            print(f"    Progress: {i}/{num_atoms} atoms ({100 * i // num_atoms}%)")

        for j in range(i + 1, num_atoms):
            dist = calculate_distance(positions[i], positions[j])

            if dist < distance_cutoff:
                # Undirected graph - add both directions
                edge_list.append([i, j])
                edge_list.append([j, i])

    print(f"  ✓ Created {len(edge_list)} edges")

    # Convert edges to tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, pos=pos)

    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

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

    # Simple hard-coded slice (edit these to change which subset is processed)
    # Set start/end here. `start` is inclusive, `end` is exclusive (Python slice semantics).
    # Example: start=0, end=353 will process the first 353 entries (indices 0..352).
    start = 664
    end = len(ids)

    # Safety clamp so we don't go past available ids
    end = min(end, len(ids))
    start = max(0, min(start, end))

    pdb_files = ids[start:end]

    last_file_basename = os.path.basename(pdb_files[-1]) if pdb_files else 'N/A'
    print(f"Using ids {start}:{end} (actual {len(pdb_files)}) from `{pdb_ids}` out of the total {len(ids)} with the last file that will be processed being: {last_file_basename}")

    print("=" * 70)
    print("CONVERTING PDB FILES TO GRAPHS")
    print("=" * 70)

    results = []
    graph_objects = []  # Store graphs for 05_visualization

    # Use absolute output directories based on PROJECT_ROOT
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    figures_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')

    # Ensure output directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    for pdb_file in pdb_files:
        try:
            # Convert to graph
            graph = pdb_to_graph(pdb_file, distance_cutoff=5.0)

            # Get PDB ID (basename, cross-platform)
            pdb_id = os.path.basename(pdb_file).replace('.pdb', '')

            # SAVE THE GRAPH using absolute path
            save_path = os.path.join(processed_dir, f'{pdb_id}_graph.pt')
            torch.save(graph, save_path)
            print(f"  💾 Saved to: {save_path}")

            result = {
                'pdb_id': pdb_id,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'status': '✓ SUCCESS'
            }
            results.append(result)
            graph_objects.append((pdb_id, graph))  # Store for viz

            print(f"  ✓ Graph saved: {graph.num_nodes} nodes, {graph.num_edges} edges\n")

        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            results.append({
                'pdb_id': os.path.basename(pdb_file).replace('.pdb', ''),
                'status': f'✗ FAILED: {e}'
            })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'PDB ID':<10} {'Nodes':<10} {'Edges':<12} {'Status':<20}")
    print("-" * 70)

    for r in results:
        nodes = r.get('num_nodes', 'N/A')
        edges = r.get('num_edges', 'N/A')
        print(f"{r['pdb_id']:<10} {nodes:<10} {edges:<12} {r['status']:<20}")

    print(f"\n✓ Conversion complete! Graphs saved to {processed_dir}")

    # ============================================================
    # VISUALIZATION - Sample the smallest graph
    # ============================================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    # Use first graph for 05_visualization (if available)
    if not graph_objects:
        print("No graphs to visualize.")
    else:
        pdb_id, graph = graph_objects[0]
        print(f"Visualizing {pdb_id} (first 150 atoms)...")

        # Sample first 150 nodes (full graph too big)
        sample_size = min(150, graph.num_nodes)

        # Get edges that connect nodes within sample
        edge_index = graph.edge_index.numpy()
        mask = (edge_index[0] < sample_size) & (edge_index[1] < sample_size)
        sampled_edges = edge_index[:, mask]

        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(sample_size))

        # Add edges
        for i in range(sampled_edges.shape[1]):
            G.add_edge(sampled_edges[0, i], sampled_edges[1, i])

        print(f"Sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Create 05_visualization
        plt.figure(figsize=(12, 12))

        # Use spring layout for nice positioning
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_size=30,
                               node_color='lightblue',
                               alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos,
                               edge_color='gray',
                               alpha=0.3,
                               width=0.5)

        plt.title(f"Molecular Graph: {pdb_id.upper()} (first {sample_size} atoms)",
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save figure using absolute path
        viz_path = os.path.join(figures_dir, 'sample_molecular_graph.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_path}")

        # Also create a statistics plot
        plt.figure(figsize=(10, 6))

        pdb_ids = [r['pdb_id'] for r in results if 'num_nodes' in r]
        node_counts = [r['num_nodes'] for r in results if 'num_nodes' in r]
        edge_counts = [r['num_edges'] for r in results if 'num_nodes' in r]

        x = range(len(pdb_ids))
        width = 0.35

        plt.bar([i - width / 2 for i in x], node_counts, width, label='Nodes', alpha=0.8)
        plt.bar([i + width / 2 for i in x], [e / 20 for e in edge_counts], width,
                label='Edges/20', alpha=0.8)

        plt.xlabel('PDB Structure', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title('Graph Sizes for Each Structure', fontsize=14, fontweight='bold')
        plt.xticks(x, pdb_ids)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        stats_path = os.path.join(figures_dir, 'graph_statistics.png')
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        print(f"✓ Statistics plot saved to: {stats_path}")

    print("\n🎉 All done! Check the results folder for visualizations!")
