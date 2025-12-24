"""
Simple script to inspect PDB files
"""

from Bio.PDB import PDBParser
import sys


def inspect_pdb(pdb_file):
    """
    Load PDB and print basic information
    """
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {pdb_file}")
    print('=' * 60)

    # Create parser
    parser = PDBParser(QUIET=True)

    # Load structure
    structure = parser.get_structure('complex', pdb_file)

    # Count atoms and chains
    total_atoms = 0
    chains_info = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_atoms = list(chain.get_atoms())
            chains_info[chain_id] = len(chain_atoms)
            total_atoms += len(chain_atoms)

    # Print results
    print(f"\nTotal atoms: {total_atoms}")
    print(f"Number of chains: {len(chains_info)}")
    print(f"\nChains breakdown:")
    for chain_id, atom_count in chains_info.items():
        print(f"  Chain {chain_id}: {atom_count} atoms")

    # Get first few atoms as examples
    print(f"\nFirst 5 atoms:")
    atom_list = list(structure.get_atoms())
    for i, atom in enumerate(atom_list[:5]):
        coord = atom.get_coord()
        print(f"  {i + 1}. {atom.element:2s} at ({coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f})")

    return structure, chains_info, total_atoms


if __name__ == "__main__":
    # Test on all 5 downloaded PDB files
    pdb_files = [
        '../data/raw/example/1a2y.pdb',
        '../data/raw/example/1fbi.pdb',
        '../data/raw/example/1dqj.pdb',
        '../data/raw/example/1fns.pdb',
        '../data/raw/example/1bj1.pdb'
    ]

    print("INSPECTING ALL PDB FILES")
    print("=" * 60)

    results = {}

    for pdb_file in pdb_files:
        try:
            structure, chains, total = inspect_pdb(pdb_file)
            pdb_id = pdb_file.split('/')[-1].replace('.pdb', '')
            results[pdb_id] = {
                'chains': chains,
                'total_atoms': total
            }
        except Exception as e:
            print(f"\nERROR loading {pdb_file}: {e}")

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(f"{'PDB ID':<10} {'Chains':<15} {'Total Atoms':<15}")
    print('-' * 60)

    for pdb_id, info in results.items():
        chains_str = ', '.join([f"{c}:{n}" for c, n in info['chains'].items()])
        print(f"{pdb_id:<10} {chains_str:<15} {info['total_atoms']:<15}")

    print("\n✓ Inspection complete!")