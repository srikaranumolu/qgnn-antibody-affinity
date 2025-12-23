"""Minimal PDB parser placeholder using simple string parsing.

In the real project this would wrap BioPython's PDB parser.
"""

def parse_pdb(pdb_path):
    """Return a list of atom lines (very small placeholder).
    """
    with open(pdb_path, 'r') as f:
        return [line.strip() for line in f if line.startswith('ATOM')]

