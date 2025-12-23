"""Run simple smoke checks to validate core placeholders without external test tools.

This script imports key modules and runs minimal functionality using the files
we added to the repository. It avoids third-party test frameworks so it can run
in a minimal Python environment.
"""

# Update: add project root to sys.path so imports work when running script directly
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.pdb_parser import parse_pdb
from src.data.graph_builder import build_graph
from models.classical.binding_gnn import BindingGNN
from models.quantum.quantum_gnn import QuantumGNN


def main():
    print('Running smoke tests...')

    # Parse PDB
    pdb_path = 'data/raw/1A2Y.pdb'
    lines = parse_pdb(pdb_path)
    print(f'Parsed {len(lines)} ATOM lines from {pdb_path}')

    # Build graph
    graph = build_graph(lines)
    print('Built graph:', graph)

    # Classical model
    cmodel = BindingGNN()
    cpred = cmodel.forward(graph)
    print('Classical model forward returned:', cpred)

    # Quantum model
    qmodel = QuantumGNN()
    qpred = qmodel.forward(graph)
    print('Quantum model forward returned:', qpred)

    print('Smoke tests completed successfully.')


if __name__ == '__main__':
    main()
