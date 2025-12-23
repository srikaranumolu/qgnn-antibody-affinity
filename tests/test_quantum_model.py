from models.quantum.quantum_gnn import QuantumGNN

def test_quantum_gnn():
    m = QuantumGNN()
    assert hasattr(m, 'forward')

