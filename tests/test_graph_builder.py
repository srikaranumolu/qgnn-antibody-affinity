from src.data.graph_builder import build_graph

def test_build_graph():
    g = build_graph(['ATOM ...','ATOM ...'])
    assert g['num_atoms'] == 2

