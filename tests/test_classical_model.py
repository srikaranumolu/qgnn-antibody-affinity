from models.classical.binding_gnn import BindingGNN

def test_binding_gnn():
    m = BindingGNN()
    assert hasattr(m, 'forward')

