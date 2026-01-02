from models.classical.binding_gat import BindingGNN

def test_binding_gnn():
    m = BindingGNN()
    assert hasattr(m, 'forward')

