"""Minimal placeholder classical GNN model."""

class BindingGNN:
    def __init__(self, num_layers=3, hidden_dim=64):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, graph):
        """Pretend to compute a prediction from a graph.

        Args:
            graph: placeholder graph object
        Returns:
            float: dummy prediction
        """
        return 0.0

