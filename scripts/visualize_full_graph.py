"""
Visualize the FULL molecular graph (warning: slow and messy!)
"""
import torch
import matplotlib.pyplot as plt
import networkx as nx

print("Loading graph...")
graph = torch.load('../data/processed/example/1a2y_graph.pt',
                   map_location='cpu',
                   weights_only=False)

print(f"Graph size: {graph.num_nodes} nodes, {graph.num_edges} edges")
print("⚠️  WARNING: This will take 2-5 minutes and look like a blob!\n")

# Create NetworkX graph with ALL nodes
G = nx.Graph()
G.add_nodes_from(range(graph.num_nodes))

# Add ALL edges
print("Adding edges (this is slow)...")
edge_index = graph.edge_index.numpy()
for i in range(0, edge_index.shape[1], 2):  # Skip reverse edges
    G.add_edge(edge_index[0, i], edge_index[1, i])

print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Layout (this takes a while!)
print("Computing layout (this takes 2-3 minutes)...")
pos = nx.spring_layout(G, k=0.1, iterations=20, seed=42)

# Draw
print("Drawing (this also takes time)...")
plt.figure(figsize=(20, 20))

nx.draw_networkx_nodes(G, pos,
                      node_size=5,
                      node_color='lightblue',
                      alpha=0.6)

nx.draw_networkx_edges(G, pos,
                      edge_color='gray',
                      alpha=0.1,
                      width=0.2)

plt.title(f"Full Molecular Graph: 1A2Y (ALL {graph.num_nodes} atoms)",
          fontsize=20, fontweight='bold')
plt.axis('off')
plt.tight_layout()

save_path = '../results/figures/full_molecular_graph.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to: {save_path}")