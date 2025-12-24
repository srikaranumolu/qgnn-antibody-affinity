"""
View what's inside a .pt file
"""
import torch

# Load the .pt file
graph = torch.load('../data/processed/example/1a2y_graph.pt',
                   map_location='cpu',
                   weights_only=False)

print("=" * 70)
print("VIEWING 1A2Y_GRAPH.PT FILE")
print("=" * 70)

print(f"\n📊 BASIC INFO:")
print(f"  Total nodes (atoms): {graph.num_nodes}")
print(f"  Total edges (connections): {graph.num_edges}")
print(f"  Node features shape: {graph.x.shape}")
print(f"  Edge index shape: {graph.edge_index.shape}")

print(f"\n🔢 FIRST 10 NODES (atoms):")
print("Format: [C, N, O, S, P, Other]")
print("-" * 70)
for i in range(10):
    features = graph.x[i].tolist()

    # Decode what element it is
    element_types = ['C', 'N', 'O', 'S', 'P', 'Other']
    element = element_types[features.index(1.0)]

    print(f"  Node {i}: {features} → Element: {element}")

print(f"\n🔗 FIRST 20 EDGES (connections):")
print("Format: Atom X ↔ Atom Y")
print("-" * 70)
for i in range(0, 20, 2):  # Show every other edge (since undirected)
    source = graph.edge_index[0][i].item()
    target = graph.edge_index[1][i].item()
    print(f"  Edge {i // 2}: Atom {source} ↔ Atom {target}")

print(f"\n📍 FIRST 5 ATOM POSITIONS (x, y, z coordinates):")
print("-" * 70)
for i in range(5):
    pos = graph.pos[i].tolist()
    print(f"  Atom {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

print("\n" + "=" * 70)
print("✓ This is the RAW DATA inside the .pt file!")
print("=" * 70)