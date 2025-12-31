"""
3D visualization using actual atom positions
"""
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load graph
graph = torch.load('../../data/processed/6a0z_model_0_graph.pt',
                   map_location='cpu',
                   weights_only=False)

print(f"Loaded graph: {graph.num_nodes} atoms")

# Get positions
positions = graph.pos.numpy()

# Color by element type
node_features = graph.x.numpy()
element_types = ['C', 'N', 'O', 'S', 'P', 'Other']
colors = []
color_map = {'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow', 'P': 'orange', 'Other': 'purple'}

for i in range(len(node_features)):
    element_idx = np.argmax(node_features[i])
    element = element_types[element_idx]
    colors.append(color_map[element])

# Create 3D plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# Plot atoms
ax.scatter(positions[:, 0],
          positions[:, 1],
          positions[:, 2],
          c=colors,
          s=10,
          alpha=0.6)

ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
ax.set_title(f'3D Molecular Structure: 1A2Y ({graph.num_nodes} atoms)',
             fontsize=16, fontweight='bold')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=color, markersize=10, label=elem)
                  for elem, color in color_map.items()]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
save_path = '../../results/figures/3d_molecular_structure.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"✓ Saved to: {save_path}")

print("\nThis shows the ACTUAL 3D structure from the .pt file!")
print("Colors: Carbon=gray, Nitrogen=blue, Oxygen=red, Sulfur=yellow")