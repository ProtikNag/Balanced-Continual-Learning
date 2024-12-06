from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="./data/TUDataset", name='COIL-RAG', use_node_attr=True)

print(f"Dataset: {dataset}")
print(f"Number of Graphs: {len(dataset)}")
print(f"Number of Classes: {dataset.num_classes}")
print(f"Number of Node Features: {dataset.num_node_features}")
print(f"Number of Edge Features: {dataset.num_edge_features}")

data = dataset[25]

# Print the properties of the graph
print("Graph Properties:")
print(f"Number of Nodes: {data.num_nodes}")
print(f"Number of Edges: {data.num_edges}")
print(f"Node Features: {data.x.shape if data.x is not None else 'None'}")
print(f"Edge Features: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
print(f"Graph Label: {data.y.item()}")
print(f"Edge Index:\n{data.edge_index}")
print(f"Edge Attributes:\n{data.edge_attr}")


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convert PyTorch Geometric graph to NetworkX
graph = to_networkx(data, to_undirected=True)

# Visualize the graph
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
plt.title(f"Graph Visualization (Label: {data.y.item()})")
plt.show()