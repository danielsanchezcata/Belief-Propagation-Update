import numpy as np
import networkx as nx
from arbKernel import read_adjacency_matrix, BP  # Importing from arbKernel.py
import time
import matplotlib.pyplot as plt

# Path to adjacency matrix file
adjacency_matrix_file = "/Users/danielsanchez/Desktop/competitions-data/test.txt"

# Start the timer
start_time = time.time()

# Read the adjacency matrix file and create graph
G = read_adjacency_matrix(adjacency_matrix_file)

# Check the number of edges between nodes
for node in G.nodes():
    successors = list(G.successors(node))
    for succ in successors:
        num_edges = G.number_of_edges(node, succ)
        print(f"Number of edges from {node} to {succ}: {num_edges}")

# Visualise the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42) 

# Draw the nodes and edges
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20)

plt.title("Graph Visualization")
plt.show()

# End the timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

