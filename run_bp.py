import numpy as np
import networkx as nx
from arbKernel import read_adjacency_matrix, BP  # Importing from arbKernel.py
import time
import matplotlib.pyplot as plt

# Path to adjacency matrix file
adjacency_matrix_file = "/Users/danielsanchez/Desktop/competitions-data/test.txt"

# Start the timer
start_time = time.time()

# Read the adjacency matrix file
edges_array, B_array, N_array, num_nodes = read_adjacency_matrix(adjacency_matrix_file)

print(num_nodes)

# Set parameters for BP
L = 32 # Number of Chebyshev points
alpha = 0.2  # Luck parameter
beta = 1 # Depth of competition parameter
gamma = 0.1 # Magnitude of skills parameter
w = 4  # Scale parameter for Cauchy distribution
tol = 1e-5  # Tolerance for BP

# Run BP
pts, A, S, lnZ, samples = BP(edges_array, B_array, N_array, num_nodes, L, alpha, beta, tol)
# Calculate the log likelihood
log_likelihood = lnZ - num_nodes * np.log(2)
# Calculate the prior term for beta (Cauchy distribution)
log_prior_beta = np.log((2 * w) / (np.pi * (beta**2 + w**2)))
# Calculate the average scores for each node
average_scores = np.array([np.sum(pts * A[i]) / np.sum(A[i]) for i in range(num_nodes)])
# Calculate the log posterior
log_posterior = log_likelihood + log_prior_beta 

# Output results
print(f"Value for luck: {alpha}")
print(f"Value for depth: {beta}")
print(f"Chebyshev Points: {pts}")
#print(f"Marginals: {A}")
print(f"Entropy: {S}")
print(f"Log Partition Function: {lnZ}")
print(f"Log Likelihood : {log_likelihood}")
print(f"Log Prior Beta: {log_prior_beta}")
print(f"Log Posterior : {log_posterior}")
samples_sorted = sorted(samples, key=lambda x: (x[0], x[1]))
print(f"Sampled Skills : {[(i, j, round(float(skill_i), 4), round(float(skill_j), 4)) for i, j, skill_i, skill_j in samples_sorted]}")



# End the timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print(f"Average scores: {average_scores}")

plt.figure(figsize=(12, 8))
for i in range(num_nodes):
    color = plt.cm.tab10(i % 10)  # Using a colormap to assign unique colors to each node's curve
    plt.plot(pts, A[i], label=f"Node {i}", color=color, linewidth=2, alpha=0.5)
    plt.scatter(average_scores[i], 0, color=color, marker='x', s=100)



plt.xlabel("x (Scores)")
plt.ylabel("Marginal Probability Density")
plt.title(f"Marginal Skill Distributions for 'Test'")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Nodes")
plt.tight_layout()
plt.grid(True)
plt.xlim([-1, 1])
plt.show()

'''
#Plot the marginal distribution over x = [-1, 1]
plt.figure(figsize=(12, 8))
plt.plot(pts, A[0], label = f"Marginal Distribution node {0}", color="b", linewidth=2)
plt.plot(pts, A[1], label = f"Marginal Distribution node {1}", color="r", linewidth=2)
plt.plot(pts, A[2], label = f"Marginal Distribution node {2}", color="g", linewidth=2)
plt.plot(pts, A[3], label = f"Marginal Distribution node {3}", color="y", linewidth=2)
plt.xlabel("x (Scores)")
plt.ylabel("Marginal Probability")
plt.title(f"Marginal Distribution for Luck (alpha={alpha}) and Depth (beta={beta})")
plt.legend()
plt.grid(True)
plt.xlim([-1, 1])  # Ensure x-axis is limited to [-1, 1]
plt.show()
'''
