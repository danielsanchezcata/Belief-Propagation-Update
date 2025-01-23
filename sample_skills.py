import numpy as np
import networkx as nx
from arbKernel import read_adjacency_matrix, BP  # Importing from arbKernel.py
import time
import matplotlib.pyplot as plt

# Path to adjacency matrix file
adjacency_matrix_file = "/Users/danielsanchez/Desktop/competitions-data/dogs.txt"

# Start the timer
start_time = time.time()

# Read the adjacency matrix file
edges_array, B_array, N_array, num_nodes = read_adjacency_matrix(adjacency_matrix_file)

print(num_nodes)

# Set parameters for BP sampling
L = 64 # Number of Chebyshev points
alpha = 0.2  # Luck parameter
beta = 10 # Depth of competition parameter
tol = 1e-5  # Tolerance for BP
num_samples = 3 # Number of samples for every pair of nodes

# Run BP and sample skills
samples = BP(edges_array, B_array, N_array, num_nodes, L, alpha, beta, num_samples, tol)
 
# Output results
print(f"Value for luck: {alpha}")
print(f"Value for depth: {beta}")
# Sort the samples by participant indices
samples_sorted = sorted(samples, key=lambda x: (x["i"], x["j"]))
# Print the sampled skills
print("Sampled Skills: ")
for sample in samples_sorted:
    print(f"i: {sample['i']}, j: {sample['j']}, skill_i: {round(sample['skill_i'], 6)}, skill_j: {round(sample['skill_j'], 6)}")


# Calculate average skill for each pair (i, j)
pair_avg_skills = {}

for sample in samples_sorted:
    i, j = sample['i'], sample['j']
    skill_i, skill_j = sample['skill_i'], sample['skill_j']
    
    # Ensure the pair (i, j) is stored in a sorted way to avoid repetition
    pair = tuple(sorted([i, j]))
    
    if pair not in pair_avg_skills:
        pair_avg_skills[pair] = {"skill_i": [], "skill_j": []}
    
    pair_avg_skills[pair]["skill_i"].append(skill_i)
    pair_avg_skills[pair]["skill_j"].append(skill_j)

# Output the average skill for each player in each pair
for pair, skills in pair_avg_skills.items():
    avg_skill_i = sum(skills["skill_i"]) / len(skills["skill_i"])
    avg_skill_j = sum(skills["skill_j"]) / len(skills["skill_j"])
    print(f"Players {pair}:")
    print(f"  Average skill player {pair[0]}: {round(avg_skill_i, 4)}")
    print(f"  Average skill player {pair[1]}: {round(avg_skill_j, 4)}")

# End the timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
