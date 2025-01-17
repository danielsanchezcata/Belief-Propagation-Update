import numpy as np
import networkx as nx
from arbKernel import read_adjacency_matrix, BP  # Importing from arbKernel.py
import time
import matplotlib.pyplot as plt

# Path to adjacency matrix file
adjacency_matrix_file = "/Users/danielsanchez/Desktop/competitions-data/dogs.txt"

# Start the timer
overall_start_time = time.time()

# Read the adjacency matrix file
edges_array, B_array, N_array, num_nodes = read_adjacency_matrix(adjacency_matrix_file)

# Set parameters for BP
L = 100  # Number of Chebyshev points
tol = 1e-5  # Tolerance for BP
w = 4  # Scale parameter for Cauchy distribution


# Define the ranges for alpha and beta
alpha_range = np.linspace(0.0, 0.2, 11)
beta_range = np.linspace(6, 16, 11)

# Create an array to store the log-likelihood values
log_posteriors = np.zeros((len(alpha_range), len(beta_range)))

# Flag to control message loading
load_messages = 0  # Start with initialising messages

# Two for loops to perform grid search
for i, alpha in enumerate(alpha_range):
    alpha_iteration_start_time = time.time()
    for j, beta in enumerate(beta_range):
    
        # Start the timer for each individual computation
        iteration_start_time = time.time()
    
        # Run BP
        pts, A, S, lnZ, samples = BP(edges_array, B_array, N_array, num_nodes, L, alpha, beta, tol, load_messages)
    
        # Calculate the log-likelihood
        log_likelihood = lnZ - num_nodes * np.log(2)
        
        # Calculate the prior term for beta (Cauchy distribution)
        log_prior_beta = np.log((2 * w / np.pi) / (beta**2 + w**2))

        # Calculate the log posterior
        log_posterior = log_likelihood + log_prior_beta
    
        # Store the log posterior value
        log_posteriors[i, j] = log_posterior
        
        
        # Update load_message to reuse messages for next alpha, beta
        load_messages = 1
        
        # End the timer for the current iteration
        iteration_end_time = time.time()
        iteration_execution_time = iteration_end_time - iteration_start_time
    
        # Output results for this alpha-beta pair
        print(f"Value for luck (alpha): {alpha}")
        print(f"Value for depth (beta): {beta}")
        samples_sorted = sorted(samples, key=lambda x: (x[0], x[1]))
        print(f"Sampled Skills : {[(i, j, round(float(skill_i), 4), round(float(skill_j), 4)) for i, j, skill_i, skill_j in samples_sorted]}")
        print(f"Log Posterior : {log_posterior}")
        print(f"Time for this iteration: {iteration_execution_time:.4f} seconds")

    # End the timer for the current iteration
    alpha_iteration_end_time = time.time()
    alpha_iteration_execution_time = alpha_iteration_end_time - alpha_iteration_start_time
    print(f"Time for this alpha iteration: {alpha_iteration_execution_time:.4f} seconds")
    
 # End the overall timer and calculate total execution time
overall_end_time = time.time()
overall_execution_time = overall_end_time - overall_start_time
print(f"Total Execution Time: {overall_execution_time:.2f} seconds")

# Replace NaN values with the minimum log-likelihood
min_log_posterior = np.nanmin(log_posteriors)
log_posteriors = np.nan_to_num(log_posteriors, nan=min_log_posterior)

# Normalise the log-likelihoods
max_log_posterior = np.max(log_posteriors)
normalised_posteriors = np.exp(log_posteriors - max_log_posterior)

# Calculate expected values for alpha and beta
expected_alpha = np.sum(alpha_range[:, np.newaxis] * normalised_posteriors) / np.sum(normalised_posteriors)
expected_beta = np.sum(beta_range[np.newaxis, :] * normalised_posteriors) / np.sum(normalised_posteriors)

# Print expected alpha and beta values
print("Expected alpha:", expected_alpha)
print("Expected beta:", expected_beta)

# Find the maximum likelihood point
max_posterior_idx = np.unravel_index(np.argmax(log_posteriors, axis=None), log_posteriors.shape)
max_alpha = alpha_range[max_posterior_idx[0]]
max_beta = beta_range[max_posterior_idx[1]]

# Print Mode alpha and beta values
print("Mode alpha:", max_alpha)
print("Mode beta:", max_beta)

# Plot the heatmap of log-likelihoods
plt.figure(figsize=(12, 6))
plt.imshow(normalised_posteriors, extent=[beta_range.min(), beta_range.max(), alpha_range.min(), alpha_range.max()],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Normalised Posterior Distribution')
# Set the x-axis to logarithmic scale
plt.xscale('log')
integer_ticks = np.arange(1, 17)
plt.xticks(integer_ticks, labels=integer_ticks)

# Plot the Red Cross at the expected alpha and beta
plt.plot(expected_beta, expected_alpha, 'rx', markersize=12, markeredgewidth=2, label="Expected (alpha, beta)")

# Plot the Blue Cross at the maximum log-likelihood alpha and beta
plt.plot(max_beta, max_alpha, 'bx', markersize=12, markeredgewidth=2, label="Max Likelihood (alpha, beta)")

plt.xlabel('Beta (Depth)')
plt.ylabel('Alpha (Luck)')
plt.title('Normalised Posterior Distribution New Model')

plt.show()
