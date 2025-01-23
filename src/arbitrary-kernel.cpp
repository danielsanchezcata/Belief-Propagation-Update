#include <random>
#include "KernelIntegrations.hpp"
#include "arbitrary-kernel-fns.hpp"
#include "sample.hpp"
#include <iostream>

std::random_device rd;
std::mt19937 rng(rd());

floatT ln_fact(floatT x) {
  if (x<=1) return 0;
  else return ln_fact(x-1) + log(x);
}

// Define global variables K and messages
message_t messages;
std::vector<cheby::ArrayXd> K;

// Defines the K matrix from precomputed python values
void set_kernel(double* y, int L) {
    // Resize the global kernel matrix to L x L
    K.resize(L);

    // Populate K with ArrayXd objects
    for (int i = 0; i < L; ++i) {
        K[i] = cheby::ArrayXd(L); 
        for (int j = 0; j < L; ++j) {
            K[i][j] = y[i * L + j]; // Fill in values from the flattened array
        }
    }
}

extern "C" {

void compute_Cheb_pts(double *out, int L) {
    auto pts = cheby::ChebPts(L);
    for (int i = 0; i < L; ++i) out[i] = (double) pts[i];
}

// Main algorithm function updated for sampling
int full_algorithm(double *K_array, int num_nodes, int num_edges, int num_BN_pairs, int *full_B_array, int *full_N_array, int *edges, int L, double alpha_in, double beta_in, int *B_array, int *N_array, double tol, double *output, int num_samples, int load_message) {
    
    // Construct the ComparisonGraph with B and N values
    ComparisonGraph G;
    for (int i = 0; i < num_nodes; ++i) G.add_node(i);
    for (int i = 0; i < num_edges; ++i) {
        int source = edges[2 * i];
        int target = edges[2 * i + 1];
        int B = B_array[i];  // Get B from array
        int N = N_array[i];  // Get N from array
        G.add_edge(source, target, B, N);  // Add edge with B and N
    }
    
    // Define K from the python values
    set_kernel(K_array, L);
 
    // Precompute I matrices for each unique (B, N) pair
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> I_cache;
    Eigen::MatrixXd I(L, L);
        
    // Loop and compute integrals for unique (B, N) pairs
    for (size_t idx = 0; idx < num_BN_pairs; ++idx) {
        int B = full_B_array[idx];
        int N = full_N_array[idx];
        std::pair<int, int> B_N_pair = {B, N};
            
        // Check if the (B, N) pair is already in the cache
        if (I_cache.find(B_N_pair) == I_cache.end()) {
            // Compute the integrals for this B, N combination
            auto I_T = KernelIntegration::compute_integrals(L, B, N);
            // Store integrals in the matrix I
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    I(i, j) = I_T[j][i];
                }
            }
            // Add the matrix I to the cache
            I_cache[B_N_pair] = I;
        }
    }
    
    elist_t m_order = ComparisonGraph_to_list(G);  // Message update order
    
    // Initialise or reuse messages based on load_message flag
    if (load_message == 0) {
        messages.clear();  // Reset messages if not loading previous
        for (const auto &[i, j] : m_order) messages[i][j] = cheby::Chebyshev(L);
        std::cout << "Messages initialised.\n";
    }else {
        std::cout << "Reusing messages from the last iteration.\n";
    }
    
    // Perform message passing until convergence or max iterations
    int s = 0;
    for (s; s < 100; ++s) {
        std::shuffle(m_order.begin(), m_order.end(), rng);  // Randomise update order
        // Update messages and check for convergence
        if (iterate_messages_delta_parallel_kernel(G, m_order, messages, L, I_cache) < tol) {
            break;
        }
    }
   
    // Compute one-node marginals
    marginal_t one_node_marginals = compute_marginals_kernel(G, messages, L, I_cache);
    
    // Sample skills from the two point marginal distributions
    std::vector<Sample> samples = two_point_samples(G, messages, L, num_samples);
    
    // Calculate energy and entropy values
    floatT U = energy_kernel(G, messages, L);    // Two-point energy
    std::cout << "Final value of U: " << U << std::endl;
    floatT S2 = message_entropy_kernel(G, messages, L);  // Two-point entropy
    std::cout << "Final value of S2: " << S2 << std::endl;
    floatT S1 = marginal_entropy(G, one_node_marginals, L); // One-point entropy
    std::cout << "Final value of S1: " << S1 << std::endl;
    floatT S = S2 - S1;                                     // Bethe entropy
    std::cout << "Final value of S: " << S << std::endl;
    floatT lnZ = U - S;                                    // log Partition function
    std::cout << "Final value of lnZ: " << lnZ << std::endl;
    
    int t = 0;
    // Store skill samples in the output array
    for (int i = 0; i < samples.size(); ++i) {
        output[t] = (double)samples[i].i;   // Participant i
        ++t;
        output[t] = (double)samples[i].j;   // Participant j
        ++t;
        output[t] = (double)samples[i].skill_i; // Sampled skill for participant i
        ++t;
        output[t] = (double)samples[i].skill_j; // Sampled skill for participant j
        ++t;
    }
    
    return s+1;
}

}
