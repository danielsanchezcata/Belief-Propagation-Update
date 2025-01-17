#include <random>
#include "KernelIntegrations.hpp"
#include "arbitrary-kernel-fns.hpp"
#include <iostream>

std::random_device rd;
std::mt19937 rng(rd());

floatT ln_fact(floatT x) {
  if (x<=1) return 0;
  else return ln_fact(x-1) + log(x);
}

// Define global variables K and messages
message_t messages; // Global messages to initialise next message passing iteration from previous messages
std::vector<cheby::ArrayXd> K; // Global kernel matrix

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

// Main algorithm function updated
int full_algorithm(double *K_array, int num_nodes, int num_edges, int num_BN_pairs, int *full_B_array, int *full_N_array, int *edges, int L, double alpha_in, double beta_in, int *B_array, int *N_array, double tol, double *output, double *edgescores, int load_message) {
    
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
    
     // std::cout << "ComparisonGraph constructed with " << num_nodes << " nodes and " << num_edges << " edges.\n";
    
    /* Debugging: Print all edges and their BN weights
        std::cout << "Edges in the ComparisonGraph:\n";
        for (const auto& node : G.nodes()) {
            auto neighbors = G.neighbors(node);
            for (const auto& neighbor : neighbors) {
                int target = neighbor.first;
                auto [B, N] = neighbor.second; // Get B and N weights
                std::cout << "Edge (" << node << " -> " << target << ") with B = " << B << ", N = " << N << "\n";
            }
        } */
    
    // Define K from the python values
    set_kernel(K_array, L);
    
    /* Print the values of the kernel matrix K
    std::cout << "Kernel matrix K:" << std::endl;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            std::cout << K[i][j] << " ";
        }
        std::cout << std::endl; // Move to the next row
    } */

    // Precompute I matrices for each unique (B, N) pair
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> I_cache;
    Eigen::MatrixXd I(L, L);
        
    // Loop and compute integrals for unique (B, N) pairs
    for (size_t idx = 0; idx < num_BN_pairs; ++idx) {
        int B = full_B_array[idx];
        int N = full_N_array[idx];
        std::pair<int, int> B_N_pair = {B, N};

        // std::cout << "Processing pair " << idx << " with B = " << B << ", N = " << N << "\n";
            
        // Check if the (B, N) pair is already in the cache
        if (I_cache.find(B_N_pair) == I_cache.end()) {
            // std::cout << "Computing integrals for new (B, N) pair: (" << B << ", " << N << ")\n";
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
            // std::cout << "Stored I matrix for (B, N) pair: (" << B << ", " << N << ")\n";
        }
    }
    
    elist_t m_order = ComparisonGraph_to_list(G);  // Message update order
    
    /* Print the message update order
    std::cout << "Message update order (m_order):" << std::endl;
    for (const auto &[i, j] : m_order) {
        std::cout << "  Edge: (" << i << ", " << j << ")" << std::endl;
    }
    */
    
    // Initialise or reuse messages based on load_message flag
    if (load_message == 0) {
        messages.clear();  // Reset messages if not loading previous
        for (const auto &[i, j] : m_order) messages[i][j] = cheby::Chebyshev(L);
        std::cout << "Messages initialized.\n";
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
    
    /* Print the final messages matrix after message passing
    std::cout << "Final messages after message passing:\n";
    for (const auto &[i, j] : m_order) {
        std::cout << "Message[" << i << "][" << j << "]: ";
        
        // Assuming messages[i][j].vals() gives the values stored in the Chebyshev object
        for (const auto &val : messages[i][j].vals()) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } */
    
    // Compute one-node marginals
    marginal_t one_node_marginals = compute_marginals_kernel(G, messages, L, I_cache);
    
    // Return the skill samples
    std::vector<Sample> samples = skill_dist_samples(G, messages, L);
    // Print out the samples
    std::cout << "Sampled Skills: \n";
    for (const auto& sample : samples) {
        std::cout << "i: " << sample.i
                  << ", j: " << sample.j
                  << ", skill_i: " << sample.skill_i
                  << ", skill_j: " << sample.skill_j << std::endl;
    } //
    
    // Calculate energy and entropy values
    floatT U = energy_kernel(G, messages, L);    // Two-point energy
    // Print the final value of U
    std::cout << "Final value of U: " << U << std::endl;
    floatT S2 = message_entropy_kernel(G, messages, L);  // Two-point entropy
    // Print the final value of S2
    std::cout << "Final value of S2: " << S2 << std::endl;
    floatT S1 = marginal_entropy(G, one_node_marginals, L); // One-point entropy
    // Print the final value of S1
    std::cout << "Final value of S1: " << S1 << std::endl;
    floatT S = S2 - S1;
    std::cout << "Final value of S: " << S << std::endl;
    floatT lnZ = U - S;
    
    // Store results in the output array
    int t = 0;
    auto pts = cheby::ChebPts(L);
    for (t; t < pts.size(); ++t) output[t] = (double)pts[t];
    
    for (int i = 0; i < G.number_of_nodes(); ++i) {
        auto X = one_node_marginals[i].vals();
        for (int x = 0; x < X.size(); ++x) {
            output[t] = (double)X[x];
            t++;
        }
    }
    
    output[t] = (double)S; ++t;
    output[t] = (double)lnZ; ++t;
    
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
    
    /* std::cout << "Output array: ";
    for (int k = 0; k < t; ++k) {
        std::cout << output[k] << " ";
    }
    std::cout << std::endl; */

    return s+1;
}

}
