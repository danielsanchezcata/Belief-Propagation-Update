#pragma once
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include "LEMP-fns.hpp"
#include <iostream>
#include <iomanip> // for formatting output
#include <random>
#include <numeric>
#include <algorithm>

extern std::vector<cheby::ArrayXd> K; // Declaration of global K

// Key to store unique (B, N) pairs
struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

Eigen::MatrixXd load_matrix(std::string f_name, int L) {
  Eigen::MatrixXd X(L,L);
  std::ifstream in_file(f_name);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      in_file >> X(i,j);
    }
  }
  return X;
}

cheby::ArrayXd dot(Eigen::MatrixXd const &X, cheby::ArrayXd const &a) {
  int L = a.size();
  cheby::ArrayXd ans(0.0,L);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      ans[i] += X(i,j)*a[j];
    }
  }
  return ans;
}

floatT oneD_integral( cheby::ArrayXd const &f ) {
	cheby::Chebyshev f_cheb(f, INIT_BY_VALUES);
	return cheby::Chebyshev_value(1.0, cheby::Chebyshev_coef_integrate(f_cheb.coefs()));
}


floatT twoD_integral( std::vector<cheby::ArrayXd> const &f ) {
	int L = f.size();
	cheby::ArrayXd g(0.0,L);
	for (int k=0; k<L; ++k) {
		g[k] = oneD_integral(f[k]);
	}
	return oneD_integral(g);
}


floatT xlogx(floatT x){
	if (x>0) return x * log(x);
	else return 0.0;
}

// Updated to take alpha as an additional parameter
std::vector<cheby::ArrayXd> KlnK_eval(int L) {
    std::vector<cheby::ArrayXd> ans(L);
    for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
    // Iterate through the global kernel matrix K
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            ans[i][j] = xlogx(K[i][j]);  // Use the precomputed kernel values from K
        }
    }
    return ans;
}

// Updated two_point_dist
std::vector<cheby::ArrayXd> two_point_dist( cheby::ArrayXd const &f1,
        cheby::ArrayXd const &f2, int B, int N) {
    
        int L = f1.size();
        std::vector<cheby::ArrayXd> ans(L);
        for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
        for (int i=0; i<L; ++i) {
            for (int j=0; j<L; ++j) {
                // Calculate each component for ans[i][j]
                double K_pow_B = std::pow(K[i][j], B);
                double K_pow_N_minus_B = std::pow(1 - K[i][j], N - B);
                double product_f1_f2 = f1[i] * f2[j]; 

                // Assign the result to ans[i][j]
                ans[i][j] = K_pow_B * K_pow_N_minus_B * product_f1_f2;

                /* Print the intermediate values and final result for debugging
                std::cout << "K[" << i << "][" << j << "]^B: " << K_pow_B
                          << ", (1 - K[" << i << "][" << j << "])^(N - B): " << K_pow_N_minus_B
                          << ", f1[" << i << "] * f2[" << j << "]: " << product_f1_f2
                          << ", ans[" << i << "][" << j << "]: " << ans[i][j] << std::endl; */
            }
        }
        /* Print the final ans matrix
        std::cout << "Final ans matrix:" << std::endl;
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                std::cout << ans[i][j] << " ";
            }
            std::cout << std::endl;
        } */
    
        return ans;
    }

// Original
std::vector<cheby::ArrayXd> two_point_dist( cheby::ArrayXd const &f1,
		cheby::ArrayXd const &f2, std::vector<cheby::ArrayXd> const &K ) {

	int L = f1.size();
	std::vector<cheby::ArrayXd> ans(L);
	for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
	for (int i=0; i<L; ++i) {
		for (int j=0; j<L; ++j) {
			ans[i][j] = (K[i][j])*f1[i]*f2[j];
		}
	}
	return ans;

}

// Structure to store the skill samples
struct Sample {
    int i, j; // Indices of participants
    floatT skill_i, skill_j; // Sampled skill values for participants
};

// Function to return skill samples from two_point_dist
std::vector<Sample> skill_dist_samples(ComparisonGraph const &G, message_t &messages, int L) {
    std::vector<Sample> samples;
    auto pts = cheby::ChebPts(L); // Chebyshev points for the grid
    // Convert valarray to vector and reverse the order of Chebyshev points
    std::vector<floatT> pts_vec(std::begin(pts), std::end(pts));
    std::reverse(pts_vec.begin(), pts_vec.end());
    std::vector<floatT> eval_pts; // Points to evaluate the CDF
    eval_pts = pts_vec; // Use reversed points for evaluation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i : G.nodes()) {
        for (const auto &[j, BN] : G.neighbors(i)) {
            if (i >= j) continue; // Avoid double-counting for undirected graph

            int B = BN.first;  // Number of wins for i over j
            int N = BN.second;  // Total games between i and j
            
            // Print out the node pair and their B, N values for debugging
            std::cout << "Processing nodes: " << i << " and " << j << std::endl;

            auto mu1 = messages.at(i).at(j);
            auto mu2 = messages.at(j).at(i);
            mu1.normalize();
            mu2.normalize();

            // Compute Q(x, y)
            std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), B, N);

            // Marginalise to get q(x) using 1D integral
            cheby::ArrayXd q_x(L);
            for (int m = 0; m < L; ++m) {
                q_x[m] = oneD_integral(Q[m]);
            }
            
            // Normalise q(x)
            cheby::Chebyshev qx_cheb(q_x, INIT_BY_VALUES); // Initialise Chebyshev polynomial
            qx_cheb.normalize();                           // Normalise so that integral(q(x)) = 1
            q_x = qx_cheb.vals();
            
            // Convert q(x) values to Chebyshev coefficients and compute CDF coefficients
            cheby::ArrayXd qx_coefs = cheby::Chebyshev_coefs_from_values(q_x);
            cheby::ArrayXd cdf_coefs = cheby::Chebyshev_coef_integrate(qx_coefs);

            // Evaluate the CDF at desired points
            std::vector<floatT> cdf_values(L);
            for (int idx = 0; idx < L; ++idx) {
                cdf_values[idx] = cheby::Chebyshev_value(eval_pts[idx], cdf_coefs);
            }

            // Print the CDF values
            std::cout << "x_points and corresponding CDF values:" << std::endl;
            for (int i = 0; i < eval_pts.size(); ++i) {
                std::cout << "x = " << eval_pts[i] << ", CDF(x) = " << cdf_values[i] << std::endl;
            }
            
            // Compute the inverse CDF values z_k
            cheby::ArrayXd z_k(L);
            for (int k = 0; k < L; ++k) {
                // Compute the target value for CDF
                floatT target = (eval_pts[k] + 1.0) / 2.0;  // F(z_k) = (x_k + 1) / 2
                // Initialise bisection bounds
                floatT low = -1.0;
                floatT high = 1.0;
                floatT mid = 0.0;
                
                int max_iter = 128; // Bisection iteration limit
                while (max_iter-- > 0 && (high - low) > std::numeric_limits<floatT>::epsilon()) {
                    mid = (low + high) / 2.0;
                    floatT cdf_mid = cheby::Chebyshev_value(mid, cdf_coefs);
                    if (cdf_mid < target) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                z_k[k] = mid; // Final root
            }

            // Print the z_k values
            std::cout << "z_k values (inverse CDF):" << std::endl;
            for (int k = 0; k < L; ++k) {
                std::cout << "z_k[" << k << "] = " << z_k[k] << std::endl;
            }
            
            // Represent inverse CDF as a Chebyshev polynomial
            cheby::Chebyshev inverse_cdf_cheb(z_k, INIT_BY_VALUES);
            
            // Check the inverse CDF Chebyshev polynomial coefficients
            std::cout << "Initialised inverse CDF Chebyshev polynomial with coefficients: ";
            for (int i = 0; i < z_k.size(); ++i) {
                std::cout << inverse_cdf_cheb.coefs()[i] << " ";
            }
            std::cout << std::endl;
            
            // Loop to get multiple samples for testing (I think we only need one sample in practice)
            for (int sample_idx = 0; sample_idx < 1; ++sample_idx) {
                
                // Sample x' from q(x)
                floatT u = dis(gen); // Uniform sample in [0, 1]
                std::cout << "Generated uniform random sample u = " << u << std::endl;
                floatT skill_j = inverse_cdf_cheb.value(2.0 * u - 1.0);
                std::cout << "Sampled skill_j using inverse CDF: " << skill_j << std::endl;
                
                // Update this part computation of Q'(x', y)
                // Find the closest value in eval_pts using lower_bound
                auto it = std::lower_bound(eval_pts.begin(), eval_pts.end(), skill_j);
                // Get the index of the nearest value
                int x_idx = std::distance(eval_pts.begin(), it);
                // Access the corresponding Q[x_idx]
                cheby::ArrayXd Q_x_prime = Q[x_idx];
               
                // Normalise using Chebyshev::normalize
                cheby::Chebyshev Qx_prime_cheb(Q_x_prime, INIT_BY_VALUES); // Initialise Chebyshev polynomial
                Qx_prime_cheb.normalize();                                 // Normalise so that integral(Q(x', y)) = 1
                Q_x_prime = Qx_prime_cheb.vals();
                
                // Compute the CDF for Q(x', y)
                cheby::ArrayXd Qx_prime_coefs = cheby::Chebyshev_coefs_from_values(Q_x_prime);
                cheby::ArrayXd CDF_Q_x_prime_coefs = cheby::Chebyshev_coef_integrate(Qx_prime_coefs);

                std::vector<floatT> CDF_Q_x_prime(L);
                for (int idx = 0; idx < L; ++idx) {
                    CDF_Q_x_prime[idx] = cheby::Chebyshev_value(eval_pts[idx], CDF_Q_x_prime_coefs);
                }
                
                // Print the CDF values
                std::cout << "y_points and corresponding CDF values:" << std::endl;
                for (int i = 0; i < eval_pts.size(); ++i) {
                    std::cout << "y = " << eval_pts[i] << ", CDF(y) = " << CDF_Q_x_prime[i] << std::endl;
                }
                
                // Compute the inverse CDF values z_k_y
                cheby::ArrayXd z_k_y(L);
                for (int k = 0; k < L; ++k) {
                    // Compute the target value for CDF
                    floatT target = (eval_pts[k] + 1.0) / 2.0;  // F(z_k) = (x_k + 1) / 2
                    // Initialise bisection bounds
                    floatT low = -1.0;
                    floatT high = 1.0;
                    floatT mid = 0.0;
                    
                    int max_iter = 128; // Bisection iteration limit
                    while (max_iter-- > 0 && (high - low) > std::numeric_limits<floatT>::epsilon()) {
                        mid = (low + high) / 2.0;
                        floatT cdf_mid = cheby::Chebyshev_value(mid, CDF_Q_x_prime_coefs);
                        if (cdf_mid < target) {
                            low = mid;
                        } else {
                            high = mid;
                        }
                    }
                    z_k_y[k] = mid; // Final root
                }
                
                // Print the z_k values
                std::cout << "z_k_y values (inverse CDF):" << std::endl;
                for (int k = 0; k < L; ++k) {
                    std::cout << "z_k_y[" << k << "] = " << z_k_y[k] << std::endl;
                }
                
                // Represent inverse CDF as a Chebyshev polynomial
                cheby::Chebyshev inverse_CDF_Q_x_prime_cheb(z_k_y, INIT_BY_VALUES);

                
                // Sample y' from Q(x', y)
                floatT v = dis(gen); // Uniform sample in [0, 1]
                std::cout << "Generated uniform random sample v = " << v << std::endl;
                floatT skill_i = inverse_CDF_Q_x_prime_cheb.value(2.0 * v - 1.0);
                std::cout << "Sampled skill_i using inverse CDF: " << skill_i << std::endl;

                // Store the sample
                samples.push_back(Sample{i, j, skill_i, skill_j});
                
            }
        }
    }

    return samples;
}


floatT message_entropy_kernel(ComparisonGraph const &G, message_t &messages, int L) {
    floatT S = 0.0;
    // Loop over nodes in the ComparisonGraph
    for (int i : G.nodes()) {
        // Loop over neighbours (instead of successors since ComparisonGraph is undirected)
        for (const auto &[j, BN] : G.neighbors(i)) {
            
            if (i >= j) continue;  // Skip to prevent double-counting for undirected graph
            int B = BN.first;  // Number of wins for i over j
            int N = BN.second;  // Total games between i and j

            auto mu1 = messages.at(i).at(j);
            auto mu2 = messages.at(j).at(i);
            mu1.normalize();
            mu2.normalize();

            std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), B, N);
            auto QlnQ = Q;
            for (int ii = 0; ii < L; ++ii) {
                QlnQ[ii] = Q[ii] * log0(Q[ii]);
            }
            
            floatT Z = twoD_integral(Q);
            // std::cout << "Two D Integral of Q:" << Z << std::endl;
            S -= log(Z) - (twoD_integral(QlnQ) / Z);
        }
    }

    return S;
}

// Updated
floatT energy_kernel(ComparisonGraph const &G, message_t &messages, int L) {
    floatT U = 0.0;
    
    // Loop over nodes in the ComparisonGraph
    for (int i : G.nodes()) {
        // Loop over neighbors (since ComparisonGraph is undirected)
        for (const auto &[j, BN] : G.neighbors(i)) {
            
            if (i >= j) continue;  // Skip to prevent double-counting for undirected graph
            int B = BN.first;  // Number of wins for i over j
            int N = BN.second;  // Total games between i and j
      
            // Initialise and compute computed_kernel specific to the (B, N) pair for (i, j)
            std::vector<cheby::ArrayXd> computed_kernel(L);
            for (int ii = 0; ii < L; ++ii) {
                computed_kernel[ii].resize(L);
                for (int jj = 0; jj < L; ++jj) {
                    computed_kernel[ii][jj] = std::pow(K[ii][jj], B) * std::pow(1 - K[ii][jj], N - B);
                }
            }
            
        auto mu1 = messages.at(i).at(j);
        auto mu2 = messages.at(j).at(i);
        mu1.normalize();
        mu2.normalize();
        
        /* Debug the values of m1 and m2
        std::cout << "mu2.vals(): ";
        for (const auto& val : mu2.vals()) {
            std::cout << val << " ";  // Print each element of mu2.vals()
        }
        std::cout << std::endl;

        std::cout << "mu1.vals(): ";
        for (const auto& val : mu1.vals()) {
            std::cout << val << " ";  // Print each element of mu1.vals()
        }
        std::cout << std::endl; */


        std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), B, N);
        auto QlnK = Q;
        for (int ii=0; ii<L; ++ii) QlnK[ii] = Q[ii] * log0(computed_kernel[ii]);
            
        floatT Z = twoD_integral(Q);
        U += twoD_integral(QlnK)/Z;
        }
  }
  return U;
}

// Updated
floatT iterate_messages_delta_parallel_kernel(ComparisonGraph const &G, elist_t const &m_order, message_t &messages, int L, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> const &I_cache) {
    
    // Print statement indicating the function has been called
    // std::cout << "Function iterate_messages_delta_parallel_kernel called." << std::endl;
    
    // Print the message update order
    // std::cout << "Message update order (m_order):" << std::endl;
    // for (const auto &[i, j] : m_order) {
    //     std::cout << "  Edge: (" << i << ", " << j << ")" << std::endl;
    // }

    floatT delta = 0.0;
    #pragma omp parallel shared(m_order, messages, G, L, I_cache) reduction(+: delta)
    for (int q = 0; q < m_order.size(); ++q) {
        int i = m_order[q].first;  // Current node
        int j = m_order[q].second; // Neighbour node
        cheby::ArrayXd new_values(1.0, L);
        
        // std::cout << "Processing edge (" << i << ", " << j << ")" << std::endl;

        // Loop over neighbours of j (since it's now undirected)
        for (const auto &[k, BN] : G.neighbors(j)) {
            if (k != i) {
                int B = BN.first;  // Get B from the weights
                int N = BN.second; // Get N from the weights

                // Retrieve the precomputed I matrix from the cache
                std::pair<int, int> B_N_pair = {B, N};
                
                // Debug statement for B, N values
                // std::cout << "  Neighbor (" << j << ", " << k << ") with (B, N) = (" << B << ", " << N << ")" << std::endl;
                
                // Check if (B, N) pair exists in I_cache
                if (I_cache.find(B_N_pair) == I_cache.end()) {
                    std::cerr << "Error: I matrix for (B, N) = (" << B << ", " << N << ") not found in I_cache." << std::endl;
                    continue; // Skip this neighbour if the I matrix is missing
                }
                
                Eigen::MatrixXd I = I_cache.at(B_N_pair);

                // Update message using the dot product with the precomputed I matrix
                new_values *= dot(I, messages.at(k).at(j).coefs());
            }
        }

        cheby::Chebyshev new_message(new_values, INIT_BY_VALUES);
        if (ENFORCE_POSITIVE) new_message.positives(0.0);
        new_message.normalize();
        
        // Update the delta value
        delta += abs(messages[j][i].vals() - new_message.vals()).sum();
        messages[j][i] = new_message; // Update the message
    }
    // std::cout << "Total delta after processing: " << delta << std::endl;
    return delta; // Return the delta value
}

// Updated
marginal_t compute_marginals_kernel(ComparisonGraph const &G, message_t &messages, int L,
        std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> const &I_cache) {
    
    marginal_t marg;
    for (int i : G.nodes()) {
        marg[i] = cheby::Chebyshev(L);
        cheby::ArrayXd marginal_values(1.0, L); // Initialise marginal values

        // Loop over neighbors in the undirected graph
        for (const auto& [j, BN] : G.neighbors(i)) {
            int B = BN.first;  // Get B from the weights
            int N = BN.second; // Get N from the weights

            // Access the precomputed I matrix for the (B, N) pair
            std::pair<int, int> B_N_pair = {B, N};
            Eigen::MatrixXd I = I_cache.at(B_N_pair);

            // Update marginal values using the dot product with I matrix
            marginal_values *= dot(I, messages.at(j).at(i).coefs());
        }

        // Normalise and store the marginals
        marg[i].set_coefs(cheby::Chebyshev_coefs_from_values(marginal_values)[std::slice(0, L, 1)]);
        if (ENFORCE_POSITIVE) marg[i].positives(0.0);
        marg[i].normalize();
    }
    return marg; // Return the computed marginals
}





