#pragma once
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include "LEMP-fns.hpp"
#include <iostream>
#include <iomanip>
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


std::vector<cheby::ArrayXd> KlnK_eval(int L) {
    std::vector<cheby::ArrayXd> ans(L);
    for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
    // Iterate through the global kernel matrix K
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            ans[i][j] = xlogx(K[i][j]);
        }
    }
    return ans;
}

// Updated two_point_dist with normalisation
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
        }
    }
    // Compute the normalisation constant
    floatT norm = twoD_integral(ans);
    // Normalise the matrix Q
    if (norm > 0) {
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                ans[i][j] /= norm;
            }
        }
    } else {
        std::cerr << "Warning: normalisation constant is zero, normalisation skipped." << std::endl;
    }
    return ans;
}
       

floatT message_entropy_kernel(ComparisonGraph const &G, message_t &messages, int L) {
    floatT S = 0.0;
    for (int i : G.nodes()) {
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
            S -= log(Z) - (twoD_integral(QlnQ) / Z);
        }
    }

    return S;
}


floatT energy_kernel(ComparisonGraph const &G, message_t &messages, int L) {
    floatT U = 0.0;
    for (int i : G.nodes()) {
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
    
        std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), B, N);
        auto QlnK = Q;
        for (int ii=0; ii<L; ++ii) QlnK[ii] = Q[ii] * log0(computed_kernel[ii]);
            
        floatT Z = twoD_integral(Q);
        U += twoD_integral(QlnK)/Z;
        }
  }
  return U;
}


floatT iterate_messages_delta_parallel_kernel(ComparisonGraph const &G, elist_t const &m_order, message_t &messages, int L, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> const &I_cache) {
    floatT delta = 0.0;
    #pragma omp parallel shared(m_order, messages, G, L, I_cache) reduction(+: delta)
    for (int q = 0; q < m_order.size(); ++q) {
        int i = m_order[q].first;  // Current node
        int j = m_order[q].second; // Neighbour node
        cheby::ArrayXd new_values(1.0, L);
        
        for (const auto &[k, BN] : G.neighbors(j)) {
            if (k != i) {
                int B = BN.first;  // Number of wins for i over j
                int N = BN.second; // Total games between i and j

                // Retrieve the precomputed I matrix from the cache
                std::pair<int, int> B_N_pair = {B, N};
                
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
    return delta;
}


marginal_t compute_marginals_kernel(ComparisonGraph const &G, message_t &messages, int L,
        std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairHash> const &I_cache) {
    
    marginal_t marg;
    for (int i : G.nodes()) {
        marg[i] = cheby::Chebyshev(L);
        cheby::ArrayXd marginal_values(1.0, L); // Initialise marginal values

        for (const auto& [j, BN] : G.neighbors(i)) {
            int B = BN.first;  // Number of wins for i over j
            int N = BN.second; // Total games between i and j

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
    return marg;
}


