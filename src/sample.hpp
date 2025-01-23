#include "cheby2.hpp"
#include "arbitrary-kernel-fns.hpp"
#include <iostream>
#include <vector>
#include <random>

std::random_device rd_sample;
std::mt19937 gen(rd_sample());
std::uniform_real_distribution<floatT> unif(0.0,1.0);

std::vector<cheby::ArrayXd> transpose(std::vector<cheby::ArrayXd> X, int L) {
    std::vector<cheby::ArrayXd> ans(L);
    for (int i=0; i<L; ++i) {
      ans[i].resize(L,0.0); 
      for (int j=0; j<L; ++j) ans[i][j] = X[j][i];
    }
    return ans;
}

floatT sample1d(cheby::Chebyshev pdf, floatT u) {
    auto cdf = pdf.integrate();
    return (cdf - u).root();
}

std::pair<floatT, floatT> sample2d(std::vector<cheby::ArrayXd> Q, floatT u1, floatT u2, int L) {

    cheby::ArrayXd qx(L);
    for (int i=0; i<L; ++i) qx[i] = cheby::Chebyshev(Q[i], INIT_BY_VALUES).integrate()(1.0);
    floatT x = sample1d(cheby::Chebyshev(qx, INIT_BY_VALUES), u1);

    auto QT = transpose(Q, L);
    cheby::ArrayXd qy(L);
    for (int i=0; i<L; ++i) qy[i] = cheby::Chebyshev(QT[i], INIT_BY_VALUES)(x);
    auto pdf_y = cheby::Chebyshev(qy, INIT_BY_VALUES);
    pdf_y.normalize();
    floatT y = sample1d(pdf_y, u2);

    return {x, y};
}
    
// Structure to store the skill samples
struct Sample {
    int i, j; // Indices of participants
    floatT skill_i, skill_j; // Sampled skill values for participants
};

// Function to return skill samples from two_point_dist
std::vector<Sample> two_point_samples(ComparisonGraph const &G, message_t &messages, int L, int num_samples) {
    std::vector<Sample> samples;

    for (int i : G.nodes()) {
        for (const auto &[j, BN] : G.neighbors(i)) {
            if (i >= j) continue; // Avoid double-counting for undirected graph

            int B = BN.first;  // Number of wins for i over j
            int N = BN.second;  // Total games between i and j
            
            std::cout << "Processing nodes: " << i << " and " << j << std::endl;

            auto mu1 = messages.at(i).at(j);
            auto mu2 = messages.at(j).at(i);
            mu1.normalize();
            mu2.normalize();

            // Compute Q(x, y)  // Note Q has to be normalised
            std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), B, N);
            // Generate multiple samples for each node pair
            for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
                // Generate random uniform samples u1 and u2 in [0, 1]
                floatT u1 = unif(gen);
                floatT u2 = unif(gen);

                auto [skill_j, skill_i] = sample2d(Q, u1, u2, L);
                // Store the sample
                samples.push_back(Sample{i, j, skill_i, skill_j});
            }
        }
    }
    return samples;
}
