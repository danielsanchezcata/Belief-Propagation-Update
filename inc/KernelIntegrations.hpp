#include <iostream>
#include <vector>
#include "cheby2.hpp"
#include "ComparisonGraph.h"
#include <unordered_map>
#include <cmath>

extern std::vector<cheby::ArrayXd> K; // Declaration of global K

namespace KernelIntegration {

typedef cheby::ArrayXd ArrayXd;

ArrayXd coef_to_vals(ArrayXd const &c) {
    ArrayXd a = c;
    a[0] = 2.0*c[0];
    ArrayXd f = cheby::dct3(a);
    return f/2.0;
}

ArrayXd vals_to_coefs(ArrayXd const &f) {
    ArrayXd c = cheby::dct2(f) / f.size();
    c[0] /= 2.0;
    return c;
}

floatT integrate_kernel(ArrayXd const &f, ArrayXd const &K) {
    ArrayXd c = vals_to_coefs(f * K);
    ArrayXd c_new = cheby::Chebyshev_coef_integrate(c);
    return cheby::Chebyshev_value(1.0, c_new);
}

ArrayXd T(ArrayXd x, floatT n) {
    return cos(n * acos(x));
}

// Updated compute_integrals to account for B and N
std::vector<ArrayXd> compute_integrals(int L, int B, int N) {
    
    // Get Chebyshev points
    ArrayXd x = cheby::ChebPts(L);
    
    // Initialise the I and T arrays
    std::vector<ArrayXd> I_T(L);
    std::vector<ArrayXd> T_j(L);
    for (int i = 0; i < L; ++i) {
        I_T[i].resize(L, 0.0);
        T_j[i].resize(L, 0.0);
    }
    
    // Precompute Chebyshev polynomials
    for (int j = 0; j < L; ++j) {
        T_j[j] = T(x, j);
    }
    
    // Loop over Chebyshev points and compute integrals
    for (int i = 0; i < L; ++i) {
        
        // Use the global K matrix
        ArrayXd K_i(L);
        for (int j = 0; j < L; ++j) {
            K_i[j] = K[j][i];  // Access the precomputed K matrix
        }
        
        // Create arrays for K_i^B and (1 - K_i)^(N - B)
        ArrayXd K_i_pow(K_i.size());
        ArrayXd one_minus_K_i(K_i.size());
        ArrayXd one_minus_K_i_pow(K_i.size());

        // Compute K_i^B and (1 - K_i)^(N - B)
        for (size_t j = 0; j < K_i.size(); ++j) {
            K_i_pow[j] = std::pow(K_i[j], B);
            one_minus_K_i[j] = 1.0 - K_i[j];
            one_minus_K_i_pow[j] = std::pow(one_minus_K_i[j], N - B);
        }

        // Multiply the two results element-wise to get K_i_pow * (1 - K_i)^(N - B)
        for (size_t j = 0; j < K_i.size(); ++j) {
            K_i_pow[j] *= one_minus_K_i_pow[j];
        }

        // Parallelise the integration
        #pragma omp parallel shared(K_i_pow, I_T, T_j)
        {
            #pragma omp for
            for (int j = 0; j < L; ++j) {
                I_T[j][i] = integrate_kernel(T_j[j], K_i_pow);
            }
        }
    }

    return I_T;
}

}
