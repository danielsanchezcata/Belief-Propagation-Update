import ctypes
import numpy as np
import networkx as nx
import re

# Updated to return edges, B_array, and N_array from the adjacency matrix
def read_adjacency_matrix(filename):
    """Read an adjacency matrix from a text file and return distinct edges, B, and N arrays."""

    adjacency_matrix = np.loadtxt(filename)
    num_nodes = adjacency_matrix.shape[0]
    edges = []
    B_array = []
    N_array = []

    # Loop through the adjacency matrix to populate edges and B, N arrays
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            B_i = int(adjacency_matrix[i][j])  # Games i won against j
            B_j = int(adjacency_matrix[j][i])  # Games j won against i
            N = B_i + B_j  # Total games between i and j
            
            if N > 0:
                # Add the directed edge (i, j)
                edges.append((i, j))
                # Store the B and N pair for the directed edge (i, j)
                B_array.append(B_i)
                N_array.append(N)

    # Convert lists to numpy arrays
    B_array = np.array(B_array)
    N_array = np.array(N_array)
    
    return np.array(edges), B_array, N_array, num_nodes

array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/arbKernel.so")

# Updated to include alpha, B, and N
lib.full_algorithm.argtypes = [
    array_double,  # Matrix K
    ctypes.c_int,  # Number of nodes
    ctypes.c_int,  # Number of edges
    ctypes.c_int,  # Number of BN pairs
    array_int,     # full B array
    array_int,     # full N array
    array_int,     # Edge array
    ctypes.c_int,  # L (Chebyshev points)
    ctypes.c_double,  # alpha
    ctypes.c_double,  # beta
    array_int,     # B array
    array_int,     # N array
    ctypes.c_double,  # tolerance
    array_double,  # Output
    ctypes.c_int,  # Number of samples for every pair of nodes
    ctypes.c_int   # Load message (0 for reinitialisation and 1 for reuse)
]
lib.full_algorithm.restype = ctypes.c_int 

lib.compute_Cheb_pts.argtypes = [ array_double, ctypes.c_int ]
lib.compute_Cheb_pts.restype = None

def ChebPts(L):
    ans = np.zeros(L)
    lib.compute_Cheb_pts(ans,L)
    return ans

def compute_kernel_matrix(L, alpha, beta):
    """
    Computes the kernel matrix K based on Chebyshev points (L), alpha and beta.
    The exact form of the kernel can be updated as necessary.
    """
    # Generate the Chebyshev points
    x = ChebPts(L)
    # Initialise the kernel matrix
    K = np.zeros((L, L))

    # Compute the kernel values between pairs of Chebyshev points
    for i in range(L):
        for j in range(L):
            # Modify this to use different kernel functions
            K[i, j] = 0.5 * alpha + (1 - alpha) / (1 + np.exp(-beta * (x[i] - x[j])))
            
    return K


def BP(edges, B_array, N_array, num_nodes, L, alpha, beta, num_samples, tol=10e-10, load_messages=0):
    num_edges = len(edges)
    edges_flat = edges.flatten().astype(ctypes.c_int)
    B_array = B_array.astype(ctypes.c_int) # These arrays are used for
    N_array = N_array.astype(ctypes.c_int) # building the comparison graph
    
    # Compute the kernel matrix K
    K = compute_kernel_matrix(L, alpha, beta)
    K_flat = np.ascontiguousarray(K.flatten(order='C').astype(np.float64))  # Use C-style flattening

    # Create two arrays for all the possible B-N pairs
    full_BN_pairs = set()
    for idx in range(len(edges)):
        B_i = B_array[idx]
        N_i = N_array[idx]
        full_BN_pairs.add((B_i, N_i))  # Add the original pair
        full_BN_pairs.add((N_i - B_i, N_i))  # Add the reverse pair (N - B, N)

    # Convert the set back to two separate NumPy arrays
    full_B_array, full_N_array = np.array(list(zip(*full_BN_pairs)))

    num_BN_pairs = len(full_B_array)
    full_B_array = full_B_array.astype(ctypes.c_int)
    full_N_array = full_N_array.astype(ctypes.c_int)
    
    # Each sample consists of 4 values (i, j, skill_i, skill_j)
    sample_size = 4
    output_size = num_edges * num_samples * 4
    output = np.zeros(output_size)  # Shared memory buffer

    # Call full_algorithm
    s = lib.full_algorithm(
        K_flat,  # Pass the kernel matrix K
        ctypes.c_int(num_nodes),  # Number of nodes
        ctypes.c_int(num_edges),  # Number of edges
        ctypes.c_int(num_BN_pairs),
        full_B_array,
        full_N_array,
        edges_flat,
        ctypes.c_int(L),
        ctypes.c_double(alpha),  # Pass alpha
        ctypes.c_double(beta),   # Pass beta
        B_array,  # Pass B
        N_array,  # Pass N
        ctypes.c_double(tol),
        output,
        ctypes.c_int(num_samples),
        ctypes.c_int(load_messages)
    )
    
    # Convert the output array back into a Python-friendly format
    samples = []
    for i in range(num_edges * num_samples):
        idx = i * sample_size
        sample = {
            "i": int(output[idx]),
            "j": int(output[idx + 1]),
            "skill_i": output[idx + 2].item(),
            "skill_j": output[idx + 3].item()
        }
        samples.append(sample)

    return samples
