import ctypes
import numpy as np
import networkx as nx

# Added a funciton to create a DiGraph from the raw gml files
def read_gml_as_graph(filename):
    """Read a GML file and return the directed graph.

    :param filename: Input filename
    :type filename: str
    :return: Directed graph
    :rtype: nx.DiGraph
    """
    # Read the directed graph from the GML file using NetworkX
    G = nx.read_gml(filename)

    # Ensure the graph is directed
    if not G.is_directed():
        G = G.to_directed()

    return G

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/arbKernel.so")

# Update the argument types to reflect the inclusion of alpha
lib.full_algorithm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        array_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        array_double,
        array_double,
        ctypes.c_int
]
lib.full_algorithm.restype = ctypes.c_int 

lib.compute_Cheb_pts.argtypes = [ array_double, ctypes.c_int ]
lib.compute_Cheb_pts.restype = None

def ChebPts(L):
    ans = np.zeros(L)
    lib.compute_Cheb_pts(ans,L)
    return ans

def BP(G, L, alpha, beta, tol=10e-10, sm=0):
    M = np.array(list(G.edges())).flatten().astype(ctypes.c_int)
    output = np.zeros(L * (G.number_of_nodes() + 1) + 2)
    edgescores = np.zeros(G.number_of_edges())

    # Updated to pass both alpha and beta
    s = lib.full_algorithm(
        ctypes.c_int(G.number_of_nodes()),
        ctypes.c_int(G.number_of_edges()),
        M,
        ctypes.c_int(L),
        ctypes.c_double(alpha),  # Pass alpha
        ctypes.c_double(beta),   # Pass beta
        ctypes.c_double(tol),
        output, edgescores,
        ctypes.c_int(sm)
    )

    pts = output[:L].copy()
    A = output[L:-2].reshape(G.number_of_nodes(), L)
    S = output[-2]
    lnZ = output[-1]
    return pts, A, S, lnZ
