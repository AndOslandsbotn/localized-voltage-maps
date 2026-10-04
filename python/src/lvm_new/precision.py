import numpy as np

DATA_DTYPE = np.float32    # data path: chunks, k-means, point-to-cell distances, kernels, placing points
SOLVE_DTYPE = np.float64   # accuracy-critical: graph, voltage solve, landmark distances, MDS (voltages down to 1e-10)
