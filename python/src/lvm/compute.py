import numpy as np
import torch

DATA_DTYPE = np.float32    # data path: chunks, k-means, point-to-cell distances, kernels, placing points
SOLVE_DTYPE = np.float64   # accuracy-critical: graph, voltage solve, landmark distances, MDS (voltages down to 1e-10)

TORCH_DATA_DTYPE = getattr(torch, np.dtype(DATA_DTYPE).name)
TORCH_SOLVE_DTYPE = getattr(torch, np.dtype(SOLVE_DTYPE).name)

# Adapted to the machine: may change speed and memory, never the results.
BLOCK_BYTES = 64 * 2 ** 20  # memory for one block's temporaries on the device


def rows_per_block(bytes_per_row: int) -> int:
    return max(1, BLOCK_BYTES // bytes_per_row)
