import math
import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    center = (size - 1) / 2
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = center - i
            y = center - j
            out[i][j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))

    out /= out.sum()
    return out.tolist()