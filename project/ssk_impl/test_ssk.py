"""
Simple text for the correctness of code
"""
import numpy as np
import ssk_kernel_c as kernel


if __name__ == '__main__':

    decay = 0.8
    k = kernel.ssk_kernel("cat", "car", 2, decay)
    ans = 1.0 / (2 + decay**2)

    assert (np.abs(k - ans) < 1e-6 )
