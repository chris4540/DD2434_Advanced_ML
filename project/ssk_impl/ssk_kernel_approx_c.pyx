import math
import numpy as np
cimport numpy as np
cimport cython

def find_indices(str s, int n, str string):

    cdef int start = 0
    cdef int end = start + n

    cdef list valid_indices = list()
    for i in range(len(s) - n + 1):
        if s[start:end] == string:
            valid_indices.append((start, end))
        start += 1
        end += 1
    return valid_indices

def _ssk_kernel_approx(str s, str t, int n, float decay, list basis):
    #
    cdef float kernel_sum =  0

    for str_ in basis:
        s_indices = find_indices(s, n, str_)
        t_indices = find_indices(t, n, str_)

        for s_index in s_indices:
            for t_index in t_indices:
                kernel_sum += np.power(
                    decay, (s_index[1] - s_index[0]) + (t_index[1] - t_index[0]))
    return kernel_sum

def ssk_kernel_approx(str s, str t, int n, float decay, list basis):
    cdef float st = _ssk_kernel_approx(s, t, n, decay, basis)
    cdef float ss = _ssk_kernel_approx(s, s, n, decay, basis)
    cdef float tt = _ssk_kernel_approx(t, s, n, decay, basis)

    if ss == 0 or tt == 0:
       return 1e-20
    cdef float ret = st / np.sqrt(ss * tt)
    return ret
