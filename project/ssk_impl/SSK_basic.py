import numpy as np
import tools


def find_all_occurences(text, c):
    """

    :param text:
    :param c: character to search in text
    :return: return indeces of ocurrences of c in text (generator)
    """
    idx = text.find(c)
    while idx != -1:
        yield idx
        idx = text.find(c, idx + 1)



def SSK(s, t, n, lam):
    assert 0 < lam <= 1, "Error, the length of the lambda constant should be between 0 and 1, but its value is %f" % lam

    s_len = len(s)
    t_len = len(t)

    if min(s_len, t_len) < n:
        return 0.

    # First compute K' and K''

    kp = np.zeros((n, s_len + 1, t_len + 1))  # The aditional 1 will be allways zero, only to return 0 when negative -1
    kpp = np.zeros((n, s_len + 1, t_len + 1))

    kp[0, :, :] = 1.

    for i in range(1, n):

        for j in range(i - 1, s_len):

            for k in range(i - 1, t_len):
                kpp[i, j, k] = lam * (kpp[i, j, k - 1] + (s[j] == t[k]) * lam * kp[i - 1, j - 1, k - 1])

                kp[i, j, k] = lam * kp[i, j - 1, k] + kpp[i, j, k]

    # Now computing Kernel

    ker = 0

    for j in range(n-1, s_len):
        ker += sum([kp[n - 1, j - 1, k - 1] for k in find_all_occurences(t, s[j])])

    return ker * lam ** 2


def SSK_normalized(s, t, n, lam):

    s=tools.clean_string(s)
    t=tools.clean_string(t)
    return SSK(s, t, n, lam) / np.sqrt(SSK(s, s, n, lam) * SSK(t, t, n, lam))




# lbda = 0.8

# print('K("cat","car") = %.4f = %.4f' % (SSK_normalized("cat", "car",2,lbda), 1 / (2 + lbda ** 2)))
