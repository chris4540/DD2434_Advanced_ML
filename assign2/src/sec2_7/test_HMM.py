import numpy as np
from generator import HiddenMarkovModel


def test_HMM():
    """
    Add a test case to ensure code works poperly
    """
    A_str = "0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0"
    e_str = "0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9"
    pi_str = "1.0 0.0 0.0 0.0"
    obs_seq_str = "0 1 2 3 0 1 2 3"
    n_state = 4
    n_obs_type = 4

    p_trans = np.fromstring(A_str, dtype=float, sep=' ').reshape((n_state, n_state))
    p_emis = np.fromstring(e_str, dtype=float, sep=' ').reshape(
        (n_state, n_obs_type))
    p_init = np.fromstring(
        pi_str, dtype=float, sep=' ').reshape((1, n_state))
    obs_seq = np.fromstring(obs_seq_str, dtype=int, sep=' ')

    model = HiddenMarkovModel(p_init, p_trans, p_emis)
    model.set_obs_seq(obs_seq)

    p_obs = model.get_obs_seq_prob()

    ans = 0.090276
    assert (np.abs(p_obs - ans) < 1e-6)
    print("test_HMM: Pass")

if __name__ == "__main__":
    test_HMM()
