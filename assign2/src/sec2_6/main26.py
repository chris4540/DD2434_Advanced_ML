import pickle
import numpy as np
from numpy.random import normal

class Tournament(object):

    mu = None
    sigma2 = None

    def __init__(self, p_init, p_trans):
        self.p_init = p_init
        self.p_trans = p_trans
        self.n_states = p_trans.shape[0]

    def get_p_emis(self, state, seq_idx, obs):
        """
        Args:
            state (int): the value of z^r_m
            seq_idx (int): the sequence index, m
            obs (float): the observation value
        """
        mean = self.mu[state, seq_idx]
        var = self.sigma2[state, seq_idx]

        ret = np.exp(-((obs - mean)**2) * 0.5 / var) / np.sqrt(2 * np.pi * var)

        if ret < 1e-12:
            ret = 1e-12
        return ret

    def set_obs_seqs(self, obs_seqs):
        self.alpha = None
        self.beta = None

        #
        self.n_obs = obs_seqs.shape[0]
        self.obs_length = obs_seqs.shape[1]
        self.obs_seqs = obs_seqs

        # random initialize our observation model params
        size = (self.n_states, self.obs_length)
        self.mu = normal(33, 10, size=size)
        self.sigma2 = np.ones(size) * 2 * 10

    def forward_pass(self):
        self.alpha = np.zeros((self.n_obs, self.obs_length, self.n_states))
        self.scaling = np.zeros((self.n_obs, self.obs_length))
        for r in range(self.n_obs):
            obs_seq = self.obs_seqs[r]
            alpha, scaling = self._forward(obs_seq)
            self.alpha[r, :, :] = alpha
            self.scaling[r, :] = scaling

    def _forward(self, obs_seq):
        alpha = np.zeros((self.obs_length, self.n_states))
        scaling = np.zeros(self.obs_length)

        obs_len = len(obs_seq)
        # base case
        for j in range(self.n_states):
            alpha[0, j] = self.p_init[j] * self.get_p_emis(j, 0, obs_seq[0])

        # scale the alpha[0, :]
        scaling[0] = 1 / np.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :] * scaling[0]

        # recursive case
        for t in range(1, obs_len):
            for j in range(self.n_states):
                p = 0
                for i in range(self.n_states):
                    p += alpha[t-1, i] * self.p_trans[j, i]

                alpha[t, j] = p * self.get_p_emis(j, t, obs_seq[t])

            scaling[t] = 1 / np.sum(alpha[t,:])
            alpha[t, :] = alpha[t, :] * scaling[t]

        return alpha, scaling

    def backward_pass(self):
        self.beta = np.zeros((self.n_obs, self.obs_length, self.n_states))

        for r in range(self.n_obs):
            obs_seq = self.obs_seqs[r]
            scaling = self.scaling[r, :]
            beta = self._backward(obs_seq, scaling)
            self.beta[r, :, :] = beta

    def _backward(self, obs_seq, scaling):
        beta = np.zeros((self.obs_length, self.n_states))

        obs_len = len(obs_seq)
        # base case
        beta[-1, :] = 1

        # recursive case
        for t in range(obs_len-2, -1, -1):
            for i in range(self.n_states):
                p = 0
                for j in range(self.n_states):
                    p += self.p_trans[i, j] * self.get_p_emis(j, t+1, obs_seq[t+1])

                beta[t, i] = p * scaling[t]
        return beta

    def forward_backward(self):

        self.gamma = np.zeros((self.n_obs, self.obs_length, self.n_states))

        for r in range(self.n_obs):
            alpha = self.alpha[r, :, :]
            beta = self.beta[r, :, :]
            self.gamma[r, :, :] = self._get_gamma(alpha, beta)

    def _get_gamma(self, alpha, beta):
        gamma = np.zeros((self.obs_length, self.n_states))
        for t in range(self.obs_length):
            p_vec = alpha[t, :] * beta[t, :]

            gamma[t, :]  = self.normalize_p_vec(p_vec)

        return gamma

    def learn(self):

        for _ in range(50):
            self.forward_pass()
            # print(self.alpha)
            self.backward_pass()
            # print(self.beta)
            self.forward_backward()
            # estimate mu again
            for i in range(self.n_states):
                for m in range(self.obs_length):
                    num = 0
                    denom = 0
                    for r in range(self.n_obs):
                        num += self.gamma[r, m, i] * self.obs_seqs[r, m]
                        denom += self.gamma[r, m, i]
                    self.mu[i, m] = num / denom

            # estimate sigma2
            for i in range(self.n_states):
                for m in range(self.obs_length):
                    num = 0
                    denom = 0
                    for r in range(self.n_obs):
                        num += self.gamma[r, m, i] * (
                                self.obs_seqs[r, m] - self.mu[i,m])**2
                        denom += self.gamma[r, m, i]
                    self.sigma2[i, m] = num / denom

    @staticmethod
    def normalize_p_vec(p_vec):
        """
        """
        return p_vec / np.sum(p_vec)


def load_obj(fname):
    """
    """
    with open(fname, 'rb') as f:
        ret = pickle.load(f)
    return ret

if __name__ == "__main__":
    # load the result from the generator
    data = load_obj("./sequence_output.pkl")

    p_init = np.array([0.25, 0.75])
    p_trans = np.array([[0.25, 0.75],
                        [0.75, 0.25]])
    tour = Tournament(p_init, p_trans)

    obs_seqs = []
    for seq in data[(1,2)].values():
        obs_seqs.append(np.array(seq)[:, 0])

    obs_seqs = np.array(obs_seqs)

    # set obs
    tour.set_obs_seqs(obs_seqs)
    tour.learn()
    print(tour.mu)
    print(tour.sigma2)
    # tour.forward_pass()
    # tour.backward_pass()
    # tour.forward_backward()

