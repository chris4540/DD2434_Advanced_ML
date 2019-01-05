import pickle
import numpy as np


class Tournament(object):

    def __init__(self, p_init, p_trans):
        self.p_init = p_init
        self.p_trans = p_trans
        self.n_states = p_trans.shape[0]

        # random initialize our observation model params

    def set_obs_seqs(self, obs_seqs):
        self.alpha = None
        self.beta = None

        self.n_obs = obs_seqs.shape[0]
        self.obs_length = obs_seqs.shape[1]
        self.obs_seqs = obs_seqs

    def forward_pass(self):
        self.alpha = np.zeros((self.n_obs, self.obs_length, self.n_states))

        # base case


    def backward_pass(self):
        self.beta = np.zeros((self.n_obs, self.obs_length, self.n_states))

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
    tour.forward_pass()
    tour.backward_pass()
