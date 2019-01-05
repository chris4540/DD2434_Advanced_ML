import pickle
import numpy as np


class Tournament(object):

    def __init__(self, p_init, p_trans):
        pass

    def set_obs_seqs(self, obs_seqs):
        self.alpha = None
        self.beta = None

        self.obs_seqs =obs_seqs

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

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
