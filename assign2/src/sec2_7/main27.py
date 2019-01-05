import numpy as np
import random as rd

################################
# GENERATOR FUNCTIONS
################################
def define_HMMs(K,R,M):
    # Class probabilities - one class is much more probable than the rest
    pi = np.zeros((K))
    class1 = rd.randint(0,K-1)
    pi[class1] = 0.5
    for k in range(K):
        if pi[k]==0.0:
            pi[k] = 0.5/(K-1)

    # Start probabilities - UNIFORM
    init = 1/R*np.ones((R))

    # Transition probabilities - from each row, there are only two possible next states, with varying probabilities
    A = np.zeros((K, R, R))
    for k in range(K):
        for r in range(R):
            rand = rd.randint(10,20)
            row1 = rd.randint(0,R-1)
            row2 = rd.randint(0,R-1)
            while(row2 == row1):
                row2 = rd.randint(0,R-1)

            A[k,r,row1] = rand/20
            A[k,r,row2] = (20-rand)/20

    # Emission probabilities - different noise for different classes, but same noise for all rows within that class
    E = np.zeros((K, R, R))
    for k in range(K):
        rand = rd.randint(10,20)
        for r in range(R):
            E[k,r,r] = rand/20
            E[k,r,(r+1)%nr_rows] = (20-rand)/40
            E[k,r,(r-1)%nr_rows] = (20-rand)/40

    return pi, init, A, E


def generate_states(k,R,M):
    init = start_prob
    X = np.zeros((M), dtype=int)

    rand = rd.random()
    sum_steps = 0.0
    for r in range(R):
        if rand>=sum_steps and rand<sum_steps+init[r]:
            X[0] = r
            break
        sum_steps += init[r]

    for m in range(1,M):
        A = transition_prob[k,X[m-1],:]
        rand = rd.random()
        sum_steps = 0.0
        for r in range(R):
            if rand>=sum_steps and rand<sum_steps+A[r]:
                X[m] = r;
                break
            sum_steps += A[r]

    return X


def generate_observations(k,R,M,X):
    Z =  np.zeros((M), dtype=int)
    for m in range(M):
        E = emission_prob[k,X[m],:]
        rand = rd.random()
        sum_steps = 0.0
        for r in range(R):
            if rand>=sum_steps and rand<sum_steps+E[r]:
                Z[m] = r
                break
            sum_steps += E[r]

    return Z


def generate_data(N,K,R,M):
    classes = np.zeros((N), dtype=int)
    observations = np.zeros((N,M), dtype=int)

    for n in range(N):
        rand = rd.random()
        sum_steps = 0.0
        for k in range(K):
            if rand>=sum_steps and rand<sum_steps+class_prob[k]:
                k_n = k;
                break
            sum_steps += class_prob[k]

        classes[n] = k_n
        observations[n,:] = generate_observations(k_n, R, M, generate_states(k_n, R, M))

    return classes, observations

################################
# SOLUTION FUNCTIONS / CLASSES
################################
class HiddenMarkovModel(object):

    p_init = None
    p_trans = None
    p_emis = None

    def __init__(self, p_init, p_trans, p_emis):
        """
        Args:
        p_init (np.ndarray): initial state probability vector
        p_trans (np.ndarray): Transition matrix,
        p_emis (np.ndarray): Emission probability matrix
        """
        self.num_state = p_emis.shape[0]
        self.obs_state = p_emis.shape[1]

        self.p_init = p_init
        self.p_trans = p_trans
        self.p_emis = p_emis

    def set_obs_seq(self, sequence):
        """
        Set observation sequecne to the model
        """
        self.obs_seq = sequence

    def get_obs_seq_prob(self):
        """
        Get the probability of observation sequence p(O_{1:T})

        Return:
            the probability of observation sequence
        """
        self.forward_pass()
        return self.prob_obs

    def forward_pass(self):
        """
        Copy from the provided code in forward_backward.py
        Modifiy to fit into this class
        """
        M = len(self.obs_seq)
        alpha = np.zeros((M, self.num_state))

        # base case
        O = []
        for r in range(self.num_state):
            O.append(self.p_emis[r, self.obs_seq[0]])
        alpha[0, :] = self.p_init * O[:]

        # recursive case
        for m in range(1, M):
            for r2 in range(self.num_state):
                for r1 in range(self.num_state):
                    transition = self.p_trans[r1, r2]
                    emission = self.p_emis[r2, self.obs_seq[m]]
                    alpha[m, r2] += alpha[m-1, r1] * transition * emission

        # save down the alpha values
        self.alpha = alpha
        self.prob_obs = np.sum(alpha[M-1, :])

class MixtureHMMs(object):

    models = []

    def __init__(self, n_models, p_init, p_trans, p_emis):
        self.models = []
        self.n_models = n_models

        for k in range(n_models):
            self.models.append(
                HiddenMarkovModel(p_init, p_trans[k], p_emis[k]))

    @staticmethod
    def renormalize_prob_vec(probs):
        """
        """
        return probs / np.sum(probs)

    def set_obs_seqs(self, obs_seqs):
        """
        Args:
        obs_seqs (np.ndarray): a set of observation sequences. The size should be
            N x L, where N is the number of observation sequences and L is the
            length of each observation sequence.
        """
        self.obs_seqs = obs_seqs

    def get_obs_probs(self):
        num_obs = self.obs_seqs.shape[0]

        self.obs_probs = np.zeros((num_obs, self.n_models))
        for n in range(num_obs):
            for k in range(self.n_models):
                model = self.models[k]
                model.set_obs_seq(self.obs_seqs[n])
                p = model.get_obs_seq_prob()
                self.obs_probs[n, k] = p

    @staticmethod
    def get_L2_norm(x, y):
        return np.sqrt(np.sum((x-y)**2))

    def cluster_data(self):
        num_obs = self.obs_seqs.shape[0]

        # cls_prob: \pi_k in our problem
        cls_prob = self.renormalize_prob_vec(np.random.rand(self.n_models))

        tau = np.zeros((num_obs, self.n_models))

        for iter_ in range(100): # iteration
            for n in range(num_obs):
                tau[n, :] = self.obs_probs[n, :] * cls_prob
                # renormalize
                tau[n, :] = self.renormalize_prob_vec(tau[n, :])

            # update class prob
            new_cls_prob = np.mean(tau, axis=0)
            if self.get_L2_norm(new_cls_prob, cls_prob) < 1e-6:
                print("Iteration:", iter_)
                break
            else:
                # update class prob
                cls_prob = new_cls_prob

        # save down estimated results
        self.cls_prob = cls_prob
        self.tau = tau

    def get_estimated_classes(self):
        est_classes = np.argmax(self.tau, axis=1)
        return est_classes


if __name__ == "__main__":
    nr_vehicles = 10
    nr_classes = 10
    nr_rows = 10
    nr_columns = 10

    class_prob, start_prob, transition_prob, emission_prob = define_HMMs(
        nr_classes, nr_rows, nr_columns)
    # print(type(emission_prob))
    # print(emission_prob.shape)
    print("Class probabilities\n", class_prob)
    # print("\nStart probabilities\n", start_prob)
    # print("\nTransition probabilities\n", transition_prob)
    # print("\nEmission probabilities\n", emission_prob)

    targets, data = generate_data(nr_vehicles, nr_classes, nr_rows, nr_columns)
    # print("\nObserved sequences\n",data)
    print("\nTrue classes\n", targets)

    mix_HMMs = MixtureHMMs(nr_classes, start_prob,
                           transition_prob, emission_prob)
    mix_HMMs.set_obs_seqs(data)
    mix_HMMs.get_obs_probs()
    mix_HMMs.cluster_data()
    est = mix_HMMs.get_estimated_classes()
    print(est)
    print(targets - est)
