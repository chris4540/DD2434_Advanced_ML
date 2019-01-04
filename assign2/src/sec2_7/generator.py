import numpy as np
import random as rd

# GENERATOR FUNCTIONS _____________________________________________________________________________________________________________________________

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

# SOLUTION FUNCTIONS _______________________________________________________________________________________________________________________
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

    def get_obs_seq_prob(self):
        """
        Get the probability of observation sequence p(O_{1:T})

        Return:
            the probability of observation sequence
        """
        return 0

class MixtureHMMs(object):

    def __init__(self, num_models):
        pass

    def classify(self):
        pass


# _____________________________________________________________________________________________________________________________________

if __name__ == "__main__":
    nr_vehicles = 10
    nr_classes = 10
    nr_rows = 10
    nr_columns = 10

    class_prob, start_prob, transition_prob, emission_prob = define_HMMs(nr_classes, nr_rows, nr_columns)
    print(type(emission_prob))
    print(emission_prob.shape)
    # print("Class probabilities\n", class_prob)
    # print("\nStart probabilities\n", start_prob)
    # print("\nTransition probabilities\n", transition_prob)
    # print("\nEmission probabilities\n", emission_prob)

    # targets, data = generate_data(nr_vehicles, nr_classes, nr_rows, nr_columns)
    # print("\nObserved sequences\n",data)
    # print("\nTrue classes\n", targets)

