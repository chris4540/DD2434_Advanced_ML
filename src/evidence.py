import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import multivariate_normal

def generateDataset():
    """
    Create the output dataset {t_i}, i = 1..9
    """
    combinations = list(product([-1, 1], repeat=9))
    sets = []
    for l in combinations:
        arr = np.asarray(l)
        grid = np.reshape(arr, (3, 3))
        sets.append(grid)
    np.random.shuffle(sets)
    return sets

def drawDataset(dataset):
    for i in range(3):
        print("|", end="")
        for j in range(3):
            print(dataset[i][j], "|", end="")
        print()


def model(mdl_idx, dataset, thetas):
    """
    Return an array of prob.
    """
    thetas = thetas.copy()
    nSamples = thetas.shape[0]
    if mdl_idx == 0:
        return np.ones(nSamples) / 512.0

    if mdl_idx == 1:
        thetas[:, 0] = 0
        thetas[:, 2] = 0

    if mdl_idx == 2:
        thetas[:, 0] = 0

    # make a mesh
    i_coords, j_coords = np.meshgrid([-1, 0, 1], [1, 0, -1], indexing='xy')

    x = np.vstack([np.ones(9), i_coords.flatten(), j_coords.flatten()])
    z = thetas.dot(x) * dataset.flatten()
    probs = 1.0 / (1.0 + np.exp(-z))
    return np.prod(probs, axis=1)


def priorSample(nSamples):
    """
    Args:
        nParams:
        nSamples:
    """
    nParams = 3
    # sigma_sq = 1e4
    # sigma_sq = 1e2
    # sigma_sq = 1e10
    # sigma_sq = 1e-3
    sigma_sq = 1e3
    cov = sigma_sq * np.identity(nParams)
    # cov[0, 1] = -400
    # cov[1, 0] = -400
    # cov[0, 2] = 700
    # cov[2, 0] = 700
    # cov[1, 2] = 900
    # cov[2, 1] = 900
    # mean = np.zeros(nParams)
    mean = np.ones(nParams) * 5
    theta = np.random.multivariate_normal(mean, cov, nSamples)
    return theta


def computeEvidence(dataset, modelNumber, samples):

    return np.mean(model(modelNumber, dataset, samples))

def create_index_set(evidence):
    """
    Reorder the dataset according to the evidence prob.

    Return:
        ordered index
    """
    vals = -np.sum(evidence, axis=0)
    sort_index = np.argsort(vals)
    return sort_index

if __name__ == "__main__":
    S = int(20000)
    l = generateDataset()
    thetas = priorSample(S)

    evidence = np.zeros([4, 512])


    for i in range(4):
        for j in range(512):
            evidence[i][j] = computeEvidence(l[j], i, thetas)

    print("Model 1: count > 1/512: ", np.count_nonzero(evidence[1,:] > 1/512))
    print("Model 2: count > 1/512: ", np.count_nonzero(evidence[2,:] > 1/512))
    print("Model 3: count > 1/512: ", np.count_nonzero(evidence[3,:] > 1/512))

    index = create_index_set(evidence)


    # fig, ax = plt.subplots()
    # plt.plot(evidence[0, index], 'm', label="P($\mathcal{D}$ | ${M}_0$)")
    # plt.plot(evidence[1, index], 'b', label="P($\mathcal{D}$ | ${M}_1$)")
    # plt.plot(evidence[2, index], 'r', label="P($\mathcal{D}$ | ${M}_2$)")
    # plt.plot(evidence[3, index], 'g', label="P($\mathcal{D}$ | ${M}_3$)")
    # plt.legend()
    # ax.set_xlim(0, 512)
    # ax.set_ylim(0, 0.13)
    # ax.set_xlabel('Data set')
    # ax.set_ylabel('Evidence')
    # fig.tight_layout()
    # plt.savefig("../fig/Q26c-all.png", dpi=100)

    # ax.set_xlim(0, 80)
    # plt.xticks(list(range(0, 81, 10)))
    # plt.savefig("../fig/Q26c-sub.png", dpi=100)

