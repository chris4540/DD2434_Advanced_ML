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


def priorSample(nParams, nSamples):
    """
    Args:
        nParams:
        nSamples:
    """
    sigma_sq = 1000
    cov = sigma_sq * np.identity(nParams)
    mean = np.zeros(nParams)
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
    thetas = priorSample(3, S)

    evidence = np.zeros([4, 512])


    for i in range(4):
        for j in range(512):
            evidence[i][j] = computeEvidence(l[j], i, thetas)


    index = create_index_set(evidence)


    # max_ = np.argmax(evidence, axis=1)
    # min_ = np.argmin(evidence, axis=1)
    # sum_ = np.sum(evidence, axis=1)
    fig, ax = plt.subplots()
    plt.plot(evidence[0, index], 'm', label="P($\mathcal{D}$ | ${M}_0$)")
    plt.plot(evidence[1, index], 'b', label="P($\mathcal{D}$ | ${M}_1$)")
    plt.plot(evidence[2, index], 'r', label="P($\mathcal{D}$ | ${M}_2$)")
    plt.plot(evidence[3, index], 'g', label="P($\mathcal{D}$ | ${M}_3$)")
    plt.legend()
    ax.set_xlim(0, 512)
    ax.set_ylim(0)
    ax.set_xlabel('Data set')
    ax.set_ylabel('Evidence')
    fig.tight_layout()
    plt.savefig("../fig/Q22-all.png", dpi=100)

    ax.set_xlim(0, 80)
    plt.xticks(list(range(0, 81, 10)))
    plt.savefig("../fig/Q22-sub.png", dpi=100)

