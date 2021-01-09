from __future__ import print_function

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

NUM_LABELS = 4

def getWordmap_new(textfile):
    model = KeyedVectors.load_word2vec_format(textfile, binary=True)
    model.syn0norm = model.syn0 # prevent recalc of normed vectors
    return model

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def getDataProcessed(batch):
    input_vecs = []
    scores = []
    for i in batch:
        input_vecs.append(i[0])
        temp = np.zeros(NUM_LABELS)
        temp[i[1]] = 1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype='float32')
    input_vecs = np.asarray(input_vecs,dtype='float32')
    return (scores,input_vecs)

def getWordWeight(weightfile, a=1e-3):
    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)
    return word2weight
