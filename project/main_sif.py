import sys, os
from time import time

import numpy as np
from datetime import datetime
from six.moves import cPickle

from SIF.params import params
import argparse
import pandas as pd

import lasagne
from sklearn.decomposition import TruncatedSVD

from SIF import data_io
from SIF.proj_model_sentiment import proj_model_sentiment
from SIF import eval

from main import DataSet


NUM_ITER = 10
##################################################

def adapt_ds(ds, labels):
    list_ds = [[ds[i].split(), labels[i]] for i in range(len(labels))]

    return list_ds


def sen2vec(sentence, w2v, w2weight):
    vec = [w2v[word] * w2weight.get(word, 1.) for word in sentence if word in w2v.vocab]

    vec = sum(vec) / len(vec)

    return vec


def str2bool(v):
    "utility function for parsing boolean arguments"
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')


def learner2bool(v):
    "utility function for parsing the argument for learning optimization algorithm"
    if v is None:
        return lasagne.updates.adam
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')


def get_pc(data, w2v, word2weight, params):
    "Comput the principal component"

    def get_weighted_average(data, w2v, word2weight):
        "Compute the weighted average vectors"
        n_samples = len(data)
        emb = np.zeros((n_samples, params.w2v_length))

        to_del = []

        for i in xrange(n_samples):

            w_vec = [word2weight.get(word, 1.) * w2v[word] for word in data[i][0] if word in w2v.vocab]

            if len(w_vec) > 0:
                emb[i, :] = sum(w_vec) / float(len(w_vec))
            else:
                to_del.append(i)

        emb = np.delete(emb, to_del, axis=0)

        return emb

    emb = get_weighted_average(data, w2v, word2weight)

    svd = TruncatedSVD(n_components=params.npc, n_iter=7, random_state=0)
    svd.fit(emb)
    return svd.components_


def train_util(model, train_data, test_data, params, categories):
    "utility function for training the model"
    start_time = time()
    try:
        for eidx in range(params.epochs):
            kf = data_io.get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:
                uidx += 1
                batch = [train_data[t] for t in train_index]

                # load the data
                (scores, input_vecs) = data_io.getDataProcessed(batch)

                # train
                cost = model.train_function(scores, input_vecs)

                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected')

            # evaluate

            # ds = eval.getAccDS(model, w2v, dev, params)
            # # rs = eval.getAccDS(model, w2v, train, params)
            # print("evaluation:")
            # print("train set -> ", rs)
            # print("Validation set -> ", ds)

            #print('Epoch ', (eidx + 1), 'Cost ', cost)
            sys.stdout.flush()

        # Get stats

        stats = eval.getStatsDS(model, test_data, categories)

        return stats

        # Save the trained model
        #
        # version = datetime.now().strftime("%Y%m%d%H%M%S")
        # print("\nSave model to version %s in %s" % (version, params.savepath))
        # version_path = params.savepath + "/" + version
        # os.mkdir(version_path)
        #
        # with open(version_path + "/model.net", "wb") as f:
        #     cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #     cPickle.dump(params.word2weight, f, protocol=cPickle.HIGHEST_PROTOCOL)



    except KeyboardInterrupt:
        print("Training interupted")
    end_time = time()
    print("total time:", (end_time - start_time))
    return None


##################################################

if __name__ == "__main__":

    # parse arguments
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-LW", help="Lambda for word embeddings (normal training).", type=float, default=1.e-5)
    parser.add_argument("-LC", help="Lambda for composition parameters (normal training).", type=float, default=1.e-6)
    parser.add_argument("-batchsize", help="Size of batch.", type=int, default=25)
    parser.add_argument("-dim", help="Size of input.", type=int, default=300)
    parser.add_argument("-memsize", help="Size of classification layer.",
                        type=int, default=150)
    parser.add_argument("-wordfile", help="Word embedding file.", default="SIF/auxiliary_data/glove.bin")
    parser.add_argument("-layersize", help="Size of output layers in models.", type=int, default=2400)
    parser.add_argument("-updatewords", help="Whether to update the word embeddings", default='False')
    parser.add_argument("-nonlinearity", help="Type of nonlinearity in projection and DAN model.",
                        type=int, default=1)
    parser.add_argument("-epochs", help="Number of epochs in training.", type=int, default=20)
    parser.add_argument("-minval", help="Min rating possible in scoring.", type=int)
    parser.add_argument("-maxval", help="Max rating possible in scoring.", type=int)
    parser.add_argument("-clip", help="Threshold for gradient clipping.", type=int)
    parser.add_argument("-eta", help="Learning rate.", type=float, default=0.05)
    parser.add_argument("-learner", help="Either AdaGrad or Adam.", default='adagrad')
    parser.add_argument("-weightfile",
                        help="The file containing the weights for words; used in weighted_proj_model_sim.",
                        default='SIF/auxiliary_data/enwiki_vocab_min200.txt')
    parser.add_argument("-weightpara", help="The parameter a used in computing word weights.", type=float, default=1e-3)
    parser.add_argument("-npc", help="The number of principal components to use.", type=int, default=0)
    parser.add_argument("-savepath", help="The path were the model will be saved", type=str, default="SIF/model")
    args = parser.parse_args()

    params = params()
    params.LW = args.LW
    params.LC = args.LC
    params.batchsize = args.batchsize
    params.hiddensize = args.dim
    params.memsize = args.memsize
    params.wordfile = args.wordfile
    params.layersize = args.layersize
    params.updatewords = str2bool(args.updatewords)
    params.epochs = args.epochs
    params.learner = learner2bool(args.learner)
    params.weightfile = args.weightfile
    params.weightpara = args.weightpara
    params.npc = args.npc
    params.savepath = args.savepath

    if args.eta:
        params.eta = args.eta
    params.clip = args.clip
    if args.clip:
        if params.clip == 0:
            params.clip = None
    params.minval = args.minval
    params.maxval = args.maxval
    if args.nonlinearity:
        if args.nonlinearity == 1:
            params.nonlinearity = lasagne.nonlinearities.linear
        if args.nonlinearity == 2:
            params.nonlinearity = lasagne.nonlinearities.tanh
        if args.nonlinearity == 3:
            params.nonlinearity = lasagne.nonlinearities.rectify
        if args.nonlinearity == 4:
            params.nonlinearity = lasagne.nonlinearities.sigmoid

    # load data

    data_reuters = DataSet()

    categories = data_reuters._labels

    train_data = adapt_ds(data_reuters.train_set, data_reuters.train_labels)
    test_data = adapt_ds(data_reuters.test_set, data_reuters.test_labels)

    w2v = data_io.getWordmap_new(params.wordfile)

    # define lenght vectors

    params.w2v_length = w2v['hello'].shape[0]

    # load weight
    if params.weightfile:
        word2weight = data_io.getWordWeight(params.weightfile, params.weightpara)
        params.word2weight = word2weight
        print('word weights computed using parameter a=' + str(params.weightpara))
    else:
        params.word2weight = dict()

    if params.npc > 0:
        params.pc = get_pc(train_data, w2v, params.word2weight, params)
    else:
        params.pc = []

    # Using SIF weighting scheme, compute embeddings for training data

    train_data_emb = [[sen2vec(s[0], w2v, params.word2weight), s[1]] for s in train_data]
    test_data_emb = [[sen2vec(s[0], w2v, params.word2weight), s[1]] for s in test_data]

    # load model
    model = proj_model_sentiment(w2v, params)

    # train
    f1 = [[],[],[],[]]
    prec = [[],[],[],[]]
    rec = [[],[],[],[]]

    for j in range(NUM_ITER):
        print("iter nº %i"%j)

        stats = train_util(model, train_data_emb, test_data_emb, params, categories)

        for i in range(len(categories)):
            f1[i].append(stats['F1'].values[i])
            prec[i].append(stats['Precision'].values[i])
            rec[i].append(stats['Recall'].values[i])


    # Reformat and print stats

    f1_mean = [np.round(np.mean(n),3) for n in f1]
    prec_mean = [np.round(np.mean(n),3) for n in prec]
    rec_mean = [np.round(np.mean(n),3) for n in rec]

    f1_std = [np.round(np.std(n),3) for n in f1]
    prec_std = [np.round(np.std(n),3) for n in prec]
    rec_std = [np.round(np.std(n),3) for n in rec]


    stats = pd.DataFrame()

    stats['Categories'] = categories
    stats['F1_mean'] = f1_mean
    stats['F1_std'] = f1_std
    stats['Precision_mean'] = prec_mean
    stats['Precision_std'] = prec_std
    stats['Recall_mean'] = rec_mean
    stats['Recall_std'] = rec_std

    print(stats.to_string())


"""  Categories  F1_mean  F1_std  Precision_mean  Precision_std  Recall_mean  Recall_std
0       earn    0.868   0.027           0.833          0.019        0.908       0.057
1        acq    0.822   0.031           0.834          0.070        0.816       0.032
2      crude    0.921   0.014           0.993          0.021        0.860       0.020
3       corn    0.857   0.035           0.927          0.080        0.800       0.000"""