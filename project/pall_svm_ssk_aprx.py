from sklearn import svm
import data
import ssk_impl.ssk_kernel_approx_c as kernel
import numpy as np
import dataset
import operator
from dataset import DataSet
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support
import functools
import time
import os

print = functools.partial(print, flush=True)
NUM_WORKERS = 22

class SVM:
    def __init__(self, name, ngram_size, lambda_decay=0.5):
        self.name = name
        self.ngram_size = ngram_size
        self.lambda_decay = lambda_decay
        self.text_clf = svm.SVC(kernel='precomputed')
        self.nb_basis = 200

    @staticmethod
    def extract_strings(docs, n):
        """
        :param docs: list of documents
        :param n: substring size
        :return: sorted list of substrings (first one is the most frequent)
        """

        strings = dict()
        for doc in docs:
            for i in range(len(doc) - n + 1):
                line = doc[i: i + n]
                if line not in strings:
                    strings[line] = 1
                else:
                    strings[line] += 1

        sorted_strings = sorted(strings.items(), key=operator.itemgetter(1), reverse=True)
        sorted_strings = [x[0] for x in sorted_strings]
        return sorted_strings

    def train(self, data_set):
        print('Train model ' + self.name)
        sorted_features = self.extract_strings(data_set.train_set, self.ngram_size)

        self.feature_basis = sorted_features[:self.nb_basis]
        #
        gram = self._get_train_gram_mat(data_set.train_set)
        self.text_clf.fit(gram, data_set.train_labels)

    def ssk_value(self, doc1, doc2):

        # todo: normalization

        ret = kernel.ssk_kernel_approx(
            doc1, doc2, self.ngram_size, self.lambda_decay, self.feature_basis)
        return ret

    def _get_train_gram_mat(self, train_set):

        n_sample = len(train_set)

        ret = np.diag(np.ones(n_sample))

        # calculate the upper-triangle matrix first.
        # The diagonal should be 1 be default
        tu_idx = np.triu_indices(n_sample, k=1)
        result = Parallel(n_jobs=NUM_WORKERS)(
            delayed(self.ssk_value)(train_set[i], train_set[j])
            for i,j in zip(*tu_idx))

        # un-pack the calculated results
        for t, (i, j) in enumerate(zip(*tu_idx)):
            res = result[t]
            ret[i,j] = res
            ret[j,i] = res
        return ret

    def _get_pred_gram_mat(self, predict_set, train_set):
        n_test = len(predict_set)
        n_train = len(train_set)
        ret = np.zeros((n_test, n_train))

        result = Parallel(n_jobs=NUM_WORKERS)(
            delayed(self.ssk_value)(predict_set[i], train_set[j])
            for i,j in np.ndindex(*ret.shape))

        # un-pack the calculated results
        for t, (i, j) in enumerate(np.ndindex(*ret.shape)):
            res = result[t]
            ret[i,j] = res

        return ret

    def predict(self, data_set):
        print('Model: {} is predicting...  '.format(self.name))
        gram = self._get_pred_gram_mat(data_set.test_set, data_set.train_set)
        pred_target = self.text_clf.predict(gram)
        return pred_target



if __name__ == '__main__':
    data_set = DataSet()
    print("# of training set", len(data_set.train_set))
    print("# of test set", len(data_set.test_set))

    k = int(os.getenv("k_val"))
    decay = float(os.getenv("decay_val"))

    model_name = "SSK_approx_k{}_decay{}".format(k, decay)
    test_model = SVM(model_name, k, lambda_decay=decay)
    stime = time.time()
    test_model.train(data_set)
    etime = time.time()
    print("Training: ", etime - stime)

    stime = time.time()
    prd = test_model.predict(data_set)
    etime = time.time()

    print("Pred: ", etime - stime)
    print("Prediction Classes:")
    print(prd)

    y_true = data_set.test_labels
    print("True Classes:")
    print(y_true)

    scores = precision_recall_fscore_support(y_true, prd, average=None)
    precision = scores[0]
    recall = scores[1]
    f1_score = scores[2]
    support = scores[3]
    print("precision:", precision)
    print("recall:", recall)
    print("f1_score:", f1_score)
    print("support:", support)
