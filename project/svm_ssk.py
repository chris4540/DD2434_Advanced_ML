from sklearn import svm
import data
import ssk_kernel.ssk_kernel_c as kernel
import numpy as np
import dataset
from dataset import DataSet
from joblib import Parallel, delayed

NUM_WORKERS = 1

class SVM:
    def __init__(self, name, ngram_size, lambda_decay=0.8):
        self.name = name
        self.ngram_size = ngram_size
        self.lambda_decay = lambda_decay
        self.text_clf = svm.SVC(kernel='precomputed')
        self.ssk_value_table = dict()

    def train(self, data_set):
        print('Train model ' + self.name)

        gram = self._get_train_gram_mat(data_set.train_set)
        # gram = self.gram_matrix(data_set.train_set, data_set.train_set)
        self.text_clf.fit(gram, data_set.train_labels)

    def predict(self, data_set):
        print('Model: {} is predicting...  '.format(self.name))
        gram = self._get_pred_gram_mat(data_set.test_set, data_set.train_set)
        pred_target = self.text_clf.predict(gram)
        data_set.evaluate_prediction(pred_target)
        return pred_target

    def ssk_value(self, doc1, doc2):

        if self.has_value_in_ssk_table(doc1, doc2):
            return self.get_value_from_ssk_table(doc1, doc2)

        ret = kernel.ssk_kernel(doc1, doc2, self.ngram_size, self.lambda_decay)
        # print(ret)
        return ret

    def has_value_in_ssk_table(self, doc1, doc2):
        doc1_, doc2_ = self._align_pair(doc1, doc2)
        return ((doc1_, doc2_) in self.ssk_value_table)

    def get_value_from_ssk_table(self, doc1, doc2):
        doc1_, doc2_ = self._align_pair(doc1, doc2)
        return self.ssk_value_table[(doc1_, doc2_)]


    def set_value_to_ssk_table(self, doc1, doc2, val):
        doc1_, doc2_ = self._align_pair(doc1, doc2)
        self.ssk_value_table[(doc1_, doc2_)] = val

    @staticmethod
    def _align_pair(doc1, doc2):
        doc1_ = doc1
        doc2_ = doc2
        if len(doc2_) < len(doc1_):
            doc1_, doc2_ = doc2_, doc1_

        return doc1_, doc2_

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
            # save the result to ssk_value_table
            doc1 = train_set[i]
            doc2 = train_set[j]

            if not self.has_value_in_ssk_table(doc1, doc2):
                self.set_value_to_ssk_table(doc1, doc2, res)

        return ret

    def _get_pred_gram_mat(self, predict_set, train_set):
        n_test = len(predict_set)
        n_train = len(train_set)
        ret = np.zeros((n_test, n_train))

        # idx = np.indices(ret.shape)


        result = Parallel(n_jobs=NUM_WORKERS)(
            delayed(self.ssk_value)(predict_set[i], train_set[j])
            for i,j in np.ndindex(*ret.shape))

        # un-pack the calculated results
        for t, (i, j) in enumerate(np.ndindex(*ret.shape)):
            res = result[t]
            ret[i,j] = res
            # save the result to ssk_value_table
            doc1 = predict_set[i]
            doc2 = train_set[j]

            if not self.has_value_in_ssk_table(doc1, doc2):
                self.set_value_to_ssk_table(doc1, doc2, res)

        return ret




if __name__ == '__main__':
    data_set = DataSet()

    # make a small subset for testing
    data_set.train_set = data_set.train_set[:5]
    data_set.train_labels = data_set.train_labels[:5]

    data_set.test_set = data_set.test_set[:10]
    data_set.test_labels = data_set.test_labels[:10]

    test_model = SVM("test_k3_lambda0.8", 3, lambda_decay=0.8)
    test_model.train(data_set)
    test_model.predict(data_set)
    # print(test_model.ssk_value_table)
