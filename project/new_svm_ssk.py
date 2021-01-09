import numpy as np
from sklearn.svm import SVC
from ssk_kernel_wrap import SSKKernel


class StringSVM:
    def __init__(self, name, k, decay):
        self.name = name
        self.kernel_obj = SSKKernel(k, decay)

        self.documents = dict()
        self.n_data = 0
        self.text_clf = SVC(kernel=self.kernel_fun)

        self.n_chunk = 3

    def train(self, train_data, train_label):
        """
        Use Iterative Chunking to train the svm
        """
        # add train data into documents
        for i, doc in enumerate(train_data):
            self.documents[i] = doc
        self.n_data = len(train_data)

        # create a index matrix
        X_idx = np.arange(len(train_data))

        # iterative chunking training
        support_vec = []
        iter_ = 0
        while len(X_idx) > 0:

            # sample chuck to have at least 2 classes
            for _ in range(1000):
                train_label = np.array(train_label)
                samples = np.random.choice(X_idx, self.n_chunk)
                if len(set(train_label[samples])) == 1:
                    # only one class was picked. resample
                    continue
                else:
                    break

            working_set = samples

            x = np.array(samples).reshape(-1, 1)
            if len(support_vec) > 0:
                x = np.vstack((x, support_vec))
            print(x)
            y = train_label[x.flatten()]
            print("Training iter:", iter_)
            self.text_clf.fit(x, y)

            support_vec = x[self.text_clf.support_]
            # remove trained samples from x_idx
            X_idx = np.delete(X_idx, support_vec.flatten())
            iter_ += 1
            print("# of support vectors for each class", self.text_clf.n_support_)
            print("# of support vectors", np.sum(self.text_clf.n_support_))
            print("# of remaining samples", len(X_idx))

    def predict(self, test_data):
        for i, doc in enumerate(test_data):
            self.documents[i+self.n_data] = doc

        X_new_idx = np.arange(len(test_data)).reshape(-1, 1) + self.n_data

        self.n_data += len(test_data)
        return self.text_clf.predict(X_new_idx)

    def kernel_fun(self, X1, X2):

        if np.array_equal(X1, X2):
            return self._get_train_gram_mat(X1)

        ret = np.zeros((len(X1), len(X2)))


        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                k = int(X1[i][0])
                l = int(X2[j][0])
                doc1 = self.documents[k]
                doc2 = self.documents[l]
                ret[i, j] = self.kernel_obj(doc1, doc2)
        return ret

    def _get_train_gram_mat(self, train_set_idx):
        """
        Use the sysmetric propertey to speed up
        """
        size = len(train_set_idx)
        ret = np.diag(np.ones(size))

        # calculate the upper-triangle matrix first.
        # The diagonal should be 1 be default
        tu_idx = np.triu_indices(size, k=1)
        for i, j in zip(*tu_idx):
            k = int(train_set_idx[i][0])
            l = int(train_set_idx[j][0])
            doc1 = self.documents[k]
            doc2 = self.documents[l]
            val = self.kernel_obj(doc1, doc2)
            ret[i, j] = val
            ret[j, i] = val
        return ret


    def save_kernel(self):
        self.kernel_obj.save_kernel_entry()


if __name__ == '__main__':
    from dataset import DataSet
    data_set = DataSet()

    # make a small subset for testing
    train_set = data_set.train_set
    train_labels = data_set.train_labels
    test_set = data_set.test_set
    test_labels = data_set.test_labels


    test_model = StringSVM("test_k5_lambda0.8", 5, 0.8)

    try:
        test_model.train(train_set, train_labels)
    except Exception as e:
        # re-raise exception
        raise e
    finally:
        # pre_target = test_model.predict(test_set)
        test_model.save_kernel()
    # print(pre_target)
    # print(test_labels)
    # print(test_model.ssk_value_table)
