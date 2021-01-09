import numpy as np
from sklearn.svm import SVC
from ssk_kernel.apprx_kernel import SSKKernelApprox
from sklearn.metrics import precision_recall_fscore_support
import functools
print = functools.partial(print, flush=True)

class StringSVM:

    max_iter = 50
    def __init__(self, name, k, decay):
        self.name = name
        self.kernel_obj = SSKKernelApprox(k, decay, save_entry=True)

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
        train_label = np.array(train_label)
        for _ in range(self.max_iter):
            if len(X_idx) < self.n_chunk:
                break

            # sample chuck to have at least 2 classes
            for _ in range(1000):
                samples = np.random.choice(X_idx, self.n_chunk)
                if len(set(train_label[samples])) == 1:
                    # only one class was picked. resample
                    continue
                else:
                    break

            x = np.array(samples).reshape(-1, 1)
            if len(support_vec) > 0:
                x = np.vstack((x, support_vec))
            y = train_label[x.flatten()]
            print("Training iter:", iter_)
            self.text_clf.fit(x, y)

            support_vec = x[self.text_clf.support_]
            # remove trained samples from x_idx
            X_idx = np.setdiff1d(X_idx, support_vec.flatten())
            iter_ += 1
            print("# of support vectors for each class", self.text_clf.n_support_)
            print("# of support vectors", np.sum(self.text_clf.n_support_))
            print("# of remaining samples", len(X_idx))


        # finalize the svm with only those support vectors
        print("=========================")
        print("Finalizing the SVM....")
        print("=========================")
        y = train_label[support_vec.flatten()]
        self.text_clf.fit(support_vec, y)
        print("# of support vectors for each class", self.text_clf.n_support_)
        print("# of support vectors", np.sum(self.text_clf.n_support_))

    def predict(self, test_data):
        for i, doc in enumerate(test_data):
            self.documents[i+self.n_data] = doc

        size = len(test_data)
        X_new_idx = np.arange(size) + self.n_data
        self.n_data += len(test_data)

        # predict by chunk
        ret = list()
        cnt = 0
        for x in np.array_split(X_new_idx,(size // self.n_chunk)):
            sub_pred = self.text_clf.predict(x.reshape(-1, 1))
            ret.extend(sub_pred.tolist())
            cnt += 1
            print("# of iteration for prediction: ", cnt)
            self.save_kernel()

        return ret

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
        self.kernel_obj.save_kernel_to_json()


if __name__ == '__main__':
    from dataset import DataSet
    import time
    data_set = DataSet()

    # make a small subset for testing
    train_set = data_set.train_set
    train_labels = data_set.train_labels
    test_set = data_set.test_set
    test_labels = data_set.test_labels


    test_model = StringSVM("test_k5_lambda0.9", 5, 0.9)

    try:
        st = time.time()
        test_model.train(train_set, train_labels)
        et = time.time()
        print("Training time {} sec".format(et - st))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # re-raise exception
        raise e
    finally:
        test_model.save_kernel()


    # do prediction
    try:
        st = time.time()
        class_pred = test_model.predict(test_set)
        et = time.time()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # re-raise exception
        raise e
    finally:
        test_model.save_kernel()
    print("Prediction time {} sec".format(et - st))
    print(class_pred)
    print(test_labels)

    scores = precision_recall_fscore_support(test_labels, class_pred, average=None)
    precision = scores[0]
    recall = scores[1]
    f1_score = scores[2]
    support = scores[3]
    print("precision:", precision)
    print("recall:", recall)
    print("f1_score:", f1_score)
