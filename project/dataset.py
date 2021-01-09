import numpy as np
import random as rd
import data
from sklearn.metrics import f1_score, precision_score, recall_score

class DataSet:
    def __init__(self, labels=('earn', 'acq', 'crude', 'corn')):
        print('Prepare data')
        self._train_set_tot, self._test_set_tot = data.extract_data()
        self._train_set_tot = [_item[:10] for _item in self._train_set_tot]
        self._labels = labels
        self._nb_labels = len(labels)
        self._nb_train = (152, 114, 76, 38)
        self._nb_test = (40, 25, 15, 10)

        self.train_set = []
        self.test_set = []
        self.train_labels = []
        self.test_labels = []
        self._sample_data()

    def _sample_data(self):
        self.train_set = []
        self.test_set = []
        self.train_labels = []
        self.test_labels = []
        tmp_train_set = [[] for _ in range(self._nb_labels)]
        tmp_test_set = [[] for _ in range(self._nb_labels)]

        for x in self._train_set_tot:
            for t in range(len(x[1])):
                for l in range(self._nb_labels):
                    if x[1][t] == self._labels[l]:
                        tmp_train_set[l].append(x[0])

        for x in self._test_set_tot:
            for t in range(len(x[1])):
                for l in range(self._nb_labels):
                    if x[1][t] == self._labels[l]:
                        tmp_test_set[l].append(x[0])

        for l in range(self._nb_labels):
            tmp_train_set[l] = rd.sample(tmp_train_set[l], self._nb_train[l])
            tmp_test_set[l] = rd.sample(tmp_test_set[l], self._nb_test[l])

        for l in range(self._nb_labels):
            self.train_labels += [l] * len(tmp_train_set[l])
            self.test_labels += [l] * len(tmp_test_set[l])

            self.train_set += tmp_train_set[l]
            self.test_set += tmp_test_set[l]

        # todo: here, should we just handle the first 100 pieces? (for ssk)
        tmp = list(zip(self.train_set, self.train_labels))
        rd.shuffle(tmp)
        self.train_set, self.train_labels = zip(*tmp)

    def iterate(self, iterations, *models):
        f1 = [[[] for _ in range(self._nb_labels)] for _ in range(len(models))]
        precision = [[[] for _ in range(self._nb_labels)] for _ in range(len(models))]
        recall = [[[] for _ in range(self._nb_labels)] for _ in range(len(models))]

        for _ in range(iterations):
            for m in range(len(models)):
                models[m].train(self)
                prediction = models[m].predict(self)

                tmp_f1 = f1_score(self.test_labels, prediction, average=None)
                tmp_precision = precision_score(self.test_labels, prediction, average=None)
                tmp_recall = recall_score(self.test_labels, prediction, average=None)
                for l in range(self._nb_labels):
                    f1[m][l].append(tmp_f1[l])
                    precision[m][l].append(tmp_precision[l])
                    recall[m][l].append(tmp_recall[l])
            self._sample_data()

        for m in range(len(models)):
            for l in range(self._nb_labels):
                f1_mean = round(float(np.mean(f1[m][l])), 3)
                f1_std = round(float(np.std(f1[m][l])), 3)
                f1[m][l] = [f1_mean, f1_std]

                precision_mean = round(float(np.mean(precision[m][l])), 3)
                precision_std = round(float(np.std(precision[m][l])), 3)
                precision[m][l] = [precision_mean, precision_std]

                recall_mean = round(float(np.mean(recall[m][l])), 3)
                recall_std = round(float(np.std(recall[m][l])), 3)
                recall[m][l] = [recall_mean, recall_std]

        print('Category+Model\t\tF1(Mean,Std)\t\tPrecision(Mean,Std)\t\tRecall(Mean,Std)')
        for l in range(self._nb_labels):
            for m in range(len(models)):
                print(self._labels[l] + '+' + models[m].name + '\t\t' + str(f1[m][l]) + '\t\t' + str(
                    precision[m][l]) + '\t\t' + str(recall[m][l]))

    def evaluate_prediction(self, prediction):
        tot = len(self.test_labels)
        hit = sum([1 if prediction[i] == self.test_labels[i] else 0 for i in range(tot)])
        print(hit, tot, hit / tot)
