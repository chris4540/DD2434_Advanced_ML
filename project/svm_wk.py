from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class SVM:
    def __init__(self, name):
        self.name = name
        self.text_clf = Pipeline([('v', CountVectorizer()),
                                  ('t', TfidfTransformer()),
                                  ('c', SGDClassifier(alpha=1e-3, random_state=42, max_iter=10))])

    def train(self, data_set):
        print('Train model ' + self.name)
        self.text_clf.fit(data_set.train_set, data_set.train_labels)

    def predict(self, data_set):
        return self.text_clf.predict(data_set.test_set)
