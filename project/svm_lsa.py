from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer


class SVM:
    """
    LSA Implementation reference:
    http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
    Latent Semantic Kernels:
    https://eprints.soton.ac.uk/259781/1/LatentSemanticKernals_JIIS_18.pdf
    """
    def __init__(self, name):
        self.name = name
        self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                          min_df=2, stop_words='english',
                                          use_idf=True)
        # Project the tf-idf vectors onto the first N principal components.
        # Though this is significantly fewer features than the original tf-idf vector,
        # they are stronger features, and the accuracy is higher.
        svd = TruncatedSVD(100)
        self.lsa = make_pipeline(svd, Normalizer(copy=False))
        self.text_clf = SGDClassifier(alpha=1e-3, random_state=42, max_iter=10)

    def train(self, data_set):
        print('Train model ' + self.name)

        # map train data to a tf-idf vector
        train_tf_idf = self.vectorizer.fit_transform(data_set.train_set)
        # Do lsa
        train_lsa = self.lsa.fit_transform(train_tf_idf)

        self.text_clf.fit(train_lsa, data_set.train_labels)

    def predict(self, data_set):
        test_tf_idf = self.vectorizer.transform(data_set.test_set)
        test_lsa = self.lsa.transform(test_tf_idf)
        return self.text_clf.predict(test_lsa)
