from dataset import DataSet
from ssk_kernel_wrap import SSKKernel
from itertools import permutations
from sklearn.feature_extraction.text import CountVectorizer
import ssk_kernel.ssk_kernel_c as ssk_kernel_c
import json

if __name__ == '__main__':

    k = 14
    n_basis = 200
    data_set = DataSet()
    train = data_set.train_set
    test = data_set.test_set


    data = list(train) + test
    print(len(data))
    vec = CountVectorizer(analyzer='char', ngram_range=(k, k))
    bag_of_words = vec.fit_transform(data)
    sum_words = bag_of_words.sum(axis=0)


    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    basis = []

    for (w , fq) in words_freq[:n_basis]:
        basis.append(w)


    json_fname = "./ssk_kernel_data/most_freq_contss_k{}.json".format(k)

    with open(json_fname, "w") as f:
        json.dump(basis, f)
