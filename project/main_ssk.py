import svm_ssk
import dataset
from dataset import DataSet

if __name__ == '__main__':
    SVM_SSK = svm_ssk.SVM('SVM_SSK_K5', ngram_size=5)
    data_set = DataSet()
    data_set.iterate(10, SVM_SSK)
