import svm_lsa
import svm_ngk
# import svm_ssk
import svm_ssk_approx
import svm_wk
import dataset
from dataset import DataSet

if __name__ == '__main__':
    # SVM_SSK = svm_ssk_approx.SVM('SVM_SSK', ngram_size=5)
    data_set = DataSet()
    # data_set.iterate(1, SVM_SSK)

    # SVM_3G = svm_ngk.SVM('SVM_3G', ngram_size=3)
    # SVM_4G = svm_ngk.SVM('SVM_4G', ngram_size=4)
    # SVM_5G = svm_ngk.SVM('SVM_5G', ngram_size=5)
    # SVM_6G = svm_ngk.SVM('SVM_6G', ngram_size=6)
    SVM_9G = svm_ngk.SVM('SVM_9G', ngram_size=9)
    SVM_12G = svm_ngk.SVM('SVM_12G', ngram_size=12)
    # SVM_WK = svm_wk.SVM('SVM_WK')
    # SVM_LSA = svm_lsa.SVM('SVM_LSA')
    #
    # data_set = DataSet()
    data_set.iterate(10, SVM_9G, SVM_12G)

''' example output
Category+Model      F1(Mean,Std)        Precision(Mean,Std)     Recall(Mean,Std)
earn+SVM_3G     [0.924, 0.018]      [0.905, 0.025]      [0.945, 0.029]
earn+SVM_4G     [0.927, 0.024]      [0.901, 0.024]      [0.955, 0.033]
earn+SVM_5G     [0.928, 0.021]      [0.904, 0.025]      [0.955, 0.033]
earn+SVM_6G     [0.927, 0.024]      [0.901, 0.024]      [0.955, 0.037]
earn+SVM_7G     [0.928, 0.02]       [0.906, 0.03]       [0.953, 0.034]
earn+SVM_9G     [0.921, 0.034]      [0.895, 0.055]      [0.95, 0.034]
earn+SVM_12G    [0.906, 0.04]       [0.871, 0.059]      [0.948, 0.036]
earn+SVM_WK     [0.925, 0.023]      [0.897, 0.028]      [0.955, 0.029]
earn+SVM_LSA    [0.931, 0.015]      [0.903, 0.03]       [0.962, 0.017]
acq+SVM_3G      [0.892, 0.032]      [0.894, 0.039]      [0.892, 0.047]
acq+SVM_4G      [0.899, 0.031]      [0.911, 0.036]      [0.888, 0.047]
acq+SVM_5G      [0.895, 0.031]      [0.904, 0.033]      [0.888, 0.053]
acq+SVM_6G      [0.895, 0.041]      [0.9, 0.043]        [0.892, 0.054]
acq+SVM_7G      [0.904, 0.04]       [0.905, 0.045]      [0.904, 0.051]
acq+SVM_9G      [0.855, 0.054]      [0.859, 0.06]       [0.856, 0.072]
acq+SVM_12G     [0.825, 0.076]      [0.818, 0.072]      [0.836, 0.099]
acq+SVM_WK      [0.9, 0.036]        [0.915, 0.038]      [0.888, 0.061]
acq+SVM_LSA     [0.904, 0.038]      [0.937, 0.04]       [0.876, 0.058]
crude+SVM_3G        [0.84, 0.061]       [0.887, 0.053]      [0.807, 0.105]
crude+SVM_4G        [0.847, 0.076]      [0.894, 0.074]      [0.813, 0.115]
crude+SVM_5G        [0.837, 0.072]      [0.884, 0.066]      [0.8, 0.103]
crude+SVM_6G        [0.843, 0.05]       [0.898, 0.045]      [0.8, 0.089]
crude+SVM_7G        [0.845, 0.053]      [0.889, 0.055]      [0.813, 0.102]
crude+SVM_9G        [0.817, 0.059]      [0.857, 0.06]       [0.793, 0.113]
crude+SVM_12G       [0.803, 0.07]       [0.884, 0.082]      [0.747, 0.111]
crude+SVM_WK        [0.866, 0.061]      [0.916, 0.068]      [0.827, 0.09]
crude+SVM_LSA       [0.846, 0.07]       [0.867, 0.076]      [0.833, 0.1]
corn+SVM_3G     [0.88, 0.073]       [0.905, 0.06]       [0.86, 0.102]
corn+SVM_4G     [0.88, 0.073]       [0.905, 0.06]       [0.86, 0.102]
corn+SVM_5G     [0.88, 0.073]       [0.905, 0.06]       [0.86, 0.102]
corn+SVM_6G     [0.865, 0.062]      [0.893, 0.05]       [0.84, 0.08]
corn+SVM_7G     [0.845, 0.076]      [0.889, 0.051]      [0.81, 0.104]
corn+SVM_9G     [0.861, 0.113]      [0.95, 0.064]       [0.8, 0.161]
corn+SVM_12G    [0.811, 0.103]      [0.924, 0.09]       [0.73, 0.127]
corn+SVM_WK     [0.88, 0.079]       [0.906, 0.074]      [0.86, 0.102]
corn+SVM_LSA    [0.874, 0.074]      [0.908, 0.072]      [0.85, 0.112]
'''
