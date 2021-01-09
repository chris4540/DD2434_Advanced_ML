import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

import pandas as pd


def getStatsDS(model,test_data, categories):


    true_labels = np.asarray([s[1] for s in test_data])
    preds = model.scoring_function(np.asarray([s[0] for s in test_data],dtype=np.float32))

    stats = pd.DataFrame()

    stats['Categories'] = categories
    stats['F1'] = f1_score(true_labels, preds[0], average=None)
    stats['Precision'] = precision_score(true_labels, preds[0], average=None)
    stats['Recall'] = recall_score(true_labels, preds[0], average=None)

    return stats
