from dataset import DataSet
import json


if __name__ == '__main__':
    data_set = DataSet()
    train = data_set.train_set
    test = data_set.test_set


    data = list(train) + test

    res = dict()
    for i, doc in enumerate(data):
        res[doc] = i

    with open("./ssk_kernel_data/data2idx.json", "w") as f:
        json.dump(res, f, indent=2)
