from dataset import DataSet


if __name__ == '__main__':

    data_set = DataSet()
    train = data_set.train_set
    test = data_set.test_set


    data = set(list(train) + test)
    print(len(data))


    data_set2 = DataSet()
    data2 = set(list(data_set2.train_set) + data_set2.test_set)


    print(data == data2)
    print(len(data_set._train_set_tot))
