from dataset import DataSet
from ssk_kernel_wrap import SSKKernel
from itertools import permutations

if __name__ == '__main__':
    k = 5
    decay =  0.01
    kernel = SSKKernel(k, decay)

    data_set = DataSet()
    train = data_set.train_set
    test = data_set.test_set


    data = list(train) + test
    print(len(data))

    try:
        for s, t in permutations(data, 2):
            if s == t:
                continue
            ret = kernel(s, t)
            print(ret)
    except KeyboardInterrupt:
        pass
    finally:
        kernel.save_kernel_entry()

