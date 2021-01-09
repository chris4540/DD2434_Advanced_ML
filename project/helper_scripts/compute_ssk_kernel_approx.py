from dataset import DataSet
from ssk_kernel_approx_wrap import SSKKernelApprox
from itertools import permutations

if __name__ == '__main__':
    k = 5
    decay =  0.5
    kernel = SSKKernelApprox(k, decay)

    data_set = DataSet()
    train = data_set.train_set
    test = data_set.test_set


    data = list(train) + test
    print(len(data))
    cnt = 0
    try:
        for s, t in permutations(data, 2):
            if s == t:
                continue
            ret = kernel(s, t)
            print(ret)
            cnt += 1

            if cnt > 100:
                cnt = 0
                kernel.save_kernel_entry()
    except KeyboardInterrupt:
        pass
    finally:
        kernel.save_kernel_entry()

