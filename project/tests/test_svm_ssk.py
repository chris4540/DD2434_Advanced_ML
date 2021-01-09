from dataset import DataSet
import ssk_kernel.ssk_kernel_c as ssk_kernel_c
from svm.string_svm import StringSVM
import time

if __name__ == '__main__':

    data_set = DataSet()

    # Initialize SVM
    # n_vals = [1, 2, 3]
    n_vals = [3,4,5]

    precisions = []
    recalls = []

    for n in n_vals:
        m_lambda = 0.9
        kernel = lambda x, y : ssk_kernel_c.ssk_kernel(x, y, n, m_lambda)
        ssvm = StringSVM(kernel)


        # Train SVM
        start = time.time()

        ssvm.recursive_fit(data_set.train_set, data_set.train_labels, 3)

        print("Training elapsed time: {:.2f} seconds".format(time.time() - start))

        # Test
        start = time.time()

        pred = ssvm.predict(data_set.test_set, data_set.test_labels)

        print("Prediction elapsed time: {:.2f} seconds".format(time.time() - start))

        true_pos = 0
        all_pos = 0
        all_pred_pos = 0

        for i in range(len(pred)):
            if test_labels[i] == 1:
                all_pos += 1
            if pred[i] == 1 and test_labels[i] == 1:
                true_pos += 1
            if pred[i] == 1:
                all_pred_pos += 1

        recall = true_pos / all_pos
        precision = true_pos / all_pred_pos

        recalls.append(recall)
        precisions.append(precision)

        print(true_pos, all_pos)

        print("Precision:", precision, ", Recall: ", recall)

    f = open('output_c.txt', 'w')

    f.write('precision:\n')
    f.write(str(precisions))
    f.write("\nrecall:\n")
    f.write(str(recalls))
