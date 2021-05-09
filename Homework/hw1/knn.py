import numpy as np
from download_mnist import load
import time
import heapq
from collections import Counter

# classify using kNN
x_train_0, y_train, x_test_0, y_test = load()
x_train_0 = x_train_0.reshape(60000,28,28)
x_test_0 = x_test_0.reshape(10000,28,28)
x_train_0 = x_train_0.astype(float)
x_test_0 = x_test_0.astype(float)

# turn x into 1-D array
x_train = []
x_test = []
for x_pic in x_train_0:
    tmp = []
    for x in x_pic:
        tmp = np.append(tmp, x)
    x_train.append(tmp)

for x_pic in x_test_0:
    tmp = []
    for x in x_pic:
        tmp = np.append(tmp, x)
    x_test.append(tmp)

x_train = np.array(x_train)
x_test = np.array(x_test)


def kNNClassify(new_input, data_set, labels, k):
    net = []

    for test_in in new_input:

        # record the k nearest neighbour for each new point
        clfr_dist = []
        clfr_label = []
        for train_en in data_set:
            clfr_dist.append(np.linalg.norm(train_en - test_in))
        k_min_index = map(clfr_dist.index, heapq.nsmallest(k, clfr_dist))

        # find the labels of those distance refer to
        for idx in k_min_index:
            clfr_label.append(labels[idx])

        # decide which class the input data should be
        label_counts = Counter(clfr_label)
        test_class = label_counts.most_common(1)
        net.append(test_class[0][0])
        print(test_class[0][0])
        print(clfr_label)

    return net


start_time = time.time()
output_labels = kNNClassify(x_test[500:550], x_train, y_train, 10)
result = y_test[500:550] - output_labels
result = (1 - np.count_nonzero(result)/len(output_labels))
print("---classification accuracy for knn on mnist: %s ---" % result)
print("---execution time: %s seconds ---" % (time.time() - start_time))
