import pickle
import numpy as np


def load_data():
    with open('mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']


x_train, y_train, x_test, y_test = load_data()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train = x_train / 255.0 * 0.99 + 0.01
x_test = x_test / 255.0 * 0.99 + 0.01

step_size = 1e-1
num = len(y_train)


def relu(Z):
    return np.maximum(0, Z)


def relu_back(dA, Z):
    dA[Z <= 0] = 0
    return dA


def softmax(x):
    row_max = x.max(axis=1, keepdims=True)
    normal_x = x - row_max
    exp_x = np.exp(normal_x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def cross_entropy_loss(output, label, epoch):
    dL = softmax(output)
    if epoch % 10 == 0:
        accuracy = np.sum(dL.argmax(axis=1) == label) / num
        print('epoch: %d, accuracy: %.2f' % (epoch, accuracy * 100))
        with open('train.txt', 'a') as f:
            f.write('%d: %.2f\n' % (epoch, accuracy))
    dL[range(num), label] -= 0.99
    return dL


def forward_prop(x, w, b):
    return x.dot(w) + b


def back_prop(dA, x, w, b, i):
    dw = x.T.dot(dA)
    db = np.sum(dA, axis=0, keepdims=True)
    dx = dA.dot(w[i].T)
    dw /= num
    db /= num
    w[i] -= step_size * dw
    b[i] -= step_size * db
    return dx


def train(data, label, w, b, epochs=400):
    for i in range(epochs):
        h1 = relu(forward_prop(data, w[0], b[0]))
        h2 = relu(forward_prop(h1, w[1], b[1]))
        output = forward_prop(h2, w[2], b[2])
        grad_output = cross_entropy_loss(output, label, i)
        grad_h2 = relu_back(grad_output, output)
        grad_h2 = back_prop(grad_h2, h2, w, b, 2)
        grad_h1 = relu_back(grad_h2, h2)
        grad_h1 = back_prop(grad_h1, h1, w, b, 1)
        back_prop(grad_h1, data, w, b, 0)
    return w, b


def test(data, label, w, b):
    num_test = len(label)
    h1 = relu(forward_prop(data, w[0], b[0]))
    h2 = relu(forward_prop(h1, w[1], b[1]))
    output = forward_prop(h2, w[2], b[2])
    output = softmax(output)
    accuracy = np.sum(output.argmax(axis=1) == label) / num_test
    print('Test accuracy: %.2f' % (accuracy * 100))
    with open('test.txt', 'a') as f:
        f.write(str(accuracy)+'\n')


def fcnn(train_data, train_label, test_data, test_label):
    factor = 1e-1
    w = [
        np.random.randn(784, 200) * factor,
        np.random.randn(200, 50) * factor,
        np.random.randn(50, 10) * factor
    ]
    b = [
        np.random.randn(1, 200) * factor,
        np.random.randn(1, 50) * factor,
        np.random.randn(1, 10) * factor
    ]
    w, b = train(train_data, train_label, w, b)
    test(test_data, test_label, w, b)


fcnn(x_train, y_train, x_test, y_test)
