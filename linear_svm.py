import math
import random

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

data_file = 'Data_Linear_SVM.csv'


def generate_data(w, border, wall, size):
    wall_ = wall / (math.sqrt(w[0] ** 2 + w[1] ** 2) * 2)
    with open(data_file, 'w+') as data_set:
        for i in range(size):
            x = random.random() * border
            y = random.random() * border
            z = w[0] * x + w[1] * y + w[2] * 1
            if z > wall_:
                s = 1
            elif z < -wall_:
                s = -1
            else:
                continue
            data_set.write(str(x) + ',' + str(y) + ',' + '1' + ',' + str(s) + '\n')


def load_data(test_ratio):
    with open(data_file, 'r') as data_set:
        lines = data_set.readlines()
        data_size = len(lines)
        test_size = int(data_size * test_ratio)
        test_index = random.sample(range(data_size), test_size)
        train_array = [[float(c) for c in lines[i].strip().split(',')] for i in range(data_size) if i not in test_index]
        test_array = [[float(c) for c in lines[i].strip().split(',')] for i in range(data_size) if i in test_index]
        train_mat = np.array(train_array)
        test_mat = np.array(test_array)
        return train_mat, test_mat


def my_svm(y, c, X):
    le = y.shape[1]
    K = np.dot(X, np.transpose(X))
    KY = np.dot(np.transpose(y), y)
    H = np.multiply(K, np.transpose(KY))
    P = matrix(H, tc='d')
    q = matrix(-1, (le, 1), tc='d')
    A = matrix(y, tc='d')
    b = matrix(0, tc='d')
    h_ = np.concatenate((np.ones((le, 1)) * c, np.zeros((le, 1))), axis=0)
    h = matrix(h_, tc='d')
    G_ = np.concatenate((np.diag([1 for i in range(le)]), np.diag([-1 for i in range(le)])), 0)
    G = matrix(G_, tc='d')
    sol = solvers.qp(P, q, G, h, A, b)
    print(sol['x'], 'primal objective', sol['primal objective'])
    alpha = sol['x']
    obj = sol['primal objective']
    threshold = 10 ** -3
    alpha_data = np.asarray(alpha)
    pos = np.where(alpha_data > threshold)[0]
    p = pos[0]
    b_head = (y[:, p] - np.sum(np.multiply(np.transpose(alpha), y) * (X * np.transpose(X[p]))))[0, 0]
    return alpha, b_head, pos, obj


def predict(train_data, test_data):
    c = 10
    X = train_data[:, :-2]
    y = np.matrix(train_data[:, -1])
    alpha, b_head, pos, obj = my_svm(y, c, X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    train_scatter1 = None
    train_scatter2 = None
    for xs, ys, zs, ts in train_data:
        if ts == 1:
            c = 'r'
            m = 'o'
            train_scatter1 = ax.scatter(xs, ys, c=c, marker=m)
        else:
            c = 'b'
            m = '^'
            train_scatter2 = ax.scatter(xs, ys, c=c, marker=m)
    test_scatter1 = None
    test_scatter2 = None
    for xs, ys, zs, ts in test_data:
        if ts == 1:
            c = 'r'
            m = 'x'
            test_scatter1 = ax.scatter(xs, ys, c=c, marker=m)
        else:
            c = 'b'
            m = 'x'
            test_scatter2 = ax.scatter(xs, ys, c=c, marker=m)

    wrong_data = []
    for i in range(test_data.shape[0]):
        if np.sum(np.multiply(np.transpose(alpha), y) * (X * np.transpose(test_data[i, :-2]))) + b_head > 0:
            r = 1
        else:
            r = -1
        if test_data[i, -1] != r:
            test_data[-1] = r
            wrong_data.append(test_data[i, :])

    prediction_acc = 1 - len(wrong_data) / test_data.shape[0]
    plt.annotate('Classification Accuracy: ' + str(prediction_acc), xy=(1, 1), xytext=(-0.5, -0.5))
    # plt.annotate('Weight Vector: ' + str(), xy=(1, 1), xytext=(-0.5, -0.2))
    wrong_scatter1 = None
    wrong_scatter2 = None
    for xs, ys, zs, ts in wrong_data:
        if ts == 1:
            c = 'r'
            m = 's'
            wrong_scatter1 = ax.scatter(xs, ys, c=c, marker=m)
        else:
            c = 'b'
            m = 's'
            wrong_scatter2 = ax.scatter(xs, ys, c=c, marker=m)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    x = np.arange(0, 6, 0.5)
    area_scatter1 = None
    area_scatter2 = None
    for xs in x:
        for ys in x:
            if np.sum(np.multiply(np.transpose(alpha), y) * (X * np.transpose([xs, ys]))) + b_head > 0:
                ts_predict = 1
            else:
                ts_predict = -1
            if ts_predict == 1:
                c = 'r'
                m = '*'
                area_scatter1 = ax.scatter(xs, ys, c=c, marker=m)
            else:
                c = 'b'
                m = '.'
                area_scatter2 = ax.scatter(xs, ys, c=c, marker=m)
    if wrong_scatter1 is not None or wrong_scatter2 is not None:
        ax.legend([train_scatter1, train_scatter2, test_scatter1, test_scatter2, wrong_scatter1, wrong_scatter2,
                   area_scatter1, area_scatter2], \
                  ['train class 1', 'train class 2', 'test class 1', 'test class 2', 'wrong prediction class 1',
                   'wrong prediction class 2', 'area class 1', 'area class 2'])
    elif test_scatter1 is not None or test_scatter2 is not None:
        ax.legend([train_scatter1, train_scatter2, test_scatter1, test_scatter2, area_scatter1, area_scatter2], \
                  ['train class 1', 'train class 2', 'test class 1', 'test class 2', 'area class 1', 'area class 2'])
    else:
        ax.legend([train_scatter1, train_scatter2, area_scatter1, area_scatter2], \
                  ['train class 1', 'train class 2', 'area class 1', 'area class 2'])
    plt.show()


generate_data([0.5, -1, 2], 8, 1, 50)
# generate_data([1, -1, 1], 8, 1, 50)
# generate_data([0.5, 1, -4], 5, 2, 50)

train_data, test_data = load_data(0.2)

predict(train_data, test_data)
