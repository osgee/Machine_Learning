import random
import matplotlib.pyplot as plt
import numpy as np

data_file = 'Data_PLA.csv'
Max_Iteration = 1000


def generate_data(w, border, size):
    with open(data_file, 'w+') as data_set:
        for i in range(size):
            x = random.random() * border
            y = random.random() * border
            z = w[0] * x + w[1] * y + w[2] * 1
            if z > 0:
                s = 1
            else:
                s = -1
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


def update(w, train_vector):
    if np.dot(train_vector[:-1], w) * train_vector[-1] > 0:
        return w, False
    else:
        return w + train_vector[-1] * np.transpose(train_vector[:-1]), True


def train(w, train_data):
    iteration = Max_Iteration
    for i in range(iteration):
        updated = False
        for t in range(train_data.shape[0]):
            w, up_out = update(w, train_data[t])
            updated = updated or up_out
        if not updated:
            break
    return w


def predict(w, train_data, test_data):
    w = train(w, train_data)
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
        if np.dot(test_data[i, :-1], w) > 0:
            r = 1
        else:
            r = -1
        if test_data[i, -1] != r:
            test_data[-1] = r
            wrong_data.append(test_data[i, :])

    prediction_acc = 1 - len(wrong_data) / test_data.shape[0]
    plt.annotate('Classification Accuracy: ' + str(prediction_acc), xy=(1, 1), xytext=(-0.5, -0.5))
    plt.annotate('Weight Vector: ' + str(w), xy=(1, 1), xytext=(-0.5, -0.2))
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
    x = np.arange(0, 5, 0.1)
    y = (-w[2] - w[0] * x) / w[1]
    line_clf, = ax.plot(x, y, label='CLF')
    if wrong_scatter1 is not None or wrong_scatter2 is not None:
        ax.legend([train_scatter1, train_scatter2, test_scatter1, test_scatter2, wrong_scatter1, wrong_scatter2, line_clf], \
                  ['train class 1', 'train class 2', 'test class 1', 'test class 2', 'wrong prediction class 1',
                   'wrong prediction class 2', 'CLF line'])
    elif test_scatter1 is not None or test_scatter2 is not None:
        ax.legend([train_scatter1, train_scatter2, test_scatter1, test_scatter2, line_clf], \
                  ['train class 1', 'train class 2', 'test class 1', 'test class 2', 'CLF line'])
    else:
        ax.legend([train_scatter1, train_scatter2, line_clf], \
                  ['train class 1', 'train class 2', 'CLF line'])
    plt.show()


# generate_data([1, -1, 1], 5, 50)
# generate_data([0.5, 1, -4], 5, 50)
generate_data([0.5, -1, 2], 5, 50)

train_data, test_data = load_data(0.1)
w = np.array([1, 1, 1])
predict(w, train_data, test_data)
