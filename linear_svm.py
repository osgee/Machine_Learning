from sklearn import svm


data_file = 'Data_Linear_SVM.csv'

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


model = svm.svc(kernel='linear', c=1, gamma=1)
model.fit(X, y)
model.score(X, y)
predicted = model.predicted(x_test)
