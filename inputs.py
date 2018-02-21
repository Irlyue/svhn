import numpy as np
from scipy.io import loadmat


class Data:
    def __init__(self):
        self.train = None
        self.cv = None
        self.test = None

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __repr__(self):
        return '\n'.join('{:<8}:X{}\n{:<9}y{}'.format(key, xx.shape, '', yy.shape)
                         for key, (xx, yy) in self.__dict__.items())


def load_one_set(type_='train'):
    path = '/home/wenfeng/datasets/SVHN/%s_32x32.mat' % type_
    data = loadmat(path)
    return data['X'], data['y']


def load_data(n_examples_for_train=None, n_examples_for_cv=10000):
    # if not specified, use all the rest as training examples
    n_examples_for_train = -n_examples_for_cv if n_examples_for_train is None else n_examples_for_train
    data = Data()
    X, y = load_one_set('train')
    X_test, y_test = load_one_set('test')
    y[y == 10] = 0
    y_test[y_test == 10] = 0

    X = np.transpose(X, (3, 0, 1, 2))
    X_test = np.transpose(X_test, (3, 0, 1, 2))

    X_train, X_cv = X[:n_examples_for_train], X[-n_examples_for_cv:]
    y_train, y_cv = y[:n_examples_for_train], y[-n_examples_for_cv:]
    data.train = (X_train, y_train)
    data.cv = (X_cv, y_cv)
    data.test = (X_test, y_test)
    return data


if __name__ == '__main__':
    data = load_data(10)
    print(data)