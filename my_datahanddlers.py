#! /usr/bin/python3

#from common_header import *
#colocar as bibliotecas necessárias
import numpy as np

#class slidding_window:
#   def __init__(self, dataset, batch_size, stride=0, labels=None, diff_mode=False):
#       assert batch_size > 0
#       self.dataset = dataset
#       self.length = -1 #ignore header
#       with open(dataset, 'r') as dt_file:
#           for _ in dt_file:
#               self.length += 1
#       self.batch_size = batch_size
#       self.stride = stride if stride > 0 else batch_size
#       self.cur_idx = 1 #ignore header
#       self.labels = labels
#       self.diff_mode = diff_mode

#   def __iter__(self):
#       return self

#   def __next__(self):
#       i = self.cur_idx
#       self.cur_idx += self.stride
#       if i+self.batch_size < self.length:
#           if self.diff_mode and i != 1:
#               return pd.read_csv(self.dataset, names=self.labels, skiprows=i-1, nrows=self.batch_size+1)
#           return pd.read_csv(self.dataset, names=self.labels, skiprows=i, nrows=self.batch_size)
#       raise StopIteration

#interval mapping
class map_to:
    def __init__(self, t_min, t_max):
        if t_max < t_min:
            t_max,t_min = t_min, t_max
        self.range = t_max - t_min
        self.t = t_min

    def __call__(self, x, x_max, x_min):
        if x_max < x_min:
            x_max,x_min = x_min,x_max
        return (x - x_min)*self.range/(x_max - x_min) + self.t

    def get_par(self):
        return self.t, self.range + self.t

class map_from(map_to):
    def __call__(self, t, x_max, x_min):
        if x_max < x_min:
            x_max,x_min = x_min,x_max
        return (t - self.t)*(x_max - x_min)/self.range + x_min

def normalization(X):
    M = X.max()
    m = X.min()
    return (X - m)/(M - m), M, m

def undo_normalization(X, M, m):
    return X*(M - m) + m


def standardization(X):
    m = np.mean(X)
    var = sum(map(lambda x: abs(x - m)**2, X))/(X.size if isinstance(X, np.ndarray) else X.size[0])
    s = np.sqrt(var)
    Y = list(map(lambda x: (x-m)/s, X))
    return Y, m, s

def undo_standardization(X, m, s):
    return list(map(lambda x: x*s+m, X))

def delta_encoding(X):
    X_0 = X[0]
    remove = X_0
    for i in range(1, X.size) if isinstance(X, np.ndarray) else range(1,X.size()[0]):
        last = X[i]
        X[i] -= remove
        remove = last
    X[0] -= X_0
    return X, X_0

def delta_decoding(X,X_0):
    X[0] += X_0
    for i in range(1,X.size) if isinstance(X, np.ndarray) else range(1,X.size()[0]):
        X[i] += X[i-1]
    return X

def pre_process(X):
    X,X_0 = delta_encoding(X)
#   norm = np.linalg.norm(X)
#   return X/norm, X_0, norm
    X,max,min = normalization(X)
    return X, X_0, max, min

def pos_process(X, X_0, max, min):
    X *= norm
    X = undo_normalization(X, max, min)
    return delta_decoding(X, X_0)

if __name__ == '__main__':
    import numpy as np

    a = np.random.uniform(-5, 10, size=20)
    print(a)
    print(pos_process(*pre_process(a)))
