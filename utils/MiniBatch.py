import numpy as np


class MiniBatch:
    def __init__(self, xs, ys, batch_size):
        self.Xs = xs
        self.ys = ys
        self.batch_size = batch_size
        self.it = 0
        self.L = np.size(ys, axis=0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.it >= self.L:
            self.it = 0
            raise StopIteration
        else:
            if self.it + self.batch_size >= len(self.ys):
                res = self.Xs[self.it: self.L, :], self.ys[self.it: self.L]
            else:
                res = self.Xs[self.it: self.it + self.batch_size, :], self.ys[self.it: self.it + self.batch_size]
            self.it += self.batch_size
            return res


class CyclicMiniBatch:
    def __init__(self, xs, ys, batch_size):
        self.Xs = xs
        self.ys = ys
        self.batch_size = batch_size
        self.it = 0
        self.L = np.size(ys, axis=0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.it >= self.L:
            self.it = 0

        if self.it + self.batch_size >= len(self.ys):
            res = self.Xs[self.it: self.L, :], self.ys[self.it: self.L]
        else:
            res = self.Xs[self.it: self.it + self.batch_size, :], self.ys[self.it: self.it + self.batch_size]
        self.it += self.batch_size
        return res