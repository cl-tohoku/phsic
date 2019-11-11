import sys

import numpy as np
from numpy.core.umath_tests import inner1d

import phsic.kernel


class PHSIC():

    def __init__(self):
        self.xp = np

    def fit_XY(self, X, Y):
        """
        params
        X: sequence of length n consisting of xs; [x_i]
            ... observed data on X (d1-dim feature vectors)
        Y: sequence of length n consisting of ys; [y_i]
            ... observed data on Y (d2-dim feature vectors)

        saves
        X     : n x d1 matrix
        Y     : n x d2 matrix
        x_mean: d1-dim array
        y_mean: d2-dim array
        CXY   : d1 x d2 cross covariance matrix between X and Y
        """
        print('phsic.fit_XY()...', file=sys.stderr)
        n = len(X)
        m = len(Y)
        assert n == m

        self.X = np.array(X)
        self.Y = np.array(Y)
        XTY = np.dot(self.X.T, self.Y)
        self.x_mean = np.sum(self.X, axis=0) / n  # \bar{x}; shape (d,)
        self.y_mean = np.sum(self.Y, axis=0) / n  # \bar{y}; shape (d,)
        self.CXY = XTY / n - np.outer(self.x_mean, self.y_mean)

    def predict(self, x, y):
        """
        returns float PHSIC(x,y; X, Y)
        """
        return np.dot(x - self.x_mean, self.CXY @ (y - self.y_mean))

    def predict_batch_XY(self, X, Y):
        """
        params
        X: sequence of length n consisting of xs; [x_i]
            ... observed data on X
        Y: sequence of length n consisting of ys; [y_i]
            ... observed data on Y

        returns
        """
        print('phsic.predict_batch_XY()...', file=sys.stderr)
        # return np.dot(X - self.x_mean, self.CXY @ (X - self.y_mean))
        return (np.matmul((X - self.x_mean), self.CXY) * (Y - self.y_mean)).sum(-1)

    def predict_batch_training_data(self):
        print('phsic.predict_batch_training_data()...', file=sys.stderr)
        X = self.X
        Y = self.Y
        return (np.matmul((X - self.x_mean), self.CXY) * (Y - self.y_mean)).sum(-1)


class PHSIC_ICD():

    def __init__(self, no_centering=False):
        self.xp = np
        self.no_centering = no_centering

    def fit(self, Z, k, l, d1, d2):
        """
        Z: sequence of length n consisting of (x,y) pairs; [(x_i, y_i)]
            ... observed data on X times Y

        k: function k(x_i, x_j) returns a real number
            ... positive definite kernel on X
        l: function k(y_i, y_j) returns a real number
            ... positive definite kernel on Y

        d: dimension of ICD on both X and Y side

        saves
        A     : n x d matrix
        B     : n x d matrix
        a_mean: d-dim array
        b_mean: d-dim array
        C_ICD : d x d matrix
        """
        print('phsic.fit()...', file=sys.stderr)

        n = len(Z)
        X, Y = tuple(zip(*Z))
        self.fit_XY(X, Y, k, l, d1, d2)

    def fit_XY(self, X, Y, k, l, d1, d2, k_batch=None, l_batch=None):
        """
        params
        X: sequence of length n consisting of xs; [x_i]
            ... observed data on X
        Y: sequence of length n consisting of ys; [y_i]
            ... observed data on Y

        k: function k(x_i, x_j) that returns a real number
            ... positive definite kernel on X
        l: function l(y_i, y_j) that returns a real number
            ... positive definite kernel on Y

        d1: dimension of ICD on X side
        d2: dimension of ICD on Y side

        saves
        A     : n x d1 matrix
        B     : n x d2 matrix
        a_mean: d1-dim array
        b_mean: d2-dim array
        C_ICD : d1 x d2 matrix
        """
        print('phsic.fit_XY()...', file=sys.stderr)

        n = len(X)
        m = len(Y)
        assert n == m

        # learning
        print('  incomplete Cholesky decomposition on X side...',
              file=sys.stderr)
        self.A, self.pivot_xids, self.pivot_xs = phsic.kernel.icd_kernel(
            X, k, d1, k_batch=k_batch)
        print('  incomplete Cholesky decomposition on Y side...',
              file=sys.stderr)
        self.B, self.pivot_yids, self.pivot_ys = phsic.kernel.icd_kernel(
            Y, l, d2, k_batch=l_batch)

        self.a_mean = np.array(self.A.T @ np.ones(shape=(n))) / n
        self.b_mean = np.array(self.B.T @ np.ones(shape=(n))) / n
        # todo: np.ones 2回作らない
        # todo: このnp.array()必要?
        self.C_ICD = np.array(phsic.kernel.Hprod(self.A).T @ self.B) / n
        # no_centering
        if self.no_centering:
            self.C_ICD_NC = (self.A.T @ self.B) / n

        self.k = k
        self.l = l

    def predict(self, x, y):
        """
        returns float
            ... PHSIC(x, y; Z, k, l)
        """
        a = phsic.kernel.icd_for_new_data(
            self.A, self.pivot_xids, self.pivot_xs, self.k, x)
        b = phsic.kernel.icd_for_new_data(
            self.B, self.pivot_yids, self.pivot_ys, self.l, y)

        if self.no_centering:
            return a @ (self.C_ICD_NC @ b)
        else:
            return (a - self.a_mean) @ (self.C_ICD @ (b - self.b_mean))

    def predict_batch_XY(self, X, Y):
        """
        params
        X: sequence of length n consisting of xs; [x_i]
            ... observed data on X
        Y: sequence of length n consisting of ys; [y_i]
            ... observed data on Y

        returns
        """

        print('phsic.predict_batch_XY()...', file=sys.stderr)

        A_new = np.array([phsic.kernel.icd_for_new_data(self.A, self.pivot_xids,
                                                        self.pivot_xs, self.k,
                                                        x) for x in X])
        B_new = np.array([phsic.kernel.icd_for_new_data(self.B, self.pivot_yids,
                                                        self.pivot_ys, self.l,
                                                        y) for y in Y])

        if self.no_centering:
            return inner1d(A_new, (self.C_ICD_NC @ B_new.T).T)
        else:
            return inner1d(A_new - self.a_mean,
                           (self.C_ICD @ (B_new - self.b_mean).T).T)
        # return np.array([self.predict(x, y) for x, y in zip(X, Y)])

    def predict_batch_training_data(self):
        print('phsic.predict_batch_training_data()...', file=sys.stderr)

        A_new = self.A
        B_new = self.B

        if self.no_centering:
            return inner1d(A_new, (self.C_ICD_NC @ B_new.T).T)
        else:
            return inner1d(A_new - self.a_mean,
                           (self.C_ICD @ (B_new - self.b_mean).T).T)
