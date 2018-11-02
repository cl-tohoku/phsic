import functools
import sys

import numpy as np
import scipy.spatial
import sklearn.metrics
from statsmodels.nonparametric import bandwidths

"""
- KERNEL_CONFIG:
    - ['Linear']
    - ['Cos']
    - ['ReluCos']
    - ['Gaussian', sigma]
    - ['Laplacian', gamma]
"""


# -----------------------------------------
# positive definite kernels (on R^n)
# -----------------------------------------

def positive_definite_kernel(kernel_config, data=None):
    """
    return kernel, kernel_batch

    kernel_batch:
    cf. sklearn.metrics - Pairwise metrics, Affinities and Kernels
        <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise>
        <http://scikit-learn.org/stable/modules/metrics.html>
    cf. scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>
    """
    kernel_type = kernel_config[0]

    if kernel_type == 'Linear':
        return inner_product, None
    elif kernel_type == 'Cos':
        return cosine, None
    elif kernel_type == 'ReluCos':
        return relu_cosine, None
    elif kernel_type == 'Gaussian':
        bw_param = kernel_config[1]
        if bw_param == 'scott':
            if len(kernel_config) > 2:
                scale = float(kernel_config[2])
                sigma_vec = bandwidths.bw_scott(data) * scale
            else:
                sigma_vec = bandwidths.bw_scott(data)
            print('bandwidth: {}'.format(np.average(sigma_vec)))
            return gaussian(sigma_vec), gaussian_pairwise(sigma_vec)

        elif bw_param == 'silverman':
            sigma_vec = bandwidths.bw_silverman(data)
            print('bandwidth: {}'.format(np.average(sigma_vec)))
            return gaussian(sigma_vec), gaussian_pairwise(sigma_vec)

        else:
            sigma = float(kernel_config[1])
            gamma = 1.0 / (2.0 * sigma ** 2)
            return gaussian(sigma), functools.partial(
                sklearn.metrics.pairwise.rbf_kernel, gamma=gamma)
    elif kernel_type == 'Laplacian':
        gamma = float(kernel_config[1])
        return laplacian(gamma), functools.partial(
            sklearn.metrics.pairwise.laplacian_kernel, gamma=gamma)


inner_product = lambda v1, v2: np.dot(v1, v2)

cosine = lambda v1, v2: 1.0 - scipy.spatial.distance.cosine(v1, v2)
assert cosine(np.array([1, 1]), np.array([1, 1])) - 1.0 < 0.0001
assert cosine(np.array([1, 1]), np.array([1, -1])) - 0.0 < 0.0001

cosine_plus_one = lambda v1, v2: cosine(v1, v2) + 1
relu_cosine = lambda v1, v2: max(cosine(v1, v2), 0)


def gaussian(sigma):
    def _gaussian(v1, v2):
        if np.array_equal(v1, v2):
            return 1.0
        else:
            return np.exp(- (scipy.spatial.distance.sqeuclidean(v1 / sigma,
                                                                v2 / sigma) / 2))

    return _gaussian


def gaussian_pairwise(sigma_vec):
    def _gaussian_pairwise(X, Y):
        K = sklearn.metrics.pairwise.euclidean_distances(
            X / sigma_vec, Y / sigma_vec, squared=True)
        K *= -0.5
        np.exp(K, K)  # exponentiate K in-place
        return K

    return _gaussian_pairwise


def laplacian(gamma):
    def _laplacian(v1, v2):
        if np.array_equal(v1, v2):
            return 1.0
        else:
            return np.exp(- gamma * np.linalg.norm(v1 - v2, ord=1))

    return _laplacian


# -----------------------------------------
# handle Gram matrices
# -----------------------------------------

def Hprod(A):
    # A: n x k numpy array
    # H: n x n numpy array
    # return H @ A (n x k numpy array)
    n = A.shape[0]

    OneTA = np.sum(A, axis=0)  # $1^T A$
    return A - ((1 / n) * OneTA)  # 引き算は各行 (broadcast) A - 1/n 1 1^T A


# -----------------------------------------
# incomplete Cholesky decomposition with kernels
# -----------------------------------------

def icd_kernel(X, k, d, k_batch=None, verbose=False):
    """
    input
    - X: sequence of length n
    - k: positive definite kernel
    - d: dim of output data (<= n)

    output:
    - A: n x d matrix
        - s.t. A @ A.T ~ K (Gram matrix)
               A[i] @ A[j] ~ k(X[i],X[j])
    - pivot_ids: d-length list consist of subset of range(n), corresponding each column of A
    - pivot_xs:  d-length list consist of subset of X,        corresponding each column of A
    """
    n = len(X)
    assert d <= n

    A = np.zeros(shape=(n, d))
    # todo: カーネルによっては固定の値で初期化できそう. e.g., cosine, Gaussian
    diag = np.array([k(x, x) for x in X])
    assert np.all(diag >= 0.)
    # todo: set にしないと (list にしちゃうと) remove のオーダーがでかい?
    remain_ids = list(range(n))
    pivot_ids = list()  # used ids

    print('    iter: {}, approx error: {}'.format(0, sum(diag)),
          file=sys.stderr)

    for i in range(d):
        # select pivot (index)
        p = np.argmax(diag)
        pivot_ids.append(p)  # p == pivot_ids[i]
        remain_ids.remove(p)

        A[p, i] = np.sqrt(diag[p])
        if k_batch:
            A[remain_ids, i] = np.array((k_batch(X,
                                                 X[p].reshape(1, -1)).reshape(
                (n,)) - A[:, 0:i] @ A[p, 0:i]) / A[p, i])[remain_ids]
        else:
            A[remain_ids, i] = np.array(
                ([k(X[i], X[p]) for i in range(n)] - A[:, 0:i] @ A[p, 0:i]) / A[
                    p, i])[remain_ids]
            # A[remain_ids, i] = ([k(X[i],X[p]) for i in remain_ids] - A[remain_ids, 0:i] @ A[p, 0:i]) / A[p, i]
        diag[remain_ids] -= A[remain_ids, i] ** 2

        diag[p] = 0.0  # eliminate selected pivot

        if i + 1 <= 5 or (i + 1) % 10 == 0:
            print('    iter: {}, approx error: {}'.format(i + 1, sum(diag)),
                  file=sys.stderr)

    pivot_xs = [X[i] for i in pivot_ids]
    print('  pivot ids selected in iteration: {}'.format(
        pivot_ids), file=sys.stderr)
    return A, pivot_ids, pivot_xs


def icd_for_new_data(A, pivot_ids, pivot_xs, k, x_new):
    d = A.shape[1]
    assert len(pivot_ids) == d
    assert len(pivot_xs) == d

    a = list()
    # todo: -> listcomp
    for i in range(d):
        a.append((k(x_new, pivot_xs[i]) - a[0:i] @ A[pivot_ids[i], 0:i]) / A[
            pivot_ids[i], i])

    return np.array(a)
