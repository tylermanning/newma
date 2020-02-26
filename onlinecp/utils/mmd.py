import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def rbf(x, y, gamma):
    sq_diff = (x - y)**2
    return np.exp(-gamma*sq_diff.sum(axis=1))


def linear_mmd(x, y, gamma=1):
    if x.ndim == 1:
        x = np.reshape(x, (len(x), 1))
    if y.ndim == 1:
        y = np.reshape(y, (len(y), 1))
    n = (x.shape[0] // 2) * 2

    mmd2 = (rbf(x[:n:2], x[1:n:2], gamma) + rbf(y[:n:2], y[1:n:2], gamma)
            - rbf(x[:n:2], y[1:n:2], gamma) - rbf(x[1:n:2], y[:n:2], gamma)).mean()
    return mmd2


def quad_mmd(X, Y, gamma=1, biased=True):
    if X.ndim == 1:
        X = np.reshape(X, (len(X), 1))
    if Y.ndim == 1:
        Y = np.reshape(Y, (len(Y), 1))
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    X_sqnorms = np.diagonal(XX)
    Y_sqnorms = np.diagonal(YY)

    K_XY = np.exp(-gamma * (
        -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = np.exp(-gamma * (
        -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = np.exp(-gamma * (
        -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd = ((K_XX.sum() - m) / (m * (m - 1))
               + (K_YY.sum() - n) / (n * (n - 1))
               - 2 * K_XY.mean())
    return mmd