# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:09:35 2018

@author: nkeriven
"""
import glob

import numpy as np
from scipy.io import wavfile, loadmat
import scipy.signal as sig
from sklearn import mixture


def gmdraw(weights, mu, Sigma, n):
    """Draw samples from GMM.

    Parameters
    ----------
    weights: np.ndarray (k, ),
        weights of the GMM.
    mu: np.ndarray (k, d),
        means of the GMM.
    Sigma: np.ndarray (k, d, d),
        covariance of the GMM.
    n: int,
        number of samples.

    Returns
    -------
    X: np.ndarray (n, d),
        samples.
    label: np.ndarray of int (n, ),
        labels of samples.
    """

    k = weights.shape[0]
    d = mu.shape[1]
    p = np.cumsum(weights)
    # label
    label = np.random.rand(n)
    for i in range(n):
        label[i] = np.sum(label[i] > p)
    # cholesky
    cSigma = np.zeros((k, d, d))
    for l in range(k):
        cSigma[l, :, :] = np.linalg.cholesky(Sigma[l, :, :])
    # data
    X = np.zeros((n, d))
    for i in range(n):
        j = int(label[i])
        X[i, :] = mu[j, :] + np.dot(np.random.randn(1, d), cSigma[j, :, :])
    return X, label


def generateGMM(d=10, k=10, n=1000, std_mean=1, concentration_wishart=30, concentration_dirichlet=5):
    """Generate random parameters of GMM with diag covariance, draw samples.

    Parameters
    ----------
    d: int,
        dimension.
    k: int,
        number of components.
    n: int,
        number of samples.
    std_mean: float,
        the means will be drawn from a centered Gaussian with covariance (std_mean**2)*Id.
    concentration_wishart: float,
        the bigger, the more concentrated the diagonal covariances are around Id.
    concentration_dirichlet: float,
        the bigger, the more concentrated the weights are around uniform 1/k.

    Returns
    -------
    generated_data: dictionary with fields
        'data' (n,d): samples
        'weights' (k,): weights
        'means' (k,d): means
        'cov' (k,d): diagonal of covariances
        'label' (n,): labels of samples
        'gmm': scikit_learn mixture object
    """

    concentration_wishart = np.max((concentration_wishart, 3))

    # weights
    weights = np.random.dirichlet(concentration_dirichlet*np.ones(k))

    # means
    mu = std_mean*k**(1/d)*np.random.randn(k, d)

    # sigma
    Sigma = np.zeros((k, d))
    for l in range(k):
        Sigma[l, :] = (concentration_wishart - 2)/np.sum(np.random.randn(int(concentration_wishart), d)**2, axis=0)

    # sklearn object
    # , weights_init = GM['weights'], means_init = GM['means'], precisions_init = GM['cov'], max_iter = 1)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='diag')
    clf.means_ = mu
    clf. covariances_ = Sigma
    clf.precisions_cholesky_ = mixture._gaussian_mixture._compute_precision_cholesky(
        Sigma, clf.covariance_type)
    clf.weights_ = weights
    X, label = clf.sample(n_samples=n)
    p = np.random.permutation(n)
    X = X[p, :]
    label = label[p]
    generated_data = {'data': X,
                      'weights': weights,
                      'means': mu,
                      'cov': Sigma,
                      'label': label,
                      'gmm': clf}
    return generated_data


def stream_GMM(d=10, k=10, n=1000, nb_change=50, std_mean=0.2, concentration_wishart=30, concentration_dirichlet=5):
    if nb_change == 0:
        GM = generateGMM(d=d, k=k, n=5*n, std_mean=std_mean, concentration_wishart=concentration_wishart,
                         concentration_dirichlet=concentration_dirichlet)
        ground_truth = np.zeros(5*n)
        return GM['data'], ground_truth

    X = np.zeros((n*(nb_change), d))
    ground_truth = np.zeros(n*(nb_change))
    for i in range(nb_change):
        GM = generateGMM(d=d, k=k, n=n, std_mean=std_mean, concentration_wishart=concentration_wishart,
                         concentration_dirichlet=concentration_dirichlet)
        X[i*n:(i+1)*n, :] = GM['data']
        if i != 0:
            ground_truth[i*n] = 1
    return X, ground_truth


