import numpy as np
from numpy.random import dirichlet
from numpy import random as rd
from numpy.random import normal
from itertools import cycle


def draw_bkps(n_samples=100, n_bkps=3):
    """Draw a random partition with specified number of samples and specified number of changes."""
    alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
    bkps = np.cumsum(dirichlet(alpha) * n_samples).astype(int).tolist()
    bkps[-1] = n_samples
    return bkps


def pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=None,
                delta=(1, 10)):
    """Return a piecewise constant signal (step function) and the associated changepoints.
    Args:
        n_samples (int): signal length
        n_features (int, optional): number of dimensions
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        delta (tuple, optional): (delta_min, delta_max) max and min jump values
    Returns:
        tuple: signal of shape (n_samples, n_features), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples)
    delta_min, delta_max = delta
    # mean value
    center = np.zeros(n_features)
    for ind in np.split(tt_, bkps):
        if ind.size > 0:
            # jump value
            jump = rd.uniform(delta_min, delta_max, size=n_features)
            spin = rd.choice([-1, 1], n_features)
            center += jump * spin
            signal[ind] = center

    if noise_std is not None:
        noise = rd.normal(size=signal.shape) * noise_std
        signal = signal + noise
    ground_truth = np.array([1 if i in bkps else 0 for i in range(signal.shape[0])])
    return signal, bkps, ground_truth


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None):
    """
    Return piecewise linear signal and the associated changepoints.
    Args:
        n_samples (int, optional): signal length
        n_features (int, optional): number of covariates
        n_bkps (int, optional): number of change points
        noise_std (float, optional): noise std. If None, no noise is added
    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """

    covar = normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(n_samples=n_samples,
                                     n_bkps=n_bkps,
                                     n_features=n_features,
                                     noise_std=None)
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += normal(scale=noise_std, size=var.shape)
    signal = np.c_[var, covar]
    ground_truth = np.array([1 if i in bkps else 0 for i in range(signal.shape[0])])
    return signal, bkps, ground_truth


def pw_normal(n_samples=200, n_bkps=3):
    """Return a 2D piecewise Gaussian signal and the associated changepoints.
    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of change points
    Returns:
        tuple: signal of shape (n_samples, 2), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.zeros((n_samples, 2), dtype=float)
    cov1 = np.array([[1, 0.9], [0.9, 1]])
    cov2 = np.array([[1, -0.9], [-0.9, 1]])
    for sub, cov in zip(np.split(signal, bkps), cycle((cov1, cov2))):
        n_sub, _ = sub.shape
        sub += rd.multivariate_normal([0, 0], cov, size=n_sub)
    ground_truth = np.array([1 if i in bkps else 0 for i in range(signal.shape[0])])
    return signal, bkps, ground_truth


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None):
    """Return a 1D piecewise wavy signal and the associated changepoints.
    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
    Returns:
        tuple: signal of shape (n_samples, 1), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    # DeprecationWarning: Calling np.sum(generator) is deprecated
    # Use np.sum(np.from_iter(generator)) or the python sum builtin instead.
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    if noise_std is not None:
        noise = normal(scale=noise_std, size=signal.shape)
        signal += noise

    ground_truth = np.array([1 if i in bkps else 0 for i in range(signal.shape[0])])
    return signal, bkps, ground_truth