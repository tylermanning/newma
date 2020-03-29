import pandas as pd
import numpy as np


import onlinecp.utils.feature_functions as feat
from sklearn.metrics.pairwise import euclidean_distances


class Bandwidth:
    def __init__(self):
        self.X = None
        self.Y = None
        self.bandwidth = None

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y

    def set_bandwidth(self, data, method='median'):
        # Median trick
        if method == 'median':
            if data is None:
                raise Exception(
                    "You chose median for sigma in generate_frequencies, but you didn't pass any data")
            distances = euclidean_distances(data, data)
            squared_distances = distances.flatten() ** 2
            sigmasq = np.median(squared_distances)
            self.bandwidth = np.sqrt(sigmasq)

    def get_guassian_kernel(self):
        if not (self.X or self.Y):
            print("Need to set X & Y to run calculation")
        feat.gauss_kernel(self.X, self.Y, self.bandwidth)
