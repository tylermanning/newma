import numpy as np
# import pandas as pd
import plotly.graph_objects as go


import onlinecp.algos as algos
import onlinecp.utils.evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.gendata as gd


class Experiment:
    def __init__(self, X, algo, thresholds):
        self.signal = X
        self.algo = algo
        self.thresholds = thresholds
        self.statistic = None
        self.ttfa = None

    def get_config(self):
        # common config
        choice_sigma = 'median'
        numel = 100
        data_sigma_estimate = self.signal[:numel]  # data for median trick to estimate sigma
        B = 250 # window size

        # Scan-B config
        N = 3  # number of windows in scan-B

        # Newma and MA config
        big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        # number of random features is set automatically with this criterion
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        m_OPU = 10 * m
        W, sigmasq = feat.generate_frequencies(m, self.signal.shape[1], data=data_sigma_estimate, choice_sigma=choice_sigma)

        return {'choice_sigma': choice_sigma, 'data_sigma_estimate': data_sigma_estimate, 'B': B, 'N': N,
                 'big_lambda': big_Lambda, 'small_lambda': small_lambda, 'thres_ff': thres_ff, 'm': m, 'W': W,
                 'sigmasq': sigmasq}

    def run_algo(self):
        config = self.get_config()
        if self.algo == 'newmaRFF':
            print('Start algo ', self.algo, '...')
            def feat_func(x):
                return feat.fourier_feat(x, config['W'])

            detector = algos.NEWMA(self.signal[0],
                                   forget_factor=config['big_lambda'],
                                   forget_factor2=config['small_lambda'],
                                   feat_func=feat_func,
                                   adapt_forget_factor=config['thres_ff'])
        elif self.algo == 'ScanB':
            print('Start algo ', self.algo, '... (can be long !)')
            detector = algos.ScanB(self.signal[0],
                                   kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(config['sigmasq'])),
                                   window_size=config['B'],
                                   nbr_windows=config['N'],
                                   adapt_forget_factor=config['thres_ff'])

        elif self.algo == 'kcusum':
            pass
        detector.apply_to_data(self.signal)
        self.statistic = detector.stat_stored

    def set_ttfa(self):
        self.ttfa = [np.where(self.statistic >= threshold)[0].min() for threshold in self.thresholds]

    def plot_stat_time_series(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(self.signal.shape[0])], y=[i[0] for i in self.statistic],
                                 mode='lines',
                                 name=f'{self.algo} statistic'))
        fig.show()

    def plot_stat_distribution(self):
        fig = go.Figure()
        fig = go.Figure(data=[go.Histogram(x=[i[0] for i in self.statistic])])
        fig.show()
