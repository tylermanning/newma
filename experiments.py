import numpy as np
import pandas as pd
import plotly.graph_objects as go


import onlinecp.algos as algos
import onlinecp.utils.evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.gendata as gd


class Experiment:
    def __init__(self, X, algo, thresholds=None):
        self.signal = X
        self.algo = algo
        self.thresholds = thresholds
        self.statistics = None
        self.ttfa = None
        self.buffer = None
        self.num_points = None
        self.start_coeff = None
        self.end_coeff = None

    def get_config(self):
        # common config
        choice_sigma = 'median'
        numel = 100
        # data for median trick to estimate sigma
        data_sigma_estimate = self.signal[:numel]
        B = 250  # window size

        # Scan-B config
        N = 3  # number of windows in scan-B

        # Newma and MA config
        big_Lambda, small_lambda = algos.select_optimal_parameters(
            B)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        # number of random features is set automatically with this criterion
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        m_OPU = 10 * m
        W, sigmasq = feat.generate_frequencies(
            m, self.signal.shape[1], data=data_sigma_estimate, choice_sigma=choice_sigma)

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
                                   kernel_func=lambda x, y: feat.gauss_kernel(
                                       x, y, np.sqrt(config['sigmasq'])),
                                   window_size=config['B'],
                                   nbr_windows=config['N'],
                                   adapt_forget_factor=config['thres_ff'])

        elif self.algo == 'kcusum':
            pass
        detector.apply_to_data(self.signal)
        self.statistics = detector.stat_stored

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def set_evaluation_settings(self, buffer, num_points, start_coeff, end_coeff):
        self.buffer = buffer
        self.num_points = num_points
        self.start_coeff = start_coeff
        self.end_coeff = end_coeff

    def set_ttfa(self, buffer):
        algo_statistic = np.array([i[0] for i in self.statistics])[buffer:]
        ttfa = []
        for threshold in self.thresholds:
            cross_indices = np.where(algo_statistic >= threshold)[0]
            if cross_indices.shape[0] > 0:
                ttfa.append(cross_indices.min())
            else:
                ttfa.append(None)
        self.ttfa = ttfa

    def plot_stat_time_series(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(self.signal.shape[0])], y=[i[0] for i in self.statistics],
                                 mode='lines',
                                 name=f'{self.algo} statistic'))
        fig.add_trace(go.Scatter(x=[i for i in range(self.signal.shape[0])], y=[i[1] for i in self.statistics],
                                 mode='lines',
                                 name=f'{self.algo} adaptive threshold'))
        fig.update_layout(title=f'{self.algo} statistic over time',
                          xaxis_title="Time",
                          yaxis_title="Statistic",
                          )
        fig.show()

    def plot_stat_distribution(self):
        fig = go.Figure()
        fig = go.Figure(data=[go.Histogram(x=[i[0] for i in self.statistics])])
        fig.show()
