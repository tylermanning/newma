import numpy as np
import pandas as pd
import plotly.graph_objects as go


import onlinecp.algos as algos
import onlinecp.utils.evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.gendata as gd


class Experiment:
    def __init__(self, X, truth_labels, no_of_dist, algo, thresholds=None):
        self.signal = X
        self.truth_labels = truth_labels
        self.no_of_dist = no_of_dist
        self.algo = algo
        self.thresholds = thresholds
        self.data_points = X.shape[0]
        self.features = X.shape[1]
        self.statistics = None
        self.padding = 3
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

    def get_stats(self, padding=3):
        # omits the first X number of CPs based on the padding, to account for initialization phase
        print(f"Omitting first {padding} distributions")
        n = self.no_of_dist
        detection_stat = np.array([i[0] for i in self.statistics])[int(padding * n):]  # padding
        online_th = np.array([i[1] for i in self.statistics])[int(padding * n):]
        ground_truth = self.truth_labels[int(padding * n):]
        return detection_stat, online_th, ground_truth

    def set_evaluation_settings(self, num_points=30, start_coeff=1.05, end_coeff=1.2):
        print(f'Calculating performance with {num_points} points')
        self.num_points = num_points
        self.start_coeff = start_coeff
        self.end_coeff = end_coeff

    def get_performance_metrics(self, detection_stat, online_th, ground_truth):
        EDD, FA, ND, results = ev.compute_curves(ground_truth, detection_stat, num_points=self.num_points,
                                        start_coeff=self.start_coeff, end_coeff=self.end_coeff)
        EDDth, FAth, NDth, results_thres = ev.compute_curves(ground_truth, detection_stat, num_points=self.num_points,
                                              thres_values=online_th, start_coeff=1, end_coeff=1)
        return {'EDD': EDD, 'FA': FA, 'ND': ND, 'EDDth': EDDth, 'FAth': FAth, 'NDth': NDth,
                'results': results, 'results_threshold': results_thres}

    def get_results(self, padding=3, num_points=30, start_coeff=1.05, end_coeff=1.2):
        stat, adaptive_th, ground_truth = self.get_stats(padding)
        self.set_evaluation_settings(num_points, start_coeff, end_coeff)
        return self.get_performance_metrics(stat, adaptive_th, ground_truth)

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
