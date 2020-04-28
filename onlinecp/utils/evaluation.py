import matplotlib.pyplot as plt
import numpy as np

def evaluate_detection(ground_truth, flagged):
    """Evaluate detection given ground_truth and flagged points.
    1. divide each window between two changes in two
    2. count each flagged sample in the half before each change as false alarm
    3. count the time until the first flagged point in the half after each change as detection delay (if no flagged point is present, count a missed detection)

    Parameters
    ----------
    ground_truth: np.ndarray of boolean (N, ),
        ground truth for changes (1 each time there's a change).
    flagged: np.ndarray of boolean (N,),
        a change is counted when going from 0 to 1.

    Returns
    -------
    results: dict with fields
            EDD : (averaged) Expected Detection Delay
            not_detected : number of missed changes
            false_alarm : (averaged) number of false alarms

    """
    n = ground_truth.shape[0]
    if n != flagged.shape[0]:
        print('error', n, flagged.shape[0])
    # change flagged into change point, going from 0 to 1
    cp = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        # consecutive flags are disregarded and only 1 is kept
        if not flagged[i] and flagged[i + 1]:
            cp[i] = 1

    EDD, not_detected, FA = 0, 0, 0
    num_change = int(ground_truth.sum())
    where_change = np.concatenate((np.argwhere(ground_truth).flatten(), np.array([n])))

    total_delays, total_undetected = 0, 0
    ttfas = []

    for i in range(num_change):
        begin_ind = where_change[i]
        end_ind = where_change[i + 1]
        middle_ind = int((begin_ind + end_ind) / 2)
        j = begin_ind
        while j <= middle_ind and not cp[j]:
            j = j + 1
        if cp[j]:
            delay = j - begin_ind
            total_delays += j - begin_ind
        else:
            total_undetected += 1
            not_detected += 1
        false_alarms = cp[middle_ind:end_ind].sum()
        if false_alarms > 0:
            ttfa = np.where(cp[middle_ind:end_ind])[0][0]
            ttfas.append(ttfa)

        FA += cp[middle_ind:end_ind].sum()

    results = {'num_changes': num_change,
               'EDD': total_delays / np.max((num_change - not_detected, 1)),
               '%_missed': 100 * not_detected / num_change,
               'total_missed': total_undetected,
               'total_false_alarms': FA,
               'false_alarm_rate': FA / num_change,
               'ttfas': ttfas,
               'avg_ttfa': sum(ttfas) / max(len(ttfas), 1),
               'cp': cp}
    print(results)
    return results


def compute_curves(ground_truth, dist,
                   num_points=50,
                   start_coeff=1.3, end_coeff=2,
                   thres_values=np.array([np.nan]),
                   thres_offset=0):
    """
    Evaluate performance for several level of thresholds, thres_values can be an array of adaptive threshold at each
    time point.

    Parameters
    ----------
    ground_truth: np.ndarray of boolean (N,),
        ground truth change.
    dist: np.ndarray (N,),
        online statistic.
    num_points: int,
        number of points in the scatter plot.
    start_coeff: float,
        start point for range of threshold (multiplicative).
    end_coeff: float,
        end point for range of threshold (multiplicative).
    thres_values: np.ndarray of floats,
        values of adaptive threshold. If nan, baseline fixed threshold = mean(dist).
    thres_offset: float,
        value of offset for the adaptive threshold.

    Returns
    -------
    EDDs: list,
        detection delay time.
    FAs: list,
        false alarms.
    NDs: list,
        missed detections.
    """
    if np.isnan(thres_values)[0]:
        thres_values = np.mean(dist)
    thres_levels = np.linspace(start_coeff, end_coeff, num_points)
    EDDs = np.zeros(num_points)
    FAs = np.zeros(num_points)
    NDs = np.zeros(num_points)
    for i in range(num_points):
        print('Evaluate performance', i+1, '/', num_points)
        flagged_points = dist > thres_levels[i] * thres_values + thres_offset
        res = evaluate_detection(ground_truth, flagged_points)
        EDDs[i] = res['EDD']
        FAs[i] = res['false_alarm_rate']
        NDs[i] = res['%_missed']
    return EDDs, FAs, NDs


def plot_metrics(edd, fa, md):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    axes[0].plot(fa, edd, 'o:')
    axes[0].set_xlabel('False alarms')
    axes[0].set_ylabel('Detection Delay')

    axes[1].plot(fa, md, 'o:')
    axes[1].set_xlabel('False alarms')
    axes[1].set_ylabel('Missed Detections (%)')
    return fig


def save_metrics(edd, fa, md, args):
    npz_filename = 'results_algo{}_d{}_m{}_n{}_nb{}_B{}.npz'.format(args.algo, args.d, args.m, args.n, args.nb, args.B)
    np.savez(npz_filename, edd=edd, fa=fa, md=md)
    return
