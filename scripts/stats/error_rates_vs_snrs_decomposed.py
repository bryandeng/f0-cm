#!/usr/bin/env python3

import os
import shelve
from itertools import product, repeat
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

noise_names = ['babble', 'factory1', 'factory2', 'pink', 'white']
snrs = [20, 15, 10, 5, 0, -5]  # dB

max_freq_deviation_percentage = 20
methods = ['martin', 'swipe', 'yin']
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

timestamps = {
    'babble': {20: '2016-06-24-19-01-46', 15: '2016-06-24-20-29-28',
               10: '2016-06-24-21-57-09', 5: '2016-06-24-23-25-03',
               0: '2016-06-25-00-53-01', -5: '2016-06-25-02-21-07'},
    'factory1': {20: '2016-06-25-03-49-19', 15: '2016-06-25-05-17-09',
                 10: '2016-06-25-06-45-00', 5: '2016-06-25-08-13-03',
                 0: '2016-06-25-09-41-09', -5: '2016-06-25-11-09-11'},
    'factory2': {20: '2016-06-25-12-36-47', 15: '2016-06-25-14-04-17',
                 10: '2016-06-25-15-31-53', 5: '2016-06-25-16-59-30',
                 0: '2016-06-25-18-27-18', -5: '2016-06-25-19-54-40'},
    'pink': {20: '2016-06-25-21-22-14', 15: '2016-06-25-22-49-53',
             10: '2016-06-26-00-17-36', 5: '2016-06-26-01-45-03',
             0: '2016-06-26-03-12-40', -5: '2016-06-26-04-40-07'},
    'white': {20: '2016-06-26-06-07-44', 15: '2016-06-26-07-35-24',
              10: '2016-06-26-09-03-01', 5: '2016-06-26-10-30-43',
              0: '2016-06-26-11-58-32', -5: '2016-06-26-13-26-34'}
}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz_1_5th/ref"
refs = os.listdir(refs_folder)


def error_count(ref, method, noise, snr, offset):
    # method: martin/swipe/yin
    ref_values = pd.read_csv(
        os.path.join(refs_folder, ref),
        sep=' ',
        header=None,
        names=['f0'],
        dtype={0: np.float64},
        usecols=[0],
        squeeze=True,
    )
    ref_values.index = ref_values.index * 10 + 16

    result = 'mic' + ref[3:-3] + '.' + method + '.f0'
    results_folder = os.path.join("/home/bdeng/datasets/results",
                                  timestamps[noise][snr])
    result_values = pd.read_csv(
        os.path.join(results_folder, result),
        header=None,
        index_col=0,
        names=['f0'],
        skiprows=11,
        dtype={0: np.int64, 1: np.float64},
        delim_whitespace=True,
        squeeze=True,
    )
    result_values.index += offset
    result_values = result_values.reindex(
        range(min(ref_values.index[0], result_values.index[0]),
              max(ref_values.index[-1], result_values.index[-1])+1)
    )
    result_values_interpolated = result_values.interpolate(method='nearest')
    result_values_interpolated.fillna(0.0, inplace=True)

    diff = result_values_interpolated.loc[ref_values.index] - ref_values
    ref_values_no_zero = ref_values.replace(0.0, 100.0)
    # the best denominator to filter out false positive voiced values, for
    # that human voice is always higher than 100 * 20% = 20 Hz
    deviation = diff.abs() / ref_values_no_zero

    n_values = len(ref_values)
    n_errors = len(deviation[deviation > max_freq_deviation_percentage / 100])

    return n_values, n_errors


def error_rate(method, noise, snr, offset):
    pool = Pool()
    error_counts = pool.starmap(
        error_count,
        zip(refs, repeat(method), repeat(noise), repeat(snr),
            repeat(offset)))
    pool.close()
    n_values_total = sum(error_count_[0] for error_count_ in error_counts)
    n_errors_total = sum(error_count_[1] for error_count_ in error_counts)
    return n_errors_total/n_values_total

os.makedirs('shelf', exist_ok=True)

with shelve.open(
        os.path.join('shelf', 'data.for_random_noise_wrt_snr.shelve')) as db:
    if 'error_stats' in db:
        error_stats = db['error_stats']
    else:
        error_stats = {}
        for key in noise_names:
            error_stats[key] = {}
        for noise, method in product(noise_names, methods):
            error_stats[noise][method] = [
                error_rate(method, noise, snr, optimal_offsets[method])
                for snr in snrs]
            db['error_stats'] = error_stats

colors = {'babble': 'red', 'factory1': 'green', 'factory2': 'blue',
          'pink': 'magenta', 'white': 'cyan'}
markers = {'martin': 'o', 'swipe': 's', 'yin': 'D'}


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

formatter = FuncFormatter(to_percent)


for noise in noise_names:
    fig_per_noise = plt.figure()
    fig_per_noise.suptitle(
        "Error rates when random noise at certain SNR is added \n"
        "(sampled from same noise file, 1/5 dataset)",
        fontsize=14, fontweight='bold')
    ax = fig_per_noise.add_subplot(111)
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel("F0 estimation error rate")
    plt.gca().yaxis.set_major_formatter(formatter)

    for method in methods:
        ax.plot(snrs, error_stats[noise][method], color=colors[noise],
                linestyle='solid', marker=markers[method], markersize=4,
                label=noise + ' ' + method)

    ax.legend(loc='upper right')
    plt.savefig(
        os.path.join('../../gallery',
                     'error_rates_random_noise_wrt_snr_' + noise + '.pdf'),
        papertype='a4')

for method in methods:
    fig_per_method = plt.figure()
    fig_per_method.suptitle(
        "Error rates when random noise at certain SNR is added \n"
        "(for same estimation algorithm, 1/5 dataset)",
        fontsize=14, fontweight='bold')
    ax = fig_per_method.add_subplot(111)
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel("F0 estimation error rate")
    plt.gca().yaxis.set_major_formatter(formatter)

    for noise in noise_names:
        ax.plot(snrs, error_stats[noise][method], color=colors[noise],
                linestyle='solid', marker=markers[method], markersize=4,
                label=noise + ' ' + method)

    ax.legend(loc='lower left')
    plt.savefig(
        os.path.join('../../gallery',
                     'error_rates_random_noise_wrt_snr_' + method + '.pdf'),
        papertype='a4')

print("Done.")
