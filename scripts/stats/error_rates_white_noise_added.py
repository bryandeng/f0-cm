#!/usr/bin/env python3

import os
from itertools import repeat
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

noise_lambdas = [0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]

max_freq_deviation_percentage = 20
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

timestamps = {
    0: '2016-02-24-18-55-41',
    0.0125: '2016-03-08-13-18-01',
    0.025: '2016-03-08-15-02-00',
    0.05: '2016-03-08-11-15-03',
    0.1: '2016-03-07-18-11-28',
    0.2: '2016-03-07-19-45-54',
    0.4: '2016-03-07-21-20-54',
    0.8: '2016-03-07-22-56-11'
}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz_1_5th/ref"
refs = os.listdir(refs_folder)


def error_count(ref, method, level, offset):
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
                                  timestamps[level])
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


def error_rate(method, level, offset):
    pool = Pool()
    error_counts = pool.starmap(
        error_count,
        zip(refs, repeat(method), repeat(level), repeat(offset)))
    pool.close()
    n_values_total = sum(error_count_[0] for error_count_ in error_counts)
    n_errors_total = sum(error_count_[1] for error_count_ in error_counts)
    return n_errors_total/n_values_total

error_stats = {}
for method in ['martin', 'swipe', 'yin']:
    error_stats[method] = [error_rate(method, level, optimal_offsets[method])
                           for level in noise_lambdas]

plt.title("Error rates when white noise is added (1/5 dataset)", fontsize=14,
          fontweight='bold')
plt.xlabel("white noise level")
plt.ylabel("F0 estimation error rate")


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
plt.gca().yaxis.set_major_formatter(formatter)

mic_m, = plt.plot(noise_lambdas, error_stats['martin'], 'r-o', label="martin")
mic_s, = plt.plot(noise_lambdas, error_stats['swipe'], 'g-o', label="swipe")
mic_y, = plt.plot(noise_lambdas, error_stats['yin'], 'b-o', label="yin")
plt.legend(handles=[mic_m, mic_s, mic_y], loc='upper left')

plt.savefig(os.path.join('../../gallery', 'error_rates_white_noise_added.pdf'),
            papertype='a4')

print("Done.")
