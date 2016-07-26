#!/usr/bin/env python3

import os
import shelve
from itertools import repeat
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

levels = [0.00048828125, 0.0009765625,
          0.001953125, 0.00390625, 0.0078125,
          0.015625, 0.03125, 0.0625,
          0.125, 0.25, 0.5,
          1]
max_freq_deviation_percentage = 20
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

timestamps = {
    1: '2016-02-24-18-55-41',
    0.5: '2016-02-25-17-43-12',
    0.25: '2016-02-26-00-59-37',
    0.125: '2016-02-26-08-16-27',
    0.0625: '2016-02-26-16-36-53',
    0.03125: '2016-02-27-00-30-17',
    0.015625: '2016-02-27-08-27-01',
    0.0078125: '2016-02-29-10-55-58',
    0.00390625: '2016-02-29-18-44-06',
    0.001953125: '2016-03-01-02-56-00',
    0.0009765625: '2016-03-01-11-53-37',
    0.00048828125: '2016-03-01-19-37-59'
}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
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

with shelve.open(os.path.join('shelf', 'data.for_signal_level.shelve')) as db:
    if 'error_stats' in db:
        error_stats = db['error_stats']
    else:
        error_stats = {}
        for method in ['martin', 'swipe', 'yin']:
            error_stats[method] = [
                error_rate(method, level, optimal_offsets[method])
                for level in levels]
        db['error_stats'] = error_stats

plt.title("Error rates under different signal levels", fontsize=14,
          fontweight='bold')
plt.xlabel("level (relative proportion)")
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
plt.xscale('log', basex=2)
plt.gca().yaxis.set_major_formatter(formatter)

mic_m, = plt.plot(levels, error_stats['martin'], 'r-o', label="martin")
mic_s, = plt.plot(levels, error_stats['swipe'], 'g-o', label="swipe")
mic_y, = plt.plot(levels, error_stats['yin'], 'b-o', label="yin")
plt.legend(handles=[mic_m, mic_s, mic_y])

plt.savefig(os.path.join('../../gallery', 'error_rates_vs_signal_levels.pdf'),
            papertype='a4')

print("Done.")
