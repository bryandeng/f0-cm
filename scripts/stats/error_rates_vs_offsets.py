#!/usr/bin/env python3

import argparse
import os
import shelve
from itertools import repeat
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

max_freq_deviation_percentage = 20
offsets = range(-40, 41)

parser = argparse.ArgumentParser(
    description='Evaluate error rates based on experiments results.')
parser.add_argument('timestamp',
                    help='result folder name (a timestamp)')
parser.add_argument("-v", "--verbose",
                    help="store result series under series/",
                    action="store_true")
args = parser.parse_args()
timestamp = args.timestamp
verbose = args.verbose

refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
results_folder = os.path.join("/home/bdeng/datasets/results", timestamp)

refs = os.listdir(refs_folder)

if verbose:
    os.makedirs(os.path.join('series', timestamp))


def error_count(ref, method, offset):
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

    if verbose:
        csv_filename = os.path.join(
            'series', timestamp,
            result + '.' + str(offset) + '.interpolated.txt')
        result_values_interpolated.to_csv(csv_filename, sep=' ', header=False)
    return n_values, n_errors


def error_rate(method, offset):
    pool = Pool()
    error_counts = pool.starmap(
        error_count,
        zip(refs, repeat(method), repeat(offset)))
    pool.close()
    n_values_total = sum(error_count_[0] for error_count_ in error_counts)
    n_errors_total = sum(error_count_[1] for error_count_ in error_counts)
    return n_errors_total/n_values_total


def error_rates(method):
    results = [error_rate(method, offset) for offset in offsets]
    return results

os.makedirs('shelf', exist_ok=True)

with shelve.open(os.path.join('shelf', 'data.' + timestamp + '.shelve')) as db:
    if 'error_stats' in db:
        error_stats = db['error_stats']
    else:
        error_stats = {}
        for method in ['martin', 'swipe', 'yin']:
            error_stats[method] = error_rates(method)
        db['error_stats'] = error_stats


plt.title("Error rates under different offsets", fontsize=14,
          fontweight='bold')
plt.xlabel("offset (ms)")
plt.ylabel("error rate")


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

mic_m, = plt.plot(offsets, error_stats['martin'], 'r-o', label="martin")
mic_s, = plt.plot(offsets, error_stats['swipe'], 'g-o', label="swipe")
mic_y, = plt.plot(offsets, error_stats['yin'], 'b-o', label="yin")
plt.legend(handles=[mic_m, mic_s, mic_y])
plt.savefig(os.path.join('../../gallery', 'error_rates_vs_offsets.pdf'),
            papertype='a4')

print("Done.")
