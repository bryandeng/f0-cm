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
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset,
                                                   zoomed_inset_axes)

noise_names = ['babble', 'factory1', 'factory2', 'pink', 'white']
noise_lambdas = [0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]

max_freq_deviation_percentage = 20
methods = ['martin', 'swipe', 'yin']
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

timestamps = {
    'babble': {0: '2016-02-24-18-55-41', 0.0125: '2016-03-11-19-04-06',
               0.025: '2016-03-11-20-32-43', 0.05: '2016-03-11-22-02-16',
               0.1: '2016-03-11-23-31-56', 0.2: '2016-03-12-01-01-44',
               0.4: '2016-03-12-02-31-52', 0.8: '2016-03-12-04-02-29'},
    'factory1': {0: '2016-02-24-18-55-41', 0.0125: '2016-03-12-05-31-51',
                 0.025: '2016-03-12-07-01-45', 0.05: '2016-03-12-08-31-58',
                 0.1: '2016-03-12-10-01-53', 0.2: '2016-03-12-11-30-51',
                 0.4: '2016-03-12-12-58-08', 0.8: '2016-03-12-14-25-27'},
    'factory2': {0: '2016-02-24-18-55-41', 0.0125: '2016-03-12-15-52-46',
                 0.025: '2016-03-12-17-20-02', 0.05: '2016-03-12-18-47-18',
                 0.1: '2016-03-12-20-14-26', 0.2: '2016-03-12-21-41-44',
                 0.4: '2016-03-12-23-08-54', 0.8: '2016-03-13-00-36-10'},
    'pink': {0: '2016-02-24-18-55-41', 0.0125: '2016-03-13-02-03-27',
             0.025: '2016-03-13-03-30-32', 0.05: '2016-03-13-04-57-48',
             0.1: '2016-03-13-06-24-54', 0.2: '2016-03-13-07-52-32',
             0.4: '2016-03-13-09-19-42', 0.8: '2016-03-13-10-47-00'},
    'white': {0: '2016-02-24-18-55-41', 0.0125: '2016-03-13-12-16-42',
              0.025: '2016-03-13-13-46-41', 0.05: '2016-03-13-15-16-42',
              0.1: '2016-03-13-16-46-50', 0.2: '2016-03-13-18-17-10',
              0.4: '2016-03-13-19-47-38', 0.8: '2016-03-13-21-18-03'}
}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz_1_5th/ref"
refs = os.listdir(refs_folder)


def error_count(ref, method, noise, level, offset):
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
                                  timestamps[noise][level])
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


def error_rate(method, noise, level, offset):
    pool = Pool()
    error_counts = pool.starmap(
        error_count,
        zip(refs, repeat(method), repeat(noise), repeat(level),
            repeat(offset)))
    pool.close()
    n_values_total = sum(error_count_[0] for error_count_ in error_counts)
    n_errors_total = sum(error_count_[1] for error_count_ in error_counts)
    return n_errors_total/n_values_total

os.makedirs('shelf', exist_ok=True)

with shelve.open(os.path.join('shelf', 'data.for_random_noise.shelve')) as db:
    if 'error_stats' in db:
        error_stats = db['error_stats']
    else:
        error_stats = {}
        for key in noise_names:
            error_stats[key] = {}
        for noise, method in product(noise_names, methods):
            error_stats[noise][method] = [
                error_rate(method, noise, level, optimal_offsets[method])
                for level in noise_lambdas]
            db['error_stats'] = error_stats

colors = {'babble': 'red', 'factory1': 'green', 'factory2': 'blue',
          'pink': 'magenta', 'white': 'cyan'}
markers = {'martin': 'o', 'swipe': 's', 'yin': 'D'}

fig = plt.figure()
fig.suptitle("Error rates when random noise is added (1/5 dataset)",
             fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.set_xlabel("noise level")
ax.set_ylabel("error rate")


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

for noise, method in product(noise_names, methods):
    ax.plot(noise_lambdas, error_stats[noise][method], color=colors[noise],
            linestyle='solid', marker=markers[method], markersize=2,
            label=noise + ' ' + method)

ax.legend(loc='upper left', fontsize='xx-small')

axins = zoomed_inset_axes(ax, 25, loc=9)

for noise, method in product(noise_names, ['martin', 'swipe']):
    axins.plot(noise_lambdas[1:3], error_stats[noise][method][1:3],
               color=colors[noise], linestyle='solid', marker=markers[method],
               markersize=1)

axins.set_ylim(0.0838, 0.088)

plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.8")

plt.savefig(os.path.join('../../gallery',
                         'error_rates_random_noise_added.pdf'),
            papertype='a4')

for noise in noise_names:
    fig_per_noise = plt.figure()
    fig_per_noise.suptitle(
        "Error rates when random noise is added "
        "(sampled from same noise file)",
        fontsize=14, fontweight='bold')
    ax = fig_per_noise.add_subplot(111)
    ax.set_xlabel("noise level")
    ax.set_ylabel("error rate")
    plt.gca().yaxis.set_major_formatter(formatter)

    for method in methods:
        ax.plot(noise_lambdas, error_stats[noise][method], color=colors[noise],
                linestyle='solid', marker=markers[method], markersize=4,
                label=noise + ' ' + method)

    ax.legend(loc='upper left')
    plt.savefig(os.path.join('../../gallery',
                             'error_rates_random_noise_' + noise + '.pdf'),
                papertype='a4')

for method in methods:
    fig_per_method = plt.figure()
    fig_per_method.suptitle(
        "Error rates when random noise is added "
        "(for same estimation algorithm)",
        fontsize=14, fontweight='bold')
    ax = fig_per_method.add_subplot(111)
    ax.set_xlabel("noise level")
    ax.set_ylabel("error rate")
    plt.gca().yaxis.set_major_formatter(formatter)

    for noise in noise_names:
        ax.plot(noise_lambdas, error_stats[noise][method], color=colors[noise],
                linestyle='solid', marker=markers[method], markersize=4,
                label=noise + ' ' + method)

    ax.legend(loc='upper left')
    plt.savefig(os.path.join('../../gallery',
                             'error_rates_random_noise_' + method + '.pdf'),
                papertype='a4')

print("Done.")
