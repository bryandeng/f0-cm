#!/usr/bin/env python3

import os
import shelve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool

timestamp = "2016-02-10-16-59-39"
refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
results_folder = os.path.join("/home/bdeng/datasets/results", timestamp)

refs = os.listdir(refs_folder)

max_freq_deviation_percentage = 20
offsets = range(-40, 41)


def error_count(ref, signal, method, offset):
    # signal: mic/lar, method: martin/swipe/yin
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

    result = signal + ref[3:-3] + "." + method + ".f0"
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
        range(0, max(ref_values.index[-1], result_values.index[-1])+1)
    )
    result_values_interpolated = result_values.interpolate(method='linear')

    diff = result_values_interpolated.loc[ref_values.index] - ref_values
    ref_values_no_zero = ref_values[ref_values == 0] = 100
    # the best denominator to filter out false positive voiced values, for
    # that human voice is always higher than 100 * 20% = 20 Hz
    deviation = diff.abs() / ref_values_no_zero

    n_values = len(ref_values)
    n_errors = len(deviation[deviation > max_freq_deviation_percentage / 100])
    return n_values, n_errors


def error_rate(signal, method, offset):
    pool = Pool()
    error_counts = pool.starmap(
        error_count,
        zip(refs, repeat(signal), repeat(method), repeat(offset)))
    pool.close()
    n_values_total = sum(error_count_[0] for error_count_ in error_counts)
    n_errors_total = sum(error_count_[1] for error_count_ in error_counts)
    return n_errors_total/n_values_total


def error_rates(signal, method):
    results = [error_rate(signal, method, offset) for offset in offsets]
    return results

d = shelve.open('error_rates_data.' + timestamp + '.shelve')
if 'error_data' in d:
    error_data = d['error_data']
else:
    error_data = {}
    error_data['mic'], error_data['lar'] = {}, {}
    for method in ['martin', 'swipe', 'yin']:
        error_data['mic'][method] = error_rates('mic', method)
        error_data['lar'][method] = error_rates('lar', method)
    d['error_data'] = error_data
d.close()

fig = plt.figure()
fig.suptitle("Error rates under different offsets", fontsize=14,
             fontweight='bold')

mic = fig.add_subplot(211)
mic.set_title("MIC signal")
mic.set_xlabel("offset (ms)")
mic.set_ylabel("error rate")
mic_m, = mic.plot(offsets, error_data['mic']['martin'], 'r-o', label="martin")
mic_s, = mic.plot(offsets, error_data['mic']['swipe'], 'g-o', label="swipe")
mic_y, = mic.plot(offsets, error_data['mic']['yin'], 'b-o', label="yin")
# mic.legend(handles=[mic_m, mic_s, mic_y])
mic.legend([mic_m, mic_s, mic_y])  # old matplotlib

lar = fig.add_subplot(212)
lar.set_title("LAR signal")
lar.set_xlabel("offset (ms)")
lar.set_ylabel("error rate")
lar_m, = lar.plot(offsets, error_data['lar']['martin'], 'r-o', label="martin")
lar_s, = lar.plot(offsets, error_data['lar']['swipe'], 'g-o', label="swipe")
lar_y, = lar.plot(offsets, error_data['lar']['yin'], 'b-o', label="yin")
# lar.legend(handles=[lar_m, lar_s, lar_y])
lar.legend([lar_m, lar_s, lar_y])  # old matplotlib

fig.set_size_inches(8.27, 11.69)  # A4
plt.savefig('error_rates.' + timestamp + '.pdf', papertype='a4')
