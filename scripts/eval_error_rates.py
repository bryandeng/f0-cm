#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from tabulate import tabulate

timestamp = "2016-02-10-16-59-39"
max_freq_deviation = 5  # Hz

refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
refs = os.listdir(refs_folder)
results_folder = os.path.join("/home/bdeng/datasets/results", timestamp)


def error_count(ref, signal, method):
    # signal: mic/lar, method: martin/swipe/yin
    ref_values = pd.read_csv(
        os.path.join(refs_folder, ref),
        sep=' ', dtype={0: np.float64},
        header=None, names=['f0'], usecols=[0])
    result = signal + ref[3:-3] + "." + method + ".f0"
    result_values = pd.read_csv(
        os.path.join(results_folder, result),
        delim_whitespace=True, dtype={0: np.float64},
        header=None, skiprows=11, names=['f0'], usecols=[1])
    diff = result_values.sub(ref_values, axis='index', fill_value=0).abs()
    n_values = len(result_values)
    n_errors = len(diff[diff['f0'] > max_freq_deviation])
    return n_values, n_errors


def error_rate(signal, method):
    error_counts = [error_count(ref, signal, method) for ref in refs]
    n_values_total = sum(error_count[0] for error_count in error_counts)
    n_errors_total = sum(error_count[1] for error_count in error_counts)
    return round(n_errors_total/n_values_total, 5)

table = [["martin", error_rate("mic", "martin"), error_rate("lar", "martin")],
         ["swipe", error_rate("mic", "swipe"), error_rate("lar", "swipe")],
         ["yin", error_rate("mic", "yin"), error_rate("lar", "yin")]]

print(tabulate(table, headers=["", "mic", "lar"]))
