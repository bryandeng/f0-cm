#!/usr/bin/env python3

import os
import shelve
from collections import OrderedDict
from itertools import product

from tabulate import tabulate

methods = ['martin', 'swipe', 'yin']
noise_names = ['babble', 'factory1', 'factory2', 'pink', 'white']

levels = [0.00048828125, 0.0009765625,
          0.001953125, 0.00390625, 0.0078125,
          0.015625, 0.03125, 0.0625,
          0.125, 0.25, 0.5,
          1]

snrs = [20, 15, 10, 5, 0, -5]  # dB

with shelve.open(os.path.join('shelf', 'data.for_signal_level.shelve')) as db:
    error_stats_level = db['error_stats']

table_for_level = OrderedDict()
table_for_level['signal level'] = levels
for method in methods:
    table_for_level[method] = error_stats_level[method]

print("Error rates under different signal levels.")
print(tabulate(table_for_level, headers="keys"))

with shelve.open(os.path.join('shelf',
                              'data.for_random_noise_wrt_snr.shelve')) as db:
    error_stats_snr = db['error_stats']

print()

table_for_snr = OrderedDict()
table_for_snr['SNR'] = snrs
for (noise_name, method) in product(noise_names, methods):
    table_for_snr[(noise_name, method)] = error_stats_snr[noise_name][method]

print("Error rates when random noise at certain SNR is added.")
print(tabulate(table_for_snr, headers="keys"))
