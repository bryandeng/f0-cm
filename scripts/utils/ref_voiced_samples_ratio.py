#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
refs = os.listdir(refs_folder)


def samples_stats(ref):
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
    n_values = len(ref_values)
    n_nonzero = len(ref_values.nonzero()[0])
    return n_values, n_nonzero

aggregated_stats = [samples_stats(ref) for ref in refs]
n_values_total = sum(stats[0] for stats in aggregated_stats)
n_nonzero_total = sum(stats[1] for stats in aggregated_stats)

print('Ratio of voiced samples in reference:', n_nonzero_total/n_values_total)
