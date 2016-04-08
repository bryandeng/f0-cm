#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

max_freq_deviation_percentage = 20
methods = ['martin', 'swipe', 'yin']
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz_1_5th/ref"
results_folder = "/home/bdeng/datasets/results/2016-02-24-18-55-41"
mfcc_folder = "/home/bdeng/datasets/mfcc_text_1_5th"

hdf5_path = "/home/bdeng/datasets/f0_estimation_data.h5"


def estimation_correctness(ref, method):
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
    result_values.index += optimal_offsets[method]
    result_values = result_values.reindex(
        range(min(ref_values.index[0], result_values.index[0]),
              max(ref_values.index[-1], result_values.index[-1])+1)
    )
    result_values_interpolated = result_values.interpolate(method='nearest')
    result_values_interpolated.fillna(0.0, inplace=True)

    estimated_values = result_values_interpolated.loc[ref_values.index]
    diff = estimated_values - ref_values
    ref_values_no_zero = ref_values.replace(0.0, 100.0)
    deviation = diff.abs() / ref_values_no_zero

    correctness = deviation < max_freq_deviation_percentage / 100
    return estimated_values, correctness

data = {}

for mfcc_file in os.listdir(mfcc_folder):
    wav_basename = os.path.splitext(mfcc_file)[0]
    ref = 'ref' + wav_basename[3:] + '.f0'

    mfcc_coeffs = pd.read_csv(
        os.path.join(mfcc_folder, mfcc_file),
        header=None,
        usecols=range(1, 14),
        skiprows=4,
        delim_whitespace=True,
    )
    # align with estimations, the actual timing is 12.8 ms, 22.8ms, 32.8 ms...
    mfcc_coeffs.index = mfcc_coeffs.index * 10 + 16
    mfcc_column_multiindex = pd.MultiIndex.from_product(
        [['mfcc'], mfcc_coeffs.columns])
    mfcc_coeffs.columns = mfcc_column_multiindex

    data_per_file = mfcc_coeffs
    for method in methods:
        estimated_values, correctness = estimation_correctness(ref, method)
        data_per_file['estimated', method] = estimated_values
        data_per_file['correctness', method] = correctness
    data_per_file.dropna(inplace=True)
    data[wav_basename] = data_per_file

dataframe = pd.concat(data)
print(dataframe)
dataframe.to_hdf(hdf5_path, 'f0_estimation')
