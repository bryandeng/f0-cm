#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

max_freq_deviation_percentage = 20
methods = ['martin', 'swipe', 'yin']
optimal_offsets = {'martin': 16, 'swipe': -4, 'yin': 6}

refs_folder = "/home/bdeng/datasets/speechdata_16kHz/ref"
features_folders = {
    'original_audio': "/home/bdeng/datasets/features/2016-06-07-17-57-51",
    'distorted_audio': "/home/bdeng/datasets/features/2016-06-08-03-24-28"}

hdf5_basedir = "/home/bdeng/datasets"


def create_samples(ref, features_folder, method):
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

    features_file = 'mic' + ref[3:-3] + '.' + method + '.features'
    feature_values = pd.read_csv(
        os.path.join(features_folder, features_file),
        index_col=0,
        skiprows=14,
        delim_whitespace=True,
    )

    result_values = feature_values.loc[:, 'f00_hz']
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
    feature_values['correctness'] = correctness
    # reference has fewer estimations near the end of audio file
    feature_values.dropna(inplace=True)
    return feature_values

for method in methods:
    data = {}
    for ref in os.listdir(refs_folder):
        wav_basename = 'mic' + ref[3:-3]
        data[wav_basename] = create_samples(
            ref, features_folders['original_audio'], method)
        data[wav_basename+'_distorted'] = create_samples(
            ref, features_folders['distorted_audio'], method)
    dataframe = pd.concat(data)
    print(dataframe)
    hdf5_path = os.path.join(hdf5_basedir, "features_data_" + method + ".h5")
    dataframe.to_hdf(hdf5_path, 'features')
