#!/usr/bin/env python3

import os

import pandas as pd

mfcc_folder = "/home/bdeng/datasets/mfcc_text_1_5th"
mfcc_hdf5_path = "/home/bdeng/datasets/mfcc.h5"

mfcc_data = {}

for mfcc_file in os.listdir(mfcc_folder):
    wav_basename = os.path.splitext(mfcc_file)[0]
    mfcc_coeffs = pd.read_csv(
        os.path.join(mfcc_folder, mfcc_file),
        header=None,
        usecols=range(1, 14),
        skiprows=4,
        delim_whitespace=True,
    )
    mfcc_multiindex = pd.MultiIndex.from_product(
        [['mfcc'], mfcc_coeffs.columns])
    mfcc_coeffs.columns = mfcc_multiindex
    mfcc_data[wav_basename] = mfcc_coeffs

mfcc_dataframe = pd.concat(mfcc_data)
mfcc_dataframe.to_hdf(mfcc_hdf5_path, 'mfcc')
