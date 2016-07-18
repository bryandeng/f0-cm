#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib

hdf5_basedir = "/home/bdeng/datasets"


def load_dataframe(method, snr):
    hdf5_path = os.path.join(
        hdf5_basedir, "features_data_" + method + '_' + str(snr) + ".h5")
    df = pd.read_hdf(hdf5_path, 'features')
    return df


def load_points(method, snr):
    df = load_dataframe(method, snr)
    X = df.drop(['correctness', 'ref-voiced'], axis=1).values
    y = df.loc[:, 'correctness'].values.astype(int)
    ref_voiced = df.loc[:, 'ref-voiced'].values.astype(int)

    # normalize to the same distribution of training data
    scaler = joblib.load(os.path.join('shelf', 'mlp_normalizer.pkl'))
    X = scaler.transform(X)

    return (X, y), ref_voiced


def load_sequences(method, snr, sequence_length=3):
    df = load_dataframe(method, snr)
    grouped_per_audio_file = df.groupby(level=0, sort=False)
    dfs_per_audio = [group for name, group in grouped_per_audio_file]

    scaler = joblib.load(os.path.join('shelf', 'lstm_normalizer.pkl'))

    X_seq, y, ref_voiced = [], [], []

    for df_per_audio in dfs_per_audio:
        X_all = scaler.transform(df_per_audio.values[:, :-2].astype('float64'))
        for i in range(len(X_all) - sequence_length):
            X_seq.append(X_all[i:i+sequence_length, :])
        y_all = df_per_audio.values[:, -2].astype(int)
        y.append(y_all[sequence_length:])
        ref_voiced_all = df_per_audio.values[:, -1].astype(int)
        ref_voiced.append(ref_voiced_all[sequence_length:])

    X_seq, y, ref_voiced = (np.asarray(X_seq), np.hstack(y),
                            np.hstack(ref_voiced))

    return (X_seq, y), ref_voiced
