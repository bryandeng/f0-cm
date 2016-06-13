#!/usr/bin/env python3

import math
import os

import numpy as np
import pandas as pd
from sklearn import cross_validation, preprocessing

hdf5_basedir = "/home/bdeng/datasets"


def load_dataframe(method):
    hdf5_path = os.path.join(hdf5_basedir, "features_data_" + method + ".h5")
    df = pd.read_hdf(hdf5_path, 'features')
    return df


def load_shuffled_points(method, test_split=0.2):
    # actually by default, Keras also shuffles samples during training
    df = load_dataframe(method)
    X = df.drop('correctness', axis=1).values
    y = df.loc[:, 'correctness'].values.astype(int)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=test_split, random_state=0)
    # normalize to zero mean and unit variance
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_test, y_test)


def load_sequences(method, sequence_length=3, test_split=0.2):
    df = load_dataframe(method)
    grouped_per_audio_file = df.groupby(level=0, sort=False)
    dfs_per_audio = [group for name, group in grouped_per_audio_file]
    # for convenience, test_split is w.r.t number of audio files
    n_train = math.floor(len(dfs_per_audio) * (1.0 - test_split))
    dfs_train, dfs_test = dfs_per_audio[:n_train], dfs_per_audio[n_train:]

    scaler = preprocessing.StandardScaler().fit(
        np.vstack(df.values for df in dfs_train)[:, :-1].astype('float64'))

    X_train, y_train, X_test, y_test = [], [], [], []

    for df_per_audio in dfs_train:
        X = scaler.transform(df_per_audio.values[:, :-1].astype('float64'))
        for i in range(len(X) - sequence_length):
            X_train.append(X[i:i+sequence_length, :])
        y = df_per_audio.values[:, -1].astype(int)
        y_train.append(y[sequence_length:])

    for df_per_audio in dfs_test:
        X = scaler.transform(df_per_audio.values[:, :-1].astype('float64'))
        for i in range(len(X) - sequence_length):
            X_test.append(X[i:i+sequence_length, :])
        y = df_per_audio.values[:, -1].astype(int)
        y_test.append(y[sequence_length:])

    X_train, y_train = np.asarray(X_train), np.hstack(y_train)
    X_test, y_test = np.asarray(X_test), np.hstack(y_test)

    return (X_train, y_train), (X_test, y_test)
