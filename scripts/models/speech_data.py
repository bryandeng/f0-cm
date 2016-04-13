#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn import cross_validation, preprocessing

hdf5_path = "/home/bdeng/datasets/f0_estimation_data.h5"
df = pd.read_hdf(hdf5_path, 'f0_estimation')

mfcc_values = df.xs('mfcc', axis=1, level=0).values


def load_data(method, test_split=0.1):
    estimated_values = df.loc[:, ('estimated', method)].values

    X = np.concatenate((mfcc_values, np.reshape(estimated_values, (-1, 1))),
                       axis=1)
    y = df.loc[:, ('correctness', method)].values.astype(int)

    # normalize to zero mean and unit variance
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=test_split, random_state=0)
    return (X_train, y_train), (X_test, y_test)
