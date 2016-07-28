#!/usr/bin/env python3

import os
import shelve

from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from speech_data_extra_testing import load_points, load_sequences

methods = ['martin', 'swipe', 'yin']
snrs = [20, 15, 10, 5, 0]

input_length = 4


def model_metrics(model, X, y, batch_size):
    loss_and_metrics = model.evaluate(X, y, batch_size=batch_size)
    predicted_classes = model.predict_classes(X, batch_size=batch_size)
    predicted_probas = model.predict_proba(X, batch_size=batch_size)

    accuracy = loss_and_metrics[1]
    roc_auc = roc_auc_score(y, predicted_probas)
    nmi = normalized_mutual_info_score(y, predicted_classes.flatten())
    return accuracy, roc_auc, nmi

mlp_model = model_from_json(
    open(os.path.join('shelf', 'mlp_model.json')).read())
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
mlp_model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=["accuracy"])

lstm_model = model_from_json(
    open(os.path.join('shelf', 'lstm_model.json')).read())
lstm_model.compile(loss='binary_crossentropy',
                   optimizer='sgd',
                   metrics=["accuracy"])

mlp_metrics = {}
lstm_metrics = {}
mlp_metrics_on_voiced = {}
mlp_metrics_on_unvoiced = {}
lstm_metrics_on_voiced = {}
lstm_metrics_on_unvoiced = {}

for method in methods:
    mlp_model.load_weights(
        os.path.join('shelf', 'mlp_model_weights-' + method + '.h5'))
    lstm_model.load_weights(
        os.path.join('shelf', 'lstm_model_weights-' + method + '.h5'))

    for snr in snrs:
        (X, y), ref_voiced = load_points(method, snr)
        mlp_metrics[(method, snr)] = model_metrics(
            mlp_model, X, y, batch_size=32)
        metrics_on_voiced = model_metrics(
            mlp_model, X[ref_voiced == 1], y[ref_voiced == 1], batch_size=32)
        metrics_on_unvoiced = model_metrics(
            mlp_model, X[ref_voiced == 0], y[ref_voiced == 0], batch_size=32)
        mlp_metrics_on_voiced[(method, snr)] = metrics_on_voiced
        mlp_metrics_on_unvoiced[(method, snr)] = metrics_on_unvoiced

        (X_seq, y), ref_voiced = load_sequences(method, snr,
                                                sequence_length=input_length)
        lstm_metrics[(method, snr)] = model_metrics(
            lstm_model, X_seq, y, batch_size=16)
        metrics_on_voiced = model_metrics(
            lstm_model, X_seq[ref_voiced == 1], y[ref_voiced == 1],
            batch_size=16)
        metrics_on_unvoiced = model_metrics(
            lstm_model, X_seq[ref_voiced == 0], y[ref_voiced == 0],
            batch_size=16)
        lstm_metrics_on_voiced[(method, snr)] = metrics_on_voiced
        lstm_metrics_on_unvoiced[(method, snr)] = metrics_on_unvoiced

with shelve.open(os.path.join('shelf', 'metrics.shelve')) as db:
    db['mlp_metrics'] = mlp_metrics
    db['lstm_metrics'] = lstm_metrics
    db['mlp_metrics_on_voiced'] = mlp_metrics_on_voiced
    db['mlp_metrics_on_unvoiced'] = mlp_metrics_on_unvoiced
    db['lstm_metrics_on_voiced'] = lstm_metrics_on_voiced
    db['lstm_metrics_on_unvoiced'] = lstm_metrics_on_unvoiced
