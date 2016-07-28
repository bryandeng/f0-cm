#!/usr/bin/env python3

import os
import shelve
from itertools import product

import matplotlib.pyplot as plt

methods = ['martin', 'swipe', 'yin']
snrs = [20, 15, 10, 5, 0]
metric_names = ['classification_accuracy', 'roc_auc', 'nmi']
metric_indices = {'classification_accuracy': 0, 'roc_auc': 1, 'nmi': 2}

with shelve.open(os.path.join('shelf', 'metrics.shelve')) as db:
    mlp_metrics = db['mlp_metrics']
    lstm_metrics = db['lstm_metrics']
    mlp_metrics_on_voiced = db['mlp_metrics_on_voiced']
    mlp_metrics_on_unvoiced = db['mlp_metrics_on_unvoiced']
    lstm_metrics_on_voiced = db['lstm_metrics_on_voiced']
    lstm_metrics_on_unvoiced = db['lstm_metrics_on_unvoiced']

locs = {'classification_accuracy': 'lower right', 'roc_auc': 'lower right',
        'nmi': 'upper left'}

for method, metric_name in product(methods, metric_names):
    fig = plt.figure()
    fig.suptitle(
        "Classification metrics on distorted audios \n"
        "(" + method + "'s data, " + metric_name + ")",
        fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel(metric_name)
    ax.set_ylim([0, 1])

    ax.plot(
        snrs,
        [mlp_metrics[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='red', linestyle='solid', marker='o', markersize=4,
        label="MLP")
    ax.plot(
        snrs,
        [lstm_metrics[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='blue', linestyle='solid', marker='o', markersize=4,
        label="LSTM")

    ax.legend(loc=locs[metric_name])
    plt.savefig(os.path.join('../../gallery',
                             'classification_metrics_whole_' +
                             method + '_' + metric_name + '.pdf'),
                papertype='a4')

    fig = plt.figure()
    fig.suptitle(
        "Classification metrics on distorted audios \n"
        "(" + method + "'s data, " + metric_name + ")",
        fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel(metric_name)
    ax.set_ylim([0, 1])

    ax.plot(
        snrs,
        [mlp_metrics_on_voiced[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='red', linestyle='solid', marker='o', markersize=4,
        label="MLP, voiced segments")
    ax.plot(
        snrs,
        [mlp_metrics_on_unvoiced[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='yellow', linestyle='solid', marker='o', markersize=4,
        label="MLP, unvoiced segments")
    ax.plot(
        snrs,
        [lstm_metrics_on_voiced[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='blue', linestyle='solid', marker='o', markersize=4,
        label="LSTM, voiced segments")
    ax.plot(
        snrs,
        [lstm_metrics_on_unvoiced[(method, snr)][metric_indices[metric_name]]
         for snr in snrs],
        color='green', linestyle='solid', marker='o', markersize=4,
        label="LSTM, unvoiced segments")

    ax.legend(loc=locs[metric_name])
    plt.savefig(os.path.join('../../gallery',
                             'classification_metrics_voicedness_' +
                             method + '_' + metric_name + '.pdf'),
                papertype='a4')

print("Done.")
