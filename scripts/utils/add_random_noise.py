#!/usr/bin/env python3

import os
import random
import subprocess
from itertools import product
from multiprocessing import Pool

input_basedir = "/home/bdeng/datasets/speechdata_16kHz_1_5th"
noise_files_folder = "/home/bdeng/datasets/noises_16kHz"
output_basedir = ("/home/bdeng/datasets/" +
                  "speechdata_16kHz_1_5th_with_noise_from_files")

# signal + noise_lambda * noise
noise_lambdas = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
noise_names = ['babble', 'factory1', 'factory2', 'pink', 'white']


def add_noise(src, noise_name, noise_lambda):
    noise_file_path = os.path.join(noise_files_folder, noise_name + '.wav')
    dst = os.path.join(output_basedir, noise_name, str(noise_lambda),
                       os.path.basename(src))
    print("Generating", dst)

    vol_adjustment_signal = subprocess.run(
        ["sox", src, "-n", "stat", "-v"],
        stderr=subprocess.PIPE, universal_newlines=True
    ).stderr
    vol_adjustment_noise = subprocess.run(
        ["sox", noise_file_path, "-n", "stat", "-v"],
        stderr=subprocess.PIPE, universal_newlines=True
    ).stderr
    duration_signal = subprocess.run(
        ['soxi', '-D', src],
        stdout=subprocess.PIPE, universal_newlines=True
    ).stdout
    duration_noise = subprocess.run(
        ['soxi', '-D', noise_file_path],
        stdout=subprocess.PIPE, universal_newlines=True
    ).stdout

    max_amp_signal = 1 / float(vol_adjustment_signal[:-1])
    max_amp_noise = 1 / float(vol_adjustment_noise[:-1])
    max_amp_sum = max_amp_signal + noise_lambda * max_amp_noise

    duration_signal = float(duration_signal[:-1])
    duration_noise = float(duration_noise[:-1])
    noise_interval_begin = random.uniform(0, duration_noise - duration_signal)

    if max_amp_sum < 1:
        subprocess.run(
            "sox " + noise_file_path + " -p trim " +
            str(noise_interval_begin) + " " + str(duration_signal) +
            " vol " + str(noise_lambda) +
            " | sox -m " + src + " - " + dst,
            shell=True)
    else:
        # avoid clipping
        alpha = 1 / max_amp_sum
        subprocess.run(
            "sox -m " +
            "<( sox " + src + " -p vol " + str(alpha) + " ) " +
            "<( sox " + noise_file_path + " -p trim " +
            str(noise_interval_begin) + " " + str(duration_signal) +
            " vol " + str(noise_lambda * alpha) + " ) " +
            "-b 16 " + dst,
            shell=True, executable="/bin/bash")

wav_paths = []

for root, dirs, files in os.walk(input_basedir):
    for name in files:
        if name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))

for pair in product(noise_names, [str(l) for l in noise_lambdas]):
    os.makedirs(os.path.join(output_basedir, *pair))

pool = Pool()
pool.starmap(add_noise, product(wav_paths, noise_names, noise_lambdas))

print("Done.")
