#!/usr/bin/env python3

import os
import subprocess
from itertools import product
from multiprocessing import Pool

input_basedir = "/home/bdeng/datasets/speechdata_16kHz_1_5th"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz_1_5th_noise_added"

# signal + noise_lambda * noise
noise_lambdas = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]


def add_white_noise(src, noise_lambda):
    dst = os.path.join(output_basedir, str(noise_lambda),
                       os.path.basename(src))
    print("Generating", dst)
    vol_adjustment_signal = subprocess.run(
        ["sox", src, "-n", "stat", "-v"],
        stderr=subprocess.PIPE, universal_newlines=True
    ).stderr
    vol_adjustment_noise = subprocess.run(
        "sox " + src + " -p synth whitenoise | sox - -n stat -v",
        stderr=subprocess.PIPE, universal_newlines=True, shell=True
    ).stderr

    max_amp_signal = 1 / float(vol_adjustment_signal[:-1])
    max_amp_noise = 1 / float(vol_adjustment_noise[:-1])
    max_amp_sum = max_amp_signal + noise_lambda * max_amp_noise

    if max_amp_sum < 1:
        subprocess.run(
            "sox " + src + " -p synth whitenoise vol " + str(noise_lambda) +
            " | sox -m " + src + " - " + dst,
            shell=True)
    else:
        # avoid clipping
        alpha = 1 / max_amp_sum
        subprocess.run(
            "sox -m " +
            "<( sox " + src + " -p vol " + str(alpha) + " ) " +
            "<( sox " + src + " -p synth whitenoise vol " +
            str(noise_lambda * alpha) + " ) " + "-b 16 " +
            dst,
            shell=True, executable="/bin/bash")


wav_paths = []

for root, dirs, files in os.walk(input_basedir):
    for name in files:
        if name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))

for noise_lambda in noise_lambdas:
    os.makedirs(os.path.join(output_basedir, str(noise_lambda)))

pool = Pool()
pool.starmap(add_white_noise, product(wav_paths, noise_lambdas))

print("Done.")
