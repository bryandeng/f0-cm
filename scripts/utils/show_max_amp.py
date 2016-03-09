#!/usr/bin/env python3

import os
import subprocess
from multiprocessing import Pool

dataset = "/home/bdeng/datasets/speechdata_16kHz"


def vol_adjustment(src):
    vol_adjustment = subprocess.run(
        ["sox", src, "-n", "stat", "-v"],
        stderr=subprocess.PIPE,
        universal_newlines=True
    ).stderr
    return float(vol_adjustment[:-1])

wav_paths = []

for root, dirs, files in os.walk(dataset):
    for name in files:
        if name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))

pool = Pool()
vol_adjustments = pool.map(vol_adjustment, wav_paths)
max_amp = 1 / min(vol_adjustments)

print('Minimum volume adjustment:', min(vol_adjustments))
print('Maximum volume adjustment:', max(vol_adjustments))
print('Maximum amplitude over dataset:', max_amp)
