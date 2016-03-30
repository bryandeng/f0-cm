#!/usr/bin/env python3

import os
import subprocess
from itertools import product
from multiprocessing import Pool

input_basedir = "/home/bdeng/datasets/speechdata_16kHz"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz_level_modified"

vols = [0.5, 0.25, 0.125,
        0.0625, 0.03125, 0.015625,
        0.0078125, 0.00390625, 0.001953125,
        0.0009765625, 0.00048828125]


def modify_level(src, vol):
    dst = os.path.join(output_basedir, str(vol), os.path.basename(src))
    print("Generating", dst)
    subprocess.check_call(["sox", src, dst, "vol", str(vol)])

for vol in vols:
    os.makedirs(os.path.join(output_basedir, str(vol)))

wav_paths = []

for root, dirs, files in os.walk(input_basedir):
    for name in files:
        if name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))

pool = Pool()
pool.starmap(modify_level, product(wav_paths, vols))

print("Done.")
