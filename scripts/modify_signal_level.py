#!/usr/bin/env python3

import os
import subprocess
from itertools import product
from multiprocessing import Pool

input_basedir = "/home/bdeng/datasets/speechdata_16kHz"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz_level_modified"

vols = [0.5, 0.25, 0.125]

wav_paths = []

for root, dirs, files in os.walk(input_basedir):
    for name in files:
        if name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))


def modify_level(src, vol):
    dst = os.path.join(output_basedir, str(vol), os.path.basename(src))
    print("Generating", dst)
    subprocess.check_call(["sox", src, dst, "vol", str(vol)])

for vol in vols:
    os.makedirs(os.path.join(output_basedir, str(vol)))

pool = Pool()
pool.starmap(modify_level, product(wav_paths, vols))

print("Done.")
