#!/usr/bin/env python3

import os
import shutil
import subprocess
from multiprocessing import Pool

dataset_basedir = "/home/bdeng/datasets/SPEECH DATA"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz"
refs_folder = os.path.join(output_basedir, "ref")
os.makedirs(refs_folder)


def resample(src):
    dst = os.path.join(output_basedir, os.path.basename(src))
    print("Generating", dst)
    subprocess.check_call(["sox", src, "-r", "16000", dst])

wav_paths, ref_paths = [], []

for root, dirs, files in os.walk(dataset_basedir):
    for name in files:
        if name.startswith('ref'):
            ref_paths.append(os.path.join(root, name))
        elif name.startswith('mic'):
            wav_paths.append(os.path.join(root, name))

for src in ref_paths:
    dst = os.path.join(refs_folder, os.path.basename(src))
    print("Generating", dst)
    shutil.copyfile(src, dst)

pool = Pool()
pool.map(resample, wav_paths)

print("Done.")
