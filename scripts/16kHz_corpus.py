#!/usr/bin/env python3

import os
import shutil
import subprocess

dataset_basedir = "/home/bdeng/datasets/SPEECH DATA"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz"
os.mkdir(output_basedir)

directories, wav_paths, others_paths = [], [], []

for root, dirs, files in os.walk(dataset_basedir):
    for name in dirs:
        directories.append(os.path.join(root, name))
    for name in files:
        if os.path.splitext(name)[1] == '.wav':
            wav_paths.append(os.path.join(root, name))
        else:
            others_paths.append(os.path.join(root, name))

for directory in directories:
    output_dir = os.path.join(output_basedir,
                              os.path.relpath(directory, dataset_basedir))
    os.mkdir(output_dir)

for src in wav_paths:
    dst = os.path.join(output_basedir,
                       os.path.relpath(src, dataset_basedir))
    print("Generating", dst)
    subprocess.check_call(["sox", src, "-r", "16000", dst])

for src in others_paths:
    dst = os.path.join(output_basedir,
                       os.path.relpath(src, dataset_basedir))
    print("Generating", dst)
    shutil.copyfile(src, dst)

print("Done.")
