#!/usr/bin/env python3

import os
import subprocess

input_basedir = "/home/bdeng/datasets/noises"
output_basedir = "/home/bdeng/datasets/noises_16kHz"
os.makedirs(output_basedir)


def resample(src):
    dst = os.path.join(output_basedir, os.path.basename(src))
    print("Generating", dst)
    subprocess.check_call(["sox", src, "-r", "16000", dst])

for file in os.listdir(input_basedir):
    resample(os.path.join(input_basedir, file))

print("Done.")
