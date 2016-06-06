#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from itertools import repeat
from multiprocessing import Pool

parser = argparse.ArgumentParser(
    description='Run F0 estimations on the given dataset.')
parser.add_argument('dataset', help='path to the dataset')
args = parser.parse_args()

dataset = args.dataset

extractor_path = "/home/bdeng/Documents/FeaturesPitchExtractionScripts"
extractor_scripts_paths = {
    "martin": os.path.join(extractor_path, "script_martin_algorithm.py"),
    "swipe": os.path.join(extractor_path, "script_swipe_algorithm.py"),
    "yin": os.path.join(extractor_path, "script_yin_algorithm.py")
}
extractor_params = [
    "--start", "16", "-t", "10"
]

features_basedir = "/home/bdeng/datasets/features"
# put results in a subfolder named after the current time
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
features_folder = os.path.join(features_basedir, timestamp)
os.makedirs(features_folder)


def estimate(wav_path, method):
    print("Calculating on", wav_path)
    features_filepath = os.path.join(
        features_folder,
        os.path.splitext(os.path.basename(wav_path))[0] + "." + method +
        ".features")
    subprocess.check_call(
        ["jython",
         extractor_scripts_paths[method],
         "-i", wav_path, "-o", features_filepath] + extractor_params)

wav_paths = []

for root, dirs, files in os.walk(dataset):
    for name in files:
        if os.path.splitext(name)[1] == '.wav':
            wav_paths.append(os.path.join(root, name))

pool = Pool()
pool.starmap(estimate, zip(wav_paths, repeat("martin")))
pool.starmap(estimate, zip(wav_paths, repeat("swipe")))
pool.starmap(estimate, zip(wav_paths, repeat("yin")))

print("Done.")
