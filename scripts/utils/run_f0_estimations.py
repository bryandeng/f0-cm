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

jsnoori_path = "/home/bdeng/Documents/jsnoorijy"
# jsnoori_jython_path = os.path.join(jsnoori_path, "jython.jar")
jsnoori_scripts_paths = {
    "martin": os.path.join(jsnoori_path, "script_martin_algorithm.py"),
    "swipe": os.path.join(jsnoori_path, "script_swipe_algorithm.py"),
    "yin": os.path.join(jsnoori_path, "script_yin_algorithm.py")
}
jsnoori_params = [
    "-t", "10"
]

results_basedir = "/home/bdeng/datasets/results"
# put results in a subfolder named after the current time
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
results_folder = os.path.join(results_basedir, timestamp)
os.makedirs(results_folder)


def estimate(wav_path, method):
    print("Calculating on", wav_path)
    result_filepath = os.path.join(
        results_folder,
        os.path.splitext(os.path.basename(wav_path))[0] + "." + method + ".f0")
    subprocess.check_call(
        ["jython",
         jsnoori_scripts_paths[method],
         "-i", wav_path, "-o", result_filepath] + jsnoori_params)

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
