#!/usr/bin/env python3

import math
import os
import random
import shutil

size_ratio = 0.2  # 20%

input_basedir = "/home/bdeng/datasets/speechdata_16kHz"
output_basedir = "/home/bdeng/datasets/speechdata_16kHz_1_5th"
input_refs_folder = os.path.join(input_basedir, 'ref')
output_refs_folder = os.path.join(output_basedir, 'ref')

os.makedirs(output_refs_folder)

refs = os.listdir(input_refs_folder)
picked_refs = random.sample(refs, math.floor(len(refs) * size_ratio))

for ref in picked_refs:
    ref_src = os.path.join(input_refs_folder, ref)
    ref_dst = os.path.join(output_refs_folder, ref)
    print("Generating", ref_dst)
    shutil.copyfile(ref_src, ref_dst)

    mic_filename = 'mic' + ref[3:-3] + '.wav'
    mic_src = os.path.join(input_basedir, mic_filename)
    mic_dst = os.path.join(output_basedir, mic_filename)
    print("Generating", mic_dst)
    shutil.copyfile(mic_src, mic_dst)

print("Done.")
