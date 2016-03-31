#!/usr/bin/env bash

# This script takes the path to a folder as argument and
# outputs MFCC features for all WAV files in that folder.
# You can find the results in mfcc_text/ folder of the current directory.

# Usage example: ./mfcc_features.sh speechdata_16kHz/

# The Perl script has dependencies, so the basedirs need to be added to PATH.
# in file prog_path:
# export PATH=$PATH:~jouvet/Prog/AcousticAnalysis/Scripts/
# export PATH=$PATH:~jouvet/Prog/SphinxBase/sphinxbase-0.6.1/bin/
source prog_path

~jouvet/Prog/AcousticAnalysis/Scripts/AcousticAnalysisDirectory.pl \
    -type 16k_MFCC_Standard_Sphinx \
    -iDir "$1" \
    -iExt .wav \
    -oDir ./mfcc/ \
    -oExt .mfcc

mkdir -p mfcc_text

for file in ./mfcc/*; do
    ~jouvet/Prog/AcousticAnalysis/bin/linux/ProcessingFeatures \
        -srcFmt Sphinx \
        -srcFrameCoeffs 13 \
        -srcFramePeriod 10 \
        -display 0 -1 \
        -srcFn "$file" \
    > "./mfcc_text/$(basename "$file" .mfcc).mfcc_text"
done
