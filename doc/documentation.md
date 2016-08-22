## `gallery/`
Various plots produced by running the scripts, which are also presented in the
publication.

## `scripts/`
Code developed for this project.

These scripts are easily configurable by setting the global variables in the
preamble.

#### `datasets/`
Related to the pre-processing, distortion and formatting of speech corpora.

Important scripts:
* `add_random_noise_wrt_snr.py` adds five types of noise respectively at
different signal-to-noise ratios.
* `add_random_noise_wrt_snr_any_noise.py` instead samples noise across
different noise audio files.
* `modify_signal_level.py` modifies signal levels (volumes) of audio files.
* `prepare_data.py` and `prepare_data_extra_testing.py` store features later
used in the machine learning part into HDF5 files.
They are easier to use and transmit than having a CSV file for each WAV file.

#### `models/`
Related to the neural network models.

* `mlp.py` and `lstm.py`: specification, training and evaluation of our two
neural network models.
Saved models, trained weights and data normalizers are available under
`shelf/`.
Please refer to [Keras documentation](https://keras.io/) for all the details.
* `extra_testing.py`: extra testing of the models regarding SNR and
voiced/unvoiced segments.
* `extra_testing_plot.py`: plot results of the extra testing.
* `speech_data.py` and `speech_data_extra_testing.py`: data loading utilities
for the neural network models.

#### `stats/`
Plot error rates of fundamental frequency estimations when the speech corpus is
distorted in various ways.

#### `utils/`
Important scripts:

* `run_f0_estimations.py` calls [JSnoori](http://jsnoori.loria.fr/) to conduct
F0 estimations.
* `run_feature_extractions.py` calls FeaturesPitchExtractionScripts to extract
the features used for machine learning.

Please note the timestamps of created results folders, as they are identifiers
used in error rates plotting and machine learning data preparation.
