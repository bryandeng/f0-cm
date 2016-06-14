#!/usr/bin/env python3

import argparse
import os

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot
from sklearn.metrics import (confusion_matrix, normalized_mutual_info_score,
                             roc_auc_score)

from speech_data import load_sequences

batch_size = 16
input_length = 2

parser = argparse.ArgumentParser(description='Run the LSTM classifier.')
parser.add_argument('method',
                    help="which algorithm's data to use (martin/swipe/yin)")
args = parser.parse_args()
method = args.method

(X_train, y_train), (X_test, y_test) = load_sequences(
    method, sequence_length=input_length, test_split=0.2)
weights_file = os.path.join('shelf', 'lstm_model_weights-' + method + '.h5')

model = Sequential()
model.add(LSTM(128, input_dim=59, input_length=input_length,
               return_sequences=True))
model.add(LSTM(128, input_dim=128, input_length=input_length,
               return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=["accuracy"])

plot(model, to_file='shelf/model-lstm.png', show_shapes=True)

if os.path.exists(weights_file):
    model.load_weights(weights_file)
else:
    earlystopper = EarlyStopping()
    model.fit(X_train, y_train,
              nb_epoch=10,
              batch_size=batch_size,
              validation_split=0.2,
              callbacks=[earlystopper])
    model.save_weights(weights_file)

print('Algorithm:', method)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test accuracy:', loss_and_metrics[1])

predicted_classes = model.predict_classes(X_test, batch_size=batch_size)
predicted_probas = model.predict_proba(X_test, batch_size=batch_size)
print('ROC AUC score:', roc_auc_score(y_test, predicted_probas))
print('Normalized Mutual Information (NMI):',
      normalized_mutual_info_score(y_test, predicted_classes.flatten()))
print('Confusion matrix:')
print(confusion_matrix(y_test, predicted_classes))
