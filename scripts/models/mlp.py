#!/usr/bin/env python3

import argparse
import os

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from sklearn.metrics import (confusion_matrix, normalized_mutual_info_score,
                             roc_auc_score)

from speech_data import load_shuffled_points

batch_size = 32

parser = argparse.ArgumentParser(description='Run the MLP classifier.')
parser.add_argument('method',
                    help="which algorithm's data to use (martin/swipe/yin)")
args = parser.parse_args()
method = args.method

(X_train, y_train), (X_test, y_test) = load_shuffled_points(method,
                                                            test_split=0.2)
weights_file = os.path.join('shelf', 'mlp_model_weights-' + method + '.h5')

model = Sequential()
model.add(Dense(128, input_dim=59, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

plot(model, to_file='shelf/model-mlp.png', show_shapes=True)
json_string = model.to_json()
open(os.path.join('shelf', 'mlp_model.json'), 'w').write(json_string)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=["accuracy"])

if os.path.exists(weights_file):
    model.load_weights(weights_file)
else:
    earlystopper = EarlyStopping()
    model.fit(X_train, y_train,
              nb_epoch=40,
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
