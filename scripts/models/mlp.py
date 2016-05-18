#!/usr/bin/env python3

import argparse
import os

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from sklearn.metrics import confusion_matrix

from speech_data import load_shuffled_points

parser = argparse.ArgumentParser(
    description='Run the MLP classifier.')
parser.add_argument('method',
                    help="which algorithm's data to use (martin/swipe/yin)")
args = parser.parse_args()
method = args.method

(X_train, y_train), (X_test, y_test) = load_shuffled_points(method,
                                                            test_split=0.1)
weights_file = os.path.join('shelf', 'mlp_model_weights-' + method + '.h5')

model = Sequential()
model.add(Dense(128, input_dim=14, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=["accuracy"])

plot(model, to_file='shelf/model-mlp.png', show_shapes=True)

if os.path.exists(weights_file):
    model.load_weights(weights_file)
else:
    model.fit(X_train, y_train,
              nb_epoch=40,
              batch_size=32)
    model.save_weights(weights_file)

predicted_classes = model.predict_classes(X_test, batch_size=32)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)

print('Test loss:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])

print('Confusion matrix:')
print(confusion_matrix(y_test, predicted_classes))
