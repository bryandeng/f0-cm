#!/usr/bin/env python3

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from speech_data import load_data

(X_train, y_train), (X_test, y_test) = load_data('martin', test_split=0.1)

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
model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
