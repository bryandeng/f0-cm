#!/usr/bin/env python3

from keras.layers import Dense, Dropout
from keras.models import Sequential

from speech_data import load_data

(X_train, y_train), (X_test, y_test) = load_data('martin', test_split=0.1)

model = Sequential()
model.add(Dense(128, input_dim=14, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse',
              optimizer='sgd',
              metrics=["accuracy"])
model.fit(X_train, y_train,
          nb_epoch=10,
          batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
