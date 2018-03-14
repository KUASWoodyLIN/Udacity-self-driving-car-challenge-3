import os
import csv

import cv2
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout




x_train = np.array(images) / 255.
y_train = np.array(measurements)

x_input = Input(shape=(160, 320, 3), name='x_input')

conv1 = Conv2D(32, (3,3), padding='same', activation='relu')(x_input)
conv2 = Conv2D(32, (3,3), padding='same', activation='relu')(conv1)
pool1 = MaxPool2D()(conv2)

conv3 = Conv2D(64, (3,3), padding='same', activation='relu')(pool1)
conv4 = Conv2D(64, (3,3), padding='same', activation='relu')(conv3)
pool2 = MaxPool2D()(conv4)

conv5 = Conv2D(128, (3,3), padding='same', activation='relu')(pool2)
pool3 = MaxPool2D()(conv5)

flat = Flatten()(pool3)

hidden1 = Dense(256)(flat)
drop1 = Dropout(0.5)(hidden1)

hidden2 = Dense(128)(drop1)
drop2 = Dropout(0.5)(hidden2)

hidden3 = Dense(32)(drop2)
drop3 = Dropout(0.5)(hidden3)

output = Dense(1)(drop3)

model = Model(x_input, output)
model.compile(optimizer='adam', loss='mae')
model.summary()
print("")

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
model.fit_generator()
model.save('model.h5')
