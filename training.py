import os
import csv

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Path Setting
ROOT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(ROOT_DIR, 'IMG')
CSV_FILE = os.path.join(ROOT_DIR, 'driving_log.csv')

# Read csv file
lines = []
with open(CSV_FILE, 'r') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append(line)

# Save the training data
header = lines.pop(0)
x_data = []
y_data = []
for line in lines:
  x_data.append(os.path.join(IMAGES_DIR, line[0].split('/')[-1]))
  y_data.append(float(line[3]))


# Shuffle, and train / validation set
x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)


# Generator
def generator(x, y, batch_size):
  batch_size = int(batch_size / 2.)
  print('batch', batch_size)
  while True:
    for start in range(0, len(x), batch_size):
      end = start + batch_size
      x_batch, y_batch = [], []
      for img_path, steer in zip(x_train[start:end], y[start:end]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_batch.append(img)
        y_batch.append(steer)
        img_flip = cv2.flip(img, 1)
        x_batch.append(img_flip)
        y_batch.append(-steer)
      x_batch = np.array(x_batch) / 255.
      y_batch = np.array(y_batch)
      yield x_batch, y_batch


def main():
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
  model.compile(optimizer='adam', loss='mse')
  model.summary()
  print("")

  # model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2, shuffle=True, verbose=2)

  # Add callback
  model.fit_generator(generator=generator(x_train, y_train, 64),
                      steps_per_epoch=int(np.ceil(len(x_train)*2/64)), epochs=3,
                      verbose=1,
                      validation_data=generator(x_valid, y_valid, 64),
                      validation_steps=int(np.ceil(len(x_valid)*2/64)))
  model.save('model.h5')


if __name__ == '__main__':
  main()
