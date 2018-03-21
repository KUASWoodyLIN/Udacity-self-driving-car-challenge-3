import os
import csv
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Cropping2D, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard

# Path Setting
ROOT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(ROOT_DIR, 'IMG')
CSV_FILE = os.path.join(ROOT_DIR, 'driving_log.csv')
LOGS_PATH = os.path.join(ROOT_DIR, 'logs')

# Learning parameter
BATCH_SIZE = 64

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
  # image path
  img_center = os.path.join(IMAGES_DIR, line[0].split('/')[-1])
  img_left = os.path.join(IMAGES_DIR, line[1].split('/')[-1])
  img_right = os.path.join(IMAGES_DIR, line[2].split('/')[-1])
  x_data.append(img_center)
  x_data.append(img_left)
  x_data.append(img_right)

  # steering angle
  steering_center = float(line[3])
  correction = 0.25  # this is a parameter to tune
  steering_left = steering_center + correction
  steering_right = steering_center - correction
  y_data.append(steering_center)
  y_data.append(steering_left)
  y_data.append(steering_right)


# Shuffle, and train / validation set
x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.15, shuffle=True)


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
      x_batch = np.array(x_batch)
      y_batch = np.array(y_batch)
      yield shuffle(x_batch, y_batch)


# Data preprocessing
def valid_proc(x, y):
  x_out = []
  y_out = []
  for img_path, steer in zip(x, y):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_out.append(img)
    y_out.append(steer)
    img_flip = cv2.flip(img, 1)
    x_out.append(img_flip)
    y_out.append(-steer)
  x_out = np.array(x_out)
  y_out = np.array(y_out)
  return shuffle(x_out, y_out)

# only run this if you don't use valid_generator
x_valid, y_valid = valid_proc(x_valid, y_valid)


def main():
  x_input = Input(shape=(160, 320, 3), name='x_input')
  crop = Cropping2D(((55,20),(0,0)))(x_input)
  normalize = Lambda(lambda x: (x /255.0) - 0.5)(crop)

  # LeNet
  #conv1 = Conv2D(6, (5, 5), padding='same', activation='relu')(normalize)
  #pool1 = MaxPooling2D()(conv1)
  #conv2 = Conv2D(6, (5, 5), padding='same', activation='relu')(pool1)
  #pool2 = MaxPooling2D()(conv2)
  #flat = Flatten()(pool2)
  #hedden1 = Dense(120)(flat)
  #hedden2 = Dense(84)(hedden1)
  #output = Dense(1)(hedden2)


  # VGG-16
  # conv1 = Conv2D(32, (3,3), padding='same', activation='relu')(x_input)
  # conv2 = Conv2D(32, (3,3), padding='same', activation='relu')(conv1)
  # pool1 = MaxPool2D()(conv2)
  # conv3 = Conv2D(64, (3,3), padding='same', activation='relu')(pool1)
  # conv4 = Conv2D(64, (3,3), padding='same', activation='relu')(conv3)
  # pool2 = MaxPool2D()(conv4)
  # conv5 = Conv2D(128, (3,3), padding='same', activation='relu')(pool2)
  # pool3 = MaxPool2D()(conv5)
  # flat = Flatten()(pool3)
  # hidden1 = Dense(256)(flat)
  # drop1 = Dropout(0.5)(hidden1)
  # hidden2 = Dense(128)(drop1)
  # drop2 = Dropout(0.5)(hidden2)
  # hidden3 = Dense(32)(drop2)
  # drop3 = Dropout(0.5)(hidden3)
  # output = Dense(1)(drop3)

  # Nvidia
  conv1 = Conv2D(24, (5, 5), padding='same', activation='relu')(normalize)
  pool1 = MaxPooling2D()(conv1)
  conv2 = Conv2D(36, (5, 5), padding='same', activation='relu')(pool1)
  pool2 = MaxPooling2D()(conv2)
  conv3 = Conv2D(48, (5, 5), padding='same', activation='relu')(pool2)
  pool3 = MaxPooling2D()(conv3)
  conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool3)
  pool4 = MaxPooling2D()(conv4)
  conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool4)
  pool5 = MaxPooling2D()(conv5)
  flat = Flatten()(pool5)
  hidden1 = Dense(100)(flat)
  drop1 = Dropout(0.5)(hidden1)
  hidden2 = Dense(50)(drop1)
  drop2 = Dropout(0.5)(hidden2)
  output = Dense(1)(drop2)


  model = Model(x_input, output)
  model.compile(optimizer='adam', loss='mse')
  model.summary()
  print("")

  # automatic stop learning when model have not improve
  early_stop = EarlyStopping(monitor='val_loss',
                             min_delta=0.00005,
                             patience=4,
                             mode='min',
                             verbose=1)

  # save the model automatic
  checkpoint = ModelCheckpoint('model.h5',
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min',
                               period=1)

#  not use for this project
#  embedding_layer_names = set(layer.name
#                              for layer in model.layers
#                              if layer.name.startswith('dense_') or layer.name.startswith('conv2d_'))

  # Tensorboard
  log_file_name = 'Driving_Car_' + str(len(glob(LOGS_PATH + '/Driving_Car_*')) + 1)
  tensorboard = TensorBoard(log_dir='./logs/' + log_file_name,
                            histogram_freq=10,
                            batch_size=BATCH_SIZE,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,)
                            #embeddings_freq=10,
                            #embeddings_layer_names=embedding_layer_names,
                            #embeddings_metadata=None)


  print('train: {}, valid: {}'.format(len(x_train), len(x_valid)))
  history_object = model.fit_generator(generator=generator(x_train, y_train, 64),
                                       steps_per_epoch=int(np.ceil(len(x_train)*2/64)),
                                       epochs=20,
                                       verbose=1,
                                       validation_data=(x_valid, y_valid),
                                       callbacks=[early_stop, checkpoint, tensorboard])


#  train_generator = generator(x_train, y_train, BATCH_SIZE)
#  valid_generator = generator(x_valid, y_valid, BATCH_SIZE)

#  history_object = model.fit_generator(generator=train_generator,
#                      steps_per_epoch=np.ceil(len(x_train)*2/BATCH_SIZE),
#                      epochs=20,
#                      validation_data=valid_generator,
#		      validation_steps=np.ceil(len(x_valid)*2/BATCH_SIZE),
#                      callbacks=[early_stop, checkpoint, tensorboard])


if __name__ == '__main__':
  main()
