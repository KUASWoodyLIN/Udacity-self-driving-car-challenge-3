import os
import csv

import cv2
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten

ROOT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(ROOT_DIR, 'IMG')
CSV_FILE = os.path.join(ROOT_DIR, 'driving_log.csv')

lines = []
with open(CSV_FILE, 'r') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  file_path = os.path.join(IMAGES_DIR, line[0].split('/')[-1])
  img = cv2.imread(file_path)
  images.append(img)
  measurements.append(float(line[3]))


x_train = np.array(images)
y_train = np.array(measurements)




print()