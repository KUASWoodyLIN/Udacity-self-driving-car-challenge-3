import os
import csv

import numpy as np

ROOT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(ROOT_DIR, 'IMG')
CSV_FILE = os.path.join(ROOT_DIR, 'driving_log.csv')

lines = []
with open(CSV_FILE, 'r') as f:
  reader = csv.reader(f)
  for line in reader:
    lines.append(line)

header = lines.pop(0)
x_train = []
y_train = []
for line in lines:
  x_train.append(os.path.join(IMAGES_DIR, line[0].split('/')[-1]))
  y_train.append(float(line[3]))


def train_generator(batch_size):
  batch_size = int(batch_size / 2.)
  print('batch' ,batch_size)
  while True:
    for start in range(0, len(x_train), batch_size):
      end = start + batch_size
      x_batch, y_batch = x_train[start:end], y_train[start:end]
      # for i in i_train_batch:
      #   x_batch.append(process_wav_file(i))
      # x_batch = np.expand_dims(np.array(x_batch), 4)
      print('start: {}  end: {}'.format(start, end))
      yield x_batch, y_batch


if __name__ == '__main__':
  a = train_generator(32)
  x, y = next(a)
  x, y = next(a)