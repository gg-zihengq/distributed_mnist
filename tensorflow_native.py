'''Adapted from tensorflow/docs/site/en/tutorials/distribute/multi_worker_with_keras.ipynb .'''
import argparse
import json
import os
import sys
import tensorflow as tf
import numpy as np
import time

os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

parser = argparse.ArgumentParser()
parser.add_argument("--hosts", nargs='+', default=['localhost'])
parser.add_argument("--host-rank", type=int, default=0)
parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )

parser = parser.parse_args()

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
      tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

tf_config = {
    'cluster': {
        'worker': [host+':12345' for host in parser.hosts]
    },
    'task': {'type': 'worker', 'index': parser.host_rank}
}
# os.environ["TF_CONFIG"]=json.dumps(tf_config)

strategy = tf.distribute.MultiWorkerMirroredStrategy()

# tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = parser.batch_size
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

start = time.time()
multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=500//num_workers)
end = time.time()
print('train 10 epochs take:'+str(end-start))
