import logging
import os
import time

import matplotlib.pyplot as plt

from src.unet.datasets import circles
from src.unet.unet import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def _parse_function(example_proto):
    feature = {'image': tf.io.FixedLenFeature([], tf.string),
               'mask': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    mask = tf.io.decode_raw(example['mask'], tf.uint8)

    return image, mask


def loadDataset():
    raw_dataset = tf.data.TFRecordDataset('trainfull_norooms_unet.tfrecords')
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset, parsed_dataset


def main():
    train_dataset, validation_dataset = loadDataset()


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
