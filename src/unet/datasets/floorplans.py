from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

size = 512
IMAGE_SIZE = (512, 512)
channels = 3
classes = 3


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image_train(x):
    image = tf.io.decode_raw(x['image'], tf.uint8)
    mask = tf.io.decode_raw(x['mask'], tf.uint8)

    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    # label = tf.cast(mask, tf.uint8)

    input_image, input_mask = normalize(image, mask)
    return input_image, input_mask

    # print(datapoint['image'].shape)
    # input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    # input_mask = tf.image.resize(datapoint['mask'], IMAGE_SIZE)
    #
    # input_image, input_mask = normalize(input_image, input_mask)
    #
    # return input_image, input_mask


def load_data(buffer_size=400, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = loadDataset().shuffle(buffer_size)
    DATASET_SIZE = len(list(dataset))
    print(DATASET_SIZE)
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    #
    # train = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # test = test_dataset.map(load_image_train)
    # train_dataset = train.cache().shuffle(buffer_size).take(train_size)
    return train_dataset, test_dataset


def _parse_function(example_proto):
    feature = {'image': tf.io.FixedLenFeature([], tf.string),
               'mask': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature)

    image, mask = decodeAllRaw(example)
    return preprocess(image, mask)


def decodeAllRaw(x):
    image = tf.io.decode_raw(x['image'], tf.uint8)
    mask = tf.io.decode_raw(x['mask'], tf.uint8)
    return image, mask


def preprocess(img, mask, size=1024):  # 1024
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [size, size, 3]) / 255
    mask = tf.reshape(mask, [size, size, 1])
    return img, mask


def loadDataset():
    # raw_dataset = tf.data.TFRecordDataset('../../../data.tfrecords')
    raw_dataset = tf.data.TFRecordDataset('data.tfrecords')
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset = load_data()
    rows = 10
    fig, axs = plt.subplots(rows, 2, figsize=(8, 30))
    for ax, (image, mask) in zip(axs, train_dataset.take(rows).batch(1)):
        ax[0].matshow(np.array(image[0]).squeeze())
        ax[1].matshow(np.array(mask[0]).squeeze())
    plt.show()
