from typing import Tuple

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


def load_data(buffer_size=32, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = loadDataset()
    DATASET_SIZE = len(list(dataset))
    print(DATASET_SIZE)
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.3 * DATASET_SIZE)

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
    image = tf.io.decode_raw(example['image'], tf.uint8)
    mask = tf.io.decode_raw(example['mask'], tf.uint8)

    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, [size, size, 3]) / 255
    mask = tf.reshape(mask, [size, size, 1])

    return image, mask


def decodeAllRaw(x):
    image = tf.io.decode_raw(x['image'], tf.uint8)
    mask = tf.io.decode_raw(x['mask'], tf.uint8)
    return image, mask


def preprocess(img, mask, size=512):  # 1024
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, size, size, 3]) / 255
    mask = tf.reshape(mask, [-1, size, size])
    return img, mask


def loadDataset(size=512):
    raw_dataset = tf.data.TFRecordDataset('trainfull_norooms_unet.tfrecords')
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
