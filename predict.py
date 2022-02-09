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
    return tf.io.parse_single_example(example_proto, feature)


def loadDataset():
    d1 = tf.data.TFRecordDataset('trainfull_norooms_unet.tfrecords')
    full_dataset = d1.map(_parse_function)
    full_dataset = full_dataset.shuffle(buffer_size=32)

    DATASET_SIZE = len(list(full_dataset))
    print(DATASET_SIZE)
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.3 * DATASET_SIZE)

    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    return train_dataset, val_dataset
    # return full_dataset, full_dataset


def main():
    # train_dataset, validation_dataset = loadDataset()
    train_dataset, validation_dataset = circles.load_data(100, nx=172, ny=172, splits=(0.8, 0.2))

    unet_model = tf.keras.models.load_model('unet_model', custom_objects={'mean_iou': unet.metrics.mean_iou,
                                                                          'dice_coefficient': unet.metrics.dice_coefficient,
                                                                          'auc': tf.keras.metrics.AUC()})

    rows = 10
    fig, axs = plt.subplots(rows, 3, figsize=(8, 30))
    for ax, (image, label) in zip(axs, train_dataset.take(rows).batch(1)):
        prediction = unet_model.predict(image)
        ax[0].matshow(np.array(image[0]).squeeze())
        ax[1].matshow(label[0, ..., 0], cmap="gray")
        ax[2].matshow(prediction[0].argmax(axis=-1), cmap="gray")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.show()
    plt.savefig(timestr)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
