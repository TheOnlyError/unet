import logging
import os
import time

import matplotlib.pyplot as plt

from src.unet import custom_objects
from src.unet.datasets import circles, floorplans
from src.unet.unet import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

def main():
    train_dataset, validation_dataset = floorplans.load_data()

    unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects)

    rows = 10
    fig, axs = plt.subplots(rows, 3, figsize=(8, 30))
    for ax, (image, label) in zip(axs, train_dataset.take(rows).batch(1)):
        prediction = unet_model.predict(image)
        ax[0].matshow(np.array(image[0]).squeeze())
        ax[1].matshow(label[0, ..., 0], cmap="gray")
        ax[2].matshow(prediction[0].argmax(axis=-1), cmap="gray")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.show()
    # plt.savefig(timestr)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
