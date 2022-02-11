import logging
import os
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src.unet import custom_objects
from src.unet.datasets import circles, floorplans
from src.unet.unet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

def main():
    unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects)

    predict = True
    if predict:
        # image = mpimg.imread('resources/single.jpg')
        # image = mpimg.imread('resources/multi_large.jpg')
        # image = mpimg.imread('resources/multi_largest.jpg')
        image = mpimg.imread('resources/m_sampled.jpg')
        # image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        prediction = unet_model.predict(image)
        result = prediction[0].argmax(axis=-1)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        mpimg.imsave("result" + timestr + ".jpg", result.astype(np.uint8))
    else:
        train_dataset, validation_dataset = floorplans.load_data()
        rows = 3
        for t in range(3):
            fig, axs = plt.subplots(rows, 3, figsize=(8, rows*3))
            for ax, (image, label) in zip(axs, validation_dataset.shuffle(64).take(rows).batch(1)):
                prediction = unet_model.predict(image)
                p = prediction[0].argmax(axis=-1)
                # p *= 2
                # p -= 2
                # p[p < 0] = 1
                ax[0].matshow(np.array(image[0]).squeeze())
                ax[1].matshow(label[0, ..., 0])
                ax[2].matshow(p)

                ax[0].axis('off')
                ax[1].axis('off')
                ax[2].axis('off')

            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.show()
            # plt.savefig(timestr)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
