import logging
import os
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src.unet import custom_objects
from src.unet.datasets import floorplans
from src.unet.unet import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

def main():
    # unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects)
    unet_model = tf.keras.models.load_model('unet_pp_model', custom_objects=custom_objects)

    predict = True
    if predict:
        single = mpimg.imread('resources/single.jpg')
        multi = mpimg.imread('resources/multi.jpg')
        # image = mpimg.imread('resources/multi_large.jpg')
        # image = mpimg.imread('resources/multi_largest.jpg')
        m_sampled = mpimg.imread('resources/m_sampled.jpg')
        m_sampled2 = mpimg.imread('resources/m_sampled2.jpg')
        mplan_s = mpimg.imread('resources/mplan_s.jpg')

        images = [single, multi, m_sampled2, m_sampled, mplan_s]
        for i, image in enumerate(images):
            shp = image.shape
            image = tf.convert_to_tensor(image, dtype=tf.uint8)
            # img = tf.image.resize(img, [size, size])
            image = tf.cast(image, dtype=tf.float32)
            image = tf.reshape(image, [-1, shp[0], shp[1], 3]) / 255

            prediction = unet_model.predict(image)
            result = prediction[0].argmax(axis=-1)

            result[result == 1] = 10
            result[result == 0] = 20
            result[result == 2] = 30

            timestr = time.strftime("%Y%m%d-%H%M%S")
            mpimg.imsave("result" + timestr + str(i) + ".jpg", result.astype(np.uint8))
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
