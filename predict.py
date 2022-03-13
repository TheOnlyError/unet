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

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def pad_to(x, stride):
    h, w = x.shape[:2]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = tf.constant([[lh, uh], [lw, uw], [0, 0]])

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = tf.pad(x, pads, "CONSTANT", 255)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]:-pad[1]]
    return x


def main():
    plusplus = True
    if plusplus:
        unet_model = tf.keras.models.load_model('unet2_model', custom_objects=custom_objects)
        # unet_model = tf.keras.models.load_model('unet_pp_model', custom_objects=custom_objects)
    else:
        unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects)

    predict = True
    if predict:
        single = mpimg.imread('resources/single.jpg')
        multi = mpimg.imread('resources/multi.jpg')
        # # image = mpimg.imread('resources/multi_large.jpg')
        # # image = mpimg.imread('resources/multi_largest.jpg')
        # m_sampled = mpimg.imread('resources/m_sampled.jpg')
        m_sampled2 = mpimg.imread('resources/m_sampled2.jpg')
        mplan_s = mpimg.imread('resources/mplan_s.jpg')
        # gv = mpimg.imread('resources/gv2.jpg')

        images = [single, multi, m_sampled2, mplan_s]
        # images = [single]
        for i, image in enumerate(images):
            shp = image.shape
            image = tf.convert_to_tensor(image, dtype=tf.uint8)

            if plusplus:
                image, pads = pad_to(image, 32)
                shp = image.shape
            image = tf.cast(image, dtype=tf.float32)
            # image = tf.reshape(image, [-1, shp[0], shp[1], 3]) / 255  # Only divide if not effnet
            image = tf.reshape(image, [-1, shp[0], shp[1], 3])

            prediction = unet_model.predict(image)
            result = prediction[0].argmax(axis=-1)

            result[result == 1] = 10
            result[result == 0] = 20
            result[result == 2] = 30

            timestr = time.strftime("%Y%m%d-%H%M%S")
            mpimg.imsave("result" + timestr + str(i) + ".jpg", result.astype(np.uint8))
    else:
        train_dataset, validation_dataset = floorplans.load_data(normalize=True)
        rows = 3
        for t in range(3):
            fig, axs = plt.subplots(rows, 3, figsize=(8, rows * 3))
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
            plt.savefig(timestr)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
