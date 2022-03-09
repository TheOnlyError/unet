import logging
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import losses, metrics
from tensorflow.keras.optimizers import Adam

import src.segmentation_models as sm
from src import unet
from src.unet.datasets import floorplans

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


os.environ['TF_CUDNN_DETERMINISTIC']='1'

logging.disable(logging.WARNING)


def main():
    # Initialize seeds
    seed = int(random.random() * 1000)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: {0}".format(seed))

    sm.set_framework('tf.keras')

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

    LEARNING_RATE = 1e-4
    # unet_model = sm.Xnet(backbone_name='vgg16', classes=3)
    unet_model = sm.Xnet(backbone_name='efficientnetb5', classes=3)

    unet_model.compile(loss=losses.SparseCategoricalCrossentropy(),
                       optimizer=Adam(learning_rate=LEARNING_RATE),
                       metrics=[metrics.SparseCategoricalAccuracy()],
                       )

    # unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects) # 160 + 80
    # unet_model = tf.keras.models.load_model('unet_pp_model') # 80

    train_dataset, validation_dataset = floorplans.load_data(normalize=False)

    trainer = unet.Trainer(checkpoint_callback=True)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=80,
                batch_size=1,
                verbose=2)

    unet_model.save("unet_pp_model")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
