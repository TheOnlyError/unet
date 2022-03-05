import logging
import os
import time

import tensorflow as tf

from tensorflow import losses, metrics

from src import unet, xnet
from src.unet.datasets import floorplans
from tensorflow.keras.optimizers import Adam

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def main():
    LEARNING_RATE = 1e-4
    unet_model = xnet.Xnet(backbone_name='efficientnetb5', classes=3)

    unet_model.compile(loss=losses.SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  )

    # unet_model = tf.keras.models.load_model('unet_model', custom_objects=custom_objects) # 160 + 80
    # unet_model = tf.keras.models.load_model('unet_pp_model') # 80

    train_dataset, validation_dataset = floorplans.load_data()

    trainer = unet.Trainer(checkpoint_callback=False)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=120,
                batch_size=1,
                verbose=2)

    unet_model.save("unet_pp_model")

if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
