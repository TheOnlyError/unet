import logging
import os
import time

from src import unet

from tensorflow import keras
from tensorflow import losses, metrics
from tensorflow.keras.optimizers import Adam

import src.segmentation_models as sm
from src.unet.datasets import floorplans

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.disable(logging.WARNING)


def main():
    sm.set_framework('tf.keras')
    keras.backend.set_image_data_format('channels_last')

    LEARNING_RATE = 1e-4
    unet_model = sm.Unet(backbone_name='efficientnetb1', classes=3, activation='sigmoid')

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
                epochs=120,
                batch_size=1,
                verbose=2)

    unet_model.save("unet2_model")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
