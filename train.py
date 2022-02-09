import logging
import os
import time

from tensorflow import losses, metrics

from src import unet, mrcnn
from src.unet.datasets import floorplans, circles, oxford_iiit_pet

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def main():
    # train_dataset, validation_dataset = loadDataset()
    train_dataset, validation_dataset = floorplans.load_data()
    # train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200, splits=(0.7, 0.3))
    # train_dataset, validation_dataset = oxford_iiit_pet.load_data()

    # print(train_dataset)

    channels = 3
    classes = 3
    LEARNING_RATE = 1e-4

    train_unet = False
    if train_unet:
        model = unet.build_model(channels=channels,
                                      num_classes=classes,
                                      layer_depth=5,
                                      filters_root=64,
                                      padding="same")
        name = 'unet_model'
    else:
        class SimpleConfig(mrcnn.config.Config):
            # Give the configuration a recognizable name
            NAME = "coco_inference"

            # set the number of GPUs to use along with the number of images per GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            NUM_CLASSES = classes

        name = 'mrcnn_model'

        model = mrcnn.model.MaskRCNN(mode="training",
                                     config=SimpleConfig(),
                                     model_dir=os.getcwd())

    unet.finalize_model(model,
                        loss=losses.SparseCategoricalCrossentropy(),
                        metrics=[metrics.SparseCategoricalAccuracy()],
                        auc=False,
                        learning_rate=LEARNING_RATE)

    trainer = unet.Trainer(checkpoint_callback=False)
    trainer.fit(model,
                train_dataset,
                validation_dataset,
                epochs=60,
                batch_size=1)
    model.save(name)

if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
