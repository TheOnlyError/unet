import logging
import os
import time

from src import unet
from src.unet.datasets import floorplans, circles

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def main():
    # train_dataset, validation_dataset = loadDataset()
    train_dataset, validation_dataset = floorplans.load_data(100)
    # train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200, splits=(0.7, 0.3))

    # print(train_dataset)

    channels = 3
    classes = 1
    LEARNING_RATE = 1e-3
    unet_model = unet.build_model(channels=channels,
                                  num_classes=classes,
                                  layer_depth=3,
                                  filters_root=64)
    unet.finalize_model(unet_model, learning_rate=LEARNING_RATE)

    trainer = unet.Trainer(checkpoint_callback=False,
                           learning_rate_scheduler=unet.SchedulerType.WARMUP_LINEAR_DECAY,
                           warmup_proportion=0.1,
                           learning_rate=LEARNING_RATE)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=10,
                batch_size=1)

    unet_model.save("unet_model")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
