import logging
import os
import time

from src import unet
import tensorflow as tf

from src.unet.datasets import circles

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'mask':tf.io.FixedLenFeature([],tf.string)}
    return tf.io.parse_single_example(example_proto,feature)

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

    channels = 1
    classes = 2
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
                epochs=60,
                batch_size=1)

    unet_model.save("unet_model")



if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} seconds'.format((toc - tic)))