import time

from src import unet
import tensorflow as tf

def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'mask':tf.io.FixedLenFeature([],tf.string)}
    return tf.io.parse_single_example(example_proto,feature)

def loadDataset():
    d1 = tf.data.TFRecordDataset('trainfull_norooms_unet.tfrecords')
    full_dataset = d1.map(_parse_function)
    full_dataset = full_dataset.shuffle(buffer_size=32)

    DATASET_SIZE = len(list(full_dataset))
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.3 * DATASET_SIZE)

    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size).take(val_size)
    return train_dataset, val_dataset

def main():
    train_dataset, validation_dataset = loadDataset()

    channels = 1
    classes = 3
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
                epochs=5,
                batch_size=1)

    unet_model.save("unet_model")



if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))