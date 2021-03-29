import tensorflow as tf
import tensorflow_datasets as tfds
from .augs import normalize_img, flip_lr, color_augs


__all__ = ['GetVoc']


# Reference: https://www.tensorflow.org/datasets/catalog/voc
class GetVoc:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_train_ds(self, shuffle=True, drop_remainder=True):
        # Training Dataset (voc2007 trainval + 2012 trainval)
        (voc2007_train, voc2007_val), voc2007_info = tfds.load(name='voc/2007', split=['train', 'validation'], with_info=True)
        (voc2012_train, voc2012_val), voc2012_info = tfds.load(name='voc/2012', split=['train', 'validation'], with_info=True)
        train_ds = voc2007_train.concatenate(voc2007_val).concatenate(voc2012_train).concatenate(voc2012_val)
        train_ds_num_examples = voc2007_info.splits['train'].num_examples + \
            voc2007_info.splits['validation'].num_examples + \
            voc2012_info.splits['train'].num_examples + \
            voc2012_info.splits['validation'].num_examples

        if shuffle:
            train_ds = train_ds.shuffle(train_ds_num_examples)
        
        train_ds = train_ds.map(normalize_img, num_parallel_calls=self.autotune)
        train_ds = train_ds.map(flip_lr, num_parallel_calls=self.autotune)
        train_ds = train_ds.map(color_augs, num_parallel_calls=self.autotune)
        train_ds = train_ds.cache()  # Loaded data first time, it's going to keep track of some of them in memory. It makes faster
        train_ds = train_ds.padded_batch(self.batch_size, drop_remainder=drop_remainder)
        train_ds = train_ds.prefetch(self.autotune)  # While running on gpu, it's going to prefetch number of batch_size examples, so they are ready to be run instantly after the gpu calls are done
        return train_ds

    def get_val_ds(self):
        (val_ds,) = tfds.load(name='voc/2007', split=['test'], with_info=False)
        val_ds = val_ds.map(normalize_img, num_parallel_calls=self.autotune)
        val_ds = val_ds.padded_batch(self.batch_size)
        val_ds = val_ds.prefetch(self.autotune)
        return val_ds
    