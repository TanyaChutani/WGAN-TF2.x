import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def image_preprocessing(images):
  image = images['image']
  image = tf.image.random_crop(image,size=(256,256,3))
  image = tf.cast(image,tf.float32)/127.5 - 1.0
  image = tf.cast(image,tf.float32)
  label = -tf.ones((1))

  return image,label


def real_data_generator(images,batch_size):
  dataset = images.map(image_preprocessing,tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size//2)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset