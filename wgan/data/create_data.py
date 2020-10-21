import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
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
=======


def image_preprocessing(image,label):
  label = -tf.ones((1))
  image = tf.expand_dims(image,-1)
  image = tf.cast(image,tf.float32)/127.5 - 1.0
  image = tf.cast(image,tf.float32)
  return image,label

def real_data_generator(x,y,batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((x,y))
  dataset = dataset.map(image_preprocessing,tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size//2)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
>>>>>>> 90f70924f956480349e2b93541a0175527e734f2
