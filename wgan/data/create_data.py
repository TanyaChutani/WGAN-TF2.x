import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def image_preprocessing(images):
  image = tf.io.read_file(images)
  image = tf.io.decode_image(image,channels=3)
  image.set_shape([None,None,3])
  image = tf.image.resize(image,[64,64])
  image = tf.image.central_crop(image,1.0)
  image = tf.cast(image,tf.float32)/127.5 - 1.0
  image = tf.cast(image,tf.float32)
  label = -tf.ones((1))
  return image,label


def real_data_generator(images,batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(images)
  dataset = dataset.map(image_preprocessing,tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size//2)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset