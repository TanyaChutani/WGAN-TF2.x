import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Critic(tf.keras.models.Model):
  def __init__(self,clip_value):
    super(Critic,self).__init__()
    self.clip_value = clip_value
    self.weight_constraint = tf.keras.constraints.MinMaxNorm(min_value = -self.clip_value,max_value= self.clip_value)
    self.weight_init = tf.keras.initializers.RandomNormal()
    self.conv_layer1 = tf.keras.layers.Conv2D(filters=8,
                                             kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                             kernel_constraint = self.weight_constraint)
    self.conv_layer2 = tf.keras.layers.Conv2D(filters=16,
                                          kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                          kernel_constraint = self.weight_constraint)
    self.conv_layer3 = tf.keras.layers.Conv2D(filters=32,
                                          kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                          kernel_constraint = self.weight_constraint)
    self.conv_layer4 = tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                          kernel_constraint = self.weight_constraint)
    self.conv_layer5 = tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                          kernel_constraint = self.weight_constraint)
    self.conv_layer6 = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init,
                                          kernel_constraint = self.weight_constraint)

    self.bn_layer = tf.keras.layers.BatchNormalization()
    self.lrelu_layer = tf.keras.layers.LeakyReLU(0.2)
    self.flatten_layer = tf.keras.layers.Flatten()
    self.avg_pooling_layer = tf.keras.layers.AveragePooling2D()
    self.dense_layer = tf.keras.layers.Dense(1,activation='linear')
  
  def call(self,input_tensor):
    x = self.conv_layer1(input_tensor)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.conv_layer2(x)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.conv_layer3(x)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.conv_layer4(x)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.conv_layer5(x)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.conv_layer6(x)
    x = self.lrelu_layer(x)
    x = self.avg_pooling_layer(x)
    x = self.flatten_layer(x)
    x = self.dense_layer(x)
    return x


class Generator(tf.keras.models.Model):
  def __init__(self):
    super(Generator,self).__init__()
    self.weight_init = tf.keras.initializers.RandomNormal()
    self.dense_layer = tf.keras.layers.Dense(4096)
    self.lrelu_layer = tf.keras.layers.LeakyReLU(0.2)
    self.reshape_layer = tf.keras.layers.Reshape(target_shape=[1,1,4096])
    self.conv_transpose_layer = tf.keras.layers.Conv2DTranspose(256,4,
                                                                kernel_initializer=self.weight_init)
    self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.7)
    self.upsample_layer = tf.keras.layers.UpSampling2D()
    self.tanh_layer = tf.keras.layers.Activation(tf.nn.tanh)
    self.conv_layer1 = tf.keras.layers.Conv2D(filters=256,
                                      kernel_size=(4,4),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer2 = tf.keras.layers.Conv2D(filters=128,
                                      kernel_size=(4,4),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer3 = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer4 = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer5 = tf.keras.layers.Conv2D(filters=16,
                                      kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer6 = tf.keras.layers.Conv2D(filters=8,
                                      kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init)
    self.conv_layer7 = tf.keras.layers.Conv2D(filters=3,
                                      kernel_size=(3,3),padding='same',kernel_initializer=self.weight_init)
    
  def call(self,input_tensor):
    x = self.dense_layer(input_tensor)
    x = self.reshape_layer(x)
    x = self.conv_transpose_layer(x)
    x = self.lrelu_layer(x)
    x = self.conv_layer1(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer2(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer3(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer4(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer5(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer6(x)
    x = self.lrelu_layer(x)
    x = self.upsample_layer(x)
    x = self.conv_layer7(x)
    x = self.tanh_layer(x)
    return x
