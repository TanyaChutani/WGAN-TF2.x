import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Critic(tf.keras.models.Model):
    def __init__(self, clip_value):
        super(Critic, self).__init__()
        self.clip_value = clip_value
        self.weight_constraint = tf.keras.constraints.MinMaxNorm(min_value=-self.clip_value, max_value=self.clip_value)
        self.weight_init = tf.keras.initializers.RandomNormal()
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(4, 4), strides=2, padding='same',
                                                  kernel_initializer=self.weight_init,
                                                  kernel_constraint=self.weight_constraint)
        self.bn_layer1 = tf.keras.layers.BatchNormalization()

        self.conv_layer2 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(4, 4), strides=2, padding='same',
                                                  kernel_initializer=self.weight_init,
                                                  kernel_constraint=self.weight_constraint)
        self.bn_layer2 = tf.keras.layers.BatchNormalization()

        self.conv_layer3 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(4, 4), strides=2, padding='same',
                                                  kernel_initializer=self.weight_init,
                                                  kernel_constraint=self.weight_constraint)
        self.bn_layer3 = tf.keras.layers.BatchNormalization()

        self.conv_layer4 = tf.keras.layers.Conv2D(filters=512,
                                                  kernel_size=(4, 4), strides=2, padding='same',
                                                  kernel_initializer=self.weight_init,
                                                  kernel_constraint=self.weight_constraint)
        self.bn_layer4 = tf.keras.layers.BatchNormalization()

        self.conv_layer5 = tf.keras.layers.Conv2D(filters=1,
                                                  kernel_size=(4, 4), kernel_initializer=self.weight_init,
                                                  kernel_constraint=self.weight_constraint)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(1, activation='linear')
        self.lrelu_layer = tf.keras.layers.LeakyReLU(0.2)

    def call(self, input_tensor, training=None):
        x = self.conv_layer1(input_tensor)
        x = self.bn_layer1(x, training=training)
        x = self.lrelu_layer(x)
        x = self.conv_layer2(x)
        x = self.bn_layer2(x, training=training)
        x = self.lrelu_layer(x)
        x = self.conv_layer3(x)
        x = self.bn_layer3(x, training=training)
        x = self.lrelu_layer(x)
        x = self.conv_layer4(x)
        x = self.bn_layer4(x, training=training)
        x = self.lrelu_layer(x)
        x = self.conv_layer5(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x


class Generator(tf.keras.models.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.weight_init = tf.keras.initializers.RandomNormal()
        self.reshape_layer = tf.keras.layers.Reshape((1, 1, 100))
        self.conv_transpose_layer1 = tf.keras.layers.Conv2DTranspose(512, 4,
                                                                     kernel_initializer=self.weight_init)
        self.bn_layer1 = tf.keras.layers.BatchNormalization(momentum=0.7)
        self.conv_transpose_layer2 = tf.keras.layers.Conv2DTranspose(256, 4, 2, padding='same',
                                                                     kernel_initializer=self.weight_init)
        self.bn_layer2 = tf.keras.layers.BatchNormalization(momentum=0.7)
        self.conv_transpose_layer3 = tf.keras.layers.Conv2DTranspose(128, 4, 2, padding='same',
                                                                     kernel_initializer=self.weight_init)
        self.bn_layer3 = tf.keras.layers.BatchNormalization(momentum=0.7)
        self.conv_transpose_layer4 = tf.keras.layers.Conv2DTranspose(64, 4, 2, padding='same',
                                                                     kernel_initializer=self.weight_init)
        self.bn_layer4 = tf.keras.layers.BatchNormalization(momentum=0.7)
        self.conv_transpose_layer5 = tf.keras.layers.Conv2DTranspose(3, 4, 2, padding='same',
                                                                     kernel_initializer=self.weight_init)

        self.tanh_layer = tf.keras.layers.Activation(tf.nn.tanh)
        self.relu_layer = tf.keras.layers.Activation(tf.nn.relu)

    def call(self, input_tensor, training=None):
        x = self.reshape_layer(input_tensor)
        x = self.conv_transpose_layer1(x)
        x = self.bn_layer1(x, training=training)
        x = self.relu_layer(x)
        x = self.conv_transpose_layer2(x)
        x = self.bn_layer2(x, training=training)
        x = self.relu_layer(x)
        x = self.conv_transpose_layer3(x)
        x = self.bn_layer3(x, training=training)
        x = self.relu_layer(x)
        x = self.conv_transpose_layer4(x)
        x = self.bn_layer4(x, training=training)
        x = self.relu_layer(x)
        x = self.conv_transpose_layer5(x)
        x = self.tanh_layer(x)
        return x
