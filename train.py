import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from wgan.data.create_data import image_preprocessing, real_data_generator
from wgan.losses.wgan_loss import Wasserstein_Loss
from wgan.model.gen_disc_model import Critic, Generator


class Wasserstein_GAN(tf.keras.models.Model):
    def __init__(self, latent_dim, critic_steps,
                 **kwargs):
        super(Wasserstein_GAN, self).__init__()
        self.generator_model = Generator()
        self.critic_model = Critic(0.01)
        self.wgan_loss = Wasserstein_Loss()
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps

    def compile(self, c_optimizer, g_optimizer):
        super(Wasserstein_GAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, real_data):
        real_image, real_label = real_data
        batch_size = tf.shape(real_image)[0]
        critic_latent_dim = tf.random.normal(shape=(batch_size, self.latent_dim))
        critic_y_fake = tf.ones((batch_size, 1))

        generator_latent_dim = tf.random.normal(shape=(batch_size * 2, self.latent_dim))
        generator_y_fake = -tf.ones((batch_size * 2, 1))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            for _ in range(self.critic_steps):
                real_output = self.critic_model(real_image, training=True)
                critic_generated_output = self.generator_model(critic_latent_dim, training=True)
                fake_output = self.critic_model(critic_generated_output, training=True)

            generated_output = self.critic_model(self.generator_model(generator_latent_dim, training=True),
                                                 training=True)

            critic_fake_loss = tf.reduce_mean(self.wgan_loss(critic_y_fake, fake_output))
            critic_real_loss = tf.reduce_mean(self.wgan_loss(real_label, real_output))
            critic_loss = critic_fake_loss + critic_real_loss
            generator_loss = tf.reduce_mean(self.wgan_loss(generator_y_fake, generated_output))

        critic_gradients = d_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        generator_gradients = g_tape.gradient(generator_loss, self.generator_model.trainable_variables)

        self.c_optimizer.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))
        self.g_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))

        return {"Critic_fake_loss": critic_fake_loss,
                "Critic_real_loss": critic_real_loss,
                "Generator_loss": generator_loss,
                "Critic_loss": critic_fake_loss - critic_real_loss}


def main():
    batch_size = 256
    epochs = 1000
    latent_dim = 100
    critic_steps = 5
    clip_value = 0.01
    image_width, image_height = 64, 64
    train_len = len(glob("/content/cats/*.jpg"))
    train_dataset_path = glob("/content/cats/*.jpg")

    discriminator_real_dataset = real_data_generator(train_dataset_path, batch_size)

    w_gan = Wasserstein_GAN(
        latent_dim=latent_dim,
        critic_steps=critic_steps)
    w_gan.generator_model.build((None, latent_dim))
    w_gan.critic_model.build((None, image_width, image_height, 3))

    w_gan.compile(c_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))
    cp_callbacks = [
        tf.keras.callbacks.ModelCheckpoint("/content/drive/My Drive/wgan_checkpoint_cats/",
                                           save_weights_only=True,
                                           monitor='loss',
                                           save_freq='epoch',
                                           period=1)
    ]
    w_gan.fit(discriminator_real_dataset, epochs=epochs, steps_per_epoch=train_len // batch_size,
              callbacks=cp_callbacks)
    w_gan.generator_model.save_weights("/content/drive/My Drive/wgan_checkpoint_generator/")


if __name__ == '__main__':
    main()