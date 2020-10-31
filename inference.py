import tensorflow as tf
import matplotlib.pyplot as plt
from wgan.model.gen_disc_model import Generator

def test_show_image():
  latent_dim = 100
  generator_model = Generator(weights="/content/drive/My Drive/wgan_checkpoint_generator/")
  generator_model.compile(optimizer=tf.keras.optimizers.Adam(2e-4,beta_1=0.5))

  test_latent_dim = tf.random.normal(shape=(25,latent_dim))
  
  plt.figure(figsize=(15,15))
  for i in range(25):
    plt.subplot(5,5,i+1)
    predicted_image = generator_model(test_latent_dim,training=False)
    plt.imshow(tf.cast((predicted_image[i]+1)*127.5,tf.uint8))
  plt.show()

test_show_image()
