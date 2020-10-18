import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Wasserstein_Loss(tf.keras.losses.Loss):
  def __init__(self,**kwargs):
    super(Wasserstein_Loss,self).__init__()
  
  def call(self,y_true,y_pred):
    return y_true*y_pred
