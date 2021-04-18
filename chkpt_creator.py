import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np

#MobileNet
#model = tf.keras.applications.MobileNet(include_top=True, input_shape=(32,32,3), weights=None, classes=10)

#model.save_weights('./chkpts/init_mnet.ckpt')

#Resnet8
def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

# Instantiate model
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
num_res_net_blocks = 8
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.save_weights("./chkpts/init_resnet8.ckpt")
