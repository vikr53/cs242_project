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

'''
# Instantiate ResNet model
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
'''
'''
# Instantiate LeNet model
inputs = keras.Input(shape=(32, 32, 3),name="input")
x = layers.Conv2D(6, 3, activation='relu',name="layer1")(inputs)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(16, 3, activation='relu',name="layer2")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Flatten()(x)

x = layers.Dense(120, activation='relu',name="FC1")(x)
x = layers.Dense(84, activation='relu',name="FC2")(x)
outputs = layers.Dense(10, activation='softmax')(x)
model_lenet = keras.Model(inputs, outputs)
print("Creating Checkpoint for LeNet model!")
model_lenet.save_weights("./chkpts/init_lenet.ckpt")
'''
# Instantiate AlexNet model
inputs = keras.Input(shape=(32, 32, 3),name="input")
x = layers.Conv2D(96, 3, strides = (4,4), name="layer1")(inputs)
x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
x = layers.Conv2D(256, 5, activation='relu', padding='same',name="layer2")(x)
x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
x = layers.Conv2D(384, 3, activation='relu',padding='same',name="layer3")(x)
x = layers.Conv2D(384, 3, activation='relu',padding='same',name="layer4")(x)
x = layers.Conv2D(256, 3, activation='relu',padding='same',name="layer5")(x)
x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
x = layers.Dense(4096, activation='relu',name="FC1")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(4096, activation='relu',name="FC2")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model_alexnet = keras.Model(inputs, outputs)

print("Creating Checkpoint for AlexNet model!")
model_alexnet.save_weights("./chkpts/init_alexnet.ckpt")
