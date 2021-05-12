import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np

from resnet_generator import ResNetForCIFAR10

base_model = "resnet"

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

def SeparableConv( x , num_filters , strides , alpha=1.0 ):
    x = layers.DepthwiseConv2D( kernel_size=3 , padding='same' )( x )
    x = layers.BatchNormalization(momentum=0.9997)( x )
    x = layers.Activation( 'relu' )( x )
    x = layers.Conv2D( np.floor( num_filters * alpha ) , kernel_size=( 1 , 1 ) , strides=strides , use_bias=False , padding='same' )( x )
    x = layers.BatchNormalization(momentum=0.9997)(x)
    x = layers.Activation('relu')(x)
    return x

def Conv( x , num_filters , kernel_size , strides=1 , alpha=1.0 ):
    x = layers.Conv2D( np.floor( num_filters * alpha ) , kernel_size=kernel_size , strides=strides , use_bias=False , padding='same' )( x )
    x = layers.BatchNormalization( momentum=0.9997 )(x)
    x = layers.Activation('relu')(x)
    return x  

if base_model == "FLNet":
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

if base_model == "lenet":
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

if base_model == "MobileNet":
  # Instantiate MobileNet model
  inputs = keras.Input(shape=(32, 32, 3),name="input")
  x = Conv( inputs , num_filters=32 , kernel_size=3)
  x = SeparableConv( x , num_filters=32 , strides=1 )
  x = Conv( x , num_filters=64 , kernel_size=1 )
  x = SeparableConv( x , num_filters=64,strides=1)
  x = Conv( x , num_filters=128 , kernel_size=1 )
  x = SeparableConv( x , num_filters=128 , strides=1  )
  x = Conv( x , num_filters=128 , kernel_size=1 )
  x = SeparableConv( x , num_filters=128,strides=1)
  x = Conv( x , num_filters=256 , kernel_size=1 )
  x = SeparableConv( x , num_filters=256 , strides=1  )
  x = Conv( x , num_filters=256 , kernel_size=1 )
  x = SeparableConv( x , num_filters=256,strides=1)
  x = Conv( x , num_filters=512 , kernel_size=1 )
  x = SeparableConv(x, num_filters=512 , strides=1 )
  x = Conv(x, num_filters=1024 , kernel_size=1 )
  x = tf.keras.layers.AveragePooling2D( pool_size=( 7 , 7 ) )( x )
  x = tf.keras.layers.Flatten()( x )
  outputs = layers.Dense(10, activation='softmax')(x)
  model = keras.Model(inputs, outputs)

  print("Creating Checkpoint for MobileNet model!")
  model.save_weights("./chkpts/init_mobilenet.ckpt")

if base_model == "resnet":
  weight_decay = 1e-4
  num_blocks = 9
  name = "resnet"
  resnet = ResNetForCIFAR10(input_shape=(32, 32, 3),name=name,block_layers_num=num_blocks, classes=10, weight_decay=weight_decay)
  print("Creating Checkpoint for ResNet model!")
  resnet.save_weights("./chkpts/init_resnet.ckpt")


