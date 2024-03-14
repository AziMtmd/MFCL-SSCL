from absl import flags
from absl import logging
import tensorflow as tf
import model as model_lib
import objective as obj_lib
import metrics
import numpy
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D 
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Input, Flatten, Dropout, Conv2DTranspose, LeakyReLU

import os

import data as data_lib
import math
from tensorflow.keras import Model

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

import resnet
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
import tensorflow_addons as tfa

# import Strain2
import Wtrain2


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
# import pandas as pd

FLAGS = flags.FLAGS

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


class autokhodamClass():
  def __init__(self, m):
    self.name = m  # Worker ID
    self.topology = None
    self.strategy = tf.distribute.MirroredStrategy()


  def encoderkhodam(self, iti):  
    FLAGS.ModeAuto='TrainAuto'
    print('iti', iti)
    self.terroo=True
    if iti==0:  
      if FLAGS.CPC==1: 
        idx = (y_train == self.name).reshape(X_train.shape[0])
        az0 = X_train[idx]; bz0 = y_train[idx]
        x_n01=np.concatenate((az0[0:250], az0[250:500]), axis=0)
        y_n01=np.concatenate((bz0[0:2500], bz0[2500:5000]), axis=0)
        lol=len(y_n01)
        vay01=np.reshape(y_n01, (lol,))
        x_nn01, y_nn01 = shuffle(np.array(x_n01), np.array(vay01))
        Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_nn01)
        Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_nn01)
        dataset0 = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))
      if FLAGS.CPC==2:
        if self.name<9:
          idx = (y_train == self.name).reshape(X_train.shape[0])
          az0 = X_train[idx]; bz0 = y_train[idx]
          idx = (y_train == self.name+2).reshape(X_train.shape[0])
          az1 = X_train[idx]; bz1 = y_train[idx]
        else:
          idx = (y_train == self.name).reshape(X_train.shape[0])
          az0 = X_train[idx]; bz0 = y_train[idx]
          idx = (y_train == self.name-8).reshape(X_train.shape[0])
          az1 = X_train[idx]; bz1 = y_train[idx]

        x_n01=np.concatenate((az0[0:2500], az1[2500:5000]), axis=0)
        y_n01=np.concatenate((bz0[0:2500], bz1[2500:5000]), axis=0)
        lol=len(y_n01)
        vay01=np.reshape(y_n01, (lol,))
        x_nn01, y_nn01 = shuffle(np.array(x_n01), np.array(vay01))
        Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_nn01)
        Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_nn01)
        dataset0 = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))        

      if FLAGS.CPC==10:
        x_n01=np.concatenate((X_train[(self.name*5000):((self.name*5000)+2500)], X_train[((self.name*5000)+2500):(self.name+1)*5000]), axis=0)
        y_n01=np.concatenate((y_train[(self.name*5000):((self.name*5000)+2500)], y_train[((self.name*5000)+2500):(self.name+1)*5000]), axis=0)
        lol=len(y_n01)
        vay01=np.reshape(y_n01, (lol,))
        x_nn01, y_nn01 = shuffle(np.array(x_n01), np.array(vay01))
        Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_nn01)
        Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_nn01)
        dataset0 = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))  

      with self.strategy.scope():
        self.Auto_ds = data_lib.build_distributed_dataset(dataset0, FLAGS.AutoBatch, True, self.strategy, self.topology)
        self.server_ds = data_lib.build_distributed_dataset(dataset0, FLAGS.serBatch, True, self.strategy, self.topology)
        
        self.Aziiterator = iter(self.Auto_ds)
      
      # Encoder
      inputs2 = Input(shape=(32, 32, 3))
      x2 = resnet.Conv2dFixedPadding(filters=64, kernel_size=3, strides=1, data_format='channels_last', trainable=self.terroo)(inputs2)
      x2 = resnet.IdentityLayer(trainable=self.terroo)(x2)
      if FLAGS.norm=='PN':
        x2 = Wtrain2.GNN(axis = -1, trainable=self.terroo)(x2) #PN2
        x2 = LeakyReLU()(x2)#PN2
      if FLAGS.norm=='GN':
        x2=tfa.layers.GroupNormalization(axis = -1, trainable=self.terroo)(x2) #GN2
        x2 = LeakyReLU()(x2)#GN2
      if FLAGS.norm=='BN':
        x2 = resnet.BatchNormRelu(data_format='channels_last', trainable=self.terroo)(x2)#BN

      x2 = resnet.Conv2dFixedPadding(filters=64, kernel_size=3, strides=1, data_format='channels_last', trainable=self.terroo)(inputs2)
      if FLAGS.norm=='PN':
        x2 = Wtrain2.GNN(axis = -1, trainable=self.terroo)(x2) #PN2
        x2 = LeakyReLU()(x2)#PN2
      if FLAGS.norm=='GN':
        x2=tfa.layers.GroupNormalization(axis = -1, trainable=self.terroo)(x2) #GN2
        x2 = LeakyReLU()(x2)#GN2
      if FLAGS.norm=='BN':
        x2 = resnet.BatchNormRelu(data_format='channels_last', trainable=self.terroo)(x2)#BN

      self.encoded = resnet.IdentityLayer(trainable=self.terroo)(x2)

      # Decoder
      x2 = MaxPool2D()(self.encoded)
      x2 = Conv2DTranspose(64, 3, strides=1, padding='same')(x2)
      x2 = LeakyReLU()(x2)
      if FLAGS.norm=='PN':
        x2 = Wtrain2.GNN(axis = -1, trainable=self.terroo)(x2) #PN2
        x2 = LeakyReLU()(x2)#PN2
      if FLAGS.norm=='GN':
        x2=tfa.layers.GroupNormalization(axis = -1, trainable=self.terroo)(x2) #GN2
        x2 = LeakyReLU()(x2)#GN2
      if FLAGS.norm=='BN':
        x2 = resnet.BatchNormRelu(data_format='channels_last', trainable=self.terroo)(x2)#BN
      
      x2 = Conv2DTranspose(64, 3, strides=1, padding='same')(x2)
      x2 = LeakyReLU()(x2)
      if FLAGS.norm=='PN':
        x2 = Wtrain2.GNN(axis = -1, trainable=self.terroo)(x2) #PN2
        x2 = LeakyReLU()(x2)#PN2
      if FLAGS.norm=='GN':
        x2=tfa.layers.GroupNormalization(axis = -1, trainable=self.terroo)(x2) #GN2
        x2 = LeakyReLU()(x2)#GN2
      if FLAGS.norm=='BN':
        x2 = resnet.BatchNormRelu(data_format='channels_last', trainable=self.terroo)(x2)#BN
        
      self.decoded = Conv2DTranspose(3, 3, activation='sigmoid', strides=(2, 2), padding='same')(x2)
      self.autoencoder = Model(inputs2, self.decoded) 
      # self.autoencoder.summary() 

      self.loss_metric = tf.keras.metrics.Mean(name='train_loss') 
      self.valid_loss_metric = tf.keras.metrics.Mean(name='v_loss')    
      
    self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')        
    self.train_loss_history=[]; self.val_loss_history=[]   
    labels = 0; features=0
    
    if iti<FLAGS.tekrar-1:
      for epoch in range(FLAGS.epochsm):
        self.loss_metric.reset_states()
        with tf.GradientTape() as tape:  
          images, labels = next(self.Aziiterator)
          features, labels = images, {'labels': labels}
          num_transforms = features.shape[3] // 3
          num_transforms = tf.repeat(3, num_transforms)
          features_list = tf.split(features, num_or_size_splits=num_transforms, axis=-1)
          features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)
          predictions = self.autoencoder(features, training=True)
          pred_loss = binary_crossentropy(features, predictions)
          self.loss_metric.update_state(pred_loss)
          self.train_loss_history.append(self.loss_metric.result())
          print(f'Epoch {epoch}, Loss {self.loss_metric.result()}')
          gerad = tape.gradient(pred_loss, self.autoencoder.trainable_variables)          
          self.autoencoder.optimizer.apply_gradients(zip(gerad, self.autoencoder.trainable_variables)) 
          for i in range(len(gerad)):
            if gerad[i]==None:
              print(i)

    

    if iti==FLAGS.tekrar-1:
      jam=self.name+1+self.name
      NetName='identity_layer_'+str(jam)
      self.encoder = Model(self.autoencoder.inputs, self.autoencoder.get_layer(NetName).output)    
      self.encoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    return 



   
