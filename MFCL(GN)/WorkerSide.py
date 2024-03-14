seedValue=0
import os
os.environ["PYTHONHASHSEED"]=str(seedValue)
import numpy as np
np.random.seed(seedValue)
import random
random.seed(seedValue)
import tensorflow as tf
tf.random.set_seed(seedValue)
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten, Dropout
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import math
from sklearn.utils import shuffle
from tensorflow.keras.losses import categorical_crossentropy
from keras.models import load_model
import time
import pandas as pd
import gc

from tensorflow.keras import initializers

random.seed(seedValue);np.random.seed(seedValue);tf.random.set_seed(seedValue)

gama=84060; EPOCHS = 1


class Devices:
  def __init__(self, name):
    self.name=name #Worker ID
    self.LW=[] #Local Weights
    self.NGW=[0]*gama #Received Noisy Global Weights in the Workers
    self.LoW=[0]*gama #Array of Lost Weights in the Downlink
    self.NGWS=[] #Array of Global Weights for the Correlation
    self.LWS=[] #Array of Local Weights for the Correlation
    self.model=0
    self.acc=0 #Local Validation Accuracy
    self.loss=0 #Local Validation Loss
    self.saveInd=[0]*gama
    self.tacc=0#Local Training Accuracy
    self.daryaft=[]  #Received Noisy Local Weights in the Sever
    self.LoW2=[] #Array of Lost Weights in the Uplink
    self.tloss=0 #Local Training Accuracy
    self.x_n=0
    self.y_n=0
    self.y_tt=0
    print(name)
    
  def local_update(self, ii, X_t_m0, X_t_m1, X_t_m2, X_t_m3, X_t_m4, X_t_m5, X_t_m6, X_t_m7, X_t_m8, X_t_m9, bz,
  y_t_m0, y_t_m1, y_t_m2, y_t_m3, y_t_m4, y_t_m5, y_t_m6, y_t_m7, y_t_m8, y_t_m9, x_t, y_t): 

    batch_size = 32
    optimizer = SGD(lr=0.01, momentum=0.9)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    if ii==1:            
      start=[0]*10
      end=[bz[0][0], bz[0][1], bz[0][2], bz[0][3], bz[0][4], bz[0][5], bz[0][6], bz[0][7], bz[0][8], bz[0][9]]
      if self.name>0:
        for fer in range (self.name-1):
          start=[end[j] for j in range(10)]
          end=[start[j]+bz[fer+1][j] for j in range(10)]
      print('start', start)
      print('end', end)

      az0=X_t_m0[start[0]:end[0]]; bz0=y_t_m0[start[0]:end[0]]
      az1=X_t_m1[start[1]:end[1]]; bz1=y_t_m1[start[1]:end[1]]
      az2=X_t_m2[start[2]:end[2]]; bz2=y_t_m2[start[2]:end[2]]
      az3=X_t_m3[start[3]:end[3]]; bz3=y_t_m3[start[3]:end[3]]
      az4=X_t_m4[start[4]:end[4]]; bz4=y_t_m4[start[4]:end[4]]
      az5=X_t_m5[start[5]:end[5]]; bz5=y_t_m5[start[5]:end[5]]
      az6=X_t_m6[start[6]:end[6]]; bz6=y_t_m6[start[6]:end[6]]
      az7=X_t_m7[start[7]:end[7]]; bz7=y_t_m7[start[7]:end[7]]
      az8=X_t_m8[start[8]:end[8]]; bz8=y_t_m8[start[8]:end[8]]
      az9=X_t_m9[start[9]:end[9]]; bz9=y_t_m9[start[9]:end[9]]

      self.x_n=np.concatenate((az0, az1, az2, az3, az4, az5, az6, az7, az8, az9), axis=0)
      y_n01=np.concatenate((bz0, bz1, bz2, bz3, bz4, bz5, bz6, bz7, bz8, bz9), axis=0)
      lol=len(y_n01)
      vay01=np.reshape(y_n01, (lol,))
      self.y_2 = to_categorical(vay01)
      self.y_tt = to_categorical(y_t)

      random.seed(seedValue);np.random.seed(seedValue);tf.random.set_seed(seedValue)
      self.model = Sequential()
      self.model.add(Flatten(input_shape=[28, 28]))
      self.model.add(Dense(100, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0.0), 
      bias_initializer=initializers.Zeros()))
      self.model.add(Dense(50, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0.0), 
      bias_initializer=initializers.Zeros()))
      self.model.add(Dense(10, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0.0), 
      bias_initializer=initializers.Zeros()))
      self.model.add(Dropout(0.3))
            
    else:
      b=[0]*6
      b[0]=np.reshape(self.NGW[0:78400], (784, 100))
      b[1]=np.reshape(self.NGW[78400:78500], (100,))
      b[2]=np.reshape(self.NGW[78500:83500], (100, 50))
      b[3]=np.reshape(self.NGW[83500:83550], (50,))
      b[4]=np.reshape(self.NGW[83550:84050], (50, 10))
      b[5]=np.reshape(self.NGW[84050:84060], (10,))

      for fer in range(6):
        b[fer] = b[fer].astype('float32')

      self.model.optimizer.apply_gradients(zip(b, self.model.trainable_variables))

    self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')  
    start1=time.time()

    numUpdates = int(self.x_n.shape[0] / batch_size)
    print('self.x_n', self.x_n.shape)
    print('self.x_n', self.y_2.shape)
    
    for epochs in range (EPOCHS):
      # print(self.y_2[0])
      # a, b = shuffle(np.array(self.x_n), np.array(self.y_2))
      for i in range(0, 200):
        # determine starting and ending slice indexes for the current batch
        start = i * batch_size
        end = start + batch_size
        X=self.x_n[start:end]; y=self.y_2[start:end]      
        with tf.GradientTape() as tape:
          pred = self.model(X)
          loss = categorical_crossentropy(y, pred)
          
          grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print('self.x_t', x_t.shape)
        print('self.y_tt', self.y_tt.shape)
        (loss, acc) = self.model.evaluate(x_t[0:32], self.y_tt[0:32])
    exit()
    stop1=time.time()    
    trtm=stop1-start1

    (loss, acc) = self.model.evaluate(x_t, self.y_tt)

    self.LW=['*']*gama
    self.LW[0:78400] = np.reshape(grads[0], (78400,))
    self.LW[78500:83500] = np.reshape(grads[2], (5000,))
    self.LW[83550:84050] = np.reshape(grads[4], (500,))
    self.LW[78400:78500] = np.reshape(grads[1], (100,))
    self.LW[83500:83550] = np.reshape(grads[3], (50,))
    self.LW[84050:84060] = np.reshape(grads[5], (10,))

    self.acc=acc
    self.loss= loss
    for kl in range(gama):
      self.LWS.append(self.LW[kl])
      if ii>1:
        self.NGWS.append(self.NGW[kl])
    self.NGW=['*']*gama
    return trtm


  def step(self, X, y, opt):
    with tf.GradientTape() as tape:
      pred = self.model(X)
      loss = categorical_crossentropy(y, pred)
      grads = tape.gradient(loss, self.model.trainable_variables)
      opt.apply_gradients(zip(grads, self.model.trainable_variables))
    return grads   

  def check(self, se):
    start=time.time()
    self.saveInd=[0]*gama
    Output = [index for index, elem in enumerate(self.LW) if elem == se.GW[index]]
    for dd in Output:
      self.saveInd[dd]=1 
    stop=time.time()
    tp=stop-start
    return

  def repairCWCS(self, alpha, ii): # To use the previous local gardients in the downlink in the case of packet loss or when the server does not sent the requested packets based on the Server-Side Strategy in the Downlink.
    for j in range(gama):
      if self.NGW[j]==5:
        self.NGW[j]=self.LW[j]   
    return

  def Clean(self, ii, alpha, ICC):
    if alpha>1:
      if len(self.NGWS)>alpha*gama:
        del self.NGWS[0:gama]
        del self.LWS[0:gama]
    if alpha==0 or alpha==1:
      if ii>ICC+1:
        del self.NGWS[0:gama]
        del self.LWS[0:gama]
    return

  def WorkerUp(self, ii, R, redundancy):
    upt=0 #Uplink Transmission Time
    self.daryaft=['*']*gama
    for dd in range(gama):
      self.daryaft[dd]=self.LW[dd]
    upt=((gama*redundancy)*32)/R
    AA=[0]*(gama)
    for dd in range(gama):
      AA[dd]=self.daryaft[dd]
    return AA, upt

  def UpChannel(self, X, UPN):
    indices = np.random.choice(len(X), replace=False, size=int(len(X)*UPN))
    for m in indices:
      X[m]=30
    return X

  def ReWorkerUp(self):
    med=[0]*gama
    for dd in range(gama):
      if self.LoW2[dd]==1:
        med[dd]=self.LW[dd]
    return med

  def WriteDn(self, tool, recovered_blocks, file_blocks_n):
    man=['*']*gama
    for dd in range(file_blocks_n):  
      if dd<file_blocks_n-1:
        data_as_float=[]
        if recovered_blocks[dd][0]!='a':
          data_bytes = np.array(recovered_blocks[dd], dtype=np.uint64)
          data_as_float = data_bytes.view(dtype=np.float64) 
        else:
          data_as_float=[5]*128
        for j in range(0, 128):
          man[(dd*128)+j]=data_as_float[j]  
      else:
        data_as_float=[]
        if recovered_blocks[dd][0]!='a':
          data_bytes = np.array(recovered_blocks[dd], dtype=np.uint64)
          data_as_float = data_bytes.view(dtype=np.float64)
        else:
          data_as_float=[5]*128 
        for j in range(0, tool):
          man[(dd*128)+j]=data_as_float[j] 
    for dd in range(gama):
      if man[dd]=='*':
        man[dd]=5
    return man

  def WorkerDn(self, sahe, R, ii, alpha, QR, redundancy): #Worker-side strategy in the downlink
    ct=0
    self.LoW=[0]*gama
    for dd in range(gama):
      self.NGW[dd]=sahe[dd]
    if alpha==0:
      for dd in range(len(self.LW)): 
        if self.NGW[dd]==5: 
          self.LoW[dd]=1    
    elif alpha==1:
      for dd in range(len(self.LW)): 
        if self.NGW[dd]==5: 
          self.NGW[dd]=self.LW[dd] 
    else:
      if ii>QR: 
        for dd in range(len(self.LW)): 
          if self.NGW[dd]==5:  
            if self.saveInd[dd]==1:
              self.NGW[dd]=self.LW[dd]              
            else:           
              azi=[]
              nafi=[]            
              for j in range (0, alpha-1):
                azi.append(self.NGWS[dd+((j+1)*len(self.LW))])
                nafi.append(self.LWS[dd+((j)*len(self.LW))])
              if np.count_nonzero(azi)==0 or np.count_nonzero(nafi)==0:
                self.NGW[dd]=self.LW[dd] 
              else:
                start2=time.time()
                Cor=np.corrcoef(azi, nafi)[0][1]   
                stop2=time.time()
                ct=ct+((stop2-start2)/4) 
                if math.isnan(Cor)==True:   
                  self.LoW[dd]=1      
                elif Cor<2/math.sqrt(alpha):  
                  self.LoW[dd]=1
                else:
                  if self.NGW[dd]==5:
                    self.NGW[dd]=self.LW[dd]                       
      else:
        for dd in range(gama): 
          if self.NGW[dd]==5:
            self.NGW[dd]=self.LW[dd]           
    return ct

  def ReWriteDn(self, tool, recovered_blocks, file_blocks_n):  
    man=['*']*gama
    for dd in range(file_blocks_n):      
      if dd<file_blocks_n-1:
        data_as_float=[]
        if recovered_blocks[dd][0]!='a':
          data_bytes = np.array(recovered_blocks[dd], dtype=np.uint64)
          data_as_float = data_bytes.view(dtype=np.float64) 
        else:
          data_as_float=[5]*128
        for j in range(0, 128):
          man[(dd*128)+j]=data_as_float[j]  
      else:
        data_as_float=[]
        if recovered_blocks[dd][0]!='a':
          data_bytes = np.array(recovered_blocks[dd], dtype=np.uint64)
          data_as_float = data_bytes.view(dtype=np.float64)
        else:
          data_as_float=[5]*128 
        for j in range(0, tool):
          man[(dd*128)+j]=data_as_float[j] 
    for dd in range(gama):
      if man[dd]=='*':
        man[dd]=5
    return man

  def ReWorkerDn1(self, UDP):
    for dd in range(gama):
      if self.LoW[dd]==1:
        self.NGW[dd]=UDP[dd]
        self.LoW[dd]=0 
    return 
    
  def ReWorkerDn2(self, UDP):  
    for dd in range(gama):
      if UDP[dd]!=5:
        if self.LoW[dd]==1:
          self.NGW[dd]=UDP[dd]
          self.LoW[dd]=0 
    return  
    

