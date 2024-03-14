from operator import add
from re import M
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import scipy
from sklearn.metrics import jaccard_score
from tensorflow.keras.optimizers import Adam
import pandas as pd

FLAGS = flags.FLAGS
dam3=[]
dam4=[]

class Ser:
  def __init__(self):
    self.myList = []
    self.myListG = [0]*40643
    self.GW = []
              
  def addList(self, m, iti):  # Receiving the loss, acc, and local gradients from the workers
    if m.name==0:
      self.myList=m.autoencoder.trainable_variables
      self.GW =m.autoencoder.trainable_variables
      # for i in range(40643):
      #   self.myListG[i]=m.geradiyan[i]
    else:
      self.myList = list(map(add, self.myList, m.autoencoder.trainable_variables))
      # self.myListG = tf.add(self.myListG, m.geradiyan)
      # self.myListG = list(map(add, self.myListG, m.geradiyan))
    return self

  def devid(self):  # Calculating the global gradients, Accuracy, and Loss
    self.GW = [x / (FLAGS.NumofWorkers) for x in self.myList]
    # self.GG = [y / (FLAGS.NumofWorkers) for y in self.myListG]
    return 

  def similarity(self, m): #Calculating the global gradients 
    sim=np.corrcoef(m.geradiyan, self.GG)[0][1]
    dam3.append(sim)
    dam4.append(sim)
    print('sim', sim)

    # correlation, p_value =  scipy.stats.pearsonr(m.geradiyan, self.GG)
    # print('correlation', correlation)
    # pakh=jaccard_score(m.geradiyan[0:100], self.GG[0:100], average='samples')
    # intersection1=set(m.geradiyan).intersection(set(self.GG))
    # # print('intersection1', intersection1)
    # print(len(intersection1))
    # unioni=len(m.geradiyan)+len(self.GG)-len(intersection1)
    # kl=len(intersection1)/unioni
    # print('pakh', kl)

    return 
  
  def filing(self):
    report3=zip(dam3, dam4)  
    filename2 = 'aziSim.xlsx'
    pd.DataFrame(report3).to_excel(filename2, header=False, index=False)
    return

  def clusters(self, NumofWorkers):
    clust=[[], [], [], [], [], [], [], [], [], []]
    for i in range(NumofWorkers):
      if self.simarray[i]>=0 and self.simarray[i]<0.2:
        clust[0].append(i)
      elif self.simarray[i]>=0.2 and self.simarray[i]<0.4:
        clust[1].append(i)
      elif self.simarray[i]>=0.4 and self.simarray[i]<0.6:
        clust[2].append(i)
      elif self.simarray[i]>=0.6 and self.simarray[i]<0.8:
        clust[3].append(i)
      elif self.simarray[i]>=0.8:
        clust[4].append(i)
      elif self.simarray[i]>=-0.2 and self.simarray[i]<0:
        clust[5].append(i)
      elif self.simarray[i]>=-0.4 and self.simarray[i]<-0.2:
        clust[6].append(i)
      elif self.simarray[i]>=-0.6 and self.simarray[i]<-0.4:
        clust[7].append(i)
      elif self.simarray[i]>=-0.8 and self.simarray[i]<-0.6:
        clust[8].append(i)
      else:
        clust[9].append(i)
    lena=0
    for i in range(NumofWorkers):
      if len(clust[i])>1:
        lena=lena+1
    print(lena)
    print('len(clust)', len(clust))
    print('(clust)', clust)
    self.simarray=[]
    return lena, clust   
  
  
  def broad(self, m):  # Calculating the global gradients, Accuracy, and Loss
    for yad in range(9):
      m.autoencoder.trainable_variables[yad].assign(self.GW[yad])
    # print('m.autoencoder.trainable_variables2[0]', m.name, m.autoencoder.trainable_variables[3][0][0][0])
    # m.geradiyan=[0]*40643
    # for jk in range(40643):
    #   m.geradiyan[jk]=self.GG[jk]
    return 







