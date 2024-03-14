seedValue=0
import os
os.environ["PYTHONHASHSEED"]=str(seedValue)
import numpy as np
np.random.seed(seedValue)
import random
random.seed(seedValue)
import tensorflow as tf
tf.random.set_seed(seedValue)
from tensorflow.keras.datasets import cifar10, fashion_mnist
import time
import argparse
import pandas as pd

from WorkerSide import Devices
from ServerSide import Server
import core
from encoder import encode
from encoder import encode2
import decoder
from decoder import decode
from sklearn.utils import shuffle
from sympy import symbols, solve

dar=0
NumofWorkers=10; gama=84060; Continue=1; seedValue=0
random.seed(seedValue);np.random.seed(seedValue);tf.random.set_seed(seedValue)

request=[] #Number of the requested gradients
answer=[] #Number of the received gradients
Global=[] #Global validation acc
Global2=[] #Global validation loss
Global3=[] #Global training acc
Global4=[] #Global training loss
Total=[] #Total learning delay
ComuniDN=[] #Total communication delay in the downlink
ComuniUP=[] #Total communication delay in the Uplink
Computi=[] #Total correlation calculation delay
tren=[] #Total training delay
nze=[0]*(NumofWorkers+1) #Packet loss rate for different workers
WOA=[] #worketrs
WOA.append(0)


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

a=10;b=200;NumDS1=5000
s = np.random.dirichlet((b, b, b, b, b, b, b, b, b, b), a)

idx = (y_train == 0).reshape(X_train.shape[0])
X_train_m0 = X_train[idx]; y_train_m0 = y_train[idx]
idx = (y_train == 1).reshape(X_train.shape[0])
X_train_m1 = X_train[idx]; y_train_m1 = y_train[idx]
idx = (y_train == 2).reshape(X_train.shape[0])
X_train_m2 = X_train[idx]; y_train_m2 = y_train[idx]
idx = (y_train == 3).reshape(X_train.shape[0])
X_train_m3 = X_train[idx]; y_train_m3 = y_train[idx]
idx = (y_train == 4).reshape(X_train.shape[0])
X_train_m4 = X_train[idx]; y_train_m4 = y_train[idx]
idx = (y_train == 5).reshape(X_train.shape[0])
X_train_m5 = X_train[idx]; y_train_m5 = y_train[idx]
idx = (y_train == 6).reshape(X_train.shape[0])
X_train_m6 = X_train[idx]; y_train_m6 = y_train[idx]
idx = (y_train == 7).reshape(X_train.shape[0])
X_train_m7 = X_train[idx]; y_train_m7 = y_train[idx]
idx = (y_train == 8).reshape(X_train.shape[0])
X_train_m8 = X_train[idx]; y_train_m8 = y_train[idx]
idx = (y_train == 9).reshape(X_train.shape[0])
X_train_m9 = X_train[idx]; y_train_m9 = y_train[idx]

bz=[[int(NumDS1*s[j][i]) for i in range(a)] for j in range(a)]

print(bz)
for i in range(1,10):
  for j in range(10):
    bz[i][j]=bz[i-1][j-1]
print(bz)


def MainChannel(Y, DLN):
  for j in range(0, int(DLN*len(Y))):
    rnd=random.randint(0, len(Y)-1-j)
    del Y[rnd]
  return Y

def solvi(N):
  x = symbols('x')
  probabilities = 0
  for k in range(2, N+1):
    probabilities = probabilities+k**(-2.1)
  expr = 0.1+probabilities*x-1
  sol = solve(expr)
  return sol

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Robust implementation of LT Codes encoding/decoding process in FL downlink channel.")
  parser.add_argument("-nd", "--DownLinknoise", help="noise level in downlink.", default=0.1, type=float)
  parser.add_argument("-nu", "--UpLinknoise", help="noise level in uplink.", default=0.1, type=float)
  parser.add_argument("-r", "--redundancy", help="the wanted redundancy.", default=1, type=float)
  parser.add_argument("-a", "--alpha", help="alpha!.", default=8, type=int)
  parser.add_argument("-k", "--kapa", help="kapa.", default=0, type=int)
  parser.add_argument("-c", "--transmissionRate", help="the transmission rate.", default=2000000, type=int)
  parser.add_argument("-acc", "--thrAcc", help="the threshold accuracy.", default=0.8, type=float)
  parser.add_argument("-mu", "--NoRetransmissionRounds", help="the No-Retransmission FL Rounds", default=7, type=float)
  parser.add_argument("-i", "--PS", help="Pretraine/Scrach", default=0, type=int)
  args = parser.parse_args()
  unoise=args.UpLinknoise
  ICC=args.PS
  noise=args.DownLinknoise
  alpha=args.alpha
  kapa=args.kapa
  R=args.transmissionRate
  thrAcc=args.thrAcc 
  QR=args.NoRetransmissionRounds

  for k in range(1, NumofWorkers+1):
    WOA.append(Devices(k))
  if ICC!=0:
    for dd in range(ICC+1):
      Global.append(0); Global2.append(0); Global3.append(0); Global4.append(0) 
      Total.append(0); ComuniDN.append(0); ComuniUP.append(0); Computi.append(0) 
      tren.append(0); request.append(0); answer.append(0)  
  
  S=Server()
  ii=ICC
  while Continue:
    ii=ii+1
    print ('**************************FL Round:', ii, '**********************************')
    redundancy=args.redundancy
    t_Up=[0]*(NumofWorkers+1) #Communication delay in the Uplink in each FL round 
    youpt=[0]*(NumofWorkers+1) #Transmission delay in the uplink in each FL round
    retratime=[0]*(NumofWorkers+1) #Retransmission delay in the uplink in each FL round
    calcul=[0]*(NumofWorkers+1) #Correlation calculation delay in the Uplink in each FL round
    fig=[1]*(NumofWorkers+1) #Number of the requested gradients from the workers in the uplink

    t_Dn=[0]*(NumofWorkers+1) #Communication delay in the downlink in each FL round
    f=[0]*(NumofWorkers+1) #Transmission delay in the downlink in each FL round
    g=[0]*(NumofWorkers+1) #Correlation calculation delay in the downlink in each FL round

    dt=[0]*(NumofWorkers+1) #Decoding delay
    re_dt=[0]*(NumofWorkers+1) #Redecoding delay        
    trt=[0]*(NumofWorkers+1) #Training delay in each FL round
    t_=[0]*(NumofWorkers+1) #Total leaning delay in each FL round        
    t_calcul=[0]*(NumofWorkers+1) #Correlation calculation delay    
    subtup=0 #Retransmission Counter in the uplink
    subtdn=0 #Retransmission Counter in the downlink
 
    #Local Updates            
    for k in range(1, NumofWorkers+1):
      trt[k]=WOA[k].local_update(ii, X_train_m0, X_train_m1, X_train_m2, X_train_m3, X_train_m4, 
      X_train_m5, X_train_m6, X_train_m7, X_train_m8, X_train_m9, bz, y_train_m0, y_train_m1, y_train_m2, y_train_m3, y_train_m4, y_train_m5,
      y_train_m6, y_train_m7, y_train_m8, y_train_m9, X_test, y_test)

      # WOA[k].savmodel(ii, QR, ICC)

    #Uplink Transmission
    if unoise>0:   
      for k in range(1, NumofWorkers+1):       
        random.seed(ii+(2*k))
        OutUp, youpt[k]=WOA[k].WorkerUp(ii, R, 3) #Uplink transmission  
        t_Up[k]=t_Up[k]+youpt[k]
        OutUp=WOA[k].UpChannel(OutUp, unoise) #Uplink Channel
        retratime[k], calcul[k]=S.ServerUp(WOA[k], OutUp, ii, alpha, QR, 3, R, ICC) #Server-side strategy in the uplink
        t_calcul[k]=t_calcul[k]+calcul[k]
        t_Up[k]=t_Up[k]+retratime[k]

        while fig[k]>0: #The workers will response as long as the sever requests for lost local gradients         
          subtup=subtup+1
          random.seed(501+ii+subtup+k)
          med= WOA[k].ReWorkerUp()
          OutUp=WOA[k].UpChannel(med, unoise) 
          fig[k], g[k]= S.ReServerUp(WOA[k], OutUp, 3, R)
          t_Up[k]=t_Up[k]+g[k]
    else:  
      t_Up[k]=t_Up[k]+((gama*3)*32)/R    
    for k in range(1, NumofWorkers+1): 
      S.addList(WOA[k]) #After finishing the uplink transmission and retransmissions, the servers stores the repaired local gradients
    
    #Gradients Averaging
    AvgAcc, Avgloss, AvgTAcc, AvgTloss=S.devid(NumofWorkers, alpha, ii, ICC)
    Global.append(AvgAcc); Global2.append(Avgloss); Global3.append(AvgTAcc); Global4.append(AvgTloss)
  
    if AvgAcc>thrAcc:
      Continue=0
    else:
      #initial Transmission in Downlink
      for k in range (1, NumofWorkers+1):
        f[k], OutDn=S.ServerDn(WOA[k], redundancy, R)
        t_Dn[k]=t_Dn[k]+f[k]        
      Belaks, tool=S.blocks_readCWCS(OutDn)
      NumofRedundBelaks = int(len(Belaks) * redundancy)      
      for k in range (1, NumofWorkers+1):        
        EncBelaks = []
        for dd in encode(Belaks, drops_quantity=NumofRedundBelaks): #LT encoding
          EncBelaks.append(dd) 
        random.seed(713+ii+k) 
        ListNoisyEncBelaks=MainChannel(EncBelaks, noise) #Noisy downlink channel
        DecBelaks, lenDecBelaks, dt[k] = decode(ListNoisyEncBelaks, blocks_quantity=len(Belaks)) #LT decoding
        OutDn[k]=WOA[k].WriteDn(tool, DecBelaks, lenDecBelaks)           
        g[k]=WOA[k].WorkerDn(OutDn[k], R, ii, alpha, QR, redundancy) #Worker-side strategy in the downlink  
        t_calcul[k]=t_calcul[k]+g[k]     
        ned=S.ask(WOA[k], NumofWorkers) #Request retransmissions from the server        
      ImpWeights, Andis, KAZA=S.retransCWCS(kapa, Global, ii, ICC) #server-side Strategy in the downlink

      for k in range(1, NumofWorkers+1):        
        t_Dn[k]=t_Dn[k]+dt[k]
        t_Up[k]=t_Up[k]+dt[k]
        WOA[k].Clean(ii, alpha, ICC) #Free-up the memory      
      request.append(ned); answer.append(KAZA) 
      
      if KAZA==0: #If the server does not retransmit any of the requested global gradients, the worker should re-use the previous local gradients instead
        for k in range(1, NumofWorkers+1):
          WOA[k].repairCWCS(alpha, ii)
      else: 
        #Downlink retransmissions        
        while KAZA>0:
          subtdn=subtdn+1   
          ret_t=(KAZA*redundancy*32)/R #Retransmission time       
          print('$$$$$$$$$$$$$$$$$$$  Retransmision round', subtdn, '$$$$$$$$$$$$$$$$$$$')
          redundancy=redundancy+0.2
          for k in range (1, NumofWorkers+1):    
            if KAZA<129: #We assume that only if the server retransmits only one packet, it will receive without error 
              WOA[k].ReWorkerDn1(ImpWeights)
          if KAZA>129:    
            Belaks, tool=S.blocks_readCWCS(ImpWeights)
            NumofRedundBelaks = int(len(Belaks) * redundancy)         
            for k in range(1, NumofWorkers+1):
              EncBelaks = []
              if KAZA<1000:
                sol=solvi(KAZA)
                for dd in encode2(sol, Belaks, drops_quantity=NumofRedundBelaks):
                  EncBelaks.append(dd) #Re-encoding with SFD
              else:
                for dd in encode(Belaks, drops_quantity=NumofRedundBelaks):
                  EncBelaks.append(dd)
              random.seed(ii+k+subtdn+173)
              ListNoisyEncBelaks=MainChannel(EncBelaks, noise)               
              DecBelaks, lenDecBelaks, re_dt[k] = decode(ListNoisyEncBelaks, blocks_quantity=len(Belaks)) #Re-decoding
              OutDn[k]=WOA[k].ReWriteDn(tool, DecBelaks, lenDecBelaks) 

          for k in range (1, NumofWorkers+1):
            WOA[k].ReWorkerDn2(OutDn[k]) #Receiving the requested global gradients and check if some of them are lost again 
            t_Up[k]=t_Up[k]+re_dt[k]
            t_Dn[k]=t_Dn[k]+re_dt[k]+ret_t                      
                                   
          for k in range(1, NumofWorkers+1):            
            boo=S.ask(WOA[k], NumofWorkers) #Requesting the lost global gradients during the retransmission 
          ImpWeights, Andis, KAZA=S.retransCWCS(kapa, Global, ii, ICC) #Server-side strategy in the downlink
          if KAZA==0:
            for k in range(1, NumofWorkers+1):
              WOA[k].repairCWCS(alpha, ii)

    for k in range(1, NumofWorkers+1):
      WOA[k].check(S)
    # Beta Check
    if dar==0:
      dar=S.Betacheck(NumofWorkers, ii, ICC)
      print('dar', dar)
      if dar!=0:
        if ii>alpha:
          QR=ii
    print('QR', QR)
    S.reset()
    #Preparing the results       
    bor=[]
    bor.append(0)
    for k in range(1, NumofWorkers+1):
      t_[k]=t_[k]+t_Dn[k]+t_Up[k]+trt[k]+t_calcul[k]
      bor.append(t_[k])
    T=max(bor)      
    print('Total learning delay in FL round', ii, '=', T)         
    ind=bor.index(max(bor))
    Total.append(T)
    tren.append(trt[ind])
    ComuniDN.append(t_Dn[ind]) 
    ComuniUP.append(t_Up[ind]) 
    Computi.append(t_calcul[ind]/4) #The code calculates 4 different correlation coefficients and we only want one of them
  report=zip(Global, Global2, Total, ComuniDN, ComuniUP, Computi, tren, request, answer)
  filename = "-ND"+str(noise)+"-NU"+str(unoise)+"-alpha"+ str(alpha)+"-kappa" + str(kapa)+ "-beta" +str(QR)+'.xlsx'
  pd.DataFrame(report).to_excel(filename, header=False, index=False)
  print('Beta:', QR)
