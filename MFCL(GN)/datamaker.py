import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf

def setmaker(x01, y01):
    lol=len(y01); vay01=np.reshape(y01, (lol,))
    x_n01, y_n01 = shuffle(np.array(x01), np.array(vay01))
    Mydatasetx01 = tf.data.Dataset.from_tensor_slices(x_n01)
    Mydatasety01 = tf.data.Dataset.from_tensor_slices(y_n01)
    dataset = tf.data.Dataset.zip((Mydatasetx01, Mydatasety01))
    yool=np.concatenate((y_n01, y_n01), axis=0)
    del x01, y01, lol, vay01, x_n01, y_n01, Mydatasetx01, Mydatasety01
    return dataset, yool

def tsnemaker(a, b, c, d, e, f):
    a = a.astype('float32')
    a /= 255
    b = np.asarray(b)
    tsne = TSNE().fit_transform(a.reshape((len(a),32*32*3)))
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))    
    print('tx.shape', tx.shape)
    print('ty.shape', ty.shape)
    print('b.shape', b.shape)
    plt.figure(figsize = (16,12))
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if e=='10':
      y_i = b == 0
      plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label=classes[0])
      y_i = b == 9
      plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label=classes[9])
    else:
      for i in range(f, d):
          y_i = b == i
          plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label=classes[i])
    plt.legend(loc=4)
    plt.gca().invert_yaxis()
    til='t-sne on raw pixelvalues in client'+e+'_'+c
    plt.title(til)
    til='/content/azi/pixel_client'+e+'_'+c+'.png'
    plt.savefig(til)
    plt.show()
    return 
    
