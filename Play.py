from keras import layers
from keras import models
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import PIL
import imagesize
import random
import tensorflow as tf
import os, shutil
from keras import *
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.image import extract_patches_2d
import h5py

from keras.callbacks import CSVLogger

f = h5py.File('train.h5', 'r')
input_train = f['data'][...]
label_train = f['label'][...]
f.close()

input_train = np.transpose(input_train,(0,2,3,1))
label_train = np.transpose(label_train,(0,2,3,1))

num_batches = 64
num_epochs = 100
num_layers = 20
num_lr = 0.01
filename = str(num_epochs) + 'ep_' + str(num_layers) + 'ls_' + str(num_batches) + 'bt_' + str(num_lr) + 'lr.h5'
# Camada de entrada do modelo
inputLayer = layers.Input(shape=(41,41,1))

#adicionando as camadas
n_layers = num_layers
for i in range(n_layers):
    if i == 0:
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(inputLayer)
    elif i == n_layers-1:
        x = layers.Conv2D(1,(3,3),activation='linear',padding='same',kernel_initializer='he_normal')(x)
    else:
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(x)

#somando as camadas final e entrada para aprendizagem residual
layer_out = layers.add([x,inputLayer])

model = models.Model(inputs=inputLayer,outputs=layer_out)

opt = optimizers.Adam(learning_rate=num_lr, decay=1E-3)

def PSNR(y_true, y_pred):    
    max_pixel = 1.0    
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

model.compile(optimizer=opt, loss='mse',metrics=[PSNR])

model.summary()

history = model.fit(input_train,label_train,
                    batch_size = num_batches,
                    epochs = num_epochs,
                    validation_split = 0.2)

model.save('.\\train_results\\' + 'TorchData_' + filename)

#golden = PIL.Image.open("F:\Images Dataset\IAPRTC\iaprtc12\images\\00\\33.jpg")
#golden = golden.convert('YCbCr')

#Input = golden
#Input = Input.resize((int(golden.size[0]/2),int(golden.size[1]/2)),PIL.Image.BICUBIC)
#Input = Input.resize((int(golden.size[0]),int(golden.size[1])),PIL.Image.BICUBIC)

#Y_g, cb_g, cr_g = golden.split()
#Y_in, cb_in, cr_in = Input.split()

#res = np.asarray(Y_g) - np.asarray(Y_in)
#res = PIL.Image.fromarray(res,'L')
#res = res.convert('YCbCr').split()[0]

#def calculate_psnr_MSE(img1, img2, max_value=1):
#        """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
#        mse = np.mean((np.array(img1, dtype=np.float64)/255 - np.array(img2, dtype=np.float64)/255) ** 2,dtype=np.float64)
#        if mse == 0:
#            return 100, mse
#        return (20 * np.log10(max_value / (np.sqrt(mse,dtype=np.float64)))), mse

#psnr, mse = calculate_psnr_MSE(res,Y_in)

##Y_g.show()
##Y_in.show()
##res.show()
#cb_g.show()
#cr_g.show()
#Y_g.show()