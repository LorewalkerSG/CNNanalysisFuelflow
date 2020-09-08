#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#[alti , Regt , Legt , Ln1 , Rn2 , Ln2 , Mach , Lff , Rff]


#[alti , Mach , Legt , Ln1  , Ln2 , Lff ]

#[alti , Mach , Regt , Rn1  , Rn2 , Rff ]


import pywt 
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.layers import LeakyReLU

def wavelet_denoising(data):

    db4 = pywt.Wavelet('db4')

    coeffs = pywt.wavedec(data, db4,level=3)

    for i in range(1,len(coeffs)):
        coeffs[i]=pywt.threshold(coeffs[i],0.05*max(coeffs[i]))

    meta = pywt.waverec(coeffs, db4)
    return meta

def excel2m(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  
    ncols = table.ncols  
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)  
        datamatrix[:, x] = cols1 
    return datamatrix

def normalization(data):
    nrows = data.shape[0]  
    ncols = data.shape[1]  
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = data[:,x]
        minVals = min(cols)
        maxVals = max(cols)
        cols1 = np.matrix(cols)  
        ranges = maxVals - minVals
        b = cols1 - minVals 
        normcols = b / ranges  
        datamatrix[:, x] = normcols 
    return datamatrix


def build_model():
    model = Sequential()
    model.add(layers.Conv1D(80,4,
                            kernel_regularizer=regularizers.l1(0.00001),
                            input_shape=(5,1)))
    model.add(layers.LeakyReLU(0.001))
    model.add(layers.Conv1D(80, 2,
                            kernel_regularizer=regularizers.l1(0.00001),
                            ))
    model.add(layers.LeakyReLU(0.001))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,
                           kernel_regularizer=regularizers.l1(0.00001),
                           ))
    model.add(layers.LeakyReLU(0.001))
    model.summary()
#    optimizers.RMSprop(learning_rate=0.001,rho=0.1)
    optimizers.SGD(lr=0.01)
    model.compile(optimizer = 'SGD' , loss = 'mse' , metrics = ['mae']) 
    return model

os.chdir('/home/lorewalker') 
filename=os.listdir()
filename.sort()

num_rows=[]
each_mae=[]
train_mae=[]
for i in range(len(filename)):
    data = xlrd.open_workbook(filename[i])
    table = data.sheets()[0]
    num_rows.append(table.nrows)

data=np.zeros([sum(num_rows),6])

for i in range(len(filename)):
    data[sum(num_rows[0:i]):sum(num_rows[0:i+1]),:]=excel2m(filename[i])

for j in range(len(filename)): 
    for i in range(6):
        data[sum(num_rows[0:i]):sum(num_rows[0:i+1]),i]=wavelet_denoising(data[sum(num_rows[0:i]):sum(num_rows[0:i+1]),i])
    
data=normalization(data)

for i in range(6,7):
#for i in range(len(filename)):
    print('in 1D CNN processing fold ',i)
    data2=data[sum(num_rows[0:i]):sum(num_rows[0:i+1]),:]
    data1=np.concatenate([data[:sum(num_rows[0:i]),:],data[sum(num_rows[0:i+1]):,:]],axis=0)
    
    x_train = data1[:, 0:5]
    y_train = data1[:,5]
    x_test = data2[:, 0:5]
    y_test = data2[:,5]
    
    
    x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
    y_train=y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    model = build_model()
    BATCH_SIZE = 256
    EPOCHS = 30
    history = model.fit(x_train, y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_data=(x_test,y_test),verbose=0)
        

        
    history_dick = history.history
    loss_values = history_dick['loss']
    mean_absolute_error = history_dick['mae']
    train_mae.append(mean_absolute_error[len(mean_absolute_error)-1])    
    
    epochs = range(1,len(loss_values)+1)
    
    
    
    
        
    predict=model.predict(x_test)
    predict=predict.T
    predict=predict.squeeze()
    real=y_test
    real=real.squeeze()
    
    
    plt.plot(range(predict.shape[0]), predict,'r--' ,label='predict')
    plt.plot(range(predict.shape[0]), real,'b--',label='real')
        
    plt.title('time-ff')
    plt.xlabel('time')
    plt.ylabel('ff')
    plt.legend()
        
    plt.show()
    plt.clf()
        
        
    plt.plot(epochs, loss_values,'bo' ,label='Training loss')
    plt.plot(epochs, mean_absolute_error,'b',label='mae')
    
    plt.title('Training loss and mean_absolute_error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and mae')
    plt.legend()
        
    plt.show()
        
    plt.clf()
    val_loss = history.history['val_loss']
    val_mean_absolute_error = history.history['val_mae']
    
    each_mae.append(val_mean_absolute_error[len(val_mean_absolute_error)-1])    
    plt.plot(epochs,val_loss,'bo',label='val_loss')
    plt.plot(epochs,val_mean_absolute_error,'b',label='val_mean_absolute_error')
    plt.title('val_loss and val_mae')
    plt.xlabel('Epochs')
    plt.ylabel('val_loss and val_mae')
        
    plt.legend()
        
    plt.show()
    plt.clf()

print('val_mae',each_mae)
print('mean_val_mae',sum(each_mae)/8)
print('train_mae',train_mae)
print('mean_train_mae',sum(train_mae)/8)


