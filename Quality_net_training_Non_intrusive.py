# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:45:24 2018


If you find this code useful in your research, please cite:
Citation: 
       [1] S.-W. Fu, Y. Tsao, H.-T. Hwang, and H.-M. Wang, “Quality-Net: An end-to-end non-intrusive speech quality assessment model based on BLSTM,” in Proc. Interspeech, 2018
Contact:
       Szu-Wei Fu
       jasonfu@citi.sinica.edu.tw
       Academia Sinica, Taipei, Taiwan
	   
@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.activations import softmax
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input
from keras.constraints import max_norm

import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
random.seed(999)

epoch=15
batch_size=1
forgetgate_bias=-3 # Please see tha paper for more details

NUM_EandN=8000
NUM_Clean=800

def frame_mse(y_true, y_pred):  # Customized loss function  (frame-level loss, the second term of equation 1 in the paper)
    True_pesq=y_true[0,0]           
    return (10**(True_pesq-4.5))*tf.reduce_mean((y_true-y_pred)**2)

def Global_average(x):
    return 4.5*tf.reduce_mean(x,axis=-2)

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


    
def Sp_and_phase(path, Noisy=False):
    
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal))
    
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    
    #Lp = np.log10(np.abs(F)**2+10**-9)
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase

    
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path


def train_data_generator(file_list):
	index=0
	while True:
         pesq_filepath=file_list[index].split(',')
         noisy_LP, _ =Sp_and_phase(pesq_filepath[1])           
         pesq=np.asarray(float(pesq_filepath[0])).reshape([1])
         
         index += 1
         if index == len(file_list):
             index = 0
             
             random.shuffle(file_list)
       
         yield noisy_LP, [pesq, pesq[0]*np.ones([1,noisy_LP.shape[1],1])]

def val_data_generator(file_list):
	index=0
	while True:
         pesq_filepath=file_list[index].split(',')
         noisy_LP, _ =Sp_and_phase(pesq_filepath[1])           
         pesq=np.asarray(float(pesq_filepath[0])).reshape([1])
         
         index += 1
         if index == len(file_list):
             index = 0
       
         yield noisy_LP, [pesq, pesq[0]*np.ones([1,noisy_LP.shape[1],1])]

#################################################################             
######################### Training data #########################
###  LSTM Enhanced ###
Enhanced_list = ListRead('/home/jasonfu/SE/PESQ_estimation/ICASSP/TrainSet_Enhanced_PESQ.list')

###  Noisy ###
Noisy_list = ListRead('/home/jasonfu/SE/PESQ_estimation/ICASSP/TrainSet_Noisy_PESQ.list')

###  Clean ###
clean_list = ListRead('/home/jasonfu/SE/PESQ_estimation/ICASSP/Clean.list')


# Full list
Enhanced_noisy_list=Enhanced_list+Noisy_list
random.shuffle(Enhanced_noisy_list)
random.shuffle(clean_list)

Train_list= Enhanced_noisy_list[0:NUM_EandN]+clean_list[0:NUM_Clean]
random.shuffle(Train_list)
Num_train=len(Train_list)

################################################################
######################### Testing data #########################
Test_list= Enhanced_noisy_list[NUM_EandN:NUM_EandN+900]+clean_list[NUM_Clean:NUM_Clean+100]
Num_testdata=len(Test_list)
           
start_time = time.time()
print 'model building...'
_input = Input(shape=(None, 257))

activations2=Bidirectional(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), merge_mode='concat')(_input)
#activations2=Bidirectional(LSTM(150, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, recurrent_constraint=max_norm(0.00001)),merge_mode='sum')(activations1)

activations3=TimeDistributed(Dense(50))(activations2)
activations3=ELU()(activations3)
activations3=Dropout(0.3)(activations3)

Frame_score=TimeDistributed(Dense(1), name='Frame_score')(activations3)

Average_score=GlobalAveragePooling1D(name='Average_score')(Frame_score)

model = Model(outputs=[Average_score, Frame_score], inputs=_input)

# Initialization of the forget gate bias (optional)
W=model.layers[1].get_weights()
bias_init=np.concatenate((np.zeros([100]), forgetgate_bias*np.ones([100]), np.zeros([200])))
model.layers[1].set_weights([W[0], W[1], bias_init, W[3], W[4], bias_init])

model.compile(loss={'Average_score': 'mse', 'Frame_score': frame_mse}, optimizer='rmsprop')

plot_model(model, to_file='model.png', show_shapes=True)
    
with open('Quality-Net_(Non-intrusive).json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='Quality-Net_(Non-intrusive).hdf5', verbose=1, save_best_only=True, mode='min')  

print 'training...'
g1 = train_data_generator(Train_list)
g2 = val_data_generator  (Test_list)

hist=model.fit_generator(g1,	steps_per_epoch=Num_train, 
  					        verbose=1,
                            validation_data=g2,
                            validation_steps=Num_testdata,
                            max_queue_size=1, 
                            workers=1,
                            callbacks=[checkpointer])


model.load_weights('Quality-Net_(Non-intrusive).hdf5')   # Load the best model                         					

print 'testing...'
PESQ_Predict=np.zeros([len(Test_list),])
PESQ_true   =np.zeros([len(Test_list),])
for i in range(len(Test_list)):
    pesq_filepath=Test_list[i].split(',')
    noisy_LP, _ =Sp_and_phase(pesq_filepath[1])           
    pesq=float(pesq_filepath[0])
    
    [Average_score, Frame_score]=model.predict(noisy_LP, verbose=0, batch_size=batch_size)
    PESQ_Predict[i]=Average_score
    PESQ_true[i]   =pesq


MSE=np.mean((PESQ_true-PESQ_Predict)**2)
print ('Test error= %f' % MSE)
LCC=np.corrcoef(PESQ_true, PESQ_Predict)
print ('Linear correlation coefficient= %f' % LCC[0][1])
SRCC=scipy.stats.spearmanr(PESQ_true.T, PESQ_Predict.T)
print ('Spearman rank correlation coefficient= %f' % SRCC[0])

# Plotting the scatter plot
M=np.max([np.max(PESQ_Predict),4.55])
plt.figure(1)
plt.scatter(PESQ_true, PESQ_Predict, s=14)
plt.xlim([0,M])
plt.ylim([0,M])
plt.xlabel('True PESQ')
plt.ylabel('Predicted PESQ')
plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
plt.show()
plt.savefig('Scatter_plot_Quality-Net_(Non-intrusive).png', dpi=150)


# plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print 'drawing the training process...'
plt.figure(2)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('Learning_curve_Quality-Net_(Non-intrusive).png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
