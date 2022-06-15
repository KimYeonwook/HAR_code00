# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:45:56 2021

@author: KYW
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import logging
import json
from util import utils
from os.path import join
import os
        
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import scipy.stats as stats

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


from keras.layers import Input
from keras.layers.merge import concatenate

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import Activation, LSTM, GRU, TimeDistributed
from keras.models import Model
from keras import backend as K
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
from keras.layers import Bidirectional
from keras.layers import Multiply
from keras.layers import Add

from tensorflow.keras.layers import *

from imblearn.over_sampling import SMOTE
from datetime import datetime
from pandas import read_csv
from numpy import dstack

from tensorflow.python.client import device_lib

import pickle

#print(device_lib.list_local_devices())
tf.config.list_physical_devices('GPU')
#print('Version of tensorflow : ', tf.__version__)

#strategy_mirrored = tf.distribute.MirroredStrategy()
#strategy_mirrored = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#print('Number of device : {}'.format(strategy_mirrored.num_replicas_in_sync))

# Read configuration
with open('./HAR_code/HAR_00_config.json', "r") as read_file:
    config = json.load(read_file)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#Dataset Option

WISDM0_UCI1 = config.get("WISDM0_UCI1")
dataset_name = config.get("dataset_name")

#Deep learning model option
epochs = config.get("epochs")
Test_on = ['','Test_']
Test_on_idx = 0
if epochs < 400:
    Test_on_idx = 1
    
validation = config.get("validation")
verbose = config.get("verbose")
batch_size = config.get("batch_size") #* strategy_mirrored
early_stopping_option = config.get("early_stopping_option")
early_stopping_patience_number = config.get("early_stopping_patience_number")

Aug_Option = config.get("Aug_Option")
name_of_augmentation_algorithm = config.get("name_of_augmentation_algorithm")

if WISDM0_UCI1 == 0:
    number_of_each_class = config.get("number_of_each_class_WISDM")
else:
    number_of_each_class = config.get("number_of_each_class_UCI")

# Model =======================================================================

ModelNumb = config.get("ModelNumb")
NameofModel = config.get("NameofModel")




# graph for accuracy in training
def plot_accuracy(history, size = (10,5), title = 'Model Accuracy', save_figure = 1, save_path = './', dpi = 300):
    plt.figure(figsize=size) 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.ylim(0,1.2)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tick_params(axis='both')
    if save_figure == 1:
        plt.savefig(save_path,bbox_inches='tight', dpi=dpi)


# graph for loss in training
def plot_loss(history, size = (10,5), title = 'Model Loss', save_figure = 1, save_path = './', dpi = 300):
    
    plt.figure(figsize=size) 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.ylim(0,10)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tick_params(axis='both')
    if save_figure == 1:
        plt.savefig(save_path,bbox_inches='tight', dpi=dpi)


# test data prediction
def predict_classes(model, test_imgs, test_labels, emotions_dict,  batch_size  = 8):    

    # Predict class of image using trained model
    class_pred = model.predict(test_imgs, batch_size = batch_size)

    # Convert vector of zeros and ones to label
    labels_pred = np.argmax(class_pred,axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Boolean array that indicates whether the predicted label is the true label
    correct = labels_pred == true_labels
    
    # Converting array of labels into emotion names
    pred_emotion_names = pd.Series(labels_pred).map(emotions_dict)
    
    results = {'Predicted_label': labels_pred, 'Predicted_score': pred_emotion_names, 'Is_correct' : correct}
    results = pd.DataFrame(results)
    return correct, results


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
        # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
        # round : 반올림한다
        y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
        y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    
        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    
        # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
        count_true_positive_false_positive = K.sum(y_pred_yn)
    
        # Precision = (True Positive) / (True Positive + False Positive)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    
        # return a single tensor value
        return precision


def f1score(y_target, y_pred):
        _recall = recall(y_target, y_pred)
        _precision = precision(y_target, y_pred)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
        
        # return a single tensor value
        return _f1score


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.__TraningTime__ = 0
        self.__EpochTime__ = 0
        self.times = []
        self.sum_epochs_time = 0
        # use this value as reference to calculate cummulative time taken
        self.start_time = time.time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(time.time())
    def on_train_end(self,logs = {}):
        self.__TraningTime__ = time.time()-self.start_time
        for i in range(len(self.times)-1):
            self.sum_epochs_time = self.sum_epochs_time + self.times[i+1] - self.times[i]
        self.__EpochTime__ = self.sum_epochs_time/(len(self.times)-1)


def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels


def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

save_path = 'HAR_code/HAR_data/' + dataset_name[WISDM0_UCI1] 
with open(save_path + "_shape_of_data.p", 'rb') as f:
        load_shape_of_data = []
        while True:
            try:
                line = pickle.load(f)
            except EOFError:
                break
            load_shape_of_data.append(line)

with open(save_path + "_X_train_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'rb') as f:
    X_train = np.zeros((load_shape_of_data[0][0], load_shape_of_data[0][1],load_shape_of_data[0][2]))
    idx = 0
    while True:
        try:
            line_of_data = pickle.load(f)
        except EOFError:
            break
        X_train[idx] = line_of_data
        idx = idx+1

with open(save_path + "_y_train_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'rb') as f:
    y_train = np.zeros(load_shape_of_data[0][0])
    idx = 0
    while True:
        try:
            line_of_data = pickle.load(f)
        except EOFError:
            break
        y_train[idx] = line_of_data
        idx = idx+1

with open(save_path + "_X_test_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'rb') as f:
    X_test = np.zeros((load_shape_of_data[1][0], load_shape_of_data[1][1],load_shape_of_data[1][2]))
    idx = 0
    while True:
        try:
            line_of_data = pickle.load(f)
        except EOFError:
            break
        X_test[idx] = line_of_data
        idx = idx+1
        
with open(save_path + "_y_test_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'rb') as f:
    y_test = np.zeros(load_shape_of_data[1][0])
    idx = 0
    while True:
        try:
            line_of_data = pickle.load(f)
        except EOFError:
            break
        y_test[idx] = line_of_data
        idx = idx+1

print('X_train.shape : {}  X_test.shape : {}'.format(X_train.shape, X_test.shape))
print('y_train.shape : {}  y_test.shape : {}'.format(y_train.shape, y_test.shape))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Finish')
# =====================================================================================
for ModelN in ModelNumb:
    # make directroy and logfile
    log_path = utils.init_logger(''.join([Test_on[Test_on_idx],NameofModel[ModelN],'_',dataset_name[WISDM0_UCI1],'_',name_of_augmentation_algorithm[Aug_Option]])) 
 
    logging.info("\nConfiguration:\n\n{}".format(
    '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # ModelNumb == 0 : Double head 1D CNN + single head GRU
    if ModelN == 0:  
        
        number_of_head = 3
        #with strategy_mirrored.scope():
        # head 1 (1D-CNN)
        inputs1 = Input(shape= X_train[0].shape, name='input_part')
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Flatten()(conv1)
        
        # head 2 (1D-CNN)
        inputs2 = Input(shape=X_train[0].shape)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Flatten()(conv2)

        # head 3 (GRU)
        inputs3 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(inputs3)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = TimeDistributed(Dense(8))(GRUlayer)                          
        GRUlayer = Flatten()(GRUlayer)

        # merge
        merged = concatenate([conv1, conv2, GRUlayer])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])

    elif ModelN == 1:
        
        number_of_head = 1
        #with strategy_mirrored.scope():
        # head 1 (1D-CNN)
        inputs1 = Input(shape= X_train[0].shape, name='input_part')
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Flatten()(conv1)

        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(conv1) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])


    elif ModelN == 2: 
        
        number_of_head = 2
        #with strategy_mirrored.scope():
        # head 1 (1D-CNN)
        inputs1 = Input(shape= X_train[0].shape, name='input_part')
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Flatten()(conv1)
        
        # head 2 (1D-CNN)
        inputs2 = Input(shape=X_train[0].shape)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Flatten()(conv2)

        # merge
        merged = concatenate([conv1, conv2])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1, inputs2], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])


    elif ModelN == 3:
        
        number_of_head = 2
        #with strategy_mirrored.scope():
        # head 1 (1D-CNN)
        inputs1 = Input(shape= X_train[0].shape, name='input_part')
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Flatten()(conv1)

        # head 3 (GRU)
        inputs3 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(inputs3)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = TimeDistributed(Dense(8))(GRUlayer)                          
        GRUlayer = Flatten()(GRUlayer)

        # merge
        merged = concatenate([conv1, GRUlayer])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1, inputs3], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])
        
    
    elif ModelN == 4:
        
        number_of_head = 1
        # head 1
        inputs1 = Input(shape= X_train[0].shape, name='input_part')
        conv_GRU = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        conv_GRU = MaxPooling1D(pool_size=2)(conv_GRU)
        conv_GRU = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_GRU)
        conv_GRU = MaxPooling1D(pool_size=2)(conv_GRU)
        conv_GRU = Bidirectional(GRU(8, return_sequences=True))(conv_GRU)
        conv_GRU = TimeDistributed(Dense(8))(conv_GRU)                          
        conv_GRU = Flatten()(conv_GRU)
                
        # interpretation
        mataLearner = Dropout(0.7)(conv_GRU)
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])  
            
            
    elif ModelN == 5:
        
        number_of_head = 1

        # head 3 (GRU)
        inputs3 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(inputs3)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = TimeDistributed(Dense(8))(GRUlayer)                          
        GRUlayer = Flatten()(GRUlayer)

        # merge
        merged = concatenate([GRUlayer])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs3], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])

    elif ModelN == 6:
        
        number_of_head = 2
        #with strategy_mirrored.scope():
        # head 1 (GRU)
        inputs1 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer1 = Bidirectional(GRU(8, return_sequences=True))(inputs1)
        GRUlayer1 = Bidirectional(GRU(8, return_sequences=True))(GRUlayer1)
        GRUlayer1 = Bidirectional(GRU(8, return_sequences=True))(GRUlayer1)
        GRUlayer1 = TimeDistributed(Dense(8))(GRUlayer1)                          
        GRUlayer1 = Flatten()(GRUlayer1)

        # head 3 (GRU)
        inputs3 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(inputs3)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = Bidirectional(GRU(8, return_sequences=True))(GRUlayer)
        GRUlayer = TimeDistributed(Dense(8))(GRUlayer)                          
        GRUlayer = Flatten()(GRUlayer)

        # merge
        merged = concatenate([GRUlayer1, GRUlayer])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs1, inputs3], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])

    elif ModelN == 7:
        
        number_of_head = 1

        # head 3 (GRU)
        inputs3 = Input(shape=X_train[0].shape)#,batch_size=epochs_batch_size[model_idx][task_idx][1])
        GRUlayer = GRU(8, return_sequences=True)(inputs3)
        GRUlayer = GRU(8, return_sequences=True)(GRUlayer)
        GRUlayer = GRU(8, return_sequences=True)(GRUlayer)
        GRUlayer = TimeDistributed(Dense(8))(GRUlayer)                          
        GRUlayer = Flatten()(GRUlayer)

        # merge
        merged = concatenate([GRUlayer])
        
        # interpretation
        #mataLearner = Dense(1024, activation='relu')(merged) UCI
        mataLearner = Dropout(0.7)(merged) # WISDM 0.7 / UCI 0.3
        mataLearner = Dense(64, activation='relu')(mataLearner)
        mataLearner = Dense(6, activation='softmax')(mataLearner)
        model = Model(inputs=[inputs3], outputs=mataLearner)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])
        
        
    #plot_model(model, to_file=log_path+'structure_of_model.png', show_shapes=True, show_layer_names=True)
      
       
    # Fit and Evaluate  ===========================================================
    multi_head_X_train = []    
    multi_head_X_test = []
    #multi_head_X_val = []
    for head_idx in range(number_of_head):
        multi_head_X_train.append(X_train)
        multi_head_X_test.append(X_test)
        #multi_head_Xval.append(Xval)   

    # fit model
    timetaken = timecallback()
    if validation == 1: 
        if early_stopping_option == 1:
            earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience_number)
            history = model.fit(multi_head_X_train, y_train, validation_data=(multi_head_X_test, y_test), epochs=epochs,batch_size=batch_size, verbose=verbose, callbacks = [timetaken, earlystopping])
        else:
            history = model.fit(multi_head_X_train, y_train, validation_data=(multi_head_X_test, y_test), epochs=epochs,batch_size=batch_size, verbose=verbose, callbacks = [timetaken])
    else : 
        history = model.fit(multi_head_X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks = [timetaken])   


    model.summary(print_fn=logging.info)
    history_df = pd.DataFrame(history.history)
    logging.info('\n\nTraining History : \n\n{}\n\n'.format(history_df))
    
    # evaluate model
    evaluation_time = time.time()
    _loss, accuracy, _precision, _recall, _f1score = model.evaluate(multi_head_X_test, y_test, batch_size=batch_size, verbose=verbose)
    evaluation_time = time.time() - evaluation_time
    
    
    y_train = np.argmax(y_train, axis=1) #onehot to integer
    y_test = np.argmax(y_test, axis=1)
    
    
    # Display confusion matraix and graph =====================================
    plt.rc('font', size=12)
    if validation == 1: 
        plot_accuracy(history, save_path = log_path + NameofModel[ModelN] + '_accuracy.png')
        plot_loss(history,save_path = log_path + NameofModel[ModelN] + '_loss.png')
        #plot_learningCurve(history, epochs)
        
    #y_pred = model.predict_classes([X_test,X_test,X_test])
    y_pred = model.predict(multi_head_X_test)

    #probability to int
    y_pred = np.argmax(y_pred, axis=1)
    mat = confusion_matrix(y_test, y_pred)
    
    # integer to onehot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    if WISDM0_UCI1 == 0:
        class_name = np.array(['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])
    else:
        class_name = np.array(["Walking","Walking Upstairs","Walking Downstairs","Sitting","Standing","Laying"])
        
    fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=class_name, show_absolute=True, show_normed=True, figsize=(7,7))
    plt.savefig(log_path + NameofModel[ModelN] +'confusion_matrix.png',bbox_inches='tight', dpi = 300)
    #cm.figure_.savefig(log_path + NameofModel[ModelN] + 'confusion_matrix.png', dpi = 300)
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    model.save_weights(log_path + NameofModel[ModelN] + '_model'+time_stamp+'.h5')

    ep = len(history.history['accuracy'])
    epoch_time = timetaken.__EpochTime__
    
    logging.info('\n\nDataset: {}  Model: {}  Augmentation: {}\n\nLoss: {}\nAccuracy: {},\nPrecision: {},\nRecall: {},\nF1 score: {}\nEvaluation(=prediction) time: {} sec\nepochs: {}\nepoch time: {} sec'
                 .format(dataset_name[WISDM0_UCI1],NameofModel[ModelN],name_of_augmentation_algorithm[Aug_Option],round(_loss,7),round(accuracy,4),round(_precision,4),round(_recall,4),round(_f1score,4),round(evaluation_time,4),ep,round(epoch_time,4)))