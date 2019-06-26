#conding = utf-8
#author kevin to zxx
#eg:DISPLAY=:0.0
import os, sys
import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras.utils import np_utils

import scipy.io.wavfile as wav
from sklearn import preprocessing
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import random
import scipy

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pandas as pd
import matplotlib.pyplot as plt

# start - 20params
from pylab import *
from scipy.io.wavfile import read
from scipy.signal import get_window
def _extract(audio, rate, numcep=13, type='delta_delta'):
    '''
    this func is to get audio's feature(like mfcc or somthing)
    :param audio: wav
    :param rate:
    :param numcep: jieshu
    :param type:
    :return: mfcc
    '''
    # mfcc_feat = mfcc(audio, rate, numcep=numcep, appendEnergy=True, lowfreq=90, highfreq=6000)
    mfcc_feat = mfcc(audio, rate, numcep=numcep, appendEnergy=True)

    if type == 'mfcc':
        return mfcc_feat

    if type == 'delta_delta':
        mfcc_feat = preprocessing.scale(mfcc_feat)
        mfcc_delta = delta(mfcc_feat, 2)
        mfcc_delta_delta = delta(mfcc_delta, 2)
        combined = np.hstack((mfcc_feat, mfcc_delta, mfcc_delta_delta))
        return combined

def _normalize(features):
    shape = features.shape
    _mean = np.mean([np.mean(t) for t in features])
    _std_mean = np.mean([np.std(t) for t in features])
    features = [(f - _mean)/_std_mean for f in features]
    return np.reshape(features, shape)

def _label_features(folder, label, is_test_mode=False):
    ubmList = []
    for root, dirs, files in os.walk(folder):
        #print('what is filesss>>',files)
        for file in files:
            if not file.endswith('.wav'): continue
            #ubmList.append('/'.join(file.split('/')[-3:]).split('.')[0])
            ubmList.append(file.split('.wav')[0])
        #print('what is ubmlist>',ubmList)

    np.random.shuffle(ubmList)
    if is_test_mode:
        ubmList = ubmList[:1]
    features = np.asarray(())

    for f in ubmList:
        #print('waht is wav dir>>',os.path.join(folder, f + '.wav'))
        sr, audio = wav.read(str(os.path.join(folder, str(f) + '.wav')))
        #print('what is wav.read output>>',sr,audio)
        vector = _extract(audio, sr, numcep=19, type='delta_delta')
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    np.random.shuffle(features)
    features = _normalize(features)
    #print('what is fea>>>>>>>>>',features,features.shape)
    id = int(label.split("user")[1])
    y = np.full((features.shape[0], 1), id)
    return features, y
if len(sys.argv) < 3:
    #print(len(sys.argv))
    print("usage: python dnn_sv.py /tmp/data  100")
    sys.exit(1)

input_folder = sys.argv[1]
training_no = int(sys.argv[2])

#genarate .h5 file dir#
model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sv_dnn.h5')
user_ids = [f for f in os.listdir(input_folder) if f != '.DS_Store']
#print(user_ids)
total_len = len(user_ids)
#choose part to train#
user_ids_train = ["user"+str(i) for i in range(1, training_no)]
print(user_ids_train,len(user_ids_train))
#use anothur to test#
user_ids_test = ["user"+str(i) for i in range(training_no, total_len)]
train_len = len(user_ids_train) + 1
test_len = len(user_ids_test)
print(user_ids_test,len(user_ids_test))
if os.path.isfile(model_file):
    print("[info] loading existed model")
    model = load_model(model_file)
    '''
    use one layer of the model to predict
    '''
    model_l6 = Model(inputs=model.input, outputs=model.get_layer('Dense_1').output)
    
else:
    print("[info] build new model")
    x_train, y_train = None, None
    for user_id in user_ids_train:
        print('user_id>>>>>>>>',user_id)
        enroll_dir = os.path.join(input_folder, str(user_id) + "/Enr")
        #print('enroll dir>>>>>>>',enroll_dir)
        #_label_features return wav_features and label #
        x, y = _label_features(enroll_dir, user_id)
        #print('what is users feature and label',user_id,x.shape,y.shape)
        if x_train is None:
            x_train = x
            y_train = y
        else:
             #what is vstack??#
            x_train = np.vstack((x_train, x))
            y_train = np.vstack((y_train, y))

        ver_dir = os.path.join(input_folder, user_id + "/Ver")
        x, y = _label_features(ver_dir, user_id)
        x_train = np.vstack((x_train, x))
        y_train = np.vstack((y_train, y))

     # Convert labels to categorical one-hot encoding#
    print('what is trian data',x_train,x_train.shape,y_train,y_train.shape)
    one_hot_labels = np_utils.to_categorical(y_train, num_classes=train_len)
    #print('one hot labes>>>',one_hot_labels)
    #######all of the above is avilable########################
    
    ####here is the network##########################
    dropout = 0.25
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu',name='Dense_1'))
    model.add(Dropout(dropout,name='dropout'))
    model.add(Dense(train_len, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(x_train, one_hot_labels,
          nb_epoch=20,
          batch_size=256)
    #model_l6 = model(inputs=model.input, outputs=model.get_layer('Dense_1').output)
    model.save(model_file)
    print("[info] model has saved")
    
# # Testing
#frr is "false reject rate".
#far is "false acceptance rate"
frr_dis = []
far_dis = []
print('user_ids_test>>>>>>>>>.',user_ids_test)
for user_id_i in user_ids_test:
    for user_id_j in user_ids_test:
        test_user = user_id_i
        imposter_user = user_id_j
        enroll_dir = os.path.join(input_folder, test_user + "/Enr")
        ver_dir = os.path.join(input_folder, imposter_user + "/Ver")

        x, y = _label_features(enroll_dir, test_user)
        #print("what is x >>>>>>>>>>>>",x,x.shape)
        #score_enroll = model.predict(x, batch_size=256)
        score_enroll = model_l6.predict(x, batch_size=256)
        #print('what is enroll models ptedict>>>>>>>>',score_enroll,score_enroll.shape)
        mean_enroll = np.mean(score_enroll, axis=0)
        #print('what is mean_enroll',mean_enroll,mean_enroll.shape)
        x, y = _label_features(ver_dir, imposter_user, True)
        #print('what is x ',x,x.shape)
        score_ver = model_l6.predict(x, batch_size=256)
        #print('what is ver model predict',score_ver,score_ver.shape)
        mean_ver = np.mean(score_ver, axis=0)
        #print('what is mean _ver',mean_ver)
        
        distance = scipy.spatial.distance.euclidean(mean_enroll, mean_ver)
        if user_id_i == user_id_j:
            frr_dis.append(distance)
        else:
            far_dis.append(distance)
#print('what is frr shape>>>>>>>',frr_dis.shape,far_dis.shape)

#plt.figure()
#plt.plot(np.(far_dis),np.(frr_dis))
#plt.show()

print("----------------frr")
frr_dis = np.asarray(frr_dis)
frr_dis_labeled = np.c_[frr_dis, np.full(frr_dis.shape, 1)]
frr_dis.sort()
print(frr_dis)

print("---------------far")
far_dis = np.asarray(far_dis)
far_dis_labeled = np.c_[far_dis, np.full(far_dis.shape, 0)]
far_dis.sort()
print(far_dis)

total_labled = np.vstack((frr_dis_labeled, far_dis_labeled))
score = total_labled[:, 0]
target = total_labled[:, 1]

fpr, tpr, thresholds = roc_curve(target, score, pos_label=1)
roc_auc = auc(fpr, tpr)

eer = 1 - brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
print("EER=",eer)
print("threshold=",thresh)
    
