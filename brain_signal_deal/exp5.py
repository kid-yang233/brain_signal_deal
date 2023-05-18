import os
from typing import List,Tuple
import numpy as np
import scipy.signal as signal
from scipy.signal import filtfilt
import joblib
import scipy.io as scio
from scipy.signal import butter,lfilter,hann
from sklearn import svm
from sklearn.preprocessing import normalize

datafile = "dataset\exp5\data.mat"

data = scio.loadmat(datafile)
train_label = data['train_l']
train_x = data['train_d'][0]
test_x = data['test_d'][0]
test_label = data['test_l']

fs= 200
Q=30 

def filter1(signal_set):#陷波滤波器
    new_set =[]
    for i in signal_set:
        b,a = signal.iirnotch(50,Q,fs)
        fitered_signal = filtfilt(b,a,i,axis=0)
        b,a = butter(4,[0.5,49.5],btype='bandpass',fs=fs)
        fitered_signal = filtfilt(b,a,i,axis=0)
        new_set.append(fitered_signal)

    return new_set

def filter2(signal_set):#带通滤波器
    Delta=[0.5,3]
    Theta=[3,8]
    Alpha=[8,12]
    Beta=[12,27]
    Gamma=[27,49]
    new_set = []
    for i in signal_set:
        b, a = butter(4,Delta,btype='bandpass',fs=fs)
        sign1 = filtfilt(b, a, i,axis=0)
        b, a = butter(4,Theta,btype='bandpass',fs=fs)
        sign2 = filtfilt(b, a, i,axis=0)
        b, a = butter(4,Alpha,btype='bandpass',fs=fs)
        sign3 = filtfilt(b, a, i,axis=0)
        b, a = butter(4,Beta,btype='bandpass',fs=fs)
        sign4 = filtfilt(b, a, i,axis=0)
        b, a = butter(4,Gamma,btype='bandpass',fs=fs)
        sign5 = filtfilt(b, a, i,axis=0)
        sign = np.concatenate((sign1,sign2,sign3,sign4,sign5),axis = 1)
        new_set.append(sign)
    return new_set


def window_feature(signal_set,label,window_length=2560,step=256):
    x_set = []
    y_set = []
    length = len(signal_set)
    for i in range(length):
        time_length,channel = signal_set[i].shape
        t = 0
        while t + window_length < time_length:
            u = signal_set[i][t:t+window_length,:]
            t = t+step
            u = np.fft.fft(u,n=1024)
            u = np.average(np.power(np.abs(u),2),axis=0)
            u = np.log2(u)
            x_set.append(u)
            y_set.append(label[i])
    return x_set,y_set


train_x = filter1(train_x)
train_x = filter2(train_x)
test_x = filter1(test_x)
test_x = filter2(test_x)

x_train,y_train = window_feature(train_x,train_label,2560,256)
x_test,y_test = window_feature(test_x,test_label,2560,256)

x_train,x_test = np.array(x_train),np.array(x_test)
x_train,x_test = normalize(x_train,axis=0),normalize(x_test,axis=0)

clf = svm.SVC()
clf.fit(x_train,y_train)

score = clf.score(x_test,y_test)
print(score)
