#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:23:55 2017

@author: hxj
"""
import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
#import base
#import sigproc
eps = 1e-5
def getlogspec(signal,samplerate=16000,winlen=0.02,winstep=0.01,
               nfilt=26,nfft=399,lowfreq=0,highfreq=None,preemph=0.97,
               winfunc=lambda x:np.ones((x,))):
    highfreq= highfreq or samplerate/2
    signal = ps.sigproc.preemphasis(signal,preemph)
    frames = ps.sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames,nfft)
    return pspec 
def read_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def zscore(data,mean,std):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        data[i,:,:,0] = (data[i,:,:,0]-mean)/(std)
    return data

def mapminmax(data):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        min = np.min(data[i,:,:,0])
        max = np.max(data[i,:,:,0])
        data[i,:,:,0] = (data[i,:,:,0] - min)/((max - min)+eps)
    return data
def generate_label(emotion,classnum):
    label = -1
    if(emotion == 'angry'):
        label = 0
    elif(emotion == 'fear'):
        label = 1
    elif(emotion == 'happy'):
        label = 2
    elif(emotion == 'neutral'):
        label = 3
    elif(emotion == 'sad'):
        label = 4
    else:
        label = 5
    return label
        
        
def read_CASIA():
    rootdir = '/home/hxj/hxj/datasets/CASIA/'
    train_num = 1241# 
    test_num = 304#must subtract 1
    filter_num = 80
    train_label = np.empty((train_num,1), dtype = np.int8)
    test_label = np.empty((test_num,1), dtype = np.int8)
    train_data = np.empty((train_num,200,filter_num,1),dtype = np.float32)
    test_data = np.empty((test_num,200,filter_num,1),dtype = np.float32)
    valid_data = np.empty((100,200,filter_num,1),dtype = np.float32)
    valid_label = np.empty((100,1), dtype = np.int8)
    train_num = 0
    test_num = 0
    for name in os.listdir(rootdir):
        sub_dir = os.path.join(rootdir,name)
        for emotion in os.listdir(sub_dir):
            file_dir = os.path.join(sub_dir, emotion, '*.wav')
            files = glob.glob(file_dir)
            for filename in files:
                data, time, rate = read_file(filename)
                mel_spec = ps.logfbank(data,rate,nfilt = filter_num)
                #mel_spec = getlogspec(data,rate)  #frames*feature_dims
                if(name == 'wangzhe'):
                    if(mel_spec.shape[0] > 200):
                        #test_num = test_num + 2
                                               
                        em = generate_label(emotion,6)
                        part1 = mel_spec[0:200]
                        part2 = mel_spec[-200:] 
                        test_data[test_num,:,:,0] = part1
                        test_label[test_num] = em
                        test_num = test_num + 1
                        test_data[test_num,:,:,0] = part2
                        test_label[test_num] = em
                        test_num = test_num + 1
                        
                    else:
                        #test_num = test_num + 1
                        
                        part = mel_spec
                        part = np.pad(part,((0,200 - part.shape[0]),(0,0)),'constant',constant_values = 0)
                        #(上，下)，（左，右）                        
                        test_data[test_num,:,:,0] = part
                        em = generate_label(emotion,6)
                        test_label[test_num] = em
                        test_num = test_num + 1
                        
                else:
                    if(mel_spec.shape[0] > 200):
                        #train_num = train_num + 2
                                                
                        part1 = mel_spec[0:200]
                        part2 = mel_spec[-200:] 
                        em = generate_label(emotion,6)
                        #print  train_num,em
                        train_data[train_num,:,:,0] = part1
                        train_label[train_num] = em
                        train_num = train_num + 1
                        train_data[train_num,:,:,0] = part2
                        train_label[train_num] = em
                        train_num = train_num + 1
                        
                    else:
                        #train_num = train_num + 1
                                                
                        part = mel_spec
                        part = np.pad(part,((0,200 - part.shape[0]),(0,0)),'constant',constant_values = 0)
                        train_data[train_num,:,:,0] = part
                        em = generate_label(emotion,6)
                        #print train_num,em
                        train_label[train_num] = em
                        train_num = train_num + 1
    
    #apply zscore
    data = np.reshape(train_data,(-1,filter_num))
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    train_data = zscore(train_data,mean,std)
    test_data = zscore(test_data,mean,std)
    
    '''
    print train_num,test_num
    #apply mapminmax
    train_data = mapminmax(train_data)
    test_data = mapminmax(test_data)
    '''
    arr = np.arange(1241)
    np.random.shuffle(arr)
    valid_data = train_data[arr[0:100]]
    valid_label = train_label[arr[0:100]]
    train_data = train_data[arr[100:]]
    train_label = train_label[arr[100:]]
    print train_label.shape
    f=open('./CASIA_80.pkl','wb') 
    cPickle.dump((train_data,train_label,test_data,test_label,valid_data,valid_label),f)
    f.close()
    return
                
        


if __name__=='__main__':
    read_CASIA()
    #print "test_num:", test_num
    #print "train_num:", train_num
