#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:57:51 2018

@author: hxj
"""

import cPickle
import math
import numpy as np
import os
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion

def read_data(dir):
    f = open(dir,'rb')
    best_valid_uw,best_valid_w,pred_test_w,test_acc_w,confusion_w,pred_test_uw,test_acc_uw,confusion_uw = cPickle.load(f)
    f.close()
    return best_valid_uw,best_valid_w,pred_test_w,test_acc_w,confusion_w,pred_test_uw,test_acc_uw,confusion_uw

def load_data():
    f = open('./IEMOCAP7.pkl','rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = cPickle.load(f)
    return test_label


def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum(axis=0)
def decsion():
    dir0 = './model_max0.pkl'
    dir1 = './model_max1.pkl'
    dir2 = './model_max2.pkl'
    dir3 = './model_max3.pkl'
    dir4 = './model_max4.pkl'
    dir5 = './model_max5.pkl'
    dir6 = './model_max6.pkl'
    dir7 = './model_max7.pkl'
    test_label = load_data()
    best_valid_uw0,best_valid_w0,pred_test_w0,test_acc_w0,confusion_w0,pred_test_uw0,test_acc_uw0,confusion_uw0 = read_data(dir0)
    best_valid_uw1,best_valid_w1,pred_test_w1,test_acc_w1,confusion_w1,pred_test_uw1,test_acc_uw1,confusion_uw1 = read_data(dir1)
    best_valid_uw2,best_valid_w2,pred_test_w2,test_acc_w2,confusion_w2,pred_test_uw2,test_acc_uw2,confusion_uw2 = read_data(dir2)
    best_valid_uw3,best_valid_w3,pred_test_w3,test_acc_w3,confusion_w3,pred_test_uw3,test_acc_uw3,confusion_uw3 = read_data(dir3)
    best_valid_uw4,best_valid_w4,pred_test_w4,test_acc_w4,confusion_w4,pred_test_uw4,test_acc_uw4,confusion_uw4 = read_data(dir4)
    best_valid_uw5,best_valid_w5,pred_test_w5,test_acc_w5,confusion_w5,pred_test_uw5,test_acc_uw5,confusion_uw5 = read_data(dir5)
    best_valid_uw6,best_valid_w6,pred_test_w6,test_acc_w6,confusion_w6,pred_test_uw6,test_acc_uw6,confusion_uw6 = read_data(dir6)
    best_valid_uw7,best_valid_w7,pred_test_w7,test_acc_w7,confusion_w7,pred_test_uw7,test_acc_uw7,confusion_uw7 = read_data(dir7)
   
    
    print test_acc_uw0,test_acc_w0
    print test_acc_uw1,test_acc_w1
    print test_acc_uw2,test_acc_w2
    print test_acc_uw3,test_acc_w3
    print test_acc_uw4,test_acc_w4
    print test_acc_uw7,test_acc_w7
    print test_acc_uw6,test_acc_w6
    print test_acc_uw5,test_acc_w5
    
    #voting
    size = pred_test_uw0[0]
    Pred_w_vote = np.empty((size,8),dtype=np.int8)
    Pred_w_vote[:,0] = np.argmax(pred_test_w0,1)
    Pred_w_vote[:,1] = np.argmax(pred_test_w1,1)
    Pred_w_vote[:,2] = np.argmax(pred_test_w2,1)
    Pred_w_vote[:,3] = np.argmax(pred_test_w3,1)
    Pred_w_vote[:,4] = np.argmax(pred_test_w4,1)
    Pred_w_vote[:,5] = np.argmax(pred_test_w5,1)
    Pred_w_vote[:,6] = np.argmax(pred_test_w6,1)
    Pred_w_vote[:,7] = np.argmax(pred_test_w7,1)
#    print Pred_w0.shape, Pred_w1.shape, Pred_w2.shape, Pred_w3.shape
#    Pred_w_vote = np.concatenate((Pred_w0,Pred_w1,Pred_w2,Pred_w3),axis=1)
    pred_w_vote = np.empty((Pred_w_vote.shape[0],1),dtype=np.int8)
    for l in range(Pred_w_vote.shape[0]):
        pred_w_vote[l] = np.argmax(np.bincount(Pred_w_vote[l]))
    Pred_uw_vote = np.empty((size,8),dtype=np.int8)
    Pred_uw_vote[:,0] = np.argmax(pred_test_uw0,1)
    Pred_uw_vote[:,1] = np.argmax(pred_test_uw1,1)
    Pred_uw_vote[:,2] = np.argmax(pred_test_uw2,1)
    Pred_uw_vote[:,3] = np.argmax(pred_test_uw3,1)
    Pred_uw_vote[:,4] = np.argmax(pred_test_uw4,1)
    Pred_uw_vote[:,5] = np.argmax(pred_test_uw5,1)
    Pred_uw_vote[:,6] = np.argmax(pred_test_uw6,1)
    Pred_uw_vote[:,7] = np.argmax(pred_test_uw7,1)
#    print Pred_uw0.shape, Pred_uw1.shape, Pred_uw2.shape, Pred_uw3.shape
#    Pred_uw_vote = np.concatenate((Pred_uw0,Pred_uw1,Pred_uw2,Pred_uw3),axis=1)
    pred_uw_vote = np.empty((Pred_uw_vote.shape[0],1),dtype=np.int8)
    for l in range(Pred_uw_vote.shape[0]):
        pred_uw_vote[l] = np.argmax(np.bincount(Pred_uw_vote[l]))
    acc_uw_vote = recall(np.argmax(test_label, 1),pred_uw_vote,average='macro')
    acc_w_vote = recall(np.argmax(test_label, 1),pred_w_vote,average='weighted')
    conf_uw_vote = confusion(np.argmax(test_label, 1),pred_uw_vote)
    conf_w_vote = confusion(np.argmax(test_label, 1),pred_w_vote)
    print '*'*30
    print "Voting UW Accuracy: %3.4g" %acc_uw_vote
    print 'Confusion Matrix(UA):["ang","sad","hap","neu"]'
    print conf_uw_vote
    print "Voting W Accuracy: %3.4g" %acc_w_vote
    print 'Confusion Matrix(A):["ang","sad","hap","neu"]'
    print conf_w_vote
   
    
if __name__=='__main__':
    decsion()   
    
    