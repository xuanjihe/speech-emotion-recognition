#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2018

@author: hxj
"""

import numpy as np
import tensorflow as tf
import crnn
import cPickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
FLAGS = crnn.FLAGS

def load_data():
    f = open(FLAGS.data_path,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,\
        Test_label,pernums_test,pernums_valid = cPickle.load(f)
    return test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
def evaluate():
    with tf.Graph().as_default() as g:
        model = crnn.CRNN('test')
        model._build_model()
        
        #load training data
        test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = load_data()
        test_label = dense_to_one_hot(test_label,4)
        valid_label = dense_to_one_hot(valid_label,4)
        Test_label = dense_to_one_hot(Test_label,4)
        Valid_label = dense_to_one_hot(Valid_label,4)
        test_size = test_data.shape[0]
        valid_size = valid_data.shape[0]
        tnum = pernums_test.shape[0]
        vnum = pernums_valid.shape[0]
        pred_test_uw = np.empty((tnum,4),dtype = np.float32)
        pred_test_w = np.empty((tnum,4),dtype = np.float32)
        valid_iter = divmod((valid_size),FLAGS.valid_batch_size)[0]
        test_iter = divmod((test_size),FLAGS.test_batch_size)[0]
        y_pred_valid = np.empty((valid_size,4),dtype=np.float32)
        y_pred_test = np.empty((test_size,4),dtype=np.float32)
        y_test = np.empty((tnum,4),dtype=np.float32)
        y_valid = np.empty((vnum,4),dtype=np.float32)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels =  model.labels, logits =  model.logits)        
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.momentum)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        flag = False
        best_valid_uw = 0
        best_valid_w = 0
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    #for validation data
                    index = 0
                    cost_valid = 0
                    if(valid_size < FLAGS.valid_batch_size):
                        validate_feed = {model.inputs:valid_data,model.labels:Valid_label}
                        y_pred_valid,loss = sess.run([model.logits,cross_entropy],feed_dict = validate_feed)
                        cost_valid = cost_valid + np.sum(loss)
                    for v in range(valid_iter):
                        v_begin = v*FLAGS.valid_batch_size
                        v_end = (v+1)*FLAGS.valid_batch_size
                        if(v == valid_iter-1):
                            if(v_end < valid_size):
                                v_end = valid_size
                        validate_feed = {model.inputs:valid_data[v_begin:v_end],model.labels:Valid_label[v_begin:v_end]}
                        loss, y_pred_valid[v_begin:v_end,:] = sess.run([cross_entropy,model.logits],feed_dict = validate_feed)
                        cost_valid = cost_valid + np.sum(loss)
                    cost_valid = cost_valid/valid_size
                    for s in range(vnum):
                        y_valid[s,:] = np.max(y_pred_valid[index:index+pernums_valid[s],:],0)
                        index = index + pernums_valid[s]
                    valid_acc_uw = recall(np.argmax(valid_label,1),np.argmax(y_valid,1),average='macro')
                    valid_acc_w = recall(np.argmax(valid_label, 1),np.argmax(y_valid,1),average='weighted')
                    valid_conf = confusion(np.argmax(valid_label, 1),np.argmax(y_valid,1))
                    
                    #for test set
                    index = 0
                    for t in range(test_iter):
                        t_begin = t*FLAGS.test_batch_size
                        t_end = (t+1)*FLAGS.test_batch_size
                        if(t == test_iter-1):
                            if(t_end < test_size):
                                t_end = test_size
                        #print t_begin,t_end,t,test_iter
                        test_feed = {model.inputs:test_data[t_begin:t_end],model.labels:Test_label[t_begin:t_end]}
                        y_pred_test[t_begin:t_end,:] = sess.run(model.logits, feed_dict = test_feed)
                        
                    for s in range(tnum):
                        y_test[s,:] = np.max(y_pred_test[index:index+pernums_test[s],:],0)
                        index = index + pernums_test[s]
                    
                    if valid_acc_uw > best_valid_uw:
                        best_valid_uw = valid_acc_uw
                        pred_test_uw = y_test
                        test_acc_uw = recall(np.argmax(test_label, 1),np.argmax(y_test,1),average='macro')
                        test_conf = confusion(np.argmax(test_label, 1),np.argmax(y_test,1))
                        confusion_uw = test_conf
                        flag = True
                   
                    if valid_acc_w > best_valid_w:
                        best_valid_w = valid_acc_w
                        pred_test_w = y_test
                        test_acc_w = recall(np.argmax(test_label, 1),np.argmax(y_test,1),average='weighted')
                        test_conf = confusion(np.argmax(test_label, 1),np.argmax(y_test,1))
                        confusion_w = test_conf
                        flag = True
                    #export
                    print "*****************************************************************"
                    print global_step                    
                    print "Epoch: %s" %global_step
                    print "Valid cost: %2.3g" %cost_valid
                    print "Valid_UA: %3.4g" %valid_acc_uw    
                    print "Valid_WA: %3.4g" %valid_acc_w
                    print "Best valid_UA: %3.4g" %best_valid_uw 
                    print "Best valid_WA: %3.4g" %best_valid_w
                    print 'Valid Confusion Matrix:["ang","sad","hap","neu"]'
                    print valid_conf
                    print "Test_UA: %3.4g" %test_acc_uw   
                    print "Test_WA: %3.4g" %test_acc_w
                    print 'Test Confusion Matrix:["ang","sad","hap","neu"]'
                    print confusion_uw
                    print "*****************************************************************" 
                    if(flag):
                        f=open(FLAGS.pred_name,'wb') 
                        cPickle.dump((best_valid_uw,best_valid_w,pred_test_w,test_acc_w,confusion_w,pred_test_uw,test_acc_uw,confusion_uw,),f)
                        f.close()
                        flag = False 

                
if __name__=='__main__':
    evaluate()