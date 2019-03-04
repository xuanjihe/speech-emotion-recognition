#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2017

@author: hxj
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import tensorflow as tf
from acrnn1 import acrnn
import cPickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os

tf.app.flags.DEFINE_integer('num_epoch', 5000, 'The number of epoches for training.')
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 60, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True,'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float   ('dropout_keep_prob',     1,        'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')

tf.app.flags.DEFINE_string  ('traindata_path', './IEMOCAP.pkl', 'total dataset includes training set')
tf.app.flags.DEFINE_string  ('validdata_path', 'inputs/valid.pkl', 'total dataset includes valid set')
tf.app.flags.DEFINE_string  ('checkpoint', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string  ('model_name', 'model4.ckpt', 'model name')

FLAGS = tf.app.flags.FLAGS

def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = cPickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = cPickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def train():
    #####load data##########
    
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = load_data(FLAGS.traindata_path)
    train_label = dense_to_one_hot(train_label,FLAGS.num_classes)
    valid_label = dense_to_one_hot(valid_label,FLAGS.num_classes)
    Valid_label = dense_to_one_hot(Valid_label,FLAGS.num_classes)
    valid_size = valid_data.shape[0]
    dataset_size = train_data.shape[0]
    vnum = pernums_valid.shape[0]
    best_valid_uw = 0
    
    
    ##########tarin model###########
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height,FLAGS.image_width,FLAGS.image_channel])
    Y = tf.placeholder(tf.int32, shape=[None, FLAGS.num_classes])
    is_training = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels =  Y, logits =  Ylogits)
    cost = tf.reduce_mean(cross_entropy)
    var_trainable_op = tf.trainable_variables()
    if FLAGS.is_adam:
        # not apply gradient clipping
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)            
    else:
        # apply gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, var_trainable_op), 5)
        opti = tf.train.AdamOptimizer(lr)
        train_op = opti.apply_gradients(zip(grads, var_trainable_op))
    correct_pred = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver=tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(FLAGS.num_epoch):
            #learning_rate = FLAGS.learning_rate            
            start = (i * FLAGS.batch_size) % dataset_size
            end = min(start+FLAGS.batch_size, dataset_size)
            [_,tcost,tracc] = sess.run([train_op,cost,accuracy], feed_dict={X:train_data[start:end,:,:,:], Y:train_label[start:end,:],
                                            is_training:True, keep_prob:FLAGS.dropout_keep_prob, lr:FLAGS.learning_rate})
            if i % 5 == 0:
                #for valid data
                valid_iter = divmod((valid_size),FLAGS.batch_size)[0]
                y_pred_valid = np.empty((valid_size,FLAGS.num_classes),dtype=np.float32)
                y_valid = np.empty((vnum,4),dtype=np.float32)
                index = 0
                cost_valid = 0
                if(valid_size < FLAGS.batch_size):
                    loss, y_pred_valid = sess.run([cross_entropy,Ylogits],feed_dict = {X:valid_data, Y:Valid_label,is_training:False, keep_prob:1})
                    cost_valid = cost_valid + np.sum(loss)
                for v in range(valid_iter):
                    v_begin = v*FLAGS.batch_size
                    v_end = (v+1)*FLAGS.batch_size
                    if(v == valid_iter-1):
                        if(v_end < valid_size):
                            v_end = valid_size
                    loss, y_pred_valid[v_begin:v_end,:] = sess.run([cross_entropy,Ylogits],feed_dict = {X:valid_data[v_begin:v_end],Y:Valid_label[v_begin:v_end],is_training:False, keep_prob:1})
                    cost_valid = cost_valid + np.sum(loss)
                cost_valid = cost_valid/valid_size
                for s in range(vnum):
                    y_valid[s,:] = np.max(y_pred_valid[index:index+pernums_valid[s],:],0)
                    index = index + pernums_valid[s]
    
                valid_acc_uw = recall(np.argmax(valid_label,1),np.argmax(y_valid,1),average='macro')
                valid_conf = confusion(np.argmax(valid_label, 1),np.argmax(y_valid,1))
                if valid_acc_uw > best_valid_uw:
                    best_valid_uw = valid_acc_uw
                    best_valid_conf = valid_conf
                    saver.save(sess, os.path.join(FLAGS.checkpoint, FLAGS.model_name), global_step = i+1)
                print ("*****************************************************************")
                print ("Epoch: %05d" %(i+1))
                print ("Training cost: %2.3g" %tcost)   
                print ("Training accuracy: %3.4g" %tracc) 
                print ("Valid cost: %2.3g" %cost_valid)
                print ("Valid_UA: %3.4g" %valid_acc_uw)    
                print ("Best valid_UA: %3.4g" %best_valid_uw) 
                print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print (valid_conf)
                print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print (best_valid_conf)
                print ("*****************************************************************" )
                 
                
if __name__=='__main__':
    train()
