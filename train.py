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
import os
FLAGS = crnn.FLAGS
def load_data():
    f = open(FLAGS.data_path,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,\
        Test_label,pernums_test,pernums_valid = cPickle.load(f)
    return train_data,train_label

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
def train(train_dir=None, model_dir=None, mode='train'):
    model = crnn.CRNN(mode)
    model._build_model()
    global_step = tf.Variable(0, trainable=False)
    #sess1 = tf.InteractiveSession()
    #load training data
    train_data,train_label = load_data()
    train_label = dense_to_one_hot(train_label,4)
    training_size = train_data.shape[0]
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels =  model.labels, logits =  model.logits)
        loss = tf.reduce_mean(cross_entropy)
#        print model.logits.get_shape()  
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.momentum, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.name_scope("train_step"):
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        training_size/FLAGS.train_batch_size,
                                        FLAGS.decay_rate,
                                        staircase=True)
        #print (lr.eval())        
        train_step = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(FLAGS.num_epochs):
            start = (i * FLAGS.train_batch_size) % training_size
            end = min(start+FLAGS.train_batch_size, training_size)
            _, loss_value, step,acc = sess.run([train_op, loss, global_step,accuracy],
                                           feed_dict={model.inputs:train_data[start:end],model.labels:train_label[start:end]})
            if i % 10 == 0:
                print "After %d training step(s), loss on training batch is %.2f, accuracy is %.3f." %(step, loss_value,acc)
                saver.save(
                        sess, os.path.join(FLAGS.checkpoint, FLAGS.model_name), global_step = global_step)
                
if __name__=='__main__':
    train()