#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:54:52 2018

@author: hexuanji
"""



import tensorflow as tf
from attention import attention

epsilon = 1e-3
def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def acrnn(inputs, num_classes=4,
                  is_training=True,
                  L1=128,
                  L2=256,
                  cell_units=128,
                  num_linear=768,
                  p=10,
                  time_step=150,
                  F1=64,
                  dropout_keep_prob=1):
    layer1_filter = tf.get_variable('layer1_filter', shape=[5, 3, 3, L1], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable('layer1_bias', shape=[L1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer1_stride = [1, 1, 1, 1]
    layer2_filter = tf.get_variable('layer2_filter', shape=[5, 3, L1, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable('layer2_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer2_stride = [1, 1, 1, 1]
    layer3_filter = tf.get_variable('layer3_filter', shape=[5, 3, L2, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer3_bias = tf.get_variable('layer3_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer3_stride = [1, 1, 1, 1]
    layer4_filter = tf.get_variable('layer4_filter', shape=[5, 3, L2, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer4_bias = tf.get_variable('layer4_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer4_stride = [1, 1, 1, 1]
    layer5_filter = tf.get_variable('layer5_filter', shape=[5, 3, L2, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer5_bias = tf.get_variable('layer5_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer5_stride = [1, 1, 1, 1]
    layer6_filter = tf.get_variable('layer6_filter', shape=[5, 3, L2, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer6_bias = tf.get_variable('layer6_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer6_stride = [1, 1, 1, 1]
    
    linear1_weight = tf.get_variable('linear1_weight', shape=[p*L2,num_linear], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    linear1_bias = tf.get_variable('linear1_bias', shape=[num_linear], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
 
    fully1_weight = tf.get_variable('fully1_weight', shape=[2*cell_units,F1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully1_bias = tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    fully2_weight = tf.get_variable('fully2_weight', shape=[F1,num_classes], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully2_bias = tf.get_variable('fully2_bias', shape=[num_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    
    layer1 = tf.nn.conv2d(inputs, layer1_filter, layer1_stride, padding='SAME')
    layer1 = tf.nn.bias_add(layer1,layer1_bias)
    layer1 = leaky_relu(layer1, 0.01)
    layer1 = tf.nn.max_pool(layer1,ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID', name='max_pool')
    layer1 = tf.contrib.layers.dropout(layer1, keep_prob=dropout_keep_prob, is_training=is_training)
    
    layer2 = tf.nn.conv2d(layer1, layer2_filter, layer2_stride, padding='SAME')
    layer2 = tf.nn.bias_add(layer2,layer2_bias)
    layer2 = leaky_relu(layer2, 0.01)
    layer2 = tf.contrib.layers.dropout(layer2, keep_prob=dropout_keep_prob, is_training=is_training)
    
    layer3 = tf.nn.conv2d(layer2, layer3_filter, layer3_stride, padding='SAME')
    layer3 = tf.nn.bias_add(layer3,layer3_bias)
    layer3 = leaky_relu(layer3, 0.01)
    layer3 = tf.contrib.layers.dropout(layer3, keep_prob=dropout_keep_prob, is_training=is_training)
    
    layer4 = tf.nn.conv2d(layer3, layer4_filter, layer4_stride, padding='SAME')
    layer4 = tf.nn.bias_add(layer4,layer4_bias)
    layer4 = leaky_relu(layer4, 0.01)
    layer4 = tf.contrib.layers.dropout(layer4, keep_prob=dropout_keep_prob, is_training=is_training)
    
    layer5 = tf.nn.conv2d(layer4, layer5_filter, layer5_stride, padding='SAME')
    layer5 = tf.nn.bias_add(layer5,layer5_bias)
    layer5 = leaky_relu(layer5, 0.01)    
    layer5 = tf.contrib.layers.dropout(layer5, keep_prob=dropout_keep_prob, is_training=is_training)

    layer6 = tf.nn.conv2d(layer5, layer6_filter, layer6_stride, padding='SAME')
    layer6 = tf.nn.bias_add(layer6,layer6_bias)
    layer6 = leaky_relu(layer6, 0.01)    
    layer6 = tf.contrib.layers.dropout(layer6, keep_prob=dropout_keep_prob, is_training=is_training)
    
    layer6 = tf.reshape(layer6,[-1,time_step,L2*p])
    layer6 = tf.reshape(layer6, [-1,p*L2])
    
    linear1 = tf.matmul(layer6,linear1_weight) + linear1_bias
    linear1 = batch_norm_wrapper(linear1,is_training)
    linear1 = leaky_relu(linear1, 0.01)
    #linear1 = batch_norm_wrapper(linear1,is_training)
    linear1 = tf.reshape(linear1, [-1, time_step, num_linear])
    
    
    
    # Define lstm cells with tensorflow
    # Forward direction cell
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    
    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell1,
                                                             cell_bw=gru_bw_cell1,
                                                             inputs= linear1,
                                                             dtype=tf.float32,
                                                             time_major=False,
                                                             scope='LSTM1')

    # Attention layer
    gru, alphas = attention(outputs1, 1, return_alphas=True)
    
    
    fully1 = tf.matmul(gru,fully1_weight) + fully1_bias
    fully1 = leaky_relu(fully1, 0.01)
    fully1 = tf.nn.dropout(fully1, dropout_keep_prob)
    
    
    Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias
    #Ylogits = tf.nn.softmax(Ylogits)
    return Ylogits
